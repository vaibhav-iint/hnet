"""Prefix output dictionary (HNetConfig.prefix_head, hnet/model/prefix_head.py): prefix bookkeeping, causality,
packing, default path unchanged, gradient paths (CLAUDE.md rules 1, 2, 10), parameter / FLOP accounting."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.prefix_head import chunk_prefix_len

PH = {"buckets": 2048, "rank": 16, "k": 8, "max_len": 16, "use_main": True, "prior_scale": 4.0}
VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]


def _text(n_rows: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(97, 123, (40, 5), generator=g)
    words[:, -1] = 32  # "words" ending in a space
    rows = []
    for _ in range(n_rows):
        idx = torch.randint(0, 40, (length // 5 + 1,), generator=g)
        rows.append(words[idx].reshape(-1)[:length])
    return torch.stack(rows)


def _cfg(on=True, vocab=True):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision = "end_mlp", "quantile"
    cfg.dc.chunk_vocab = VOCAB if vocab else None
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 512}
    cfg.prefix_head = dict(PH) if on else {}
    return cfg


def _model(on=True, steps=3, vocab=True):
    torch.manual_seed(0)
    m = build_model(_cfg(on, vocab))
    m.init_weights()
    m.train()
    if steps:
        opt = torch.optim.SGD(m.parameters(), lr=5e-2)
        ids = _text(2, 256, seed=1)
        for _ in range(steps):
            o = m(ids, labels=ids)
            opt.zero_grad()
            (o.loss + o.aux_loss).backward()
            opt.step()
    return m


def test_chunk_prefix_len():
    mask = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0, 0, 0], dtype=torch.bool)
    assert chunk_prefix_len(mask).tolist() == [0, 1, 2, 0, 1, 0, 0, 1, 2, 3]


def test_prefix_hash_depends_only_on_the_chunk_prefix():
    m = _model(steps=0)
    ph = m.prefix_head
    from hnet.model.chunk_vocab import hash_bank
    from hnet.model.ctx import SeqCtx

    ids = torch.tensor([5, 1, 2, 3, 9, 7, 1, 2, 3, 4])
    mask = torch.tensor([1, 0, 0, 0, 1, 1, 0, 0, 0, 1], dtype=torch.bool)  # chunks "5" "123 9" "7"...
    ctx = SeqCtx.single(10)
    h = ph.prefix_hash(hash_bank(ids, ctx, 1, 16), mask)
    assert h[3] == h[8]  # both at prefix "123" of a chunk, after different chunks
    assert h[0] == h[4] == h[5] == h[9]  # chunk ends: empty prefix
    assert h[2] != h[3]


def test_default_model_has_no_prefix_parameters_and_zero_init_is_a_no_op():
    off, on = _model(on=False, steps=0), _model(on=True, steps=0)
    assert not any(k.startswith("prefix_head") for k in off.state_dict())
    common = {k: v for k, v in on.state_dict().items() if not k.startswith("prefix_head")}
    assert common.keys() == off.state_dict().keys()
    for k, v in off.state_dict().items():
        assert torch.equal(v, common[k]), k
    ids = _text(2, 256)
    off.eval(), on.eval()
    with torch.no_grad():
        a, b = off(ids, labels=ids), on(ids, labels=ids)
    assert torch.allclose(a.logits.float(), b.logits.float(), atol=1e-5)
    assert torch.allclose(a.loss, b.loss, atol=1e-6)


@pytest.mark.parametrize("vocab", [True, False])
def test_future_bytes_do_not_change_earlier_logits(vocab):
    m = _model(vocab=vocab).eval()
    with torch.no_grad():
        m.prefix_head.table.weight.normal_()  # make the correction large so a leak would show
    ids = _text(1, 300, seed=5)
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    with torch.no_grad():
        a = m(ids).logits[0, :200]
        b = m(alt).logits[0, :200]
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_packed_segments_equal_separate_sequences():
    m = _model().eval()
    with torch.no_grad():
        m.prefix_head.table.weight.normal_()
    a, b = _text(1, 180, seed=7), _text(1, 220, seed=8)
    with torch.no_grad():
        la, lb = m(a).logits[0], m(b).logits[0]
        lp = m(torch.cat([a[0], b[0]]), cu_seqlens=torch.tensor([0, 180, 400], dtype=torch.int32),
               max_seqlen=220).logits
    # tolerance: fp32 summation-order noise on O(10) rescored logits (a flipped top-k candidate would be O(1))
    assert torch.allclose(lp[:180], la, atol=1e-4), (lp[:180] - la).abs().max()
    assert torch.allclose(lp[180:], lb, atol=1e-4), (lp[180:] - lb).abs().max()


def test_gradient_paths():
    """At init (zero table) the byte CE reaches the table only; once the table is non-zero it also reaches both
    query projections and, through q_main, the main network and the router (STE in main_out)."""
    m = _model(steps=0)
    ids = _text(2, 256)
    out = m(ids, labels=ids)
    ph = m.prefix_head

    def gnorm(loss, ps):
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        return sum(float(g.pow(2).sum()) for g in gs if g is not None)

    assert gnorm(out.loss, [ph.table.weight]) > 0
    assert gnorm(out.loss, [ph.q_dec.weight, ph.q_main.weight]) == 0  # zero table: no signal to the queries yet
    with torch.no_grad():
        ph.table.weight.normal_(std=0.1)
    out = m(ids, labels=ids)
    assert gnorm(out.loss, [ph.q_dec.weight]) > 0 and gnorm(out.loss, [ph.q_main.weight]) > 0
    # the prefix-head correction alone (rescored minus base logits) reaches the main network and the router
    cap = {}
    hook = m.backbone.register_forward_hook(lambda mod, args, res: cap.update(h=res[0]))
    ids_flat = ids.reshape(-1)
    out = m(ids, labels=ids)
    hook.remove()
    from hnet.model.chunk_vocab import hash_bank
    from hnet.model.lm import pack_inputs

    _, ctx, _ = pack_inputs(ids, None, None)
    phash = ph.prefix_hash(hash_bank(ids_flat, ctx, 1, ph.max_len), out.bpred[0].boundary_mask)
    base = m.lm_head(cap["h"]).float()
    corr = (ph(base, cap["h"], out.bpred[0].main_out, phash)[0] - base).pow(2).sum()
    assert gnorm(corr, list(m.backbone.main_network.parameters())) > 0
    assert gnorm(corr, list(m.backbone.routing_module.parameters())) > 0


def test_param_count_and_flops_of_the_candidate_config():
    cfg = model_config_from_yaml("configs/evolve/models/g1-wild-02.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_ws_L03.yaml")
    with torch.device("meta"):
        m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert count_params(cfg) <= count_params(ref) <= 20.02e6
    f = model_flops(cfg, seq_len=8512)["flops_per_byte"]
    f_ref = model_flops(ref, seq_len=8512)["flops_per_byte"]
    assert abs(f / f_ref - 1) < 0.02
