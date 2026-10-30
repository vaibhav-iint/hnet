"""Chunk pointer copy (HNetConfig.chunk_copy, hnet/model/chunk_copy.py, evolve g3-copy-01): exact pointers vs brute
force, default path unchanged, zero-init is a no-op, causality, packing, gradient paths (CLAUDE.md rules 1, 2, 10),
parameter / FLOP accounting (fairness contract)."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.chunk_copy import chunk_pointers
from hnet.model.chunking.select import select_positions
from hnet.model.ctx import SeqCtx

MM = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256, "suffix_lens": [2, 4]}]
CC = {"orders": [3, 6]}


def _repetitive(n_rows: int, length: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(97, 123, (12, 6), generator=g)
    rows = []
    for _ in range(n_rows):
        idx = torch.randint(0, 12, (length // 6 + 1,), generator=g)
        rows.append(words[idx].reshape(-1)[:length])
    return torch.stack(rows)


def _cfg(copy=True, match=True, vocab=True):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision = "end_mlp", "quantile"
    cfg.dc.chunk_vocab = VOCAB if vocab else None
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 512}
    cfg.match_model = dict(MM) if match else {}
    cfg.chunk_copy = dict(CC) if copy else {}
    return cfg


def _model(copy=True, steps=0, **kw):
    torch.manual_seed(0)
    m = build_model(_cfg(copy, **kw))
    m.init_weights()
    m.train()
    if steps:
        opt = torch.optim.SGD(m.parameters(), lr=5e-2)
        ids = _repetitive(2, 256, seed=1)
        for _ in range(steps):
            o = m(ids, labels=ids)
            opt.zero_grad()
            (o.loss + o.aux_loss).backward()
            opt.step()
    return m


def _perturb(m):
    """Non-zero copy parameters, so the path is exercised (zero-init makes it a no-op)."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in m.chunk_copy.parameters():
            p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return m


# ----------------------------------------------------------------------------- exact pointers
def test_chunk_pointers_match_brute_force():
    ids = _repetitive(1, 240, seed=3)[0]
    cu = torch.tensor([0, 100, 240], dtype=torch.int32)
    ctx = SeqCtx.from_cu_seqlens(cu, 140, 240)
    g = torch.Generator().manual_seed(4)
    mask = torch.rand(240, generator=g) < 0.3
    mask[cu[:-1].long()] = True
    _, idx, ctx_inner = select_positions(mask, ctx)
    orders = [2, 5]
    src, oi = chunk_pointers(ids, ctx, ctx_inner, orders)
    x, E = ids.tolist(), idx.tolist()
    seg = ctx.seq_idx[0].tolist()
    starts = [0, 100]
    for j, e in enumerate(E):
        want_src, want_oi = -1, len(orders)
        for i, n in enumerate(orders):  # ascending: the longest order with a match wins
            s0 = starts[seg[e]]
            if e - n + 1 < s0:
                continue
            for jp in range(j - 1, -1, -1):
                ep = E[jp]
                if seg[ep] != seg[e] or ep - n + 1 < s0:
                    continue
                if x[ep - n + 1: ep + 1] == x[e - n + 1: e + 1]:
                    want_src, want_oi = jp + 1, i
                    break
        assert (int(src[j]), int(oi[j])) == (want_src, want_oi), j
        if want_src >= 0:
            assert want_src <= j and seg[E[want_src]] == seg[e]  # causal and inside the document


# ----------------------------------------------------------------------------- default path / zero init
def test_default_model_has_no_copy_parameters_and_zero_init_is_bit_identical():
    off = _model(copy=False)
    on = _model(copy=True)
    assert not any(k.startswith("chunk_copy") for k in off.state_dict())
    common = {k: v for k, v in on.state_dict().items() if not k.startswith("chunk_copy")}
    assert common.keys() == off.state_dict().keys()
    for k, v in off.state_dict().items():  # same RNG stream: the copy module is built under a forked RNG
        assert torch.equal(v, common[k]), k
    ids = _repetitive(2, 256)
    for mode in ("train", "eval"):
        getattr(off, mode)(), getattr(on, mode)()
        torch.manual_seed(5)
        a = off(ids, labels=ids)
        torch.manual_seed(5)
        b = on(ids, labels=ids)
        assert torch.equal(a.logits, b.logits), mode
        assert torch.equal(a.loss, b.loss) and torch.equal(a.aux_loss, b.aux_loss), mode
    assert float(b.stats["chunk_copy_cov/s0"]) > 0.2, "the test input must exercise the pointer"


# ----------------------------------------------------------------------------- invariances (rule 10)
@pytest.mark.parametrize("vocab", [True, False])
def test_future_bytes_do_not_change_earlier_logits(vocab):
    m = _perturb(_model(steps=2, vocab=vocab)).eval()
    ids = _repetitive(1, 300, seed=5)
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    with torch.no_grad():
        oa, ob = m(ids), m(alt)
    assert float(oa.stats["chunk_copy_cov/s0"]) > 0.2
    assert torch.allclose(oa.logits[0, :200], ob.logits[0, :200], atol=1e-5), \
        (oa.logits[0, :200] - ob.logits[0, :200]).abs().max()


def test_packed_segments_equal_separate_sequences_even_when_segments_repeat_each_other():
    m = _perturb(_model(steps=2)).eval()
    a = _repetitive(1, 180, seed=7)
    b = torch.cat([a[:, :120], _repetitive(1, 100, seed=8)], 1)  # b starts with a copy of a: no cross-segment pointer
    with torch.no_grad():
        la = m(a).logits[0]
        lb = m(b).logits[0]
        packed = torch.cat([a[0], b[0]])
        cu = torch.tensor([0, 180, 400], dtype=torch.int32)
        lp = m(packed, cu_seqlens=cu, max_seqlen=220).logits
    assert torch.allclose(lp[:180], la, atol=1e-5), (lp[:180] - la).abs().max()
    assert torch.allclose(lp[180:], lb, atol=1e-5), (lp[180:] - lb).abs().max()


def test_the_pointer_changes_the_logits_once_nonzero():
    m = _model().eval()
    ids = _repetitive(1, 256, seed=9)
    with torch.no_grad():
        a = m(ids).logits
        b = _perturb(m)(ids).logits
    assert not torch.allclose(a, b)


# ----------------------------------------------------------------------------- gradient paths (rule 1)
def test_byte_ce_reaches_both_copy_parameters_at_init_and_the_copied_vector():
    m = _model()
    ids = _repetitive(2, 256)
    out = m(ids, labels=ids)
    cc = m.chunk_copy
    for name, p in (("proj", cc.proj.weight), ("offset", cc.offset.weight)):
        (g,) = torch.autograd.grad(out.loss, [p], retain_graph=True, allow_unused=True)
        assert g is not None and float(g.norm()) > 0, f"byte CE sends no gradient to chunk_copy.{name} at init"
    # once P != 0, the copied vector x_{j'+1} is a gradient path into the encoder of the earlier chunk
    _perturb(m)
    out = m(ids, labels=ids)
    enc = [p for n, p in m.named_parameters() if n.startswith("backbone.encoder.")]
    gs = torch.autograd.grad(out.loss, enc, allow_unused=True)
    assert sum(float(g.pow(2).sum()) for g in gs if g is not None) > 0


def test_pointers_carry_no_gradient():
    ids = _repetitive(1, 128)[0]
    ctx = SeqCtx.single(128)
    mask = torch.zeros(128, dtype=torch.bool)
    mask[::4] = True
    _, _, ci = select_positions(mask, ctx)
    src, oi = chunk_pointers(ids, ctx, ci, [3, 6])
    assert not src.requires_grad and not oi.requires_grad


# ----------------------------------------------------------------------------- accounting (fairness contract)
def test_param_count_and_flops_match_the_module():
    cfg = _cfg()
    m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    d0 = cfg.d_model[0]
    assert m.chunk_copy.num_params() == d0 * d0 + 3 * d0


def test_param_count_and_flops_of_the_candidate_config():
    cfg = model_config_from_yaml("configs/evolve/models/g3-copy-01.yaml")
    parent = model_config_from_yaml("configs/evolve/models/g2-match-mem.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_ws_L03.yaml")
    with torch.device("meta"):
        m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert count_params(cfg) <= 20_023_040  # the BPE arm
    f = model_flops(cfg, seq_len=8512)["flops_per_byte"]
    f_ref = model_flops(ref, seq_len=8512)["flops_per_byte"]
    assert abs(f / f_ref - 1) < 0.02
    assert cfg.chunk_copy and not parent.chunk_copy
