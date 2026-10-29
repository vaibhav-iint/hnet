"""In-context match model (HNetConfig.match_model, hnet/model/match.py): exact candidates, causality, packing,
default path unchanged, gradient paths (CLAUDE.md rules 1, 2, 10), parameter / FLOP accounting (fairness contract)."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.match import NONE_ID, match_candidates, previous_occurrence

MM = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]


def _repetitive(n_rows: int, length: int, seed: int = 0) -> torch.Tensor:
    """Byte rows built from a small random 'vocabulary' of 6-byte words, so long exact repeats are frequent."""
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(97, 123, (12, 6), generator=g)
    rows = []
    for _ in range(n_rows):
        idx = torch.randint(0, 12, (length // 6 + 1,), generator=g)
        rows.append(words[idx].reshape(-1)[:length])
    return torch.stack(rows)


def _cfg(match=True, quantile=True):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    if quantile:
        cfg.dc.router, cfg.dc.decision, cfg.dc.chunk_vocab = "end_mlp", "quantile", VOCAB
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 512}
    cfg.match_model = dict(MM) if match else {}
    return cfg


def _model(match=True, steps=3, quantile=True):
    torch.manual_seed(0)
    m = build_model(_cfg(match, quantile))
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


# ----------------------------------------------------------------------------- exact candidates
def test_previous_occurrence_matches_brute_force():
    g = torch.Generator().manual_seed(0)
    key = torch.randint(0, 6, (300,), generator=g)
    valid = torch.rand(300, generator=g) > 0.2
    prev = previous_occurrence(key, valid)
    for t in range(300):
        want = -1
        if valid[t]:
            for s in range(t - 1, -1, -1):
                if valid[s] and key[s] == key[t]:
                    want = s
                    break
        assert int(prev[t]) == want, t


def test_match_candidates_match_brute_force_string_search():
    ids = _repetitive(1, 200, seed=3)[0]
    cu = torch.tensor([0, 90, 200], dtype=torch.int32)
    ctx = SeqCtx.from_cu_seqlens(cu, 110, 200)
    orders = [1, 3, 5, 9]
    C = match_candidates(ids, ctx, orders)
    x = ids.tolist()
    starts = [0, 90]
    for i, n in enumerate(orders):
        for t in range(200):
            s0 = max(s for s in starts if s <= t)
            want = NONE_ID
            if t - n + 1 >= s0:
                for tp in range(t - 1, s0 + n - 2, -1):
                    if x[tp - n + 1: tp + 1] == x[t - n + 1: t + 1]:
                        want = x[tp + 1]
                        break
            assert int(C[i, t]) == want, (n, t)


# ----------------------------------------------------------------------------- default path / zero init
def test_default_model_has_no_match_parameters_and_zero_init_is_a_no_op():
    off = _model(match=False, steps=0)
    on = _model(match=True, steps=0)
    assert not any(k.startswith("match_model") for k in off.state_dict())
    common = {k: v for k, v in on.state_dict().items() if not k.startswith("match_model")}
    assert common.keys() == off.state_dict().keys()
    for k, v in off.state_dict().items():  # same RNG stream: the match model is built last
        assert torch.equal(v, common[k]), k
    ids = _repetitive(2, 256)
    off.eval(), on.eval()
    with torch.no_grad():
        a, b = off(ids, labels=ids), on(ids, labels=ids)
    assert torch.allclose(a.logits.float(), b.logits.float(), atol=1e-5)
    assert torch.allclose(a.loss, b.loss, atol=1e-6)


# ----------------------------------------------------------------------------- invariances (rule 10)
@pytest.mark.parametrize("quantile", [True, False])
def test_future_bytes_do_not_change_earlier_logits(quantile):
    m = _model(quantile=quantile).eval()
    ids = _repetitive(1, 300, seed=5)
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    with torch.no_grad():
        a = m(ids).logits[0, :200]
        b = m(alt).logits[0, :200]
        n_match = int((m.match_model.candidates(ids[0], SeqCtx.single(300)) != NONE_ID).sum())
    assert n_match > 300, "the test input must exercise the match path"
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_packed_segments_equal_separate_sequences_even_when_segments_repeat_each_other():
    m = _model().eval()
    a = _repetitive(1, 180, seed=7)
    b = torch.cat([a[:, :120], _repetitive(1, 100, seed=8)], 1)  # b starts with a copy of a: no cross-segment match
    with torch.no_grad():
        la = m(a).logits[0]
        lb = m(b).logits[0]
        packed = torch.cat([a[0], b[0]])
        cu = torch.tensor([0, 180, 400], dtype=torch.int32)
        lp = m(packed, cu_seqlens=cu, max_seqlen=220).logits
    assert torch.allclose(lp[:180], la, atol=1e-5), (lp[:180] - la).abs().max()
    assert torch.allclose(lp[180:], lb, atol=1e-5), (lp[180:] - lb).abs().max()


# ----------------------------------------------------------------------------- gradient paths (rule 1)
def test_byte_ce_reaches_every_match_parameter_at_init():
    m = _model(steps=0)
    ids = _repetitive(2, 256)
    out = m(ids, labels=ids)
    mm = m.match_model
    for name, ps in {"enc": list(mm.enc.parameters()), "dec": list(mm.dec.parameters()),
                     "gate": list(mm.lm_head.parameters())}.items():
        grads = torch.autograd.grad(out.loss, ps, retain_graph=True, allow_unused=True)
        norm = sum(float(g.pow(2).sum()) for g in grads if g is not None)
        assert norm > 0, f"byte CE sends no gradient to the match model's {name} parameters at init"
    # rows of the "none" id and of matched bytes both learn; the candidate computation itself has no gradient
    cand = mm.candidates(ids.reshape(-1), SeqCtx.from_cu_seqlens(torch.tensor([0, 256, 512], dtype=torch.int32),
                                                                  256, 512))
    assert not cand.requires_grad


def test_gate_bonus_lands_on_the_candidate_byte():
    m = _model(steps=0)
    mm = m.match_model
    with torch.no_grad():
        mm.lm_head.bias.copy_(torch.tensor([1.0, 2.0, 4.0]))
    h = torch.zeros(3, m.config.d_model[0])
    cand = torch.tensor([[5, NONE_ID, 5], [5, 9, NONE_ID], [6, 9, NONE_ID]])  # [orders, T]
    bonus = mm.logit_bonus(h, cand, 256)
    assert bonus[0, 5] == 3 and bonus[0, 6] == 4
    assert bonus[1, 9] == 6 and bonus[1].sum() == 6
    assert bonus[2, 5] == 1 and bonus[2].sum() == 1


# ----------------------------------------------------------------------------- accounting (fairness contract)
def test_param_count_and_flops_of_the_candidate_config():
    cfg = model_config_from_yaml("configs/evolve/models/g1-wild-01.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_ws_L03.yaml")
    with torch.device("meta"):
        m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert count_params(cfg) <= count_params(ref) <= 20.02e6
    f = model_flops(cfg, seq_len=8512)["flops_per_byte"]
    f_ref = model_flops(ref, seq_len=8512)["flops_per_byte"]
    assert abs(f / f_ref - 1) < 0.02
