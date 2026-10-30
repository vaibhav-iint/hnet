"""Token-level in-context match model for the dense BPE baseline (DenseConfig.match_model, TokenMatchModel in
hnet/model/match.py): the fairness control of the byte match model (tests/test_match_model.py). Exact candidates,
causality, packing, default path unchanged, gradient paths (CLAUDE.md rules 1, 2, 10), the chunked CE with a sparse
logit bonus, parameter / FLOP accounting."""

import pytest
import torch

import hnet.model.lm as lm
from hnet.config import DenseConfig, model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.match import match_candidates

V = 300
MM = {"orders": [1, 2, 4], "enc": True, "dec": True, "gate": True}


def _repetitive(n_rows: int, length: int, seed: int = 0) -> torch.Tensor:
    """Token rows built from 10 random 'phrases' of 3 token ids in [0, V), so exact repeats are frequent."""
    g = torch.Generator().manual_seed(seed)
    words = torch.randint(0, V, (10, 3), generator=g)
    rows = []
    for _ in range(n_rows):
        idx = torch.randint(0, 10, (length // 3 + 1,), generator=g)
        rows.append(words[idx].reshape(-1)[:length])
    return torch.stack(rows)


def _cfg(match=True, **kw):
    mm = dict(MM, **kw) if match else {}
    return DenseConfig(vocab_size=V, d_model=32, n_layers=3, d_intermediate=64, num_heads=2, match_model=mm)


def _model(match=True, steps=3, **kw):
    torch.manual_seed(0)
    m = build_model(_cfg(match, **kw))
    m.init_weights()
    m.train()
    if steps:
        opt = torch.optim.SGD(m.parameters(), lr=5e-2)
        ids = _repetitive(2, 120, seed=1)
        for _ in range(steps):
            o = m(ids, labels=ids)
            opt.zero_grad()
            o.loss.backward()
            opt.step()
    return m


# ----------------------------------------------------------------------------- exact candidates
def test_token_candidates_match_brute_force_search_with_segments():
    ids = _repetitive(1, 150, seed=3)[0] + 1000  # ids above 256: nothing byte-specific
    cu = torch.tensor([0, 70, 150], dtype=torch.int32)
    ctx = SeqCtx.from_cu_seqlens(cu, 80, 150)
    orders = [1, 2, 4, 7]
    none = 5000
    C = match_candidates(ids, ctx, orders, none_id=none)
    x = ids.tolist()
    for i, n in enumerate(orders):
        for t in range(150):
            s0 = 0 if t < 70 else 70
            want = none
            if t - n + 1 >= s0:
                for tp in range(t - 1, s0 + n - 2, -1):
                    if x[tp - n + 1: tp + 1] == x[t - n + 1: t + 1]:
                        want = x[tp + 1]
                        break
            assert int(C[i, t]) == want, (n, t)
    m = _model(steps=0)
    assert torch.equal(m.match_model.candidates(ids % V, ctx),
                       match_candidates(ids % V, ctx, MM["orders"], none_id=V))


# ----------------------------------------------------------------------------- default path / zero init
def test_default_model_unchanged_and_zero_init_is_a_no_op():
    off = _model(match=False, steps=0)
    on = _model(match=True, steps=0)
    assert off.match_model is None and not any(k.startswith("match_model") for k in off.state_dict())
    common = {k: v for k, v in on.state_dict().items() if not k.startswith("match_model")}
    assert common.keys() == off.state_dict().keys()
    for k, v in off.state_dict().items():  # same RNG stream: the match model is built under a forked RNG
        assert torch.equal(v, common[k]), k
    ids = _repetitive(2, 120)
    off.eval(), on.eval()
    with torch.no_grad():
        a, b = off(ids, labels=ids), on(ids, labels=ids)
        a2, b2 = off(ids, labels=ids, return_logits=False), on(ids, labels=ids, return_logits=False)
    assert torch.equal(a.logits.float(), b.logits), (a.logits.float() - b.logits).abs().max()
    assert torch.allclose(a.loss, b.loss, atol=1e-6) and torch.allclose(a2.loss, b2.loss, atol=1e-6)


# ----------------------------------------------------------------------------- invariances (rule 10)
@pytest.mark.parametrize("dec_layer", [1, 2])
def test_future_tokens_do_not_change_earlier_logits(dec_layer):
    m = _model(dec_layer=dec_layer).eval()
    ids = _repetitive(1, 200, seed=5)
    alt = ids.clone()
    alt[0, 130:] = (alt[0, 130:] + 7) % V
    with torch.no_grad():
        a = m(ids).logits[0, :130]
        b = m(alt).logits[0, :130]
        n_match = int((m.match_model.candidates(ids[0], SeqCtx.single(200)) != V).sum())
    assert n_match > 300, "the test input must exercise the match path"
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_packed_segments_equal_separate_sequences_even_when_segments_repeat_each_other():
    m = _model().eval()
    a = _repetitive(1, 110, seed=7)
    b = torch.cat([a[:, :80], _repetitive(1, 60, seed=8)], 1)  # b starts with a copy of a: no cross-segment match
    with torch.no_grad():
        la = m(a).logits[0]
        lb = m(b).logits[0]
        packed = torch.cat([a[0], b[0]])
        cu = torch.tensor([0, 110, 250], dtype=torch.int32)
        lp = m(packed, cu_seqlens=cu, max_seqlen=140).logits
        bb = m(torch.cat([a, a], 0)).logits  # [B, L] input: rows are separate documents
    assert torch.allclose(lp[:110], la, atol=1e-5), (lp[:110] - la).abs().max()
    assert torch.allclose(lp[110:], lb, atol=1e-5), (lp[110:] - lb).abs().max()
    assert torch.allclose(bb[1], la, atol=1e-5)


# ----------------------------------------------------------------------------- gradient paths (rules 1, 2)
def _grad_norms(m, ids):
    out = m(ids, labels=ids)
    mm = m.match_model
    groups = {f"proj/{s}": [mm.proj[s].weight] for s in mm.sites}
    groups.update({f"offset/{s}": [mm.offset[s].weight] for s in mm.sites})
    groups.update({f"gain/{s}": [mm.gain[s]] for s in mm.sites})
    groups["gate"] = list(mm.lm_head.parameters())
    res = {}
    for name, ps in groups.items():
        grads = torch.autograd.grad(out.loss, ps, retain_graph=True, allow_unused=True)
        res[name] = sum(float(g.pow(2).sum()) for g in grads if g is not None)
    return res


def test_token_ce_reaches_the_match_parameters_at_init_and_the_gains_after_one_step():
    ids = _repetitive(2, 120)
    g0 = _grad_norms(_model(steps=0), ids)
    for name, v in g0.items():
        if name.startswith("gain/"):  # gain multiplies the input of the zero-init projection P_s: no path at init
            assert v == 0, name
        else:
            assert v > 0, f"token CE sends no gradient to the match model's {name} at init"
    g1 = _grad_norms(_model(steps=1), ids)
    assert all(v > 0 for v in g1.values()), g1


def test_gate_bonus_lands_on_the_candidate_token_and_chunked_ce_equals_dense_ce(monkeypatch):
    m = _model(steps=2)
    mm = m.match_model
    with torch.no_grad():
        mm.lm_head.bias.copy_(torch.tensor([1.0, 2.0, 4.0]))
    h = torch.zeros(3, 32)
    cand = torch.tensor([[5, V, 5], [5, 9, V], [6, 9, V]])  # [orders, T]
    cols, vals = mm.gate_values(h, cand)
    bonus = lm._bonus_logits(h, torch.zeros(V, 32), cols, vals)
    assert bonus[0, 5] == 3 and bonus[0, 6] == 4 and bonus[0].sum() == 7
    assert bonus[1, 9] == 6 and bonus[1].sum() == 6
    assert bonus[2, 5] == 1 and bonus[2].sum() == 1
    # the chunked CE with the sparse bonus (training path) == the materialized CE, value and gradients
    ids = _repetitive(2, 120, seed=11)
    lab = ids.clone()
    lab[:, :7] = -100
    ref = m(ids, labels=lab, return_logits=True)
    gref = torch.autograd.grad(ref.loss, list(m.parameters()), allow_unused=True)
    monkeypatch.setattr(lm, "CE_CHUNK_ELEMENTS", 50 * V)  # 50 rows per chunk
    out = m(ids, labels=lab, return_logits=False)
    assert out.logits is None
    g = torch.autograd.grad(out.loss, list(m.parameters()), allow_unused=True)
    assert torch.allclose(out.loss, ref.loss, atol=1e-6)
    for (n, _), a, b in zip(m.named_parameters(), g, gref):
        assert (a is None) == (b is None), n
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-4), n
    with torch.no_grad():  # evaluation path (no grad): chunked, same value
        assert torch.allclose(m(ids, labels=lab, return_logits=False).loss, ref.loss, atol=1e-6)


def test_disabled_sites_and_stats():
    m = _model(steps=0, enc=False, gate=False)
    assert m.match_model.sites == ["dec"] and m.match_model.lm_head is None
    ids = _repetitive(2, 60)
    out = m(ids, labels=ids)
    assert torch.isfinite(out.loss)
    m = _model(steps=0)
    out = m(ids, labels=ids)
    assert 0 < float(out.stats["match_cov/o1"]) <= 1 and "match_acc/o4" in out.stats
    with pytest.raises(AssertionError):
        build_model(_cfg(dec_layer=0))


# ----------------------------------------------------------------------------- accounting (fairness contract)
def test_param_count_and_flops_of_the_control_config():
    cfg = model_config_from_yaml("configs/evolve/models/g2-bpe-match.yaml")
    ref = model_config_from_yaml("configs/ladder/models/bpe_L03.yaml")
    assert {k: v for k, v in cfg.to_dict().items() if k != "match_model"} == \
        {k: v for k, v in ref.to_dict().items() if k != "match_model"}
    with torch.device("meta"):
        m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert count_params(_cfg()) == sum(p.numel() for p in _model(steps=0).parameters())
    assert 0 < count_params(cfg) / count_params(ref) - 1 < 0.02
    f = model_flops(cfg, seq_len=2048, bytes_per_token=4.145)["flops_per_byte"]
    f_ref = model_flops(ref, seq_len=2048, bytes_per_token=4.145)["flops_per_byte"]
    assert 0 < f / f_ref - 1 < 0.02
