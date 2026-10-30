"""Copy-aware compression (HNetConfig.copy_route, hnet/model/chunk_copy.py CopyRoute, evolve g3-copy-02): exact copy
state vs brute force, default path unchanged, zero bias is bit-identical, the boundaries leave verified copies at an
unchanged boundary count, causality, packing, gradient paths (CLAUDE.md rules 1, 2, 10), accounting."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.chunk_copy import N_STATES, copy_state
from hnet.model.ctx import SeqCtx
from hnet.model.match import NONE_ID, match_candidates

MM = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256, "suffix_lens": [2, 4]}]


def _repetitive(n_rows: int, length: int, seed: int = 0) -> torch.Tensor:
    """Rows built from 6 random 24-byte 'lines' in random order: long verified copies inside a line, breaks at the
    line ends (code-like repetition)."""
    g = torch.Generator().manual_seed(seed)
    lines = torch.randint(97, 123, (6, 24), generator=g)
    rows = []
    for _ in range(n_rows):
        idx = torch.randint(0, 6, (length // 24 + 1,), generator=g)
        rows.append(lines[idx].reshape(-1)[:length])
    return torch.stack(rows)


def _cfg(route=None):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision, cfg.dc.chunk_vocab = "end_mlp", "quantile", VOCAB
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 512}
    cfg.match_model = dict(MM)
    cfg.copy_route = dict(route) if route is not None else {}
    return cfg


def _model(route=None, steps=0):
    torch.manual_seed(0)
    m = build_model(_cfg(route))
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


ROUTE = {"kappa": 2.5, "kappa_break": 1.0, "run": 3, "min_order": 4}


# ----------------------------------------------------------------------------- exact copy state
def test_copy_state_matches_brute_force():
    ids = _repetitive(1, 300, seed=3)[0]
    ids[150:160] = 33  # a few breaks
    cu = torch.tensor([0, 120, 300], dtype=torch.int32)
    ctx = SeqCtx.from_cu_seqlens(cu, 180, 300)
    orders = [2, 4, 8]
    C = match_candidates(ids, ctx, orders)
    st = copy_state(ids, C, ctx, orders, min_order=4, run_min=3)
    x = ids.tolist()
    starts = {0, 120}

    def longest(t):
        c = NONE_ID
        for i, n in enumerate(orders):
            if n >= 4 and int(C[i, t]) != NONE_ID:
                c = int(C[i, t])
        return c

    run = 0
    for t in range(300):
        pred = NONE_ID if (t in starts or t == 0) else longest(t - 1)
        hit = pred != NONE_ID and pred == x[t]
        prev_run = run if t not in starts else 0
        run = run + 1 if hit else 0
        brk = (t not in starts) and prev_run >= 3 and pred != NONE_ID and not hit
        assert int(st["run"][t]) == run, t
        assert bool(st["brk"][t]) == brk, t
        assert bool(st["has_next"][t]) == (longest(t) != NONE_ID), t
        assert bool(st["in_copy"][t]) == (run >= 3 and longest(t) != NONE_ID), t
        rb = 0 if run == 0 else min(5, run.bit_length())
        assert int(st["state"][t]) == (N_STATES - 1 if brk else 2 * rb + int(longest(t) != NONE_ID)), t
    assert st["in_copy"].float().mean() > 0.2 and st["brk"].any()


# ----------------------------------------------------------------------------- default path / zero bias
def test_default_model_unchanged_and_zero_bias_is_bit_identical():
    off = _model()
    on = _model({"kappa": 0.0, "kappa_break": 0.0, "run": 3, "min_order": 4, "learn": True})
    assert not any(k.startswith("copy_route") for k in off.state_dict())
    common = {k: v for k, v in on.state_dict().items() if not k.startswith("copy_route")}
    assert common.keys() == off.state_dict().keys()
    for k, v in off.state_dict().items():
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
        assert torch.equal(a.bpred[0].boundary_mask, b.bpred[0].boundary_mask)
    assert float(b.stats["copy_route/in_copy"]) > 0.2, "the test input must exercise the copy state"


# ----------------------------------------------------------------------------- the mechanism
def test_boundaries_leave_verified_copies_at_an_unchanged_count():
    ids = _repetitive(2, 512, seed=11)
    off, on = _model(), _model(dict(ROUTE, kappa=6.0))
    a, b = off(ids, labels=ids), on(ids, labels=ids)  # training mode: exact rank decisions
    ma, mb = a.bpred[0].boundary_mask, b.bpred[0].boundary_mask
    assert int(ma.sum()) == int(mb.sum())  # same number of main-network positions = same FLOPs
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 512, 1024], dtype=torch.int32), 512, 1024)
    flat = ids.reshape(-1)
    st = copy_state(flat, on.match_model.candidates(flat, ctx), ctx, MM["orders"], 4, 3)
    inc, brk = st["in_copy"], st["brk"]
    frac_off = float((ma & inc).sum()) / float(ma.sum())
    frac_on = float((mb & inc).sum()) / float(mb.sum())
    assert frac_off > 0.2 and frac_on < 0.25 * frac_off, (frac_off, frac_on)
    assert abs(float(b.stats["copy_route/bnd_in_copy"]) - frac_on) < 1e-6
    # the freed boundaries go to novel text: its compression drops below the target ratio
    novel_off = float((~inc).sum()) / float((ma & ~inc).sum())
    assert float(b.stats["copy_route/novel_comp"]) < novel_off
    assert float((mb & brk).sum()) >= float((ma & brk).sum())  # copy breaks are cut at least as often


# ----------------------------------------------------------------------------- invariances (rule 10)
def _trained_on():
    m = _model(ROUTE, steps=2)
    with torch.no_grad():
        m.copy_route.resid.copy_(0.3 * torch.randn(N_STATES, generator=torch.Generator().manual_seed(2)))
    return m.eval()


def test_future_bytes_do_not_change_earlier_logits():
    m = _trained_on()
    ids = _repetitive(1, 300, seed=5)
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    with torch.no_grad():
        oa, ob = m(ids), m(alt)
    assert float(oa.stats["copy_route/in_copy"]) > 0.2
    assert torch.equal(oa.bpred[0].boundary_mask[:200], ob.bpred[0].boundary_mask[:200])
    assert torch.allclose(oa.logits[0, :200], ob.logits[0, :200], atol=1e-5), \
        (oa.logits[0, :200] - ob.logits[0, :200]).abs().max()


def test_packed_segments_equal_separate_sequences_even_when_segments_repeat_each_other():
    m = _trained_on()
    a = _repetitive(1, 180, seed=7)
    b = torch.cat([a[:, :120], _repetitive(1, 100, seed=8)], 1)
    with torch.no_grad():
        la = m(a).logits[0]
        lb = m(b).logits[0]
        packed = torch.cat([a[0], b[0]])
        cu = torch.tensor([0, 180, 400], dtype=torch.int32)
        lp = m(packed, cu_seqlens=cu, max_seqlen=220).logits
    assert torch.allclose(lp[:180], la, atol=1e-5), (lp[:180] - la).abs().max()
    assert torch.allclose(lp[180:], lb, atol=1e-5), (lp[180:] - lb).abs().max()


# ----------------------------------------------------------------------------- gradient paths (rule 1)
@pytest.mark.parametrize("kappa", [0.0, 2.5])
def test_byte_ce_reaches_the_learned_residual_at_init(kappa):
    m = _model(dict(ROUTE, kappa=kappa))
    ids = _repetitive(2, 256)
    out = m(ids, labels=ids)
    (g,) = torch.autograd.grad(out.loss, [m.copy_route.resid], allow_unused=True)
    assert g is not None and float(g.norm()) > 0
    assert int((g != 0).sum()) >= 4  # several copy states are credited, not just one


def test_copy_state_carries_no_gradient():
    ids = _repetitive(1, 128)[0]
    ctx = SeqCtx.single(128)
    st = copy_state(ids, match_candidates(ids, ctx, [2, 4, 8]), ctx, [2, 4, 8], 4, 3)
    assert not any(v.requires_grad for v in st.values())


def test_copy_route_requires_match_model_and_quantile():
    cfg = _cfg(ROUTE)
    cfg.match_model = {}
    with pytest.raises(AssertionError):
        build_model(cfg)
    cfg = _cfg(ROUTE)
    cfg.dc.decision = "threshold"
    with pytest.raises(AssertionError):
        build_model(cfg)


# ----------------------------------------------------------------------------- accounting (fairness contract)
def test_param_count_and_flops_match_the_module():
    cfg = _cfg(ROUTE)
    m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert m.copy_route.num_params() == N_STATES


def test_param_count_and_flops_of_the_candidate_config():
    cfg = model_config_from_yaml("configs/evolve/models/g3-copy-02.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_ws_L03.yaml")
    with torch.device("meta"):
        m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    assert count_params(cfg) <= 20_023_040
    f = model_flops(cfg, seq_len=8512)["flops_per_byte"]
    f_ref = model_flops(ref, seq_len=8512)["flops_per_byte"]
    assert abs(f / f_ref - 1) < 0.02
    assert cfg.copy_route and not cfg.chunk_copy
