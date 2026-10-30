"""decision_kwargs score features (hnet/model/routing/decide.py): a next-byte probe on the detached encoder output whose
entropy enters the rank decision as a pointwise feature (prior-free word-basin steering, evolve g3-route-02) and/or a
causal region mean (adaptive compression at an exact batch budget, evolve g3-route-01).

Checks (CLAUDE.md rules 1-5, 10; docs/EVOLVE.md guardrails 2, 3, 5): the default is off and bit-identical; turning the
features on leaves every other module's initialization unchanged; the probe's CE reaches only the probe and the byte
CE never reaches the probe (its features are detached) but does reach the gains; the batch budget is exact; the mixed
score's scale is anchored (no gain can saturate the confidences); the statistics survive constant, bimodal and
heavy-tailed inputs; the region feature moves boundaries between segments at a fixed total; eval decisions are causal
and packing-invariant; parameters and FLOPs are accounted for.
"""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.base import router_output
from hnet.model.routing.decide import H_FLOOR, QuantileDecider, causal_window_mean, trimmed_stats

VOCAB = [{"buckets": 512, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 64}]
PT = {"probe_rank": 8, "feat_point": 1.0}
RG = {"probe_rank": 8, "feat_region": 0.5, "region_window": 8}
BOTH = {"probe_rank": 8, "feat_point": 1.0, "feat_region": 0.5, "region_window": 8}


def _cfg(kw=None, match=False):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision, cfg.dc.chunk_vocab = "end_mlp", "quantile", VOCAB
    cfg.dc.decision_kwargs = kw
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 257}
    if match:
        cfg.match_model = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
    return cfg


def _model(kw=None, seed=0, match=False):
    torch.manual_seed(seed)
    m = build_model(_cfg(kw, match))
    m.init_weights()
    return m


def _text(B=2, L=320, seed=0):
    """English-like bytes: words of 2-7 letters separated by spaces (so whitespace statistics are meaningful)."""
    g = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(B):
        row = []
        while len(row) < L:
            n = int(torch.randint(2, 8, (1,), generator=g))
            row += torch.randint(97, 123, (n,), generator=g).tolist() + [0x20]
        rows.append(row[:L])
    return torch.tensor(rows)


# ----------------------------------------------------------------------------- default off, bit-identical
def test_default_is_off():
    d = QuantileDecider(4.0)
    assert d.router_probe is None and d.feat_names == []
    assert list(d.state_dict()) == ["mu", "sigma", "z_tau", "initialized"]
    assert not list(d.parameters())


@pytest.mark.parametrize("train", [True, False])
def test_explicit_defaults_are_bit_identical(train):
    ids = _text()
    outs = []
    for kw in (None, {"probe_rank": 0, "feat_point": 0.0, "feat_region": 0.0}):
        m = _model(kw).train(train)
        torch.manual_seed(1)
        o = m(ids, labels=ids)
        outs.append((m, o))
    (ma, a), (mb, b) = outs
    assert list(ma.state_dict()) == list(mb.state_dict())
    assert torch.equal(a.logits, b.logits) and torch.equal(a.loss, b.loss) and torch.equal(a.aux_loss, b.aux_loss)
    assert set(a.stats) == set(b.stats)


@pytest.mark.parametrize("kw", [PT, RG, BOTH], ids=["point", "region", "both"])
def test_features_leave_every_other_init_unchanged(kw):
    a, b = _model(None).state_dict(), _model(kw).state_dict()
    extra = set(b) - set(a)
    assert extra and all(k.startswith("backbone.decider.") for k in extra), extra
    for k, v in a.items():
        assert torch.equal(v, b[k]), k


def test_probe_is_a_stage0_option_only():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router, cfg.dc.decision = "end_mlp", "quantile"
    cfg.dc.decision_kwargs = [None, PT]
    with pytest.raises(AssertionError, match="stage-0"):
        build_model(cfg)


# ----------------------------------------------------------------------------- gradient paths (rule 1)
def _grads(m, loss, names):
    ps = [p for n, p in m.named_parameters() if any(n.startswith(x) for x in names)]
    gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
    return sum(float(g.pow(2).sum()) for g in gs if g is not None) ** 0.5, any(g is not None for g in gs)


@pytest.mark.parametrize("kw", [PT, RG], ids=["point", "region"])
def test_gradient_paths(kw):
    m = _model(kw, match=True).train()
    ids = _text()
    o = m(ids, labels=ids)
    probe = o.bpred[0].probe_loss
    assert probe is not None and probe.requires_grad
    # the probe CE reaches the probe and nothing else (it reads the encoder output detached)
    n, _ = _grads(m, probe, ["backbone.decider.router_probe."])
    assert n > 0
    for other in ["backbone.encoder.", "backbone.routing_module.", "embeddings.", "ngram_embeddings.", "lm_head.",
                  "backbone.decoder.", "backbone.main_network.", "match_model.", "backbone.decider.feat_gain"]:
        nn_, used = _grads(m, probe, [other])
        assert nn_ == 0.0, other
    # the byte CE reaches the gains (through the STE / EMA confidences) but never the probe (features detached)
    n_gain, _ = _grads(m, o.loss, ["backbone.decider.feat_gain"])
    assert n_gain > 0
    n_probe, used = _grads(m, o.loss, ["backbone.decider.router_probe."])
    assert n_probe == 0.0 and not used
    # the aux loss carries the probe CE (weight 1 by default)
    o0 = _model(None, match=True).train()(ids, labels=ids)
    ratio_term = m.config.dc.ratio_loss_alpha * sum(loss for loss, _ in m.backbone.aux_losses(o.bpred))
    assert float((o.aux_loss - ratio_term).detach() - m.backbone.chunk_vocab.pred_weight
                 * m.backbone.chunk_vocab.pred_loss(o.bpred[0].inner_out, o.bpred[0].vocab_tgt)) == pytest.approx(
        float(probe), rel=1e-4, abs=1e-4)
    assert "probe_ce/s0" in o.stats and "probe_ce/s0" not in o0.stats


def test_probe_targets_stay_inside_segments():
    d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0).train()
    ids = torch.tensor([5, 6, 7, 9, 9, 9])
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 3, 6], dtype=torch.int32), 3, 6)
    h = torch.randn(6, 8)
    _, loss, _ = d._features(h, ctx, ids)
    logp = torch.log_softmax(d.router_probe(h), -1)
    ref = -(logp[0, 6] + logp[1, 7] + logp[3, 9] + logp[4, 9]) / 4  # no target from position 2 (next segment)
    assert torch.allclose(loss, ref)


# ----------------------------------------------------------------------------- exact budget, anchored scale
def _decide(d, s, raw, ctx, train=True):
    """Run the decider on given router scores and given raw features (the probe bypassed)."""
    d.train(train)
    d._features = lambda h, c, i: (raw, None, raw[0])
    r = router_output(torch.sigmoid(s), ctx, "end", logit=s)
    return d(r, ctx, h=torch.zeros(1))


def test_exact_budget_and_anchored_scale():
    torch.manual_seed(0)
    T = 2000
    ctx = SeqCtx.single(T)
    s = torch.randn(T, requires_grad=True)
    raw = torch.rand(1, T) * 5
    for g in (0.3, 1.0, 100.0):
        d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=g)
        out = _decide(d, s, raw, ctx)
        free = ctx.position_ids != 0
        assert int(out.boundary_mask[free].sum()) == round(int(free.sum()) / 4.0)
        # the mixed score is restandardized: its spread does not grow with the gain, so p never saturates wholesale
        p = out.p[free]
        assert float(((p > 0.02) & (p < 0.98)).float().mean()) > 0.5, g
    # adding a constant to / scaling the router's scores leaves the decisions unchanged with features on
    d1 = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0)
    d2 = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0)
    a = _decide(d1, s.detach(), raw, ctx)
    b = _decide(d2, 7.0 * s.detach() + 3.0, raw, ctx)
    assert torch.equal(a.boundary_mask, b.boundary_mask)
    assert torch.allclose(a.p, b.p, atol=1e-3)  # up to the +1e-3 in the trimmed std


def test_gain_gradient_is_finite_and_nonzero():
    torch.manual_seed(0)
    T = 600
    ctx = SeqCtx.single(T)
    s = torch.randn(T)
    raw = torch.rand(1, T)
    d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0)
    out = _decide(d, s, raw, ctx)
    (out.selected_probs.squeeze(-1) * torch.randn(T)).sum().backward()
    assert d.feat_gain.grad is not None and torch.isfinite(d.feat_gain.grad).all() and float(d.feat_gain.grad.abs()) > 0


# ----------------------------------------------------------------------------- ugly distributions (rule 5)
@pytest.mark.parametrize("kind", ["constant", "bimodal_5_6", "heavy_tail", "outliers"])
def test_statistics_survive_ugly_distributions(kind):
    torch.manual_seed(0)
    T = 1200
    if kind == "constant":
        x = torch.full((T,), 2.0)
    elif kind == "bimodal_5_6":
        x = torch.where(torch.rand(T) < 5 / 6, torch.tensor(-3.0), torch.tensor(4.0)) + 1e-4 * torch.randn(T)
    elif kind == "heavy_tail":
        x = torch.distributions.StudentT(1.0).sample((T,))
    else:
        x = torch.randn(T)
        x[:5] = 1e6
    m, sd = trimmed_stats(x, 0.01)
    assert torch.isfinite(m) and torch.isfinite(sd) and float(sd) >= 1e-3
    z = (x - m) / sd
    assert torch.isfinite(z).all()
    # as router scores and as features, through the full decider (training and eval), with exact budget
    ctx = SeqCtx.single(T)
    d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0, feat_region=1.0, region_window=8)
    s = x.clone().requires_grad_(True)
    out = _decide(d, s, torch.stack([x.abs(), x.abs()]) if kind != "constant" else torch.stack([x, x]), ctx)
    assert torch.isfinite(out.p).all()
    assert int(out.boundary_mask[1:].sum()) == round((T - 1) / 4.0)
    out.selected_probs.sum().backward()
    assert torch.isfinite(s.grad).all() and torch.isfinite(d.feat_gain.grad).all()
    ev = _decide(d, x, torch.stack([x.abs(), x.abs()]), ctx, train=False)
    assert torch.isfinite(ev.p).all()


def test_constant_entropy_gives_zero_feature():
    d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_point=1.0)
    raw = torch.full((1, 50), 5.545) + 1e-6 * torch.randn(1, 50)
    f, _, sd = d._standardize_features(raw, torch.ones(50, dtype=torch.bool), True)
    assert float(sd) < H_FLOOR and float(f.abs().max()) < 1e-3


# ----------------------------------------------------------------------------- the region feature moves the budget
def test_causal_window_mean_matches_brute_force_and_packing():
    torch.manual_seed(0)
    x = torch.rand(25)
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 10, 25], dtype=torch.int32), 15, 25)
    got = causal_window_mean(x, ctx, 4)
    for t in range(25):
        s0 = 0 if t < 10 else 10
        lo = max(s0, t - 3)
        assert float(got[t]) == pytest.approx(float(x[lo:t + 1].mean()), rel=1e-6)
    sep = torch.cat([causal_window_mean(x[:10], SeqCtx.single(10), 4), causal_window_mean(x[10:], SeqCtx.single(15), 4)])
    assert torch.equal(got, sep)


def test_region_feature_reallocates_boundaries_at_a_fixed_total():
    """Two documents with identically distributed router scores; one is 'hard' (high probe entropy), one 'easy'
    (a copy the model predicts). Without the feature both get ~1/N; with it the hard one gets more, the total is
    exact. This is the adaptive compression a fixed BPE vocabulary cannot do."""
    torch.manual_seed(0)
    L = 1000
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, L, 2 * L], dtype=torch.int32), L, 2 * L)
    s = torch.randn(2 * L)
    H = torch.cat([2.0 + 0.3 * torch.rand(L), 0.2 + 0.3 * torch.rand(L)])  # nats: hard doc, then easy doc
    raw = causal_window_mean(H, ctx, 16)[None]
    counts = {}
    for g in (0.0001, 0.5, 1.0):
        d = QuantileDecider(4.0, d_model=8, probe_rank=4, feat_region=g, region_window=16)
        m = _decide(d, s, raw, ctx).boundary_mask
        counts[g] = (int(m[:L].sum()), int(m[L:].sum()))
        assert sum(counts[g]) == 2 + round((2 * L - 2) / 4.0)
    assert abs(counts[0.0001][0] - counts[0.0001][1]) < 40
    assert counts[0.5][0] > counts[0.5][1] + 100
    assert counts[1.0][0] > counts[0.5][0]


# ----------------------------------------------------------------------------- invariances (rule 10), accounting
def _trained(kw, match=False):
    m = _model(kw, match=match).train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    for i in range(3):
        ids = _text(seed=i)
        o = m(ids, labels=ids)
        opt.zero_grad()
        (o.loss + o.aux_loss).backward()
        opt.step()
    return m.eval()


@pytest.mark.parametrize("kw", [PT, RG, BOTH], ids=["point", "region", "both"])
def test_causal_and_packing_invariant_in_eval(kw):
    m = _trained(kw, match=True)
    ids = _text(1, 300, seed=7)
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    with torch.no_grad():
        a, b = m(ids).logits[0, :200], m(alt).logits[0, :200]
        assert torch.allclose(a, b, atol=1e-5)
        x, y = _text(1, 180, seed=8), _text(1, 220, seed=9)
        ox, oy = m(x), m(y)
        op = m(torch.cat([x[0], y[0]]), cu_seqlens=torch.tensor([0, 180, 400], dtype=torch.int32), max_seqlen=220)
    # decisions identical; logits to 1e-4: after 3 steps the probe is near-constant, its entropy std (1e-3 nats) sits
    # below the H_FLOOR = 0.25 floor, so the ~1e-6 packed-vs-separate float noise of the encoder output is amplified
    # 4x into the confidences
    assert torch.equal(op.bpred[0].boundary_mask, torch.cat([ox.bpred[0].boundary_mask, oy.bpred[0].boundary_mask]))
    lp = op.logits
    assert torch.allclose(lp[:180], ox.logits[0], atol=1e-4) and torch.allclose(lp[180:], oy.logits[0], atol=1e-4)


@pytest.mark.parametrize("kw", [PT, BOTH], ids=["point", "both"])
def test_params_and_flops_accounted(kw):
    cfg = _cfg(kw)
    with torch.device("meta"):
        m = build_model(cfg)
    assert sum(p.numel() for p in m.parameters()) == count_params(cfg)
    base = model_flops(_cfg(None), seq_len=2048)["flops_per_byte"]
    new = model_flops(cfg, seq_len=2048)["flops_per_byte"]
    d, r, V = cfg.d_model[0], kw["probe_rank"], cfg.vocab_size
    nf = int(kw.get("feat_point", 0) != 0) + int(kw.get("feat_region", 0) != 0)
    fwd = 2 * (d * r + r * V) + 5 * V + kw.get("region_window", 16) + 4 * nf  # matmuls + softmax / entropy / mix
    assert new - base == pytest.approx(3 * fwd, rel=1e-6)  # training = 3x forward (convention), once per byte
