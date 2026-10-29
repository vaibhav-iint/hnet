"""Rank-based boundary decisions on normalized scores (hnet.model.routing.decide)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.base import router_output
from hnet.model.routing.decide import QuantileDecider


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def test_training_selects_exactly_the_target_fraction_whatever_the_offset_or_scale():
    torch.manual_seed(0)
    ctx = _ctx([500, 500])
    dec = QuantileDecider(target_ratio=3.0).train()
    free = ctx.position_ids != 0
    for offset, scale in ((-3.0, 0.1), (0.0, 1.0), (2.5, 40.0)):  # shifts/scales of the logits change nothing
        logit = torch.randn(1000) * scale + offset
        r = dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
        assert int(r.boundary_mask[free].sum()) == round(int(free.sum()) / 3), (offset, scale)
        assert r.boundary_mask[~free].all()
        # recentred: the threshold maps to 0.5, selected positions have p' > 0.5, unselected p' < 0.5
        assert (r.p[r.boundary_mask & free] > 0.5).all()
        assert (r.p[~r.boundary_mask] < 0.5).all()
        # the recentred confidences do not depend on the raw scale: temp * std(z) = temp
        assert abs(float(r.logit.std()) - scale) < 0.1 * scale  # raw score passed through untouched


def test_saturated_scores_do_not_collapse_the_selection():
    """EXP-014: when more than 1/N of the p's are exactly 1.0 (logits > 17 in fp32) the old p-space quantile sat
    on the tie and selected nothing. Ranking fp32 logits selects exactly k positions regardless."""
    ctx = _ctx([900])
    free = ctx.position_ids != 0
    logit = torch.cat([torch.full((500,), 30.0), torch.linspace(-5, 5, 400)])[torch.randperm(900, generator=torch.Generator().manual_seed(0))]
    p = torch.sigmoid(logit)
    assert (p == 1.0).float().mean() > 0.5
    dec = QuantileDecider(target_ratio=3.0).train()
    r = dec(router_output(p, ctx, "end", logit=logit), ctx)
    assert int(r.boundary_mask[free].sum()) == round(int(free.sum()) / 3)
    assert r.boundary_mask[free][logit[free] == 30.0].float().mean() > 0.59  # the selected ones are all saturated
    # even without a logit (cosine routers), ties at the clamp are broken by rank, never "none selected"
    r2 = dec(router_output(p, ctx, "end"), ctx)
    assert int(r2.boundary_mask[free].sum()) == round(int(free.sum()) / 3)


def test_normalization_removes_shift_and_scale_gradients():
    """A gradient on the logits that is a pure common shift or a pure rescaling of the scores is zero: the forward
    pass does not depend on them, so nothing can inflate the router's logits (the EXP-014 drift)."""
    torch.manual_seed(0)
    ctx = _ctx([400])
    free = ctx.position_ids != 0
    logit = (torch.randn(400) * 3 + 1).requires_grad_(True)
    dec = QuantileDecider(target_ratio=3.0).train()
    r = dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
    # an arbitrary downstream loss on the confidences
    loss = (r.selected_probs.squeeze(-1)[free] * torch.randn(int(free.sum()))).sum()
    (g,) = torch.autograd.grad(loss, logit)
    gf, lf = g[free], logit.detach()[free]
    assert abs(float(gf.sum())) < 1e-4 * float(gf.abs().sum())  # no shift component
    assert abs(float((gf * (lf - lf.mean())).sum())) < 1e-4 * float((gf.abs() * (lf - lf.mean()).abs()).sum())  # no scale component
    # the un-normalized variant does have a shift component (the drift direction)
    dec0 = QuantileDecider(target_ratio=3.0, normalize=False).train()
    r0 = dec0(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
    torch.manual_seed(0)
    loss0 = (r0.selected_probs.squeeze(-1)[free] * torch.randn(int(free.sum()))).sum()
    (g0,) = torch.autograd.grad(loss0, logit)
    assert abs(float(g0[free].sum())) > 1e-3 * float(g0[free].abs().sum())


def test_eval_uses_running_threshold():
    torch.manual_seed(0)
    ctx = _ctx([1000])
    free = ctx.position_ids != 0
    dec = QuantileDecider(target_ratio=4.0, momentum=0.0).train()  # momentum 0: stats = last batch
    logit = torch.randn(1000)
    dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
    tau = dec.tau.clone()
    dec.eval()
    r = dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
    assert torch.equal(dec.tau, tau)  # eval does not update
    expect = ((logit > tau) & free) | ~free
    assert torch.equal(r.boundary_mask, expect)
    assert abs(int(expect[free].sum()) - int(free.sum()) / 4) <= 1  # same batch: same count (up to the midpoint)


def test_model_trains_with_quantile_decision_and_ratio_loss_is_inert():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.decision_kwargs = {"temp": 3.0}
    torch.manual_seed(0)
    m = build_model(cfg)
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    for s in (0, 1):
        assert abs(float(out.stats[f"compression/s{s}"]) - 3.0) < 0.15
    # under rank decisions F = 1/N exactly, so the ratio loss is the constant 1 (zero gradient)
    for s in (0, 1):
        assert abs(float(out.stats[f"aux/s{s}"]) - 1.0) < 0.02
    for k in ("tau/s0", "score_std/s0", "score_mu/s1"):
        assert k in out.stats
    m.eval()
    out2 = m(ids, labels=ids)
    assert torch.isfinite(out2.loss)
    assert abs(float(out2.stats["compression/s0"]) - 3.0) < 0.3  # running stats from one batch: same data


def test_robust_normalization_keeps_typical_confidences_away_from_half_under_outliers():
    """EXP-018: a few huge scores inflated the std, pushed every other confidence to ~0.5 and the hierarchy was
    never used. With median/MAD the bulk keeps its margins; the outliers saturate (clipped) on their own."""
    torch.manual_seed(0)
    ctx = _ctx([2000])
    free = ctx.position_ids != 0
    logit = torch.randn(2000)
    logit[:20] = 300.0  # 1% outliers
    for robust, lo in ((True, 0.25), (False, 0.0)):
        dec = QuantileDecider(target_ratio=3.0, robust=robust).train()
        r = dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
        bulk = free.clone()
        bulk[:20] = False
        margin = float((r.p[bulk] - 0.5).abs().mean())
        if robust:
            assert margin > lo, margin
        else:
            assert margin < 0.05, margin  # the failure mode
        assert int(r.boundary_mask[free].sum()) == round(int(free.sum()) / 3)


def test_bimodal_scores_do_not_blow_up_the_scale_or_the_gradient():
    """EXP-021: 5/6 of the scores identical (a word tokenizer's "not a boundary" logit) made the MAD zero; the
    trimmed std stays of order the gap between the modes and the gradient stays bounded."""
    ctx = _ctx([1200])
    free = ctx.position_ids != 0
    logit = torch.full((1200,), -5.0)
    logit[::6] = 3.0  # exactly 1/6 boundaries, everything else identical
    logit = (logit + 1e-3 * torch.randn(1200)).requires_grad_(True)
    dec = QuantileDecider(target_ratio=6.0).train()
    r = dec(router_output(torch.sigmoid(logit), ctx, "end", logit=logit), ctx)
    assert float(dec.sigma) > 1.0, float(dec.sigma)
    (g,) = torch.autograd.grad(r.selected_probs.sum(), logit)
    assert torch.isfinite(g).all() and float(g.abs().max()) < 10.0
    assert int(r.boundary_mask[free].sum()) == round(int(free.sum()) / 6)
