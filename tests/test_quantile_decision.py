"""Rank-based boundary decisions (hnet.model.routing.decide)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.base import router_output
from hnet.model.routing.decide import QuantileDecider


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def test_training_selects_exactly_the_target_fraction_whatever_the_offset():
    torch.manual_seed(0)
    ctx = _ctx([500, 500])
    dec = QuantileDecider(target_ratio=3.0).train()
    for offset in (-3.0, 0.0, 2.5):  # a uniform shift of the router's logits must not change the fraction
        p = torch.sigmoid(torch.randn(1000) * 0.1 + offset)
        r = dec(router_output(p, ctx, "end"), ctx)
        frac = r.boundary_mask[ctx.position_ids != 0].float().mean()
        assert abs(frac - 1 / 3) < 0.01, (offset, frac)
        assert r.boundary_mask[ctx.position_ids == 0].all()
        # recentred: the threshold maps to 0.5, selected positions have p' > 0.5
        assert (r.p[r.boundary_mask & (ctx.position_ids != 0)] > 0.5).all()


def test_eval_uses_running_threshold():
    torch.manual_seed(0)
    ctx = _ctx([1000])
    dec = QuantileDecider(target_ratio=4.0, momentum=0.0).train()  # momentum 0: tau = last batch quantile
    p = torch.sigmoid(torch.randn(1000))
    dec(router_output(p, ctx, "end"), ctx)
    tau = dec.tau.clone()
    dec.eval()
    r = dec(router_output(p, ctx, "end"), ctx)
    assert torch.equal(dec.tau, tau)  # eval does not update
    expect = (p > tau) | (ctx.position_ids == 0)
    assert torch.equal(r.boundary_mask, expect)


def test_model_trains_with_quantile_decision_and_ratio_loss_is_inert():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
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
    assert "tau/s0" in out.stats
    m.eval()
    out2 = m(ids, labels=ids)
    assert torch.isfinite(out2.loss)
