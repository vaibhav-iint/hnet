"""Annealed Gumbel-top-k exploration of rank decisions (dc.decision_kwargs.explore, routing/decide.py).

Default off and bit-identical (no random numbers drawn); training-only (eval forward identical); compression exact;
the clean confidence and running statistics are unchanged, so noise-selected positions are tentative (p' < 0.5);
the noise reorders only positions near the threshold (a decided, bimodal router is left alone).
"""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.base import router_output
from hnet.model.routing.decide import QuantileDecider


def _ctx(T):
    return SeqCtx.from_cu_seqlens(torch.tensor([0, T], dtype=torch.int32), T, T)


def _r(scores):
    ctx = _ctx(scores.numel())
    return router_output(torch.sigmoid(scores), ctx, "end", logit=scores), ctx


def test_default_off():
    assert QuantileDecider(4.0).explore == 0.0
    assert QuantileDecider(4.0, explore=0.5).explore_tau(1.0) == 0.0


def test_anneal_schedule():
    d = QuantileDecider(4.0, explore=0.6, explore_frac=0.3)
    assert d.explore_tau(0.0) == pytest.approx(0.6)
    assert d.explore_tau(0.15) == pytest.approx(0.3)
    assert d.explore_tau(0.3) == 0.0 and d.explore_tau(0.9) == 0.0
    assert QuantileDecider(4.0, explore=0.6, explore_frac=0.0).explore_tau(0.9) == pytest.approx(0.6)


def test_exact_compression_clean_statistics_and_tentative_boundaries():
    torch.manual_seed(0)
    s = torch.randn(4001)
    clean, noisy = QuantileDecider(4.0).train(), QuantileDecider(4.0, explore=1.0).train()
    r, ctx = _r(s)
    a = clean(r, ctx)
    torch.manual_seed(1)
    b = noisy(r, ctx, progress=0.0)
    assert int(a.boundary_mask.sum()) == int(b.boundary_mask.sum())  # exact compression
    assert torch.equal(a.p, b.p)  # confidence from the clean scores
    for name in ("mu", "sigma", "z_tau"):
        assert torch.equal(getattr(clean, name), getattr(noisy, name))
    moved = b.boundary_mask & ~a.boundary_mask
    assert int(moved.sum()) > 0
    assert float(b.p[moved].max()) < 0.5  # selected only by the noise: tentative
    assert torch.equal(b.selected_probs[moved, 0], b.p[moved])
    assert 0 < float(b.extra["explore_swap"]) < 1 and float(b.extra["explore_tau"]) == 1.0


def test_decided_router_is_left_alone():
    """Bimodal scores far from the threshold: the noise moves (almost) nothing; near-threshold scores move."""
    T = 4001
    s = torch.full((T,), -3.0)
    s[1::4] = 3.0  # exactly 1/4 of the free positions are decided boundaries
    s = s + 0.01 * torch.randn(T)
    d = QuantileDecider(4.0, explore=0.3).train()
    r, ctx = _r(s)
    torch.manual_seed(0)
    out = d(r, ctx, progress=0.0)
    assert float(out.extra["explore_swap"]) < 0.02  # only Gumbel's exponential right tail moves a few
    s2 = 0.01 * torch.randn(T)  # undecided: everything at the threshold
    out2 = QuantileDecider(4.0, explore=0.3).train()(*_r(s2), progress=0.0)
    assert float(out2.extra["explore_swap"]) > 0.1  # Gaussian (undecided) scores: ~19% of boundaries move at 0.3


def _model(**dk):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.decision_kwargs = dk or None
    cfg.dc.chunk_vocab = [{"buckets": 512, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 64}]
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    return m


@pytest.mark.parametrize("train,progress", [(True, 0.5), (False, 0.0)])
def test_forward_identical_when_off(train, progress):
    """Default config vs explore on but annealed away (train) or in eval mode: bit-identical outputs, same RNG."""
    ids = torch.randint(0, 256, (2, 300), generator=torch.Generator().manual_seed(0))
    outs = []
    for dk in ({}, {"explore": 0.5, "explore_frac": 0.3}):
        m = _model(**dk).train(train)
        m.set_progress(progress)
        torch.manual_seed(7)
        o = m(ids, labels=ids)
        outs.append((o, torch.get_rng_state()))
    (a, ra), (b, rb) = outs
    assert torch.equal(a.logits, b.logits) and torch.equal(a.loss, b.loss) and torch.equal(a.aux_loss, b.aux_loss)
    assert torch.equal(ra, rb)


def test_exploration_changes_training_segmentation_only_at_start():
    ids = torch.randint(0, 256, (2, 400), generator=torch.Generator().manual_seed(0))
    m0, m1 = _model().train(), _model(explore=1.0, explore_frac=0.3).train()
    for m in (m0, m1):
        m.set_progress(0.0)
    torch.manual_seed(3)
    a = m0(ids, labels=ids)
    torch.manual_seed(3)
    b = m1(ids, labels=ids)
    ma, mb = a.bpred[0].boundary_mask, b.bpred[0].boundary_mask
    assert int(ma.sum()) == int(mb.sum()) and not torch.equal(ma, mb)
    assert torch.isfinite(b.loss)
    router = [p for n, p in m1.named_parameters() if "routing_module" in n]
    g = torch.autograd.grad(b.loss, router, allow_unused=True)
    assert sum(float(x.norm()) for x in g if x is not None) > 0  # STE / EMA still train the router
