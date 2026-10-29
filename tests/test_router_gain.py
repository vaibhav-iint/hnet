"""dc.router_gain_weight: REINFORCE on the hierarchy's measured gain trains the router; forward pass unchanged."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model


def _cfg(**dc):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.residual_init = "identity"  # a live bypass, so CE_res is meaningful at init
    for k, v in dc.items():
        setattr(cfg.dc, k, v)
    return cfg


def test_gain_objective_reaches_the_router_and_logs_the_gain():
    torch.manual_seed(0)
    m = build_model(_cfg(router_gain_weight=1.0))
    m.init_weights()
    m.train()
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    assert "router_gain/s0" in out.stats and "router_gain_std/s0" in out.stats
    router = [p for n, p in m.named_parameters() if "routing_module" in n]
    g_total = torch.autograd.grad(out.aux_loss, router, retain_graph=True, allow_unused=True)
    assert sum(float(g.abs().sum()) for g in g_total if g is not None) > 0
    # the gain term alone, isolated: zero the ratio-loss path by comparing with the same model without the term
    torch.manual_seed(0)
    m0 = build_model(_cfg(router_gain_weight=0.0))
    m0.init_weights()
    m0.train()
    out0 = m0(ids, labels=ids)
    g0 = torch.autograd.grad(out0.aux_loss, [p for n, p in m0.named_parameters() if "routing_module" in n],
                             allow_unused=True)
    a = sum(float(g.abs().sum()) for g in g_total if g is not None)
    b = sum(float(g.abs().sum()) for g in g0 if g is not None)
    assert abs(a - b) > 1e-6


def test_gain_objective_leaves_eval_forward_unchanged():
    ids = torch.randint(0, 256, (2, 200))
    outs = []
    for w in (0.0, 1.0):
        torch.manual_seed(0)
        m = build_model(_cfg(router_gain_weight=w)).eval()
        outs.append(m(ids, labels=ids).loss)
    assert torch.allclose(outs[0], outs[1])


def test_gain_running_statistics_update_only_in_training():
    torch.manual_seed(0)
    m = build_model(_cfg(router_gain_weight=1.0))
    m.init_weights()
    ids = torch.randint(0, 256, (2, 200))
    m.eval()
    m(ids, labels=ids)
    assert float(m.gain_mu) == 0.0
    m.train()
    m(ids, labels=ids)
    assert float(m.gain_mu) != 0.0
