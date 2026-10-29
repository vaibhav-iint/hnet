"""Self-consistency regularizer of the byte-level router (HNetConfig.dc.router_consistency_weight)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model
from hnet.model.ctx import SeqCtx


def _model(weight=1.0, n=4):
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.router_consistency_weight = weight
    cfg.dc.router_consistency_n = n
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    return m


def test_penalty_is_the_within_group_variance_of_the_logit():
    m = _model(n=2)
    ids = torch.tensor([0, 5, 6, 5, 6, 5, 6, 9])  # "56" ends at positions 2, 4, 6; others are singletons
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 8], dtype=torch.int32))
    logit = torch.tensor([0.0, 1.0, 2.0, 1.5, 5.0, 2.0, 8.0, 3.0])
    pen = m._router_consistency(logit, ids, ctx)
    g56 = torch.tensor([2.0, 5.0, 8.0])  # "56" ends at 2, 4, 6; "65" ends at 3, 5 (1.5, 2.0); position 0 (BOS) invalid
    g65 = torch.tensor([1.5, 2.0])
    expect = (((g56 - g56.mean()) ** 2).sum() + ((g65 - g65.mean()) ** 2).sum()) / 7  # singletons contribute 0
    assert abs(float(pen) - float(expect)) < 1e-6


def test_penalty_is_zero_when_the_router_is_a_function_of_the_last_n_bytes():
    m = _model(n=3)
    ids = torch.tensor([0, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 10], dtype=torch.int32))
    logit = torch.tensor([0.0, 0.0, 0.0, 7.0, 1.0, 2.0, 7.0, 1.0, 2.0, 7.0])  # same 3-gram -> same logit
    assert float(m._router_consistency(logit, ids, ctx)) == 0.0


def test_model_reports_and_trains_the_penalty():
    m = _model(weight=2.0)
    ids = torch.randint(0, 20, (2, 300))
    out = m(ids, labels=ids)
    assert "router_consistency/s0" in out.stats and float(out.stats["router_consistency/s0"]) >= 0
    (out.loss + out.aux_loss).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in m.named_parameters() if "routing_module" in n)
