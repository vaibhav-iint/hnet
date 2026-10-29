"""Whitespace tokenizer curriculum on the byte-level router (dc.router_prior_weight)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model


def _model():
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.router_prior_weight = 1.0
    cfg.dc.router_prior_frac = 0.1
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    return m


def test_prior_is_on_during_the_curriculum_and_off_after():
    m = _model().train()
    ids = torch.tensor([list(b"\x00the cat sat on the mat and ran away from the big dog quickly ok")] * 2)
    m.set_progress(0.0)
    out = m(ids, labels=ids)
    assert "router_prior/s0" in out.stats and 0 < float(out.stats["router_prior/s0"]) < 2
    full = float(out.aux_loss)
    m.set_progress(0.05)  # half-way through the curriculum: half weight
    half = float(m(ids, labels=ids).aux_loss)
    m.set_progress(0.2)
    out2 = m(ids, labels=ids)
    assert "router_prior/s0" not in out2.stats
    base = float(out2.aux_loss)
    assert abs((full - base) - 2 * (half - base)) < 1e-3 * max(1.0, abs(full))
    m.eval()
    assert "router_prior/s0" not in m(ids, labels=ids).stats


def test_prior_gradient_pushes_whitespace_logits_up():
    m = _model().train()
    m.set_progress(0.0)
    ids = torch.tensor([list(b"\x00aa bb cc dd ee ff gg hh ii jj kk ll")])
    out = m(ids, labels=ids)
    (out.aux_loss - out.aux_loss.detach() + 0 * out.loss).backward()  # aux only (ratio loss is constant under rank decisions)
    before = out.bpred[0].logit.detach().clone()
    opt = torch.optim.SGD([p for n, p in m.named_parameters() if "routing_module" in n], lr=0.5)
    opt.step()
    after = m(ids, labels=ids).bpred[0].logit.detach()
    sp = (ids.reshape(-1) == 0x20)
    assert float((after - before)[sp].mean()) > float((after - before)[~sp].mean())
