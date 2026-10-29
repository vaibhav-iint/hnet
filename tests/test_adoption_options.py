"""dc.chunk_ste (router gradient from chunk-level losses) and dc.residual_gate (bypass gated by router margin)."""

import torch
from torch import nn

from hnet.config import model_config_from_yaml
from hnet.model import build_model


def _cfg(**kw):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.chunk_vocab = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]
    for k, v in kw.items():
        setattr(cfg.dc, k, v)
    return cfg


def _router_grad_from_chunk_loss(cfg):
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    for t in m.backbone.chunk_vocab.tables:  # the table is zero-initialised: at init the next-chunk CE has no gradient
        nn.init.normal_(t.weight, std=0.1)
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    # the chunk-level objective alone: next-chunk CE on the main network's output (computed before dechunking)
    st, r = m.backbone, out.bpred[0]
    chunk_loss = st.chunk_vocab.pred_loss(r.inner_out, r.vocab_tgt)
    grads = torch.autograd.grad(chunk_loss, [p for n, p in m.named_parameters() if "routing_module" in n],
                                allow_unused=True)
    return sum(float(g.abs().sum()) for g in grads if g is not None)


def test_chunk_ste_lets_the_next_chunk_loss_train_the_router():
    assert _router_grad_from_chunk_loss(_cfg(chunk_ste=False)) == 0.0  # paper: no path from inner outputs to p
    assert _router_grad_from_chunk_loss(_cfg(chunk_ste=True)) > 0.0


def test_chunk_ste_does_not_change_the_forward_pass():
    ids = torch.randint(0, 256, (2, 200))
    outs = []
    for flag in (False, True):
        torch.manual_seed(0)
        m = build_model(_cfg(chunk_ste=flag)).eval()
        outs.append(m(ids, labels=ids).loss)
    assert torch.allclose(outs[0], outs[1])


def test_residual_gate_scales_the_bypass_by_the_router_margin():
    ids = torch.randint(0, 256, (2, 200))
    torch.manual_seed(0)
    m = build_model(_cfg(residual_gate="margin", residual_init="identity")).eval()
    seen = {}
    m.backbone.decoder.register_forward_pre_hook(
        lambda mod, args, kwargs: (seen.setdefault("h", args[0].detach().clone()), None)[1], with_kwargs=True)
    out = m(ids, labels=ids)
    r = out.bpred[0]
    margin = 2 * (r.p - 0.5).abs().mean()
    assert 0 < float(margin) < 1
    assert torch.allclose(r.extra["residual_gate"].float(), margin.float(), atol=1e-5)
    # with the gate off the decoder input carries the full residual; with it on, the residual part is scaled
    torch.manual_seed(0)
    m2 = build_model(_cfg(residual_gate="none", residual_init="identity")).eval()
    seen2 = {}
    m2.backbone.decoder.register_forward_pre_hook(
        lambda mod, args, kwargs: (seen2.setdefault("h", args[0].detach().clone()), None)[1], with_kwargs=True)
    m2(ids, labels=ids)
    assert not torch.allclose(seen["h"], seen2["h"])
