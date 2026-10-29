"""dc.fusion = "balanced" (scale-balanced decoder input; candidate g1-arch-01).

    add (default):  h_dec = dechunk(z) * STE(c) + residual_proj(h_enc)
    balanced:       h_dec = g_h * RMSNorm(dechunk(z)) * STE(c) + g_r * RMSNorm(residual_proj(h_enc)),  g_h = 1, g_r = 0 at init
"""

import pytest
import torch

import test_invariances as inv
from hnet.config import model_config_from_yaml
from hnet.flops import count_params
from hnet.model import build_model

VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]


def _cfg(**kw):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision, cfg.dc.chunk_vocab = "end_mlp", "quantile", VOCAB
    for k, v in kw.items():
        setattr(cfg.dc, k, v)
    return cfg


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = build_model(_cfg(**kw))
    m.init_weights()
    return m


def _decoder_input(m, ids):
    seen = {}
    hook = m.backbone.decoder.register_forward_pre_hook(
        lambda mod, args: seen.__setitem__("h", args[0].detach().clone()))
    out = m(ids, labels=ids)
    hook.remove()
    return seen["h"], out


def test_default_is_add_and_adds_nothing():
    """The default model has no new parameters and its forward / backward pass is the same as with fusion="add"
    given explicitly (bit-identical logits and gradients)."""
    assert _cfg().dc.fusion == "add"
    ids = torch.randint(0, 256, (2, 300), generator=torch.Generator().manual_seed(1))
    res = []
    for kw in ({}, {"fusion": "add"}):
        m = _model(**kw).train()
        assert not any("fuse_" in n for n, _ in m.named_parameters())
        out = m(ids, labels=ids)
        (out.loss + out.aux_loss).backward()
        res.append((out.logits.detach(), [p.grad for _, p in sorted(m.named_parameters())],
                    sorted(m.state_dict())))
    assert torch.equal(res[0][0], res[1][0])
    assert res[0][2] == res[1][2]
    assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(res[0][1], res[1][1]))


def test_bypass_is_closed_at_init_including_the_bias():
    """With g_r = 0 the decoder sees only the normalized hierarchy at init (the add fusion leaks the live
    residual_proj bias, rms ~0.06)."""
    ids = torch.randint(0, 256, (2, 300))
    m = _model(fusion="balanced").eval()
    assert torch.count_nonzero(m.backbone.fuse_bypass.weight) == 0
    assert torch.equal(m.backbone.fuse_hier.weight, torch.ones_like(m.backbone.fuse_hier.weight))
    h, out = _decoder_input(m, ids)
    rms = h.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)  # unit-rms hierarchy, nothing else
    assert float(out.stats["residual_rms/s0"]) == 0.0
    m_add = _model().eval()
    assert float(m_add(ids).stats["residual_rms/s0"]) > 0.0  # the bias the add fusion lets through


def test_bypass_scale_is_set_by_its_gain_not_by_residual_proj():
    """The point of the mechanism: growing residual_proj (what Muon does to it in training) does not grow the
    bypass. Scaling its weight and bias by 10 leaves the forward pass unchanged; scaling g_r does change it."""
    ids = torch.randint(0, 256, (2, 300))
    m = _model(fusion="balanced").eval()
    with torch.no_grad():
        m.backbone.fuse_bypass.weight.fill_(0.7)
        torch.nn.init.normal_(m.backbone.residual_proj.weight, std=0.05)
    h0, _ = _decoder_input(m, ids)
    with torch.no_grad():
        m.backbone.residual_proj.weight.mul_(10.0)
        m.backbone.residual_proj.bias.mul_(10.0)
    h1, _ = _decoder_input(m, ids)
    assert torch.allclose(h0, h1, atol=2e-3)  # equal up to the norm eps (1e-5 against a small mean square)
    with torch.no_grad():
        m.backbone.fuse_bypass.weight.mul_(3.0)
    h2, _ = _decoder_input(m, ids)
    assert not torch.allclose(h0, h2, atol=1e-3)


def _grads(m, ids, loss_fn=lambda o: o.loss):
    out = m(ids, labels=ids)
    named = dict(m.named_parameters())
    groups = {
        "router": [p for n, p in named.items() if "routing_module" in n],
        "residual_proj": [named["backbone.residual_proj.weight"]],
        "g_h": [named["backbone.fuse_hier.weight"]],
        "g_r": [named["backbone.fuse_bypass.weight"]],
        "main": [p for n, p in named.items() if n.startswith("backbone.main_network.")],
        "encoder": [p for n, p in named.items() if n.startswith("backbone.encoder.")],
    }
    loss = loss_fn(out)
    res = {}
    for k, ps in groups.items():
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        res[k] = sum(float(g.abs().sum()) for g in gs if g is not None)
    return res


def test_gradient_paths_at_init_and_after_one_step():
    """Rule 1. Byte CE reaches the router through the STE (the norm sits before it), both gains, the main network
    and the encoder. At init g_r = 0, so residual_proj gets no gradient (closed bypass); one optimizer step on g_r
    opens the path."""
    ids = torch.randint(0, 256, (2, 300))
    m = _model(fusion="balanced").train()
    g = _grads(m, ids)
    assert g["router"] > 0 and g["g_h"] > 0 and g["g_r"] > 0 and g["main"] > 0 and g["encoder"] > 0
    assert g["residual_proj"] == 0.0
    opt = torch.optim.Adam([m.backbone.fuse_bypass.weight], lr=1e-2)  # Adam-like first step on the gain only
    out = m(ids, labels=ids)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
    g = _grads(m, ids)
    assert g["residual_proj"] > 0 and g["router"] > 0


def test_norm_after_the_ste_would_cut_the_router():
    """Why the norm is applied before the STE: RMSNorm is invariant to a per-position scale, so the STE's
    confidence factor would get exactly zero gradient through it."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(50, 16, dtype=torch.float64, generator=g)
    c = (0.5 + 0.5 * torch.rand(50, 1, dtype=torch.float64, generator=g)).requires_grad_()  # confidences in [0.5, 1]
    from hnet.model.norm import RMSNorm

    n = RMSNorm(16, eps=0.0).double()
    w = torch.randn(50, 16, dtype=torch.float64, generator=g)
    (g_after,) = torch.autograd.grad((n(x * c) * w).sum(), c)
    (g_before,) = torch.autograd.grad((n(x) * c * w).sum(), c)
    assert float(g_after.abs().max()) < 1e-4 * float(g_before.abs().max())  # zero up to the norm's fp32 compute


def test_param_count_matches_flops_and_l03_budget():
    cfg = _cfg(fusion="balanced")
    m = build_model(cfg)
    assert sum(p.numel() for p in m.parameters()) == count_params(cfg)
    assert count_params(cfg) == count_params(_cfg()) + 2 * cfg.d_model[0]
    l03 = model_config_from_yaml("configs/evolve/models/g1-arch-01.yaml")
    assert l03.dc.fusion == "balanced"
    assert count_params(l03) <= 20_023_040  # bpe_L03


@pytest.mark.parametrize("ngram", [False, True])
def test_invariances_with_balanced_fusion(monkeypatch, ngram):
    monkeypatch.setitem(inv.VARIANTS, "balanced", {**inv.VARIANTS["end_mlp_quantile_vocab"], "fusion": "balanced"})
    inv.test_future_bytes_do_not_change_earlier_logits("balanced", ngram)
    inv.test_packed_segments_equal_separate_sequences("balanced", ngram)
