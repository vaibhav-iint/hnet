"""train.muon.residual_wd (evolve candidate g1-opt-02): a scale anchor for the H-Net residual bypass under Muon.

Default (None) must leave every group, and therefore training, bit-identical. With a value, the residual_proj
weights move to their own Muon group (same LR and multiplier) with that decoupled weight decay, whose fixed point
under Muon's orthogonalized updates is a spectral norm of about 1/wd.
"""

import pytest
import torch
from torch import nn

from hnet.config import HNetConfig
from hnet.model import build_model
from hnet.optim import build_optimizer, set_lr
from hnet.optim.muon import MuonAdamW

torch.set_num_threads(1)


def tiny_hnet_cfg() -> HNetConfig:
    return HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T1"], "m1"], d_model=[32, 48], d_intermediate=[0, 96],
        ssm=dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8),
        attn=dict(num_heads=2, rotary_emb_dim=[8, 12], window_size=[-1, -1]),
        dc=dict(target_ratios=[3.0], router="end_mlp", decision="quantile"),
        lr_multiplier="auto",
    ))


def make(cfg, seed=0):
    torch.manual_seed(seed)
    m = build_model(cfg)
    m.init_weights()
    return m


def tc(**muon):
    return {"optimizer": "muon", "weight_decay": 0.1,
            "muon": {"lr": 0.02, "lr_embed": 0.003, "lr_head": 0.003, "lr_other": 0.003, **muon}}


def summary(opt):
    return [(g["kind"], g["stage"], g["weight_decay"], g["base_lr"], g["lr_mult"], g["lr"], g.get("role"),
             tuple(g["names"])) for g in opt.param_groups]


def batch(seed, n=96):
    g = torch.Generator().manual_seed(seed)
    words = [b"the ", b"cat ", b"sat, ", b"on ", b"a ", b"mat. ", b"x1 "]
    out = b"".join(words[i] for i in torch.randint(0, len(words), (n,), generator=g).tolist())
    return torch.tensor(list(out[: n + 1]))


def train_steps(model, opt, steps=3):
    for s in range(steps):
        set_lr(opt, 0.5 + 0.1 * s)
        ids = batch(s)
        out = model(ids[:-1], labels=ids[1:], return_logits=False)
        (out.loss + out.aux_loss).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return {n: p.detach().clone() for n, p in model.named_parameters()}


RES = "backbone.residual_proj.weight"


def test_default_groups_and_training_unchanged():
    cfg = tiny_hnet_cfg()
    m0, m1 = make(cfg), make(cfg)
    o0 = build_optimizer(m0, cfg, tc())  # flag absent
    o1 = build_optimizer(m1, cfg, tc(residual_wd=None))
    assert summary(o0) == summary(o1)
    assert not any(g.get("role") for g in o0.param_groups)
    g_res = [g for g in o0.param_groups if RES in g["names"]]
    assert len(g_res) == 1 and g_res[0]["kind"] == "muon" and g_res[0]["weight_decay"] == 0.0
    p0, p1 = train_steps(m0, o0), train_steps(m1, o1)
    for n in p0:
        assert torch.equal(p0[n], p1[n]), n


def test_residual_group_split():
    cfg = tiny_hnet_cfg()
    m = make(cfg)
    ref = build_optimizer(make(cfg), cfg, tc()).param_groups
    opt = build_optimizer(m, cfg, tc(residual_wd=1.0))
    assert len(opt.param_groups) == len(ref) + 1
    res = [g for g in opt.param_groups if g.get("role") == "residual"]
    assert len(res) == 1 and res[0]["names"] == [RES] and res[0]["use_muon"] and res[0]["weight_decay"] == 1.0
    src = next(g for g in ref if RES in g["names"])
    for k in ("base_lr", "lr_mult", "lr", "stage", "momentum", "nesterov"):
        assert res[0][k] == src[k], k
    # every other parameter keeps its group's settings; exact cover
    ref_by_name = {n: g for g in ref for n in g["names"]}
    seen = []
    for g in opt.param_groups:
        seen += [id(p) for p in g["params"]]
        if g.get("role"):
            continue
        for n in g["names"]:
            r = ref_by_name[n]
            assert (g["kind"], g["stage"], g["weight_decay"], g["lr_mult"], g["base_lr"]) == \
                   (r["kind"], r["stage"], r["weight_decay"], r["lr_mult"], r["base_lr"]), n
    assert sorted(seen) == sorted(id(p) for p in m.parameters())
    p = train_steps(m, opt)
    assert all(torch.isfinite(v).all() for v in p.values())


def test_weight_decay_fixed_point_under_muon():
    """Constant-direction gradient on a zero-init square matrix (the bypass at init): without decay the spectral
    norm grows like sum(lr); with decoupled decay wd it settles at ~1/wd, whatever the LR."""
    torch.manual_seed(0)
    G = torch.randn(24, 24)
    norms = {}
    for wd, lr in ((0.0, 0.05), (1.0, 0.05), (1.0, 0.02), (0.5, 0.05)):
        p = nn.Parameter(torch.zeros(24, 24))
        opt = MuonAdamW([dict(params=[p], use_muon=True, lr=lr, weight_decay=wd, momentum_warmup_steps=0)])
        for _ in range(400):
            p.grad = G.clone()
            opt.step()
        norms[(wd, lr)] = torch.linalg.svdvals(p.detach())[0].item()
    assert norms[(0.0, 0.05)] > 0.6 * 400 * 0.05  # unanchored: grows linearly in sum(lr)
    assert 0.6 < norms[(1.0, 0.05)] < 1.25 and 0.6 < norms[(1.0, 0.02)] < 1.25
    assert 1.2 < norms[(0.5, 0.05)] < 2.5


def test_forward_unchanged_by_flag():
    cfg = tiny_hnet_cfg()
    m0, m1 = make(cfg), make(cfg)
    build_optimizer(m1, cfg, tc(residual_wd=1.0))
    ids = batch(7)
    with torch.no_grad():
        assert torch.equal(m0.eval()(ids).logits, m1.eval()(ids).logits)


def test_model_config_route():
    """model.residual_proj_wd (where the fairness gate expects H-Net-only LR knobs, like lr_multiplier) feeds
    train.muon.residual_wd; both routes build the same groups and the same training."""
    cfg = tiny_hnet_cfg()
    cfg_m = tiny_hnet_cfg()
    cfg_m.residual_proj_wd = 1.0
    m0, m1 = make(cfg), make(cfg_m)
    o0 = build_optimizer(m0, cfg, tc(residual_wd=1.0))
    o1 = build_optimizer(m1, cfg_m, tc())
    assert summary(o0) == summary(o1)
    p0, p1 = train_steps(m0, o0), train_steps(m1, o1)
    for n in p0:
        assert torch.equal(p0[n], p1[n]), n
    with pytest.raises(AssertionError):
        build_optimizer(make(cfg_m), cfg_m, tc(residual_wd=0.5))
    with pytest.raises(AssertionError):
        build_optimizer(make(cfg_m), cfg_m, {"optimizer": "adamw", "lr": 1e-3})
