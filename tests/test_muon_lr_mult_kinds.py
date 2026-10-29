"""train.muon.lr_mult_kinds (evolve candidate g1-opt-01): per-stage LR multipliers on a subset of optimizer kinds.

Default (None) must leave every group, and therefore training, bit-identical; [embed, head, other] must give the
Muon matrices lr_mult 1 while the Adam part keeps the stage multipliers.
"""

import copy

import pytest
import torch

from hnet.config import HNetConfig
from hnet.model import build_model
from hnet.optim import build_optimizer, set_lr
from hnet.optim.muon import KINDS

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
    return [(g["kind"], g["stage"], g["weight_decay"], g["base_lr"], g["lr_mult"], g["lr"], tuple(g["names"]))
            for g in opt.param_groups]


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


def test_default_groups_and_training_unchanged():
    cfg = tiny_hnet_cfg()
    m0, m1 = make(cfg), make(cfg)
    o0 = build_optimizer(m0, cfg, tc())  # flag absent
    o1 = build_optimizer(m1, cfg, tc(lr_mult_kinds=list(KINDS)))  # every kind: same as the default
    assert summary(o0) == summary(o1)
    # the default applies the stage multipliers to every kind, Muon included
    mults = {g["stage"]: g["lr_mult"] for g in o0.param_groups}
    assert mults[0] == pytest.approx((4.6 * 48 / 32) ** 0.5) and mults[1] == pytest.approx((4.6 / 3) ** 0.5)
    assert all(g["lr_mult"] == mults[g["stage"]] for g in o0.param_groups)
    p0, p1 = train_steps(m0, o0), train_steps(m1, o1)
    for n in p0:
        assert torch.equal(p0[n], p1[n]), n


def test_adam_only_multipliers():
    cfg = tiny_hnet_cfg()
    m = make(cfg)
    ref = build_optimizer(make(cfg), cfg, tc()).param_groups
    opt = build_optimizer(m, cfg, tc(lr_mult_kinds=["embed", "head", "other"]))
    assert len(opt.param_groups) == len(ref)
    seen = []
    for g, r in zip(opt.param_groups, ref):
        assert g["names"] == r["names"] and g["base_lr"] == r["base_lr"] and g["kind"] == r["kind"]
        assert g["weight_decay"] == r["weight_decay"]
        if g["kind"] == "muon":
            assert g["lr_mult"] == 1.0 and g["lr"] == pytest.approx(0.02)
        else:
            assert g["lr_mult"] == r["lr_mult"]
        seen += [id(p) for p in g["params"]]
    assert sorted(seen) == sorted(id(p) for p in m.parameters())  # exact cover, same parameters
    set_lr(opt, 0.5)
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(g["base_lr"] * g["lr_mult"] * 0.5)
    stage0_muon = [g for g in opt.param_groups if g["kind"] == "muon" and g["stage"] == 0]
    assert stage0_muon and "backbone.residual_proj.weight" in stage0_muon[0]["names"]
    # trains, finite
    p = train_steps(m, opt)
    assert all(torch.isfinite(v).all() for v in p.values())


def test_rejects_unknown_kind():
    cfg = tiny_hnet_cfg()
    with pytest.raises(AssertionError):
        build_optimizer(make(cfg), cfg, tc(lr_mult_kinds=["muon", "router"]))


def test_forward_unchanged_by_flag():
    """Optimizer-only flag: the model and its forward pass do not depend on it (same init, same logits)."""
    cfg = tiny_hnet_cfg()
    m0, m1 = make(cfg), make(cfg)
    build_optimizer(m1, copy.deepcopy(cfg), tc(lr_mult_kinds=["embed"]))
    ids = batch(7)
    with torch.no_grad():
        assert torch.equal(m0.eval()(ids).logits, m1.eval()(ids).logits)
