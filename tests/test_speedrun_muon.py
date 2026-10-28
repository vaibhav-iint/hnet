"""Muon (hnet/optim/muon.py): Newton-Schulz orthogonalization, convergence, parameter routing."""

import pytest
import torch
from torch import nn

from hnet import registry
from hnet.config import DenseConfig, IGNORE_INDEX
from hnet.model.dense_modded import ModdedGPT
from hnet.optim.muon import MuonAdamW, build_muon_optimizer, classify_params, zeropower_via_newtonschulz5

torch.set_num_threads(1)


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (48, 48), (96, 8)])
def test_newton_schulz_is_near_orthogonal(shape):
    torch.manual_seed(0)
    G = torch.randn(*shape)
    X = zeropower_via_newtonschulz5(G)
    assert X.shape == G.shape and X.dtype == torch.float32
    s = torch.linalg.svdvals(X)
    # NS5 with slope-maximizing coefficients lands singular values in ~[0.68, 1.14], not exactly at 1
    assert s.min() > 0.6 and s.max() < 1.2
    # same singular vectors as the exact polar factor U V^T
    U, _, Vh = torch.linalg.svd(G, full_matrices=False)
    cos = (X * (U @ Vh)).sum() / (X.norm() * (U @ Vh).norm())
    assert cos > 0.97
    Xb = zeropower_via_newtonschulz5(G, dtype=torch.bfloat16)
    assert Xb.dtype == torch.bfloat16 and (Xb.float() - X).abs().max() < 0.05


def test_muon_step_is_scaled_orthogonal_update():
    torch.manual_seed(0)
    p = nn.Parameter(torch.randn(32, 8))  # tall: scale sqrt(32 / 8) = 2
    opt = MuonAdamW([dict(params=[p], use_muon=True, lr=0.1, momentum_warmup_steps=0)])
    before = p.detach().clone()
    p.grad = torch.randn_like(p)
    opt.step()
    s = torch.linalg.svdvals((before - p.detach()) / (0.1 * 2.0))
    assert s.min() > 0.6 and s.max() < 1.2


def test_momentum_warmup_and_state_dict():
    p = nn.Parameter(torch.zeros(4, 4))
    opt = MuonAdamW([dict(params=[p], use_muon=True, momentum=0.95, momentum_warmup_start=0.85,
                          momentum_warmup_steps=10)])
    group = opt.param_groups[0]
    assert opt._muon_momentum(group) == pytest.approx(0.85)
    for _ in range(5):
        p.grad = torch.randn_like(p)
        opt.step()
    assert opt._muon_momentum(group) == pytest.approx(0.90)
    opt2 = MuonAdamW([dict(params=[p], use_muon=True, momentum=0.95, momentum_warmup_steps=10)])
    opt2.load_state_dict(opt.state_dict())
    assert opt2.param_groups[0]["step"] == 5


def test_muon_solves_least_squares():
    torch.manual_seed(0)
    X = torch.randn(256, 16)
    W_true = torch.randn(8, 16)
    Y = X @ W_true.T
    W = nn.Parameter(torch.zeros(8, 16))
    opt = MuonAdamW([dict(params=[W], use_muon=True, lr=0.3, momentum_warmup_steps=0)])
    steps = 300
    losses = []
    for t in range(steps):
        opt.param_groups[0]["lr"] = 0.3 * (1 - t / steps)  # Muon's fixed-size steps need a decaying lr
        loss = ((X @ W.T - Y) ** 2).mean()
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < 1e-3 * losses[0]


def modded_tiny():
    cfg = DenseConfig(variant="modded", vocab_size=64, d_model=32, n_layers=4, d_intermediate=128, num_heads=2,
                      extra=dict(n_value_embeds=1, skip_attn_layers=[1]))
    return ModdedGPT(cfg)


def assert_exact_cover(model, opt):
    seen = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "a parameter is in two groups"
    assert set(seen) == {id(p) for p in model.parameters()}


def test_dense_param_routing_and_lrs():
    model = modded_tiny()
    opt = build_muon_optimizer(model, {"muon": {"lr": 0.05}})
    assert opt is not None and registry.get("optimizer", "muon") is build_muon_optimizer
    assert_exact_cover(model, opt)
    kind = {n: k for k, named in classify_params(model).items() for n, _ in named}
    assert kind["embed.weight"] == kind["value_embeds.0.weight"] == "embed"
    assert kind["lm_head.weight"] == "head"
    assert kind["layers.0.attn.c_q.weight"] == kind["layers.3.mlp.c_proj.weight"] == "muon"
    assert kind["skip_weights"] == kind["layers.0.lambdas"] == kind["layers.0.attn.lambdas"] == "other"
    lrs = {g["kind"]: (g["lr"], g["use_muon"]) for g in opt.param_groups}
    assert lrs == {"muon": (0.05, True), "embed": (0.6, False), "head": (0.008, False), "other": (0.04, False)}
    with pytest.raises(AssertionError):
        build_muon_optimizer(model, {"muon": {"lrr": 0.05}})


def test_with_trainer_param_groups_fn():
    """WS-D's param_groups_fn: per-stage lr_mult and decay/no-decay split; set_lr scales base_lr * lr_mult."""
    optim = pytest.importorskip("hnet.optim")
    model = modded_tiny()
    fn = optim.make_param_groups_fn(model.cfg, {"lr_multiplier_override": [2.0]})
    opt = build_muon_optimizer(model, {"muon": {"weight_decay": 0.1}}, fn)
    assert_exact_cover(model, opt)
    for g in opt.param_groups:
        assert g["lr_mult"] == 2.0 and g["lr"] == pytest.approx(2.0 * g["base_lr"])
        if g["kind"] == "muon":
            assert g["base_lr"] == 0.05 and g["weight_decay"] == 0.1
    optim.set_lr(opt, 0.5)
    assert all(g["lr"] == pytest.approx(0.05) for g in opt.param_groups if g["kind"] == "muon")


class ToyHNetLike(nn.Module):
    """Parameter names/shapes of an H-Net stage: byte embedding, Mamba-2 mixer, router, norm, head."""

    def __init__(self):
        super().__init__()
        from hnet.model.mixers.mamba2 import Mamba2

        self.embeddings = nn.Embedding(256, 32)
        self.mixer = Mamba2(32, d_state=16, headdim=8, chunk_size=16)
        self.routing_module = nn.Module()
        self.routing_module.q_proj_layer = nn.Linear(32, 32, bias=False)
        self.routing_module.k_proj_layer = nn.Linear(32, 32, bias=False)
        self.residual_proj = nn.Linear(32, 32)
        self.norm_f = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, 256, bias=False)


@pytest.mark.parametrize("router", ["adam", "muon"])
def test_hnet_like_param_routing(router):
    pytest.importorskip("hnet.model.mixers.mamba2")
    model = ToyHNetLike()
    opt = build_muon_optimizer(model, {"muon": {"router": router}})
    assert_exact_cover(model, opt)
    kind = {n: k for k, named in classify_params(model, router=router).items() for n, _ in named}
    for n in ("mixer.in_proj.weight", "mixer.out_proj.weight", "residual_proj.weight"):
        assert kind[n] == "muon", n
    for n in ("mixer.conv1d.weight", "mixer.conv1d.bias", "mixer.A_log", "mixer.D", "mixer.dt_bias",
              "mixer.norm.weight", "residual_proj.bias", "norm_f.weight"):
        assert kind[n] == "other", n
    expected = "muon" if router == "muon" else "other"
    assert kind["routing_module.q_proj_layer.weight"] == kind["routing_module.k_proj_layer.weight"] == expected
    assert kind["embeddings.weight"] == "embed" and kind["lm_head.weight"] == "head"


def test_real_hnet_param_cover():
    model_pkg = pytest.importorskip("hnet.model")
    if not hasattr(model_pkg, "build_model"):
        pytest.skip("hnet.model.build_model not available yet")
    from hnet.config import HNetConfig, load_yaml

    cfg = HNetConfig.from_dict({k: v for k, v in load_yaml("configs/tiny_cpu.yaml")["model"].items() if k != "type"})
    model = model_pkg.build_model(cfg)
    opt = build_muon_optimizer(model, {"muon": {}})
    assert_exact_cover(model, opt)
    assert any(g["use_muon"] for g in opt.param_groups)


def test_modded_model_trains_with_muon():
    torch.manual_seed(0)
    model = modded_tiny()
    opt = build_muon_optimizer(model, {"muon": {"lr": 0.02, "lr_embed": 0.1, "lr_head": 0.01}})
    ids = torch.randint(0, 64, (8,)).repeat(6)  # a repeated pattern is easy to learn
    labels = torch.roll(ids, -1)
    labels[-1] = IGNORE_INDEX
    losses = []
    for _ in range(30):
        loss = model(ids, labels=labels).loss
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < 0.3 * losses[0]
