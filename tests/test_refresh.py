"""First-byte chunk refresh (HNetConfig.refresh, evolve g5-compute-01/02): default-off identity, gradient paths
(CLAUDE.md rule 1, verified with autograd), exact capacity in training, the region mask, and the FLOPs / parameter
accounting. Causality and packed == separate are in tests/test_invariances.py (variants *_refresh*)."""

import copy

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model


def _cfg(refresh=None, main="T2"):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.arch_layout = ["m2", [main], "m2"]
    cfg.dc.router, cfg.dc.decision = "end_mlp", "quantile"
    if refresh is not None:
        cfg.refresh = refresh
    return cfg


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    m = build_model(cfg)
    m.init_weights()
    return m


def _grads(loss, params):
    return torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)


def _norm(g):
    return 0.0 if g is None else float(g.float().norm())


def test_default_off_and_zero_gain_are_bit_identical():
    assert not any("refresh" in n for n, _ in _build(_cfg()).named_parameters())
    for refresh in ({"layout": "T1"}, {"layout": "T1", "capacity": 0.5, "probe_rank": 8}):
        base = _build(_cfg())
        m = _build(_cfg(refresh))
        sd = {k: v for k, v in m.state_dict().items() if ".refresh." not in k}
        assert sd.keys() == base.state_dict().keys()
        for k, v in base.state_dict().items():  # every shared parameter initialized identically (forked RNG)
            assert torch.equal(sd[k], v), k
        ids = torch.randint(0, 256, (2, 200), generator=torch.Generator().manual_seed(1))
        for train in (True, False):
            base.train(train)
            m.train(train)
            with torch.no_grad():
                a, b = base(ids, labels=ids), m(ids, labels=ids)
            assert torch.equal(a.logits, b.logits), (refresh, train)
            assert torch.equal(a.loss, b.loss)


def test_gradient_paths_at_init_and_after_gain_opens():
    cfg = _cfg({"layout": "T1"})
    m = _build(cfg).train()
    rf = m.backbone.main_network.refresh
    ids = torch.randint(0, 256, (2, 256), generator=torch.Generator().manual_seed(2))
    out = m(ids, labels=ids)
    groups = {"gain": [rf.gain], "layers": list(rf.layers.parameters()), "up": [rf.up.weight],
              "norm": [rf.norm.weight]}
    g = {k: [_norm(x) for x in _grads(out.loss, v)] for k, v in groups.items()}
    assert g["gain"][0] > 0  # the zero-init gain is the only refresh parameter the CE reaches at init
    assert max(g["layers"]) == 0 and g["up"][0] == 0 and g["norm"][0] == 0
    with torch.no_grad():
        rf.gain.fill_(0.3)
    out = m(ids, labels=ids)
    main = list(m.backbone.main_network.main_network.parameters())
    enc = list(m.backbone.encoder.parameters())
    g_l = [_norm(x) for x in _grads(out.loss, list(rf.layers.parameters()) + [rf.up.weight])]
    assert min(g_l) > 0, "every refresh matrix gets CE gradient once the gain is open"
    # the refresh is a second CE path into the main network (through zfull) and the encoder (through e_{b+1})
    m_only = _build(cfg).train()
    with torch.no_grad():
        m_only.load_state_dict(m.state_dict())
        m_only.backbone.main_network.refresh.gain.zero_()
    o2 = m_only(ids, labels=ids)
    g_main_on = sum(_norm(x) for x in _grads(out.loss, main))
    g_main_off = sum(_norm(x) for x in _grads(o2.loss, list(m_only.backbone.main_network.main_network.parameters())))
    assert g_main_on != pytest.approx(g_main_off)
    assert sum(_norm(x) for x in _grads(out.loss, enc)) > 0


def test_probe_loss_reaches_only_the_probe_and_ce_does_not_reach_it():
    cfg = _cfg({"layout": "T1", "capacity": 0.5, "probe_rank": 8, "gain_init": 0.3})
    m = _build(cfg).train()
    rf = m.backbone.main_network.refresh
    ids = torch.randint(0, 256, (2, 256), generator=torch.Generator().manual_seed(3))
    out = m(ids, labels=ids)
    pl = out.bpred[0].refresh_loss
    assert pl is not None and torch.isfinite(pl)
    probe = list(rf.probe.parameters())
    others = [p for n, p in m.named_parameters() if ".refresh.probe." not in n]
    assert min(_norm(x) for x in _grads(pl, probe)) > 0
    assert max(_norm(x) for x in _grads(pl, others)) == 0, "the probe reads the main output detached"
    assert max(_norm(x) for x in _grads(out.loss, probe)) == 0, "chunk selection is discrete: no CE path to the probe"
    # the probe's CE is part of the training objective (weighted), so the probe learns
    g = _grads(out.loss + out.aux_loss, probe)
    assert min(_norm(x) for x in g) > 0


def test_capacity_is_exact_in_training_and_regions_are_respected():
    cfg = _cfg({"layout": "T1", "capacity": 0.5, "probe_rank": 8, "gain_init": 0.3})
    m = _build(cfg).train()
    rf = m.backbone.main_network.refresh
    seen = {}
    orig = rf.layers.forward

    def spy(h, ctx, *a, **k):
        seen["K"] = h.shape[-2] if h.dim() == 3 else h.shape[0]
        return orig(h, ctx, *a, **k)

    rf.layers.forward = spy
    ids = torch.randint(0, 256, (3, 300), generator=torch.Generator().manual_seed(4))
    out = m(ids, labels=ids)
    M = int(out.bpred[0].boundary_mask.sum())
    assert seen["K"] == round(0.5 * M)
    assert float(out.stats["refresh/frac/s0"]) == pytest.approx(seen["K"] / M)
    assert bool(rf.thr_init)
    # eval: a per-chunk running threshold (some but not all chunks), add is zero on boundary bytes
    m.eval()
    with torch.no_grad():
        o = m(ids)
    frac = float(o.stats["refresh/frac/s0"])
    assert 0.0 < frac < 1.0


def test_add_is_zero_on_boundary_bytes_and_only_after_the_first_byte():
    cfg = _cfg({"layout": "T1", "gain_init": 0.3})
    m = _build(cfg).eval()
    rf = m.backbone.main_network.refresh
    rec = {}
    orig = rf.forward

    def spy(*a, **k):
        add, pl, st = orig(*a, **k)
        rec["add"], rec["mask"] = add, a[3]
        return add, pl, st

    rf.forward = spy
    ids = torch.randint(0, 256, (1, 300), generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        m(ids)
    add, mask = rec["add"], rec["mask"].reshape(-1)
    assert add[mask].abs().max() == 0
    assert add[~mask].abs().sum(-1).gt(0).float().mean() > 0.9


@pytest.mark.parametrize("refresh,main", [({"layout": "T1"}, "T2"),
                                          ({"layout": "T1", "capacity": 0.5, "probe_rank": 8}, "T3"),
                                          ({"layout": "T2", "capacity": 0.25, "probe_rank": 16}, "T2")])
def test_flops_and_params_accounting(refresh, main):
    cfg = _cfg(refresh, main)
    m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    base = _cfg(None, main)
    r1 = model_flops(cfg, seq_len=2048)["per_stage"][1]["components"]["refresh"]
    rb = model_flops(base, seq_len=2048)
    assert "refresh" not in rb["per_stage"][1]["components"]
    # the refreshed stack costs at most capacity x (one main-rate stack of the same layout) + the probe
    from hnet.flops import _conv, _iso

    L = 2048 / 6.0
    _, full = _iso(cfg, 1, refresh["layout"], L, _conv(None))
    cap = refresh.get("capacity", 1.0)
    per_byte_stack = 3 * cap * full / 6.0
    assert r1["flops_per_byte"] >= per_byte_stack * (0.99 if cap == 1.0 else 0.5)
    assert r1["flops_per_byte"] <= 3 * (cap * full + 2 * 96 * 256 + 2 * 96 + 2 * (256 * 16 + 16 * 256) + 5 * 256) / 6.0


@pytest.mark.parametrize("name,layout,cap", [("g5-compute-01", "T2", 1.0), ("g5-compute-02", "T3", 0.5)])
def test_candidate_configs_fit_the_budget(name, layout, cap):
    from hnet.config import model_config_from_yaml

    cfg = model_config_from_yaml(f"configs/evolve/models/{name}.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_L03.yaml")
    assert cfg.arch_layout[1] == [layout] and float(cfg.refresh["capacity"]) == cap
    assert count_params(cfg) <= 20_023_040  # bpe_L03
    rel = model_flops(cfg, seq_len=8512)["flops_per_byte"] / model_flops(ref, seq_len=8512)["flops_per_byte"] - 1
    assert abs(rel) < 0.02
