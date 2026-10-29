"""hnet.analysis.scaling + scripts/analysis/fit_ladder.py on synthetic ladders with known compute multipliers."""

import glob
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

from hnet.analysis import scaling as SC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Chinchilla-style loss surface (bits per byte; N parameters, D training bytes, C = 6 N D)
E0, A0, a0, B0, b0 = 0.7, 50.0, 0.34, 200.0, 0.28


def chin(N, D):
    return E0 + A0 / N ** a0 + B0 / D ** b0


def n_opt(C):
    G = (a0 * A0 / (b0 * B0)) ** (1 / (a0 + b0))
    return G * (C / 6) ** (b0 / (a0 + b0))


def synth_runs(m=1.3, budgets=(1e16, 1e17, 1e18), sizes=5, span=1.0, seeds=1, noise=0.0, extra=None, seed=0,
               offset=0.0):
    """H-Net(N, D) = BPE(N sqrt(m), D sqrt(m)): its compute-optimal frontier is BPE's at m x the compute, so the
    true compute multiplier is exactly m at every loss. ``extra``: {metric: multiplier} for more loss metrics
    (same surface, different multiplier, e.g. per code language)."""
    rng = np.random.default_rng(seed)
    out = []
    for C in budgets:
        for arch in ("bpe", "hnet"):
            centre = n_opt(C) * (1.0 if arch == "bpe" else 1 / math.sqrt(m)) * math.exp(offset)
            for N in centre * np.exp(np.linspace(-span, span, sizes)):
                D = C / (6 * N)
                for s in range(seeds):
                    met = {}
                    for k, mm in {"val": m, **(extra or {})}.items():
                        f = 1.0 if arch == "bpe" else math.sqrt(mm)
                        met[k] = chin(N * f, D * f) + noise * rng.standard_normal()
                    out.append(SC.Run(f"{arch}_{C:.0e}_{N:.4g}_s{s}", arch, C, float(N), float(D), s, None, met))
    return out


def test_power_law_forms():
    C = np.array([1e16, 1e17, 1e18, 1e19])
    L = 0.6 + 2.0 * (C / 1e18) ** -0.2
    pl = SC.fit_power_law(C, L, E="free")
    assert pl.form == "free" and pl.E == pytest.approx(0.6, abs=2e-3) and pl.alpha == pytest.approx(0.2, rel=1e-2)
    assert float(pl.compute(pl.loss(3e17))) == pytest.approx(3e17, rel=1e-6)
    pure = SC.fit_power_law(C, 2.0 * (C / 1e18) ** -0.1, E=None)
    assert pure.form == "pure" and pure.alpha == pytest.approx(0.1) and pure.A == pytest.approx(2.0)
    assert SC.fit_power_law(C[:1], L[:1]) is None
    assert SC.fit_power_law(C[:2], L[:2], E="free") is None  # free offset needs >= 3 points
    assert np.isinf(pl.compute(0.5))  # below the irreducible loss


def test_shared_E_recovers_multiplier_with_three_budgets():
    C = np.array([1e16, 1e17, 1e18])
    law = lambda c: 0.6 + 2.0 * (c / 1e18) ** -0.2  # noqa: E731
    laws = SC.fit_shared_E({"bpe": (C, law(C)), "hnet": (C, law(1.5 * C))})
    assert laws["bpe"].E == pytest.approx(0.6, abs=5e-3) == laws["hnet"].E
    L = np.linspace(law(1.5e18), law(1e16), 7)
    assert np.allclose(laws["bpe"].compute(L) / laws["hnet"].compute(L), 1.5, rtol=1e-2)


def test_isoflop_status():
    x = np.log([1e7, 2e7, 4e7, 8e7])
    y = 1 + 0.3 * (x - np.log(3e7)) ** 2
    f = SC.fit_isoflop(x, y, logD=np.log([8e9, 4e9, 2e9, 1e9]))
    assert f["status"] == "quadratic" and math.exp(f["x_opt"]) == pytest.approx(3e7, rel=1e-6)
    assert math.exp(f["logD_opt"]) == pytest.approx(8e9 * 1e7 / 3e7, rel=1e-6)
    assert SC.fit_isoflop(x, 1 + 0.3 * (x - np.log(3e8)) ** 2)["status"] == "edge"
    assert SC.fit_isoflop(x[:2], y[:2])["status"] == "min_observed"
    assert SC.fit_isoflop(x, -y)["status"] == "min_observed"  # concave: no interior minimum


@pytest.mark.parametrize("budgets,form", [((1e16, 1e17, 1e18), "shared"), ((1e16, 1e17, 1e18, 1e19), "free"),
                                          ((1e16, 1e17), "pure")])
def test_recovers_compute_multiplier(budgets, form):
    res = SC.analyze_ladder(synth_runs(1.3, budgets), n_boot=0)
    r = res["metrics"]["val"]
    assert r["form"] == form
    assert all(p["status"] == "quadratic" for f in r["frontier"].values() for p in f)
    assert r["multiplier"]["multiplier"] == pytest.approx(1.3, rel=0.03)
    assert not r["multiplier"]["extrapolated"]
    assert r["multiplier_interp"]["multiplier"] == pytest.approx(1.3, rel=0.05)
    for p in r["multiplier"]["per_budget"]:
        assert p["multiplier"] == pytest.approx(1.3, rel=0.05)
    al = res["allocation"]
    expo = b0 / (a0 + b0)
    assert al["bpe"]["N_opt"]["exponent"] == pytest.approx(expo, abs=0.02)
    assert al["bpe"]["D_opt"]["exponent"] == pytest.approx(1 - expo, abs=0.02)
    for p, C in zip(al["bpe"]["points"], budgets):
        assert p["N_opt"] == pytest.approx(n_opt(C), rel=0.05)


def test_off_centre_grid_and_seed_bootstrap():
    runs = synth_runs(1.6, seeds=3, noise=0.002, offset=0.3, seed=1)
    res = SC.analyze_ladder(runs, n_boot=200, seed=0)
    m = res["metrics"]["val"]["multiplier"]
    assert m["bootstrap"]["mode"] == "seeds" and m["bootstrap"]["n_valid"] >= 190
    assert m["ci95"]["lo"] < 1.6 < m["ci95"]["hi"] and m["ci95"]["hi"] / m["ci95"]["lo"] < 1.5
    assert m["multiplier"] == pytest.approx(1.6, rel=0.06)


def test_residual_bootstrap_single_seed():
    res = SC.analyze_ladder(synth_runs(1.3, sizes=6, noise=0.003, seed=2), n_boot=100)
    m = res["metrics"]["val"]["multiplier"]
    assert m["bootstrap"]["mode"] == "residual" and m["ci95"] is not None
    assert m["ci95"]["lo"] <= m["multiplier"] <= m["ci95"]["hi"]


def test_per_domain_and_code_language_multipliers():
    extra = {"src:python": 2.0, "src:haskell": 3.1, "group:code": 2.2}
    runs = synth_runs(1.3, extra=extra)
    for r in runs:
        r.info["sources"] = {"python": {"group": "code", "lang": "python", "held_out": False},
                             "haskell": {"group": "code", "lang": "haskell", "held_out": True}}
    res = SC.analyze_ladder(runs, n_boot=0)
    for k, mm in extra.items():
        assert res["metrics"][k]["multiplier"]["multiplier"] == pytest.approx(mm, rel=0.04), k
    assert res["source_meta"]["haskell"]["held_out"] is True


def test_loss_to_loss_multiplier_analytic():
    """Pure power-law val law C_ref(L) = C0 (L/A)^(-1/alpha); downstream y = K_arch x^kappa. Then
    m_l2l = (K_new / K_ref)^(-1 / (kappa alpha)) at every L, and compute->down = m_val x m_l2l."""
    runs = synth_runs(1.4, budgets=(1e16, 1e17))
    kappa, K = 1.5, {"bpe": 1.2, "hnet": 1.05}
    for r in runs:
        r.metrics["down:avg"] = K[r.arch] * r.metrics["val"] ** kappa
    res = SC.analyze_ladder(runs, n_boot=50)
    l2 = res["loss_to_loss"]
    assert l2["laws"]["bpe"]["kappa"] == pytest.approx(kappa) and l2["laws"]["hnet"]["K"] == pytest.approx(1.05)
    alpha = res["metrics"]["val"]["laws"]["bpe"]["alpha"]
    want = (K["hnet"] / K["bpe"]) ** (-1 / (kappa * alpha))
    assert l2["val_to_down_multiplier"] == pytest.approx(want, rel=1e-3)
    assert l2["compute_to_down_multiplier"] == pytest.approx(want * l2["compute_to_val_multiplier"], rel=0.02)
    assert l2["compute_to_val_multiplier"] == pytest.approx(1.4, rel=0.05)
    # the direct compute -> downstream fit agrees with the product (exact here: y is a power of the val loss)
    assert res["metrics"]["down:avg"]["multiplier"]["multiplier"] == pytest.approx(
        l2["compute_to_down_multiplier"], rel=0.1)


def test_single_budget_is_degenerate_but_reports_gap():
    res = SC.analyze_ladder(synth_runs(1.3, budgets=(3e15,), sizes=1), n_boot=100)
    r = res["metrics"]["val"]
    assert r["multiplier"] is None and r["form"] is None and r["laws"] == {}
    assert any("single compute budget" in n for n in res["notes"])
    assert any("needs >= 2 budgets" in n for n in r["notes"])
    assert any("best observed run" in n for n in r["notes"])
    assert len(r["matched_budget_gap"]) == 1 and r["matched_budget_gap"][0]["delta"] < 0
    json.dumps(SC.public(res))  # serializable


def test_budget_clustering():
    runs = [SC.Run(f"r{i}", "bpe", c, 1e7) for i, c in enumerate([0.95e16, 1.05e16, 1e17, 1.1e17, 0.9e18])]
    SC.assign_budgets(runs)
    assert len({r.budget for r in runs}) == 3
    assert runs[0].budget == runs[1].budget == pytest.approx(1e16, rel=0.05)


def test_integrated_flops():
    recs = [{"bytes_seen": 100, "flops_per_byte": 10.0}, {"bytes_seen": 300, "flops_per_byte": 20.0},
            {"bytes_seen": 400}]
    C, D = SC.integrated_flops(recs)
    assert D == 400 and C == 100 * 10 + 200 * 20 + 100 * 20


# ------------------------------------------------------------------------------------------------- CLI / run dirs
def _fit_ladder():
    spec = importlib.util.spec_from_file_location("fit_ladder", os.path.join(ROOT, "scripts", "analysis",
                                                                             "fit_ladder.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fit_ladder"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_run_dir(d, r: SC.Run, model_type: str, sources: dict):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        f.write(f"model:\n  type: custom_{model_type}\ntrain:\n  seed: {r.seed}\n")
    with open(os.path.join(d, "metrics.jsonl"), "w") as f:
        f.write(json.dumps({"step": 1, "bytes_seen": r.D / 2, "flops_per_byte": r.flops / r.D}) + "\n")
        f.write(json.dumps({"step": 2, "bytes_seen": r.D, "flops_per_byte": r.flops / r.D}) + "\n")
    with open(os.path.join(d, "eval.jsonl"), "w") as f:
        f.write(json.dumps({"step": 2, "val/bpb": r.metrics["val"],
                            "downstream/tasks/hellaswag/metrics/soft_bpb/value": r.metrics["down:avg"]}) + "\n")
    open(os.path.join(d, "done.json"), "w").write("{}")
    json.dump({"arch": r.arch, "params": r.params}, open(os.path.join(d, "ladder.json"), "w"))
    ps = {"sources": {}, "groups": {}, "settings": {"byte_seq_len": 2048, "token_seq_len": 448,
                                                     "split_tokenizer": "gpt2"}}
    for name, meta in sources.items():
        ps["sources"][name] = {**meta, "bpb": r.metrics[f"src:{name}"], "n_bytes": 1000,
                               ("bytes_per_main_position" if r.arch == "hnet" else "bytes_per_token"):
                                   meta["bpp"][r.arch], "split_bytes_per_token": 4.5}
    ps["groups"]["code"] = {"bpb": r.metrics["src:python"]}
    json.dump(ps, open(os.path.join(d, "per_source.json"), "w"))


def test_fit_ladder_cli_on_run_dirs(tmp_path):
    srcs = {"python": {"group": "code", "lang": "python", "held_out": False, "bpp": {"bpe": 3.5, "hnet": 4.2}},
            "haskell": {"group": "code", "lang": "haskell", "held_out": True, "bpp": {"bpe": 3.0, "hnet": 4.0}},
            "web": {"group": "english", "lang": None, "held_out": False, "bpp": {"bpe": 4.6, "hnet": 4.6}}}
    runs = synth_runs(1.3, sizes=4, seeds=2, noise=0.001,
                      extra={"src:python": 2.0, "src:haskell": 3.0, "src:web": 1.2})
    for r in runs:
        r.metrics["down:avg"] = (1.2 if r.arch == "bpe" else 1.05) * r.metrics["val"] ** 1.5
    for i, r in enumerate(runs):
        _write_run_dir(str(tmp_path / "runs" / f"{r.arch}_{i:03d}"), r, r.arch, srcs)
    fl = _fit_ladder()
    figdir = tmp_path / "fig"
    out = tmp_path / "fit.json"
    res = fl.main(["--runs", str(tmp_path / "runs" / "*"), "--n-boot", "50", "--figdir", str(figdir),
                   "--out", str(out)])
    assert isinstance(res, dict)
    assert res["metrics"]["val"]["multiplier"]["multiplier"] == pytest.approx(1.3, rel=0.05)
    assert res["metrics"]["src:haskell"]["multiplier"]["multiplier"] == pytest.approx(3.0, rel=0.06)
    assert res["source_meta"]["haskell"]["held_out"] is True
    assert res["runs"][0]["seed"] in (0, 1)
    names = {os.path.basename(p) for p in glob.glob(str(figdir / "*.png"))}
    assert {"ladder_loss_vs_compute.png", "ladder_isoflop.png", "ladder_code_multipliers.png",
            "ladder_domain_multipliers.png", "ladder_compression.png", "ladder_loss_to_loss.png",
            "ladder_allocation.png"} <= names
    saved = json.load(open(out))
    assert saved["compression"]["hnet"]["per_source"]["haskell"] == pytest.approx(4.0)
    assert "_laws" not in saved["metrics"]["val"]


def test_fit_ladder_no_runs(tmp_path, capsys):
    assert _fit_ladder().main(["--runs", str(tmp_path / "nothing*"), "--no-plots"]) == 2


@pytest.mark.skipif(not glob.glob(os.path.join(ROOT, "runs/gcp/exp006_close_gap_tiny/bpe_s*/done.json")) or
                    not glob.glob(os.path.join(ROOT, "runs/gcp/exp024_adoption_curriculum/prior_s*/done.json")),
                    reason="tiny EXP-006 / EXP-024 run dirs not present")
def test_smoke_existing_tiny_runs_single_budget(tmp_path):
    """EXP-006 BPE vs EXP-024 'prior' H-Net: one budget (~3e15 FLOPs), one model size, 3 seeds each."""
    res = _fit_ladder().main([
        "--runs", "bpe=" + os.path.join(ROOT, "runs/gcp/exp006_close_gap_tiny/bpe_s*"),
        "--runs", "hnet=" + os.path.join(ROOT, "runs/gcp/exp024_adoption_curriculum/prior_s*"),
        "--n-boot", "20", "--figdir", str(tmp_path), "--metrics", "val,down:avg"])
    assert len(res["budgets"]) == 1 and 1e15 < res["budgets"][0] < 1e16
    assert any("single compute budget" in n for n in res["notes"])
    r = res["metrics"]["val"]
    assert r["multiplier"] is None and len(r["matched_budget_gap"]) == 1
    assert 1.5 < r["frontier"]["bpe"][0]["L"] < 1.8 and 1.5 < r["frontier"]["hnet"][0]["L"] < 1.8
    assert os.path.exists(tmp_path / "ladder_loss_vs_compute.png")


def test_load_run_reads_ladder_sweep_job(tmp_path):
    """scripts/ladder specs: budget from the spec name, architecture label from the sweep arm."""
    r = SC.Run("x", "hnet", 1.0e17, 3e7, 2e9, 0, None, {"val": 1.2, "down:avg": 1.5, "src:python": 2.0})
    d = tmp_path / "1e17" / "L05" / "hnet_ws_s1"
    _write_run_dir(str(d), r, "hnet", {"python": {"group": "code", "lang": "python", "held_out": False,
                                                  "bpp": {"bpe": 3.5, "hnet": 4.2}}})
    os.remove(d / "ladder.json")
    json.dump({"spec": "ladder_1e17_L05", "name": "hnet_ws", "arch": "hnet_ws", "seed": 1},
              open(d / "sweep_job.json", "w"))
    run = SC.load_run(str(d))
    assert run.arch == "hnet_ws" and run.budget == 1e17 and run.seed == 1
    assert run.flops == pytest.approx(1e17) and run.D == pytest.approx(2e9)
    assert SC.load_run(str(d), arch="hnet").arch == "hnet"
