"""scripts/search/report.py + results.py on a synthetic finished sweep (no training): statistics, best setting,
paired deltas, divergence handling, best_params.yaml, data multiplier, figures; logbook EXP-002 wrappers."""

import json
import math
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(REPO, "scripts", "search"), os.path.join(REPO, "logbook")]

import report  # noqa: E402
import results  # noqa: E402
from spec import Spec  # noqa: E402

LRS = [1e-3, 3e-3, 1e-2]
OFFSET = {"bpe": 0.0, "hnet": -0.05}  # H-Net 0.05 bpb better at its best LR
BEST = {"bpe": 3e-3, "hnet": 1e-2}


def final_bpb(arch, lr, seed):
    return 1.5 + OFFSET[arch] + 0.2 * abs(math.log10(lr / BEST[arch])) + 0.01 * seed


def write_run(d, arch, lr, seed, diverged=False):
    d.mkdir(parents=True)
    steps = list(range(10, 210, 10))
    metrics, evals = [], []
    fin = final_bpb(arch, lr, seed)
    for s in steps:
        t = s * 1000.0
        lf = 1.0 if 20 <= s <= 160 else 0.5
        rec = {"step": s, "tokens_eq": t, "bytes_seen": int(4.6 * t), "bpb": fin + 3.0 * (s / 10) ** -0.7,
               "lr_factor": lf, "loss": 1.0}
        if arch == "hnet":
            rec.update({"compression/s0": 3.0 + 0.001 * s, "compression/s1": 2.5})
        metrics.append(rec)
        if s % 20 == 0:
            ev = {"step": s, "tokens_eq": t, "bytes_seen": int(4.6 * t), "val/bpb": fin + 2.0 * (s / 10) ** -0.7}
            if s == steps[-1]:
                ev.update({"val/bpb": fin, "downstream/averages/average/acc_norm_char/value": 0.3 + 0.01 * seed,
                           "downstream/tasks/piqa/metrics/soft_bpb/value": 2.0,
                           "downstream/tasks/arc_easy/metrics/soft_bpb/value": 3.0})
            evals.append(ev)
    if diverged:
        metrics = metrics[:3]
        evals = evals[:1]
        (d / "diverged.json").write_text(json.dumps({"step": 30, "loss": float("nan")}))
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in metrics) + "\n")
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in evals) + "\n")
    (d / "done.json").write_text(json.dumps({"rc": 0, **({"diverged": True} if diverged else {})}))
    if arch == "hnet" and not diverged:
        st = {"bytes_per_chunk": 6.5 + seed, "pct_cut_after_space": 80.0, "pct_word_internal": 2.0,
              "pct_mid_utf8": 0.0, "pct_cut_before_space": 5.0, "n_cuts": 100}
        (d / "diag.json").write_text(json.dumps({"s0": {**st, "bytes_per_chunk": 3.0}, "s1": st}))


@pytest.fixture
def sweep_dir(tmp_path):
    spec = {"name": "toy_lr", "out_root": str(tmp_path / "runs"), "seeds": [0, 1, 2], "fairness": "tuned",
            "grid": {"train.lr": LRS}, "report": {"baseline": "bpe"},
            "runs": [{"name": "bpe", "config": "configs/runs/tiny_bpe_2m.yaml"},
                     {"name": "hnet", "config": "configs/runs/tiny_hnet2_2m.yaml"}]}
    sp = tmp_path / "spec.yaml"
    sp.write_text(yaml.safe_dump(spec))
    s = Spec.load(str(sp))
    for j in s.jobs():
        write_run(tmp_path / "runs" / j.name, j.arch, j.point["train.lr"], j.seed,
                  diverged=(j.arch == "bpe" and j.point["train.lr"] == 1e-2 and j.seed == 1))
    return sp, tmp_path / "runs"


def test_mean_and_paired_ci():
    ci = results.mean_ci([1.0, 2.0, 3.0])
    assert ci["mean"] == 2.0 and ci["hi"] - 2.0 == pytest.approx(4.303 * 1.0 / math.sqrt(3), rel=1e-4)
    d = results.paired_ci({0: 1.1, 1: 2.1, 2: 3.1, 3: 9.0}, {0: 1.0, 1: 2.0, 2: 3.0})
    assert d["seeds"] == [0, 1, 2] and d["mean"] == pytest.approx(0.1) and d["hi"] - d["lo"] < 1e-9
    assert results.mean_ci([1.0])["n"] == 1 and math.isnan(results.mean_ci([1.0])["lo"])


def test_report_tables_best_and_params(sweep_dir):
    sp, runs = sweep_dir
    sweep = results.load_sweep(Spec.load(str(sp)), runs)
    best = report.best_points(sweep)
    assert best["bpe"]["point"] == {"train.lr": 3e-3} and best["hnet"]["point"] == {"train.lr": 1e-2}
    md, summary, bp = report.build_report(sweep, n_boot=50)
    assert bp == {"best_lr": {"bpe": 3e-3, "hnet": 1e-2}}
    hb = summary["best"]["hnet"]
    assert hb["val_bpb"]["mean"] == pytest.approx(1.45 + 0.01)
    assert hb["delta_vs_baseline"]["mean"] == pytest.approx(-0.05) and hb["delta_vs_baseline"]["seeds"] == [0, 1, 2]
    row = next(r for r in summary["rows"] if r["arch"] == "bpe" and r["point"] == {"train.lr": 1e-2})
    assert row["n_diverged"] == 1 and math.isinf(row["val_bpb"]["mean"])
    assert "1 diverged" in md and "diverged" in md and "grid edge" in md  # hnet's best LR is the largest
    assert summary["chunks"]["hnet"]["s1_bytes_per_chunk"]["mean"] == pytest.approx(7.5)
    assert "blog's improved H-Net" in md
    m = hb["multiplier"]
    assert m is not None and m["multiplier"] > 1.0  # H-Net reaches a lower bpb than BPE does at equal tokens
    r = next(r for r in summary["rows"] if r["arch"] == "hnet" and r["point"] == {"train.lr": 1e-2})
    assert r["soft_bpb"]["mean"] == pytest.approx(2.5) and r["acc_norm"]["mean"] == pytest.approx(31.0)
    assert r["bytes_per_pos"][1]["mean"] == pytest.approx((3.0 + 0.2) * 2.5)


def test_report_cli_writes_outputs_and_figures(sweep_dir, tmp_path):
    sp, runs = sweep_dir
    out = tmp_path / "rep"
    assert report.main([str(sp), "--runs", str(runs), "--out", str(out), "--n-boot", "20"]) == 0
    for f in ("report.md", "summary.json", "best_params.yaml"):
        assert (out / f).exists(), f
    figs = {p.name for p in (out / "figures").iterdir()}
    assert {"overview.png", "val_bpb.png", "train_bpb.png", "lr_sweep.png", "compression.png"} <= figs
    assert yaml.safe_load((out / "best_params.yaml").read_text())["best_lr"]["hnet"] == 1e-2


def test_best_params_feed_a_later_spec(sweep_dir, tmp_path):
    """report's best_params.yaml is a valid --params-file for a spec that references ${best_lr.<arch>}."""
    sp, runs = sweep_dir
    out = tmp_path / "rep"
    report.main([str(sp), "--runs", str(runs), "--out", str(out), "--no-figures", "--n-boot", "0"])
    later = {"name": "later", "out_root": str(tmp_path / "l"), "params": {"best_lr": {"bpe": None, "hnet": None}},
             "runs": [{"name": "bpe", "config": "configs/runs/tiny_bpe_2m.yaml", "set": ["train.lr=${best_lr.bpe}"]}]}
    lp = tmp_path / "later.yaml"
    lp.write_text(yaml.safe_dump(later))
    jobs = Spec.load(str(lp), params_file=str(out / "best_params.yaml")).jobs()
    assert "train.lr=0.003" in jobs[0].sets


def test_panels_have_at_most_five_series(sweep_dir):
    sp, runs = sweep_dir
    sweep = results.load_sweep(Spec.load(str(sp)), runs)
    panels = report.panel_groups(sweep, "bpe", archs=["bpe"] + [f"a{i}" for i in range(9)])
    assert len(panels) == 3 and all(len(m) + 1 <= report.MAX_SERIES for m in panels.values())  # + baseline
    sweep.spec.raw["report"]["groups"] = {"G": [f"a{i}" for i in range(6)]}
    panels = report.panel_groups(sweep, "bpe", archs=["bpe"] + [f"a{i}" for i in range(9)])
    assert list(panels) == ["G (1/2)", "G (2/2)"]


def test_exp002_wrappers(tmp_path):
    """logbook/exp002_progress.py and exp002_summary.py run on (partial) EXP-002-shaped run dirs."""
    import exp002_progress
    import exp002_summary

    runs = tmp_path / "runs"
    write_run(runs / "bpe_s0", "bpe", 3e-3, 0)
    write_run(runs / "paper_s0", "hnet", 1e-2, 0)
    exp002_progress.main(["--no-pull", "--runs", str(runs), "--out", str(tmp_path / "p.png")])
    assert (tmp_path / "p.png").exists()
    exp002_summary.main(["--runs", str(runs), "--out", str(tmp_path / "s.md")])
    text = (tmp_path / "s.md").read_text()
    assert "paper router (LH-Net)" in text and "-0.050" in text
