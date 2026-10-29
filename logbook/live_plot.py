"""Live progress plot of running sweeps against reference runs.

    python logbook/live_plot.py --live <dir with sweep mirrors> --out logbook/figures/live_progress.png

Left: validation bpb vs step for every running arm (mean over seeds, thin lines per seed) with reference arms dashed.
Right: train loss (EMA) vs step for the same runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCES = {
    "BPE (EXP-006)": f"{ROOT}/runs/gcp/exp006_close_gap_tiny/bpe_s*",
    "H-Net prior, AdamW (EXP-024)": f"{ROOT}/runs/gcp/exp024_adoption_curriculum/prior_s*",
}


def _jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _evals(run):
    return [(r["step"], r["val/bpb"]) for r in _jsonl(f"{run}/eval.jsonl") if "val/bpb" in r]


def _train(run, alpha=0.9):
    pts, ema = [], None
    for r in _jsonl(f"{run}/metrics.jsonl"):
        v = r.get("bpb")
        if v is None:
            continue
        ema = v if ema is None else alpha * ema + (1 - alpha) * v
        pts.append((r["step"], ema))
    return pts


def arm_of(run):
    name = os.path.basename(run)
    return name.rsplit("_s", 1)[0]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--sweeps", nargs="+", required=True)
    ap.add_argument("--out", default=f"{ROOT}/logbook/figures/live_progress.png")
    a = ap.parse_args(argv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    ci = 0
    summary = {}
    for label, pat in REFERENCES.items():
        runs = sorted(glob.glob(pat))
        ev = defaultdict(list)
        for r in runs:
            for s, v in _evals(r):
                ev[s].append(v)
        if ev:
            xs = sorted(ev)
            ax1.plot(xs, [sum(ev[x]) / len(ev[x]) for x in xs], "--", color="gray" if "BPE" in label else "black",
                     lw=1.5, label=label)
            summary[label] = (xs[-1], sum(ev[xs[-1]]) / len(ev[xs[-1]]), len(ev[xs[-1]]))
    for sweep in a.sweeps:
        by_arm = defaultdict(list)
        for r in sorted(glob.glob(f"{a.live}/{sweep}/*_s[0-9]")):
            by_arm[arm_of(r)].append(r)
        for arm, runs in sorted(by_arm.items()):
            c = colors[ci % len(colors)]
            ci += 1
            ev = defaultdict(list)
            for r in runs:
                e = _evals(r)
                if e:
                    ax1.plot(*zip(*e), color=c, lw=0.6, alpha=0.4)
                for s, v in e:
                    ev[s].append(v)
                t = _train(r)
                if t:
                    ax2.plot(*zip(*t), color=c, lw=0.8, alpha=0.7)
            if ev:
                xs = sorted(ev)
                ax1.plot(xs, [sum(ev[x]) / len(ev[x]) for x in xs], "-o", ms=3, color=c, lw=1.8,
                         label=f"{sweep.split('_')[0]} {arm}")
                summary[f"{sweep.split('_')[0]} {arm}"] = (xs[-1], sum(ev[xs[-1]]) / len(ev[xs[-1]]), len(ev[xs[-1]]))
            ax2.plot([], [], color=c, label=f"{sweep.split('_')[0]} {arm}")
    for ax, t in ((ax1, "validation bpb (lines = seed mean, faint = seeds)"), (ax2, "train bpb (EMA), per seed")):
        ax.set_xlabel("step")
        ax.set_ylabel("bits per byte")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        ax.set_ylim(1.55, 2.6)
        ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=110)
    for k, (s, v, n) in summary.items():
        print(f"{k:40s} step {s:5d}  val bpb {v:.4f}  ({n} seeds)")
    print(a.out)


if __name__ == "__main__":
    main()
