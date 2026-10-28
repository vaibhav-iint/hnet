"""Time / FLOPs / tokens / bytes to reach the speedrun target bpb, for several runs of our trainer.

    python scripts/speedrun/speedrun_report.py runs/s1_modded runs/s2_hnet1_muon ... \
        [--target-bpb 1.0667] [--labels S1 S2-1] [--plot curves.png] [--json report.json]

Reads ``<run>/metrics.jsonl`` (``step``, ``step_time``, ``bytes_seen``, ``tokens_eq``, ``flops_per_byte``) and
``<run>/eval.jsonl`` (``step``, ``tokens_eq``, ``bytes_seen``, ``val/bpb``). Training time and FLOPs are
accumulated from the metrics rows (time = sum of step_time x steps, FLOPs = sum of flops_per_byte x bytes; eval
time excluded) and interpolated at the eval steps. The crossing point is linearly interpolated between the
last eval above the target and the first at or below it. Default target: 3.28 nats/token converted with
scripts/speedrun/target.py (needs the FineWeb val shard).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

QUANTITIES = ("time_s", "flops", "tokens_eq", "bytes")


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_curve(run_dir: str, metric: str = "val/bpb") -> dict:
    """Eval curve of one run with cumulative time and FLOPs at every eval step."""
    metrics = sorted(read_jsonl(os.path.join(run_dir, "metrics.jsonl")), key=lambda r: r["step"])
    evals = sorted((r for r in read_jsonl(os.path.join(run_dir, "eval.jsonl")) if metric in r),
                   key=lambda r: r["step"])
    steps, time_s, flops = [0], [0.0], [0.0]
    prev_bytes = 0.0
    for r in metrics:
        steps.append(r["step"])
        time_s.append(time_s[-1] + r["step_time"] * (r["step"] - steps[-2]))
        flops.append(flops[-1] + (r.get("flops_per_byte") or 0.0) * (r["bytes_seen"] - prev_bytes))
        prev_bytes = r["bytes_seen"]
    es = np.array([r["step"] for r in evals], dtype=float)
    return {
        "step": es,
        "bpb": np.array([r[metric] for r in evals], dtype=float),
        "time_s": np.interp(es, steps, time_s),
        "flops": np.interp(es, steps, flops),
        "tokens_eq": np.array([r["tokens_eq"] for r in evals], dtype=float),
        "bytes": np.array([r["bytes_seen"] for r in evals], dtype=float),
    }


def crossing(curve: dict, target: float) -> dict | None:
    """Quantities at the first point where bpb reaches the target (linear interpolation), or None."""
    hit = np.flatnonzero(curve["bpb"] <= target)
    if len(hit) == 0:
        return None
    i = int(hit[0])
    if i == 0:
        return {q: float(curve[q][0]) for q in ("step", *QUANTITIES)}
    b0, b1 = curve["bpb"][i - 1], curve["bpb"][i]
    w = (b0 - target) / (b0 - b1)
    return {q: float(curve[q][i - 1] + w * (curve[q][i] - curve[q][i - 1])) for q in ("step", *QUANTITIES)}


def default_target() -> float:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import target

    return target.compute(target.val_shard_path(None), (3.28,))["target_bpb"]["3.28"]


def report(runs: list[str], labels: list[str], target: float) -> list[dict]:
    rows = []
    for run, label in zip(runs, labels):
        curve = run_curve(run)
        hit = crossing(curve, target) if len(curve["bpb"]) else None
        rows.append({"run": label, "dir": run, "reached": hit is not None,
                     "best_bpb": float(curve["bpb"].min()) if len(curve["bpb"]) else None,
                     **({f"{k}_to_target": v for k, v in hit.items()} if hit else {}), "_curve": curve})
    return rows


def print_table(rows: list[dict], target: float):
    print(f"target bpb {target:.4f}\n")
    print("| run | reached | best bpb | step | time (min) | PFLOPs | tokens_eq (M) | bytes (M) |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        best = f"{r['best_bpb']:.4f}" if r["best_bpb"] is not None else "-"
        if r["reached"]:
            print(f"| {r['run']} | yes | {best} | {r['step_to_target']:.0f} | {r['time_s_to_target'] / 60:.1f} | "
                  f"{r['flops_to_target'] / 1e15:.1f} | {r['tokens_eq_to_target'] / 1e6:.0f} | "
                  f"{r['bytes_to_target'] / 1e6:.0f} |")
        else:
            print(f"| {r['run']} | no | {best} | - | - | - | - | - |")


def plot(rows: list[dict], target: float, path: str):
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(repo, "logbook"))
    import plotstyle

    plotstyle.apply()
    import matplotlib.pyplot as plt

    panels = [("flops", "training FLOPs", 1.0), ("bytes", "training bytes (GB)", 1e-9),
              ("time_s", "training time (min)", 1 / 60)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    assert len(rows) <= len(plotstyle.SERIES), "more runs than categorical slots: split the report"
    for ax, (q, xlabel, scale) in zip(axes, panels):
        for r, color in zip(rows, plotstyle.SERIES):
            c = r["_curve"]
            ax.plot(c[q] * scale, c["bpb"], color=color, label=r["run"])
            if q == panels[-1][0] and len(rows) <= 4 and len(c["bpb"]):
                plotstyle.direct_label(ax, c[q][-1] * scale, c["bpb"][-1], r["run"])
        ax.axhline(target, color=plotstyle.INK_2, linewidth=1, linestyle="--")
        ax.set_xlabel(xlabel)
        if q == "flops":
            ax.set_xscale("log")
    axes[0].set_ylabel("val bpb")
    axes[0].set_title("Speedrun val bpb (dashed: 3.28-nats target)")
    axes[-1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+", help="run output directories")
    ap.add_argument("--labels", nargs="+", help="names for the runs (default: directory names)")
    ap.add_argument("--target-bpb", type=float, help="default: 3.28 nats/token converted (target.py)")
    ap.add_argument("--plot", help="write bpb-vs-FLOPs/bytes/time curves to this PNG")
    ap.add_argument("--json", help="write the table as json")
    args = ap.parse_args(argv)
    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    assert len(labels) == len(args.runs), "one label per run"
    target = args.target_bpb if args.target_bpb is not None else default_target()
    rows = report(args.runs, labels, target)
    print_table(rows, target)
    if args.json:
        with open(args.json, "w") as f:
            json.dump([{k: v for k, v in r.items() if k != "_curve"} for r in rows], f, indent=1)
    if args.plot:
        plot(rows, target, args.plot)


if __name__ == "__main__":
    main()
