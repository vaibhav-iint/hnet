"""EXP-002 progress figures: pull run metrics from GCS and plot training curves (re-runnable while training).

    python logbook/exp002_progress.py [--no-pull]

Writes logbook/figures/exp002_progress.png with three rows:
  1. training bpb vs bytes seen (EMA-smoothed, mean over seeds)
  2. validation bpb vs bytes seen (same val text for every run)
  3. realized compression per stage (bytes per stage-s position) for the H-Nets
Left column: boundary routers (paper LH-Net router vs end-of-chunk guesses). Right column: ablations of the paper router.
The BPE baseline is the gray reference line in every panel.
"""

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GCS = "gs://neat-dispatch-501718-a5-hnet/runs/exp002_tiny2m"
LOCAL = ROOT / "runs" / "gcp" / "exp002_tiny2m"
GROUPS = {
    "Boundary routers": ["paper", "end_linear", "end_mlp", "end_cosine_shift"],
    "Ablations of the paper router": ["paper", "paper_r42", "paper_nosmooth", "paper_nolrmult", "paper_alpha01"],
}
LABELS = {"bpe": "BPE baseline", "paper": "paper router (LH-Net)", "end_linear": "end-of-chunk, linear",
          "end_mlp": "end-of-chunk, MLP", "end_cosine_shift": "end-of-chunk, cosine",
          "paper_r42": "targets (4,2)", "paper_nosmooth": "no smoothing", "paper_nolrmult": "no LR multipliers",
          "paper_alpha01": "ratio loss α=0.1"}
BASELINE_COLOR = "#8c8b86"


def pull():
    subprocess.run(["gcloud", "storage", "rsync", "-r", "--exclude", r".*/ckpt/.*", GCS, str(LOCAL)], check=True)


def read_jsonl(p: Path):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def load_runs():
    runs = defaultdict(list)  # variant -> list of (train records, eval records)
    for d in sorted(LOCAL.glob("*_s*")):
        if not d.is_dir():
            continue
        variant = d.name.rsplit("_s", 1)[0]
        runs[variant].append((read_jsonl(d / "metrics.jsonl"), read_jsonl(d / "eval.jsonl")))
    return runs


def ema(y, alpha=0.05):
    out, m = [], None
    for v in y:
        m = v if m is None else (1 - alpha) * m + alpha * v
        out.append(m)
    return np.array(out)


def mean_curve(seeds, key, source, smooth=False):
    """Mean over seeds on the shortest common prefix of bytes_seen."""
    curves = []
    for train, ev in seeds:
        recs = [r for r in (train if source == "train" else ev) if key in r and "bytes_seen" in r]
        if recs:
            x = np.array([r["bytes_seen"] for r in recs], dtype=float)
            y = np.array([r[key] for r in recs], dtype=float)
            curves.append((x, ema(y) if smooth else y))
    if not curves:
        return None, None
    n = min(len(c[0]) for c in curves)
    return curves[0][0][:n], np.mean([c[1][:n] for c in curves], axis=0)


def plot(runs):
    fig, axes = plt.subplots(3, 2, figsize=(12, 11.5), sharex=True)
    for col, (title, variants) in enumerate(GROUPS.items()):
        for row, (key, source, smooth, ylabel) in enumerate([
            ("bpb", "train", True, "Training bits per byte"),
            ("val/bpb", "eval", False, "Validation bits per byte"),
        ]):
            ax = axes[row, col]
            x, y = mean_curve(runs.get("bpe", []), key, source, smooth)
            if x is not None:
                ax.plot(x / 1e6, y, color=BASELINE_COLOR, lw=2, label=LABELS["bpe"])
            for i, v in enumerate(variants):
                x, y = mean_curve(runs.get(v, []), key, source, smooth)
                if x is not None:
                    ax.plot(x / 1e6, y, color=plotstyle.SERIES[i], label=LABELS[v],
                            marker="o" if source == "eval" else None, ms=3)
            ax.set_ylabel(ylabel)
            ax.set_ylim(top=min(ax.get_ylim()[1], 3.5))
            if row == 0:
                ax.set_title(title)
        ax = axes[2, col]
        for i, v in enumerate(variants):
            for s, ls in ((0, "-"), (1, "--")):
                x, y = mean_curve(runs.get(v, []), f"compression/s{s}", "train", smooth=True)
                if x is not None:
                    bpp = y if s == 0 else y * mean_curve(runs[v], "compression/s0", "train", smooth=True)[1][:len(y)]
                    ax.plot(x / 1e6, bpp, color=plotstyle.SERIES[i], ls=ls)
        ax.set_ylabel("Bytes per position (solid s0, dashed s1)")
        ax.set_xlabel("Training bytes seen (M)")
        axes[0, col].legend(loc="upper right", fontsize=8)
    n_done = sum(1 for d in LOCAL.glob("*_s*/done.json"))
    n_runs = sum(len(v) for v in runs.values())
    fig.suptitle(f"EXP-002 · tiny (~2M-param, FLOPs-matched) architecture search · {n_runs} runs logging, "
                 f"{n_done} finished", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "Mean over seeds. FineWeb-Edu, equal bytes/step and FLOPs/byte (fairness.md). "
             "Targets (3,3) = 3 and 9 bytes per position; (4,2) = 4 and 8.", fontsize=8, color=plotstyle.INK_2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    out = ROOT / "logbook" / "figures" / "exp002_progress.png"
    fig.savefig(out)
    return out, n_runs, n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    a = ap.parse_args()
    if not a.no_pull:
        pull()
    runs = load_runs()
    if not runs:
        raise SystemExit(f"no runs under {LOCAL} yet")
    out, n_runs, n_done = plot(runs)
    print(f"wrote {out} ({n_runs} runs, {n_done} done)")


if __name__ == "__main__":
    main()
