"""Early training dynamics of the tiny 2-stage H-Net (probe run, metrics logged every step).

    python logbook/probe_dynamics.py [--metrics runs/gcp/probe/tiny_hnet2_dyn/metrics.jsonl]

Writes logbook/figures/probe_dynamics.png: (1) compression per stage, (2) router confidence margin |p - 0.5|,
(3) decoder input: RMS of the dechunked hierarchy signal vs the residual path, (4) per-stage gradient norms.
"""

import argparse
import json
from pathlib import Path

import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=str(ROOT / "runs/gcp/probe/tiny_hnet2_dyn/metrics.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "logbook/figures/probe_dynamics.png"))
    a = ap.parse_args()
    rs = [json.loads(line) for line in open(a.metrics) if line.strip()]
    x = [r["step"] for r in rs]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    c = plotstyle.SERIES
    for s in (0, 1):
        ax[0, 0].plot(x, [r[f"compression/s{s}"] for r in rs], color=c[s], label=f"stage {s}")
        ax[0, 1].plot(x, [r[f"p_margin/s{s}"] for r in rs], color=c[s], label=f"stage {s}")
        ax[1, 0].plot(x, [r[f"dechunk_rms/s{s}"] for r in rs], color=c[s], label=f"stage {s}: hierarchy (dechunked)")
        ax[1, 0].plot(x, [r[f"residual_rms/s{s}"] for r in rs], color=c[s], ls="--", label=f"stage {s}: residual")
    for s, name in ((0, "stage 0 (byte enc/dec)"), (1, "stage 1 enc/dec"), (2, "main network")):
        ax[1, 1].plot(x, [r[f"grad_norm/stage{s}"] for r in rs], color=c[s], label=name)
    ax[0, 0].set_yscale("log")
    ax[0, 0].axhline(3, color=plotstyle.INK_2, lw=0.8, ls=":")
    ax[0, 0].set_ylabel("Bytes per chunk (vs previous stage)")
    ax[0, 0].set_title("Compression (target 3 per stage)")
    ax[0, 1].set_ylabel("Mean |p − 0.5|  (0 = coin flips)")
    ax[0, 1].set_title("Router confidence")
    ax[1, 0].set_ylabel("RMS entering the decoder")
    ax[1, 0].set_title("Hierarchy signal vs residual path")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_ylabel("Gradient norm")
    ax[1, 1].set_title("Gradient norm per stage")
    for a_ in ax.flat:
        a_.set_xscale("log")
        a_.set_xlim(1, x[-1])
        a_.legend(fontsize=8)
    for a_ in ax[1]:
        a_.set_xlabel("Training step")
    for xs, lab in ((20, "router\ncollapse"), (80, "residual\novertakes")):
        for a_ in ax.flat:
            a_.axvline(xs, color=plotstyle.INK_2, lw=0.6, ls=":")
        ax[0, 0].text(xs, ax[0, 0].get_ylim()[1] * 0.35, " " + lab, fontsize=8, color=plotstyle.INK_2)
    fig.suptitle("Tiny 2-stage H-Net (paper router): the first 400 steps, logged every step", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
