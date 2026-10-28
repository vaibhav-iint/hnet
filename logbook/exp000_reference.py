"""EXP-000: Cartesia blog I Fig. 7 reference curves and stable-vs-stable data multipliers."""

import csv
import math
from pathlib import Path

import numpy as np

import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "configs" / "reference"


def load(name):
    with open(REF / name) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


h, s = load("blog1_fig7_hnet.csv"), load("blog1_fig7_smollm3.csv")
s_stable = [r for r in s if r["phase"] == 0]
# Smooth SmolLM3's stage-1 (stable) curve: quadratic in log tokens.
coef = np.polyfit([math.log(r["tokens_T"]) for r in s_stable], [r["average6"] for r in s_stable], 2)
smooth = np.poly1d(coef)


def smol_tokens_for(score):
    grid = np.exp(np.linspace(math.log(0.05), math.log(7.9), 4000))
    above = grid[smooth(np.log(grid)) >= score]
    return float(above[0]) if len(above) else None


fig, ax = plt.subplots(figsize=(8.2, 4.6))
blue, orange = plotstyle.SERIES[0], plotstyle.SERIES[1]
ax.scatter([r["tokens_T"] for r in s], [r["average6"] for r in s], s=8, color=orange, alpha=0.35, linewidths=0)
ax.scatter([r["tokens_T"] for r in h], [r["average6"] for r in h], s=8, color=blue, alpha=0.35, linewidths=0)
xs = np.exp(np.linspace(math.log(0.094), math.log(7.9), 200))
ax.plot(xs, smooth(np.log(xs)), color=orange)
hs = [r for r in h if r["phase"] == 0]
hx = np.array([r["tokens_T"] for r in hs]); hy = np.array([r["average6"] for r in hs])
k = 7
hy_s = np.convolve(hy, np.ones(k) / k, mode="valid"); hx_s = hx[k // 2: -(k // 2)]
ax.plot(hx_s, hy_s, color=blue)
cd = [r for r in h if r["phase"] == 1]
ax.plot([r["tokens_T"] for r in cd], [r["average6"] for r in cd], color=blue, linestyle=(0, (2, 2)))
ax.axvspan(cd[0]["tokens_T"], cd[-1]["tokens_T"], color=plotstyle.GRID, alpha=0.5, lw=0)
ax.text(cd[0]["tokens_T"] * 1.02, 52.6, "H-Net\ncooldown", color=plotstyle.INK_2, fontsize=8)

for t in (0.094, 0.189):
    r = min(hs, key=lambda r: abs(r["tokens_T"] - t))
    T = smol_tokens_for(r["average6"])
    ax.annotate("", xy=(T, r["average6"]), xytext=(r["tokens_T"], r["average6"]),
                arrowprops=dict(arrowstyle="->", color=plotstyle.INK_2, lw=1))
    ax.text(math.sqrt(T * r["tokens_T"]), r["average6"] - 1.1, f"{T / r['tokens_T']:.1f}x",
            ha="center", color=plotstyle.INK, fontsize=9)

ax.set_xscale("log")
ax.set_xlim(0.005, 14)
ax.set_ylim(38, 66)
ax.set_xlabel("Training tokens (T, log scale)")
ax.set_ylabel("Six-task average (%)")
ax.set_title("Cartesia's H-Net vs SmolLM3: stable-phase data multipliers are 1.7x and 3.0x, not 10x")
plotstyle.direct_label(ax, hx_s[-1], hy_s[-1], "H-Net 3.7B (stable)", dx=-6, dy=10, ha="right")
plotstyle.direct_label(ax, s[-1]["tokens_T"], s[-1]["average6"], "SmolLM3-3B", dx=-4, dy=10, ha="right")
fig.text(0.01, 0.01, "Data: blog I Fig. 7 (extracted from the page's chart data). Six tasks: HellaSwag, ARC, PIQA, "
         "WinoGrande, CSQA, MMLU. Dots = checkpoints; lines = smoothed.", fontsize=7.5, color=plotstyle.INK_2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = ROOT / "logbook" / "figures" / "exp000_cartesia_fig7.png"
fig.savefig(out)
print("wrote", out)
for t in (0.094, 0.189, 0.301):
    r = min(hs, key=lambda r: abs(r["tokens_T"] - t))
    print(t, r["average6"], smol_tokens_for(r["average6"]))
