"""Shareable TL;DR images from logbook/tldr/data.json (python logbook/tldr/make_images.py).

tldr_main.png: how much better than a standard BPE model, at the small and the bigger scale.
tldr_journey.png: the best H-Net per generation of the search, against BPE and BPE + copying (small scale)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
D = {e["id"]: e for e in json.load(open(os.path.join(HERE, "data.json")))["experiments"]}
INK, INK2, MUTED, GRID, BG = "#141413", "#55534e", "#8a877f", "#e6e4dd", "#fbfaf7"
HNET, HNET_LIGHT, BPE = "#2a78d6", "#9ec5f4", "#eb6834"
fam = [f.name for f in font_manager.fontManager.ttflist]
plt.rcParams.update({"font.family": "Inter" if "Inter" in fam else "DejaVu Sans", "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def gain(eid, ref):  # % fewer bits than the standard BPE model at the same scale (positive = better)
    return 100 * (D[ref]["final"] - D[eid]["final"]) / D[ref]["final"]


def main_image():
    fig = plt.figure(figsize=(12, 6.75), dpi=150, facecolor=BG)
    fig.text(0.04, 0.92, "We got a tokenizer-free model to beat a standard tokenizer model.", fontsize=21, weight="bold")
    fig.text(0.04, 0.865, "Then we made the models bigger, and the win disappeared.", fontsize=21, weight="bold", color=INK2)
    panels = [("Small model · 20M parameters", "", "bpe", ["hnet", "bpe_copy", "g4_combo_02"]),
              ("Bigger model · 38M parameters, 10× the compute", "big_", "big_bpe", ["big_hnet", "big_bpe_copy", "big_g4_combo_02"])]
    names = ["Byte-level model (H-Net), as published", "Standard model + copy trick (the fair bar)",
             "Our best byte-level model (H-Net + copy trick + 4 more ideas)"]
    colors = [HNET_LIGHT, BPE, HNET]
    lo, hi = -19, 18
    for k, (title, _, ref, ids) in enumerate(panels):
        ax = fig.add_axes([0.04 + k * 0.49, 0.2, 0.43, 0.5], facecolor=BG)
        vals = [gain(i, ref) for i in ids]
        y = [2, 1, 0]
        ax.barh(y, vals, height=0.56, color=colors, zorder=3)
        ax.axvline(0, color=INK, lw=1.2, zorder=4)
        for yy, v in zip(y, vals):
            ax.text(v + (0.6 if v >= 0 else -0.6), yy, f"{v:+.1f}%", va="center", ha="left" if v >= 0 else "right",
                    fontsize=14, weight="bold", color=INK, zorder=5)
        ax.set_xlim(lo, hi); ax.set_ylim(-0.6, 2.6); ax.set_yticks([])
        ax.set_xticks([-10, 0, 10]); ax.set_xticklabels(["10% worse", "standard\nmodel", "10% better"], fontsize=10)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(title, loc="left", fontsize=13, weight="bold", pad=10)
        win = vals[2] > vals[1]
        diff = abs(D[ids[2]]["final"] - D[ids[1]]["final"]) / D[ids[1]]["final"] * 100
        ax.text(0.0, -0.22, ("Byte-level model wins" if win else "Standard model wins") + f" by {diff:.1f}%",
                transform=ax.transAxes, fontsize=13, weight="bold", color=HNET if win else BPE)
    # legend row, one entry per bar colour (direct names for the three bars)
    for i, (n, c) in enumerate(zip(names, colors)):
        x = 0.04 + [0, 0.29, 0.6][i]
        fig.patches.append(plt.Rectangle((x, 0.045), 0.012, 0.022, transform=fig.transFigure, color=c))
        fig.text(x + 0.018, 0.056, n, fontsize=10, va="center", color=INK2)
    fig.text(0.04, 0.80, "Bits needed per character of held-out text, compared with a standard BPE model at the same size "
             "(right = fewer bits = better).", fontsize=10.5, color=INK2)
    fig.text(0.04, 0.012, "Same compute, same parameter count, same data for every model; 2–3 training runs each. "
             "github.com/vaibhav-iint/hnet", fontsize=8.5, color=MUTED)
    out = os.path.join(HERE, "tldr_main.png"); fig.savefig(out, facecolor=BG); print(out)


def journey_image():
    fig = plt.figure(figsize=(12, 6.75), dpi=150, facecolor=BG)
    fig.text(0.05, 0.92, "Five rounds of AI-driven search to make byte-level models win", fontsize=21, weight="bold")
    fig.text(0.05, 0.87, "Bits per character on held-out text, small model (lower is better)", fontsize=12, color=INK2)
    ax = fig.add_axes([0.08, 0.14, 0.58, 0.66], facecolor=BG)
    best = [("Start", "hnet_ws"), ("Round 1", "g1_wild_01"), ("Round 2", "g2_match_mem"), ("Round 3", "g3_mem_02"),
            ("Round 4", "g4_combo_02"), ("Round 5", "g5_compute_02")]
    xs = list(range(len(best)))
    ys = [D[i]["final"] for _, i in best]
    for ref, c, lab in [("bpe", BPE, "Standard model (BPE)"), ("bpe_copy", BPE, "Standard model + copy trick")]:
        v = D[ref]["final"]
        ax.axhline(v, color=c, lw=2, ls="-" if ref == "bpe" else (0, (5, 3)), zorder=2)
        ax.text(len(best) - 0.65, v + 0.008, f"{lab}  {v:.3f}", va="bottom", ha="right", fontsize=11, color=INK)
    ax.plot(xs, ys, color=HNET, lw=2, zorder=3)
    ax.scatter(xs, ys, s=70, color=HNET, edgecolor=BG, linewidth=2, zorder=4)
    for x, y in zip(xs, ys):
        below = y < 1.47
        ax.text(x, y - 0.022 if below else y + 0.018, f"{y:.3f}", ha="center", va="top" if below else "bottom",
                fontsize=11, weight="bold", color=INK)
    ax.set_xticks(xs); ax.set_xticklabels([b for b, _ in best], fontsize=11)
    ax.set_ylim(1.38, 1.88); ax.set_xlim(-0.4, len(best) - 0.6)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(0.12, ys[0], "Byte-level model (H-Net)", ha="left", va="center", fontsize=11, color=HNET, weight="bold")
    note = ("Round 1 found the big lever: copying\nfrom earlier in the document.\n\n"
            "Rounds 3–4 combined five ideas and\npassed the standard model with the\nsame copy trick, by 1.9%.\n\n"
            "At a bigger model size (10× compute)\nthe order flips: the standard model\nwins by 2.5%.")
    fig.text(0.71, 0.3, note, fontsize=10.5, color=INK2, va="bottom", linespacing=1.35)
    fig.text(0.05, 0.03, "Each round: AI agents proposed ideas, wrote the code and tests; every idea trained 2–3 times on "
             "equal compute and data. github.com/vaibhav-iint/hnet", fontsize=8.5, color=MUTED)
    out = os.path.join(HERE, "tldr_journey.png"); fig.savefig(out, facecolor=BG); print(out)


main_image()
journey_image()
