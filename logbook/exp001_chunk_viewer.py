"""EXP-001: what Cartesia's chunk-viewer data says about the improved H-Net router (blog I Figs. 10-11)."""

import json
import sys
from pathlib import Path

import numpy as np

import plotstyle

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "reference"))
from analyze_chunk_viewer import cuts_from_ends, cuts_from_starts, summarize  # noqa: E402

exs = json.load(open(ROOT / "configs" / "reference" / "chunk_viewer.json"))


def group(e):
    cjk = any("぀" <= ch <= "鿿" or "가" <= ch <= "힯" for ch in e["text"][:400])
    if "code" in e["bundle"]:
        return "Code"
    return "CJK" if cjk else ("English" if e.get("src") == "English" else "Other languages")


groups = ["English", "Other languages", "CJK", "Code"]
models = {
    "Improved H-Net (stage 1)": lambda e: cuts_from_ends(e["s"][1]),
    "LH-Net, paper (stage 1)": lambda e: cuts_from_starts(e["v1"][1]),
    "Llama-3 BPE": lambda e: cuts_from_starts(e["bpe"]["llama-3"]),
}
rows = {m: {g: summarize(m, [e for e in exs if group(e) == g], f) for g in groups} for m, f in models.items()}

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
w = 0.26
x = np.arange(len(groups))
for k, (m, color) in enumerate(zip(models, plotstyle.SERIES[:3])):
    axes[0].bar(x + (k - 1) * w, [rows[m][g]["pct_mid_utf8"] for g in groups], w - 0.02, color=color, label=m)
    axes[1].bar(x + (k - 1) * w, [rows[m][g]["pct_cut_after_space"] for g in groups], w - 0.02, color=color, label=m)
axes[0].set_title("Chunk boundaries that split a UTF-8 character")
axes[1].set_title("Chunks that end right after a space ('word␣')")
for ax in axes:
    ax.set_xticks(x, groups)
    ax.set_ylabel("% of boundaries")
    ax.grid(axis="x", visible=False)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.995))
fig.text(0.01, 0.01, "Data: Cartesia's chunk-viewer examples (96 texts) parsed from the blog page. Stage-1 chunks for the H-Nets; "
         "'Code' has no multi-byte characters.", fontsize=7.5, color=plotstyle.INK_2)
fig.tight_layout(rect=(0, 0.04, 1, 0.92))
fig.savefig(ROOT / "logbook" / "figures" / "exp001_chunk_viewer.png")

# A text rendering of one English example for the log.
e = next(e for e in exs if e.get("src") == "English")
data = e["text"].encode()[:96]


def seg(cuts):
    cuts = [c for c in cuts if 0 < c < len(data)]
    parts, prev = [], 0
    for c in cuts + [len(data)]:
        parts.append(data[prev:c].decode("utf-8", "replace").replace(" ", "␣").replace("\n", "⏎"))
        prev = c
    return "|".join(parts)


print("H-Net s0 :", seg(cuts_from_ends(e["s"][0])))
print("H-Net s1 :", seg(cuts_from_ends(e["s"][1])))
print("LH-Net s0:", seg(cuts_from_starts(e["v1"][0])))
print("LH-Net s1:", seg(cuts_from_starts(e["v1"][1])))
print("Llama-3  :", seg(cuts_from_starts(e["bpe"]["llama-3"])))
for m in models:
    print(m, {g: round(rows[m][g]["bytes_per_chunk"], 2) for g in groups})
