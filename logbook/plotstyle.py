"""Shared matplotlib style for logbook figures (static PNGs).

Palette and mark rules follow the reference data-viz palette: fixed categorical order, 2px lines,
recessive grid/axes, text in ink colors (never the series color), direct labels for <= 4 series.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Categorical slots in fixed order (light mode). Assign by entity, never by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 4, "font.size": 10, "legend.frameon": False,
        "legend.fontsize": 9, "figure.dpi": 150,
    })


def direct_label(ax, x, y, text, dx=4, dy=0, **kw):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center",
                color=INK, fontsize=9, **kw)
