"""Aggregate evaluation results into markdown tables and matplotlib plots.

    python -m hnet.eval.report --results r1.json r2.json ... [--md report.md]
    python -m hnet.eval.report --runs runs/a1 runs/a2 --plots out_dir [--baseline runs/a1]
                               [--metric val/bpb] [--x tokens_eq|flops] [--reference configs/reference/blog1_fig7_smollm3.csv]

* ``--results``: ``hnet.eval.run`` json files -> one table per suite (rows = models, columns = tasks'
  primary metric in %, ``±`` half-width of the 95% bootstrap CI) plus averages and robustness.
* ``--runs``: trainer run dirs (``eval.jsonl`` with ``step, tokens_eq, bytes_seen, val/bpb, downstream/...``;
  ``metrics.jsonl`` for ``flops_per_byte``) -> curves of ``--metric`` vs tokens and vs training FLOPs
  (bytes_seen x flops_per_byte). With ``--baseline``, every other run's last point is annotated with the data
  multiplier vs the baseline curve (``hnet.eval.stats.data_multiplier``, power law for bpb, isotonic for scores).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

from hnet.eval.stats import data_multiplier

# Validated categorical order (dataviz reference palette, light mode); assigned by run order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


# ====================================================================== io
def read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def load_run_curve(run_dir: str, metric: str = "val/bpb") -> Dict[str, np.ndarray]:
    """Points of ``metric`` from ``eval.jsonl`` (falls back to ``metrics.jsonl``), with tokens, bytes, FLOPs."""
    ev = [r for r in read_jsonl(os.path.join(run_dir, "eval.jsonl")) if metric in r]
    mets = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    if not ev:
        ev = [r for r in mets if metric in r]
    fpb = [r["flops_per_byte"] for r in mets if r.get("flops_per_byte")]
    fpb = float(np.median(fpb)) if fpb else float("nan")
    ev.sort(key=lambda r: r.get("step", 0))
    tok = np.array([r.get("tokens_eq", np.nan) for r in ev], dtype=float)
    byt = np.array([r.get("bytes_seen", np.nan) for r in ev], dtype=float)
    val = np.array([r[metric] for r in ev], dtype=float)
    return {"tokens": tok, "bytes": byt, "flops": byt * fpb, "value": val, "flops_per_byte": fpb}


# ====================================================================== tables
def _fmt(m: dict, pct: bool = True) -> str:
    if m is None:
        return "–"
    k = 100.0 if pct else 1.0
    v = m["value"] * k
    if m.get("lo") is not None and np.isfinite(m.get("lo", np.nan)):
        hw = (m["hi"] - m["lo"]) / 2 * k
        return f"{v:.1f} ± {hw:.1f}" if pct else f"{v:.3f} ± {hw:.3f}"
    return f"{v:.1f}" if pct else f"{v:.3f}"


def results_tables(results: List[dict]) -> str:
    lines = []
    suites = []
    for r in results:
        for s in r.get("suites", {}):
            if s not in suites:
                suites.append(s)
    for s in suites:
        rows = [(r.get("model"), r["suites"][s]) for r in results if s in r.get("suites", {})]
        lines.append(f"### {s}\n")
        if s == "bpb":
            lines.append("| model | bpb | bytes |\n|---|---|---|")
            for name, d in rows:
                lines.append(f"| {name} | {d['bpb']:.4f} | {d.get('n_bytes', '')} |")
        elif s == "robustness":
            perts = list(rows[0][1]["perturbations"])
            lines.append("| model | clean | " + " | ".join(perts) + " | average | robustness score |")
            lines.append("|---" * (len(perts) + 4) + "|")
            for name, d in rows:
                cells = [_fmt(d["clean"])] + [_fmt(d["perturbations"][p]) for p in perts] + [_fmt(d["average"])]
                sc = d["robustness_score"]
                cells.append(f"{sc['value']:.1f}")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        else:
            tasks = list(rows[0][1]["tasks"])
            derived = list(rows[0][1].get("derived", {}))
            avgs = [a for a in rows[0][1].get("averages", {}) if a.endswith(("/primary", "/acc_norm_token"))]
            head = []
            for t in tasks:
                head.append(f"{t} ({rows[0][1]['tasks'][t]['primary']})")
            head += [f"{d} (mean)" for d in derived] + avgs
            lines.append("| model | " + " | ".join(head) + " |")
            lines.append("|---" * (len(head) + 1) + "|")
            for name, d in rows:
                cells = [_fmt(d["tasks"][t]["metrics"].get(d["tasks"][t]["primary"])) if t in d["tasks"] else "–"
                         for t in tasks]
                for dn in derived:
                    prim = d["tasks"][DERIVED_FIRST(d, dn)]["primary"]
                    cells.append(_fmt(d.get("derived", {}).get(dn, {}).get(prim)))
                cells += [_fmt(d.get("averages", {}).get(a)) for a in avgs]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            # soft accuracy (gold-continuation bpb)
            lines.append("\nsoft accuracy (bpb of the gold continuation, lower is better)\n")
            lines.append("| model | " + " | ".join(t for t in tasks) + " |")
            lines.append("|---" * (len(tasks) + 1) + "|")
            for name, d in rows:
                cells = [_fmt(d["tasks"][t]["metrics"].get("soft_bpb", d["tasks"][t]["metrics"].get("bpb")), pct=False)
                         if t in d["tasks"] else "–" for t in tasks]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def DERIVED_FIRST(summary: dict, name: str) -> str:
    from hnet.eval.tasks import DERIVED

    return DERIVED[summary["suite"]][name][0]


# ====================================================================== plots
def _style(ax, xlabel, ylabel, logx=True):
    ax.set_facecolor(SURFACE)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, color=GRID, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)


def plot_runs(runs: Dict[str, Dict[str, np.ndarray]], out_dir: str, metric: str = "val/bpb",
              baseline: Optional[str] = None, method: Optional[str] = None) -> dict:
    """Two charts (vs tokens, vs FLOPs); returns the data multipliers vs ``baseline``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    lower_better = "bpb" in metric or "loss" in metric
    mults = {}
    if baseline is not None and baseline in runs:
        b = runs[baseline]
        ok = np.isfinite(b["tokens"]) & np.isfinite(b["value"]) & (b["tokens"] > 0)
        meth = method or ("powerlaw" if lower_better else "isotonic")
        for name, r in runs.items():
            if name == baseline or len(r["value"]) == 0 or ok.sum() < 3:
                continue
            m = data_multiplier(b["tokens"][ok], b["value"][ok], r["tokens"][-1], r["value"][-1], method=meth,
                                increasing=not lower_better, n_boot=300)
            mults[name] = m
    for xkey, xlabel in (("tokens", "training tokens (tok-eq)"), ("flops", "training FLOPs")):
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        for k, (name, r) in enumerate(runs.items()):
            c = SERIES[k] if k < len(SERIES) else INK2
            x, y = r[xkey], r["value"]
            ok = np.isfinite(x) & np.isfinite(y) & (x > 0)
            if not ok.any():
                continue
            ax.plot(x[ok], y[ok], color=c, linewidth=2, marker="o", markersize=4, label=name)
            label = name
            if name in mults and np.isfinite(mults[name]["multiplier"]):
                mm = mults[name]
                label += f"  {mm['multiplier']:.2f}x [{mm.get('ci_lo', np.nan):.2f}, {mm.get('ci_hi', np.nan):.2f}]"
            ax.annotate(label, (x[ok][-1], y[ok][-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK, va="center")
        _style(ax, xlabel, metric)
        if len(runs) >= 2:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        ax.set_title(f"{metric} vs {xlabel}" + (f"; data multiplier vs {baseline}" if mults else ""),
                     color=INK, fontsize=10, loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"{metric.replace('/', '_')}_vs_{xkey}.png"), facecolor=SURFACE)
        plt.close(fig)
    return mults


def plot_task_curves(runs: Dict[str, Dict[str, np.ndarray]], out_path: str, title: str):
    """Small helper: several runs' curves of one downstream metric vs tokens (one chart per metric)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for k, (name, r) in enumerate(runs.items()):
        ok = np.isfinite(r["tokens"]) & np.isfinite(r["value"]) & (r["tokens"] > 0)
        ax.plot(r["tokens"][ok], r["value"][ok], color=SERIES[k % len(SERIES)] if k < len(SERIES) else INK2,
                linewidth=2, marker="o", markersize=4, label=name)
    _style(ax, "training tokens", title)
    if len(runs) >= 2:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def load_reference_csv(path: str, column: str) -> Dict[str, np.ndarray]:
    """configs/reference/blog1_fig7_*.csv: tokens_T plus task columns (scores in %)."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["tokens_T"]) * 1e12 for r in rows])
    v = np.array([float(r[column]) / 100.0 for r in rows])
    return {"tokens": t, "bytes": t * np.nan, "flops": t * np.nan, "value": v}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", nargs="*", default=[])
    ap.add_argument("--md", default=None)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--plots", default=None)
    ap.add_argument("--metric", default="val/bpb")
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--method", default=None, choices=[None, "powerlaw", "quadratic_log", "isotonic"])
    ap.add_argument("--reference", default=None, help="reference csv (adds a curve for --metric's column)")
    a = ap.parse_args(argv)
    md = []
    if a.results:
        res = []
        for p in a.results:
            with open(p) as f:
                res.append(json.load(f))
        md.append(results_tables(res))
    if a.runs and a.plots:
        runs = {os.path.basename(os.path.normpath(r)): load_run_curve(r, a.metric) for r in a.runs}
        if a.reference:
            runs[os.path.basename(a.reference)] = load_reference_csv(a.reference, a.metric.split("/")[-1])
        base = os.path.basename(os.path.normpath(a.baseline)) if a.baseline else None
        mults = plot_runs(runs, a.plots, a.metric, base, a.method)
        if mults:
            md.append(f"### data multiplier vs {base} ({a.metric})\n")
            md.append("| run | tokens | baseline tokens to match | multiplier | 95% CI |\n|---|---|---|---|---|")
            for name, m in mults.items():
                md.append(f"| {name} | {runs[name]['tokens'][-1]:.3g} | {m['tokens_needed']:.3g} | "
                          f"{m['multiplier']:.2f} | [{m.get('ci_lo', math.nan):.2f}, {m.get('ci_hi', math.nan):.2f}] |")
    text = "\n".join(md)
    if a.md:
        with open(a.md, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
