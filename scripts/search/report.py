"""Report of a sweep: tables (markdown + json), best parameters, figures.

    python scripts/search/report.py SPEC [--runs DIR] [--gcs gs://...] [--out DIR] [--param name=value ...]
        [--params-file F] [--no-figures] [--n-boot 300]

Reads every job of the spec from ``--runs`` (default: the spec's ``out_root``); with ``--gcs`` the prefix is first
mirrored into ``--runs`` without checkpoints. Unfinished runs appear in the curves; tables use finished seeds.
Writes to ``--out`` (default ``<runs>/report``):

* ``report.md``: per architecture x grid setting: final val bpb (mean ± 95% t-interval over seeds), Δ vs the
  baseline at the same setting with a paired interval (same seeds), paper7 soft bpb and acc_norm, realized
  bytes/position; best setting per architecture (the tuned comparison: each architecture at its best value vs
  the baseline at its best, paired over seeds), data multiplier vs the baseline (stable phase vs stable phase,
  hnet.eval.stats), stage-0/1 chunk diagnostics (diag.json) next to the blog's targets (EXP-001);
* ``summary.json`` (all numbers) and ``best_params.yaml`` (``best_<alias>: {arch: value}``, ready for a later
  spec's ``--params-file`` or to paste into its ``params:``);
* ``figures/``: ``overview.png`` (train bpb, val bpb, bytes per position vs bytes; each architecture at its best
  setting), ``val_bpb.png`` / ``train_bpb.png`` (small multiples: architecture groups x grid settings),
  ``lr_sweep.png`` (final val bpb vs the swept value per architecture: U-curves) and ``compression.png``.
  Mean over seeds with a min-max seed band; at most 5 series per panel (the baseline in gray in every panel;
  spec ``report.groups`` defines panels, else architectures are chunked in spec order).

The baseline is spec ``report.baseline`` (default: a run named ``bpe``, else the first run).
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(_Path(__file__).resolve().parent), str(REPO_ROOT / "logbook")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Optional  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from results import (BLOG_TARGET_S1, Sweep, acc_norm, bytes_per_position, final_train_record,  # noqa: E402
                     load_sweep, mean_ci, multiplier, paired_ci, pull, soft_bpb)
from spec import Spec, fmt_value  # noqa: E402

MAX_SERIES = 5
BASELINE_COLOR = "#8c8b86"
DIAG_KEYS = ["bytes_per_chunk", "pct_cut_after_space", "pct_word_internal", "pct_mid_utf8"]


# ------------------------------------------------------------------------------------------ selection helpers
def baseline_of(spec: Spec) -> str:
    b = spec.report_cfg().get("baseline")
    names = spec.run_names()
    if b:
        if b not in names:
            raise ValueError(f"report.baseline {b!r} is not a run of {spec.name}")
        return b
    return "bpe" if "bpe" in names else names[0]


def label_of(spec: Spec, arch: str) -> str:
    return (spec.report_cfg().get("labels") or {}).get(arch, arch)


def point_label(point: dict, aliases: Dict[str, str]) -> str:
    return ", ".join(f"{aliases.get(k, k)}={fmt_value(v)}" for k, v in point.items()) or "-"


def best_points(sweep: Sweep) -> Dict[str, dict]:
    """{arch: {"point", "mean", "n"}}: the grid point with the lowest mean final val bpb (finished seeds;
    a diverged seed counts as +inf). Architectures without any finished run are absent."""
    out = {}
    for arch in sweep.archs():
        best = None
        for p in sweep.points(arch):
            vals = sweep.final("val/bpb", arch, p)
            if not vals:
                continue
            m = float(np.mean(list(vals.values())))
            if best is None or m < best["mean"]:
                best = {"point": p, "mean": m, "n": len(vals)}
        if best is not None:
            out[arch] = best
    return out


def panel_groups(sweep: Sweep, baseline: str, archs: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """{panel title: archs} with at most MAX_SERIES series per panel (baseline included in every panel)."""
    present = archs if archs is not None else sweep.archs()
    groups = sweep.spec.report_cfg().get("groups")
    raw = {}
    if groups:
        for title, members in groups.items():
            raw[title] = [a for a in members if a in present and a != baseline]
    else:
        others = [a for a in present if a != baseline]
        if others:
            raw["Architectures"] = others
    out = {}
    for title, members in raw.items():
        chunks = [members[i:i + MAX_SERIES - 1] for i in range(0, len(members), MAX_SERIES - 1)] or [[]]
        for k, ch in enumerate(chunks):
            if ch or baseline in present:
                out[title if len(chunks) == 1 else f"{title} ({k + 1}/{len(chunks)})"] = ch
    return out


# ------------------------------------------------------------------------------------------ formatting
def f_ci(ci: dict, d: int = 3) -> str:
    if ci["n"] == 0:
        return "—"
    if not math.isfinite(ci["mean"]):
        return "diverged" if ci["mean"] > 0 else "—"
    if not math.isfinite(ci["lo"]):
        return f"{ci['mean']:.{d}f} (n={ci['n']})"
    return f"{ci['mean']:.{d}f} ± {(ci['hi'] - ci['lo']) / 2:.{d}f}"


def f_delta(ci: dict, d: int = 3) -> str:
    if ci["n"] == 0 or not math.isfinite(ci["mean"]):
        return "—"
    if not math.isfinite(ci["lo"]):
        return f"{ci['mean']:+.{d}f} (n={ci['n']})"
    return f"{ci['mean']:+.{d}f} [{ci['lo']:+.{d}f}, {ci['hi']:+.{d}f}]"


def f_mult(m: Optional[dict]) -> str:
    if not m or not math.isfinite(m.get("multiplier", float("nan"))):
        return "—"
    s = f"{m['multiplier']:.2f}x"
    if "ci_lo" in m:
        hi = "∞" if not math.isfinite(m["ci_hi"]) else f"{m['ci_hi']:.2f}"
        s += f" [{m['ci_lo']:.2f}, {hi}]"
    return s + ("" if m.get("in_range") else " (extrapolated)")


def _diag_ci(runs, stage: str, key: str) -> dict:
    return mean_ci([r.diag[stage][key] for r in runs if r.diag and stage in r.diag and key in r.diag[stage]])


# ------------------------------------------------------------------------------------------ tables
def build_report(sweep: Sweep, n_boot: int = 300):
    """-> (markdown, summary dict, best_params dict)."""
    spec = sweep.spec
    base = baseline_of(spec)
    grid_keys = list(dict.fromkeys(k for r in sweep.runs for k in r.job.point))
    aliases = spec.aliases(grid_keys) if grid_keys else {}
    best = best_points(sweep)
    counts = sweep.counts()
    summary = {"spec": spec.name, "baseline": base, "fairness": spec.fairness_mode(), "counts": counts,
               "grid_keys": grid_keys, "rows": [], "best": {}, "chunks": {}}
    md = [f"# Sweep report: {spec.name}", "",
          f"Spec `{spec.path}` · fairness **{spec.fairness_mode()}** · baseline **{label_of(spec, base)}** · "
          f"generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", "",
          "Runs: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) + f" (of {len(sweep.runs)}). "
          "Tables use finished seeds; ± is the 95% t-interval over seeds; Δ intervals are paired over seeds "
          "(same seed = same data order).", ""]
    if spec.missing:
        md += ["Not run (config files missing): " + ", ".join(f"{r} (`{p}`)" for r, p in spec.missing), ""]

    # ---- per architecture x setting
    md += ["## Final validation bpb per architecture and setting", "",
           "| architecture | setting | seeds | val bpb | Δ vs baseline (same setting) | soft bpb | acc_norm (%) | "
           "bytes/pos s0 | bytes/pos s1 |", "|---" * 9 + "|"]
    for arch in sweep.archs():
        for p in sweep.points(arch):
            runs = [r for r in sweep.select(arch, p) if r.finished and not r.diverged]
            vb = sweep.final("val/bpb", arch, p)
            base_vb = sweep.final("val/bpb", base, p) if p in sweep.points(base) else {}
            delta = paired_ci(vb, base_vb) if arch != base and base_vb else {"n": 0, "mean": float("nan")}
            sb = mean_ci([soft_bpb(r.final_eval()) for r in runs if soft_bpb(r.final_eval()) is not None])
            an = mean_ci([acc_norm(r.final_eval()) for r in runs if acc_norm(r.final_eval()) is not None])
            bpp = [bytes_per_position(final_train_record(r)) for r in runs]
            bpp0 = mean_ci([b[0] for b in bpp if len(b) > 0])
            bpp1 = mean_ci([b[1] for b in bpp if len(b) > 1])
            n_tot = len(sweep.select(arch, p))
            n_div = sum(r.diverged for r in sweep.select(arch, p))
            star = " ★" if grid_keys and arch in best and best[arch]["point"] == p else ""
            md.append(f"| {label_of(spec, arch)} | {point_label(p, aliases)}{star} | {len(vb)}/{n_tot}"
                      + (f" ({n_div} diverged)" if n_div else "") + f" | {f_ci(mean_ci(vb.values()))} | "
                      f"{'—' if arch == base else f_delta(delta)} | {f_ci(sb)} | {f_ci(an, 1)} | {f_ci(bpp0, 2)} | "
                      f"{f_ci(bpp1, 2)} |")
            summary["rows"].append({"arch": arch, "point": p, "val_bpb": mean_ci(vb.values()), "seeds": vb,
                                    "delta_vs_baseline": delta, "soft_bpb": sb, "acc_norm": an,
                                    "bytes_per_pos": [bpp0, bpp1], "n_total": n_tot, "n_diverged": n_div})
    md.append("")

    # ---- best setting per architecture (tuned comparison) + data multiplier
    best_params = {}
    if base in best:
        bp = best[base]["point"]
        base_best = sweep.final("val/bpb", base, bp)
        title = "## Best setting per architecture (each at its best value; Δ and multiplier vs the baseline at " \
                "its best)" if grid_keys else "## Comparison with the baseline"
        md += [title, "", "| architecture | " + ("best setting | " if grid_keys else "") + "val bpb | Δ vs baseline "
               "(paired) | data multiplier vs baseline | note |", "|---" * (5 + bool(grid_keys)) + "|"]
        for arch in sweep.archs():
            if arch not in best:
                continue
            p = best[arch]["point"]
            vb = sweep.final("val/bpb", arch, p)
            delta = paired_ci(vb, base_best) if arch != base else {"n": 0, "mean": float("nan")}
            mult = None if arch == base else multiplier(
                [r for r in sweep.select(base, bp)], [r for r in sweep.select(arch, p)], n_boot=n_boot)
            note = []
            for k in grid_keys:
                vals = sorted({q[k] for q in sweep.points(arch)}, key=lambda v: (str(type(v)), v))
                if len(vals) > 2 and p[k] in (vals[0], vals[-1]) and isinstance(p[k], (int, float)):
                    note.append(f"best {aliases[k]} at the grid edge: extend the grid")
            md.append(f"| {label_of(spec, arch)} | " + (f"{point_label(p, aliases)} | " if grid_keys else "")
                      + f"{f_ci(mean_ci(vb.values()))} | {'—' if arch == base else f_delta(delta)} | "
                      f"{'—' if arch == base else f_mult(mult)} | {'; '.join(note)} |")
            summary["best"][arch] = {"point": p, "val_bpb": mean_ci(vb.values()), "delta_vs_baseline": delta,
                                     "multiplier": mult}
            for k in grid_keys:
                best_params.setdefault(f"best_{aliases[k]}", {})[arch] = p[k]
        md += ["", "Data multiplier: baseline tokens needed to reach the architecture's val bpb ÷ its tokens, both "
               "in the stable phase (lr_factor = 1, before the WSD decay); power-law fit of the baseline's mean "
               "curve, bootstrap interval (hnet.eval.stats.data_multiplier).", ""]

    # ---- chunk diagnostics
    hnet_archs = [a for a in sweep.archs() if any(r.diag for r in sweep.select(a))]
    if hnet_archs:
        md += ["## Chunk diagnostics (diag.json: blog English examples + FineWeb-Edu val docs)", "",
               "| architecture | setting | s0 bytes/chunk | s1 bytes/chunk | s1 % cuts after space | "
               "s1 % word-internal | s1 % mid-UTF-8 |", "|---" * 7 + "|"]
        for arch in hnet_archs:
            p = best.get(arch, {}).get("point", sweep.points(arch)[0])
            runs = [r for r in sweep.select(arch, p) if r.diag]
            row = {f"{s}_{k}": _diag_ci(runs, s, k) for s in ("s0", "s1") for k in DIAG_KEYS}
            summary["chunks"][arch] = {"point": p, **row}
            md.append(f"| {label_of(spec, arch)} | {point_label(p, aliases)} | {f_ci(row['s0_bytes_per_chunk'], 2)} | "
                      f"{f_ci(row['s1_bytes_per_chunk'], 2)} | {f_ci(row['s1_pct_cut_after_space'], 1)} | "
                      f"{f_ci(row['s1_pct_word_internal'], 1)} | {f_ci(row['s1_pct_mid_utf8'], 2)} |")
        t = BLOG_TARGET_S1
        md += [f"| *blog's improved H-Net (EXP-001, English)* | | | {t['bytes_per_chunk']} | "
               f"{t['pct_cut_after_space']} | {t['pct_word_internal']} | {t['pct_mid_utf8']} |", ""]
    summary["best_params"] = best_params
    return "\n".join(md) + "\n", summary, best_params


# ------------------------------------------------------------------------------------------ figures
def smooth_curve(y, k: int = 5):
    """Centred rolling mean over k logged points (no lag; edges use the available neighbours)."""
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    for i in range(len(y)):
        lo, hi = max(0, i - k // 2), min(len(y), i + k // 2 + 1)
        out[i] = y[lo:hi].mean()
    return out


def seed_curve(runs, key: str, source: str, smooth: bool = False, scale=None):
    """(x bytes, mean, min, max) over seeds on the steps every seed has logged; None if nothing logged."""
    per = []
    for r in runs:
        recs = [q for q in (r.evals if source == "eval" else r.metrics) if key in q and q[key] is not None]
        if not recs:
            continue
        y = np.array([float(q[key]) if scale is None else scale(q) for q in recs])
        per.append(({int(q["step"]): i for i, q in enumerate(recs)}, recs,
                    smooth_curve(y) if smooth else y))
    if not per:
        return None
    steps = sorted(set.intersection(*[set(m) for m, _, _ in per]))
    if not steps:
        return None
    x = np.array([np.mean([recs[m[s]]["bytes_seen"] for m, recs, _ in per]) for s in steps])
    ys = np.array([[y[m[s]] for s in steps] for m, _, y in per])
    return x, ys.mean(0), ys.min(0), ys.max(0)


def _colors(sweep: Sweep, baseline: str) -> Dict[str, str]:
    import plotstyle

    others = [a for a in sweep.spec.run_names() if a != baseline]
    return {baseline: BASELINE_COLOR, **{a: plotstyle.SERIES[i % len(plotstyle.SERIES)] for i, a in enumerate(others)}}


def _draw(ax, curve, color, label, marker=None, ls="-"):
    if curve is None:
        return
    x, m, lo, hi = curve
    ax.plot(x / 1e6, m, color=color, label=label, marker=marker, ms=3, ls=ls)
    if np.any(hi > lo):
        ax.fill_between(x / 1e6, lo, hi, color=color, alpha=0.15, lw=0)


def _bpp(stage: int):
    def f(q):
        b = bytes_per_position(q)
        return b[stage] if len(b) > stage else float("nan")
    return f


def _clip_bpb(ax):
    ax.set_ylim(top=min(ax.get_ylim()[1], 3.5))


def fig_curves(sweep, path, key, source, smooth, ylabel, by_setting: bool, best, title):
    import matplotlib.pyplot as plt

    spec, base = sweep.spec, baseline_of(sweep.spec)
    colors = _colors(sweep, base)
    groups = panel_groups(sweep, base)
    points = sweep.points(spec.run_names()[0]) if by_setting else [None]
    grid_keys = list(points[0]) if points[0] else []
    aliases = spec.aliases(grid_keys) if grid_keys else {}
    fig, axes = plt.subplots(len(groups), len(points), figsize=(3.6 * len(points) + 1.5, 3.4 * len(groups) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    for gi, (gtitle, members) in enumerate(groups.items()):
        for pi, p in enumerate(points):
            ax = axes[gi, pi]
            for arch in [base] + members:
                pt = p if p is not None else best.get(arch, {}).get("point", sweep.points(arch)[0])
                if arch not in sweep.archs():
                    continue
                _draw(ax, seed_curve(sweep.select(arch, pt), key, source, smooth), colors[arch],
                      label_of(spec, arch), marker="o" if source == "eval" else None)
            _clip_bpb(ax)
            head = gtitle if p is None else (f"{gtitle} · {point_label(p, aliases)}" if len(groups) > 1
                                             else point_label(p, aliases))
            ax.set_title(head, fontsize=10)
            if pi == 0:
                ax.set_ylabel(ylabel)
            if gi == len(groups) - 1:
                ax.set_xlabel("Training bytes seen (M)")
        axes[gi, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_overview(sweep, path, best, title):
    """Rows: train bpb, val bpb, bytes per position; columns: architecture groups; best setting per arch."""
    import matplotlib.pyplot as plt

    import plotstyle

    spec, base = sweep.spec, baseline_of(sweep.spec)
    colors = _colors(sweep, base)
    groups = panel_groups(sweep, base)
    fig, axes = plt.subplots(3, len(groups), figsize=(5.8 * len(groups), 11), sharex=True, squeeze=False)
    for col, (gtitle, members) in enumerate(groups.items()):
        for row, (key, source, smooth, ylabel) in enumerate([
                ("bpb", "train", True, "Training bits per byte"), ("val/bpb", "eval", False, "Validation bits per byte")]):
            ax = axes[row, col]
            for arch in [base] + members:
                if arch in sweep.archs():
                    pt = best.get(arch, {}).get("point", sweep.points(arch)[0])
                    _draw(ax, seed_curve(sweep.select(arch, pt), key, source, smooth), colors[arch],
                          label_of(spec, arch), marker="o" if source == "eval" else None)
            ax.set_ylabel(ylabel)
            _clip_bpb(ax)
            if row == 0:
                ax.set_title(gtitle)
        ax = axes[2, col]
        for arch in members:
            if arch in sweep.archs():
                pt = best.get(arch, {}).get("point", sweep.points(arch)[0])
                for s, ls in ((0, "-"), (1, "--")):
                    _draw(ax, seed_curve(sweep.select(arch, pt), f"compression/s{s}", "train", True, _bpp(s)),
                          colors[arch], None, ls=ls)
        ax.set_ylabel("Bytes per position (solid s0, dashed s1)")
        ax.set_xlabel("Training bytes seen (M)")
        axes[0, col].legend(loc="upper right", fontsize=8)
    counts = sweep.counts()
    fig.suptitle(f"{title} · {sum(counts.get(k, 0) for k in ('running', 'done', 'diverged'))} runs logging, "
                 f"{counts.get('done', 0) + counts.get('diverged', 0)} finished", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.005, "Mean over seeds, band = seed min-max; each architecture at its best setting. Equal "
             "bytes/step and FLOPs/byte (fairness.md).", fontsize=8, color=plotstyle.INK_2)
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_lr_sweep(sweep, path, title):
    """Final val bpb vs the (single, numeric) grid key per architecture; diverged settings marked at the top."""
    import matplotlib.pyplot as plt

    spec, base = sweep.spec, baseline_of(sweep.spec)
    keys = list(sweep.points(spec.run_names()[0])[0])
    if len(keys) != 1:
        return None
    key = keys[0]
    colors = _colors(sweep, base)
    groups = panel_groups(sweep, base)
    fig, axes = plt.subplots(1, len(groups), figsize=(5.2 * len(groups), 4.2), squeeze=False, sharey=True)
    for gi, (gtitle, members) in enumerate(groups.items()):
        ax = axes[0, gi]
        top = []
        for arch in [base] + members:
            if arch not in sweep.archs():
                continue
            xs, ms, err, div = [], [], [], []
            for p in sweep.points(arch):
                if not isinstance(p[key], (int, float)):
                    return None
                vals = sweep.final("val/bpb", arch, p)
                if not vals:
                    continue
                if any(not math.isfinite(v) for v in vals.values()):
                    div.append(p[key])
                    continue
                ci = mean_ci(vals.values())
                xs.append(p[key])
                ms.append(ci["mean"])
                err.append(0 if not math.isfinite(ci["lo"]) else (ci["hi"] - ci["lo"]) / 2)
            if xs:
                ax.errorbar(xs, ms, yerr=err, color=colors[arch], marker="o", capsize=3, label=label_of(spec, arch))
                top.append(max(ms))
            for x in div:
                ax.annotate("×", (x, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top",
                            color=colors[arch], fontsize=12)
        ax.set_xscale("log")
        ax.set_xlabel(key + "  (× at the top: a seed diverged)")
        ax.set_title(gtitle, fontsize=10)
        if gi == 0:
            ax.set_ylabel("Final validation bits per byte")
        ax.legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_compression(sweep, path, best, title):
    import matplotlib.pyplot as plt

    spec, base = sweep.spec, baseline_of(sweep.spec)
    colors = _colors(sweep, base)
    hn = [a for a in sweep.archs() if any(bytes_per_position(q) for r in sweep.select(a) for q in r.metrics[-1:])]
    if not hn:
        return None
    chunks = [hn[i:i + MAX_SERIES] for i in range(0, len(hn), MAX_SERIES)]
    fig, axes = plt.subplots(1, len(chunks), figsize=(6 * len(chunks), 4.2), squeeze=False, sharey=True)
    for ci_, members in enumerate(chunks):
        ax = axes[0, ci_]
        for arch in members:
            pt = best.get(arch, {}).get("point", sweep.points(arch)[0])
            for s, ls in ((0, "-"), (1, "--")):
                _draw(ax, seed_curve(sweep.select(arch, pt), f"compression/s{s}", "train", True, _bpp(s)),
                      colors[arch], label_of(spec, arch) if s == 0 else None, ls=ls)
        ax.set_xlabel("Training bytes seen (M)")
        if ci_ == 0:
            ax.set_ylabel("Bytes per position (solid s0, dashed s1)")
        ax.legend(fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figures(sweep: Sweep, out: Path, best) -> List[Path]:
    import plotstyle

    plotstyle.apply()
    out.mkdir(parents=True, exist_ok=True)
    name = sweep.spec.name
    has_grid = bool(sweep.points(sweep.spec.run_names()[0])[0])
    paths = [fig_overview(sweep, out / "overview.png", best, name),
             fig_curves(sweep, out / "val_bpb.png", "val/bpb", "eval", False, "Validation bits per byte", has_grid,
                        best, f"{name} · validation bpb"),
             fig_curves(sweep, out / "train_bpb.png", "bpb", "train", True, "Training bits per byte", has_grid, best,
                        f"{name} · training bpb (centred 5-point mean)"),
             fig_lr_sweep(sweep, out / "lr_sweep.png", f"{name} · final val bpb vs swept value") if has_grid else None,
             fig_compression(sweep, out / "compression.png", best, f"{name} · bytes per stage position")]
    return [p for p in paths if p is not None]


# ------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--runs", default=None, help="directory holding the run dirs (default: spec out_root)")
    ap.add_argument("--gcs", default=None, help="mirror this gs:// prefix into --runs first (no checkpoints)")
    ap.add_argument("--out", default=None, help="output dir (default: <runs>/report)")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--params-file", default=None)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--n-boot", type=int, default=300, help="bootstrap draws for the data multiplier")
    a = ap.parse_args(argv)
    spec = Spec.load(a.spec, a.param, a.params_file)
    runs_dir = Path(a.runs or spec.out_root)
    if a.gcs:
        pull(a.gcs, runs_dir)
    sweep = load_sweep(spec, runs_dir)
    if not any(r.exists for r in sweep.runs):
        raise SystemExit(f"no run directories of {spec.name} under {runs_dir}")
    out = Path(a.out) if a.out else runs_dir / "report"
    out.mkdir(parents=True, exist_ok=True)
    md, summary, best_params = build_report(sweep, n_boot=a.n_boot)
    figs = [] if a.no_figures else make_figures(sweep, out / "figures", best_points(sweep))
    if figs:
        md += "\n## Figures\n\n" + "\n".join(f"![{p.stem}](figures/{p.name})" for p in figs) + "\n"
    (out / "report.md").write_text(md)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "best_params.yaml").write_text(yaml.safe_dump(best_params, sort_keys=False) if best_params else "{}\n")
    print(md)
    print(f"[report] wrote {out}/report.md, summary.json, best_params.yaml" + (f", {len(figs)} figures" if figs else ""))
    return 0


if __name__ == "__main__":
    sys = _sys
    sys.exit(main())
