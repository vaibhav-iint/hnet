#!/usr/bin/env python
"""Fit a BPE-vs-H-Net scaling ladder: iso-FLOP frontiers, L(C) power laws, compute multipliers (validation, per
domain, per code language, downstream), loss-to-loss laws, N_opt(C) / D_opt(C); blog-style figures.

    python scripts/analysis/fit_ladder.py --runs 'bpe=runs/ladder/bpe_*' --runs 'hnet=runs/ladder/hnet_*' \
        [--out logbook/tables/ladder_fit.json] [--figdir logbook/figures] [--n-boot 1000] [--form auto]
    python scripts/analysis/fit_ladder.py --records synthetic_runs.json      # list of Run dicts (tests)

``--runs ARCH=GLOB`` (repeatable; ARCH optional: model.type dense -> bpe, hnet -> hnet). Each run dir is read by
``hnet.analysis.scaling.load_run`` (config.yaml, metrics.jsonl, eval.jsonl, + per_source.json from
scripts/eval/per_source.py and downstream_bpb.json from scripts/eval/downstream_bpb.py when present). A
``ladder.json`` in a run dir overrides flops / params / budget / arch / seed. See docs/ANALYSIS_LADDER.md.

Figures (``<figdir>/<prefix>_*.png``): loss_vs_compute, isoflop, domain_multipliers, code_multipliers,
compression, loss_to_loss, allocation. Each is produced only when its inputs exist.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hnet.analysis import scaling as SC  # noqa: E402

LABEL = {"bpe": "BPE Transformer", "hnet": "H-Net (1-stage)"}


def _style():
    sys.path.insert(0, os.path.join(ROOT, "logbook"))
    import plotstyle  # noqa: E402

    plotstyle.apply()
    import matplotlib.pyplot as plt

    return plt, plotstyle


def _plain_y(ax):
    from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

    ax.yaxis.set_major_locator(LogLocator(base=10, subs=np.arange(1, 10) * 0.1, numticks=30))
    f = ScalarFormatter()
    f.set_scientific(False)
    ax.yaxis.set_major_formatter(f)
    ax.yaxis.set_minor_formatter(NullFormatter())


def _color(ps, arch, ref, new):
    return ps.SERIES[0] if arch == ref else ps.SERIES[1] if arch == new else ps.SERIES[2]


def _fmt_mult(m):
    if not m or m.get("multiplier") is None:
        return "n/a"
    s = f"{m['multiplier']:.2f}x"
    ci = m.get("ci95")
    if ci:
        s += f" [{ci['lo']:.2f}, {ci['hi']:.2f}]"
    if m.get("extrapolated"):
        s += " (extrap.)"
    return s


# ====================================================================================================== figures
def plot_loss_vs_compute(res, runs, path, metric="val", ylabel="validation bpb"):
    r = res["metrics"].get(metric)
    if r is None:
        return None
    plt, ps = _style()
    ref, new = res["ref"], res["new"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    allC = [x.flops for x in runs if metric in x.metrics]
    for arch in (ref, new):
        col = _color(ps, arch, ref, new)
        xs = [(x.flops, x.metrics[metric]) for x in runs if x.arch == arch and metric in x.metrics]
        if xs:
            ax.scatter(*zip(*xs), s=14, color=col, alpha=0.35, linewidths=0, zorder=2)
        fr = r["_fronts"].get(arch, [])
        if fr:
            ax.scatter([p["C"] for p in fr], [p["L"] for p in fr], s=46, color=col, edgecolor=ps.SURFACE,
                       linewidth=1.5, zorder=4, label=f"{LABEL.get(arch, arch)} (compute-optimal)")
        law = r["_laws"].get(arch)
        if law is not None and allC:
            cc = np.logspace(math.log10(min(allC)) - 0.15, math.log10(max(allC)) + 0.3, 100)
            ax.plot(cc, law.loss(cc), color=col, lw=2, zorder=3)
            ps.direct_label(ax, cc[-1], float(law.loss(cc[-1])), f"{arch} α={law.alpha:.3f}", dx=4,
                            dy=6 if arch == ref else -6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    _plain_y(ax)
    ax.set_xlabel("training FLOPs")
    ax.set_ylabel(ylabel)
    ax.margins(x=0.12)
    m = r.get("multiplier")
    title = f"{ylabel}: compute multiplier {LABEL.get(new, new)} vs {LABEL.get(ref, ref)}"
    ax.set_title(title, fontsize=10)
    txt = f"multiplier {_fmt_mult(m)}" if m else "single budget: multiplier undefined"
    if m:
        txt += f"\nloss range {m['loss_range'][0]:.3f}-{m['loss_range'][1]:.3f}, law: {r['form']}"
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=9, color=ps.INK, va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_isoflop(res, path, metric="val"):
    r = res["metrics"].get(metric)
    if r is None:
        return None
    budgets = res["budgets"]
    plt, ps = _style()
    n = len(budgets)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n + 0.6, 3.4), squeeze=False)
    ref, new = res["ref"], res["new"]
    cells = SC.make_cells([SC.Run.from_dict(d) for d in res["runs"]], metric)
    for ax, b in zip(axes[0], budgets):
        for arch in (ref, new):
            col = _color(ps, arch, ref, new)
            cs = [c for c in cells if c.arch == arch and c.budget == b]
            if not cs:
                continue
            N = np.exp([c.logN for c in cs])
            ax.scatter(N, [c.y for c in cs], s=22, color=col, zorder=3, label=LABEL.get(arch, arch))
            p = next((p for p in r["_fronts"].get(arch, []) if p["budget"] == b), None)
            if p and p["coef"]:
                c0, c1, c2, xm = p["coef"]
                xx = np.linspace(min(c.logN for c in cs) - 0.2, max(c.logN for c in cs) + 0.2, 60)
                ax.plot(np.exp(xx), c0 + c1 * (xx - xm) + c2 * (xx - xm) ** 2, color=col, lw=2, zorder=2)
            if p:
                ax.scatter([p["N_opt"]], [p["L"]], marker="*", s=180, color=col, edgecolor=ps.SURFACE,
                           linewidth=1.0, zorder=4)
        ax.set_xscale("log")
        ax.set_title(f"{b:.2g} FLOPs", fontsize=10)
        ax.set_xlabel("parameters N")
    axes[0][0].set_ylabel(metric if metric != "val" else "validation bpb")
    axes[0][0].legend(loc="upper center")
    fig.suptitle("Iso-FLOP curves (quadratic in log N; star = compute-optimal)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _bars(ax, ps, names, vals, cis, colors):
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.7, edgecolor=ps.SURFACE, linewidth=2, zorder=2)
    for i, ci in enumerate(cis):
        if ci:
            ax.plot([ci["lo"], ci["hi"]], [i, i], color=ps.INK_2, lw=1.2, zorder=3)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}x", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=ps.INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.axvline(1.0, color=ps.INK_2, lw=1, ls="--", zorder=1)
    ax.invert_yaxis()


def _mults(res, prefix):
    out = {}
    for m, r in res["metrics"].items():
        if m.startswith(prefix) and r.get("multiplier") and r["multiplier"].get("multiplier"):
            out[m[len(prefix):]] = r["multiplier"]
    return out


def plot_domain_multipliers(res, path):
    ms = {}
    if res["metrics"].get("val", {}).get("multiplier"):
        ms["validation (all)"] = res["metrics"]["val"]["multiplier"]
    for k, v in _mults(res, "group:").items():
        ms[k] = v
    if res["metrics"].get("down:avg", {}).get("multiplier"):
        ms["downstream (paper7 bpb)"] = res["metrics"]["down:avg"]["multiplier"]
    if not ms:
        return None
    plt, ps = _style()
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(ms) + 1.2))
    names = list(ms)
    _bars(ax, ps, names, [ms[n]["multiplier"] for n in names], [ms[n].get("ci95") for n in names],
          [ps.SERIES[1]] * len(names))
    ax.set_xlabel(f"compute multiplier ({LABEL.get(res['new'])} over {LABEL.get(res['ref'])})")
    ax.set_title("Compute multiplier per domain", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_code_multipliers(res, path):
    meta = res.get("source_meta", {})
    ms = {k: v for k, v in _mults(res, "src:").items() if (meta.get(k) or {}).get("group") == "code"}
    if not ms:
        return None
    plt, ps = _style()
    held = {k: bool((meta.get(k) or {}).get("held_out")) for k in ms}
    names = sorted(ms, key=lambda k: (held[k], ms[k]["multiplier"]))
    fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(names) + 1.4))
    c_in, c_out = ps.SERIES[2], ps.SERIES[6]
    cols = [c_out if held[n] else c_in for n in names]
    labels = [((meta.get(n) or {}).get("lang") or n) for n in names]
    _bars(ax, ps, labels, [ms[n]["multiplier"] for n in names], [ms[n].get("ci95") for n in names], cols)
    for flag, col, lab in ((False, c_in, "in distribution"), (True, c_out, "held out")):
        v = [ms[n]["multiplier"] for n in names if held[n] == flag]
        if v:
            med = float(np.median(v))
            ax.axvline(med, color=col, lw=1.5, zorder=1)
            ax.bar([0], [0], color=col, label=f"{lab} (median {med:.2f}x)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.6 / max(len(names), 1)), ncol=2)
    ax.set_xlabel("compute multiplier at matched per-language bpb (vertical lines: medians)")
    ax.set_title("Compute multiplier per code language", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def compression_table(res, runs):
    """Mean bytes per main-network position (H-Net) / per token (BPE) per source, over the runs of the largest
    budget of each architecture."""
    out = {}
    for arch in (res["ref"], res["new"]):
        rs = [r for r in runs if r.arch == arch and r.info.get("compression")]
        if not rs:
            continue
        top = max(r.budget for r in rs)
        rs = [r for r in rs if r.budget == top]
        acc = {}
        for r in rs:
            for s, c in r.info["compression"].items():
                v = c.get("bytes_per_main_position") or c.get("bytes_per_token")
                if v:
                    acc.setdefault(s, []).append(v)
                if c.get("split_bytes_per_token"):
                    acc.setdefault("__split__" + s, []).append(c["split_bytes_per_token"])
        out[arch] = {"budget": top, "n_runs": len(rs), "per_source": {k: float(np.mean(v)) for k, v in acc.items()}}
    return out


def plot_compression(res, runs, path):
    tab = compression_table(res, runs)
    if not tab:
        return None
    plt, ps = _style()
    ref, new = res["ref"], res["new"]
    srcs = sorted({s for a in tab.values() for s in a["per_source"] if not s.startswith("__split__")})
    if not srcs:
        return None
    meta = res.get("source_meta", {})
    srcs.sort(key=lambda s: ((meta.get(s) or {}).get("group") or "", bool((meta.get(s) or {}).get("held_out")), s))
    fig, ax = plt.subplots(figsize=(max(6.0, 0.5 * len(srcs) + 2), 4.0))
    x = np.arange(len(srcs))
    w = 0.38
    for k, arch in enumerate((ref, new)):
        if arch not in tab:
            continue
        v = [tab[arch]["per_source"].get(s, np.nan) for s in srcs]
        lab = ("bytes / token" if arch == ref else "bytes / main-network position")
        ax.bar(x + (k - 0.5) * w, v, width=w - 0.02, color=_color(ps, arch, ref, new), edgecolor=ps.SURFACE,
               linewidth=1, label=f"{LABEL.get(arch, arch)}: {lab}", zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([((meta.get(s) or {}).get("lang") or s) + ("*" if (meta.get(s) or {}).get("held_out") else "")
                        for s in srcs], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("bytes per position (higher = more compression)")
    ax.set_title("Compression per source (* = held out)", fontsize=11)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_loss_to_loss(res, path):
    l2 = res.get("loss_to_loss")
    if not l2 or not any(l2["points"].values()):
        return None
    plt, ps = _style()
    ref, new = res["ref"], res["new"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for arch in (ref, new):
        pts = l2["points"].get(arch) or []
        if not pts:
            continue
        col = _color(ps, arch, ref, new)
        ax.scatter(*zip(*pts), s=20, color=col, zorder=3, label=LABEL.get(arch, arch))
        law = l2["laws"].get(arch)
        if law:
            xx = np.linspace(law["x_range"][0] * 0.97, law["x_range"][1] * 1.03, 50)
            ax.plot(xx, law["K"] * xx ** law["kappa"], color=col, lw=2, zorder=2)
    ax.set_xlabel("validation bpb")
    ax.set_ylabel("downstream bpb (paper7 average)")
    txt = []
    if l2.get("val_to_down_multiplier"):
        ci = l2.get("val_to_down_ci95")
        txt.append(f"val->down multiplier {l2['val_to_down_multiplier']:.2f}x" +
                   (f" [{ci['lo']:.2f}, {ci['hi']:.2f}]" if ci else ""))
    if l2.get("compute_to_down_multiplier"):
        txt.append(f"compute->down (product) {l2['compute_to_down_multiplier']:.2f}x")
    if txt:
        ax.text(0.02, 0.96, "\n".join(txt), transform=ax.transAxes, fontsize=9, va="top", color=ps.INK)
    ax.set_title("Loss-to-loss: validation -> downstream", fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_allocation(res, path):
    al = res.get("allocation")
    if not al or not any(len(v["points"]) >= 2 for v in al.values()):
        return None
    plt, ps = _style()
    ref, new = res["ref"], res["new"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for arch, v in al.items():
        col = _color(ps, arch, ref, new)
        C = [p["C"] for p in v["points"]]
        axes[0].plot(C, [p["N_opt"] for p in v["points"]], "o-", color=col, label=LABEL.get(arch, arch))
        if all(p["D_opt"] for p in v["points"]):
            axes[1].plot(C, [p["D_opt"] for p in v["points"]], "o-", color=col, label=LABEL.get(arch, arch))
        for ax, key in ((axes[0], "N_opt"), (axes[1], "D_opt")):
            if key in v:
                ps.direct_label(ax, C[-1], v["points"][-1][key] or 1, f"∝C^{v[key]['exponent']:.2f}")
    for ax, lab in ((axes[0], "N_opt (parameters)"), (axes[1], "D_opt (training bytes)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training FLOPs")
        ax.set_ylabel(lab)
    axes[0].legend(loc="upper left")
    axes[0].set_title("Compute-optimal allocation", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def make_figures(res, runs, figdir, prefix="ladder"):
    os.makedirs(figdir, exist_ok=True)
    out = {}
    jobs = {
        "loss_vs_compute": lambda p: plot_loss_vs_compute(res, runs, p),
        "isoflop": lambda p: plot_isoflop(res, p),
        "domain_multipliers": lambda p: plot_domain_multipliers(res, p),
        "code_multipliers": lambda p: plot_code_multipliers(res, p),
        "compression": lambda p: plot_compression(res, runs, p),
        "loss_to_loss": lambda p: plot_loss_to_loss(res, p),
        "allocation": lambda p: plot_allocation(res, p),
    }
    for name, fn in jobs.items():
        p = fn(os.path.join(figdir, f"{prefix}_{name}.png"))
        if p:
            out[name] = p
    return out


# ====================================================================================================== report
def report(res) -> str:
    ref, new = res["ref"], res["new"]
    L = []
    counts = ", ".join(f"{a}: {sum(r['arch'] == a for r in res['runs'])}" for a in (ref, new))
    budgets = ", ".join(f"{b:.3g}" for b in res["budgets"])
    L.append(f"runs: {len(res['runs'])}  ({counts})  budgets: {budgets}")
    for n in res["notes"]:
        L.append(f"NOTE: {n}")
    for m, r in res["metrics"].items():
        L.append("")
        L.append(f"== {m}   (law: {r['form'] or 'none'})")
        for a, f in r["frontier"].items():
            for p in f:
                L.append(f"   {a:6s} C={p['C']:.3g}  L*={p['L']:.4f}  N_opt={p['N_opt']:.3g}  "
                         f"[{p['status']}, {p['n_sizes']} sizes]")
        for a, law in r["laws"].items():
            ci = law.get("alpha_ci95")
            L.append(f"   {a:6s} L(C) = {law['E']:.4f} + {law['A']:.4f} (C/1e18)^-{law['alpha']:.4f}"
                     + (f"  alpha ci95 [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "") + f"  rmse {law['rmse']:.2e}")
        if r["multiplier"]:
            mm = r["multiplier"]
            ends = "  ".join(f"{lab} {mm[k]:.3g}" for k, lab in (("at_low_loss", "low-loss end"),
                                                                 ("at_high_loss", "high-loss end"))
                             if mm.get(k) is not None)
            L.append(f"   multiplier ({new} over {ref}): {_fmt_mult(mm)}  [{ends}]")
        if r.get("multiplier_interp") and r["multiplier_interp"].get("multiplier"):
            L.append(f"   multiplier (law-free interpolation): {r['multiplier_interp']['multiplier']:.3f}")
        for g in r["matched_budget_gap"]:
            L.append(f"   matched budget {g['budget']:.3g}: {ref} {g['L_ref']:.4f}  {new} {g['L_new']:.4f}  "
                     f"delta {g['delta']:+.4f} ({g['rel']:+.2%})")
        for n in r["notes"]:
            L.append(f"   note: {n}")
    l2 = res.get("loss_to_loss")
    if l2:
        L.append("")
        L.append(f"== loss-to-loss {l2['x']} -> {l2['y']}")
        for a, g in l2["laws"].items():
            if g:
                L.append(f"   {a:6s} y = {g['K']:.4f} x^{g['kappa']:.4f}   rel. rmse {g['rmse_rel']:.2%}  (n={g['n']})")
        for k in ("compute_to_val_multiplier", "val_to_down_multiplier", "compute_to_down_multiplier"):
            if l2.get(k):
                L.append(f"   {k}: {l2[k]:.3f}")
        for n in l2["notes"]:
            L.append(f"   note: {n}")
    al = res.get("allocation") or {}
    for a, v in al.items():
        if "N_opt" in v:
            s = f"   {a:6s} N_opt ∝ C^{v['N_opt']['exponent']:.3f}"
            if "D_opt" in v:
                s += f"   D_opt ∝ C^{v['D_opt']['exponent']:.3f}"
            L.append(s)
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", action="append", default=[], help="[ARCH=]GLOB of run dirs (repeatable)")
    ap.add_argument("--records", default=None, help="JSON list of Run dicts instead of run dirs")
    ap.add_argument("--ref", default="bpe")
    ap.add_argument("--new", default="hnet")
    ap.add_argument("--metrics", default=None, help="comma list (default: every metric both archs have)")
    ap.add_argument("--form", default="auto", choices=["auto", "free", "shared", "pure", "fixed"])
    ap.add_argument("--E", type=float, default=None, help="irreducible loss for --form fixed")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot", default="auto", choices=["auto", "seeds", "residual"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-tol-dex", type=float, default=0.15)
    ap.add_argument("--param-key", default="params", choices=["params", "params_nonembed"])
    ap.add_argument("--include-unfinished", action="store_true")
    ap.add_argument("--l2l-y", default="down:avg", help="downstream metric of the loss-to-loss law")
    ap.add_argument("--out", default=None, help="results JSON")
    ap.add_argument("--figdir", default=os.path.join(ROOT, "logbook", "figures"))
    ap.add_argument("--prefix", default="ladder")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)
    if a.records:
        with open(a.records) as f:
            runs = [SC.Run.from_dict(d) for d in json.load(f)]
    else:
        specs = []
        for s in a.runs:
            arch, sep, pat = s.partition("=")
            specs.append((arch, pat) if sep and "/" not in arch and "*" not in arch else (None, s))
        runs = SC.load_runs(specs, include_unfinished=a.include_unfinished, param_key=a.param_key)
    if not runs:
        print("no runs found (need run dirs with metrics.jsonl, done.json and a parsable config)", file=sys.stderr)
        return 2
    archs = {r.arch for r in runs}
    missing = [x for x in (a.ref, a.new) if x not in archs]
    if missing:
        print(f"no runs for architecture(s) {missing}; found {sorted(archs)} (use ARCH=GLOB or --ref/--new)",
              file=sys.stderr)
        return 2
    res = SC.analyze_ladder(runs, a.ref, a.new, a.metrics.split(",") if a.metrics else None, a.form, a.E,
                            a.n_boot, a.seed, a.boot, a.budget_tol_dex, a.l2l_y)
    print(report(res))
    if not a.no_plots:
        figs = make_figures(res, runs, a.figdir, a.prefix)
        res["figures"] = figs
        for k, p in figs.items():
            print(f"[figure] {k}: {p}")
    res["compression"] = compression_table(res, runs)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(SC.public(res), f, indent=1, default=float)
        print(f"-> {a.out}", file=sys.stderr)
    return res


if __name__ == "__main__":
    r = main()
    sys.exit(r if isinstance(r, int) else 0)
