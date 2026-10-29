"""Training visibility report: TL;DR + tables + plots for every sweep mirrored under a live directory.

    python logbook/training_report.py --live <dir with sweep mirrors> --out logbook/figures/training_report.html

Per sweep (one sub-directory per sweep, one run directory per seed, each with metrics.jsonl / eval.jsonl /
done.json):
  * a table of arms: seeds, finished, latest step (of the sweep's known total), val bpb mean ± sd, zero-main probe,
    train bpb (EMA), throughput, and an alert column (collapse signatures from CLAUDE.md rule 11);
  * validation bpb vs step for every arm (thin lines per seed) with the reference levels dashed;
  * for H-Net arms, mechanism panels: compression, router margin, raw-score std, hierarchy vs bypass rms, grad norm;
  * for LR sweeps (arm names ``<arch>__mlr<lr>``), val bpb vs LR per architecture.
The TL;DR at the top is generated from the same numbers: best arm per sweep vs the reference, what finished, what is
in flight, and every alert. The page is self-contained (PNGs embedded as base64). ``--md`` also writes a markdown
version of the TL;DR and tables for the chat.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import glob
import html
import io
import json
import math
import os
import re
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# tiny-scale references (50M token-eq, EXP-0xx sweeps only); ladder sweeps compare arms with each other
REFERENCES = {"BPE (EXP-006, 3 seeds)": 1.634, "H-Net + whitespace prior (EXP-024)": 1.635}


def _refs(sweep):
    return REFERENCES if sweep.startswith("exp") else {}
MECH = [("p_margin/s0", "router margin |p-0.5|"), ("p_mid/s0", "undecided fraction (p in 0.4-0.6)"),
        ("score_std/s0", "raw score std (scale drift)"), ("rms", "hierarchy (solid) vs bypass (dashed) rms"),
        ("bnd_on_ws/s0", "boundaries on whitespace (basin)"), ("grad_norm", "grad norm")]


def _jsonl(path):
    out = []
    if os.path.exists(path):
        for line in open(path):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _ema(xs, a=0.3):
    out, m = [], None
    for x in xs:
        m = x if m is None else (1 - a) * m + a * x
        out.append(m)
    return out


def load_sweep(d):
    """{arm: [run dict]} for one sweep directory."""
    arms = defaultdict(list)
    for rd in sorted(glob.glob(f"{d}/*_s[0-9]*")):
        if not os.path.isdir(rd):
            continue
        name = os.path.basename(rd)
        arm, seed = name.rsplit("_s", 1)
        m = _jsonl(f"{rd}/metrics.jsonl")
        e = [r for r in _jsonl(f"{rd}/eval.jsonl") if "val/bpb" in r]
        arms[arm].append({"seed": seed, "metrics": m, "evals": e, "done": os.path.exists(f"{rd}/done.json"),
                          "step": max([r["step"] for r in m] + [r["step"] for r in e] + [0]),
                          "hnet": any(k.startswith("compression/") for k in (m[-1] if m else {}))})
    return arms


def alerts(run):
    """Collapse signatures (CLAUDE.md rule 11) in the latest metrics of one run."""
    out = []
    m = run["metrics"]
    if not m:
        return out
    last = m[-1]
    for k, v in last.items():
        if k.startswith("compression/") and isinstance(v, (int, float)) and (v > 64 or v < 1.5):
            out.append(f"{k}={v:.0f}")
        if k.startswith("score_std/") and isinstance(v, (int, float)) and v < 0.05:
            out.append(f"{k}→0")
        if (k.startswith("tau/") and isinstance(v, (int, float)) and v > 0.999
                and abs(last.get(k.replace("tau", "compression"), 0) - 1 / max(last.get(k.replace("tau", "F"), 1e-9), 1e-9)) > 0.5):
            out.append(f"{k}=1 with compression off target")  # under rank decisions tau drifts harmlessly
        if k.startswith("p_margin/") and isinstance(v, (int, float)) and v < 0.08:
            out.append(f"{k}={v:.2f} (undecided)")
    w = last.get("bnd_on_ws/s0")
    if isinstance(w, (int, float)) and m[-1]["step"] >= 300 and w < 0.3:
        out.append(f"suffix basin (bnd_on_ws {w:.2f})")  # word basin ~0.9; decided by step ~300
    gn = [r.get("grad_norm") for r in m if isinstance(r.get("grad_norm"), (int, float))]
    if len(gn) >= 4 and gn[-1] > 50 * statistics.median(gn[:-1]) and gn[-1] > 10:
        out.append(f"grad_norm×{gn[-1] / statistics.median(gn[:-1]):.0f}")
    bpb = [r.get("bpb") for r in m if isinstance(r.get("bpb"), (int, float))]
    if bpb and (math.isnan(bpb[-1]) or bpb[-1] > 6):
        out.append("train bpb diverged")
    return out


def arm_summary(arm, runs, total):
    """One table row."""
    vals = [r["evals"][-1]["val/bpb"] for r in runs if r["evals"]]
    steps = [r["evals"][-1]["step"] for r in runs if r["evals"]]
    # per-seed values that are not at the sweep's final step are interim: shown with their step, kept out of the mean
    final = [r["evals"][-1]["val/bpb"] for r in runs if r["evals"] and (total is None or r["evals"][-1]["step"] >= total)]
    seed_txt = " ".join(f"{r['evals'][-1]['val/bpb']:.3f}" + ("" if total is None or r["evals"][-1]["step"] >= total
                        else f"@{r['evals'][-1]['step']}") for r in runs if r["evals"])
    zm = [r["evals"][-1].get("val/zero_main_s0_bpb") for r in runs if r["evals"]]
    zm = [z for z in zm if isinstance(z, (int, float))]
    tb = [_ema([x["bpb"] for x in r["metrics"]])[-1] for r in runs if r["metrics"]]
    bps = [r["metrics"][-1].get("bytes_per_sec") for r in runs if r["metrics"]]
    bps = [b for b in bps if isinstance(b, (int, float))]
    al = sorted({a for r in runs for a in alerts(r)})
    row = {
        "arm": arm, "seeds": len(runs), "done": sum(r["done"] for r in runs),
        "step": max(r["step"] for r in runs), "total": total,
        "eval_step": max(steps) if steps else None,
        "val": statistics.mean(final) if final else (statistics.mean(vals) if vals else None),
        "val_sd": statistics.stdev(final) if len(final) > 1 else None,
        "val_final": bool(final),
        "val_seeds": vals,
        "seed_txt": seed_txt,
        "zero_main": statistics.mean(zm) if zm else None,
        "train": statistics.mean(tb) if tb else None,
        "bytes_per_sec": statistics.mean(bps) if bps else None,
        "alerts": al,
    }
    return row


# layout: one reading column; tables scroll inside their own box; plots are full width of the column
STYLE = """
:root{--bg:#f7f5f0;--fg:#1f2320;--muted:#6a6f68;--line:#d9d5cb;--panel:#ffffff;--accent:#3d6b5e;--warn-bg:#f3e3c6;
 --warn-fg:#6b4a08;--ok-bg:#dcebe2;--ok-fg:#1f5d44;--run-bg:#dde6f2;--run-fg:#234a7a;
 --sans:'IBM Plex Sans',system-ui,sans-serif;--mono:'IBM Plex Mono',ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15181a;--fg:#e4e6e2;--muted:#9aa09b;
 --line:#30363a;--panel:#1d2124;--accent:#8fc5b3;--warn-bg:#4a3812;--warn-fg:#f1d79a;--ok-bg:#1d3b2d;--ok-fg:#b7e2cc;
 --run-bg:#1f3350;--run-fg:#b9d1f0;color-scheme:dark}}
:root[data-theme=dark]{--bg:#15181a;--fg:#e4e6e2;--muted:#9aa09b;--line:#30363a;--panel:#1d2124;--accent:#8fc5b3;
 --warn-bg:#4a3812;--warn-fg:#f1d79a;--ok-bg:#1d3b2d;--ok-fg:#b7e2cc;--run-bg:#1f3350;--run-fg:#b9d1f0;color-scheme:dark}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.45;
 padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:24px}
main{max-width:1100px;margin:0 auto;display:flex;flex-direction:column;gap:28px}
h1{font-size:1.6rem;margin:0;text-wrap:balance}h2{font-size:1.15rem;margin:0 0 10px;border-bottom:1px solid var(--line);padding-bottom:4px}
p.meta{color:var(--muted);margin:4px 0 0;font-size:.9rem}ul{margin:0;padding-left:1.2em}li{margin:4px 0}li li{color:var(--muted)}
code{font-family:var(--mono);font-size:.9em}.scroll{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:6px}
table{border-collapse:collapse;font-size:.85rem;min-width:100%;font-variant-numeric:tabular-nums}
th{text-align:left;color:var(--muted);font-weight:600;font-size:.75rem;letter-spacing:.04em;text-transform:uppercase}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);white-space:nowrap}tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:1px 7px;border-radius:999px;font-size:.75rem;margin-right:4px}
.pill.done{background:var(--ok-bg);color:var(--ok-fg)}.pill.run{background:var(--run-bg);color:var(--run-fg)}
.pill.wait{background:var(--line);color:var(--muted)}.pill.warn{background:var(--warn-bg);color:var(--warn-fg)}
img{max-width:100%;display:block;margin-top:12px;border-radius:4px;background:#fff}section{min-width:0}
"""


def _inline(b: str) -> str:
    s = html.escape(b)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", s)


def _fmt(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def table_md(rows):
    hdr = "| arm | seeds done | step | val bpb (mean ± sd) | per seed | zero-main | train bpb | kB/s | alerts |"
    lines = [hdr, "|" + "---|" * 9]
    for r in rows:
        tot = f"/{r['total']}" if r["total"] else ""
        sd = f" ± {r['val_sd']:.3f}" if r["val_sd"] is not None else ""
        seeds = r["seed_txt"]
        kbs = _fmt(r["bytes_per_sec"] / 1e3 if r["bytes_per_sec"] else None, 0)
        lines.append(f"| {r['arm']} | {r['done']}/{r['seeds']} | {r['step']}{tot} | {_fmt(r['val'])}{sd}"
                     f"{' @' + str(r['eval_step']) if r['eval_step'] and r['eval_step'] != r['step'] else ''} | {seeds}"
                     f" | {_fmt(r['zero_main'])} | {_fmt(r['train'])} | {kbs} | {', '.join(r['alerts']) or '–'} |")
    return "\n".join(lines)


def table_html(rows):
    cols = ["arm", "seeds done", "step", "val bpb", "per seed", "zero-main", "train bpb", "kB/s", "alerts"]
    h = ["<div class='scroll'><table><thead><tr>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr></thead><tbody>"]
    for r in rows:
        tot = f"/{r['total']}" if r["total"] else ""
        sd = f" ± {r['val_sd']:.3f}" if r["val_sd"] is not None else ""
        at = f" <small>@{r['eval_step']}</small>" if r["eval_step"] and r["eval_step"] != r["step"] else ""
        seeds = r["seed_txt"]
        kbs = _fmt(r["bytes_per_sec"] / 1e3 if r["bytes_per_sec"] else None, 0)
        state = "done" if r["done"] == r["seeds"] else ("run" if r["step"] else "wait")
        al = "".join(f"<span class='pill warn'>{html.escape(a)}</span>" for a in r["alerts"]) or "–"
        h.append(f"<tr><td><code>{html.escape(r['arm'])}</code></td><td><span class='pill {state}'>{r['done']}/{r['seeds']}"
                 f"</span></td><td>{r['step']}{tot}</td><td><b>{_fmt(r['val'])}</b>{sd}{at}</td><td>{seeds}</td>"
                 f"<td>{_fmt(r['zero_main'])}</td><td>{_fmt(r['train'])}</td><td>{kbs}</td><td>{al}</td></tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def plot_val(sweep, arms):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (arm, runs) in enumerate(sorted(arms.items())):
        c = colors[i % len(colors)]
        agg = defaultdict(list)
        for r in runs:
            if r["evals"]:
                xs = [e["step"] for e in r["evals"]]
                ys = [e["val/bpb"] for e in r["evals"]]
                ax.plot(xs, ys, color=c, lw=0.7, alpha=0.4)
                for x, y in zip(xs, ys):
                    agg[x].append(y)
        if agg:
            xs = sorted(agg)
            ax.plot(xs, [statistics.mean(agg[x]) for x in xs], color=c, lw=2, label=f"{arm} ({len(runs)} seeds)")
    for j, (name, v) in enumerate(_refs(sweep).items()):
        ax.axhline(v, ls="--", lw=1, color="gray", alpha=0.8)
        x0, x1 = ax.get_xlim()
        ax.text(x0 if j % 2 == 0 else x1, v, f" {name} {v:.3f} ", va="bottom" if j % 2 == 0 else "top",
                ha="left" if j % 2 == 0 else "right", fontsize=7, color="gray")
    ax.set_xlabel("step")
    ax.set_ylabel("val bpb")
    ax.set_title(f"{sweep}: validation bpb")
    lo = min(list(_refs(sweep).values()) + [e["val/bpb"] for runs in arms.values() for r in runs for e in r["evals"]])
    ax.set_ylim(lo - 0.05, min(3.0, lo + 1.2))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return fig_to_b64(fig)


def plot_mech(sweep, arms):
    hn = {a: rs for a, rs in arms.items() if any(r["hnet"] for r in rs)}
    if not hn:
        return None
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    axes = axes.ravel()
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (arm, runs) in enumerate(sorted(hn.items())):
        c = colors[i % len(colors)]
        for r in runs:
            m = r["metrics"]
            if not m:
                continue
            xs = [x["step"] for x in m]
            for ax, (key, _) in zip(axes, MECH):
                if key == "rms":
                    ax.plot(xs, [x.get("dechunk_rms/s0", float("nan")) for x in m], color=c, lw=1)
                    ax.plot(xs, [x.get("residual_rms/s0", float("nan")) for x in m], color=c, lw=1, ls="--")
                else:
                    ax.plot(xs, [x.get(key, float("nan")) for x in m], color=c, lw=1,
                            label=arm if r is runs[0] and key == MECH[0][0] else None)
    for ax, (key, title) in zip(axes, MECH):
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
        if key == "grad_norm":
            ax.set_yscale("log")
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{sweep}: router / hierarchy mechanism (stage 0)", fontsize=10)
    fig.tight_layout()
    return fig_to_b64(fig)


def plot_lr(sweep, rows):
    pts = defaultdict(list)
    for r in rows:
        mm = re.match(r"(.*)__mlr([0-9.e-]+)$", r["arm"])
        if mm and r["val"] is not None:
            pts[mm.group(1)].append((float(mm.group(2)), r["val"]))
    if not pts:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for arch, p in sorted(pts.items()):
        p.sort()
        ax.plot([x for x, _ in p], [y for _, y in p], "o-", label=arch)
    ax.set_xscale("log")
    ax.set_xlabel("Muon LR")
    ax.set_ylabel("val bpb")
    ax.set_title(f"{sweep}: LR sweep")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig_to_b64(fig)


def tldr(all_rows):
    """Bullets from the numbers."""
    out = []
    for sweep, rows in all_rows.items():
        scored = [r for r in rows if r["val"] is not None]
        fin = sum(r["done"] for r in rows)
        tot = sum(r["seeds"] for r in rows)
        if scored:
            best = min(scored, key=lambda r: r["val"])
            refs = _refs(sweep)
            if refs:
                ref = min(refs.values())
                delta = best["val"] - ref
                tag = "matches" if abs(delta) < 0.005 else ("beats" if delta < 0 else "trails")
                cmp_ = f"{tag} the BPE reference {ref:.3f} by {abs(delta):.3f}"
            else:
                second = sorted(scored, key=lambda r: r["val"])[1:2]
                cmp_ = f"next `{second[0]['arm']}` {second[0]['val']:.3f}" if second else "single arm"
            out.append(f"**{sweep}**: best arm `{best['arm']}` at {best['val']:.3f} bpb (step {best['eval_step']}), "
                       f"{cmp_}; {fin}/{tot} runs finished.")
        else:
            out.append(f"**{sweep}**: no evaluations yet; {fin}/{tot} runs finished.")
        for r in rows:
            if r["alerts"]:
                out.append(f"  - alert `{r['arm']}`: {', '.join(r['alerts'])}")
            if r["val_sd"] is not None and r["val_sd"] > 0.03:
                out.append(f"  - seeds of `{r['arm']}` disagree (sd {r['val_sd']:.3f}): a bifurcation, not noise "
                           f"(CLAUDE.md rule 7); per seed {r['seed_txt']}")
    return out


def build(live, out_html, out_md=None, title="Training report"):
    sweeps = sorted(d for d in os.listdir(live) if os.path.isdir(f"{live}/{d}"))
    sections, all_rows = [], {}
    for sw in sweeps:
        arms = load_sweep(f"{live}/{sw}")
        if not arms:
            continue
        done_steps = [r["step"] for rs in arms.values() for r in rs if r["done"]]
        total = max(done_steps) if done_steps else None
        rows = [arm_summary(a, rs, total) for a, rs in sorted(arms.items())]
        all_rows[sw] = rows
        imgs = [plot_val(sw, arms)]
        for f in (plot_mech(sw, arms), plot_lr(sw, rows)):
            if f:
                imgs.append(f)
        sections.append((sw, rows, imgs))
    bullets = tldr(all_rows)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts = [f"<title>{html.escape(title)}</title><meta name='claude:viewport-fit-sides' content='cover'><style>"
             + STYLE + "</style><main>"
             f"<h1>{html.escape(title)}</h1><p class='meta'>generated {now} · val bpb on 1 MB of held-out FineWeb-Edu;"
             " references dashed; thin lines are single seeds</p><section><h2>TL;DR</h2><ul>"]
    for b in bullets:
        if b.startswith("  - "):
            parts.append(f"<ul><li>{_inline(b[4:])}</li></ul>")
        else:
            parts.append(f"<li>{_inline(b)}</li>")
    parts.append("</ul></section>")
    for sw, rows, imgs in sections:
        parts.append(f"<section><h2>{html.escape(sw)}</h2>{table_html(rows)}")
        parts += [f"<img src='data:image/png;base64,{b}' alt='{html.escape(sw)} plot'>" for b in imgs]
        parts.append("</section>")
    parts.append("</main>")
    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    open(out_html, "w").write("\n".join(parts))
    md = [f"# {title} ({now})", "", "## TL;DR", *[b if b.startswith("  - ") else f"- {b}" for b in bullets], ""]
    for sw, rows, _ in sections:
        md += [f"## {sw}", table_md(rows), ""]
    if out_md:
        open(out_md, "w").write("\n".join(md))
    return "\n".join(md)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", default="logbook/figures/training_report.html")
    ap.add_argument("--md", default=None)
    ap.add_argument("--title", default="H-Net training report")
    a = ap.parse_args(argv)
    print(build(a.live, a.out, a.md, a.title))


if __name__ == "__main__":
    main()
