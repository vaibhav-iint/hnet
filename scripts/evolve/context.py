#!/usr/bin/env python
"""Write the proposer context for an evolve generation (docs/EVOLVE.md step 1).

    python scripts/evolve/context.py --gen 1 [--notes logbook/evolve/notes_g0.md] [--out logbook/evolve/context_g1.md]

Contents: the benchmark and guardrails (pointer), the population ranked by fitness with per-seed values and
mechanism metrics, the orchestrator's notes on why candidates won or lost, and the documents every proposer must read.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import ROOT, bpe_ref, load_db  # noqa: E402

READ = [
    ("docs/EVOLVE.md", "the loop, the benchmark, the fairness contract, the candidate format"),
    ("CLAUDE.md", "working rules; the bug-finding meta (gradient paths, step-0 audit, lazy optimizer)"),
    ("docs/BLOG_GAP_ANALYSIS.md", "what the blog's recipe has that ours lacks; the blog's own per-run data"),
    ("docs/IDEAS.md", "ideas already considered, with status and falsifiers"),
    ("docs/IMPL_AUDIT.md", "implementation audit vs the official code and the paper (bugs found)"),
    ("logbook/interp/REPORT.md", "mechanistic interpretability of every experiment so far"),
    ("logbook/EXPERIMENTS.md", "results of EXP-002..028 (tiny scale)"),
    ("hnet/config.py", "DCConfig: every existing router / decision / hierarchy option, documented"),
]


def fmt_mech(m):
    if not m:
        return "–"
    parts = []
    for k in ("bypass_ratio", "p_margin", "p_mid", "score_std"):
        if k in m:
            parts.append(f"{k} {m[k]:.2f}")
    return ", ".join(parts)


def render(gen, notes):
    db = load_db()
    bpe = bpe_ref(db)
    lines = [f"# Evolve generation {gen}: proposer context", "",
             f"Benchmark: L03 at 1e16 FLOPs on mix_v1 (docs/EVOLVE.md). BPE reference val bpb: **{bpe:.4f}**. "
             "Fitness = H-Net val bpb − BPE (lower is better; negative beats BPE). Target: beat BPE by a wide margin.", "",
             "## Population (ranked)", "",
             "| id | gen | track | parents | fitness | mean ± sd | seeds | mechanism at end (seed 0) | hypothesis |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = sorted(db, key=lambda e: (e.get("fitness") is None, e.get("fitness") or 0))
    for e in rows:
        m0 = next(iter((e.get("mechanism") or {}).values()), {})
        sd = f" ± {e['sd']:.3f}" if e.get("sd") else ""
        fit = f"{e['fitness']:+.4f}" if e.get("fitness") is not None else "ref"
        hyp = (e.get("hypothesis") or e.get("note") or "").replace("\n", " ")[:160]
        lines.append(f"| {e['id']} | {e['gen']} | {e.get('track') or '–'} | {','.join(e.get('parents') or []) or '–'} | "
                     f"{fit} | {e['mean']:.4f}{sd} | {e['seeds']} | {fmt_mech(m0)} | {hyp} |")
    lines += ["", "## Orchestrator notes (why things won or lost)", ""]
    lines += [open(notes).read() if notes and os.path.exists(notes) else "(none yet)"]
    lines += ["", "## Read before proposing", ""]
    lines += [f"- `{p}`: {why}" + ("" if os.path.exists(os.path.join(ROOT, p)) else " (not written yet)") for p, why in READ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--notes", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(ROOT, f"logbook/evolve/context_g{a.gen}.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(render(a.gen, a.notes))
    print(out)


if __name__ == "__main__":
    sys.exit(main())
