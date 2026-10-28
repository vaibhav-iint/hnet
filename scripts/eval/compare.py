"""Paired comparison of two models from their ``--items`` jsonl files, and data multiplier vs a reference curve.

    # per-task paired bootstrap (same items), e.g. H-Net vs BPE at equal tokens
    python scripts/eval/compare.py paired a.items.jsonl b.items.jsonl [--metric acc_norm_char]

    # tokens a baseline curve needs to match a score (e.g. SmolLM3 stage-1 six-task average from Cartesia's Fig. 7)
    python scripts/eval/compare.py multiplier --ref configs/reference/blog1_fig7_smollm3.csv --column average6 \
        --phase 0 --tokens 94e9 --value 55.7 [--method quadratic_log]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from hnet.eval.stats import data_multiplier, paired_bootstrap  # noqa: E402


def _load(path):
    d = defaultdict(dict)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            d[(r["suite"], r["task"])][str(r["doc_id"])] = r
    return d


def paired(a_path, b_path, metric, n_boot):
    A, B = _load(a_path), _load(b_path)
    print(f"| suite/task | n | A | B | A-B | 95% CI | p |\n|---|---|---|---|---|---|---|")
    for key in sorted(set(A) & set(B)):
        ids = sorted(set(A[key]) & set(B[key]))
        m = metric if metric in A[key][ids[0]] else "acc"
        a = np.array([A[key][i][m] for i in ids], dtype=float)
        b = np.array([B[key][i][m] for i in ids], dtype=float)
        groups = [A[key][i].get("group") for i in ids]
        g = groups if any(x is not None for x in groups) else None
        d = paired_bootstrap(a, b, n_boot=n_boot, groups=g)
        print(f"| {key[0]}/{key[1]} ({m}) | {len(ids)} | {100 * a.mean():.1f} | {100 * b.mean():.1f} | "
              f"{100 * d['value']:+.2f} | [{100 * d['lo']:+.2f}, {100 * d['hi']:+.2f}] | {d['p_value']:.3f} |")


def multiplier(ref, column, phase, tokens, value, method, n_boot):
    with open(ref) as f:
        rows = [r for r in csv.DictReader(f) if phase is None or r.get("phase") in (None, str(phase))]
    t = np.array([float(r["tokens_T"]) * 1e12 for r in rows])
    y = np.array([float(r[column]) for r in rows])
    m = data_multiplier(t, y, tokens, value, method=method, increasing=True, n_boot=n_boot)
    m.pop("curve", None)
    print(json.dumps(m, indent=1, default=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paired")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--metric", default="acc_norm_char")
    p.add_argument("--n-boot", type=int, default=2000)
    q = sub.add_parser("multiplier")
    q.add_argument("--ref", required=True)
    q.add_argument("--column", default="average6")
    q.add_argument("--phase", type=int, default=None, help="stage filter (0 = SmolLM3 stage 1)")
    q.add_argument("--tokens", type=float, required=True)
    q.add_argument("--value", type=float, required=True)
    q.add_argument("--method", default="quadratic_log", choices=["powerlaw", "quadratic_log", "isotonic"])
    q.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    if a.cmd == "paired":
        paired(a.a, a.b, a.metric, a.n_boot)
    else:
        multiplier(a.ref, a.column, a.phase, a.tokens, a.value, a.method, a.n_boot)


if __name__ == "__main__":
    main()
