"""EXP-002 results table (thin wrapper around scripts/search/report.py; run after exp002_progress.py pulled the runs).

    python logbook/exp002_summary.py [--runs DIR]  ->  logbook/tables/exp002_summary.md

Per variant: final val bpb (mean ± 95% t-interval over seeds) and Δ vs BPE (paired over seeds), data multiplier,
paper7 soft bpb and acc_norm, realized bytes per position, stage-0/1 chunk statistics next to the blog's improved
H-Net targets (EXP-001). Unfinished runs are counted but not tabulated.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "search")]

from report import build_report  # noqa: E402
from results import load_sweep  # noqa: E402
from spec import Spec  # noqa: E402

SPEC = ROOT / "configs" / "search" / "exp002_tiny2m.yaml"
LOCAL = ROOT / "runs" / "gcp" / "exp002_tiny2m"
OUT = ROOT / "logbook" / "tables" / "exp002_summary.md"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(LOCAL))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    md, _, _ = build_report(load_sweep(Spec.load(str(SPEC)), a.runs))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
