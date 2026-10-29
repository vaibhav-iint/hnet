"""EXP-002 progress figure (thin wrapper around scripts/search/report.py; re-runnable while training).

    python logbook/exp002_progress.py [--no-pull]

Pulls the sweep's run dirs from GCS (no checkpoints) into runs/gcp/exp002_tiny2m and writes
logbook/figures/exp002_progress.png: rows = training bpb (centred 5-point mean), validation bpb, bytes per stage
position; columns = the spec's report groups (boundary routers; ablations of the paper router), BPE in gray.
Mean over seeds with a min-max seed band. The full report: python scripts/search/report.py
configs/search/exp002_tiny2m.yaml --runs runs/gcp/exp002_tiny2m.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "search"), str(ROOT / "logbook")]

import plotstyle  # noqa: E402
from report import best_points, fig_overview  # noqa: E402
from results import load_sweep, pull  # noqa: E402
from spec import Spec  # noqa: E402

SPEC = ROOT / "configs" / "search" / "exp002_tiny2m.yaml"
GCS = "gs://neat-dispatch-501718-a5-hnet/runs/exp002_tiny2m"
LOCAL = ROOT / "runs" / "gcp" / "exp002_tiny2m"
OUT = ROOT / "logbook" / "figures" / "exp002_progress.png"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--runs", default=str(LOCAL))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    if not a.no_pull:
        pull(GCS, Path(a.runs))
    sweep = load_sweep(Spec.load(str(SPEC)), a.runs)
    if not any(r.metrics for r in sweep.runs):
        raise SystemExit(f"no runs under {a.runs} yet")
    plotstyle.apply()
    out = fig_overview(sweep, Path(a.out), best_points(sweep),
                       "EXP-002 · tiny (~2M-param, FLOPs-matched) architecture search")
    c = sweep.counts()
    print(f"wrote {out} ({sum(c.get(k, 0) for k in ('running', 'done', 'diverged'))} runs logging, "
          f"{c.get('done', 0)} done)")


if __name__ == "__main__":
    main()
