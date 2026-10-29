"""Entry point of the interp pass: every analysis behind logbook/interp/REPORT.md.

    OMP_NUM_THREADS=4 PYTHONPATH=. python scripts/interp/run_all.py [--quick] [--skip-ckpt]

Order: metrics-only analyses (phases, predictability; seconds), checkpoint analyses (analyze_ckpt over the set in
run_ckpts.py; ~1.5-4 min each on 4 CPU threads, skipped when the JSON exists), dense early checkpoints
(early_dynamics), cross-seed CKA, token-aligned BPE comparison, figures/summary from the checkpoint JSONs.
``--quick``: 4 checkpoints with 8 documents, a subset of runs for the metrics analyses (a smoke test); its outputs go
to logbook/interp/quick/ so the full results are never overwritten.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def step(args):
    t = time.time()
    print("==>", " ".join(args), flush=True)
    r = subprocess.run([sys.executable, *args], cwd=ROOT)
    print(f"    rc={r.returncode} ({time.time() - t:.0f}s)", flush=True)
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-ckpt", action="store_true")
    a = ap.parse_args(argv)
    q = ["--quick"] if a.quick else []
    if a.quick:  # never overwrite the full outputs with smoke-test ones
        os.environ["INTERP_OUT"] = str(ROOT / "logbook" / "interp" / "quick")
    step([str(HERE / "phases.py"), *q])
    step([str(HERE / "predictability.py")])
    step([str(HERE / "seed_rankings.py")])
    if not a.skip_ckpt:
        step([str(HERE / "init_basin.py"), "--docs", "4" if a.quick else "16"])
        step([str(HERE / "run_ckpts.py"), *q])
        step([str(HERE / "early_dynamics.py"), *q])
        step([str(HERE / "cka.py"), *q])
        step([str(HERE / "bpe_compare.py"), *q])
        step([str(HERE / "vocab_ablation.py"), *q])
    step([str(HERE / "figures_ckpt.py")])


if __name__ == "__main__":
    main()
