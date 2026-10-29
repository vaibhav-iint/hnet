"""Run analyze_ckpt.py over the checkpoint set of the interp pass (skips outputs that already exist).

    python scripts/interp/run_ckpts.py [--quick] [--only SUBSTR] [--docs 32]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, ROOT  # noqa: E402

L16 = "runs/gcp/hnet-ladder-1e16/search/ladder/1e16"
CKPTS = [
    # 1-stage word-scale line: word vs suffix seeds over training, warm start, no-vocabulary baseline
    ("exp023_vp_s0__0500", "runs/gcp/exp023_word_vocab_trimmed/vp_s0/ckpt/step_0000500"),
    ("exp023_vp_s1__0500", "runs/gcp/exp023_word_vocab_trimmed/vp_s1/ckpt/step_0000500"),
    ("exp023_vp_s0__1000", "runs/gcp/exp023_word_vocab_trimmed/vp_s0/ckpt/step_0001000"),
    ("exp023_vp_s1__1000", "runs/gcp/exp023_word_vocab_trimmed/vp_s1/ckpt/step_0001000"),
    ("exp023_vp_s1__final", "runs/gcp/exp023_word_vocab_trimmed/vp_s1"),
    ("exp018_h1_vocab_pred_s0__final", "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s0"),
    ("exp018_h1_vocab_pred_s1__final", "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s1"),
    ("exp024_prior_s0__final", "runs/gcp/exp024_adoption_curriculum/prior_s0"),
    ("exp023_vp_ng_s2__final", "runs/gcp/exp023_word_vocab_trimmed/vp_ng_s2"),
    ("exp018_h1_quant_s0__final", "runs/gcp/exp018_vocab_shapes/h1_quant_s0"),
    # ladder 1e16 (Muon lr 0.02): is the bypass-dominated hierarchy dead?
    ("ladder_L03_hnet_s0__final", f"{L16}/L03/hnet_s0"),
    ("ladder_L03_hnet_ws_s0__final", f"{L16}/L03/hnet_ws_s0"),
    ("ladder_L04_hnet_s0__final", f"{L16}/L04/hnet_s0"),
    ("ladder_L04_hnet_ws_s0__final", f"{L16}/L04/hnet_ws_s0"),
    # 2-stage paper-era models
    ("exp002_paper_s0__0500", "runs/gcp/exp002_tiny2m/paper_s0/ckpt/step_0000500"),
    ("exp002_paper_s0__1000", "runs/gcp/exp002_tiny2m/paper_s0/ckpt/step_0001000"),
    ("exp002_paper_s0__final", "runs/gcp/exp002_tiny2m/paper_s0"),
    ("exp002_end_mlp_s1__0500", "runs/gcp/exp002_tiny2m/end_mlp_s1/ckpt/step_0000500"),
    ("exp002_end_mlp_s1__1000", "runs/gcp/exp002_tiny2m/end_mlp_s1/ckpt/step_0001000"),
    ("exp002_end_mlp_s1__final", "runs/gcp/exp002_tiny2m/end_mlp_s1"),
    ("exp002_end_cosine_shift_s0__final", "runs/gcp/exp002_tiny2m/end_cosine_shift_s0"),
    ("exp002_paper_nosmooth_s1__final", "runs/gcp/exp002_tiny2m/paper_nosmooth_s1"),
    ("exp008_base_s0__final", "runs/gcp/exp008_hierarchy_use/base_s0"),
    ("exp013_lr6_nomult_s0__final", "runs/gcp/exp013_lr_allocation/lr6_nomult_s0"),
    ("exp006_hnet1_s0__final", "runs/gcp/exp006_close_gap_tiny/hnet1_s0"),
    ("exp006_hnet1_ng_s0__1500", "runs/gcp/exp006_close_gap_tiny/hnet1_ng_s0/ckpt/step_0001500"),
]
QUICK = ["exp023_vp_s0__1000", "exp023_vp_s1__1000", "exp024_prior_s0__final", "ladder_L04_hnet_s0__final"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--docs", type=int, default=32)
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th checkpoint starting at i")
    a = ap.parse_args(argv)
    i, k = map(int, a.shard.split("/"))
    todo = [(n, p) for n, p in CKPTS if (not a.quick or n in QUICK) and (a.only is None or a.only in n)][i::k]
    for name, path in todo:
        out = DATA / "ckpt" / f"{name}.json"
        if out.exists() or not (ROOT / path).exists():
            continue
        cmd = [sys.executable, str(Path(__file__).parent / "analyze_ckpt.py"), path, "--docs",
               str(8 if a.quick else a.docs), "--attn-docs", "2" if a.quick else "4", "--out", f"{name}.json"]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        print(name, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-800:], flush=True)


if __name__ == "__main__":
    main()
