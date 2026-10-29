#!/bin/bash
# Reproduce the EXP-002 "illegal memory access" in mamba_ssm's fused SSD backward (ssd_combined.py:971).
# Runs each case alone for ~70 steps with synchronous CUDA errors so the traceback names the faulting kernel, and
# reports every case's exit code (a crash in one case must not hide the result of the others).
set -uo pipefail
export CUDA_LAUNCH_BLOCKING=1
COMMON=(--set train.compile=false --set train.train_tokens=2e6 --set train.ckpt_every=0 --set eval.every=0
        --set eval.at_end=false)
declare -A CASES=(
  [paper_mb16_s0]="--config configs/runs/tiny_hnet2_2m.yaml --set train.seed=0"
  [paper_mb16_s1]="--config configs/runs/tiny_hnet2_2m.yaml --set train.seed=1"
  [end_linear_mb16_s0]="--config configs/runs/tiny_hnet2_2m.yaml --set model.dc.router=end_linear --set train.seed=0"
  [nosmooth_mb16_s0]="--config configs/runs/tiny_hnet2_2m.yaml --set model.dc.dechunker=repeat --set train.seed=0"
  [paper_mb64_s0]="--config configs/runs/tiny_hnet2_2m.yaml --set data.micro_batch=64 --set train.seed=0"
)
declare -A RC
for name in paper_mb16_s0 paper_mb16_s1 end_linear_mb16_s0 nosmooth_mb16_s0 paper_mb64_s0; do
  echo "=== case $name"
  # shellcheck disable=SC2086
  python3 -m hnet.train.train ${CASES[$name]} "${COMMON[@]}" --set train.out_dir=runs/debug/$name \
    > runs_debug_$name.log 2>&1
  RC[$name]=$?
  grep -v "^frame #" runs_debug_$name.log | grep -E "^step |Error|error|File \"" | tail -25
  echo "=== case $name rc=${RC[$name]}"
done
echo "=== SUMMARY"; for k in "${!RC[@]}"; do echo "$k rc=${RC[$k]}"; done
for k in "${!RC[@]}"; do [ "${RC[$k]}" -eq 0 ] || exit 1; done
