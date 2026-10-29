#!/usr/bin/env bash
# One benchmark session on a fresh GPU machine (after `python -m infra.prime setup NAME --no-pytest`), from the repo
# root:  bash scripts/bench/gpu_session.sh [smoke|main|profile|realdata|all]   (default all)
#  smoke     20-step BPE + H-Net runs (compile path sanity) + the CUDA kernel parity tests
#  main      configs/bench/plan_h100.yaml (rungs, micro-batch, eager vs compile, optimizer, packing)
#  profile   configs/bench/plan_profile.yaml (torch.profiler op tables)
#  realdata  configs/bench/plan_realdata.yaml (FineWeb-Edu shards; downloads one shard first)
# Results: runs/bench/<plan>/summary.md (+ summary.json, per-run metrics.jsonl / stdout.log).
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=${HNET_PY:-python}
what=${1:-all}
log() { echo "[gpu_session $(date -u +%H:%M:%S)] $*"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$PY -c "import torch, importlib.util as u; print('torch', torch.__version__, 'cuda', torch.version.cuda, {m: bool(u.find_spec(m)) for m in ('flash_attn', 'mamba_ssm', 'causal_conv1d')})"

if [[ $what == realdata || $what == all ]]; then
  log "data prep (one FineWeb-Edu shard) in the background"
  ($PY scripts/data/prepare.py --dataset finewebedu10B --n-train 1 > runs/bench_data_prep.log 2>&1; echo "exit $?" >> runs/bench_data_prep.log) &
  PREP=$!
fi
if [[ $what == smoke || $what == all ]]; then
  log "smoke: 20 compiled steps each"
  for c in bpe_L04 hnet_L04; do
    $PY -m hnet.train.train --config configs/bench/$c.yaml --set train.steps=20 --set log.every=5 \
      --set train.out_dir=runs/bench/smoke/$c 2>&1 | grep -E "^\[train\] \{|^step|Error|error" | tail -6
  done
  log "CUDA kernel parity tests"
  timeout 900 $PY -m pytest -q -x -p no:cacheprovider tests/test_ssd.py tests/test_mamba2.py tests/test_flex_mask.py \
    tests/test_model_hnet.py 2>&1 | tail -5
fi
if [[ $what == main || $what == all ]]; then
  log "main plan"; $PY scripts/bench/bench_train.py run configs/bench/plan_h100.yaml --skip 2
fi
if [[ $what == profile || $what == all ]]; then
  log "profile plan"; $PY scripts/bench/bench_train.py run configs/bench/plan_profile.yaml --skip 1
fi
if [[ $what == realdata || $what == all ]]; then
  wait $PREP; tail -3 runs/bench_data_prep.log
  log "real-data plan"; $PY scripts/bench/bench_train.py run configs/bench/plan_realdata.yaml --skip 2
fi
log "done"
