#!/usr/bin/env bash
# PLAN run #0: eval-only harness checks on a GPU box (validates prompts, normalization and scorers).
#   bash scripts/eval/reference_evals.sh [out_dir] [limit]
# Pass criteria (PLAN sec. 5):
#   * official:cartesia-ai/hnet_2stage_L paper7 row within +-0.5 of paper Table 2
#     (bpb 0.743 | LMB 46.9 | Hella 57.4 | PIQA 72.0 | ARC-e 71.7 | ARC-c 39.2 | Wino 60.4 | OBQA 40.6 | avg 55.5);
#   * GPT-2 lm-eval numbers (approx., from public lm-eval runs; re-check) (124M: HellaSwag acc_norm 31.1, LAMBADA acc 32.6 / ppl 40.1, PIQA 62.9, ARC-e 43.8,
#     ARC-c acc_norm 22.7, Winogrande 51.6, OBQA acc_norm 27.2);
#   * SmolLM3 stage-1 revisions vs configs/reference/blog1_fig7_smollm3.csv (smollm3_cf, decide token- vs char-norm).
set -euo pipefail
OUT=${1:-results/reference}
LIMIT=${2:-}
LIM=${LIMIT:+--limit $LIMIT}
mkdir -p "$OUT"
PY=${PY:-python}

run() {  # name model suites
  $PY -m hnet.eval.run --model "$2" --suite "$3" $LIM --out "$OUT/$1.json" --items "$OUT/$1.items.jsonl" --progress
}

run hnet_2stage_L official:cartesia-ai/hnet_2stage_L paper7,robustness,bpb
run hnet_1stage_L official:cartesia-ai/hnet_1stage_L paper7,robustness
run gpt2 hf:gpt2 paper7,smollm3_cf,bpb
run gpt2-medium hf:gpt2-medium paper7,smollm3_cf
run smollm2-135m hf:HuggingFaceTB/SmolLM2-135M paper7,smollm3_cf
run smollm2-360m hf:HuggingFaceTB/SmolLM2-360M paper7,smollm3_cf
# Two SmolLM3 stage-1 points of Cartesia's Fig. 7 baseline curve (tokens = step * 2.36M)
for step in 40000 80000; do
  run smollm3-stage1-$step "hf:HuggingFaceTB/SmolLM3-3B-checkpoints@stage1-step-$step" smollm3_cf
done
$PY -m hnet.eval.report --results "$OUT"/*.json --md "$OUT/report.md"
