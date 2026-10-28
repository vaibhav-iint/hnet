#!/usr/bin/env bash
# S0: run modded-nanogpt's own scripts, unmodified, as the same-hardware reference for our S1/S2 runs.
#
#   scripts/speedrun/run_reference.sh setup              # clone the pinned commit, download FineWeb shards
#   scripts/speedrun/run_reference.sh current            # current track-1 record (#92, 0.665 min): 8 GPUs only
#   scripts/speedrun/run_reference.sh record18 [NGPU]    # record #18 (2025-01-04, 3.4 min): the recipe S1 ports
#   scripts/speedrun/run_reference.sh medium [NGPU]      # current GPT-2 medium record (2.92 target)
#
# Environment: SPEEDRUN_DIR (default ~/speedrun-ref), MICRO_TOKENS (record18 only, per-GPU micro-batch,
# default 65536), TORCHRUN (default torchrun). Logs land in $SPEEDRUN_DIR/modded-nanogpt/logs/.
#
# Fewer than 8 GPUs (see docs/SPEEDRUN.md, "Reference runs"):
# * current: impossible as published. track_1_short/distributed.py asserts WORLD_SIZE == 8 (the sampled softmax,
#   sharded n-gram table and sparse row exchanges assume one micro-batch on each of 8 ranks), and the FA3 kernel
#   needs Hopper + a CUDA 13 runtime. Rent one 8xH100 node for ~15 minutes (7 min compile + run).
# * record18: runs on any NGPU dividing 8. Each rank takes 524288/NGPU tokens per step and splits them into
#   MICRO_TOKENS chunks, i.e. gradient accumulation is built in. With MICRO_TOKENS=65536 the windows, data order
#   and val batches are exactly those of the 8-GPU run; the gradient is 8/NGPU times larger (micro-batch means
#   are summed, not averaged), which Muon (normalized) and Adam (scale-invariant up to eps=1e-8) ignore.
#   So the loss curve should match; only wall clock changes (~8x on one H100; peak memory ~31 GB at 64K).
#   On a 24 GB card set MICRO_TOKENS=32768: windows are then cut at 32K instead of 64K (slightly different
#   document truncation), a negligible but non-bit-exact change.
# * medium: grad_accum_steps = 8 // WORLD_SIZE is built in, so NGPU in {1, 2, 4, 8} reproduces the 8-GPU math.
set -euo pipefail

COMMIT=4ea6b937337a4889b8cfe3f38a93d120048d8f71  # modded-nanogpt master as of 2026-09-28 (record #92)
RECORD18_LOG=records/track_1_short/2025-01-04_SoftCap/31d6c427-f1f7-4d8a-91be-a67b5dcd13fd.txt
DIR=${SPEEDRUN_DIR:-$HOME/speedrun-ref}
REPO=$DIR/modded-nanogpt
TORCHRUN=${TORCHRUN:-torchrun}

cmd=${1:-}
ngpu=${2:-8}
case "$ngpu" in 1|2|4|8) ;; *) echo "NGPU must divide 8" >&2; exit 1 ;; esac

need_repo() { [ -d "$REPO" ] || { echo "run '$0 setup' first" >&2; exit 1; }; }

case "$cmd" in
  setup)
    mkdir -p "$DIR"
    [ -d "$REPO" ] || git clone https://github.com/KellerJordan/modded-nanogpt.git "$REPO"
    git -C "$REPO" checkout -q "$COMMIT"
    # Shards are 100M tokens each. current: 9 (README); record18: 1390 x 524288 = 729M -> 8;
    # medium: ~4740 steps x <=524288 -> 25. Download the max you need (val shard always included).
    (cd "$REPO" && python data/cached_fineweb10B.py "${N_SHARDS:-25}")
    echo "Python env: current/medium records pin torch==2.10 (cu128 wheels, CUDA 13 runtime, see README);"
    echo "record18 needs torch>=2.6 with FlexAttention (it was timed on 2.6.0.dev20241231+cu126)."
    ;;
  current)
    need_repo
    [ "$ngpu" = 8 ] || { echo "the current record runs on exactly 8 GPUs" >&2; exit 1; }
    cd "$REPO" && ./run.sh
    ;;
  record18)
    need_repo
    cd "$REPO"
    # The record's script is embedded verbatim at the top of its log, ending at the first line of 100 '='.
    awk '/^====/ {exit} {print}' "$RECORD18_LOG" > train_gpt_record18.py
    micro=${MICRO_TOKENS:-65536}
    if [ "$micro" != 65536 ]; then
      sed -i "s/max_device_batch_size = 64\*1024/max_device_batch_size = $micro/" train_gpt_record18.py
    fi
    "$TORCHRUN" --standalone --nproc_per_node="$ngpu" train_gpt_record18.py
    ;;
  medium)
    need_repo
    cd "$REPO" && "$TORCHRUN" --standalone --nproc_per_node="$ngpu" train_gpt_medium.py
    ;;
  *)
    sed -n '2,25p' "$0"
    exit 1
    ;;
esac
