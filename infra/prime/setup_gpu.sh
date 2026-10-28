#!/usr/bin/env bash
# Remote environment setup. Run from the repo root (infra.prime `setup` does this detached).
#  1. find python (install python3+pip via apt if absent)
#  2. torch: keep a working CUDA torch; else install from the index matching the driver's CUDA
#  3. requirements.txt + base deps
#  4. optional kernels from PREBUILT wheels only (mamba_ssm, causal_conv1d, flash_attn);
#     mamba_ssm falls back to its Triton-only install (no CUDA compile). Compiling is opt-in via
#     HNET_ALLOW_COMPILE=1 and capped at 10 min per package.
#  5. capability summary (runs/setup/capabilities.json) + import / pytest smoke test
# Env: HNET_PY (python), HNET_SETUP_PYTEST=0/1, HNET_SKIP_TORCH=1, HNET_SKIP_KERNELS=1, HNET_ALLOW_COMPILE=1
set -uo pipefail
cd "$(dirname "$0")/../.."
T0=$(date +%s)
log() { echo "[setup +$(( $(date +%s) - T0 ))s] $*"; }
export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore

SUDO=""; [ "$(id -u)" != 0 ] && command -v sudo >/dev/null && SUDO="sudo -n"
PY="${HNET_PY:-}"
if [ -z "$PY" ]; then
  for c in python3 python; do command -v $c >/dev/null && { PY=$c; break; }; done
fi
if [ -z "$PY" ] || ! $PY -m pip --version >/dev/null 2>&1; then
  log "installing python3 + pip via apt"
  $SUDO apt-get update -qq && $SUDO DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3 python3-pip python3-venv git >/dev/null
  PY=python3
fi
log "python: $($PY --version 2>&1) at $(command -v $PY)"
PIP="$PY -m pip install -q"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---- torch ----
if [ "${HNET_SKIP_TORCH:-0}" != 1 ]; then
  if $PY -c "import torch,sys; sys.exit(0 if (torch.cuda.is_available() or not __import__('shutil').which('nvidia-smi')) else 1)" 2>/dev/null; then
    log "torch present: $($PY -c 'import torch;print(torch.__version__, torch.version.cuda)')"
  else
    IDX=$($PY infra/prime/gpu_caps.py torch-index)
    log "installing torch from $IDX"
    $PIP torch --index-url "$IDX" || log "WARN torch install failed"
  fi
fi

# ---- python deps ----
log "installing requirements"
$PIP numpy einops pytest pyyaml tokenizers safetensors tiktoken huggingface_hub packaging ninja || log "WARN base deps failed"
[ -f requirements.txt ] && { $PIP -r requirements.txt || log "WARN requirements.txt failed"; }

# ---- kernels ----
try_wheel() {  # $1 = package key
  local url; url=$($PY infra/prime/gpu_caps.py wheel "$1" 2>/dev/null)
  if [ -n "$url" ]; then log "$1: prebuilt wheel $(basename "$url")"; $PIP --no-deps "$url" && return 0; fi
  return 1
}
if [ "${HNET_SKIP_KERNELS:-0}" != 1 ] && $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  for pkg in causal_conv1d mamba_ssm flash_attn; do
    $PY -c "import $pkg" 2>/dev/null && { log "$pkg already importable"; continue; }
    try_wheel $pkg && continue
    case $pkg in
      mamba_ssm)
        log "mamba_ssm: no matching wheel; Triton-only install (no CUDA build)"
        MAMBA_SKIP_CUDA_BUILD=TRUE timeout 600 $PY -m pip install -q --no-build-isolation --no-deps mamba-ssm \
          || log "WARN mamba_ssm triton install failed; pure-torch SSD fallback will be used";;
      *)
        if [ "${HNET_ALLOW_COMPILE:-0}" = 1 ]; then
          log "$pkg: no wheel; compiling (max 10 min)"
          MAX_JOBS=${MAX_JOBS:-8} timeout 600 $PY -m pip install -q --no-build-isolation "${pkg//_/-}" \
            || log "WARN $pkg compile failed/timed out; fallback path will be used"
        else
          log "$pkg: no matching prebuilt wheel; skipping (pure-torch fallback)"
        fi;;
    esac
  done
else
  log "no CUDA torch: skipping fast kernels"
fi

# ---- summary + smoke ----
mkdir -p runs/setup
log "capability summary:"
$PY infra/prime/gpu_caps.py summary runs/setup/capabilities.json
log "import smoke test"
$PY -c "import hnet, torch; print('import hnet OK; torch', torch.__version__)" || log "WARN import hnet failed"
if [ "${HNET_SETUP_PYTEST:-1}" = 1 ] && [ -d tests ]; then
  log "pytest smoke (max 15 min)"
  OMP_NUM_THREADS=4 timeout 900 $PY -m pytest -q -x tests 2>&1 | tail -n 25 || true
fi
log "done"
