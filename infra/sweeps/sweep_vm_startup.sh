#!/bin/bash
# GCE startup script for one experiment sweep (scripts/search/sweep.py). Fails loudly: any error stops the script,
# the full log goes to GCS, and the VM shuts down (so a broken run never burns GPU hours silently).
# Required metadata (instance attributes): hnet-commit, hnet-spec, hnet-gcs (gs:// prefix for this sweep).
set -euo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
attr() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
COMMIT=$(attr hnet-commit); SPEC=$(attr hnet-spec); GCS=$(attr hnet-gcs); PARALLEL=$(attr hnet-parallel || echo 3)
# Optional, to split one sweep across VMs (restore and mirror are restricted to this VM's jobs):
#   hnet-shard  i/n: jobs i, i+n, ... of the spec's job order (scripts/search/spec.py; infra/sweeps/launch.py sets it)
#   hnet-only   comma-separated run names (architectures, no seed suffix)
#   hnet-allow-missing  1: skip runs whose config files do not exist (otherwise the sweep refuses to start)
ONLY=$(attr hnet-only || echo ""); SHARD=$(attr hnet-shard || echo ""); ALLOW_MISSING=$(attr hnet-allow-missing || echo "")
LOG=/var/log/hnet_sweep.log
exec > >(tee -a "$LOG" | sed -u 's/^/HNET /' > /dev/console) 2>&1
upload_log() { gcloud storage cp "$LOG" "$GCS/_logs/$(hostname)-$(date -u +%Y%m%dT%H%M%S).log" || echo "WARN log upload failed"; }
finish() { rc=$?; echo "HNET_DONE rc=$rc"; upload_log; sleep 30; shutdown -h now; }
trap finish EXIT

echo "boot $(date -u) commit=$COMMIT spec=$SPEC gcs=$GCS shard=${SHARD:-all} only=${ONLY:-all}"
[ -d /opt/hnet/.git ] || git clone -q https://github.com/vaibhav-iint/hnet.git /opt/hnet
cd /opt/hnet && git fetch -q origin && git checkout -q "$COMMIT"
export PYTHONPATH=/opt/hnet
python3 -m pip install -q -r requirements.txt pytest
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
# Fast kernels are required (prebuilt wheels only); then the CUDA parity tests must pass before any training.
for pkg in causal_conv1d mamba_ssm; do
  python3 -c "import $pkg" 2>/dev/null && continue
  url=$(python3 infra/prime/gpu_caps.py wheel $pkg | tail -1)
  case "$url" in http*) python3 -m pip install -q --no-deps "$url" ;; *) echo "no prebuilt wheel for $pkg: $url"; exit 1 ;; esac
done
python3 infra/prime/gpu_caps.py summary
python3 -m pytest -q -p no:cacheprovider -m "not slow" tests/test_ssd.py tests/test_mamba2.py tests/test_model_components.py tests/test_model_hnet.py
mkdir -p data/finewebedu10B
gcloud storage rsync -r gs://neat-dispatch-501718-a5-hnet/data/finewebedu10B data/finewebedu10B
python3 scripts/data/prepare.py --dataset finewebedu10B --n-train 1
ARGS=(--parallel "$PARALLEL" --gcs "$GCS" --restore --mirror-every 120)
[ -n "$SHARD" ] && ARGS+=(--shard "$SHARD")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
[ "$ALLOW_MISSING" = "1" ] && ARGS+=(--allow-missing)
# sweep.py restores this selection's runs from GCS (finished runs and checkpoints, after a preemption or on another
# VM; run dirs already on this disk are kept), mirrors them every 120 s, and uploads each run when it ends.
python3 scripts/search/sweep.py "$SPEC" "${ARGS[@]}"
echo "sweep finished OK"
