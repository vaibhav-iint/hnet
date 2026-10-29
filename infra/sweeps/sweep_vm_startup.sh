#!/bin/bash
# GCE startup script for one experiment sweep (scripts/search/sweep.py). Fails loudly: any error stops the script,
# the full log goes to GCS, and the VM shuts down (so a broken run never burns GPU hours silently).
# Required metadata (instance attributes): hnet-commit, hnet-spec, hnet-gcs (gs:// prefix for this sweep).
set -euo pipefail
MD=http://metadata.google.internal/computeMetadata/v1/instance/attributes
attr() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }
COMMIT=$(attr hnet-commit); SPEC=$(attr hnet-spec); GCS=$(attr hnet-gcs); PARALLEL=$(attr hnet-parallel || echo 3)
LOG=/var/log/hnet_sweep.log
exec > >(tee -a "$LOG" | sed -u 's/^/HNET /' > /dev/console) 2>&1
upload_log() { gcloud storage cp "$LOG" "$GCS/_logs/$(hostname)-$(date -u +%Y%m%dT%H%M%S).log" || true; }
finish() { rc=$?; echo "HNET_DONE rc=$rc"; upload_log; sleep 30; shutdown -h now; }
trap finish EXIT

echo "boot $(date -u) commit=$COMMIT spec=$SPEC gcs=$GCS"
[ -d /opt/hnet/.git ] || git clone -q https://github.com/vaibhav-iint/hnet.git /opt/hnet
cd /opt/hnet && git fetch -q origin && git checkout -q "$COMMIT"
export PYTHONPATH=/opt/hnet
python3 -m pip install -q -r requirements.txt
python3 -c "import torch; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p data/finewebedu10B
gcloud storage rsync -r gs://neat-dispatch-501718-a5-hnet/data/finewebedu10B data/finewebedu10B
python3 scripts/data/prepare.py --dataset finewebedu10B --n-train 1
OUT=$(python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['out_root'])" "$SPEC")
mkdir -p "$OUT"
( while true; do sleep 120; gcloud storage rsync -r "$OUT" "$GCS" >/dev/null 2>&1 || echo "WARN live mirror failed"; done ) &
python3 scripts/search/sweep.py "$SPEC" --parallel "$PARALLEL" --gcs "$GCS"
echo "sweep finished OK"
