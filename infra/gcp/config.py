"""Paths, machine shapes, zones and price estimates for infra.gcp. No network here.

Prices are conservative ESTIMATES (USD/h, us-central1-ish list prices rounded up), not billing data.
Override any entry with HNET_GCP_PRICES='{"g2-standard-8": {"spot": 0.30}}' or `launch --price-hr`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRATCHPAD = Path("/tmp/claude-0/-home-user/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad")

NAME_PREFIX = "hnet-"
LABELS = {"hnet-owner": "infra-gcp", "hnet-project": "hnet-smollm3"}
REPO_URL = os.environ.get("HNET_GCP_REPO_URL", "https://github.com/vaibhav-iint/hnet.git")
IMAGE_FAMILY = os.environ.get("HNET_GCP_IMAGE_FAMILY", "pytorch-2-9-cu129-ubuntu-2204-nvidia-580")
IMAGE_PROJECT = os.environ.get("HNET_GCP_IMAGE_PROJECT", "deeplearning-platform-release")
MIN_DISK_GB = 100  # the DL image needs >= 100 GB
DISK_TYPE = "pd-balanced"
DISK_USD_PER_GB_MONTH = 0.10  # pd-balanced estimate
HOURS_PER_MONTH = 730.0

DEFAULT_BUDGET_TOTAL = float(os.environ.get("HNET_GCP_BUDGET", "300"))
DEFAULT_LAUNCH_CAP = float(os.environ.get("HNET_GCP_LAUNCH_CAP", "25"))


def state_dir() -> Path:
    env = os.environ.get("HNET_GCP_STATE_DIR")
    if env:
        p = Path(env)
    elif _SCRATCHPAD.is_dir():
        p = _SCRATCHPAD / "gcp_state"
    else:
        p = Path.home() / ".cache" / "hnet-gcp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gcloud_config_dir() -> str | None:
    """CLOUDSDK_CONFIG to use: the env value, else the scratchpad gcloud dir if present."""
    if os.environ.get("CLOUDSDK_CONFIG"):
        return os.environ["CLOUDSDK_CONFIG"]
    d = _SCRATCHPAD / "gcloud"
    return str(d) if d.is_dir() else None


# ---- machine shapes: gpu kind -> count -> (machine type, accelerator flag or None) ----
# g2 / a2 machine types have their GPUs built in; n1 needs an explicit --accelerator.
MACHINES = {
    "l4": {1: ("g2-standard-8", None), 2: ("g2-standard-24", None), 4: ("g2-standard-48", None),
           8: ("g2-standard-96", None)},
    "a100": {1: ("a2-highgpu-1g", None), 2: ("a2-highgpu-2g", None), 4: ("a2-highgpu-4g", None),
             8: ("a2-highgpu-8g", None)},
    "t4": {1: ("n1-standard-8", "nvidia-tesla-t4"), 2: ("n1-standard-16", "nvidia-tesla-t4"),
           4: ("n1-standard-32", "nvidia-tesla-t4")},
    "cpu": {0: ("e2-standard-8", None)},
}
ACCEL_NAME = {"l4": "nvidia-l4", "a100": "nvidia-tesla-a100", "t4": "nvidia-tesla-t4"}

# ---- $/h estimates (machine incl. built-in GPUs; n1 entries include the T4s) ----
PRICES = {
    "g2-standard-8": {"standard": 0.85, "spot": 0.35},
    "g2-standard-24": {"standard": 2.00, "spot": 0.80},
    "g2-standard-48": {"standard": 4.00, "spot": 1.60},
    "g2-standard-96": {"standard": 8.00, "spot": 3.20},
    "a2-highgpu-1g": {"standard": 3.67, "spot": 1.45},
    "a2-highgpu-2g": {"standard": 7.35, "spot": 2.90},
    "a2-highgpu-4g": {"standard": 14.70, "spot": 5.80},
    "a2-highgpu-8g": {"standard": 29.40, "spot": 11.60},
    "n1-standard-8": {"standard": 0.75, "spot": 0.30},    # + 1x T4
    "n1-standard-16": {"standard": 1.50, "spot": 0.60},   # + 2x T4
    "n1-standard-32": {"standard": 3.00, "spot": 1.20},   # + 4x T4
    "e2-standard-8": {"standard": 0.27, "spot": 0.10},
}

# ---- zone preference (quota exists in these regions); `--zone auto` tries them in order ----
ZONES = {
    "l4": ["us-central1-a", "us-central1-b", "us-central1-c", "us-east1-b", "us-east1-c", "us-east1-d",
           "us-west1-a", "us-west1-b", "us-west1-c", "europe-west4-a", "europe-west4-b", "europe-west4-c"],
    "a100": ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f", "us-east1-b", "us-west1-b",
             "europe-west4-a", "europe-west4-b"],
    "t4": ["us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f", "us-east1-c", "us-east1-d",
           "us-west1-a", "us-west1-b", "europe-west4-b", "europe-west4-c"],
    "cpu": ["us-central1-a", "us-central1-b", "us-east1-b", "us-west1-a"],
}


def machine_for(gpu: str, count: int) -> tuple[str, str | None, int]:
    """(machine type, accelerator type or None, gpu count) for a GPU kind and count."""
    gpu = gpu.lower()
    if gpu not in MACHINES:
        raise ValueError(f"unknown --gpu {gpu!r}; choose from {sorted(MACHINES)}")
    if gpu == "cpu":
        return MACHINES["cpu"][0][0], None, 0
    if count not in MACHINES[gpu]:
        raise ValueError(f"--count {count} not supported for {gpu}; choose from {sorted(MACHINES[gpu])}")
    mt, acc = MACHINES[gpu][count]
    return mt, acc, count


def price_table() -> dict:
    t = json.loads(json.dumps(PRICES))
    ov = os.environ.get("HNET_GCP_PRICES")
    if ov:
        for k, v in json.loads(ov).items():
            t.setdefault(k, {}).update(v)
    return t


def price_hr(machine_type: str, spot: bool) -> float:
    t = price_table()
    if machine_type not in t:
        raise KeyError(f"no price estimate for {machine_type}; pass --price-hr or set HNET_GCP_PRICES")
    return float(t[machine_type]["spot" if spot else "standard"])


def disk_price_hr(disk_gb: int) -> float:
    return disk_gb * DISK_USD_PER_GB_MONTH / HOURS_PER_MONTH
