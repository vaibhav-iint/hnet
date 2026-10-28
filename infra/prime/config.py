"""Paths, defaults, prices and credential loading for infra.prime.

Nothing here talks to the network. Credentials come from the environment (PRIME_API_KEY,
PRIME_TEAM_ID) or from an env file named by HNET_PRIME_ENV (KEY=VALUE lines, optional
`export`). The key is never printed or written anywhere by this package.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Session scratchpad used by the agents that built this project; falls back to ~/.cache.
_SCRATCHPAD = Path("/tmp/claude-0/-home-user/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad")


def state_dir() -> Path:
    env = os.environ.get("HNET_PRIME_STATE_DIR")
    if env:
        p = Path(env)
    elif _SCRATCHPAD.is_dir():
        p = _SCRATCHPAD / "prime_state"
    else:
        p = Path.home() / ".cache" / "hnet-prime"
    p.mkdir(parents=True, exist_ok=True)
    return p


def env_file() -> Path | None:
    env = os.environ.get("HNET_PRIME_ENV")
    if env:
        return Path(env)
    cand = _SCRATCHPAD / "prime.env"
    return cand if cand.is_file() else None


def load_credentials() -> None:
    """Populate PRIME_API_KEY / PRIME_TEAM_ID from the env file if they are not already set."""
    if os.environ.get("PRIME_API_KEY"):
        return
    f = env_file()
    if f is None or not f.is_file():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in ("PRIME_API_KEY", "PRIME_TEAM_ID") and not os.environ.get(k):
            os.environ[k] = v


# ---- budget defaults (override with env or CLI flags) ----
DEFAULT_BUDGET_TOTAL = float(os.environ.get("HNET_PRIME_BUDGET", "500"))
DEFAULT_LAUNCH_CAP = float(os.environ.get("HNET_PRIME_LAUNCH_CAP", "50"))

# ---- sandbox pricing (docs.primeintellect.ai/sandboxes/overview, valid through 2026-12-22) ----
SANDBOX_CPU_PER_VCPU_HR = 0.02
SANDBOX_MEM_PER_GIB_HR = 0.0125
SANDBOX_DISK_PER_GIB_HR = 0.0002
# GPU sandboxes need an explicit grant per the docs and have no public price list/API yet.
# Known type names (from the CLI help). Override prices with HNET_PRIME_SANDBOX_GPU_PRICES='{"H200_141GB": 5.4}'.
SANDBOX_GPU_TYPES = ["RTX_PRO_6000", "H200_141GB"]

# ---- resource labels / remote layout ----
LABEL = "hnet-smollm3"
NAME_PREFIX = "hnet-"
SANDBOX_REMOTE_ROOT = "/workspace/hnet-smollm3"
POD_REMOTE_ROOT = "~/hnet-smollm3"
DEFAULT_SANDBOX_IMAGE_CPU = "python:3.12-slim"
DEFAULT_SANDBOX_IMAGE_GPU = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
DEFAULT_POD_IMAGE = "cuda_12_6_pytorch_2_7"
