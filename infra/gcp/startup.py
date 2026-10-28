"""Render the GCE startup script: a small bash wrapper that writes job.json + vm_agent.py and starts the
agent as a transient systemd unit (so it outlives the startup-script service). Runs on every boot;
the agent itself is idempotent."""

from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

AGENT_PATH = Path(__file__).with_name("vm_agent.py")

TEMPLATE = r"""#!/bin/bash
# hnet infra.gcp startup script (generated). Re-runs on every boot; the agent skips finished steps.
set -u
S=/opt/hnet-state
mkdir -p "$S" /opt/hnet-cache
echo "__JOB_B64__" | base64 -d > "$S/job.json"
echo "__AGENT_B64__" | base64 -d > "$S/vm_agent.py"
if systemctl is-active --quiet hnet-agent 2>/dev/null; then
  echo "hnet-agent already running"; exit 0
fi
systemctl reset-failed hnet-agent 2>/dev/null || true
PY=/usr/bin/python3; [ -x "$PY" ] || PY=$(command -v python3)
if command -v systemd-run >/dev/null; then
  systemd-run --unit=hnet-agent --collect -p StandardOutput=append:$S/agent.log \
    -p StandardError=append:$S/agent.log "$PY" "$S/vm_agent.py" "$S/job.json"
else
  setsid nohup "$PY" "$S/vm_agent.py" "$S/job.json" >> "$S/agent.log" 2>&1 < /dev/null &
fi
echo "hnet-agent started"
"""


def job_dict(*, repo: str, commit: str, cmd: str, data: str = "auto", linger_min: float = 5.0,
             auto_shutdown: bool = True, metrics_every: int = 1, kernels: bool = True, allow_compile: bool = False,
             env: dict | None = None, watch_dirs: list[str] | None = None, nonce: str | None = None) -> dict:
    """`nonce` identifies one run request: the agent re-runs a finished command only if it changed."""
    return {"nonce": nonce or uuid.uuid4().hex[:12], "repo": repo, "commit": commit, "cmd": cmd, "data": data, "linger_min": linger_min,
            "auto_shutdown": auto_shutdown, "metrics_every": metrics_every, "kernels": kernels,
            "allow_compile": allow_compile, "env": env or {}, "watch_dirs": watch_dirs or []}


MAX_OVERLAY_BYTES = 128 * 1024  # metadata values are capped at 256 KB (startup script incl. agent ~45 KB)


def make_overlay(paths: list[str], root: Path) -> str:
    """tar.gz (base64) of repo-relative files/dirs, for files the commit lacks (e.g. untracked configs)."""
    import io
    import tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in paths:
            rel = Path(p)
            if rel.is_absolute():
                rel = rel.resolve().relative_to(root.resolve())
            full = root / rel
            if not full.exists():
                raise FileNotFoundError(f"overlay path {p} not found under {root}")
            tf.add(str(full), arcname=str(rel),
                   filter=lambda ti: None if ("__pycache__" in ti.name or ti.name.endswith(".pyc")) else ti)
    data = buf.getvalue()
    if len(data) > MAX_OVERLAY_BYTES:
        raise ValueError(f"overlay is {len(data)} bytes compressed (> {MAX_OVERLAY_BYTES}); commit those files instead")
    return base64.b64encode(data).decode()


def render(job: dict) -> str:
    jb = base64.b64encode(json.dumps(job).encode()).decode()
    ab = base64.b64encode(AGENT_PATH.read_bytes()).decode()
    return TEMPLATE.replace("__JOB_B64__", jb).replace("__AGENT_B64__", ab)
