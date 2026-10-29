"""Local-side access to the run bucket (layout written by vm_agent.Gcs):

  gs://<bucket>/runs/<vm>/<run>/{metrics.jsonl, eval.jsonl, config.yaml, ckpt/step_*/...}
  gs://<bucket>/runs/<vm>/_logs/{agent.log, setup.log, cmd.log, ...}, _summary.json
  gs://<bucket>/data/<dataset>/...           shared shard cache

All calls go through infra.gcp.gce (gcloud storage); errors raise.
"""

from __future__ import annotations

from pathlib import Path

from . import gce
from . import serial as S
from .vm_agent import latest_complete_ckpt

CKPT_CHOICES = ("latest", "all", "none")


def bucket_of(rec: dict) -> str | None:
    """The bucket a VM's job syncs to (None for serial-only VMs)."""
    gcs = ((rec.get("meta") or {}).get("job") or {}).get("gcs") or {}
    return gcs.get("bucket")


def vm_url(bucket: str, vm: str) -> str:
    return f"gs://{bucket}/runs/{vm}"


def run_name(run: str) -> str:
    """'runs/tiny/x' or 'tiny/x' -> 'tiny/x' (the name under the VM prefix)."""
    run = run.strip("/")
    return run[5:] if run.startswith("runs/") else run


def fetch_jsonl(bucket: str, vm: str, cache: Path) -> tuple[dict, dict]:
    """(metrics, evals) of every run under the VM prefix, keyed like serial run dirs ('runs/<run>')."""
    base = vm_url(bucket, vm) + "/"
    metrics, evals = {}, {}
    for u in gce.storage_ls_sizes(base):
        rel = u[len(base):]
        if rel.startswith("_logs/") or not rel.endswith(("/metrics.jsonl", "/eval.jsonl")):
            continue
        dst = cache / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        gce.storage_cp([u], str(dst), timeout=600)
        store = metrics if rel.endswith("metrics.jsonl") else evals
        store["runs/" + rel.rsplit("/", 1)[0]] = S.read_jsonl_text(dst.read_text())
    return metrics, evals


def select_objects(rels: list[str], ckpt: str) -> list[str]:
    """Which objects of a run dir to pull: everything, no checkpoints, or only the newest complete one."""
    if ckpt not in CKPT_CHOICES:
        raise ValueError(f"ckpt must be one of {CKPT_CHOICES}")
    if ckpt == "all":
        return list(rels)
    step = latest_complete_ckpt(rels) if ckpt == "latest" else None
    return [r for r in rels if not r.startswith("ckpt/") or (step and r.startswith(f"ckpt/{step}/"))]


def pull_run(bucket: str, vm: str, run: str, out: Path, ckpt: str = "latest") -> tuple[int, int]:
    """Copy gs://B/runs/<vm>/<run>/ into `out`; returns (objects, bytes). FileNotFoundError if empty."""
    src = f"{vm_url(bucket, vm)}/{run_name(run)}"
    objs = gce.storage_ls_sizes(src)
    if not objs:
        raise FileNotFoundError(f"nothing under {src}/")
    rels = [u[len(src) + 1:] for u in objs]
    keep = select_objects(rels, ckpt)
    by_dir: dict[Path, list[str]] = {}
    for r in keep:
        by_dir.setdefault((out / r).parent, []).append(f"{src}/{r}")
    for d, urls in by_dir.items():
        d.mkdir(parents=True, exist_ok=True)
        gce.storage_cp(urls, str(d) + "/", timeout=6 * 3600)
    return len(keep), sum(objs[f"{src}/{r}"] for r in keep)
