"""Checkpoints: ``<out>/ckpt/step_XXXXXXX/`` with

* ``model.pt``   model state_dict (fp32 master weights; unwrapped, so loadable by ``build_model(cfg)``);
* ``optim.pt``   optimizer state_dict;
* ``rank{r}.pt`` per-rank state: RNG (python, numpy, torch CPU, CUDA), loader state, local counters;
* ``meta.json``  step, tokens_eq, bytes_seen, world_size, the full run config;
* ``KEEP``       marker for permanent checkpoints (never pruned).

Writes go to ``.tmp_step_XXXXXXX`` and are renamed into place by rank 0 after every rank finished, so a
directory named ``step_*`` is always complete (a kill mid-save leaves only a ``.tmp_*`` dir, which is ignored and
replaced by the next save).

Resume guarantees (tested in tests/test_train_resume*.py): resuming from ``step_N`` reproduces the uninterrupted
run exactly (bitwise on CPU): model + optimizer state (AdamW moments, Muon momentum buffers and step counters),
LR schedule (a function of the step), data position (the loaders are a pure function of (seed, step, rank)),
RNG streams, token/byte counters and the not-yet-logged metric accumulators. :func:`resolve_resume` maps
``train.resume`` to a checkpoint; :func:`config_mismatches` lists config changes that would break exactness.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from typing import Optional

import numpy as np
import torch

_STEP_RE = re.compile(r"^step_(\d+)$")


def ckpt_dir(out_dir: str) -> str:
    return os.path.join(out_dir, "ckpt")


def step_name(step: int) -> str:
    return f"step_{step:07d}"


def list_checkpoints(out_dir: str):
    d = ckpt_dir(out_dir)
    if not os.path.isdir(d):
        return []
    out = []
    for n in os.listdir(d):
        m = _STEP_RE.match(n)
        if m and os.path.exists(os.path.join(d, n, "meta.json")):
            out.append((int(m.group(1)), os.path.join(d, n)))
    return sorted(out)


def latest_checkpoint(out_dir: str) -> Optional[str]:
    c = list_checkpoints(out_dir)
    return c[-1][1] if c else None


def rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: dict):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


def _barrier(world_size: int):
    if world_size > 1:
        torch.distributed.barrier()


def save_checkpoint(out_dir: str, step: int, model, optimizer, rank_state: dict, meta: dict, rank: int,
                    world_size: int, keep: bool = False):
    d = ckpt_dir(out_dir)
    tmp = os.path.join(d, ".tmp_" + step_name(step))
    final = os.path.join(d, step_name(step))
    if rank == 0:
        os.makedirs(d, exist_ok=True)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
    _barrier(world_size)
    torch.save({**rank_state, "rng": rng_state()}, os.path.join(tmp, f"rank{rank}.pt"))
    if rank == 0:
        torch.save(model.state_dict(), os.path.join(tmp, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(tmp, "optim.pt"))
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)
        if keep:
            open(os.path.join(tmp, "KEEP"), "w").close()
    _barrier(world_size)
    if rank == 0:
        if os.path.exists(final):
            shutil.rmtree(final)
        os.replace(tmp, final)
    _barrier(world_size)
    return final


def prune_checkpoints(out_dir: str, keep_last: int):
    """Delete rolling checkpoints beyond the newest ``keep_last`` (KEEP-marked ones are never deleted)."""
    if keep_last < 1:
        raise ValueError(f"train.ckpt_keep must be >= 1 (the newest checkpoint is the resume point), got {keep_last}")
    rolling = [(s, p) for s, p in list_checkpoints(out_dir) if not os.path.exists(os.path.join(p, "KEEP"))]
    for s, p in rolling[: max(0, len(rolling) - keep_last)]:
        shutil.rmtree(p)


def resolve_resume(resume, out_dir: str) -> Optional[str]:
    """``train.resume`` -> checkpoint dir or None (fresh start).

    * ``auto``: newest ``<out_dir>/ckpt/step_*`` if any, else None;
    * ``None`` / ``"none"`` / ``False``: None;
    * a checkpoint dir (has ``meta.json``) or a run dir (its newest checkpoint). A path that does not exist or
      holds no checkpoint is an error: an explicit resume request never silently starts from scratch.
    """
    if resume in (None, False, "none", "null", ""):
        return None
    if resume == "auto":
        return latest_checkpoint(out_dir)
    path = str(resume)
    if os.path.exists(os.path.join(path, "meta.json")):
        return path
    if os.path.isdir(ckpt_dir(path)):
        latest = latest_checkpoint(path)
        if latest:
            return latest
    raise FileNotFoundError(f"train.resume={path!r}: not a checkpoint dir (no meta.json) and no "
                            f"{ckpt_dir(path)}/step_* checkpoint inside")


# Config entries that must match between a checkpoint and the resuming run for the continuation to be exact.
# Data keys that only affect speed/placement (workers, prefetch, data root, micro-batch split) may change.
RESUME_KEYS = ["model", "data", "train.seed", "train.optimizer", "train.lr", "train.betas", "train.eps",
               "train.weight_decay", "train.wd_embeddings", "train.no_wd_1d", "train.lr_multiplier_override",
               "train.muon", "train.schedule", "train.steps", "train.train_tokens", "train.train_bytes",
               "train.grad_clip", "train.precision"]
RESUME_DATA_FREE = {"num_workers", "prefetch", "root", "micro_batch"}


def _norm(v):
    return json.loads(json.dumps(v, default=str))


def config_mismatches(saved: dict, current: dict) -> list:
    """Human-readable differences on :data:`RESUME_KEYS` between the checkpoint's config and the current one."""
    out = []
    for key in RESUME_KEYS:
        a, b = saved, current
        for p in key.split("."):
            a = a.get(p) if isinstance(a, dict) else None
            b = b.get(p) if isinstance(b, dict) else None
        a, b = _norm(a), _norm(b)
        if key == "data" and isinstance(a, dict) and isinstance(b, dict):
            a = {k: v for k, v in a.items() if k not in RESUME_DATA_FREE}
            b = {k: v for k, v in b.items() if k not in RESUME_DATA_FREE}
        if a != b:
            if isinstance(a, dict) and isinstance(b, dict):
                for k in sorted(set(a) | set(b)):
                    if a.get(k) != b.get(k):
                        out.append(f"{key}.{k}: checkpoint {a.get(k)!r} -> now {b.get(k)!r}")
            else:
                out.append(f"{key}: checkpoint {a!r} -> now {b!r}")
    return out


def load_checkpoint(path: str, model, optimizer, rank: int, map_location="cpu") -> dict:
    """Loads model/optimizer in place; returns ``{"meta": ..., "rank_state": ...}`` (RNG already restored)."""
    with open(os.path.join(path, "meta.json")) as f:
        meta = json.load(f)
    sd = torch.load(os.path.join(path, "model.pt"), map_location=map_location, weights_only=True)
    model.load_state_dict(sd)
    if optimizer is not None:
        osd = torch.load(os.path.join(path, "optim.pt"), map_location=map_location, weights_only=False)
        optimizer.load_state_dict(osd)
    rp = os.path.join(path, f"rank{rank}.pt")
    rank_state = {}
    if os.path.exists(rp):
        rank_state = torch.load(rp, map_location="cpu", weights_only=False)
        if "rng" in rank_state:
            set_rng_state(rank_state["rng"])
    return {"meta": meta, "rank_state": rank_state}
