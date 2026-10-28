"""Checkpoints: ``<out>/ckpt/step_XXXXXXX/`` with

* ``model.pt``   model state_dict (fp32 master weights; unwrapped, so loadable by ``build_model(cfg)``);
* ``optim.pt``   optimizer state_dict;
* ``rank{r}.pt`` per-rank state: RNG (python, numpy, torch CPU, CUDA), loader state, local counters;
* ``meta.json``  step, tokens_eq, bytes_seen, world_size, the full run config;
* ``KEEP``       marker for permanent checkpoints (never pruned).

Writes go to ``.tmp_step_XXXXXXX`` and are renamed into place by rank 0 after every rank finished, so a
directory named ``step_*`` is always complete.
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
    rolling = [(s, p) for s, p in list_checkpoints(out_dir) if not os.path.exists(os.path.join(p, "KEEP"))]
    for s, p in rolling[: max(0, len(rolling) - keep_last)]:
        shutil.rmtree(p, ignore_errors=True)


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
