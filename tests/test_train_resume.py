"""Exact resume: 10 steps + checkpoint + resume == uninterrupted 15 steps (bitwise on CPU)."""

import json
import os

import torch

from hnet.train import checkpoint as ckpt
from hnet.train.config import load_run_config
from hnet.train.train import Trainer

torch.set_num_threads(1)


def cfg(out, **train):
    return load_run_config(base={
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 32, "n_layers": 2},
        "data": {"source": "synthetic", "seq_len": 64, "micro_batch": 2, "batch_windows": 4},
        "train": {"out_dir": str(out), "steps": 15, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "schedule": {"name": "wsd", "warmup": 3, "decay": 5}, **train},
        "eval": {"every": 0, "at_end": False},
        "log": {"every": 1},
    })


def losses(out):
    with open(os.path.join(out, "metrics.jsonl")) as f:
        return {r["step"]: (r["loss"], r["grad_norm"], r["tokens_eq"]) for r in map(json.loads, f)}


def test_exact_resume(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    Trainer(cfg(a)).train()
    Trainer(cfg(b, stop_at_step=10)).train()
    assert ckpt.latest_checkpoint(str(b)).endswith("step_0000010")
    tr = Trainer(cfg(b))
    assert tr.step == 10 and tr.resumed_from is not None
    tr.train()
    la, lb = losses(a), losses(b)
    assert sorted(lb) == list(range(1, 16))
    for s in range(11, 16):
        assert la[s] == lb[s], (s, la[s], lb[s])
    sa = torch.load(os.path.join(a, "ckpt", "step_0000015", "model.pt"))
    sb = torch.load(os.path.join(b, "ckpt", "step_0000015", "model.pt"))
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_keep_and_prune(tmp_path):
    Trainer(cfg(tmp_path, steps=12, ckpt_every=2, ckpt_keep=2, ckpt_steps=[4])).train()
    steps = [s for s, _ in ckpt.list_checkpoints(str(tmp_path))]
    assert steps == [4, 8, 10, 12]  # rolling 8, 10 + permanent 4 + final 12


def test_signal_stops_with_checkpoint(tmp_path):
    tr = Trainer(cfg(tmp_path, steps=50))
    orig = tr.opt.step

    def step_and_signal():
        orig()
        if tr.step == 6:  # the 7th update: request a stop
            tr._signal(15, None)

    tr.opt.step = step_and_signal
    tr.train()
    assert tr.step == 7
    assert ckpt.latest_checkpoint(str(tmp_path)).endswith("step_0000007")
