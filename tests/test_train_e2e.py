"""End-to-end trainer tests on CPU (tiny models, synthetic packed data)."""

import json
import os

import pytest
import torch

from hnet.train.config import load_run_config
from hnet.train.train import Trainer, main

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_metrics(out, name="metrics.jsonl"):
    with open(os.path.join(out, name)) as f:
        return [json.loads(l) for l in f]


def toy_cfg(out, **train):
    return {
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 32, "n_layers": 2},
        "data": {"source": "synthetic", "seq_len": 128, "micro_batch": 2, "batch_windows": 8},
        "train": {"out_dir": str(out), "steps": 30, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "schedule": {"name": "wsd", "warmup": 0.1, "decay": 0.2}, "lr_multiplier_override": [2.0, 1.0],
                  **train},
        "eval": {"every": 0, "at_end": True},
        "log": {"every": 1},
    }


REQUIRED = ["step", "tokens_eq", "bytes_seen", "lr", "loss", "ce", "bpb", "aux_loss", "grad_norm",
            "grad_norm/stage0", "grad_norm/stage1", "compression/s0", "F/s0", "G/s0", "flops_per_byte", "mfu",
            "bytes_per_sec", "mem_gb", "step_time"]


def test_toy_training_loss_decreases_and_metrics(tmp_path):
    cfg = load_run_config(base=toy_cfg(tmp_path))
    Trainer(cfg).train()
    m = read_metrics(tmp_path)
    assert [r["step"] for r in m] == list(range(1, 31))
    for k in REQUIRED:
        assert k in m[0], k
    assert m[-1]["loss"] < 0.6 * m[0]["loss"]
    gn = m[0]
    assert gn["grad_norm"] == pytest.approx((gn["grad_norm/stage0"] ** 2 + gn["grad_norm/stage1"] ** 2) ** 0.5,
                                            rel=1e-4)
    assert m[-1]["bytes_seen"] > m[0]["bytes_seen"] > 0
    ev = read_metrics(tmp_path, "eval.jsonl")
    assert ev[-1]["step"] == 30 and 0 < ev[-1]["val/bpb"] < 8
    assert os.path.exists(tmp_path / "ckpt" / "step_0000030" / "KEEP")


def test_token_kind_bpb_uses_text_bytes(tmp_path):
    cfg = toy_cfg(tmp_path, steps=3)
    cfg["data"].update({"kind": "tokens", "eot": 255})
    cfg = load_run_config(base=cfg)
    Trainer(cfg).train()
    m = read_metrics(tmp_path)
    r = m[0]
    # bytes per token for synthetic word tokens is ~ 4-7, so bpb ~ ce / ln2 / bpt
    assert r["tokens_eq"] == 3 * 0 + 8 * 128  # one step: 8 windows x 128 tokens
    assert r["bpb"] < r["ce"] / 0.6931 / 2


def test_dry_run_cli(tmp_path, capsys):
    path = tmp_path / "run.yaml"
    import yaml

    with open(path, "w") as f:
        yaml.safe_dump(toy_cfg(tmp_path / "out"), f)
    main(["--config", str(path), "--dry-run", "--set", "train.steps=100"])
    out = capsys.readouterr().out
    assert "[dry-run] ok" in out and '"total_steps": 100' in out
    assert not os.path.exists(tmp_path / "out" / "ckpt")


def _have_model():
    try:
        from hnet.model import build_model  # noqa: F401

        return True
    except Exception:
        return False


needs_model = pytest.mark.skipif(not _have_model(), reason="hnet.model (WS-A2) not available")


@needs_model
def test_tiny_hnet_trains(tmp_path):
    cfg = {
        "model": {"include": os.path.join(REPO, "configs/tiny_cpu.yaml")},
        "data": {"source": "synthetic", "seq_len": 256, "micro_batch": 1, "batch_windows": 2},
        "train": {"out_dir": str(tmp_path), "steps": 30, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "save_final": False, "schedule": {"name": "wsd", "warmup": 3, "decay": 0.2}},
        "eval": {"every": 0, "at_end": False},
        "log": {"every": 5},
    }
    Trainer(load_run_config(base=cfg)).train()
    m = read_metrics(tmp_path)
    assert m[-1]["loss"] < m[0]["loss"] - 0.5
    for k in ["compression/s0", "compression/s1", "grad_norm/stage2", "F/s0", "flops_per_byte"]:
        assert k in m[-1] and m[-1][k] is not None, k


@needs_model
def test_tiny_dense_trains(tmp_path):
    cfg = {
        "model": {"type": "dense", "variant": "llama", "vocab_size": 256, "d_model": 64, "n_layers": 2,
                  "d_intermediate": 128, "num_heads": 4},
        "data": {"source": "synthetic", "kind": "tokens", "eot": 255, "seq_len": 128, "micro_batch": 2,
                 "batch_windows": 4},
        "train": {"out_dir": str(tmp_path), "steps": 30, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "save_final": False, "schedule": {"name": "wsd", "warmup": 3, "decay": 0.2}},
        "eval": {"every": 0, "at_end": False},
        "log": {"every": 5},
    }
    Trainer(load_run_config(base=cfg)).train()
    m = read_metrics(tmp_path)
    assert m[-1]["loss"] < m[0]["loss"] - 0.5


def test_activation_checkpointing_same_losses(tmp_path):
    runs = {}
    for ac in (False, True):
        out = tmp_path / str(ac)
        Trainer(load_run_config(base=toy_cfg(out, steps=4, activation_checkpointing=ac, save_final=False))).train()
        runs[ac] = [r["loss"] for r in read_metrics(out)]
    assert runs[True] == pytest.approx(runs[False], rel=1e-6)
