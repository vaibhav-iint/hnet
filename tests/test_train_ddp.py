"""DDP (gloo, 2 CPU processes via torchrun) matches 1 process with doubled grad accumulation."""

import json
import os
import socket
import subprocess
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cfg(out):
    return {
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 32, "n_layers": 2},
        "data": {"source": "synthetic", "seq_len": 64, "micro_batch": 1, "batch_windows": 4},
        "train": {"out_dir": str(out), "steps": 6, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "device": "cpu", "save_final": False, "schedule": {"name": "constant", "warmup": 2},
                  "check_signals_every": 2},
        "eval": {"every": 3, "at_end": True},
        "log": {"every": 1},
    }


def _run(cfg_path, nproc):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "PYTHONPATH": REPO}
    if nproc == 1:
        cmd = [sys.executable, "-m", "hnet.train.train", "--config", str(cfg_path)]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(nproc),
               "--master_port", str(_free_port()), "-m", "hnet.train.train", "--config", str(cfg_path)]
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]


def _metrics(out, name="metrics.jsonl"):
    with open(os.path.join(out, name)) as f:
        return [json.loads(l) for l in f]


def test_ddp_matches_single_process(tmp_path):
    outs = {}
    for n in (1, 2):
        out = tmp_path / f"w{n}"
        p = tmp_path / f"w{n}.yaml"
        with open(p, "w") as f:
            yaml.safe_dump(_cfg(out), f)
        _run(p, n)
        outs[n] = out
    m1, m2 = _metrics(outs[1]), _metrics(outs[2])
    assert len(m1) == len(m2) == 6
    for a, b in zip(m1, m2):
        assert a["step"] == b["step"]
        assert a["bytes_seen"] == b["bytes_seen"]  # same windows per step, globally counted
        assert b["loss"] == pytest.approx(a["loss"], rel=2e-3, abs=2e-3)
        assert b["grad_norm/stage0"] == pytest.approx(a["grad_norm/stage0"], rel=2e-2)
    assert m2[0]["loss"] == pytest.approx(m1[0]["loss"], rel=1e-5)
    e1, e2 = _metrics(outs[1], "eval.jsonl"), _metrics(outs[2], "eval.jsonl")
    assert [e["step"] for e in e2] == [3, 6]
    assert e2[-1]["val/bytes"] == e1[-1]["val/bytes"]
    assert e2[-1]["val/bpb"] == pytest.approx(e1[-1]["val/bpb"], rel=5e-3)
