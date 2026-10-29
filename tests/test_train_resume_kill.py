"""Kill a training process mid-run (SIGTERM = graceful stop with checkpoint, SIGKILL = hard kill, resume from the
last periodic checkpoint), resume it, and check that every logged train/eval record and the final weights are
bitwise identical to an uninterrupted run. CPU, toy model, AdamW and Muon.

Covers: optimizer state (AdamW moments, Muon momentum buffers + step counters), LR schedule, data position,
RNG, token/byte counters, metric accumulators of a log interval cut by the kill, metrics.jsonl truncation after
a hard kill, and resuming from an explicit checkpoint path into a new run directory.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import pytest
import torch
import yaml

from hnet.train import checkpoint as ckpt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEPS = 40
TIMING = {"step_time", "bytes_per_sec", "tokens_per_sec", "mfu", "mem_gb"}


def run_cfg(out, optimizer, ckpt_every, sleep_ms=10, ckpt_keep=1):
    train = {"out_dir": str(out), "steps": STEPS, "lr": 3e-3, "optimizer": optimizer, "precision": "fp32",
             "device": "cpu", "compile": False, "ckpt_every": ckpt_every, "ckpt_keep": ckpt_keep,
             "schedule": {"name": "wsd", "warmup": 5, "decay": 10}}
    if optimizer == "muon":
        train["muon"] = {"momentum_warmup_steps": 20, "lr": 0.02, "lr_embed": 3e-3, "lr_head": 3e-3,
                         "lr_other": 3e-3}
    return {"model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 32, "n_layers": 2,
                      "sleep_ms": sleep_ms},
            "data": {"source": "synthetic", "seq_len": 64, "micro_batch": 2, "batch_windows": 4, "num_workers": 1},
            "train": train,
            "eval": {"every": 10, "at_end": True},
            "log": {"every": 3}}  # log interval 3 so kills land between log steps


def write(path, cfg):
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return str(path)


def env():
    e = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=REPO)
    return e


def train_cmd(cfg_path, *extra):
    return [sys.executable, "-m", "hnet.train.train", "--config", cfg_path, *extra]


def run_to_end(cfg_path, *extra):
    r = subprocess.run(train_cmd(cfg_path, *extra), cwd=REPO, env=env(), capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return r


def logged_steps(out):
    p = os.path.join(out, "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        steps = [json.loads(line)["step"] for line in f if line.strip().endswith("}")]
    return max(steps, default=0)


def start_and_kill(cfg_path, out, sig, after_step):
    p = subprocess.Popen(train_cmd(cfg_path), cwd=REPO, env=env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    while logged_steps(out) < after_step:
        assert p.poll() is None, f"run ended before the kill:\n{p.stdout.read()[-3000:]}"
        assert time.time() - t0 < 240, "timeout waiting for training progress"
        time.sleep(0.005)
    p.send_signal(sig)
    log = p.communicate(timeout=120)[0]
    return p.returncode, log


def records(out, name):
    with open(os.path.join(out, name)) as f:
        return [{k: v for k, v in json.loads(line).items() if k not in TIMING} for line in f if line.strip()]


def final_weights(out):
    return torch.load(os.path.join(out, "ckpt", ckpt.step_name(STEPS), "model.pt"), weights_only=True)


def assert_same_run(ref, out, first_step=1):
    for name in ("metrics.jsonl", "eval.jsonl"):
        a = [r for r in records(ref, name) if r["step"] >= first_step]
        b = records(out, name)
        assert [r["step"] for r in b] == [r["step"] for r in a], name
        for ra, rb in zip(a, b):
            assert ra == rb, (name, ra["step"], {k: (ra.get(k), rb.get(k)) for k in ra if ra.get(k) != rb.get(k)})
    wa, wb = final_weights(ref), final_weights(out)
    for k in wa:
        assert torch.equal(wa[k], wb[k]), k


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Uninterrupted runs, one per optimizer (built lazily)."""
    cache = {}

    def get(optimizer):
        if optimizer not in cache:
            d = tmp_path_factory.mktemp(f"ref_{optimizer}")
            run_to_end(write(d / "run.yaml", run_cfg(d / "out", optimizer, ckpt_every=7)))
            cache[optimizer] = str(d / "out")
        return cache[optimizer]

    return get


@pytest.mark.parametrize("optimizer,sig", [("adamw", signal.SIGTERM), ("muon", signal.SIGTERM),
                                           ("muon", signal.SIGKILL)])
def test_kill_and_resume_matches_uninterrupted(tmp_path, reference, optimizer, sig):
    ref = reference(optimizer)
    out = tmp_path / "out"
    cfg_path = write(tmp_path / "run.yaml", run_cfg(out, optimizer, ckpt_every=7))
    rc, log = start_and_kill(cfg_path, out, sig, after_step=16)
    latest = ckpt.latest_checkpoint(str(out))
    assert latest is not None, log[-3000:]
    killed_at = int(latest.rsplit("_", 1)[1])
    if sig == signal.SIGTERM:
        assert rc == 0, log[-3000:]
        assert "stopping at step" in log
        assert 16 <= killed_at < STEPS
    else:
        assert rc == -signal.SIGKILL
        assert killed_at % 7 == 0 and killed_at < STEPS
    run_to_end(cfg_path)  # resume: auto
    assert_same_run(ref, str(out))


def test_resume_from_explicit_checkpoint_into_new_dir(tmp_path, reference):
    """--resume PATH (a checkpoint or a run dir) continues exactly in a fresh --out; only later steps are logged."""
    ref = reference("muon")
    src = tmp_path / "src"
    cfg_path = write(tmp_path / "run.yaml", run_cfg(src, "muon", ckpt_every=7, sleep_ms=0, ckpt_keep=2))
    run_to_end(cfg_path, "--stop-at-step", "17")  # checkpoints 14 (rolling) and 17 (stop)
    ck = os.path.join(src, "ckpt", ckpt.step_name(14))
    shutil.rmtree(os.path.join(src, "ckpt", ckpt.step_name(17)))
    assert os.path.isdir(ck)
    run_to_end(cfg_path, "--out", str(tmp_path / "branch"), "--resume", ck)
    assert_same_run(ref, str(tmp_path / "branch"), first_step=15)
    run_to_end(cfg_path, "--out", str(tmp_path / "branch2"), "--resume", str(src))  # run dir -> newest ckpt
    assert_same_run(ref, str(tmp_path / "branch2"), first_step=15)
