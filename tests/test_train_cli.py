"""Trainer command line (hnet.train.cli): first-class flags, precedence vs --set, --model swap, --print-config;
plus the resume/checkpoint hardening knobs (resume_strict, explicit resume paths, ckpt_tokens, divergence stop)."""

import json
import math
import os

import pytest
import torch
import yaml

from hnet.train import checkpoint as ckpt
from hnet.train import cli
from hnet.train.config import load_run_config
from hnet.train.train import Trainer, main

torch.set_num_threads(1)
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINY_HNET = os.path.join(REPO, "configs", "runs", "tiny_hnet2_2m.yaml")


def compose(*argv):
    return cli.compose(cli.parse_args(["--config", TINY_HNET, *argv]))


def test_flags_map_onto_config_keys():
    cfg = compose("--lr", "3e-3", "--seed", "2", "--optimizer", "muon", "--muon-lr", "0.02", "--micro-batch", "8",
                  "--out", "runs/x", "--precision", "fp32", "--eval-every", "100", "--ckpt-every", "50",
                  "--ckpt-keep", "3", "--ckpt-steps", "10,20", "--ckpt-tokens", "1e6,2.5e6", "--warmup", "0.05",
                  "--log-every", "5", "--max-hours", "1.5", "--resume", "none", "--no-compile")
    t = cfg["train"]
    assert t["lr"] == 3e-3 and t["seed"] == 2 and t["optimizer"] == "muon" and t["muon"]["lr"] == 0.02
    assert cfg["data"]["micro_batch"] == 8 and t["out_dir"] == "runs/x" and t["precision"] == "fp32"
    assert cfg["eval"]["every"] == 100 and t["ckpt_every"] == 50 and t["ckpt_keep"] == 3
    assert t["ckpt_steps"] == [10, 20] and t["ckpt_tokens"] == [1e6, 2.5e6]
    assert t["schedule"]["warmup"] == 0.05 and t["schedule"]["name"] == "wsd" and t["schedule"]["decay"] == 0.2
    assert cfg["log"]["every"] == 5 and t["max_hours"] == 1.5 and t["resume"] is None and t["compile"] is False
    assert compose("--compile")["train"]["compile"] == "blocks"
    assert compose("--compile", "model")["train"]["compile"] == "model"
    assert compose("--resume", "auto")["train"]["resume"] == "auto"


def test_budget_flags_clear_the_other_budgets():
    cfg = compose("--steps", "100")
    assert cfg["train"]["steps"] == 100 and cfg["train"]["train_tokens"] is None  # run file had train_tokens
    cfg = compose("--train-tokens", "1e6")
    assert cfg["train"]["train_tokens"] == 1e6 and cfg["train"]["steps"] is None
    with pytest.raises(SystemExit):
        compose("--steps", "1.5")


def test_set_after_flags_and_conflicts():
    cfg = compose("--set", "train.lr=1e-3", "train.lr=2e-3")  # --set: later wins
    assert cfg["train"]["lr"] == 2e-3
    assert compose("--lr", "1e-3", "--set", "train.lr=1e-3")["train"]["lr"] == 1e-3  # same value: fine
    with pytest.raises(ValueError, match="train.lr"):
        compose("--lr", "1e-3", "--set", "train.lr=2e-3")
    with pytest.raises(ValueError, match="overlaps"):
        compose("--warmup", "0.05", "--set", "train.schedule={name: constant}")
    with pytest.raises(ValueError, match="train.steps"):  # --train-tokens also sets steps=null
        compose("--train-tokens", "1e6", "--set", "train.steps=10")


def test_composition_is_deterministic():
    argv = ["--lr", "3e-3", "--set", "model.dc.router=end_mlp", "--seed", "1"]
    assert compose(*argv) == compose(*argv)
    assert compose(*argv)["model"]["dc"]["router"] == "end_mlp"


def test_every_flag_is_documented_in_help(capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(["--help"])
    out = capsys.readouterr().out
    for f in cli.FLAGS:
        assert f.name in out and f.key in out, f.name
    assert "precedence" in out and "--print-config" in out


def test_print_config_prints_composed_yaml_and_exits(tmp_path, capsys):
    out_dir = tmp_path / "never"
    cfg = main(["--config", TINY_HNET, "--lr", "4e-3", "--out", str(out_dir), "--print-config"])
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed == cfg and printed["train"]["lr"] == 4e-3
    assert not out_dir.exists()


def test_model_swap_keeps_run_level_model_keys(tmp_path):
    m = os.path.join(REPO, "configs", "models", "tiny_hnet2_2m_end_mlp.yaml")
    cfg = compose("--model", m)
    assert cfg["model"]["dc"]["router"] == "end_mlp"
    assert cfg["model"]["lr_multiplier_n_gpt"] == "data"  # the run file's local model key still applies
    assert cfg["data"]["batch_bytes"] == 131072
    run = tmp_path / "run.yaml"
    run.write_text(yaml.safe_dump({"model": {"type": "hnet", "d_model": [8]}, "train": {"steps": 1}}))
    with pytest.raises(ValueError, match="no `model: {include"):
        load_run_config(str(run), model=m)


# ---------------------------------------------------------------- checkpoint / resume hardening (in-process)
def toy(out, **train):
    return load_run_config(base={
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 16, "n_layers": 2},
        "data": {"source": "synthetic", "seq_len": 32, "micro_batch": 2, "batch_windows": 2},
        "train": {"out_dir": str(out), "steps": 12, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "schedule": {"name": "wsd", "warmup": 2, "decay": 3}, **train},
        "eval": {"every": 0, "at_end": False}, "log": {"every": 1}})


def test_resume_strict_refuses_changed_hyperparameters(tmp_path):
    Trainer(toy(tmp_path, stop_at_step=5)).train()
    with pytest.raises(ValueError, match="train.lr"):
        Trainer(toy(tmp_path, lr=1e-3))
    tr = Trainer(toy(tmp_path, lr=1e-3, resume_strict=False))  # explicit opt-in
    assert tr.step == 5
    tr = Trainer(toy(tmp_path, ckpt_keep=5, ckpt_every=3))  # non-semantic keys may change
    assert tr.step == 5


def test_explicit_resume_path_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        Trainer(toy(tmp_path / "a", resume=str(tmp_path / "missing")))
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        Trainer(toy(tmp_path / "b", resume=str(tmp_path / "empty")))
    assert Trainer(toy(tmp_path / "c", resume=None)).step == 0
    assert ckpt.resolve_resume("none", str(tmp_path)) is None


def test_permanent_checkpoints_by_tokens_and_steps(tmp_path):
    # 2 windows x 32 bytes / 4.6 bytes per token ~ 13.9 tok-eq per step
    Trainer(toy(tmp_path, ckpt_every=2, ckpt_keep=1, ckpt_steps=[3], ckpt_tokens=[50.0])).train()
    kept = {s for s, p in ckpt.list_checkpoints(str(tmp_path)) if os.path.exists(os.path.join(p, "KEEP"))}
    steps = [s for s, _ in ckpt.list_checkpoints(str(tmp_path))]
    assert kept == {3, 4, 12}, steps  # step 4 is the first with tokens_eq >= 50; 12 = final
    assert steps == [3, 4, 10, 12]  # + the newest rolling one (keep 1)
    with pytest.raises(ValueError, match="ckpt_keep"):
        ckpt.prune_checkpoints(str(tmp_path), 0)


def test_divergence_stops_run_and_writes_marker(tmp_path):
    tr = Trainer(toy(tmp_path, steps=30, stop_on_nonfinite=2))
    with torch.no_grad():
        tr.raw.lm_head.weight.fill_(float("nan"))
    tr.train()
    assert tr.step == 2  # two consecutive non-finite log records (log every step)
    d = json.loads((tmp_path / "diverged.json").read_text())
    assert d["step"] == 2 and not math.isfinite(d["loss"])
    assert ckpt.latest_checkpoint(str(tmp_path)) is None  # no checkpoint of a diverged model
