"""Fairness gate (scripts/search/fairness.py): strict vs tuned modes, grid identity, per-architecture keys with
provenance, FLOPs mismatch, missing configs, and the repo's EXP-002..005 specs."""

import json
import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "search"))

from fairness import main as fairness  # noqa: E402

BPE = "configs/runs/tiny_bpe_2m.yaml"
HNET = "configs/runs/tiny_hnet2_2m.yaml"


def spec(tmp_path, **kw):
    d = {"name": "f", "out_root": str(tmp_path / "out"), "seeds": [0],
         "runs": [{"name": "bpe", "config": BPE}, {"name": "paper", "config": HNET}]}
    d.update(kw)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(p)


def problems(tmp_path):
    return json.loads((tmp_path / "out" / "fairness.json").read_text())["problems"]


def test_strict_passes_without_grid_and_fails_with_lr_grid(tmp_path):
    assert fairness([spec(tmp_path)]) == 0
    p = spec(tmp_path, grid={"train.lr": [1e-3, 3e-3]})
    assert fairness([p]) == 1
    assert any("use fairness mode tuned" in x for x in problems(tmp_path))
    assert fairness([p, "--mode", "tuned"]) == 0


def test_strict_catches_a_held_field_and_flops(tmp_path):
    runs = [{"name": "bpe", "config": BPE}, {"name": "paper", "config": HNET, "set": ["train.lr=3e-3"]},
            {"name": "big", "config": HNET, "set": ["model.d_intermediate=[0, 384, 2048]"]}]
    assert fairness([spec(tmp_path, runs=runs)]) == 1
    pr = problems(tmp_path)
    assert any(x.startswith("paper: optimizer differs") for x in pr)
    assert any(x.startswith("big: gflops_per_byte") for x in pr)


def test_tuned_requires_identical_grids_and_seeds(tmp_path):
    runs = [{"name": "bpe", "config": BPE, "grid": {"train.lr": [1e-3, 3e-3]}},
            {"name": "paper", "config": HNET, "grid": {"train.lr": [1e-3, 6e-3]}, "seeds": [0, 1]}]
    assert fairness([spec(tmp_path, runs=runs, fairness="tuned")]) == 1
    pr = problems(tmp_path)
    assert any("paper: grid train.lr" in x for x in pr) and any("paper: seeds" in x for x in pr)


def test_tuned_still_holds_everything_else_equal(tmp_path):
    runs = [{"name": "bpe", "config": BPE}, {"name": "paper", "config": HNET, "set": ["train.weight_decay=0.0"]}]
    assert fairness([spec(tmp_path, runs=runs, fairness="tuned", grid={"train.lr": [1e-3, 3e-3]})]) == 1
    assert any("optimizer differs" in x and "wd=0.0" in x for x in problems(tmp_path))


def test_per_arch_keys_need_tuned_mode_and_param_provenance(tmp_path):
    runs = [{"name": "bpe", "config": BPE, "set": ["train.lr=${lr.bpe}"]},
            {"name": "paper", "config": HNET, "set": ["train.lr=${lr.paper}"]}]
    params = {"lr": {"bpe": 1.5e-3, "paper": 3e-3}}
    ok = spec(tmp_path, runs=runs, params=params, fairness={"mode": "tuned", "per_arch": ["train.lr"]})
    assert fairness([ok]) == 0
    assert "Tuned per architecture" in (tmp_path / "out" / "fairness.md").read_text()
    assert fairness([ok, "--mode", "strict"]) == 1  # per_arch requires tuned
    runs[1]["set"] = ["train.lr=3e-3"]  # literal value: no provenance
    assert fairness([spec(tmp_path, runs=runs, params=params,
                          fairness={"mode": "tuned", "per_arch": ["train.lr"]})]) == 1
    assert any("paper: per_arch key train.lr must be set" in x for x in problems(tmp_path))


def test_missing_configs_fail_unless_allowed(tmp_path):
    runs = [{"name": "bpe", "config": BPE}, {"name": "paper", "config": HNET},
            {"name": "ghost", "config": HNET, "model": "configs/models/nope_does_not_exist.yaml"}]
    p = spec(tmp_path, runs=runs)
    assert fairness([p]) == 1
    assert fairness([p, "--allow-missing"]) == 0
    assert "NOT CHECKED (config files missing)" in (tmp_path / "out" / "fairness.md").read_text()


# ---------------------------------------------------------------- the repo's experiment specs
def _repo(name):
    return os.path.join(REPO, "configs", "search", name)


def _isolated(tmp_path, name):
    """Run the gate on a repo spec but write fairness.md under tmp (out_root redirected)."""
    d = yaml.safe_load(open(_repo(name)))
    d["out_root"] = str(tmp_path / "out")
    p = tmp_path / name
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(p)


def test_exp002_passes_strict(tmp_path):
    assert fairness([_isolated(tmp_path, "exp002_tiny2m.yaml")]) == 0


def test_exp003_passes_tuned_not_strict(tmp_path):
    p = _isolated(tmp_path, "exp003_lr_adamw.yaml")
    assert fairness([p]) == 0
    assert fairness([p, "--mode", "strict"]) == 1


def test_exp004_passes_tuned_once_params_are_set(tmp_path):
    p = _isolated(tmp_path, "exp004_muon.yaml")
    assert fairness([p]) == 1  # params not filled in yet
    params = ["--param=best_lr.bpe=1.5e-3", "--param=best_lr.paper=3e-3", "--param=best_lr.end_mlp=3e-3",
              "--param=best_lr.end_cosine_shift=1.5e-3"]
    assert fairness([p, *params]) == 0


def test_exp005_passes_tuned_for_existing_configs(tmp_path):
    p = _isolated(tmp_path, "exp005_attn_variants.yaml")
    params = ["--param=lr.bpe=1.5e-3", "--param=lr.hnet=3e-3", "--param=hnet_router=end_mlp"]
    assert fairness([p, *params, "--allow-missing"]) == 0
