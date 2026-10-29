"""Sweep specs v2 (scripts/search/spec.py): grid x seeds expansion, names, order, sharding, parameters, missing
configs, model swap, and the repo's own specs."""

import os
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "search"))

from spec import Spec, SpecError, compose_config, parse_shard, status_filename  # noqa: E402

BPE = "configs/runs/tiny_bpe_2m.yaml"
HNET = "configs/runs/tiny_hnet2_2m.yaml"


def write_spec(tmp_path, **kw):
    d = {"name": "t", "out_root": str(tmp_path / "out"), "seeds": [0, 1],
         "runs": [{"name": "bpe", "config": BPE}, {"name": "end_mlp", "config": HNET,
                                                   "set": ["model.dc.router=end_mlp"]}]}
    d.update(kw)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(p)


def test_grid_expansion_names_and_order(tmp_path):
    p = write_spec(tmp_path, grid={"train.lr": ["5e-4", 1.5e-3]}, set=["train.compile=false"])
    jobs = Spec.load(p).jobs()
    assert [j.name for j in jobs] == [
        "bpe__lr0.0005_s0", "bpe__lr0.0015_s0", "end_mlp__lr0.0005_s0", "end_mlp__lr0.0015_s0",
        "bpe__lr0.0005_s1", "bpe__lr0.0015_s1", "end_mlp__lr0.0005_s1", "end_mlp__lr0.0015_s1"]  # seeds outermost
    j = jobs[2]
    assert j.arch == "end_mlp" and j.point == {"train.lr": 5e-4} and j.seed == 0  # "5e-4" (YAML 1.1 str) -> float
    assert j.sets == ["train.compile=false", "model.dc.router=end_mlp", "train.lr=0.0005", "train.seed=0",
                      f"train.out_dir={tmp_path / 'out' / 'end_mlp__lr0.0005_s0'}"]
    cfg = compose_config(j)
    assert cfg["train"]["lr"] == 5e-4 and cfg["model"]["dc"]["router"] == "end_mlp" and cfg["train"]["seed"] == 0
    assert [x.index for x in jobs] == list(range(8))
    assert j.trainer_args()[:4] == ["-m", "hnet.train.train", "--config", HNET]


def test_no_grid_keeps_v1_names(tmp_path):
    jobs = Spec.load(write_spec(tmp_path)).jobs()
    assert [j.name for j in jobs] == ["bpe_s0", "end_mlp_s0", "bpe_s1", "end_mlp_s1"] and jobs[0].setting == "-"


def test_shards_partition_the_jobs_deterministically(tmp_path):
    sp = Spec.load(write_spec(tmp_path, grid={"train.lr": [1e-3, 2e-3, 3e-3]}, seeds=[0, 1, 2]))
    all_names = [j.name for j in sp.jobs()]
    shards = [[j.name for j in sp.jobs(shard=(i, 4))] for i in range(4)]
    assert sorted(sum(shards, [])) == sorted(all_names) and sum(map(len, shards)) == len(all_names)
    assert all(abs(len(s) - len(all_names) / 4) < 1 for s in shards)
    assert shards == [[j.name for j in Spec.load(sp.path).jobs(shard=(i, 4))] for i in range(4)]  # reproducible
    only = [j.name for j in sp.jobs(shard=(1, 4), only={"bpe"})]  # only filters within the shard
    assert only == [n for n in shards[1] if n.startswith("bpe_")]
    assert parse_shard("2/5") == (2, 5)
    for bad in ("5/5", "1-4", "0/0"):
        with pytest.raises(SpecError):
            parse_shard(bad)
    assert status_filename((1, 4)) == "sweep_status.shard-1-of-4.json" and status_filename() == "sweep_status.json"


def test_params_substitution_and_missing_params(tmp_path):
    runs = [{"name": "bpe", "config": BPE, "set": ["train.lr=${best_lr.bpe}"]},
            {"name": "end_mlp", "config": HNET, "set": ["train.lr=${best_lr.end_mlp}", "model.dc.router=${router}"]}]
    p = write_spec(tmp_path, runs=runs, params={"best_lr": {"bpe": None, "end_mlp": None}, "router": "end_mlp"})
    with pytest.raises(SpecError, match=r"best_lr.bpe', 'best_lr.end_mlp'"):  # every unset parameter, at once
        Spec.load(p).jobs()
    pf = tmp_path / "best.yaml"
    pf.write_text(yaml.safe_dump({"best_lr": {"bpe": 3e-3, "end_mlp": 1e-3}}))
    jobs = Spec.load(p, params=["best_lr.end_mlp=6e-3"], params_file=str(pf)).jobs()  # --param beats the file
    assert "train.lr=0.003" in jobs[0].sets and "train.lr=0.006" in jobs[1].sets
    assert "model.dc.router=end_mlp" in jobs[1].sets


def test_missing_configs_are_listed_or_skipped(tmp_path):
    runs = [{"name": "bpe", "config": BPE},
            {"name": "x", "config": BPE, "model": "configs/models/does_not_exist_a.yaml"},
            {"name": "y", "config": "configs/runs/does_not_exist_b.yaml"}]
    p = write_spec(tmp_path, runs=runs)
    with pytest.raises(SpecError) as e:
        Spec.load(p).jobs()
    assert "does_not_exist_a.yaml  (run x)" in str(e.value) and "does_not_exist_b.yaml  (run y)" in str(e.value)
    sp = Spec.load(p)
    jobs = sp.jobs(allow_missing=True)
    assert {j.arch for j in jobs} == {"bpe"} and [r for r, _ in sp.missing] == ["x", "y"]


def test_model_swap_job(tmp_path):
    runs = [{"name": "hnet_mlp", "config": HNET, "model": "configs/models/tiny_hnet2_2m_end_mlp.yaml"}]
    cfg = compose_config(Spec.load(write_spec(tmp_path, runs=runs)).jobs()[0])
    assert cfg["model"]["dc"]["router"] == "end_mlp" and cfg["model"]["lr_multiplier_n_gpt"] == "data"


def test_spec_validation_errors(tmp_path):
    with pytest.raises(SpecError, match="unknown spec keys"):
        Spec.load(write_spec(tmp_path, seed=[0]))
    with pytest.raises(SpecError, match="unknown keys"):
        Spec.load(write_spec(tmp_path, runs=[{"name": "a", "config": BPE, "sets": []}]))
    with pytest.raises(SpecError, match="duplicate"):
        Spec.load(write_spec(tmp_path, runs=[{"name": "a", "config": BPE}, {"name": "a", "config": BPE}]))
    with pytest.raises(SpecError, match="fairness mode"):
        Spec.load(write_spec(tmp_path, fairness="loose"))
    with pytest.raises(SpecError, match="aliases collide"):
        Spec.load(write_spec(tmp_path, grid={"train.lr": [1], "train.muon.lr": [2]})).jobs()


def test_v1_expand_compat(tmp_path):
    from sweep import expand

    jobs = expand(yaml.safe_load(open(write_spec(tmp_path))), only=["bpe"])
    assert [j["name"] for j in jobs] == ["bpe_s0", "bpe_s1"] and jobs[0]["config"] == BPE


@pytest.mark.parametrize("name,n_jobs", [("exp002_tiny2m", 18), ("exp003_lr_adamw", 60)])
def test_repo_specs_expand(name, n_jobs):
    sp = Spec.load(os.path.join(REPO, "configs", "search", f"{name}.yaml"))
    assert len(sp.jobs()) == n_jobs


def test_repo_specs_with_params():
    params = ["best_lr.bpe=1.5e-3", "best_lr.paper=1.5e-3", "best_lr.end_mlp=3e-3", "best_lr.end_cosine_shift=3e-3"]
    sp = Spec.load(os.path.join(REPO, "configs", "search", "exp004_muon.yaml"), params)
    assert len(sp.jobs()) == 4 * 3 * 3
    with pytest.raises(SpecError, match="parameters not set"):
        Spec.load(os.path.join(REPO, "configs", "search", "exp004_muon.yaml")).jobs()
    sp5 = Spec.load(os.path.join(REPO, "configs", "search", "exp005_attn_variants.yaml"),
                    ["lr.bpe=1.5e-3", "lr.hnet=1.5e-3", "hnet_router=end_mlp"])
    jobs = sp5.jobs(allow_missing=True)
    assert len(sp5.run_names()) == 16 and len(jobs) == 3 * (16 - len(sp5.missing))
