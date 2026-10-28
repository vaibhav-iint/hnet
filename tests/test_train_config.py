import os

import yaml

from hnet.train.config import load_run_config


def _w(path, d):
    with open(path, "w") as f:
        yaml.safe_dump(d, f)
    return str(path)


def test_include_composition_and_overrides(tmp_path):
    (tmp_path / "models").mkdir()
    _w(tmp_path / "models" / "m.yaml", {"model": {"type": "hnet", "d_model": [1, 2], "dc": {"router": "a",
                                                                                             "ste": True}},
                                         "matching": {"x": 1}})
    _w(tmp_path / "base.yaml", {"train": {"lr": 1e-3, "steps": 10}, "data": {"seq_len": 4096}})
    run = _w(tmp_path / "run.yaml", {
        "include": "base.yaml",
        "model": {"include": "models/m.yaml", "dc": {"router": "b"}},
        "train": {"steps": 20},
    })
    cfg = load_run_config(run, ["train.lr=3e-4", "model.d_model=[4,5]"])
    assert cfg["model"]["dc"] == {"router": "b", "ste": True}
    assert cfg["model"]["d_model"] == [4, 5]
    assert "matching" not in cfg["model"]
    assert cfg["train"]["steps"] == 20 and cfg["train"]["lr"] == 3e-4
    assert cfg["data"]["seq_len"] == 4096 and cfg["data"]["micro_batch"] == 1  # default filled
    assert cfg["train"]["optimizer"] == "adamw"


def test_repo_run_configs_compose():
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "runs")
    if not os.path.isdir(root):
        return
    for n in sorted(os.listdir(root)):
        if n.endswith(".yaml"):
            cfg = load_run_config(os.path.join(root, n))
            assert cfg["model"].get("type") in ("hnet", "dense"), n
            assert cfg["train"]["out_dir"], n
