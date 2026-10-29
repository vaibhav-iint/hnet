"""scripts/search/sweep.py end to end on CPU: grid x seeds jobs train through the real trainer (toy model,
synthetic data), a diverging LR is recorded as `diverged`, status/manifest/markers are written, a re-launch skips
finished jobs, sharding selects jobs, and GCS restore/mirror go through gcloud (mocked)."""

import json
import os
import subprocess
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts", "search"))

import sweep  # noqa: E402


@pytest.fixture
def toy_spec(tmp_path, monkeypatch):
    run = {"model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 16, "n_layers": 2},
           "data": {"source": "synthetic", "seq_len": 32, "micro_batch": 2, "batch_windows": 2},
           "train": {"steps": 4, "precision": "fp32", "device": "cpu", "ckpt_every": 0, "stop_on_nonfinite": 1,
                     "schedule": {"name": "constant"}},
           "eval": {"every": 0, "at_end": True}, "log": {"every": 1}}
    (tmp_path / "toy.yaml").write_text(yaml.safe_dump(run))
    spec = {"name": "toy", "out_root": str(tmp_path / "out"), "seeds": [0], "fairness": "tuned",
            "grid": {"train.lr": [3e-3, 1e30]},
            "runs": [{"name": "a", "config": str(tmp_path / "toy.yaml")},
                     {"name": "b", "config": str(tmp_path / "toy.yaml"), "set": ["model.n_layers=4"]}]}
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    # the toy model has no FLOPs model / chunk diagnostics: stub the gate and diag (tested elsewhere)
    calls = {"fairness": []}
    fake = type(sys)("fairness")
    fake.main = lambda argv: calls["fairness"].append(argv) or 0
    monkeypatch.setitem(sys.modules, "fairness", fake)
    monkeypatch.setattr(sweep, "diag_texts", lambda: [])
    monkeypatch.setattr(sweep, "run_diag", lambda out, texts: {})
    return str(p), tmp_path / "out", calls


def test_sweep_runs_grid_records_divergence_and_resumes(toy_spec):
    p, out, calls = toy_spec
    assert sweep.main([p, "--parallel", "2"]) == 0
    status = json.loads((out / "sweep_status.json").read_text())["runs"]
    assert status == {"a__lr0.003_s0": "done", "a__lr1e+30_s0": "diverged", "b__lr0.003_s0": "done",
                      "b__lr1e+30_s0": "diverged"}
    assert json.loads((out / "a__lr1e+30_s0" / "done.json").read_text())["diverged"] is True
    assert (out / "a__lr1e+30_s0" / "diverged.json").exists()
    job = json.loads((out / "b__lr0.003_s0" / "sweep_job.json").read_text())
    assert job["arch"] == "b" and job["point"] == {"train.lr": 0.003} and job["seed"] == 0
    cfg = yaml.safe_load((out / "b__lr0.003_s0" / "config.yaml").read_text())
    assert cfg["model"]["n_layers"] == 4 and cfg["train"]["lr"] == 0.003
    manifest = json.loads((out / "sweep_manifest.json").read_text())
    assert [j["name"] for j in manifest["jobs"]] == list(status)
    assert calls["fairness"] and calls["fairness"][0][0] == p
    mtime = (out / "a__lr0.003_s0" / "metrics.jsonl").stat().st_mtime
    assert sweep.main([p]) == 0  # re-launch: everything finished, nothing retrained
    assert (out / "a__lr0.003_s0" / "metrics.jsonl").stat().st_mtime == mtime


def test_list_shard_and_dry_run(toy_spec, capsys):
    p, out, _ = toy_spec
    assert sweep.main([p, "--list", "--shard", "1/2"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [ln.split()[1] for ln in lines] == ["a__lr1e+30_s0", "b__lr1e+30_s0"]
    assert sweep.main([p, "--dry-run", "--only", "b"]) == 0
    cmds = capsys.readouterr().out.strip().splitlines()
    assert len(cmds) == 2 and all("hnet.train.train" in c and "model.n_layers=4" in c for c in cmds)
    assert not out.exists()


def test_failed_job_stops_the_sweep(toy_spec, tmp_path):
    p, out, _ = toy_spec
    d = yaml.safe_load(open(p))
    d["runs"][0]["set"] = ["model.d_model=not_a_number"]
    open(p, "w").write(yaml.safe_dump(d))
    assert sweep.main([p, "--shard", "0/2"]) == 1
    status = json.loads((out / "sweep_status.shard-0-of-2.json").read_text())["runs"]
    assert status["a__lr0.003_s0"].startswith("failed") and status["b__lr0.003_s0"] == "skipped (earlier failure)"


def test_gcs_restore_and_uploads_are_scoped_to_the_selection(toy_spec, monkeypatch):
    p, out, _ = toy_spec
    calls = []

    def fake_gcloud(args):
        calls.append(args)
        if args[:2] == ["storage", "ls"]:
            return subprocess.CompletedProcess(args, 0, "gs://b/toy/a__lr0.003_s0/\ngs://b/toy/b__lr0.003_s0/\n"
                                                        "gs://b/toy/fairness.md\n", "")
        if args[:2] == ["storage", "rsync"] and args[-1].endswith("a__lr0.003_s0") and args[-2].startswith("gs://"):
            os.makedirs(args[-1], exist_ok=True)  # "download" a finished run
            with open(os.path.join(args[-1], "done.json"), "w") as f:
                f.write('{"rc": 0}')
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(sweep, "gcloud", fake_gcloud)
    assert sweep.main([p, "--gcs", "gs://b/toy", "--restore", "--shard", "0/2"]) == 0
    downloads = [c for c in calls if c[:2] == ["storage", "rsync"] and c[-2].startswith("gs://")]
    assert [c[-2] for c in downloads] == ["gs://b/toy/a__lr0.003_s0", "gs://b/toy/b__lr0.003_s0"]  # shard 0 only
    uploads = [c[-1] for c in calls if c[:2] == ["storage", "rsync"] and c[-1].startswith("gs://")]
    assert "gs://b/toy/b__lr0.003_s0" in uploads and not any("1e+30" in u for u in uploads)  # shard 1 untouched
    status = json.loads((out / "sweep_status.shard-0-of-2.json").read_text())["runs"]
    assert status == {"a__lr0.003_s0": "done", "b__lr0.003_s0": "done"}
    assert not (out / "a__lr0.003_s0" / "metrics.jsonl").exists()  # restored as finished: not retrained
    assert any(c[:2] == ["storage", "cp"] and c[-1].endswith("sweep_status.shard-0-of-2.json") for c in calls)


def test_failed_upload_raises(toy_spec, monkeypatch):
    p, out, _ = toy_spec
    monkeypatch.setattr(sweep, "gcloud", lambda args: subprocess.CompletedProcess(args, 1, "", "AccessDenied"))
    with pytest.raises(RuntimeError, match="AccessDenied"):
        sweep.main([p, "--gcs", "gs://b/toy", "--only", "a", "--shard", "0/2"])
