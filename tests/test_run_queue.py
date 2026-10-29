"""scripts/ladder/run_queue.py: GPU packing, done markers / resume, retries, export (CPU only, shell jobs)."""

import importlib.util
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "ladder" / "run_queue.py")
rq = importlib.util.module_from_spec(_spec)
sys.modules["run_queue"] = rq
_spec.loader.exec_module(rq)


def _queue(tmp_path, runs, **top):
    q = {"sweep": "t", "out_root": str(tmp_path / "sweep"), "runs": runs, **top}
    p = tmp_path / "q.yaml"
    p.write_text(yaml.safe_dump(q))
    return str(p)


def _job(name, extra=""):
    # records its GPU assignment and the number of concurrently running jobs on that GPU
    cmd = (f"mkdir -p {{out_dir}} && echo $CUDA_VISIBLE_DEVICES > {{out_dir}}/gpu && "
           f"touch {{out_dir}}/../running_{name}_$CUDA_VISIBLE_DEVICES && sleep 0.3 && "
           f"ls {{out_dir}}/.. | grep -c running_.*_$CUDA_VISIBLE_DEVICES > {{out_dir}}/conc; "
           f"rm {{out_dir}}/../running_{name}_$CUDA_VISIBLE_DEVICES; {extra} true")
    return {"name": name, "cmd": cmd}


def test_packing_and_resume(tmp_path):
    q = _queue(tmp_path, [_job(f"j{i}") for i in range(6)])
    argv = ["--queue", q, "--gpus", "0,1", "--per-gpu", "3", "--poll", "0.05"]
    assert rq.main(argv) == 0
    root = tmp_path / "sweep"
    gpus = [int((root / f"j{i}" / "gpu").read_text()) for i in range(6)]
    assert sorted(set(gpus)) == [0, 1]
    assert max(int((root / f"j{i}" / "conc").read_text()) for i in range(6)) <= 3
    for i in range(6):
        assert json.loads((root / f"j{i}" / "done.json").read_text())["rc"] == 0
    status = json.loads((root / "queue_status.json").read_text())["runs"]
    assert all(r["status"] == "done" for r in status.values())
    # a second launch skips finished runs
    (root / "j0" / "gpu").unlink()
    assert rq.main(argv) == 0
    assert not (root / "j0" / "gpu").exists()


def test_whole_gpu_runs_one_per_gpu(tmp_path):
    q = _queue(tmp_path, [_job(f"j{i}") for i in range(4)])
    assert rq.main(["--queue", q, "--gpus", "0,1", "--poll", "0.05"]) == 0
    root = tmp_path / "sweep"
    assert all(int((root / f"j{i}" / "conc").read_text()) == 1 for i in range(4))


def test_retry_then_fail(tmp_path):
    flaky = {"name": "flaky", "cmd": "mkdir -p {out_dir}; f={out_dir}/n; echo x >> $f; "
                                     "[ $(wc -l < $f) -ge 2 ]"}  # fails on the first attempt only
    bad = {"name": "bad", "cmd": "exit 7"}
    q = _queue(tmp_path, [flaky, bad, _job("ok")])
    assert rq.main(["--queue", q, "--gpus", "0", "--max-attempts", "2", "--poll", "0.05"]) == 1
    status = json.loads((tmp_path / "sweep" / "queue_status.json").read_text())["runs"]
    assert status["flaky"]["status"] == "done" and status["flaky"]["attempts"] == 2
    assert status["bad"]["status"] == "failed rc=7" and status["bad"]["attempts"] == 2
    assert status["ok"]["status"] == "done"


def test_trainer_command_and_export(tmp_path, monkeypatch):
    q = _queue(tmp_path, [{"name": "a", "config": "configs/x.yaml", "set": ["train.steps=3"], "gpus": 2}],
               defaults={"set": ["train.compile=blocks"]})
    out_root, jobs = rq.load_queue(q, extra_sets=["log.every=1"])
    (j,) = jobs
    assert j.argv[1:4] == ["-m", "torch.distributed.run", "--standalone"]
    assert j.argv[-8:] == ["--set", "train.compile=blocks", "--set", "train.steps=3", "--set", "log.every=1",
                           "--set", f"train.out_dir={out_root}/a"]
    run = tmp_path / "runs" / "sw" / "r1"
    (run / "ckpt" / "step_0000010").mkdir(parents=True)
    (run / "ckpt" / "step_0000010" / "model.pt").write_text("x")
    (run / "metrics.jsonl").write_text("{}\n")
    (run / "config.yaml").write_text("a: 1\n")
    (tmp_path / "runs" / "sw" / "queue_status.json").write_text("{}")
    monkeypatch.chdir(tmp_path)
    dst = rq.export("runs/sw")
    assert dst == Path("runs/_export/sw")
    files = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
    assert files == ["queue_status.json", "r1/config.yaml", "r1/metrics.jsonl"]
