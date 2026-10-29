"""infra/sweeps/launch.py with subprocess mocked (nothing reaches gcloud or git): create (print / execute with
zone fallback, budget cap, commit checks), status (GCS sweep_status of every shard), teardown; plus the VM
startup script's shard handling."""

import json
import os
import subprocess
import sys

import pytest
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "infra", "sweeps"), os.path.join(REPO, "scripts", "search")]

import launch  # noqa: E402

SHA = "a" * 40


class FakeSubprocess:
    """Records every command; answers git and gcloud like a pushed repo and a GCP project would."""

    def __init__(self, create_errors=None, vms=None, status_files=None, remote_branch="origin/main"):
        self.calls, self.create_errors = [], dict(create_errors or {})
        self.vms, self.status_files, self.remote_branch = vms or [], status_files or {}, remote_branch

    def __call__(self, cmd, capture_output=True, text=True, env=None, cwd=None):
        self.calls.append((cmd, env))
        ok = lambda out="": subprocess.CompletedProcess(cmd, 0, out, "")  # noqa: E731
        if cmd[:2] == ["git", "rev-parse"]:
            return ok(SHA + "\n")
        if cmd[:3] == ["git", "branch", "-r"]:
            return ok(self.remote_branch)
        if cmd[:2] == ["git", "diff"]:
            return ok()
        if cmd[:4] == ["gcloud", "compute", "instances", "create"]:
            zone = next(a for a in cmd if a.startswith("--zone=")).split("=", 1)[1]
            err = self.create_errors.get((cmd[4], zone))
            return subprocess.CompletedProcess(cmd, 1, "", err) if err else ok("[]")
        if cmd[:4] == ["gcloud", "compute", "instances", "list"]:
            return ok(json.dumps(self.vms))
        if cmd[:3] == ["gcloud", "storage", "ls"]:
            if not self.status_files:
                return subprocess.CompletedProcess(cmd, 1, "", "ERROR: One or more URLs matched no objects.")
            return ok("\n".join(f"gs://b/exp/{n}" for n in self.status_files))
        if cmd[:3] == ["gcloud", "storage", "cat"]:
            return ok(json.dumps(self.status_files[cmd[3].rsplit("/", 1)[-1]]))
        if cmd[:4] == ["gcloud", "compute", "instances", "delete"]:
            return ok()
        raise AssertionError(f"unexpected command {cmd}")

    def creates(self):
        return [c for c, _ in self.calls if c[:4] == ["gcloud", "compute", "instances", "create"]]


@pytest.fixture
def spec_path(tmp_path):
    d = {"name": "exp_t", "out_root": str(tmp_path / "out"), "seeds": [0, 1], "fairness": "tuned",
         "grid": {"train.lr": [1e-3, 3e-3]},
         "runs": [{"name": "bpe", "config": "configs/runs/tiny_bpe_2m.yaml"},
                  {"name": "paper", "config": "configs/runs/tiny_hnet2_2m.yaml"}]}
    p = tmp_path / "exp_t.yaml"
    p.write_text(yaml.safe_dump(d))
    return str(p)


@pytest.fixture
def fake(monkeypatch):
    f = FakeSubprocess()
    monkeypatch.setattr(launch.subprocess, "run", f)
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "SECRET-TOKEN-VALUE")
    return f


def create(spec_path, *extra):
    return launch.main(["create", spec_path, "--vms", "3", "--no-fairness", "--max-hours", "4", *extra])


def test_create_prints_one_sharded_spot_vm_per_shard(spec_path, fake, capsys):
    assert create(spec_path) == 0
    out = capsys.readouterr().out
    assert "SECRET-TOKEN-VALUE" not in out and "unset CLOUDSDK_AUTH_ACCESS_TOKEN" in out
    cmds = [ln for ln in out.splitlines() if ln.startswith("gcloud compute instances create")]
    assert len(cmds) == 3 and not fake.creates()  # printed, not run
    for i, c in enumerate(cmds):
        assert f"hx-exp-t-{i}of3 " in c and f"hnet-shard={i}/3" in c and f"hnet-shard={i}of3" in c
        assert f"hnet-commit={SHA}" in c and "hnet-spec=" + spec_path in c and "hnet-sweep=exp-t" in c
        assert "--provisioning-model=SPOT" in c and "--max-run-duration=14400s" in c
        assert "startup-script=infra/sweeps/sweep_vm_startup.sh" in c and "--zone=us-central1-a" in c
    assert "shard 0: 3, shard 1: 3, shard 2: 2" in out and "worst-case cost $" in out


def test_create_execute_falls_back_across_zones(spec_path, fake, capsys):
    fake.create_errors = {("hx-exp-t-1of3", "us-central1-a"): "ZONE_RESOURCE_POOL_EXHAUSTED: no L4 here",
                          ("hx-exp-t-2of3", "us-central1-a"): "The resource 'hx-exp-t-2of3' already exists"}
    assert create(spec_path, "--execute") == 0
    zones = [(c[4], next(a for a in c if a.startswith("--zone="))) for c in fake.creates()]
    assert zones == [("hx-exp-t-0of3", "--zone=us-central1-a"), ("hx-exp-t-1of3", "--zone=us-central1-a"),
                     ("hx-exp-t-1of3", "--zone=us-central1-b"), ("hx-exp-t-2of3", "--zone=us-central1-a")]
    out = capsys.readouterr().out
    assert "already exists; left alone" in out
    for cmd, env in fake.calls:
        if cmd[0] == "gcloud":
            assert "CLOUDSDK_AUTH_ACCESS_TOKEN" not in env and env["CLOUDSDK_CORE_DISABLE_PROMPTS"] == "1"


def test_create_execute_raises_on_other_errors(spec_path, fake, capsys):
    fake.create_errors = {("hx-exp-t-0of3", "us-central1-a"): "PERMISSION_DENIED: compute.instances.create"}
    assert create(spec_path, "--execute") == 2
    assert "PERMISSION_DENIED" in capsys.readouterr().err and len(fake.creates()) == 1


def test_create_refusals(spec_path, fake, capsys):
    assert create(spec_path, "--budget-cap", "0.5") == 2
    assert "exceeds --budget-cap" in capsys.readouterr().err
    fake.remote_branch = ""
    assert create(spec_path) == 2
    assert "not on any remote branch" in capsys.readouterr().err
    assert launch.main(["create", spec_path, "--vms", "99", "--no-fairness"]) == 2


def test_status_reads_every_shard_and_suggests_restart(spec_path, fake, capsys):
    fake.vms = [{"name": "hx-exp-t-0of2", "zone": "zones/us-central1-a", "status": "RUNNING",
                 "labels": {"hnet-shard": "0of2"}},
                {"name": "hx-exp-t-1of2", "zone": "zones/us-east1-c", "status": "TERMINATED",
                 "labels": {"hnet-shard": "1of2"}}]
    fake.status_files = {
        "sweep_status.shard-0-of-2.json": {"updated": "T0", "runs": {"bpe__lr0.001_s0": "done",
                                                                     "paper__lr0.001_s0": "running"}},
        "sweep_status.shard-1-of-2.json": {"updated": "T1", "runs": {"bpe__lr0.003_s0": "diverged"}}}
    assert launch.main(["status", spec_path, "--gcs", "gs://b/exp"]) == 0
    out = capsys.readouterr().out
    assert "8 jobs" in out and "done=1" in out and "running=1" in out and "diverged=1" in out
    assert "gcloud compute instances start hx-exp-t-1of2 --zone=us-east1-c" in out
    assert not [c for c, _ in fake.calls if c[:4] == ["gcloud", "compute", "instances", "start"]]  # only printed


def test_status_before_anything_ran(spec_path, fake, capsys):
    assert launch.main(["status", spec_path]) == 0
    out = capsys.readouterr().out
    assert "not started=8" in out and "no VMs labelled" in out


def test_teardown_prints_then_executes(spec_path, fake, capsys):
    fake.vms = [{"name": "hx-exp-t-0of2", "zone": "zones/us-central1-a", "status": "RUNNING", "labels": {}}]
    assert launch.main(["teardown", spec_path]) == 0
    assert "gcloud compute instances delete hx-exp-t-0of2 --zone=us-central1-a --quiet" in capsys.readouterr().out
    assert not [c for c, _ in fake.calls if "delete" in c]
    assert launch.main(["teardown", spec_path, "--execute"]) == 0
    assert [c[4] for c, _ in fake.calls if c[:4] == ["gcloud", "compute", "instances", "delete"]] == ["hx-exp-t-0of2"]
    list_cmd = next(c for c, _ in fake.calls if c[:4] == ["gcloud", "compute", "instances", "list"])
    assert "--filter=labels.hnet-sweep=exp-t" in list_cmd


def test_vm_names_are_valid_gce_names():
    import re

    for name in ("exp003_lr_adamw", "EXP_005 attn.variants", "x" * 80):
        n = launch.vm_name(name, 11, 12)
        assert re.fullmatch(r"[a-z]([-a-z0-9]{0,61}[a-z0-9])?", n), n


def test_startup_script_passes_shard_to_sweep():
    path = os.path.join(REPO, "infra", "sweeps", "sweep_vm_startup.sh")
    assert subprocess.run(["bash", "-n", path]).returncode == 0
    text = open(path).read()
    assert "attr hnet-shard" in text and 'ARGS+=(--shard "$SHARD")' in text
    assert "--restore --mirror-every" in text and "set -euo pipefail" in text
