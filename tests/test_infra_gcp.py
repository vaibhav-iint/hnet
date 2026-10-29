"""Unit tests for infra.gcp with gcloud (subprocess) mocked: no network, no spend."""

import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from infra.gcp import cli, config as C, gce, ledger as L, serial as S, startup
from infra.gcp import vm_agent as VA
from infra.prime.state import State


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HNET_GCP_STATE_DIR", str(tmp_path / "gcp_state"))
    monkeypatch.setenv("CLOUDSDK_AUTH_ACCESS_TOKEN", "fake-token")
    monkeypatch.delenv("HNET_GCP_PRICES", raising=False)
    monkeypatch.delenv("HNET_GCP_PROJECT", raising=False)
    yield


_REAL_RUN = subprocess.run


class FakeGcloud:
    """Replaces subprocess.run for gcloud calls; `handler(args) -> (code, stdout, stderr)`."""

    def __init__(self, handler):
        self.handler, self.calls, self.envs = handler, [], []

    def __call__(self, args, **kw):
        if args[0] == "git":  # local repo queries run for real
            return _REAL_RUN(args, **kw)
        assert args[0] == "gcloud", args
        self.calls.append(args)
        self.envs.append(kw.get("env"))
        code, out, err = self.handler(args)
        return subprocess.CompletedProcess(args, code, out, err)


def _spec(tmp_path, **kw):
    p = tmp_path / "s.sh"
    p.write_text("#!/bin/bash\n")
    d = dict(name="hnet-t", machine_type="g2-standard-8", accelerator=None, gpu_count=1, spot=True,
             max_hours=0.5, disk_gb=100, startup_path=str(p))
    d.update(kw)
    return gce.CreateSpec(**d)


# ----------------------------------------------------------------------------- command construction
def test_create_args_spot_l4(tmp_path):
    a = gce.create_args(_spec(tmp_path), "us-central1-a")
    assert a[:4] == ["compute", "instances", "create", "hnet-t"]
    for flag in ["--zone=us-central1-a", "--machine-type=g2-standard-8", "--no-service-account", "--no-scopes",
                 "--max-run-duration=1800s", "--maintenance-policy=TERMINATE", "--provisioning-model=SPOT",
                 "--instance-termination-action=STOP", "--boot-disk-size=100GB",
                 f"--image-family={C.IMAGE_FAMILY}", f"--image-project={C.IMAGE_PROJECT}"]:
        assert flag in a, flag
    assert not any(x.startswith("--accelerator") for x in a)  # g2 has the L4 built in
    md = next(x for x in a if x.startswith("--metadata="))
    assert "enable-guest-attributes=TRUE" in md
    assert any(x.startswith("--metadata-from-file=startup-script=") for x in a)
    assert any("hnet-owner=infra-gcp" in x for x in a if x.startswith("--labels="))


def test_create_args_on_demand_t4_cpu(tmp_path):
    mt, acc, n = C.machine_for("t4", 1)
    a = gce.create_args(_spec(tmp_path, machine_type=mt, accelerator=acc, gpu_count=n, spot=False, max_hours=2,
                              disk_gb=50), "us-east1-c")
    assert "--accelerator=type=nvidia-tesla-t4,count=1" in a
    assert "--instance-termination-action=DELETE" in a and "--provisioning-model=STANDARD" in a
    assert "--max-run-duration=7200s" in a
    assert "--boot-disk-size=100GB" in a  # DL image minimum
    mt, acc, n = C.machine_for("cpu", 1)
    a = gce.create_args(_spec(tmp_path, machine_type=mt, accelerator=acc, gpu_count=n), "us-central1-a")
    assert mt.startswith("e2-") and "--maintenance-policy=TERMINATE" not in a
    assert C.machine_for("a100", 1)[0] == "a2-highgpu-1g"
    with pytest.raises(ValueError):
        C.machine_for("l4", 3)


def test_gcloud_env_drops_preset_token(monkeypatch):
    seen = FakeGcloud(lambda a: (0, "[]", ""))
    monkeypatch.setattr(subprocess, "run", seen)
    monkeypatch.setenv("CLOUDSDK_CONFIG", "/x/gcloud")
    gce.run(["compute", "instances", "list"])
    env = seen.envs[0]
    assert "CLOUDSDK_AUTH_ACCESS_TOKEN" not in env and env["CLOUDSDK_CONFIG"] == "/x/gcloud"
    assert seen.calls[0][-1] == "--format=json"
    monkeypatch.setenv("HNET_GCP_KEEP_TOKEN", "1")
    gce.run(["compute", "instances", "list"])
    assert seen.envs[1]["CLOUDSDK_AUTH_ACCESS_TOKEN"] == "fake-token"


# ----------------------------------------------------------------------------- zone fallback
STOCKOUT = ("ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n"
            " - The zone 'projects/p/zones/us-central1-a' does not have enough resources available to fulfill "
            "the request.  '(resource type:compute)'.\n")
QUOTA = "ERROR: (gcloud.compute.instances.create) Could not fetch resource:\n - Quota 'NVIDIA_L4_GPUS' exceeded.  Limit: 16.0 in region us-central1.\n"
NOTFOUND = "ERROR: (gcloud.compute.instances.describe) Could not fetch resource:\n - The resource 'projects/p/zones/z/instances/hnet-t' was not found\n"


def test_zone_fallback_moves_on_capacity_and_quota(tmp_path, monkeypatch):
    def h(a):
        if a[3] == "create":
            z = next(x for x in a if x.startswith("--zone=")).split("=")[1]
            if z == "us-central1-a":
                return 1, "", STOCKOUT
            if z == "us-central1-b":
                return 1, "", QUOTA
            return 0, json.dumps([{"name": "hnet-t", "status": "RUNNING"}]), ""
        if a[3] == "describe":
            return 1, "", NOTFOUND
        raise AssertionError(a)

    fake = FakeGcloud(h)
    monkeypatch.setattr(subprocess, "run", fake)
    r = gce.create_with_fallback(_spec(tmp_path), ["us-central1-a", "us-central1-b", "us-east1-b"], log=lambda *x: None)
    assert r.ok and r.zone == "us-east1-b" and len(r.tried) == 2
    assert [c[3] for c in fake.calls] == ["create", "describe", "create", "describe", "create"]


def test_zone_fallback_stops_on_hard_error_and_cleans_leftover(tmp_path, monkeypatch):
    state = {"deleted": []}

    def h(a):
        if a[3] == "create":
            return 1, "", "ERROR: (gcloud.compute.instances.create) Required 'compute.instances.create' permission denied"
        if a[3] == "describe":
            return 0, json.dumps({"name": "hnet-t", "status": "TERMINATED"}), ""
        if a[3] == "delete":
            state["deleted"].append(a[4])
            return 0, "", ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    with pytest.raises(gce.GcloudError, match="permission denied"):
        gce.create_with_fallback(_spec(tmp_path), ["z1", "z2"], log=lambda *x: None)
    assert state["deleted"] == ["hnet-t"]


def test_zone_fallback_never_touches_existing(tmp_path, monkeypatch):
    def h(a):
        if a[3] == "create":
            return 1, "", "ERROR: The resource 'projects/p/zones/z1/instances/hnet-t' already exists"
        raise AssertionError(f"unexpected call {a}")

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    with pytest.raises(gce.GcloudError, match="already exists"):
        gce.create_with_fallback(_spec(tmp_path), ["z1", "z2"], log=lambda *x: None)


def test_zone_fallback_all_zones_exhausted(tmp_path, monkeypatch):
    def h(a):
        if a[3] == "create":
            return 1, "", STOCKOUT
        if a[3] == "describe":
            return 1, "", NOTFOUND
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    with pytest.raises(gce.NoCapacity) as ei:
        gce.create_with_fallback(_spec(tmp_path), ["z1", "z2"], log=lambda *x: None)
    assert [z for z, _ in ei.value.tried] == ["z1", "z2"]


def test_retryable_classification():
    assert gce.is_retryable(STOCKOUT) and gce.is_retryable(QUOTA)
    assert gce.is_retryable("ZONE_RESOURCE_POOL_EXHAUSTED")
    assert gce.is_retryable(" - Invalid value for field 'resource.machineType': 'projects/p/zones/z/machineTypes/g2-standard-8' was not found")
    assert not gce.is_retryable(" - The resource 'projects/p/global/images/family/x' was not found")
    assert not gce.is_retryable("permission denied")


# ----------------------------------------------------------------------------- serial transport
def test_parse_serial_response_forms():
    d = gce.parse_serial_response(json.dumps({"contents": "abc\n", "start": "10", "next": "14"}), "", 10)
    assert d == {"contents": "abc\n", "start": 10, "next": 14}
    with pytest.raises(ValueError):  # we always ask for JSON; anything else is an error, not a fallback
        gce.parse_serial_response("hello\n", "", 100)


class FakeSerial:
    """A rolling serial buffer of `cap` bytes; fetch(start) behaves like the GCE API."""

    def __init__(self, cap=10_000):
        self.buf, self.total, self.cap = b"", 0, cap

    def write(self, s: str):
        b = s.encode()
        self.buf += b
        self.total += len(b)
        self.buf = self.buf[-self.cap:]

    def fetch(self, start):
        first = self.total - len(self.buf)
        s = max(start, first)
        s = min(s, self.total)
        return {"contents": self.buf[s - first:].decode(), "start": s, "next": self.total}


def test_tracker_offsets_partial_lines_and_gap(tmp_path):
    fs = FakeSerial(cap=400)
    tr = S.SerialTracker(tmp_path / "ser")
    m1 = 'HNET_METRIC runs/a {"step": 10, "bpb": 2.0}\n'
    fs.write("boot noise\n" + m1 + 'HNET_METRIC runs/a {"step": 2')  # partial line at the end
    tr.fetch(fs.fetch)
    p = S.parse_text(tr.read_raw())
    assert [r["step"] for r in p.metrics["runs/a"]] == [10] and p.corrupt == 0
    fs.write('0, "bpb": 1.9}\n')  # completes the partial line
    tr.fetch(fs.fetch)
    p = S.parse_text(tr.read_raw())
    assert [r["step"] for r in p.metrics["runs/a"]] == [10, 20]
    assert tr.cur["next"] == fs.total
    # offsets persist: a new tracker object continues where the last one stopped
    tr2 = S.SerialTracker(tmp_path / "ser")
    assert tr2.cur["next"] == fs.total and tr2.fetch(fs.fetch) == ""
    # overflow the 400-byte buffer between polls: gap marker, the half-lost line is dropped
    fs.write('HNET_METRIC runs/a {"step": 30, "bpb": 1.8, "pad": "' + "x" * 300 + '"}\n')
    fs.write('HNET_METRIC runs/a {"step": 40, "bpb": 1.7}\n' * 3)
    tr2.fetch(fs.fetch)
    assert tr2.cur["lost_bytes"] > 0
    p = S.parse_text(tr2.read_raw())
    assert p.gaps == 1
    steps = [r["step"] for r in p.metrics["runs/a"]]
    assert steps[:2] == [10, 20] and steps[-1] == 40 and 30 not in steps


def test_tracker_reset_after_restart(tmp_path):
    fs = FakeSerial()
    tr = S.SerialTracker(tmp_path / "ser")
    fs.write("x" * 500 + "\nHNET_DONE 0\n")
    tr.fetch(fs.fetch)
    fs2 = FakeSerial()  # restarted VM: counter starts at 0 again
    fs2.write("HNET_STAGE {\"stage\": \"boot\", \"boot\": 2}\n")

    def fetch(start):
        if start > fs2.total:
            return {"contents": "", "start": 0, "next": fs2.total}
        return fs2.fetch(start)

    nxt = tr.cur["next"]
    tr.fetch(lambda start: {"contents": "", "start": 0, "next": 0})  # a stopping VM: empty, not a reset
    assert tr.cur["resets"] == 0 and tr.cur["next"] == nxt
    tr.fetch(fetch)
    assert tr.cur["resets"] == 1 and tr.cur["next"] == fs2.total
    p = S.parse_text(tr.read_raw())
    assert p.resets == 1 and p.done is None and p.stages[-1]["boot"] == 2


def test_parse_lines_kinds_and_corruption():
    text = "\n".join([
        "[   12.3] kernel: something",
        'Oct 28 12:00:00 vm google_metadata_script_runner[1]: startup-script: HNET_STATUS run {"step": 5, "gpu_util": [97]}',
        'login: HNET_METRIC runs/tiny/x {"step": 1, "loss": 3.0}',
        'HNET_METRIC runs/tiny/x {"step": 2, "loss": 2.',  # corrupted (interleaved) line
        'HNET_EVAL runs/tiny/x {"step": 2, "val/bpb": 2.5}',
        'HNET_METRIC runs/with%20space {"step": 1}',
        'HNET_INFO {"torch": "2.9.1", "hnet_kernels": {"ssd_triton": true}}',
        'HNET_SUMMARY {"exit": 0}',
        "HNET_LOG hello world",
        "HNET_DONE 0",
        "HNET_DONE 1 partial-tail-not-terminated",
    ])
    p = S.parse_text(text)
    assert p.corrupt == 1
    assert p.status.data == {"state": "run", "step": 5, "gpu_util": [97]}
    assert [r["step"] for r in p.metrics["runs/tiny/x"]] == [1]
    assert p.evals["runs/tiny/x"][0]["val/bpb"] == 2.5
    assert "runs/with space" in p.metrics
    assert p.info["hnet_kernels"]["ssd_triton"] is True and p.summary == {"exit": 0}
    assert p.logs == ["hello world"] and p.done == 0  # last line has no newline yet


def test_resume_semantics_and_reconstruct(tmp_path):
    lines = [f'HNET_METRIC runs/tiny/a {{"step": {s}, "bpb": {3 - s / 100}}}' for s in (10, 20, 30, 40)]
    lines += [f'HNET_METRIC runs/tiny/a {{"step": {s}, "bpb": {2.5 - s / 1000}}}' for s in (30, 40, 50)]  # resumed at 20
    lines += ['HNET_METRIC runs/tiny/a {"step": 50, "bpb": 2.45}']  # exact duplicate collapses
    lines += ['HNET_EVAL runs/tiny/a {"step": 50, "tokens_eq": 1000, "val/bpb": 2.4}']
    p = S.parse_text("\n".join(lines) + "\n")
    assert [r["step"] for r in p.metrics["runs/tiny/a"]] == [10, 20, 30, 40, 50]
    assert p.metrics["runs/tiny/a"][2]["bpb"] == pytest.approx(2.47)
    dirs = S.reconstruct(p, tmp_path / "out")
    assert dirs == [tmp_path / "out" / "tiny" / "a"]
    m = [json.loads(x) for x in (dirs[0] / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in m] == [10, 20, 30, 40, 50]
    from hnet.eval.report import read_jsonl
    assert read_jsonl(str(dirs[0] / "eval.jsonl"))[0]["val/bpb"] == 2.4
    assert S.local_run_dir(tmp_path, "../../etc/x") == tmp_path / "etc" / "x"


# ----------------------------------------------------------------------------- ledger / budget
def _rec(**kw):
    r = {"name": "hnet-a", "price_hr": 1.0, "disk_hr": 0.0, "max_hours": 2.0, "created_at": 0.0,
         "segments": [{"start": 0.0, "deadline": 7200.0, "end": None}]}
    r.update(kw)
    return r


def test_ledger_actual_vs_committed():
    r = _rec()
    assert L.actual_cost(r, now=1800) == pytest.approx(0.5)
    assert L.committed_cost(r, now=1800) == pytest.approx(2.0)  # whole reservation while running
    assert L.actual_cost(r, now=99999) == pytest.approx(2.0)  # capped at the max-run-duration deadline
    r["segments"][0]["end"] = 3600
    assert L.committed_cost(r, now=5000) == pytest.approx(1.0)
    r["segments"].append({"start": 10000.0, "deadline": 17200.0, "end": None})  # resumed
    assert L.committed_cost(r, now=10100) == pytest.approx(3.0)
    r2 = _rec(disk_hr=0.1, ended_at=36000.0)
    r2["segments"][0]["end"] = 3600
    assert L.actual_cost(r2) == pytest.approx(1.0 + 0.1 * 10)  # disk paid while stopped, until deleted


def test_apply_observation():
    r = _rec()
    st = L.apply_observation(r, {"status": "TERMINATED", "lastStopTimestamp": "1970-01-01T00:30:00.000-00:00"},
                             now=5000)
    assert st == "TERMINATED" and r["segments"][0]["end"] == pytest.approx(1800)
    L.apply_observation(r, {"status": "RUNNING", "lastStartTimestamp": "1970-01-01T02:00:00Z"}, now=7300)
    assert len(r["segments"]) == 2 and r["segments"][1]["start"] == pytest.approx(7200)
    L.apply_observation(r, None, now=8000)
    assert r["status"] == "DELETED" and r["ended_at"] == 8000 and r["segments"][1]["end"] == 8000


def test_budget_guard():
    g = L.check_launch(L.projected_cost(0.35, 0.014, 0.5), committed=0, budget_total=300, launch_cap=25)
    assert g.ok
    g = L.check_launch(L.projected_cost(3.67, 0.014, 10), committed=0, budget_total=300, launch_cap=25)
    assert not g.ok and "per-launch cap" in g.reason
    g = L.check_launch(36.84, committed=0, budget_total=300, launch_cap=25, allow_over=True)
    assert g.ok
    g = L.check_launch(20, committed=290, budget_total=300, launch_cap=25, allow_over=True)
    assert not g.ok and "total budget" in g.reason


def test_prices_overridable(monkeypatch):
    assert C.price_hr("g2-standard-8", spot=True) == 0.35 and C.price_hr("a2-highgpu-1g", spot=False) == 3.67
    monkeypatch.setenv("HNET_GCP_PRICES", json.dumps({"g2-standard-8": {"spot": 0.3}}))
    assert C.price_hr("g2-standard-8", spot=True) == 0.3
    assert C.disk_price_hr(730) == pytest.approx(0.10)


# ----------------------------------------------------------------------------- CLI with mocked gcloud
def _launch_args(**kw):
    a = ["launch", "--name", "t1", "--gpu", "l4", "--spot", "--max-hours", "0.5", "--cmd", "echo hi",
         "--commit", "a" * 40]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return a


def test_cli_launch_records_ledger_and_refuses_over_budget(monkeypatch, capsys):
    monkeypatch.setattr(cli, "resolve_commit", lambda c: (c, True))

    def h(a):
        if a[3] == "create":
            return 0, json.dumps([{"name": a[4], "status": "RUNNING"}]), ""
        if a[3] == "describe":
            return 0, json.dumps({"name": a[4], "status": "RUNNING"}), ""
        raise AssertionError(a)

    fake = FakeGcloud(h)
    monkeypatch.setattr(subprocess, "run", fake)
    assert cli.main(_launch_args()) == 0
    st = State(C.state_dir() / "state.json")
    rec = st.get("hnet-t1")
    assert rec["zone"] == "us-central1-a" and rec["spot"] and rec["price_hr"] == 0.35
    sh = (C.state_dir() / "startup" / "hnet-t1.sh").read_text()
    job = json.loads(base64.b64decode(re.search(r'echo "([A-Za-z0-9+/=]+)" \| base64 -d > "\$S/job.json"', sh).group(1)))
    assert job["cmd"] == "echo hi" and job["commit"] == "a" * 40 and job["auto_shutdown"] is True
    # same name while active -> refused
    assert cli.main(_launch_args()) == 2
    # per-launch cap
    assert cli.main(_launch_args(name="t2", max_hours=100)) == 3
    assert "REFUSED" in capsys.readouterr().out
    n_create = sum(1 for c in fake.calls if c[3] == "create")
    assert n_create == 1
    # --spot/--on-demand is mandatory
    a = [x for x in _launch_args(name="t3") if x != "--spot"]
    assert cli.main(a) == 2


def test_cli_teardown_all_only_ours(monkeypatch):
    deleted = []
    gone = set()

    def h(a):
        if a[3] == "list":
            return 0, json.dumps([
                {"name": "hnet-x", "zone": "zones/us-central1-a", "labels": {"hnet-owner": "infra-gcp"}},
                {"name": "hnet-foreign", "zone": "zones/us-central1-a", "labels": {}},
            ]), ""
        if a[3] == "delete":
            deleted.append(a[4])
            gone.add(a[4])
            return 0, "", ""
        if a[3] == "describe":
            return (1, "", NOTFOUND) if a[4] in gone else (0, json.dumps({"status": "RUNNING"}), "")
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    assert cli.main(["teardown", "--all"]) == 0
    assert deleted == []  # not in our ledger: never touched without --include-untracked
    assert cli.main(["teardown", "--all", "--include-untracked"]) == 0
    assert deleted == ["hnet-x"]


def test_cli_sync_metrics_end_to_end(tmp_path, monkeypatch):
    st = State(C.state_dir() / "state.json")
    with st.locked():
        st.add({"name": "hnet-s", "id": "hnet-s", "zone": "us-central1-a", "machine_type": "g2-standard-8",
                "spot": True, "price_hr": 0.35, "disk_hr": 0.0, "max_hours": 0.5, "created_at": time.time(),
                "started_at": time.time(), "status": "RUNNING",
                "segments": [{"start": time.time(), "deadline": time.time() + 1800, "end": None}]})
    fs = FakeSerial(cap=100_000)
    fs.write('HNET_STAGE {"stage": "run"}\nHNET_METRIC runs/tiny/t {"step": 10, "bpb": 3.0, "tokens_eq": 100}\n')

    def h(a):
        if a[3] == "get-serial-port-output":
            start = int(next(x for x in a if x.startswith("--start=")).split("=")[1])
            return 0, json.dumps(fs.fetch(start)), ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    out = tmp_path / "runs_gcp"
    assert cli.main(["sync-metrics", "hnet-s", "--out", str(out)]) == 0
    fs.write('HNET_EVAL runs/tiny/t {"step": 10, "tokens_eq": 100, "val/bpb": 2.9}\nHNET_DONE 0\n')
    assert cli.main(["sync-metrics", "s", "--out", str(out)]) == 0
    m = (out / "tiny" / "t" / "metrics.jsonl").read_text().splitlines()
    e = (out / "tiny" / "t" / "eval.jsonl").read_text().splitlines()
    assert len(m) == 1 and json.loads(e[0])["val/bpb"] == 2.9


# ----------------------------------------------------------------------------- startup script + agent
def test_render_startup_embeds_agent_and_job():
    job = startup.job_dict(repo="https://x/y.git", commit="abc", cmd="echo 'q' && true")
    sh = startup.render(job)
    assert sh.startswith("#!/bin/bash") and "systemd-run --unit=hnet-agent" in sh and len(sh) < 200_000
    blobs = re.findall(r'echo "([A-Za-z0-9+/=]+)" \| base64 -d', sh)
    assert json.loads(base64.b64decode(blobs[0])) == job
    assert base64.b64decode(blobs[1]) == startup.AGENT_PATH.read_bytes()


def test_agent_train_invocations_and_needs():
    cmd = ("python -m hnet.train.train --config configs/runs/tiny_hnet2_2m.yaml --set train.train_tokens=5e6 "
           "eval.every=0 && python -m hnet.train.train --config=configs/runs/A1.yaml")
    inv = VA._train_invocations(cmd)
    assert inv == [("configs/runs/tiny_hnet2_2m.yaml", ["train.train_tokens=5e6", "eval.every=0"]),
                   ("configs/runs/A1.yaml", [])]
    needs = VA.merge_needs([{"source": "hnet", "dataset": "finewebedu10B", "n_train_shards": 1, "val": "standard"},
                            {"source": "hnet", "dataset": "fineweb10B", "n_train_shards": 2, "val": "speedrun"},
                            {"source": "synthetic", "val": None}])
    assert needs == {"finewebedu10B": 1, "fineweb10B": 2}
    assert VA.merge_needs([{"source": "hnet", "dataset": "fineweb10B", "n_train_shards": None, "val": "standard"}]) \
        == {"fineweb10B": 103, "finewebedu10B": 0}
    assert VA.parse_data_spec("finewebedu10B:1,fineweb10B:0") == {"finewebedu10B": 1, "fineweb10B": 0}
    assert VA.parse_data_spec("none") == {}


def test_agent_mirror_new_lines_only(tmp_path):
    ser = tmp_path / "ttyS0"
    ser.write_text("")
    em = VA.Emitter(dev=str(ser))
    run = tmp_path / "repo" / "runs" / "tiny" / "r"
    run.mkdir(parents=True)
    mf = run / "metrics.jsonl"
    mf.write_text('{"step": 1, "loss": 3.0}\n{"step": 2, "loss"')  # second line incomplete
    mir = VA.Mirror(str(tmp_path / "repo"), em, str(tmp_path / "mirror.json"))
    assert mir.scan() == 1
    with open(mf, "a") as f:
        f.write(': 2.5}\n')
    (run / "eval.jsonl").write_text('{"step": 2, "val/bpb": 2.0}\n')
    assert mir.scan() == 2 and mir.last_step == 2
    assert VA.Mirror(str(tmp_path / "repo"), em, str(tmp_path / "mirror.json")).scan() == 0  # offsets persisted
    mf.write_text('{"step": 1, "loss": 9.0}\n')  # truncated (fresh run) -> re-read from 0
    assert mir.scan() == 1
    p = S.parse_text(ser.read_text())
    assert [r["loss"] for r in p.metrics["runs/tiny/r"]] == [9.0]  # fresh run replaced the old steps
    assert p.evals["runs/tiny/r"][0]["val/bpb"] == 2.0


def test_agent_full_run_offline(tmp_path, monkeypatch):
    """The whole agent loop with setup stages stubbed: cmd writes metrics, serial gets HNET lines, DONE."""
    repo = tmp_path / "repo"
    repo.mkdir()
    for k, v in {"STATE": tmp_path / "state", "REPO": repo, "VENV": tmp_path / "venv"}.items():
        monkeypatch.setattr(VA, k, str(v))
    monkeypatch.setattr(VA, "POLL_SEC", 0.05)
    monkeypatch.setattr(VA, "HB_SEC", 0.1)
    monkeypatch.setattr(VA, "guest_attr", lambda k, v: True)
    for st in ("clone", "venv", "kernels", "data"):
        monkeypatch.setattr(VA.Agent, st, lambda self: None)
    ser = tmp_path / "ttyS0"
    ser.write_text("")
    script = (f"import json,os,time\nos.makedirs('runs/t/r',exist_ok=True)\n"
              f"f=open('runs/t/r/metrics.jsonl','w')\n"
              f"for s in range(1,6):\n f.write(json.dumps({{'step':s*10,'bpb':3-s/10}})+'\\n'); f.flush(); time.sleep(0.05)\n"
              f"open('runs/t/r/eval.jsonl','w').write(json.dumps({{'step':50,'val/bpb':2.4}})+'\\n')\n"
              f"print('trained'); time.sleep(0.3)\n")
    (repo / "fake_train.py").write_text(script)
    job = startup.job_dict(repo="x", commit="abc", cmd=f"{sys.executable} fake_train.py; exit 3", linger_min=0,
                           auto_shutdown=False)
    em = VA.Emitter(dev=str(ser))
    VA.Agent(job, em).main()
    p = S.parse_text(ser.read_text())
    assert p.done == 3
    assert [r["step"] for r in p.metrics["runs/t/r"]] == [10, 20, 30, 40, 50]
    assert p.evals["runs/t/r"][0]["val/bpb"] == 2.4
    assert [s["stage"] for s in p.stages][:2] == ["boot", "clone"] and p.stages[-1]["stage"] == "run"
    assert p.summary["exit"] == 3 and p.summary["last_metric"]["step"] == 50
    assert p.status is not None  # at least one heartbeat
    assert any("trained" in ln for ln in p.logs)
    # reboot after completion: no re-run, DONE re-announced
    ser.write_text("")
    VA.Agent(job, VA.Emitter(dev=str(ser))).main()
    p2 = S.parse_text(ser.read_text())
    assert p2.done == 3 and not p2.metrics and any("trained" in ln for ln in p2.logs)
    # a new job nonce (resume --rerun) runs the command again
    ser.write_text("")
    job2 = dict(job, nonce="second")
    VA.Agent(job2, VA.Emitter(dev=str(ser))).main()
    p3 = S.parse_text(ser.read_text())
    assert p3.done == 3 and [r["step"] for r in p3.metrics["runs/t/r"]] == [10, 20, 30, 40, 50]


def test_cli_resume_rerun_pushes_new_job(monkeypatch):
    st = State(C.state_dir() / "state.json")
    job = startup.job_dict(repo="r", commit="c" * 40, cmd="echo 1", nonce="n1")
    t = time.time() - 3600
    with st.locked():
        st.add({"name": "hnet-r", "id": "hnet-r", "zone": "us-central1-a", "machine_type": "g2-standard-8",
                "spot": True, "price_hr": 0.35, "disk_hr": 0.0, "max_hours": 0.5, "created_at": t, "started_at": t,
                "status": "TERMINATED", "segments": [{"start": t, "deadline": t + 1800, "end": t + 600}],
                "meta": {"job": job}})
    calls = []

    def h(a):
        calls.append(a[3])
        if a[3] == "describe":
            return 0, json.dumps({"status": "TERMINATED"}), ""
        if a[3] in ("add-metadata", "start"):
            return 0, "", ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    assert cli.main(["resume", "hnet-r", "--cmd", "echo 2"]) == 0
    assert calls[-2:] == ["add-metadata", "start"]
    rec = State(C.state_dir() / "state.json").get("hnet-r")
    assert rec["meta"]["job"]["cmd"] == "echo 2" and rec["meta"]["job"]["nonce"] != "n1"
    assert len(rec["segments"]) == 2 and rec["segments"][1]["end"] is None


def test_overlay_roundtrip_and_missing_config_check(tmp_path, monkeypatch):
    root = tmp_path / "src"
    (root / "configs" / "runs").mkdir(parents=True)
    (root / "configs" / "runs" / "a.yaml").write_text("train: {}\n")
    (root / "configs" / "runs" / "__pycache__").mkdir()
    (root / "configs" / "runs" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    b64 = startup.make_overlay(["configs/runs"], root)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(VA, "REPO", str(repo))
    ser = tmp_path / "tty"
    ser.write_text("")
    ag = VA.Agent({"overlay_b64": b64}, VA.Emitter(dev=str(ser)))
    ag.apply_overlay()
    assert (repo / "configs" / "runs" / "a.yaml").read_text() == "train: {}\n"
    assert not (repo / "configs" / "runs" / "__pycache__" / "x.pyc").exists()
    with pytest.raises(FileNotFoundError):
        startup.make_overlay(["nope"], root)
    # configs/runs is .gitignored by `runs/` in this repo: the launch check must notice untracked configs
    sha = subprocess.run(["git", "-C", str(C.REPO_ROOT), "rev-parse", "HEAD"], capture_output=True,
                         text=True).stdout.strip()
    if sha:
        cmd = "python -m hnet.train.train --config configs/does_not_exist.yaml"
        assert cli.missing_configs(cmd, sha, []) == ["configs/does_not_exist.yaml"]
        assert cli.missing_configs(cmd, sha, ["configs"]) == []
        assert cli.missing_configs("python -m hnet.train.train --config configs/README.md", sha, []) == []


def test_stale_local_state_is_reconciled(monkeypatch):
    """Local ledger says RUNNING (e.g. the controller restarted) but the VM is TERMINATED / restarted / gone."""
    def iso(t):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))
    t0 = time.time() - 7200
    st = State(C.state_dir() / "state.json")
    with st.locked():
        for n in ("hnet-a", "hnet-b", "hnet-c"):
            st.add({"name": n, "id": n, "zone": "us-central1-a", "machine_type": "g2-standard-8", "spot": True,
                    "price_hr": 1.0, "disk_hr": 0.0, "max_hours": 0.5, "created_at": t0, "started_at": t0,
                    "status": "RUNNING", "segments": [{"start": t0, "deadline": t0 + 1800, "end": None}]})
    obs = {"hnet-a": (0, json.dumps({"status": "TERMINATED", "lastStopTimestamp": iso(t0 + 600)}), ""),
           "hnet-b": (0, json.dumps({"status": "RUNNING", "lastStartTimestamp": iso(t0 + 3600),
                                     "lastStopTimestamp": iso(t0 + 900)}), ""),
           "hnet-c": (1, "", NOTFOUND)}

    def h(a):
        if a[3] == "describe":
            return obs[a[4]]
        if a[3] == "list":
            return 0, "[]", ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    assert cli.main(["ls"]) == 0
    st = State(C.state_dir() / "state.json")
    a, b, c = (st.get(n) for n in ("hnet-a", "hnet-b", "hnet-c"))
    assert a["status"] == "TERMINATED" and a["segments"][0]["end"] == pytest.approx(t0 + 600, abs=1)
    assert L.actual_cost(a) == pytest.approx(600 / 3600, abs=2 / 3600) and not a.get("ended_at")
    assert b["status"] == "RUNNING" and len(b["segments"]) == 2
    assert b["segments"][0]["end"] == pytest.approx(t0 + 900, abs=1) and b["segments"][1]["end"] is None
    assert c["status"] == "DELETED" and c["ended_at"]
    assert [r["name"] for r in st.active()] == ["hnet-a", "hnet-b"]


# ----------------------------------------------------------------------------- GCS extension
def test_create_args_with_service_account(tmp_path):
    a = gce.create_args(_spec(tmp_path, service_account="sa@p.iam.gserviceaccount.com"), "us-central1-a")
    assert "--service-account=sa@p.iam.gserviceaccount.com" in a and "--scopes=cloud-platform" in a
    assert "--no-service-account" not in a and "--no-scopes" not in a


def test_gcs_helpers():
    assert VA.run_rel("runs/tiny/x") == "tiny/x" and VA.run_rel("out/y/") == "out/y"
    assert VA.wanted_data_object("finewebedu_val_000000.bin", 0)
    assert VA.wanted_data_object("bytes/finewebedu_train_000001.bytes.bin", 1)
    assert not VA.wanted_data_object("bytes/finewebedu_train_000002.idx.npy", 1)
    assert VA.wanted_data_object("bytes/convert_report.json", 0)
    rels = ["metrics.jsonl", "ckpt/step_0000100/meta.json", "ckpt/step_0000100/model.pt",
            "ckpt/step_0000200/model.pt",  # incomplete (being uploaded / deleted)
            "ckpt/step_0000050/meta.json", "ckpt/step_0000050/model.pt"]
    assert VA.latest_complete_ckpt(rels) == "step_0000100"
    assert VA.latest_complete_ckpt(["metrics.jsonl"]) is None
    assert gce.parse_ls_sizes("     12  2026-10-28T00:00:00Z  gs://b/x/a.bin\nTOTAL: 1 objects, 12 bytes (12B)\n") \
        == {"gs://b/x/a.bin": 12}


class FakeBucket:
    """In-memory bucket answering the agent's `gcloud storage` calls (ls -l / cp / rsync)."""

    def __init__(self):
        self.obj: dict[str, bytes] = {}
        self.calls = []

    def exec(self, args, timeout=0):
        self.calls.append(args)
        verb = args[1]
        if verb == "ls":
            pre = args[-1][:-3] if args[-1].endswith("/**") else args[-1]
            hits = {u: len(b) for u, b in self.obj.items() if u.startswith(pre)}
            if not hits:
                return 1, "", "ERROR: (gcloud.storage.ls) One or more URLs matched no objects."
            return 0, "".join(f"{s}  2026-01-01T00:00:00Z  {u}\n" for u, s in hits.items()), ""
        if verb == "cp":
            rec = "-r" in args
            srcs, dst = [a for a in args[2:-1] if a != "-r"], args[-1]
            for src in srcs:
                if src.startswith("gs://"):
                    items = [(u, u[len(src.rsplit("/", 1)[0]) + 1:]) for u in self.obj if u == src or
                             (rec and u.startswith(src + "/"))]
                    for u, rel in items:
                        tgt = os.path.join(dst, rel) if dst.endswith("/") else dst
                        os.makedirs(os.path.dirname(tgt), exist_ok=True)
                        open(tgt, "wb").write(self.obj[u])
                else:
                    tgt = dst + os.path.basename(src) if dst.endswith("/") else dst
                    self.obj[tgt] = open(src, "rb").read()
            return 0, "", ""
        if verb == "rsync":
            src, dst = args[-2], args[-1]
            excl = [a.split("=", 1)[1] for a in args if a.startswith("--exclude=")]
            delete = "--delete-unmatched-destination-objects" in args
            seen = set()
            for dp, _, fs in os.walk(src):
                for f in fs:
                    rel = os.path.relpath(os.path.join(dp, f), src)
                    if any(re.search(x, rel) for x in excl):
                        continue
                    self.obj[f"{dst}/{rel}"] = open(os.path.join(dp, f), "rb").read()
                    seen.add(f"{dst}/{rel}")
            if delete:
                for u in [u for u in self.obj if u.startswith(dst + "/") and u not in seen]:
                    del self.obj[u]
            return 0, "", ""
        raise AssertionError(args)


def _gcs_agent(tmp_path, monkeypatch, bucket, prefix="runs/hnet-a", restore_prefix=None):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(VA, "REPO", str(repo))
    monkeypatch.setattr(VA, "STATE", str(tmp_path / "state"))
    os.makedirs(tmp_path / "state", exist_ok=True)
    monkeypatch.setattr(VA, "find_gcloud", lambda: "/usr/bin/gcloud")
    ser = tmp_path / "tty"
    ser.write_text("")
    ag = VA.Agent({"gcs": {"bucket": "b", "prefix": prefix, "restore_prefix": restore_prefix or prefix}},
                  VA.Emitter(dev=str(ser)))
    monkeypatch.setattr(ag.gcs, "_exec", bucket.exec)
    return ag, repo


def test_gcs_data_cache_roundtrip(tmp_path, monkeypatch):
    bk = FakeBucket()
    ag, repo = _gcs_agent(tmp_path, monkeypatch, bk)
    root = tmp_path / "data1"
    d = root / "finewebedu10B"
    (d / "bytes").mkdir(parents=True)
    for n in ("finewebedu_val_000000.bin", "finewebedu_train_000001.bin", "finewebedu_train_000002.bin",
              "bytes/finewebedu_train_000001.bytes.bin", "bytes/finewebedu_train_000002.bytes.bin",
              "bytes/convert_report.json"):
        (d / n).write_bytes(n.encode())
    (d / ".cache" / "huggingface").mkdir(parents=True)
    (d / ".cache" / "huggingface" / "x.metadata").write_text("junk")
    assert ag.gcs.upload_data("finewebedu10B", str(root)) == 6
    assert not any(".cache" in k for k in bk.obj)
    assert ag.gcs.upload_data("finewebedu10B", str(root)) == 0  # already cached
    assert "gs://b/data/finewebedu10B/bytes/finewebedu_train_000002.bytes.bin" in bk.obj
    # a fresh VM needing 1 train shard copies val + shard 1 + reports, not shard 2
    root2 = tmp_path / "data2"
    assert ag.gcs.restore_data("finewebedu10B", 1, str(root2)) == 4
    got = sorted(str(p.relative_to(root2)) for p in root2.rglob("*") if p.is_file())
    assert got == ["finewebedu10B/bytes/convert_report.json", "finewebedu10B/bytes/finewebedu_train_000001.bytes.bin",
                   "finewebedu10B/finewebedu_train_000001.bin", "finewebedu10B/finewebedu_val_000000.bin"]
    assert ag.gcs.restore_data("finewebedu10B", 1, str(root2)) == 0  # present with the same size


def test_gcs_run_sync_and_restore(tmp_path, monkeypatch):
    bk = FakeBucket()
    ag, repo = _gcs_agent(tmp_path, monkeypatch, bk)
    run = repo / "runs" / "tiny" / "x"
    (run / "ckpt" / "step_0000010").mkdir(parents=True)
    (run / "ckpt" / ".tmp_step_0000020").mkdir()
    (run / "metrics.jsonl").write_text('{"step": 10}\n')
    (run / "config.yaml").write_text("a: 1\n")
    for f in ("meta.json", "model.pt", "optim.pt"):
        (run / "ckpt" / "step_0000010" / f).write_text(f)
    (run / "ckpt" / ".tmp_step_0000020" / "model.pt").write_text("partial")
    (tmp_path / "state" / "cmd.log").write_text("log\n")
    assert ag.gcs.sync_runs(["runs"]) == 1
    keys = sorted(bk.obj)
    assert "gs://b/runs/hnet-a/tiny/x/metrics.jsonl" in keys and "gs://b/runs/hnet-a/_logs/cmd.log" in keys
    assert "gs://b/runs/hnet-a/tiny/x/ckpt/step_0000010/model.pt" in keys
    assert not any(".tmp_" in k for k in keys)
    # rolling checkpoint replaced locally -> the old one disappears from the bucket too
    import shutil as _sh
    _sh.rmtree(run / "ckpt" / "step_0000010")
    (run / "ckpt" / "step_0000030").mkdir()
    for f in ("meta.json", "model.pt"):
        (run / "ckpt" / "step_0000030" / f).write_text(f)
    ag.gcs.sync_runs(["runs"])
    assert not any("step_0000010" in k for k in bk.obj) and any("step_0000030" in k for k in bk.obj)
    # disk gone: a new VM (fresh repo) restores the newest complete checkpoint + jsonl before training
    ag2, repo2 = _gcs_agent(tmp_path / "vm2", monkeypatch, bk)
    assert ag2.gcs.restore_run("runs/tiny/x") == "step_0000030"
    assert (repo2 / "runs/tiny/x/ckpt/step_0000030/model.pt").read_text() == "model.pt"
    assert (repo2 / "runs/tiny/x/metrics.jsonl").read_text() == '{"step": 10}\n'
    assert ag2.gcs.restore_run("runs/tiny/x") is None  # local checkpoint present -> nothing to do
    ag3, _ = _gcs_agent(tmp_path / "vm3", monkeypatch, bk, prefix="runs/hnet-b", restore_prefix="runs/hnet-a")
    assert ag3.gcs.restore_run("runs/tiny/x") == "step_0000030"


def test_sync_metrics_prefers_bucket_then_serial(tmp_path, monkeypatch):
    st = State(C.state_dir() / "state.json")
    t = time.time()
    with st.locked():
        st.add({"name": "hnet-g", "id": "hnet-g", "zone": "us-central1-a", "machine_type": "g2-standard-8",
                "spot": True, "price_hr": 0.35, "disk_hr": 0.0, "max_hours": 0.5, "created_at": t, "started_at": t,
                "status": "RUNNING", "segments": [{"start": t, "deadline": t + 1800, "end": None}],
                "meta": {"job": {"gcs": {"bucket": "bk"}}}})
    fs = FakeSerial(cap=500)  # small: the serial buffer lost the early lines
    fs.write("x" * 600 + "\n")
    fs.write('HNET_METRIC runs/tiny/t {"step": 30, "bpb": 2.8}\nHNET_METRIC runs/tiny/t {"step": 40, "bpb": 2.7}\n')
    bucket = {"gs://bk/runs/hnet-g/tiny/t/metrics.jsonl": '{"step": 10, "bpb": 3.0}\n{"step": 20, "bpb": 2.9}\n'
                                                          '{"step": 30, "bpb": 2.8}\n{"step": 4',
              "gs://bk/runs/hnet-g/tiny/t/eval.jsonl": '{"step": 20, "val/bpb": 2.95}\n',
              "gs://bk/runs/hnet-g/_logs/cmd.log": "log"}

    def h(a):
        if a[1] == "storage" and a[2] == "ls":
            return 0, "".join(f"{len(v)}  2026-01-01T00:00:00Z  {u}\n" for u, v in bucket.items()), ""
        if a[1] == "storage" and a[2] == "cp":
            Path(a[-2] if False else a[-1]).write_text(bucket[a[3]])
            return 0, "", ""
        if a[3] == "get-serial-port-output":
            start = int(next(x for x in a if x.startswith("--start=")).split("=")[1])
            return 0, json.dumps(fs.fetch(start)), ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    out = tmp_path / "o"
    assert cli.main(["sync-metrics", "hnet-g", "--out", str(out)]) == 0
    m = [json.loads(x) for x in (out / "tiny" / "t" / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in m] == [10, 20, 30, 40]  # bucket history + fresher serial line
    assert json.loads((out / "tiny" / "t" / "eval.jsonl").read_text())["val/bpb"] == 2.95


def test_cli_pull_latest_checkpoint(tmp_path, monkeypatch):
    st = State(C.state_dir() / "state.json")
    t = time.time()
    with st.locked():
        st.add({"name": "hnet-p", "id": "hnet-p", "zone": "z", "machine_type": "g2-standard-8", "spot": True,
                "price_hr": 0.35, "disk_hr": 0.0, "max_hours": 0.5, "created_at": t, "started_at": t,
                "status": "TERMINATED", "segments": [], "meta": {"job": {"gcs": {"bucket": "bk"}}}})
    objs = {f"gs://bk/runs/hnet-p/tiny/t/{r}": r for r in
            ["metrics.jsonl", "ckpt/step_0000010/meta.json", "ckpt/step_0000010/model.pt",
             "ckpt/step_0000020/meta.json", "ckpt/step_0000020/model.pt"]}
    copied = []

    def h(a):
        if a[2] == "ls":
            return 0, "".join(f"{len(v)}  2026-01-01T00:00:00Z  {u}\n" for u, v in objs.items()), ""
        if a[2] == "cp":
            copied.extend(a[3:-1])
            return 0, "", ""
        raise AssertionError(a)

    monkeypatch.setattr(subprocess, "run", FakeGcloud(h))
    assert cli.main(["pull", "hnet-p", "runs/tiny/t", "--out", str(tmp_path / "o")]) == 0
    assert sorted(u.rsplit("tiny/t/", 1)[1] for u in copied) == \
        ["ckpt/step_0000020/meta.json", "ckpt/step_0000020/model.pt", "metrics.jsonl"]


def _offline_agent(tmp_path, monkeypatch, job_kw, stub=("clone", "venv", "kernels", "data")):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    for k, v in {"STATE": tmp_path / "state", "REPO": repo, "VENV": tmp_path / "venv"}.items():
        monkeypatch.setattr(VA, k, str(v))
    os.makedirs(tmp_path / "state", exist_ok=True)
    monkeypatch.setattr(VA, "POLL_SEC", 0.05)
    monkeypatch.setattr(VA, "HB_SEC", 0.1)
    monkeypatch.setattr(VA, "SYNC_RETRY_WAIT", 0.0)
    monkeypatch.setattr(VA, "guest_attr", lambda k, v: None)
    for st in stub:
        monkeypatch.setattr(VA.Agent, st, lambda self: None)
    ser = tmp_path / "tty"
    ser.write_text("")
    job = startup.job_dict(repo="x", commit="abc", linger_min=0, auto_shutdown=False, **job_kw)
    return VA.Agent(job, VA.Emitter(dev=str(ser))), ser


def test_agent_stage_failure_is_reported_and_raised(tmp_path, monkeypatch):
    ag, ser = _offline_agent(tmp_path, monkeypatch, {"cmd": "true"}, stub=("clone", "venv", "kernels"))

    def boom(self):
        raise RuntimeError("prepare exploded")

    monkeypatch.setattr(VA.Agent, "data", boom)
    with pytest.raises(RuntimeError, match="prepare exploded"):
        ag.main()
    p = S.parse_text(ser.read_text())
    assert p.done == VA.INFRA_FAILURE and p.summary["failed_stage"] == "data"
    assert any("prepare exploded" in ln for ln in p.logs)
    assert "Traceback" in (tmp_path / "state" / "agent.log").read_text()
    assert json.loads((tmp_path / "state" / "done.json").read_text())["failed_stage"] == "data"


def test_agent_bucket_failure_stops_the_command(tmp_path, monkeypatch):
    ag, ser = _offline_agent(tmp_path, monkeypatch, {"cmd": "sleep 30"})
    ag.gcs = VA.Gcs({"bucket": "b", "prefix": "runs/x", "sync_every_s": 0.05})
    ag.gcs.gcloud = "/usr/bin/gcloud"
    calls = []

    def failing(args, timeout):
        calls.append(args)
        return 1, "", "ERROR: 503 backend unavailable"

    monkeypatch.setattr(ag.gcs, "_exec", failing)
    t = time.time()
    with pytest.raises(RuntimeError, match="background thread failed"):
        ag.main()
    assert time.time() - t < 20  # the 30 s command was stopped
    p = S.parse_text(ser.read_text())
    assert p.done == VA.INFRA_FAILURE
    assert sum("bucket sync failed" in ln for ln in p.logs) >= VA.SYNC_RETRIES


def test_agent_command_failure_exit_code(tmp_path, monkeypatch):
    ag, ser = _offline_agent(tmp_path, monkeypatch, {"cmd": "echo bad; exit 4"})
    assert ag.main() == 4
    p = S.parse_text(ser.read_text())
    assert p.done == 4 and any("bad" in ln for ln in p.logs)


def test_bucket_select_objects():
    from infra.gcp import bucket as B
    rels = ["metrics.jsonl", "ckpt/step_0000010/meta.json", "ckpt/step_0000010/model.pt",
            "ckpt/step_0000020/meta.json", "ckpt/step_0000020/model.pt", "ckpt/step_0000030/model.pt"]
    assert B.select_objects(rels, "latest") == ["metrics.jsonl", "ckpt/step_0000020/meta.json",
                                                "ckpt/step_0000020/model.pt"]
    assert B.select_objects(rels, "none") == ["metrics.jsonl"]
    assert B.select_objects(rels, "all") == rels
    with pytest.raises(ValueError):
        B.select_objects(rels, "newest")
    assert B.run_name("runs/tiny/x/") == "tiny/x" and B.bucket_of({"meta": {}}) is None
