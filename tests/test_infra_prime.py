"""Unit tests for infra.prime with the Prime SDK mocked (no network, no spend)."""

import json
import subprocess
import sys
import tarfile
import io
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from infra.prime import budget as B
from infra.prime import cli, offers as O, pack, remote
from infra.prime.backends import ExecResult, parse_ssh
from infra.prime.state import State

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infra" / "prime"))
import gpu_caps  # noqa: E402


@pytest.fixture(autouse=True)
def _state_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HNET_PRIME_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PRIME_API_KEY", "test-not-a-key")
    yield


# ----------------------------------------------------------------------------- budget
def test_projected_and_sandbox_price():
    assert B.projected_cost(5.4, 2, 10) == pytest.approx(108.0)
    assert B.projected_cost(1.0, 1, 2, extra_hr=0.5) == pytest.approx(3.0)
    # docs example: 1 vCPU, 2 GiB, 5 GiB disk -> $0.046/h
    assert B.sandbox_price_hr(1, 2, 5) == pytest.approx(0.046)
    assert B.sandbox_price_hr(0.5, 2, 5) == pytest.approx(0.046)  # CPU rounds up to 1
    assert B.sandbox_price_hr(8, 64, 100, 1, 5.4) == pytest.approx(0.16 + 0.8 + 0.02 + 5.4)


def test_check_launch_rules():
    ok = B.check_launch(40, 100, 500, 50)
    assert ok.ok
    over_total = B.check_launch(60, 450, 500, 50, allow_over=True)
    assert not over_total.ok and "total budget" in over_total.reason
    over_cap = B.check_launch(60, 0, 500, 50)
    assert not over_cap.ok and "per-launch cap" in over_cap.reason
    assert B.check_launch(60, 0, 500, 50, allow_over=True).ok
    assert not B.check_launch(40, 0, 500, 50, wallet_balance=30).ok
    assert B.check_launch(50, 450, 500, 50).ok  # exactly at both limits


def test_committed_vs_actual():
    now = 1_000_000.0
    active = {"price_hr": 2.0, "started_at": now - 3600, "deadline": now + 3 * 3600, "ended_at": None}
    ended = {"price_hr": 2.0, "started_at": now - 7200, "deadline": now + 3600, "ended_at": now - 3600}
    billed = {"price_hr": 1.0, "started_at": now - 3600, "deadline": now, "ended_at": now,
              "billed_usd": 3.5}
    overdue = {"price_hr": 1.0, "started_at": now - 5 * 3600, "deadline": now - 4 * 3600, "ended_at": None}
    assert B.actual_cost(active, now) == pytest.approx(2.0)
    assert B.committed_cost(active, now) == pytest.approx(8.0)  # whole reservation
    assert B.committed_cost(ended, now) == pytest.approx(2.0)
    assert B.committed_cost(billed, now) == pytest.approx(3.5)  # platform billing wins when larger
    assert B.actual_cost(overdue, now) == pytest.approx(1.0)  # capped at deadline
    assert B.committed_total([active, ended, billed], now) == pytest.approx(13.5)
    assert B.spent_so_far([active, ended], now) == pytest.approx(4.0)


# ----------------------------------------------------------------------------- state
def test_state_roundtrip(tmp_path):
    st = State(tmp_path / "s.json")
    with st.locked():
        st.add({"name": "hnet-a", "id": "sb1", "kind": "sandbox", "price_hr": 1.0,
                "started_at": 1.0, "deadline": 2.0})
        st.set_wallet_baseline(100.0)
        st.set_wallet_baseline(50.0)  # only first one sticks
    st2 = State(tmp_path / "s.json")
    assert st2.get("hnet-a")["id"] == "sb1"
    assert st2.get("sb1")["name"] == "hnet-a"  # lookup by id
    assert st2.data["wallet_baseline"]["balance"] == 100.0
    with pytest.raises(ValueError):
        st2.add({"name": "hnet-a", "id": "sb2", "kind": "sandbox", "price_hr": 1, "started_at": 3})
    with st2.locked():
        st2.add_run("hnet-a", "r1", "echo hi")
        st2.mark_ended("hnet-a", when=5.0)
        st2.add({"name": "hnet-a", "id": "sb2", "kind": "sandbox", "price_hr": 1, "started_at": 6})
    st3 = State(tmp_path / "s.json")
    assert st3.get("hnet-a")["id"] == "sb2"
    assert [r["id"] for r in st3.data["history"]] == ["sb1"]
    assert st3.data["history"][0]["runs"][0]["cmd"] == "echo hi"
    assert len(st3.active()) == 1 and len(st3.all_records()) == 2
    with pytest.raises(KeyError):
        st3.get("nope")


# ----------------------------------------------------------------------------- tarball
def test_tarball_contents(tmp_path):
    root = tmp_path / "repo"
    files = {
        "hnet/__init__.py": "x", "hnet/model/a.py": "y", "configs/c.yaml": "z", "requirements.txt": "r",
        "infra/prime/setup_gpu.sh": "s", "data/shard.bin": "d", "runs/r1/stdout.log": "l",
        ".venv/lib/x.py": "v", "hnet/__pycache__/a.cpython-313.pyc": "p", "hnet/b.pyc": "p",
        ".git/config": "g", "ckpt/model.pt": "m", "x.egg-info/PKG-INFO": "e", ".pytest_cache/v": "c",
    }
    for rel, txt in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(txt)
    (root / "big.txt").write_bytes(b"0" * (pack.MAX_FILE_BYTES + 1))
    got = pack.list_files(root)
    assert got == sorted(["configs/c.yaml", "hnet/__init__.py", "hnet/model/a.py",
                          "infra/prime/setup_gpu.sh", "requirements.txt"])
    data = pack.make_tarball(root)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        assert sorted(tf.getnames()) == got
        assert tf.extractfile("hnet/model/a.py").read() == b"y"
        assert all(m.uid == 0 for m in tf.getmembers())


def test_tarball_of_this_repo_has_package_and_no_data():
    files = pack.list_files(Path(__file__).resolve().parents[1])
    assert "infra/prime/cli.py" in files and "infra/prime/setup_gpu.sh" in files
    assert not any(f.startswith(("data/", "runs/", ".venv/", ".git/")) or "__pycache__" in f for f in files)


# ----------------------------------------------------------------------------- remote shell snippets
def _sh(cmd):
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)


def test_detached_run_tail_ps_locally(tmp_path):
    root = str(tmp_path / "r")
    Path(root).mkdir()
    rid = "t1"
    cmd = "echo hello; echo '{\"step\": 3, \"loss\": 1.25}' > $HNET_RUN_DIR/metrics.jsonl; sleep 0.3; exit 7"
    out = _sh(remote.detached(root, rid, cmd, env={"FOO": "bar baz"}))
    assert out.returncode == 0 and f"started {rid}" in out.stdout
    for _ in range(50):
        if (Path(root) / "runs/t1/exit_code").exists():
            break
        time.sleep(0.1)
    assert (Path(root) / "runs/t1/exit_code").read_text().strip() == "7"
    assert "hello" in (Path(root) / "runs/t1/stdout.log").read_text()
    t = _sh(remote.tail(root, rid, 5, metrics=True))
    assert "hello" in t.stdout and '"step": 3' in t.stdout
    p = _sh(remote.ps(root, [rid, "missing"]))
    assert "t1: exited 7" in p.stdout and "missing: not running" in p.stdout
    assert remote.format_metrics_line('{"step": 3, "loss": 1.25, "x": 1}') == "step=3 loss=1.25"


def test_sync_unpack_locally(tmp_path):
    src = tmp_path / "src"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg/m.py").write_text("print(1)")
    tarp = tmp_path / "x.tar.gz"
    pack.make_tarball(src, tarp)
    dest = tmp_path / "dest dir"
    r = _sh(remote.sync_unpack(str(dest), str(tarp)))
    assert r.returncode == 0, r.stderr
    assert (dest / "pkg/m.py").read_text() == "print(1)" and not tarp.exists()


def test_paths_and_ids():
    assert remote.resolve_remote_path("/workspace/x", "runs/a/m.jsonl") == "/workspace/x/runs/a/m.jsonl"
    assert remote.resolve_remote_path("/workspace/x", "/etc/hosts") == "/etc/hosts"
    assert remote.root_expr("~/hnet") == '"$HOME"/hnet'
    rid = remote.new_run_id("python -m hnet.train.train --config a.yaml", now=0)
    assert rid.startswith("19700101-000000-hnet-train")
    assert parse_ssh("root@1.2.3.4 -p 2222") == ("root", "1.2.3.4", 2222)
    assert parse_ssh(["ubuntu@5.6.7.8"]) == ("ubuntu", "5.6.7.8", 22)


# ----------------------------------------------------------------------------- offers
def _avail(gtype, n, price, disk_default=100, disk_ppu=0.0001, included=False, stock="Available"):
    return NS(gpu_type=gtype, gpu_count=n, prices=NS(on_demand=price, community_price=None),
              disk=NS(default_count=disk_default, price_per_unit=disk_ppu, default_included_in_price=included),
              provider="nebius", country="US", data_center="us1", stock_status=stock,
              vcpu=NS(default_count=16), memory=NS(default_count=200), gpu_memory=80, is_spot=None,
              cloud_id="c", socket="SXM5")


class FakeAvail:
    def get(self):
        return {"H200_141GB": [_avail("H200_141GB", 1, 5.4), _avail("H200_141GB", 8, 40.0)],
                "A100_80GB": [_avail("A100_80GB", 1, 1.5, included=True)],
                "CPU_NODE": [_avail("CPU_NODE", 1, 0.066)]}


def test_pod_offers_sorted_and_filtered():
    offs = O.pod_offers(client=FakeAvail())
    assert [o.gpu_type for o in offs] == ["A100_80GB", "H200_141GB", "H200_141GB"]
    assert offs[0].per_gpu_hr == pytest.approx(1.5)            # disk included
    assert offs[1].per_gpu_hr == pytest.approx((40.0 + 0.01) / 8)
    assert offs[2].price_hr == pytest.approx(5.41)
    assert [o.gpu_type for o in O.pod_offers("h200", client=FakeAvail())] == ["H200_141GB"] * 2
    assert any(o.gpu_type == "CPU_NODE" for o in O.pod_offers(client=FakeAvail(), include_cpu=True))
    best = O.cheapest_pod_offer("H200", 1, offs)
    assert best.gpu_count == 1 and best.price_hr == pytest.approx(5.41)
    sb = O.sandbox_offers(None, offs)
    h200 = [o for o in sb if o.gpu_type == "H200_141GB"][0]
    assert h200.per_gpu_hr == pytest.approx(5.41) and "unknown" in h200.note
    assert "A100_80GB" in O.format_table(offs)


def test_sandbox_price_override(monkeypatch):
    monkeypatch.setenv("HNET_PRIME_SANDBOX_GPU_PRICES", json.dumps({"RTX_PRO_6000": 1.2}))
    sb = O.sandbox_offers("RTX_PRO")
    assert sb[0].per_gpu_hr == 1.2 and "HNET_PRIME" in sb[0].note


# ----------------------------------------------------------------------------- cli with fake backend
class FakeBackend:
    kind = "sandbox"
    instances: dict = {}

    def create(self, name, **kw):
        rid = f"sb{len(self.instances)}"
        self.instances[rid] = {"status": "RUNNING", "kw": kw, "name": name}
        return {"id": rid, "status": "PENDING"}

    def wait_ready(self, rid, timeout_s=0):
        pass

    def status(self, rid):
        return {"status": self.instances.get(rid, {}).get("status", "NOT_FOUND")}

    def exec(self, rid, cmd, timeout=0):
        return ExecResult(0, "ok\n", "")

    def delete(self, rid):
        self.instances[rid]["status"] = "TERMINATED"

    def list_labeled(self):
        return [{"id": k, "name": v["name"], "status": v["status"]} for k, v in self.instances.items()]


@pytest.fixture
def fake(monkeypatch):
    FakeBackend.instances = {}
    fb = FakeBackend()
    monkeypatch.setattr(cli, "_backend", lambda kind: fb)
    monkeypatch.setattr(cli, "wallet_info", lambda ids=None: {"balance": 1000.0, "billed": {}})
    monkeypatch.setattr(O, "pod_offers", lambda gpu=None, client=None, include_cpu=False:
                        FakeAvail and O_offers_cache)
    return fb


O_offers_cache = [O.Offer(kind="pod", gpu_type="H200_141GB", gpu_count=1, price_hr=5.4, per_gpu_hr=5.4,
                          stock="Available")]


def test_cli_launch_guard_and_teardown(fake, capsys):
    # 10 h of an H200-priced sandbox = ~$56 > $50 cap -> refused
    assert cli.main(["launch", "--name", "big", "--gpu", "H200_141GB", "--max-hours", "10"]) == 3
    assert "REFUSED" in capsys.readouterr().out and not fake.instances
    # dry run under cap
    assert cli.main(["launch", "--name", "t", "--gpu", "H200_141GB", "--max-hours", "2", "--dry-run"]) == 0
    assert not fake.instances
    # real (fake) CPU launch, sets platform timeout = ceil(hours*60)
    assert cli.main(["launch", "--name", "t", "--max-hours", "0.25"]) == 0
    (rid, inst), = fake.instances.items()
    assert inst["name"] == "hnet-t" and inst["kw"]["timeout_minutes"] == 15
    assert inst["kw"]["gpu_count"] == 0 and inst["kw"]["image"].startswith("python:")
    st = State()
    rec = st.get("hnet-t")
    assert rec["price_hr"] == pytest.approx(B.sandbox_price_hr(1, 2, 10))
    assert rec["deadline"] - rec["started_at"] == pytest.approx(900)
    # total budget guard counts the active reservation
    assert cli.main(["launch", "--name", "u", "--max-hours", "1", "--budget", "0.01"]) == 3
    assert cli.main(["run", "t", "--", "python", "-c", "print(1)"]) == 0
    assert len(State().get("hnet-t")["runs"]) == 1
    assert cli.main(["cost"]) == 0
    assert cli.main(["teardown", "t"]) == 0
    assert fake.instances[rid]["status"] == "TERMINATED"
    assert State().get("hnet-t")["ended_at"] is not None
    out = capsys.readouterr().out
    assert "verified gone" in out and "verify sandboxs: 0 active" in out


def test_teardown_all_catches_untracked_labeled(fake, capsys):
    fake.instances["stray"] = {"status": "RUNNING", "kw": {}, "name": "hnet-stray"}
    assert cli.main(["teardown", "--all"]) == 0
    assert fake.instances["stray"]["status"] == "TERMINATED"


def test_pick_wheel():
    tag = {"cuda_major": "12", "torch": "2.8", "abi": "TRUE", "py": "cp311"}
    names = ["mamba_ssm-2.3.2.post1+cu12torch2.7cxx11abiTRUE-cp311-cp311-linux_x86_64.whl",
             "mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl",
             "mamba_ssm-2.3.2.post1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"]
    assert gpu_caps.pick_wheel(names, "mamba_ssm", tag) == names[2]
    assert gpu_caps.pick_wheel(names[:2], "mamba_ssm", tag) is None
    assert gpu_caps.pick_wheel(["flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"],
                               "flash_attn", tag)
    assert gpu_caps.wheel_name("causal_conv1d", "1.5.2", tag) == \
        "causal_conv1d-1.5.2+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
