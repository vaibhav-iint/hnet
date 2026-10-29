"""Launch one sweep spec across n spot GPU VMs (one shard each), check on it, tear it down.

    python infra/sweeps/launch.py create   SPEC --vms N [--execute] [--commit SHA] [--gcs gs://...] [--parallel 3]
                                           [--max-hours 6] [--gpu l4] [--zones auto|z1,z2] [--disk-gb 150]
                                           [--budget-cap 25] [--allow-missing] [--no-fairness] [--allow-unpushed]
    python infra/sweeps/launch.py status   SPEC [--gcs gs://...]
    python infra/sweeps/launch.py teardown SPEC [--execute]

``create`` validates the spec (all parameters set, config files present, fairness gate) and that the commit is
pushed and holds this working tree's spec and configs, then prints the exact ``gcloud compute instances create``
command of every VM. VM ``i`` (name ``hx-<spec>-<i>of<n>``) runs ``infra/sweeps/sweep_vm_startup.sh`` with
metadata ``hnet-shard=i/n`` and trains jobs ``i, i+n, ...`` of the spec (scripts/search/spec.py), mirroring them
to ``<gcs>/<job>/``. Each VM is spot (stopped, not deleted, on preemption, so ``status`` can restart it), has a
hard ``--max-run-duration`` and labels ``hnet-sweep=<spec>``, ``hnet-shard=<i>of<n>``. With ``--execute`` the
commands run, trying zones in order and moving on only on capacity/quota errors; existing VMs of the same name
are left alone. The worst-case cost (n x $/h x max hours) must stay under ``--budget-cap``.

``status`` (read-only: always runs) lists the sweep's VMs and each shard's ``sweep_status*.json`` from GCS, and
prints the restart command for VMs that stopped before their shard finished. ``teardown`` deletes every VM
labelled with the sweep (prints the commands; ``--execute`` runs them).

gcloud runs with the isolated config of infra/gcp (``CLOUDSDK_CONFIG``, default the scratchpad gcloud dir; a preset
``CLOUDSDK_AUTH_ACCESS_TOKEN`` is dropped). Nothing here prints or stores credentials.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts" / "search")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shlex  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from infra.gcp import config as C  # noqa: E402
from infra.gcp import gce  # noqa: E402

STARTUP = "infra/sweeps/sweep_vm_startup.sh"
VM_PREFIX = "hx-"
LABEL_OWNER = {"hnet-owner": "sweeps", "hnet-project": "hnet-smollm3"}


class LaunchError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------- gcloud / git
def run(cmd: list[str], check: bool = True, cwd=None) -> subprocess.CompletedProcess:
    """Every external command goes through here (tests mock ``subprocess.run``)."""
    env = gce.gcloud_env() if cmd and cmd[0] == "gcloud" else None
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd or REPO_ROOT)
    if check and r.returncode != 0:
        raise LaunchError(f"{' '.join(cmd[:4])} ... failed (exit {r.returncode}):\n{(r.stderr or '')[-2000:]}")
    return r


def gcloud(*args: str) -> list[str]:
    return ["gcloud", *args, *gce.project_flag()]


def env_header() -> list[str]:
    """Shell lines that select the isolated gcloud config (paths only, no secrets)."""
    lines = ["unset CLOUDSDK_AUTH_ACCESS_TOKEN"]
    cfg = C.gcloud_config_dir()
    if cfg:
        lines.append(f"export CLOUDSDK_CONFIG={shlex.quote(cfg)}")
    return lines + [f"cd {shlex.quote(str(REPO_ROOT))}"]


def check_commit(commit: str, spec_path: str, config_paths: list[str], allow_unpushed: bool) -> str:
    """Resolve ``commit`` and require that it is on a remote branch and holds the local spec/config files."""
    sha = run(["git", "rev-parse", commit]).stdout.strip()
    if allow_unpushed:
        return sha
    remote = run(["git", "branch", "-r", "--contains", sha]).stdout.strip()
    if not remote:
        raise LaunchError(f"commit {sha[:10]} is not on any remote branch: push it first (VMs clone from GitHub)")
    paths = sorted({spec_path, *config_paths})
    r = run(["git", "diff", "--quiet", sha, "--", *paths], check=False)
    if r.returncode != 0:
        raise LaunchError(f"the spec/config files differ between the working tree and {sha[:10]}: commit and push "
                          f"them (VMs read the spec from git): {paths}")
    return sha


# ---------------------------------------------------------------------------------------------- naming
def slug(name: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return s[:maxlen].rstrip("-")


def vm_name(spec_name: str, i: int, n: int) -> str:
    return f"{VM_PREFIX}{slug(spec_name)}-{i}of{n}"


def default_gcs(spec_name: str, spec_gcs: str | None) -> str:
    return spec_gcs or f"gs://{C.BUCKET}/runs/{spec_name}"


@dataclass
class Plan:
    names: list[str]
    specs: list  # gce.CreateSpec
    zones: list[str]
    cost_cap_usd: float


def plan_vms(spec, n: int, *, commit: str, gcs: str, parallel: int, max_hours: float, gpu: str, zones: list[str],
             disk_gb: int, allow_missing: bool) -> Plan:
    machine, accel, count = C.machine_for(gpu, 1)
    specs, names = [], []
    for i in range(n):
        name = vm_name(spec.name, i, n)
        md = {"hnet-commit": commit, "hnet-spec": spec.path, "hnet-gcs": gcs, "hnet-parallel": str(parallel),
              "hnet-shard": f"{i}/{n}"}
        if allow_missing:
            md["hnet-allow-missing"] = "1"
        labels = {**LABEL_OWNER, "hnet-sweep": slug(spec.name, 63), "hnet-shard": f"{i}of{n}"}
        specs.append(gce.CreateSpec(name=name, machine_type=machine, accelerator=accel, gpu_count=count, spot=True,
                                    max_hours=max_hours, disk_gb=disk_gb, startup_path=STARTUP, labels=labels,
                                    service_account=C.VM_SERVICE_ACCOUNT, extra_metadata=md))
        names.append(name)
    per_vm = C.price_hr(machine, spot=True) + C.disk_price_hr(max(disk_gb, C.MIN_DISK_GB))
    return Plan(names, specs, zones, round(n * per_vm * max_hours, 2))


def create_cmd(cs, zone: str) -> list[str]:
    return gcloud(*gce.create_args(cs, zone))


# ---------------------------------------------------------------------------------------------- commands
def load_spec(a, allow_missing: bool = False):
    from spec import Spec, SpecError

    try:
        spec = Spec.load(a.spec)
        jobs = spec.jobs(allow_missing=allow_missing)
    except SpecError as e:
        raise LaunchError(f"spec check failed: {e}")
    return spec, jobs


def cmd_create(a) -> int:
    spec, jobs = load_spec(a, a.allow_missing)
    if spec.missing:
        print(f"# WARNING: runs skipped for missing configs: {spec.missing}")
    if a.vms < 1 or a.vms > len(jobs):
        raise LaunchError(f"--vms must be in 1..{len(jobs)} (the spec has {len(jobs)} jobs)")
    if not a.no_fairness:
        from fairness import main as fairness_main

        if fairness_main([a.spec] + (["--allow-missing"] if a.allow_missing else [])) != 0:
            raise LaunchError("fairness gate failed (see <out_root>/fairness.md); fix the spec first")
    cfg_paths = sorted({p for j in jobs for p in (j.config, j.model) if p})
    commit = check_commit(a.commit, spec.path, cfg_paths, a.allow_unpushed)
    gcs = a.gcs or default_gcs(spec.name, spec.raw.get("gcs"))
    zones = C.ZONES[a.gpu] if a.zones == "auto" else [z.strip() for z in a.zones.split(",") if z.strip()]
    plan = plan_vms(spec, a.vms, commit=commit, gcs=gcs, parallel=a.parallel, max_hours=a.max_hours, gpu=a.gpu,
                    zones=zones, disk_gb=a.disk_gb, allow_missing=a.allow_missing)
    per_shard = Counter(j.index % a.vms for j in spec.jobs(allow_missing=a.allow_missing))
    print(f"# sweep {spec.name}: {len(jobs)} jobs on {a.vms} spot {a.gpu} VMs "
          f"({', '.join(f'shard {i}: {per_shard[i]}' for i in range(a.vms))}), {a.parallel} runs in parallel per VM")
    print(f"# commit {commit}  gcs {gcs}  cap {a.max_hours} h/VM  worst-case cost ${plan.cost_cap_usd:.2f} "
          f"(estimate, {C.machine_for(a.gpu, 1)[0]} spot)")
    if plan.cost_cap_usd > a.budget_cap:
        raise LaunchError(f"worst-case cost ${plan.cost_cap_usd:.2f} exceeds --budget-cap ${a.budget_cap:.2f}; "
                          f"lower --max-hours/--vms or raise the cap deliberately")
    print(f"# zone order (fallback on capacity/quota errors): {', '.join(zones)}")
    for line in env_header():
        print(line)
    for cs in plan.specs:
        print(shlex.join(create_cmd(cs, zones[0])))
    if not a.execute:
        print("# dry run: nothing created. Re-run with --execute to create the VMs.")
        return 0
    created = []
    for cs in plan.specs:
        zone = create_with_fallback(cs, zones)
        created.append((cs.name, zone))
    print("# created: " + ", ".join(f"{n} ({z})" for n, z in created if z))
    print(f"# monitor: python infra/sweeps/launch.py status {a.spec}")
    return 0


def create_with_fallback(cs, zones: list[str]) -> str | None:
    """Create one VM, trying ``zones`` in order. Returns the zone, or None if the VM already exists (left
    alone). Non-capacity errors raise; so does running out of zones."""
    tried = []
    for z in zones:
        r = run(create_cmd(cs, z), check=False)
        if r.returncode == 0:
            print(f"[create] {cs.name} in {z}", flush=True)
            return z
        err = r.stderr or ""
        if "already exists" in err.lower():
            print(f"[create] {cs.name} already exists; left alone (see `status`)", flush=True)
            return None
        if not gce.is_retryable(err):
            raise LaunchError(f"creating {cs.name} in {z} failed (not a capacity error):\n{err[-2000:]}")
        tried.append((z, err.strip().splitlines()[-1][:200] if err.strip() else ""))
        print(f"[create] {cs.name}: {z} unavailable -> next zone", flush=True)
    raise LaunchError(f"no zone could create {cs.name}:\n" + "\n".join(f"  {z}: {e}" for z, e in tried))


def list_vms(spec_name: str) -> list[dict]:
    r = run(gcloud("compute", "instances", "list", f"--filter=labels.hnet-sweep={slug(spec_name, 63)}",
                   "--format=json"))
    return json.loads(r.stdout or "[]")


def read_status_files(gcs: str) -> dict:
    """{status file name: parsed json} for every sweep_status*.json under the prefix."""
    r = run(gcloud("storage", "ls", f"{gcs.rstrip('/')}/sweep_status*.json"), check=False)
    if r.returncode != 0:
        if "matched no objects" in (r.stderr or ""):
            return {}
        raise LaunchError(f"gcloud storage ls {gcs} failed:\n{(r.stderr or '')[-2000:]}")
    out = {}
    for url in [u.strip() for u in r.stdout.splitlines() if u.strip()]:
        cat = run(gcloud("storage", "cat", url))
        out[url.rsplit("/", 1)[-1]] = json.loads(cat.stdout)
    return out


def cmd_status(a) -> int:
    spec, jobs = load_spec(a, allow_missing=True)  # launched jobs never include runs whose configs were missing
    gcs = a.gcs or default_gcs(spec.name, spec.raw.get("gcs"))
    vms = list_vms(spec.name)
    files = read_status_files(gcs)
    states = {}
    for f in files.values():
        states.update(f.get("runs", f))  # v1 status files are a flat {run: state}
    counts = Counter(_bucket(states.get(j.name, "not started")) for j in jobs)
    print(f"sweep {spec.name}: {len(jobs)} jobs  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"{'vm':40s} {'zone':16s} {'vm status':12s} shard  done/total  running  failed  updated")
    shards = {}
    for fname, f in files.items():
        m = re.search(r"shard-(\d+)-of-(\d+)", fname)
        if m:
            shards[int(m.group(1))] = f
    restart = []
    for vm in sorted(vms, key=lambda v: v["name"]):
        zone = (vm.get("zone") or "").rsplit("/", 1)[-1]
        lab = (vm.get("labels") or {}).get("hnet-shard", "")
        m = re.fullmatch(r"(\d+)of(\d+)", lab)
        i, n = (int(m.group(1)), int(m.group(2))) if m else (None, None)
        mine = [j for j in jobs if i is not None and j.index % n == i]
        st = Counter(_bucket(states.get(j.name, "not started")) for j in mine)
        finished = st["done"] + st["diverged"]
        upd = (shards.get(i) or {}).get("updated", "-")
        print(f"{vm['name']:40s} {zone:16s} {vm.get('status', '?'):12s} {lab:6s} {finished:4d}/{len(mine):<6d} "
              f"{st['running']:7d}  {st['failed']:6d}  {upd}")
        if vm.get("status") in ("TERMINATED", "STOPPED", "SUSPENDED") and finished < len(mine) and not st["failed"]:
            restart.append(shlex.join(gcloud("compute", "instances", "start", vm["name"], f"--zone={zone}")))
    if not vms:
        print("(no VMs labelled for this sweep)")
    failed = sorted(k for k, v in states.items() if _bucket(v) == "failed")
    if failed:
        print(f"FAILED runs: {failed}  (logs: {gcs}/<run>/stdout.log, VM logs: {gcs}/_logs/)")
    if restart:
        print("stopped before finishing (spot preemption or max-run-duration); restart (resumes from GCS):")
        for c in restart:
            print("  " + c)
    if counts["done"] + counts["diverged"] == len(jobs):
        print(f"all jobs finished: python scripts/search/report.py {a.spec} --gcs {gcs}; then teardown")
    return 0


def _bucket(state: str) -> str:
    s = str(state)
    if s.startswith("failed"):
        return "failed"
    if s.startswith("skipped"):
        return "skipped"
    return s if s in ("done", "diverged", "running", "pending") else "not started"


def cmd_teardown(a) -> int:
    from spec import Spec

    spec = Spec.load(a.spec)
    vms = list_vms(spec.name)
    if not vms:
        print(f"no VMs labelled hnet-sweep={slug(spec.name, 63)}")
        return 0
    for line in env_header():
        print(line)
    cmds = [gcloud("compute", "instances", "delete", vm["name"],
                   f"--zone={(vm.get('zone') or '').rsplit('/', 1)[-1]}", "--quiet", "--delete-disks=all")
            for vm in vms]
    for c in cmds:
        print(shlex.join(c))
    if not a.execute:
        print("# dry run: nothing deleted. Re-run with --execute to delete these VMs (and their disks).")
        return 0
    for c in cmds:
        run(c)
        print(f"[teardown] deleted {c[4]}", flush=True)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="print (or --execute) the VM create commands for a sharded sweep")
    c.add_argument("spec")
    c.add_argument("--vms", type=int, required=True, help="number of VMs = shards")
    c.add_argument("--execute", action="store_true", help="run the commands (default: print only)")
    c.add_argument("--commit", default="HEAD", help="git commit the VMs check out (must be pushed)")
    c.add_argument("--gcs", default=None, help="gs:// prefix (default: spec gcs, else gs://<bucket>/runs/<name>)")
    c.add_argument("--parallel", type=int, default=3, help="runs trained at once per VM (3 fit a 24 GB L4)")
    c.add_argument("--max-hours", type=float, default=6.0, help="hard --max-run-duration per VM")
    c.add_argument("--gpu", default="l4", choices=sorted(k for k in C.MACHINES if k != "cpu"))
    c.add_argument("--zones", default="auto", help="auto (infra/gcp/config.py order) or z1,z2,...")
    c.add_argument("--disk-gb", type=int, default=150)
    c.add_argument("--budget-cap", type=float, default=C.DEFAULT_LAUNCH_CAP, help="max worst-case $ per launch")
    c.add_argument("--allow-missing", action="store_true", help="launch without runs whose configs are missing")
    c.add_argument("--no-fairness", action="store_true", help="skip the local fairness gate (the VMs still run it)")
    c.add_argument("--allow-unpushed", action="store_true", help="skip the pushed/committed checks")
    s = sub.add_parser("status", help="VM states + per-shard sweep_status from GCS (read-only)")
    s.add_argument("spec")
    s.add_argument("--gcs", default=None)
    t = sub.add_parser("teardown", help="delete every VM of the sweep (print; --execute runs)")
    t.add_argument("spec")
    t.add_argument("--execute", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return {"create": cmd_create, "status": cmd_status, "teardown": cmd_teardown}[a.cmd](a)
    except LaunchError as e:
        print(f"[launch] ERROR: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
