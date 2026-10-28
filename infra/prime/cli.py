"""python -m infra.prime <cmd> -- launch, sync, run, monitor and tear down Prime Intellect machines.

  offers   [--gpu H200] [--cpu]                 GPU offers ($/GPU-h) for pods and sandboxes
  launch   --name N --gpu TYPE --count K --max-hours H [--kind sandbox|pod] [--dry-run]
           [--budget 500] [--launch-cap 50] [--allow-over] [--price-per-gpu-hour P]
           [--cpu C --mem-gb M --disk-gb D --image IMG --volume name:/mnt[:ro] --env K=V]
  sync     N                                      upload the repo (tracked files) and unpack
  setup    N [--no-pytest] [--timeout-min 45]     run infra/prime/setup_gpu.sh remotely
  run      N [--run-id ID] [--env K=V] [--wait] -- CMD...   detached; logs runs/<id>/stdout.log
  ps       N                                      resource status + run status + GPU usage
  tail     N [--run ID] [-n 40] [--metrics]       end of a run's log (+ latest metrics.jsonl)
  pull     N REMOTE LOCAL                         file or directory (relative to repo root)
  exec     N -- CMD...                            run a short command synchronously
  cost                                            uptime x price per resource, spend vs budget
  teardown N | --all                              delete and verify gone
  reap     [--loop SEC]                           mark finished resources; kill ones past deadline
  ls                                              local state

State/ledger: <scratchpad>/prime_state/state.json (HNET_PRIME_STATE_DIR overrides).
Credentials: PRIME_API_KEY / PRIME_TEAM_ID env, or HNET_PRIME_ENV=<env file>.
"""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from . import budget as B
from . import config as C
from . import offers as O
from . import pack, remote
from .backends import GONE, SandboxBackend, PodBackend
from .state import State


def _backend(kind: str):
    return SandboxBackend() if kind == "sandbox" else PodBackend()


def _p(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- wallet
def wallet_info(ids: set[str] | None = None, max_rows: int = 400) -> dict:
    """Balance and billing rows attributed to our resource ids (best effort)."""
    from prime_cli.api.wallet import WalletClient
    from prime_cli.core import APIClient
    wc = WalletClient(APIClient())
    team = os.environ.get("PRIME_TEAM_ID") or None
    w = wc.get(limit=100, offset=0, team_id=team)
    billed: dict[str, float] = {}
    rows = list(w.recent_billings)
    off = len(rows)
    while ids and off < min(max_rows, w.total_billings) and len(rows) == off:
        more = wc.get(limit=100, offset=off, team_id=team).recent_billings
        if not more:
            break
        rows += more
        off += len(more)
    for r in rows:
        if ids and r.resource_id in ids:
            billed[r.resource_id] = billed.get(r.resource_id, 0.0) + r.amount_usd
    return {"balance": w.balance_usd, "billed": billed}


# ----------------------------------------------------------------------------- refresh / reap
def refresh(st: State, kill_overdue: bool = False, grace_s: float = 300) -> None:
    now = time.time()
    for rec in st.active():
        try:
            s = _backend(rec["kind"]).status(rec["id"])
        except Exception as e:  # network hiccup: leave as is
            _p(f"[warn] status {rec['name']}: {e}")
            continue
        rec["status"] = s.get("status")
        if s.get("status") in GONE:
            st.mark_ended(rec["name"], when=s.get("terminated_at") or min(now, rec.get("deadline") or now),
                          status=s["status"])
            _p(f"[reap] {rec['name']} ({rec['id']}) is {s['status']}; marked ended")
        elif kill_overdue and rec.get("deadline") and now > rec["deadline"] + grace_s:
            _p(f"[reap] {rec['name']} past deadline by {(now - rec['deadline']) / 60:.0f} min; deleting")
            _delete_and_verify(st, rec)


def _delete_and_verify(st: State, rec: dict, wait_s: int = 180) -> bool:
    be = _backend(rec["kind"])
    try:
        be.delete(rec["id"])
    except Exception as e:
        if "404" not in str(e) and "not found" not in str(e).lower():
            _p(f"[error] delete {rec['name']}: {e}")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        s = be.status(rec["id"])
        if s.get("status") in GONE:
            st.mark_ended(rec["name"], when=s.get("terminated_at") or time.time(), status=s["status"])
            _p(f"[teardown] {rec['name']} ({rec['id']}): {s['status']} - verified gone")
            return True
        time.sleep(5)
    _p(f"[error] {rec['name']} ({rec['id']}) still {s.get('status')} after {wait_s}s")
    return False


# ----------------------------------------------------------------------------- commands
def cmd_offers(a) -> int:
    pods = O.pod_offers(a.gpu, include_cpu=a.cpu)
    sbx = O.sandbox_offers(a.gpu, pods)
    _p(O.format_table(pods + sbx))
    return 0


def _parse_env(items) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def cmd_launch(a) -> int:
    st = State()
    name = a.name if a.name.startswith(C.NAME_PREFIX) else C.NAME_PREFIX + a.name
    gpu = None if (a.gpu or "").upper() in ("", "CPU", "NONE") else a.gpu
    count = a.count if gpu else 0
    with st.locked():
        refresh(st)
    offer = None
    if a.kind == "sandbox":
        gpu_price = 0.0
        if gpu:
            if a.price_per_gpu_hour is not None:
                gpu_price = a.price_per_gpu_hour
            else:
                ov = O.sandbox_gpu_prices()
                if O._norm(gpu) in ov:
                    gpu_price = ov[O._norm(gpu)]
                else:
                    ref = [o for o in O.pod_offers(gpu) if o.per_gpu_hr]
                    if not ref:
                        _p(f"refusing: no price known for sandbox GPU {gpu}; pass --price-per-gpu-hour")
                        return 2
                    gpu_price = min(o.per_gpu_hr for o in ref)
                    _p(f"[note] sandbox GPU price unknown; assuming pod price ${gpu_price:.2f}/GPU-h")
        cpu = a.cpu if a.cpu is not None else (8 if gpu else 1)
        mem = a.mem_gb if a.mem_gb is not None else (64 if gpu else 2)
        disk = a.disk_gb if a.disk_gb is not None else (120 if gpu else 10)
        price_hr = B.sandbox_price_hr(cpu, mem, disk, count, gpu_price)
    else:
        offers = O.pod_offers(gpu or "CPU_NODE", include_cpu=not gpu)
        offer = O.cheapest_pod_offer(gpu or "CPU_NODE", max(count, 1), offers)
        if offer is None:
            _p(f"refusing: no available pod offer for {gpu or 'CPU'} x{max(count, 1)}")
            return 2
        price_hr = offer.price_hr
        d = offer.raw.disk
        if a.disk_gb and d is not None and d.price_per_unit:
            price_hr += max(0, a.disk_gb - (d.default_count or 0)) * d.price_per_unit
    projected = price_hr * a.max_hours
    committed = B.committed_total(st.all_records())
    try:
        wi = wallet_info()
        bal = wi["balance"]
        with st.locked():
            st.set_wallet_baseline(bal)
    except Exception as e:
        _p(f"[warn] wallet unavailable: {e}")
        bal = None
    g = B.check_launch(projected, committed, a.budget, a.launch_cap, a.allow_over, bal)
    _p(f"launch {name}: kind={a.kind} gpu={gpu or 'none'} x{count} price=${price_hr:.3f}/h "
       f"max_hours={a.max_hours} projected=${projected:.2f} committed=${committed:.2f} "
       f"budget=${a.budget:.0f} cap=${a.launch_cap:.0f} wallet={'?' if bal is None else f'${bal:.2f}'}")
    if not g.ok:
        _p(f"REFUSED ({g.reason})")
        return 3
    if a.dry_run:
        _p("dry run: budget guard OK, nothing created")
        return 0
    env = _parse_env(a.env)
    timeout_min = max(1, math.ceil(a.max_hours * 60))
    be = _backend(a.kind)
    if a.kind == "sandbox":
        image = a.image or (C.DEFAULT_SANDBOX_IMAGE_GPU if gpu else C.DEFAULT_SANDBOX_IMAGE_CPU)
        vols = []
        for v in a.volume or []:
            parts = v.split(":")
            vols.append((parts[0], parts[1], len(parts) > 2 and parts[2] == "ro"))
        info = be.create(name, image=image, cpu=cpu, mem_gb=mem, disk_gb=disk, gpu_type=gpu,
                         gpu_count=count, timeout_minutes=timeout_min,
                         idle_timeout_minutes=a.idle_minutes, env=env, volumes=vols)
        root = C.SANDBOX_REMOTE_ROOT
    else:
        image = a.image or (C.DEFAULT_POD_IMAGE if gpu else "ubuntu_22_cuda_12")
        info = be.create(name, offer=offer.raw, image=image, disk_gb=a.disk_gb, gpu_count=max(count, 1),
                         env=env)
        if info.get("price_hr"):
            price_hr = max(price_hr, info["price_hr"])
        root = C.POD_REMOTE_ROOT
    now = time.time()
    with st.locked():
        st.add({"name": name, "id": info["id"], "kind": a.kind, "gpu_type": gpu, "gpu_count": count,
                "price_hr": price_hr, "started_at": now, "deadline": now + a.max_hours * 3600,
                "status": info.get("status"), "remote_root": root, "image": image})
    _p(f"created {a.kind} {info['id']} as {name}; platform timeout {timeout_min} min"
       if a.kind == "sandbox" else f"created pod {info['id']} as {name}; deadline enforced by reap")
    try:
        be.wait_ready(info["id"])
    except Exception as e:
        _p(f"[error] not ready: {e}")
        if not a.keep_on_error:
            with st.locked():
                _delete_and_verify(st, st.get(name))
        return 4
    if a.kind == "pod":
        mins = max(1, timeout_min)
        be.exec(info["id"], f"(sudo -n shutdown -h +{mins} || shutdown -h +{mins}) >/dev/null 2>&1 || true",
                timeout=60)
    r = be.exec(info["id"], "nproc; free -g | sed -n 2p; df -h / | tail -1; "
                "(nvidia-smi -L 2>/dev/null || echo no-gpu)", timeout=60)
    with st.locked():
        st.get(name)["status"] = "RUNNING"
    _p(f"ready: {name} ({info['id']})\n{r.stdout.strip()}")
    return 0


def _rec(name: str) -> tuple[State, dict]:
    st = State()
    try:
        rec = st.get(name)
    except KeyError:
        rec = st.get(C.NAME_PREFIX + name)
    return st, rec


def cmd_sync(a) -> int:
    st, rec = _rec(a.name)
    be = _backend(rec["kind"])
    with tempfile.TemporaryDirectory(dir=C.state_dir()) as td:
        tp = Path(td) / "repo.tar.gz"
        pack.make_tarball(C.REPO_ROOT, tp)
        n = len(pack.list_files(C.REPO_ROOT))
        tar_remote = f"/tmp/hnet-sync-{int(time.time())}.tar.gz"
        _p(f"uploading {n} files ({tp.stat().st_size / 1e6:.2f} MB) to {rec['name']}")
        be.upload(rec["id"], str(tp), tar_remote)
    r = be.exec(rec["id"], remote.sync_unpack(rec["remote_root"], tar_remote), timeout=300)
    _p((r.stdout + r.stderr).strip())
    return r.exit_code


def _follow(be, rec, run_id: str, timeout_s: float, poll_s: float = 10) -> int:
    root = remote.root_expr(rec["remote_root"])
    off, t0 = 0, time.time()
    while True:
        cmd = (f"cd {root}; f=runs/{run_id}/stdout.log; n=$(wc -c < $f 2>/dev/null || echo 0); "
               f"tail -c +{off + 1} $f 2>/dev/null | head -c $((n-{off})); echo; echo \"@@SIZE $n\"; "
               f"[ -f runs/{run_id}/exit_code ] && echo \"@@EXIT $(cat runs/{run_id}/exit_code)\"; true")
        r = be.exec(rec["id"], cmd, timeout=120)
        lines = r.stdout.split("\n")
        code = None
        body = []
        for ln in lines:
            if ln.startswith("@@SIZE "):
                off = int(ln.split()[1] or 0)
            elif ln.startswith("@@EXIT "):
                code = int(ln.split()[1])
            else:
                body.append(ln)
        txt = "\n".join(body).strip("\n")
        if txt:
            _p(txt)
        if code is not None:
            return code
        if time.time() - t0 > timeout_s:
            _p(f"[timeout] still running after {timeout_s:.0f}s; check with `tail`")
            return 124
        time.sleep(poll_s)


def _start(st, rec, cmd: str, run_id: str | None, env: dict) -> str:
    run_id = run_id or remote.new_run_id(cmd)
    be = _backend(rec["kind"])
    r = be.exec(rec["id"], remote.detached(rec["remote_root"], run_id, cmd, env), timeout=120)
    if r.exit_code != 0:
        raise RuntimeError(f"start failed ({r.exit_code}): {r.stderr.strip()[-800:]}")
    _p(r.stdout.strip())
    with st.locked():
        st.add_run(rec["name"], run_id, cmd)
    return run_id


def cmd_setup(a) -> int:
    st, rec = _rec(a.name)
    env = {"HNET_SETUP_PYTEST": "0" if a.no_pytest else "1"}
    rid = _start(st, rec, "bash infra/prime/setup_gpu.sh", f"setup-{int(time.time())}", env)
    return _follow(_backend(rec["kind"]), rec, rid, a.timeout_min * 60, poll_s=15)


def cmd_run(a) -> int:
    st, rec = _rec(a.name)
    argv = a.cmd
    if not argv:
        _p("nothing to run")
        return 2
    cmd = argv[0] if len(argv) == 1 else shlex.join(argv)
    rid = _start(st, rec, cmd, a.run_id, _parse_env(a.env))
    if a.wait:
        return _follow(_backend(rec["kind"]), rec, rid, a.wait_timeout)
    return 0


def cmd_exec(a) -> int:
    _, rec = _rec(a.name)
    argv = a.cmd
    if not argv:
        _p("nothing to run")
        return 2
    cmd = argv[0] if len(argv) == 1 else shlex.join(argv)
    r = _backend(rec["kind"]).exec(rec["id"], f"cd {remote.root_expr(rec['remote_root'])} 2>/dev/null; {cmd}",
                                    timeout=a.timeout)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.exit_code


def cmd_ps(a) -> int:
    st, rec = _rec(a.name)
    be = _backend(rec["kind"])
    s = be.status(rec["id"])
    left = (rec["deadline"] - time.time()) / 3600 if rec.get("deadline") else float("nan")
    _p(f"{rec['name']} {rec['kind']} {rec['id']} status={s.get('status')} gpu={rec.get('gpu_type')}"
       f"x{rec.get('gpu_count')} ${rec['price_hr']:.3f}/h up={B.actual_hours(rec):.2f}h "
       f"deadline_in={left:.2f}h")
    if s.get("status") in GONE:
        return 0
    runs = [r["id"] for r in rec.get("runs", [])][-8:]
    r = be.exec(rec["id"], remote.ps(rec["remote_root"], runs), timeout=60)
    _p(r.stdout.strip())
    return 0


def cmd_tail(a) -> int:
    _, rec = _rec(a.name)
    runs = rec.get("runs", [])
    rid = a.run or (runs[-1]["id"] if runs else None)
    if rid is None:
        _p("no runs recorded")
        return 2
    r = _backend(rec["kind"]).exec(rec["id"], remote.tail(rec["remote_root"], rid, a.n, a.metrics, a.metrics_n),
                                   timeout=60)
    out = r.stdout.rstrip("\n").split("\n")
    in_metrics = False
    for ln in out:
        if ln.startswith("== ") and ln.endswith("metrics.jsonl"):
            in_metrics = True
            _p(ln)
            continue
        _p(remote.format_metrics_line(ln) if in_metrics and ln.startswith("{") else ln)
    if r.exit_code:
        sys.stderr.write(r.stderr)
    return r.exit_code


def cmd_pull(a) -> int:
    _, rec = _rec(a.name)
    be = _backend(rec["kind"])
    src = remote.resolve_remote_path(rec["remote_root"], a.remote)
    q = remote.root_expr(src) if src.startswith("~") else shlex.quote(src)
    r = be.exec(rec["id"], f"if [ -d {q} ]; then echo DIR; elif [ -f {q} ]; then echo FILE; else echo NONE; fi",
                timeout=60)
    kind = r.stdout.strip().split("\n")[-1]
    if kind == "NONE":
        _p(f"remote path not found: {src}")
        return 2
    if kind == "FILE":
        abs_src = be.exec(rec["id"], f"readlink -f {q}", timeout=60).stdout.strip()
        be.download(rec["id"], abs_src, a.local)
        _p(f"pulled {abs_src} -> {a.local} ({os.path.getsize(a.local)} B)")
        return 0
    tmp = f"/tmp/hnet-pull-{int(time.time())}.tar.gz"
    r = be.exec(rec["id"], f"tar -czf {tmp} -C {q} .", timeout=1800)
    if r.exit_code:
        _p(r.stderr)
        return r.exit_code
    os.makedirs(a.local, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=C.state_dir()) as td:
        lt = os.path.join(td, "pull.tar.gz")
        be.download(rec["id"], tmp, lt)
        with tarfile.open(lt) as tf:
            tf.extractall(a.local, filter="data")
    be.exec(rec["id"], f"rm -f {tmp}", timeout=60)
    _p(f"pulled dir {src} -> {a.local}")
    return 0


def cmd_cost(a) -> int:
    st = State()
    with st.locked():
        refresh(st, kill_overdue=False)
        recs = st.all_records()
        ids = {r["id"] for r in recs}
        bal = None
        try:
            wi = wallet_info(ids)
            bal = wi["balance"]
            for r in recs:
                if r["id"] in wi["billed"]:
                    r["billed_usd"] = round(wi["billed"][r["id"]], 4)
            st.set_wallet_baseline(bal)
        except Exception as e:
            _p(f"[warn] wallet unavailable: {e}")
    now = time.time()
    _p(f"{'name':24} {'kind':7} {'gpu':14} {'$/h':>7} {'hours':>7} {'est$':>8} {'billed$':>8} "
       f"{'committed$':>10} status")
    for r in sorted(recs, key=lambda r: r["started_at"]):
        _p(f"{r['name']:24} {r['kind']:7} {str(r.get('gpu_type') or 'cpu') + 'x' + str(r.get('gpu_count')):14} "
           f"{r['price_hr']:7.3f} {B.actual_hours(r, now):7.2f} {r['price_hr'] * B.actual_hours(r, now):8.2f} "
           f"{r.get('billed_usd') or 0:8.2f} {B.committed_cost(r, now):10.2f} "
           f"{'ended' if r.get('ended_at') else r.get('status')}")
    spent, committed = B.spent_so_far(recs, now), B.committed_total(recs, now)
    _p(f"spent ${spent:.2f}  committed ${committed:.2f}  budget ${a.budget:.2f}  "
       f"remaining ${a.budget - committed:.2f}")
    wb = st.data.get("wallet_baseline")
    if bal is not None and wb:
        _p(f"wallet ${bal:.2f} (baseline ${wb['balance']:.2f} at "
           f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(wb['at']))}Z; team-wide delta "
           f"${wb['balance'] - bal:.2f}, includes other team members' usage)")
    return 0


def cmd_teardown(a) -> int:
    st = State()
    ok = True
    with st.locked():
        if a.all:
            targets = list(st.active())
            for kind in ("sandbox", "pod"):
                try:
                    labeled = _backend(kind).list_labeled()
                except Exception as e:
                    _p(f"[warn] list {kind}s: {e}")
                    continue
                known = {r["id"] for r in st.all_records()}
                for x in labeled:
                    if x["id"] not in known and x["status"] not in GONE:
                        _p(f"[teardown] untracked {kind} {x['id']} ({x['name']}) carries our label; deleting")
                        st.add({"name": f"untracked-{x['id']}", "id": x["id"], "kind": kind,
                                "gpu_type": None, "gpu_count": 0, "price_hr": 0.0,
                                "started_at": time.time(), "deadline": None, "status": x["status"],
                                "remote_root": ""})
                        targets.append(st.get(f"untracked-{x['id']}"))
        else:
            try:
                targets = [st.get(a.name)]
            except KeyError:
                targets = [st.get(C.NAME_PREFIX + a.name)]
        for rec in targets:
            if rec.get("ended_at"):
                _p(f"{rec['name']} already ended")
                continue
            ok &= _delete_and_verify(st, rec)
    # final verification against the platform
    for kind in ("sandbox", "pod"):
        try:
            left = [x for x in _backend(kind).list_labeled() if x["status"] not in GONE]
        except Exception as e:
            _p(f"[warn] verify {kind}s: {e}")
            continue
        if a.all and left:
            ok = False
        tgt = {r["id"] for r in targets}
        bad = [x for x in left if x["id"] in tgt]
        _p(f"verify {kind} list: {len(left)} active with our label/prefix"
           + (f"; NOT GONE: {bad}" if bad else ""))
        ok &= not bad
    return 0 if ok else 1


def cmd_reap(a) -> int:
    while True:
        st = State()
        with st.locked():
            refresh(st, kill_overdue=True, grace_s=a.grace_min * 60)
        if not a.loop:
            return 0
        time.sleep(a.loop)


def cmd_ls(a) -> int:
    st = State()
    _p(f"state: {st.path}")
    for r in st.all_records():
        _p(f"{r['name']:24} {r['kind']:7} {r['id']:28} {r.get('status')!s:12} "
           f"{'ended' if r.get('ended_at') else 'active'} runs={len(r.get('runs', []))}")
    return 0


# ----------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m infra.prime", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("offers")
    s.add_argument("--gpu")
    s.add_argument("--cpu", action="store_true", help="include CPU pod nodes")
    s.set_defaults(fn=cmd_offers)

    s = sp.add_parser("launch")
    s.add_argument("--name", required=True)
    s.add_argument("--gpu", default="CPU", help="GPU type (e.g. H200_141GB, RTX_PRO_6000) or CPU")
    s.add_argument("--count", type=int, default=1)
    s.add_argument("--max-hours", type=float, required=True)
    s.add_argument("--kind", choices=["sandbox", "pod"], default="sandbox")
    s.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_TOTAL)
    s.add_argument("--launch-cap", type=float, default=C.DEFAULT_LAUNCH_CAP)
    s.add_argument("--allow-over", action="store_true", help="lift the per-launch cap (not the total)")
    s.add_argument("--price-per-gpu-hour", type=float)
    s.add_argument("--cpu", type=float)
    s.add_argument("--mem-gb", type=float)
    s.add_argument("--disk-gb", type=int)
    s.add_argument("--image")
    s.add_argument("--idle-minutes", type=int, default=None,
                   help="sandbox idle timeout; detached jobs do NOT count as activity")
    s.add_argument("--volume", action="append")
    s.add_argument("--env", action="append")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--keep-on-error", action="store_true")
    s.set_defaults(fn=cmd_launch)

    for nm, fn in (("sync", cmd_sync), ("ps", cmd_ps)):
        s = sp.add_parser(nm)
        s.add_argument("name")
        s.set_defaults(fn=fn)

    s = sp.add_parser("setup")
    s.add_argument("name")
    s.add_argument("--no-pytest", action="store_true")
    s.add_argument("--timeout-min", type=float, default=45)
    s.set_defaults(fn=cmd_setup)

    s = sp.add_parser("run")
    s.add_argument("name")
    s.add_argument("--run-id")
    s.add_argument("--env", action="append")
    s.add_argument("--wait", action="store_true")
    s.add_argument("--wait-timeout", type=float, default=3600)
    s.add_argument("cmd", nargs="*")
    s.set_defaults(fn=cmd_run)

    s = sp.add_parser("exec")
    s.add_argument("name")
    s.add_argument("--timeout", type=int, default=300)
    s.add_argument("cmd", nargs="*")
    s.set_defaults(fn=cmd_exec)

    s = sp.add_parser("tail")
    s.add_argument("name")
    s.add_argument("--run")
    s.add_argument("-n", type=int, default=40)
    s.add_argument("--metrics", action="store_true")
    s.add_argument("--metrics-n", type=int, default=5)
    s.set_defaults(fn=cmd_tail)

    s = sp.add_parser("pull")
    s.add_argument("name")
    s.add_argument("remote")
    s.add_argument("local")
    s.set_defaults(fn=cmd_pull)

    s = sp.add_parser("cost")
    s.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_TOTAL)
    s.set_defaults(fn=cmd_cost)

    s = sp.add_parser("teardown")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("name", nargs="?")
    g.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_teardown)

    s = sp.add_parser("reap")
    s.add_argument("--loop", type=float, default=0, help="repeat every N seconds")
    s.add_argument("--grace-min", type=float, default=5)
    s.set_defaults(fn=cmd_reap)

    s = sp.add_parser("ls")
    s.set_defaults(fn=cmd_ls)
    return p


def main(argv=None) -> int:
    C.load_credentials()
    argv = list(sys.argv[1:] if argv is None else argv)
    tail = None
    if "--" in argv:  # everything after the first `--` is the remote command (run / exec)
        i = argv.index("--")
        argv, tail = argv[:i], argv[i + 1:]
    a = build_parser().parse_args(argv)
    if tail is not None:
        a.cmd = tail
    return int(a.fn(a) or 0)
