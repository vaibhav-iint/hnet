"""python -m infra.gcp <cmd> -- GCE GPU runner that needs only HTTPS (gcloud API calls; no SSH, no GCS).

  launch   --name N --gpu l4|a100|t4|cpu [--count 1] (--spot | --on-demand) --max-hours H
           --cmd "<shell, run in the repo>" [--commit SHA] [--zone auto|Z1,Z2] [--disk-gb 100]
           [--data auto|none|DATASET:N[,..]] [--linger-min 5] [--no-shutdown] [--metrics-every K]
           [--overlay PATH] [--env K=V] [--allow-compile] [--no-kernels] [--price-hr P] [--budget 300] [--launch-cap 25]
           [--allow-over] [--dry-run] [--bucket B|none] [--sa EMAIL|none] [--sync-every-min 5]
           [--restore-from VM]
  status   N                     instance state, guest attributes, latest HNET_STATUS / stage / exit
  tail     N [-n 60] [--all]     raw serial output fetched since the last tail (offset tracked)
  sync-metrics N [--out DIR]     rebuild DIR/<run>/metrics.jsonl + eval.jsonl (default runs/gcp/<N>) for
                                 hnet.train.plot / hnet.eval.report: the bucket copy (synced every
                                 --sync-every-min) plus newer lines from the serial console
  pull     N RUN [--ckpt latest|all|none] [--out DIR]   run dir incl. checkpoints from the bucket
  watch    N [--interval 60] [--teardown] [--timeout-min M]   sync until HNET_DONE and the VM stopped
  resume   N [--max-hours H] [--rerun] [--cmd C] [--commit SHA]
                                 start a stopped (preempted / capped) VM again; the run resumes.
                                 --rerun/--cmd/--commit push a new job (startup-script metadata) first
  stop     N                     stop a running VM (keeps the disk)
  teardown N | --all             delete (with disk) and verify gone; --all = every hnet- VM we own
  ls                             ledger records + live hnet-* instances
  cost     [--budget 300]        uptime x estimated $/h per VM, spend vs budget

Bucket (default gs://neat-dispatch-501718-a5-hnet, HNET_GCP_BUCKET): runs/<vm>/<run>/ (jsonl, config,
checkpoints, rsynced every few minutes and at the end), runs/<vm>/_logs/, data/<dataset>/ (shard cache:
new VMs copy shards instead of downloading/converting). A VM whose disk has no checkpoint for a run
restores the newest complete one from runs/<vm or --restore-from>/<run>/ckpt before starting.

State: <scratchpad>/gcp_state/ (HNET_GCP_STATE_DIR overrides): state.json (ledger), serial/<N>/,
startup/<N>.sh. gcloud: CLOUDSDK_CONFIG (default <scratchpad>/gcloud); CLOUDSDK_AUTH_ACCESS_TOKEN is
ignored. Prices in infra/gcp/config.py are estimates (HNET_GCP_PRICES overrides).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from infra.prime.state import State

from . import config as C
from . import gce
from . import ledger as L
from . import serial as S
from . import startup


def _p(*a):
    print(*a, flush=True)


def _state() -> State:
    return State(C.state_dir() / "state.json")


def _norm_name(n: str) -> str:
    return n if n.startswith(C.NAME_PREFIX) else C.NAME_PREFIX + n


def _rec(st: State, name: str) -> dict:
    return st.get(_norm_name(name))


def _tracker(name: str) -> S.SerialTracker:
    return S.SerialTracker(C.state_dir() / "serial" / name)


def refresh(st: State, names: list[str] | None = None) -> None:
    """Re-observe active records (status, segments) under the state lock."""
    with st.locked():
        for rec in st.active():
            if names and rec["name"] not in names:
                continue
            if not rec.get("zone"):
                continue
            try:
                inst = gce.describe(rec["name"], rec["zone"])
            except gce.GcloudError as e:
                _p(f"[warn] describe {rec['name']}: {e}")
                continue
            L.apply_observation(rec, inst)


# ----------------------------------------------------------------------------- git helpers
def resolve_commit(commit: str | None) -> tuple[str, bool]:
    """(full sha, pushed?) for `commit` (default HEAD) in the local repo."""
    ref = commit or "HEAD"
    try:
        sha = subprocess.run(["git", "-C", str(C.REPO_ROOT), "rev-parse", ref], capture_output=True, text=True,
                             timeout=30).stdout.strip()
    except Exception:
        sha = ""
    if not sha:
        if commit and len(commit) == 40:
            return commit, True  # trust a full sha we cannot resolve locally
        raise ValueError(f"cannot resolve commit {ref!r}")
    try:
        r = subprocess.run(["git", "-C", str(C.REPO_ROOT), "branch", "-r", "--contains", sha], capture_output=True,
                           text=True, timeout=30)
        pushed = bool(r.stdout.strip())
    except Exception:
        pushed = False
    return sha, pushed


def missing_configs(cmd: str, sha: str, overlay: list[str]) -> list[str]:
    """--config files of hnet.train.train calls in `cmd` that are neither in commit `sha` nor overlaid."""
    from .vm_agent import _train_invocations
    out = []
    for cfg, _ in _train_invocations(cmd):
        if any(cfg == o.rstrip("/") or cfg.startswith(o.rstrip("/") + "/") for o in overlay):
            continue
        try:
            r = subprocess.run(["git", "-C", str(C.REPO_ROOT), "cat-file", "-e", f"{sha}:{cfg}"],
                               capture_output=True, timeout=30)
        except Exception:
            continue
        if r.returncode != 0:
            out.append(cfg)
    return out


# ----------------------------------------------------------------------------- launch
def _parse_env(items) -> dict:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def cmd_launch(a) -> int:
    if a.spot is None:
        _p("refusing: choose --spot or --on-demand explicitly")
        return 2
    name = _norm_name(a.name)
    try:
        mt, acc, count = C.machine_for(a.gpu, a.count)
    except ValueError as e:
        _p(f"refusing: {e}")
        return 2
    if not (0 < a.max_hours <= 168):
        _p("refusing: --max-hours must be in (0, 168]")
        return 2
    disk_gb = max(a.disk_gb or C.MIN_DISK_GB, C.MIN_DISK_GB)
    price = a.price_hr if a.price_hr is not None else C.price_hr(mt, a.spot)
    disk_hr = C.disk_price_hr(disk_gb)
    st = _state()
    refresh(st)
    try:
        old = st.get(name)
        if not old.get("ended_at"):
            _p(f"refusing: {name} is still in the ledger (status {old.get('status')}); teardown first")
            return 2
    except KeyError:
        pass
    projected = L.projected_cost(price, disk_hr, a.max_hours)
    committed = L.committed_total(st.all_records())
    g = L.check_launch(projected, committed, a.budget, a.launch_cap, a.allow_over)
    _p(f"launch {name}: {mt}{f' + {count}x {acc}' if acc else ''} gpu={a.gpu} x{count} "
       f"{'spot' if a.spot else 'on-demand'} est ${price:.2f}/h + disk ${disk_hr:.3f}/h  max_hours={a.max_hours} "
       f"projected=${projected:.2f} committed=${committed:.2f} budget=${a.budget:.0f} cap=${a.launch_cap:.0f}")
    if not g.ok:
        _p(f"REFUSED ({g.reason})")
        return 3
    try:
        sha, pushed = resolve_commit(a.commit)
    except ValueError as e:
        _p(f"refusing: {e}")
        return 2
    if not pushed and not a.allow_unpushed:
        _p(f"refusing: commit {sha[:12]} is not on any remote branch (push it, or --allow-unpushed)")
        return 2
    missing = missing_configs(a.cmd, sha, a.overlay or [])
    if missing:
        _p(f"refusing: {', '.join(missing)} not in commit {sha[:12]} (untracked or .gitignored?). "
           f"Commit it, or ship it with --overlay {missing[0]}")
        return 2
    job = startup.job_dict(repo=a.repo or C.REPO_URL, commit=sha, cmd=a.cmd, data=a.data, linger_min=a.linger_min,
                           auto_shutdown=not a.no_shutdown, metrics_every=a.metrics_every,
                           kernels=not a.no_kernels, allow_compile=a.allow_compile, env=_parse_env(a.env),
                           watch_dirs=a.watch_dir)
    bucket = None if (a.bucket or "").lower() in ("", "none") else a.bucket.replace("gs://", "").strip("/")
    sa = None if (a.sa or "").lower() == "none" else (a.sa or (C.VM_SERVICE_ACCOUNT if bucket else None))
    if bucket and not sa:
        _p("[note] --sa none: the VM has no credentials, so the bucket is not used (serial only)")
        bucket = None
    if bucket:
        job["gcs"] = {"bucket": bucket, "prefix": f"runs/{name}",
                      "restore_prefix": f"runs/{_norm_name(a.restore_from) if a.restore_from else name}",
                      "data_prefix": "data", "sync_every_s": int(a.sync_every_min * 60)}
    if a.overlay:
        try:
            job["overlay_b64"] = startup.make_overlay(a.overlay, C.REPO_ROOT)
        except (OSError, ValueError) as e:
            _p(f"refusing: {e}")
            return 2
        _p(f"overlay: {', '.join(a.overlay)} ({len(job['overlay_b64']) * 3 // 4} bytes gz)")
    sdir = C.state_dir() / "startup"
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{name}.sh"
    spath.write_text(startup.render(job))
    spec = gce.CreateSpec(name=name, machine_type=mt, accelerator=acc, gpu_count=count, spot=a.spot,
                          max_hours=a.max_hours, disk_gb=disk_gb, startup_path=str(spath), service_account=sa)
    zones = C.ZONES[a.gpu.lower()] if a.zone in (None, "", "auto") else a.zone.split(",")
    _p(f"commit {sha[:12]}  cmd: {a.cmd}")
    _p(f"bucket: {'gs://' + bucket + '/runs/' + name if bucket else 'off (serial console only)'}  "
       f"service account: {sa or 'none'}")
    _p(f"zones: {', '.join(zones)}")
    if a.dry_run:
        _p("dry run: budget guard OK, nothing created. gcloud args:")
        _p("  gcloud " + " ".join(gce.create_args(spec, zones[0])))
        return 0
    now = time.time()
    rec = {"name": name, "id": name, "kind": "gce", "zone": None, "machine_type": mt, "gpu": a.gpu.lower(),
           "gpu_count": count, "spot": a.spot, "price_hr": price, "disk_gb": disk_gb, "disk_hr": disk_hr,
           "max_hours": a.max_hours, "created_at": now, "started_at": now, "status": "CREATING", "commit": sha,
           "cmd": a.cmd, "segments": [{"start": now, "deadline": now + a.max_hours * 3600, "end": None}],
           "meta": {"data": a.data, "linger_min": a.linger_min, "job": job, "startup_path": str(spath)}}
    with st.locked():
        st.add(rec)
    res = gce.create_with_fallback(spec, zones, log=_p)
    with st.locked():
        r = st.get(name)
        if not res.ok:
            r["segments"][-1]["end"] = r["segments"][-1]["start"]
            r["ended_at"] = time.time()
            r["created_at"] = r["ended_at"]
            r["status"] = "CREATE_FAILED"
            r["meta"]["tried"] = res.tried
        else:
            t = time.time()
            r["zone"] = res.zone
            r["status"] = (res.instance or {}).get("status", "RUNNING")
            r["segments"][-1].update({"start": t, "deadline": t + a.max_hours * 3600})
            r["meta"]["tried"] = res.tried
    if not res.ok:
        _p("FAILED: no zone could create the VM:")
        for z, e in res.tried:
            _p(f"  {z}: {e}")
        return 4
    _p(f"created {name} in {res.zone}. Follow with:\n  python -m infra.gcp watch {name}\n"
       f"  python -m infra.gcp sync-metrics {name}")
    return 0


# ----------------------------------------------------------------------------- serial-based commands
def _fetch(rec: dict, tr: S.SerialTracker | None = None) -> str:
    tr = tr or _tracker(rec["name"])
    return tr.fetch(lambda start: gce.serial_output(rec["name"], rec["zone"], start))


def _sync(rec: dict, out: str | None = None, quiet: bool = False) -> S.Parsed:
    tr = _tracker(rec["name"])
    try:
        _fetch(rec, tr)
    except gce.GcloudError as e:
        if not quiet:
            _p(f"[warn] serial fetch failed ({str(e)[:200]}); using the local copy")
    p = S.parse_text(tr.read_raw())
    out_root = Path(out) if out else C.REPO_ROOT / "runs" / "gcp" / rec["name"]
    src = "serial"
    g = _gcs_records(rec, quiet)
    if g is not None and (g[0] or g[1]):
        p.metrics = S.merge_sources(g[0], p.metrics)
        p.evals = S.merge_sources(g[1], p.evals)
        src = "bucket+serial"
    dirs = S.reconstruct(p, out_root)
    if not quiet:
        _p(f"  source: {src}")
        for d in dirs:
            rd = next((k for k in set(p.metrics) | set(p.evals) if S.local_run_dir(out_root, k) == d), "?")
            _p(f"  {rd} -> {d}  metrics={len(p.metrics.get(rd, []))} eval={len(p.evals.get(rd, []))}")
        if p.corrupt or p.gaps or p.resets:
            _p(f"  [note] corrupt lines={p.corrupt} gaps={p.gaps} (lost {tr.cur['lost_bytes']} bytes) "
               f"resets={p.resets}")
    return p


def _bucket_of(rec: dict) -> str | None:
    gcs = ((rec.get("meta") or {}).get("job") or {}).get("gcs") or {}
    return gcs.get("bucket")


def _gcs_records(rec: dict, quiet: bool = False) -> tuple[dict, dict] | None:
    """(metrics, evals) keyed like serial run dirs ('runs/<run>') from gs://B/runs/<vm>/; None if no bucket."""
    bucket = _bucket_of(rec)
    if not bucket:
        return None
    base = f"gs://{bucket}/runs/{rec['name']}/"
    try:
        objs = gce.storage_ls_sizes(base)
    except gce.GcloudError as e:
        if not quiet:
            _p(f"[warn] bucket listing failed ({str(e)[:200]}); serial only")
        return None
    want = [u for u in objs if u.endswith(("/metrics.jsonl", "/eval.jsonl")) and "/_logs/" not in u]
    cache = C.state_dir() / "gcs" / rec["name"]
    metrics, evals = {}, {}
    for u in want:
        rel = u[len(base):]
        dst = cache / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            gce.storage_cp([u], str(dst), timeout=600)
        except gce.GcloudError as e:
            if not quiet:
                _p(f"[warn] {u}: {str(e)[:200]}")
            continue
        run = "runs/" + rel.rsplit("/", 1)[0]
        recs = S.read_jsonl_text(dst.read_text())
        (metrics if rel.endswith("metrics.jsonl") else evals)[run] = recs
    return metrics, evals


def cmd_pull(a) -> int:
    """Copy a run dir (checkpoints included) from the bucket: gs://B/runs/<vm>/<run>/ -> <out>."""
    st = _state()
    rec = _rec(st, a.name)
    bucket = a.bucket or _bucket_of(rec) or C.BUCKET
    run = a.run.strip("/")
    if run.startswith("runs/"):
        run = run[5:]
    src = f"gs://{bucket}/runs/{rec['name']}/{run}"
    out = Path(a.out) if a.out else C.REPO_ROOT / "runs" / "gcp" / rec["name"] / run
    out.mkdir(parents=True, exist_ok=True)
    objs = gce.storage_ls_sizes(src)
    if not objs:
        _p(f"nothing under {src}/")
        return 1
    rels = [u[len(src) + 1:] for u in objs]
    from .vm_agent import latest_complete_ckpt
    if a.ckpt == "latest":
        step = latest_complete_ckpt(rels)
        keep = [r for r in rels if not r.startswith("ckpt/") or (step and r.startswith(f"ckpt/{step}/"))]
    elif a.ckpt == "none":
        keep = [r for r in rels if not r.startswith("ckpt/")]
    else:
        keep = rels
    total = sum(objs[f"{src}/{r}"] for r in keep)
    _p(f"pull {src}/ -> {out}: {len(keep)} objects, {total / 1e6:.1f} MB (ckpt={a.ckpt})")
    by_dir: dict[str, list[str]] = {}
    for r in keep:
        by_dir.setdefault(str((out / r).parent), []).append(f"{src}/{r}")
    for d, urls in by_dir.items():
        Path(d).mkdir(parents=True, exist_ok=True)
        gce.storage_cp(urls, d + "/", timeout=6 * 3600)
    _p("done")
    return 0


def _fmt_status(p: S.Parsed) -> str:
    parts = []
    if p.stages:
        parts.append(f"stage={p.stages[-1].get('stage')}")
    if p.status:
        d = dict(p.status.data)
        last = d.pop("last", None)
        parts.append("status=" + json.dumps(d, separators=(",", ":")))
        if last:
            parts.append(f"last: {last}")
    if p.done is not None:
        parts.append(f"DONE exit={p.done}")
    return "  ".join(parts) or "(no HNET_ lines yet)"


def cmd_status(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    refresh(st, [rec["name"]])
    rec = _rec(st, a.name)
    _p(f"{rec['name']}: {rec.get('status')} zone={rec.get('zone')} {rec['machine_type']} "
       f"{'spot' if rec.get('spot') else 'on-demand'} up {L.running_hours(rec):.2f} h "
       f"est ${L.actual_cost(rec):.2f}")
    if rec.get("status") in L.RUNNING:
        ga = gce.guest_attributes(rec["name"], rec["zone"])
        if ga:
            _p("guest attributes: " + json.dumps(ga))
    if rec.get("zone") and rec.get("status") not in ("DELETED",):
        p = _sync(rec, quiet=True)
        _p(_fmt_status(p))
        if p.info:
            _p("info: " + json.dumps({k: p.info.get(k) for k in ("gpus", "torch", "torch_cuda", "mamba_ssm",
                                                                    "causal_conv1d", "hnet_kernels")}))
        if p.summary:
            _p("summary: " + json.dumps(p.summary))
    return 0


def cmd_tail(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    tr = _tracker(rec["name"])
    before = tr.raw.stat().st_size if tr.raw.exists() else 0
    try:
        _fetch(rec, tr)
    except gce.GcloudError as e:
        _p(f"[warn] serial fetch failed: {str(e)[:300]}")
    raw = tr.read_raw()
    if a.all:
        text = raw
    else:
        with open(tr.raw, "rb") as f:
            f.seek(before)
            text = f.read().decode("utf-8", "replace")
    lines = text.splitlines()
    if a.hnet_only:
        lines = [ln for ln in lines if "HNET_" in ln or "[hnet-gcp]" in ln]
    for ln in lines[-a.n:] if a.n else lines:
        _p(ln)
    _p(f"[tail] next offset {tr.cur['next']} (lost {tr.cur['lost_bytes']} bytes so far)")
    return 0


def cmd_sync(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    p = _sync(rec, a.out)
    _p(_fmt_status(p))
    return 0


def cmd_watch(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    t0 = time.time()
    seen_stage = None
    while True:
        refresh(st, [rec["name"]])
        rec = _rec(st, a.name)
        p = _sync(rec, a.out, quiet=True)
        stage = p.stages[-1].get("stage") if p.stages else None
        if stage != seen_stage:
            seen_stage = stage
        n_m = sum(len(v) for v in p.metrics.values())
        _p(f"[{time.strftime('%H:%M:%S')}] {rec.get('status')} metrics={n_m} {_fmt_status(p)}")
        if rec.get("status") in L.STOPPED + ("DELETED",) and rec.get("status") != "STOPPING":
            if p.done is not None or rec.get("status") == "DELETED":
                break
            _p("[watch] VM stopped without HNET_DONE (preempted or hit max-run-duration); "
               f"`python -m infra.gcp resume {rec['name']}` continues it")
            return 5
        if a.timeout_min and time.time() - t0 > a.timeout_min * 60:
            _p("[watch] timeout")
            return 6
        time.sleep(a.interval)
    _p(f"[watch] finished: exit={p.done} status={rec.get('status')}")
    if p.summary:
        _p("summary: " + json.dumps(p.summary))
    if a.teardown and rec.get("status") != "DELETED":
        return _teardown_one(st, rec)
    return 0 if (p.done in (0, None)) else 1


def cmd_resume(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    refresh(st, [rec["name"]])
    rec = _rec(st, a.name)
    if rec.get("status") in L.RUNNING:
        _p(f"{rec['name']} is already {rec['status']}")
        return 0
    if rec.get("status") == "DELETED" or rec.get("ended_at"):
        _p(f"{rec['name']} no longer exists")
        return 2
    hours = a.max_hours or rec["max_hours"]
    committed = L.committed_total(st.all_records())
    projected = rec["price_hr"] * hours
    g = L.check_launch(projected, committed, a.budget, a.launch_cap, a.allow_over)
    _p(f"resume {rec['name']}: projected ${projected:.2f} committed ${committed:.2f} budget ${a.budget:.0f}")
    if not g.ok:
        _p(f"REFUSED ({g.reason})")
        return 3
    if a.rerun or a.cmd or a.commit or a.overlay:
        job = dict(rec.get("meta", {}).get("job") or {})
        if not job:
            _p("refusing: no job recorded for this VM; cannot re-run")
            return 2
        if a.cmd:
            job["cmd"] = a.cmd
        if a.overlay:
            job["overlay_b64"] = startup.make_overlay(a.overlay, C.REPO_ROOT)
        if a.commit:
            sha, pushed = resolve_commit(a.commit)
            if not pushed and not a.allow_unpushed:
                _p(f"refusing: commit {sha[:12]} is not on any remote branch")
                return 2
            job["commit"] = sha
        if a.bucket and a.bucket.lower() != "none" and not job.get("gcs"):
            job["gcs"] = {"bucket": a.bucket.replace("gs://", "").strip("/"), "prefix": f"runs/{rec['name']}",
                          "restore_prefix": f"runs/{rec['name']}", "data_prefix": "data", "sync_every_s": 300}
        if job.get("gcs"):
            inst = gce.describe(rec["name"], rec["zone"]) or {}
            if not inst.get("serviceAccounts"):
                gce.run(["compute", "instances", "set-service-account", rec["name"], f"--zone={rec['zone']}",
                         f"--service-account={C.VM_SERVICE_ACCOUNT}", f"--scopes={C.VM_SCOPES}"], json_out=False)
                _p(f"attached service account {C.VM_SERVICE_ACCOUNT} (bucket access)")
        job["nonce"] = startup.job_dict(repo="", commit="", cmd="")["nonce"]
        spath = C.state_dir() / "startup" / f"{rec['name']}.sh"
        spath.parent.mkdir(parents=True, exist_ok=True)
        spath.write_text(startup.render(job))
        gce.run(["compute", "instances", "add-metadata", rec["name"], f"--zone={rec['zone']}",
                 f"--metadata-from-file=startup-script={spath}"], json_out=False)
        with st.locked():
            st.get(rec["name"])["meta"]["job"] = job
            st.get(rec["name"])["cmd"] = job["cmd"]
            st.get(rec["name"])["commit"] = job["commit"]
        _p(f"new job {job['nonce']}: re-runs `{job['cmd']}` at {job['commit'][:12]} (disk caches reused)")
    if hours != rec["max_hours"]:
        gce.run(["compute", "instances", "set-scheduling", rec["name"], f"--zone={rec['zone']}",
                 f"--max-run-duration={gce.max_run_duration(hours)}"], json_out=False)
    for attempt in range(a.retries + 1):
        try:
            gce.start(rec["name"], rec["zone"])
            break
        except gce.GcloudError as e:
            # a spot VM's disk pins it to its zone; capacity there comes and goes
            if attempt == a.retries or not gce.is_retryable(e.stderr):
                raise
            _p(f"[resume] {gce._short(e.stderr)} -> retry in {a.retry_wait:.0f}s ({attempt + 1}/{a.retries})")
            time.sleep(a.retry_wait)
    with st.locked():
        r = st.get(rec["name"])
        t = time.time()
        r["max_hours"] = hours
        r.setdefault("segments", []).append({"start": t, "deadline": t + hours * 3600, "end": None})
        r["status"] = "RUNNING"
    _p(f"started {rec['name']}; the agent resumes the command (trainer resume=auto)")
    return 0


def cmd_stop(a) -> int:
    st = _state()
    rec = _rec(st, a.name)
    gce.stop(rec["name"], rec["zone"])
    refresh(st, [rec["name"]])
    _p(f"{rec['name']}: {_rec(st, a.name).get('status')}")
    return 0


def _teardown_one(st: State, rec: dict, wait_s: int = 300) -> int:
    name, zone = rec["name"], rec.get("zone")
    if not name.startswith(C.NAME_PREFIX):
        _p(f"refusing to delete {name}: not an {C.NAME_PREFIX} VM")
        return 2
    if zone:
        try:
            gce.delete(name, zone)
        except gce.GcloudError as e:
            _p(f"[error] delete {name}: {str(e)[:300]}")
        t0 = time.time()
        while time.time() - t0 < wait_s:
            if gce.describe(name, zone) is None:
                break
            time.sleep(5)
        else:
            _p(f"[error] {name} still exists after {wait_s}s")
            return 1
    with st.locked():
        try:
            L.apply_observation(st.get(name), None)
        except KeyError:
            pass
    _p(f"[teardown] {name} ({zone}): deleted - verified gone")
    return 0


def cmd_teardown(a) -> int:
    st = _state()
    if a.all:
        rc = 0
        targets = {r["name"]: r for r in st.active()}
        for inst in gce.list_ours():
            labels = inst.get("labels") or {}
            if labels.get("hnet-owner") != C.LABELS["hnet-owner"]:
                _p(f"[skip] {inst['name']}: hnet- name but not labelled by infra.gcp")
                continue
            targets.setdefault(inst["name"], {"name": inst["name"], "zone": gce.zone_of(inst)})
            targets[inst["name"]]["zone"] = gce.zone_of(inst)
        if not targets:
            _p("nothing to tear down")
        for rec in targets.values():
            rc |= _teardown_one(st, rec)
        return rc
    return _teardown_one(st, _rec(st, a.name))


def cmd_ls(a) -> int:
    st = _state()
    refresh(st)
    recs = sorted(st.data["resources"].values(), key=lambda r: r.get("created_at", 0))
    _p(f"{'name':28} {'zone':16} {'machine':16} {'kind':9} {'status':14} {'hours':>6} {'est $':>7}")
    for r in recs:
        _p(f"{r['name']:28} {str(r.get('zone')):16} {r['machine_type']:16} {'spot' if r.get('spot') else 'ondemand':9} "
           f"{str(r.get('status')):14} {L.running_hours(r):6.2f} {L.actual_cost(r):7.2f}")
    try:
        live = gce.list_ours()
        _p(f"live hnet-* instances: {', '.join(i['name'] + ':' + i.get('status', '?') for i in live) or 'none'}")
    except gce.GcloudError as e:
        _p(f"[warn] list failed: {str(e)[:200]}")
    return 0


def cmd_cost(a) -> int:
    st = _state()
    refresh(st)
    recs = st.all_records()
    _p("ESTIMATES (uptime x configured $/h, see infra/gcp/config.py), not billing data")
    _p(f"{'name':28} {'machine':16} {'kind':9} {'$/h':>6} {'run h':>6} {'actual':>8} {'committed':>9} status")
    for r in sorted(recs, key=lambda r: r.get("created_at", 0)):
        _p(f"{r['name']:28} {r['machine_type']:16} {'spot' if r.get('spot') else 'ondemand':9} {r['price_hr']:6.2f} "
           f"{L.running_hours(r):6.2f} {L.actual_cost(r):8.2f} {L.committed_cost(r):9.2f} {r.get('status')}")
    act, com = L.actual_total(recs), L.committed_total(recs)
    _p(f"total: actual ${act:.2f}  committed ${com:.2f}  budget ${a.budget:.0f}  remaining ${a.budget - com:.2f}")
    return 0


# ----------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m infra.gcp", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    s = sp.add_parser("launch")
    s.add_argument("--name", required=True)
    s.add_argument("--gpu", required=True, choices=["l4", "a100", "t4", "cpu"], type=str.lower)
    s.add_argument("--count", type=int, default=1)
    g = s.add_mutually_exclusive_group()
    g.add_argument("--spot", dest="spot", action="store_true", default=None)
    g.add_argument("--on-demand", dest="spot", action="store_false")
    s.add_argument("--max-hours", type=float, required=True)
    s.add_argument("--cmd", required=True)
    s.add_argument("--commit", default=None, help="git sha/ref (default local HEAD; must be pushed)")
    s.add_argument("--repo", default=None, help=f"clone URL (default {C.REPO_URL})")
    s.add_argument("--allow-unpushed", action="store_true")
    s.add_argument("--zone", default="auto")
    s.add_argument("--disk-gb", type=int, default=C.MIN_DISK_GB)
    s.add_argument("--data", default="auto", help="auto (from the train config) | none | DATASET:N[,DATASET:N]")
    s.add_argument("--linger-min", type=float, default=5.0, help="minutes to stay up after HNET_DONE")
    s.add_argument("--no-shutdown", action="store_true", help="do not shut down after the command")
    s.add_argument("--metrics-every", type=int, default=1, help="mirror every K-th metrics line")
    s.add_argument("--watch-dir", action="append", help="extra dir (repo-relative) to mirror *.jsonl from")
    s.add_argument("--overlay", action="append",
                   help="repo-relative file/dir shipped with the job and unpacked over the clone (<=128 KB gz)")
    s.add_argument("--bucket", default=C.BUCKET, help="GCS bucket for run mirrors + data cache ('none' = off)")
    s.add_argument("--sa", default=None, help="VM service account (default the project SA when a bucket is set; "
                                              "'none' = no credentials on the VM)")
    s.add_argument("--sync-every-min", type=float, default=5.0, help="bucket sync period for run dirs")
    s.add_argument("--restore-from", default=None,
                   help="VM name whose bucket checkpoints seed this run when the disk has none (default: same name)")
    s.add_argument("--env", action="append")
    s.add_argument("--allow-compile", action="store_true", help="compile causal_conv1d if no wheel (<=10 min)")
    s.add_argument("--no-kernels", action="store_true")
    s.add_argument("--price-hr", type=float, default=None)
    s.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_TOTAL)
    s.add_argument("--launch-cap", type=float, default=C.DEFAULT_LAUNCH_CAP)
    s.add_argument("--allow-over", action="store_true", help="lift the per-launch cap (not the total)")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_launch)

    s = sp.add_parser("status")
    s.add_argument("name")
    s.set_defaults(fn=cmd_status)

    s = sp.add_parser("tail")
    s.add_argument("name")
    s.add_argument("-n", type=int, default=60)
    s.add_argument("--all", action="store_true", help="from the local copy's beginning, not just new output")
    s.add_argument("--hnet-only", action="store_true")
    s.set_defaults(fn=cmd_tail)

    s = sp.add_parser("sync-metrics")
    s.add_argument("name")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=cmd_sync)

    s = sp.add_parser("pull")
    s.add_argument("name")
    s.add_argument("run", help="run dir as on the VM (runs/tiny/x or tiny/x)")
    s.add_argument("--out", default=None, help="default runs/gcp/<vm>/<run>")
    s.add_argument("--ckpt", choices=["latest", "all", "none"], default="latest")
    s.add_argument("--bucket", default=None)
    s.set_defaults(fn=cmd_pull)

    s = sp.add_parser("watch")
    s.add_argument("name")
    s.add_argument("--out", default=None)
    s.add_argument("--interval", type=float, default=60)
    s.add_argument("--timeout-min", type=float, default=0)
    s.add_argument("--teardown", action="store_true", help="delete the VM once it finished")
    s.set_defaults(fn=cmd_watch)

    s = sp.add_parser("resume")
    s.add_argument("name")
    s.add_argument("--max-hours", type=float, default=None)
    s.add_argument("--retries", type=int, default=10, help="retries on zone stockout")
    s.add_argument("--retry-wait", type=float, default=60)
    s.add_argument("--rerun", action="store_true", help="run the command again even if it finished")
    s.add_argument("--cmd", default=None, help="replace the command (implies --rerun)")
    s.add_argument("--commit", default=None, help="check out another commit first (implies --rerun)")
    s.add_argument("--overlay", action="append", help="ship repo-relative files with the new job (implies --rerun)")
    s.add_argument("--bucket", default=None, help="add bucket sync to a VM launched without it (with --rerun)")
    s.add_argument("--allow-unpushed", action="store_true")
    s.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_TOTAL)
    s.add_argument("--launch-cap", type=float, default=C.DEFAULT_LAUNCH_CAP)
    s.add_argument("--allow-over", action="store_true")
    s.set_defaults(fn=cmd_resume)

    s = sp.add_parser("stop")
    s.add_argument("name")
    s.set_defaults(fn=cmd_stop)

    s = sp.add_parser("teardown")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("name", nargs="?")
    g.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_teardown)

    s = sp.add_parser("ls")
    s.set_defaults(fn=cmd_ls)

    s = sp.add_parser("cost")
    s.add_argument("--budget", type=float, default=C.DEFAULT_BUDGET_TOTAL)
    s.set_defaults(fn=cmd_cost)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return a.fn(a)
    except KeyError as e:
        _p(f"error: {e}")
        return 2
    except gce.GcloudError as e:
        _p(f"gcloud error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
