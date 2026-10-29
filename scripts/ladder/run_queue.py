"""Run a queue of training runs across all GPUs of one machine (one run per GPU, or several small runs per GPU).

    python scripts/ladder/run_queue.py --spec configs/ladder/SPEC.yaml [--param k=v] [--only a,b] [--shard i/n]
    python scripts/ladder/run_queue.py --queue QUEUE.yaml
        [--gpus 0,1,2,3] [--per-gpu 2] [--max-attempts 2] [--order list|longest] [--list] [--dry-run]
    python scripts/ladder/run_queue.py export runs/<sweep> [--dest runs/_export/<sweep>]   # small files only

Inputs. ``--spec`` takes a sweep spec (``scripts/search/spec.py``: runs x grid x seeds, out_root); ``--queue`` a
YAML (or JSON) file::

    sweep: ladder_1e17                 # run dirs: <out_root or runs/<sweep>>/<name>
    out_root: runs/ladder/ladder_1e17  # optional
    defaults: {gpu_frac: 1.0, set: [train.compile=blocks], env: {}}
    runs:
      - {name: bpe_L08, config: configs/ladder/bpe_L08.yaml, set: [train.train_tokens=2.1e8], est_hours: 0.2}
      - {name: hnet_L04, config: ..., gpu_frac: 0.5}          # two of these share one GPU
      - {name: big, config: ..., gpus: 2}                     # torchrun (DDP) on 2 whole GPUs
      - {name: custom, cmd: "python my_script.py --out {out_dir}"}

Every job trains with ``python -m hnet.train.train ... --set train.out_dir=<run dir>`` (spec jobs: exactly the
sweep's trainer arguments) and logs to ``<run dir>/stdout.log``. Scheduling: jobs start in order (``--order
longest`` sorts by ``est_hours``, longest first) on the first GPU(s) with enough free capacity; a GPU holds runs
whose ``gpu_frac`` sum to <= 1 (``--per-gpu N`` sets the default frac to 1/N). ``CUDA_VISIBLE_DEVICES`` and
``OMP_NUM_THREADS`` (CPUs / concurrent runs) are set per job.

Resumable: a finished run has ``done.json`` (same marker as scripts/search/sweep.py; ``diverged`` when the trainer
stopped on a non-finite loss) and is skipped; an unfinished one is started again and resumes from its newest
checkpoint (trainer ``resume: auto``). A failed run is retried up to ``--max-attempts`` times, then recorded as
failed while the others go on. SIGTERM/SIGINT are forwarded to the running jobs (the trainer checkpoints and exits),
so re-launching the same command continues the queue. Status: ``<out_root>/queue_status.json``; one runner per
out_root (lock file).

On a Prime machine: ``python -m infra.prime run NAME -- python scripts/ladder/run_queue.py --spec ...``, then
``python -m infra.prime exec NAME -- python scripts/ladder/run_queue.py export runs/<sweep>`` and
``python -m infra.prime pull NAME runs/_export/<sweep> runs/<sweep>`` (metrics, configs, logs; no checkpoints).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SEARCH_DIR = REPO_ROOT / "scripts" / "search"
EXPORT_FILES = ("config.yaml", "metrics.jsonl", "eval.jsonl", "stdout.log", "done.json", "diverged.json",
                "queue_job.json", "sweep_job.json", "diag.json", "profile_cuda.txt", "profile_cpu.txt")


@dataclass
class QJob:
    name: str
    out: str
    argv: List[str]  # full command (python ... or a shell command via bash -c)
    gpus: int = 1
    gpu_frac: float = 1.0
    env: Dict[str, str] = field(default_factory=dict)
    est_hours: Optional[float] = None
    meta: dict = field(default_factory=dict)
    # runtime
    attempts: int = 0
    status: str = "pending"
    assigned: List[int] = field(default_factory=list)
    proc: Optional[subprocess.Popen] = None
    t0: float = 0.0


# ----------------------------------------------------------------------------------------------- inputs
def trainer_argv(gpus: int, args: List[str]) -> List[str]:
    if gpus > 1:
        return [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={gpus}", *args]
    return [sys.executable, *args]


def jobs_from_spec(path: str, params=None, params_file=None, only=None, shard=None, extra_sets=()) -> tuple:
    if str(SEARCH_DIR) not in sys.path:
        sys.path.insert(0, str(SEARCH_DIR))
    from spec import Spec, parse_shard  # noqa: E402

    spec = Spec.load(path, params or [], params_file)
    out = []
    for j in spec.jobs(only=set(only.split(",")) if only else None, shard=parse_shard(shard)):
        args = j.trainer_args() + [a for s in extra_sets for a in ("--set", s)]
        out.append(QJob(name=j.name, out=j.out, argv=trainer_argv(1, args),
                        meta={"spec": spec.name, **j.to_json()}))
    return spec.out_root, out


def load_queue(path: str, extra_sets=()) -> tuple:
    import yaml

    with open(path) as f:
        q = yaml.safe_load(f)
    sweep = q.get("sweep") or Path(path).stem
    out_root = q.get("out_root") or f"runs/{sweep}"
    d = q.get("defaults") or {}
    jobs, seen = [], set()
    for r in q["runs"]:
        r = {**d, **r, "env": {**(d.get("env") or {}), **(r.get("env") or {})},
             "set": list(d.get("set") or []) + list(r.get("set") or []) + list(extra_sets)}
        name = r["name"]
        assert name not in seen, f"duplicate run name {name!r}"
        seen.add(name)
        out = r.get("out") or f"{out_root}/{name}"
        gpus = int(r.get("gpus", 1))
        if r.get("cmd"):
            cmd = r["cmd"].format(out_dir=out, name=name)
            argv = ["bash", "-c", cmd]
        else:
            args = ["-m", "hnet.train.train", "--config", r["config"]]
            if r.get("model"):
                args += ["--model", r["model"]]
            for s in r["set"]:
                args += ["--set", s]
            args += ["--set", f"train.out_dir={out}"]
            argv = trainer_argv(gpus, args)
        jobs.append(QJob(name=name, out=out, argv=argv, gpus=gpus,
                         gpu_frac=float(r["gpu_frac"]) if r.get("gpu_frac") is not None else -1.0,
                         env={k: str(v) for k, v in r["env"].items()}, est_hours=r.get("est_hours"),
                         meta={"queue": str(path), **{k: v for k, v in r.items() if k != "env"}}))
    return out_root, jobs


def visible_gpus() -> List[int]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None and env.strip():
        return [int(x) for x in env.split(",") if x.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        n = sum(1 for ln in out.splitlines() if ln.startswith("GPU "))
    except (OSError, subprocess.SubprocessError):
        n = 0
    return list(range(n))


# ----------------------------------------------------------------------------------------------- scheduler
class Runner:
    def __init__(self, out_root: str, jobs: List[QJob], gpus: List[int], max_attempts: int = 2, poll: float = 5.0,
                 cpus: Optional[int] = None, log=print):
        self.root = Path(out_root)
        self.jobs, self.gpus, self.max_attempts, self.poll = jobs, gpus, max_attempts, poll
        self.free = {g: 1.0 for g in gpus}
        self.cpus = cpus or os.cpu_count() or 1
        self.log = log
        self.stopping = False
        self.status_path = self.root / "queue_status.json"

    # ---- bookkeeping
    def save_status(self):
        rows = {j.name: {"status": j.status, "attempts": j.attempts, "gpus": j.assigned, "out": j.out}
                for j in self.jobs}
        tmp = self.status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "gpus": self.gpus, "runs": rows}, indent=1))
        tmp.replace(self.status_path)

    def _fits(self, j: QJob) -> Optional[List[int]]:
        eps = 1e-6
        if j.gpus > 1:
            whole = [g for g in self.gpus if self.free[g] >= 1.0 - eps]
            return whole[: j.gpus] if len(whole) >= j.gpus else None
        best = None  # fullest GPU that still fits (packs small runs, leaves whole GPUs for big ones)
        for g in self.gpus:
            if self.free[g] + eps >= j.gpu_frac and (best is None or self.free[g] < self.free[best]):
                best = g
        return None if best is None else [best]

    def _start(self, j: QJob, gpus: List[int], n_running: int):
        Path(j.out).mkdir(parents=True, exist_ok=True)
        j.attempts += 1
        j.assigned = gpus
        env = dict(os.environ)
        env.update(j.env)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env.setdefault("OMP_NUM_THREADS", str(max(1, self.cpus // max(1, n_running + 1))))
        env["PYTHONUNBUFFERED"] = "1"
        (Path(j.out) / "queue_job.json").write_text(json.dumps(
            {"name": j.name, "argv": j.argv, "gpus": gpus, "gpu_frac": j.gpu_frac, "attempt": j.attempts,
             "meta": j.meta}, indent=1, default=str))
        log = open(Path(j.out) / "stdout.log", "a")
        log.write(f"\n[run_queue] attempt {j.attempts} on GPU {gpus} at "
                  f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}: {shlex.join(j.argv)}\n")
        log.flush()
        j.proc = subprocess.Popen(j.argv, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)
        log.close()
        j.t0 = time.time()
        j.status = "running"
        for g in gpus:
            self.free[g] -= 1.0 if j.gpus > 1 else j.gpu_frac
        self.log(f"[run_queue] start {j.name} on GPU {gpus} (attempt {j.attempts}, pid {j.proc.pid})")

    def _finish(self, j: QJob, rc: int):
        for g in j.assigned:
            self.free[g] += 1.0 if j.gpus > 1 else j.gpu_frac
        secs = round(time.time() - j.t0, 1)
        j.proc = None
        out = Path(j.out)
        if rc == 0:
            diverged = (out / "diverged.json").exists()
            (out / "done.json").write_text(json.dumps({"rc": 0, "seconds": secs, "gpus": j.assigned,
                                                        "attempts": j.attempts, **({"diverged": True} if diverged else {})}))
            j.status = "diverged" if diverged else "done"
            self.log(f"[run_queue] end {j.name}: {j.status} in {secs:.0f}s")
        elif self.stopping:
            j.status = "interrupted"
            self.log(f"[run_queue] {j.name} stopped (rc={rc}) after {secs:.0f}s; it resumes on the next launch")
        elif j.attempts < self.max_attempts:
            j.status = "pending"
            self.log(f"[run_queue] {j.name} failed rc={rc} after {secs:.0f}s; retrying (resumes from checkpoint)")
        else:
            j.status = f"failed rc={rc}"
            self.log(f"[run_queue] FAILED {j.name} rc={rc}; see {out / 'stdout.log'}:\n{_tail(out / 'stdout.log')}")

    def _signal(self, signum, frame):
        if not self.stopping:
            self.log(f"[run_queue] signal {signum}: forwarding SIGTERM to running jobs")
        self.stopping = True
        for j in self.jobs:
            if j.proc is not None and j.proc.poll() is None:
                try:
                    os.killpg(j.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    # ---- main loop
    def run(self) -> int:
        for j in self.jobs:
            d = Path(j.out) / "done.json"
            if d.exists():
                j.status = "diverged" if json.loads(d.read_text() or "{}").get("diverged") else "done"
        pending = [j for j in self.jobs if j.status == "pending"]
        self.log(f"[run_queue] {len(self.jobs)} jobs ({len(self.jobs) - len(pending)} already done), "
                 f"GPUs {self.gpus}, out_root {self.root}")
        for j in pending:
            if j.gpus > len(self.gpus) or (j.gpus == 1 and j.gpu_frac > 1.0 + 1e-6):
                raise SystemExit(f"job {j.name} needs {j.gpus} GPUs (frac {j.gpu_frac}) but only {self.gpus} exist")
        old = {s: signal.signal(s, self._signal) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            self.save_status()
            while True:
                running = [j for j in self.jobs if j.proc is not None]
                for j in running:
                    rc = j.proc.poll()
                    if rc is not None:
                        self._finish(j, rc)
                        self.save_status()
                running = [j for j in self.jobs if j.proc is not None]
                if not self.stopping:
                    for j in [j for j in self.jobs if j.status == "pending"]:
                        g = self._fits(j)
                        if g is not None:
                            self._start(j, g, len(running))
                            running.append(j)
                            self.save_status()
                if not running and (self.stopping or not any(j.status == "pending" for j in self.jobs)):
                    break
                time.sleep(self.poll)
        finally:
            for s, h in old.items():
                signal.signal(s, h)
            self.save_status()
        counts: Dict[str, int] = {}
        for j in self.jobs:
            counts[j.status.split()[0]] = counts.get(j.status.split()[0], 0) + 1
        self.log(f"[run_queue] finished: {counts}")
        if self.stopping:
            return 143
        return 1 if any(j.status.startswith("failed") for j in self.jobs) else 0


def _tail(path: Path, n: int = 30) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def export(sweep_dir: str, dest: Optional[str] = None) -> Path:
    """Copy the small result files of every run under ``sweep_dir`` (no checkpoints) to ``dest``."""
    src = Path(sweep_dir)
    if dest:
        dst = Path(dest)
    elif src.parts and src.parts[0] == "runs":
        dst = Path("runs", "_export", *src.parts[1:])
    else:
        dst = Path(f"{sweep_dir}_export")
    n = 0
    for f in src.rglob("*"):
        rel = f.relative_to(src)
        if any(p.startswith(("ckpt", "step_")) for p in rel.parts[:-1]) or not f.is_file():
            continue
        if f.name in EXPORT_FILES or f.parent == src:
            t = dst / f.relative_to(src)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
            n += 1
    print(f"[run_queue] exported {n} files from {src} to {dst}")
    return dst


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--spec", help="sweep spec (scripts/search/spec.py format)")
    src.add_argument("--queue", help="queue file (YAML/JSON, see module docstring)")
    ap.add_argument("--param", action="append", default=[], help="spec parameter name=value")
    ap.add_argument("--params-file")
    ap.add_argument("--only", help="spec: comma-separated architectures to keep")
    ap.add_argument("--shard", help="spec: i/n")
    ap.add_argument("--set", action="append", default=[], help="extra --set for every training job")
    ap.add_argument("--gpus", help="comma-separated GPU ids (default: all)")
    ap.add_argument("--per-gpu", type=int, default=1, help="default runs per GPU (gpu_frac = 1/N)")
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--order", choices=["list", "longest"], default="list")
    ap.add_argument("--poll", type=float, default=5.0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "export":
        ep = argparse.ArgumentParser(prog="run_queue.py export")
        ep.add_argument("sweep_dir")
        ep.add_argument("--dest")
        e = ep.parse_args(argv[1:])
        export(e.sweep_dir, e.dest)
        return 0
    a = build_parser().parse_args(argv)
    if a.spec:
        out_root, jobs = jobs_from_spec(a.spec, a.param, a.params_file, a.only, a.shard, a.set)
    elif a.queue:
        out_root, jobs = load_queue(a.queue, a.set)
    else:
        build_parser().error("one of --spec / --queue is required")
    for j in jobs:
        if j.gpu_frac < 0:
            j.gpu_frac = 1.0 / max(1, a.per_gpu)
    if a.order == "longest":
        jobs.sort(key=lambda j: -(j.est_hours or 0.0))
    if a.list or a.dry_run:
        for j in jobs:
            done = (Path(j.out) / "done.json").exists()
            print(f"{j.name:40s} gpus={j.gpus} frac={j.gpu_frac:.2f} {'done' if done else 'todo'}"
                  + (f"  {shlex.join(j.argv)}" if a.dry_run else ""))
        return 0
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else visible_gpus()
    if not gpus:
        print("[run_queue] no GPUs found (set --gpus or CUDA_VISIBLE_DEVICES)", flush=True)
        return 2
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = open(root / ".run_queue.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[run_queue] another runner holds {root / '.run_queue.lock'}", flush=True)
        return 3
    return Runner(out_root, jobs, gpus, a.max_attempts, a.poll, log=lambda s: print(s, flush=True)).run()


if __name__ == "__main__":
    sys.exit(main())
