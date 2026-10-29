"""Run an experiment sweep (many short training runs) on one machine, then chunking diagnostics per run.

    python scripts/search/sweep.py SPEC [--shard i/n] [--parallel 3] [--only run1,run2] [--gcs gs://.../NAME]
        [--restore] [--mirror-every 120] [--param name=value ...] [--params-file f.yaml] [--allow-missing]
        [--list] [--dry-run]

Spec format, job names, ordering and sharding: ``scripts/search/spec.py``; walkthrough: docs/EXPERIMENTS_HOWTO.md.

Each job trains with ``python -m hnet.train.train`` into ``<out_root>/<job name>`` and writes ``sweep_job.json``
(architecture, grid point, seed) there. Finished jobs (a ``done.json`` marker) are skipped and unfinished ones
resume from their newest checkpoint (trainer ``resume: auto``), so the sweep can be re-launched after a
preemption. A run whose loss goes non-finite stops itself (trainer ``stop_on_nonfinite``) and is recorded as
``diverged`` (a result, e.g. too high an LR; not a failure). After training, H-Net runs get ``diag.json``
(hnet.analysis.chunks on the blog's English texts and FineWeb-Edu val documents).

Before starting, the fairness gate (``fairness.py``, strict or tuned per the spec) must pass. The first failed
job stops new jobs from starting (running ones finish); the exit code is then 1.

GCS (``--gcs``): ``--restore`` first downloads this selection's run dirs (finished runs, checkpoints) from the
prefix; ``--mirror-every S`` uploads them every S seconds while training (a failed periodic upload is reported
on stdout and retried next period); after each job and at the end, uploads are mandatory (a failure raises).
Status: ``<out_root>/sweep_status[.shard-i-of-n].json`` per selection, so several VMs can share one prefix.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _sys.path:  # allow `python scripts/...` from anywhere without PYTHONPATH
    _sys.path.insert(0, str(REPO_ROOT))
_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

from spec import Spec, SpecError, parse_shard, status_filename  # noqa: E402

ROOT = REPO_ROOT


def expand(spec: dict, only=None):
    """v1 API (dict spec -> list of job dicts with name/config/sets/out); kept for old callers."""
    import tempfile

    import yaml

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(spec, f)
    jobs = Spec.load(f.name).jobs(only=set(only) if only else None)
    Path(f.name).unlink()
    return [dict(name=j.name, config=j.config, sets=j.sets, out=j.out, model=j.model) for j in jobs]


def diag_texts(n_docs: int = 40, max_bytes: int = 3000):
    """English examples from the blog's chunk viewer + the first FineWeb-Edu val documents."""
    texts = []
    ref = ROOT / "configs" / "reference" / "chunk_viewer.json"
    if ref.exists():
        texts += [e["text"][:max_bytes] for e in json.load(open(ref)) if e.get("src") == "English"]
    from hnet.data.download import data_root
    from hnet.data.shards import byte_paths, doc_text_bytes, read_byte_shard, read_doc_index

    bin_path, idx_path = byte_paths(data_root() / "finewebedu10B" / "bytes", "finewebedu_val_000000")
    if bin_path.exists():
        data, idx = read_byte_shard(bin_path), read_doc_index(idx_path)
        for row in idx[:n_docs]:
            b = bytes(doc_text_bytes(data, row)[:max_bytes])
            texts.append(b.decode("utf-8", "ignore"))
    return texts


def run_diag(out_dir: str, texts) -> dict:
    import torch

    from hnet.analysis.chunks import chunk_cuts, chunk_report, render
    from hnet.eval.run import load_run_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mcfg, meta = load_run_checkpoint(out_dir, device=device)
    if not hasattr(mcfg, "n_stages"):
        return {}
    rep = chunk_report(model, texts, device=device)
    sample = texts[0]
    rep["examples"] = {f"s{s}": render(sample, c) for s, c in enumerate(chunk_cuts(model, sample, device))}
    rep["checkpoint"] = meta.get("checkpoint")
    return rep


# ---------------------------------------------------------------------------------------------------- GCS
def gcloud(args):
    """``gcloud <args>`` (on the VM: its service account); callers check the return code."""
    return subprocess.run(["gcloud", *args], capture_output=True, text=True, check=False)


def gcs_sync(local: str, remote: str):
    """Mirror a directory to GCS; a failed upload is an error (results must not silently stay local)."""
    r = gcloud(["storage", "rsync", "-r", local, remote])
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage rsync {local} -> {remote} failed:\n{r.stderr[-2000:]}")


def gcs_list_dirs(prefix: str) -> set:
    """Names of the immediate sub-directories of a gs:// prefix (empty if the prefix does not exist yet)."""
    r = gcloud(["storage", "ls", prefix.rstrip("/") + "/"])
    if r.returncode != 0:
        if "matched no objects" in r.stderr or "One or more URLs matched no" in r.stderr:
            return set()
        raise RuntimeError(f"gcloud storage ls {prefix} failed:\n{r.stderr[-2000:]}")
    return {line.rstrip("/").rsplit("/", 1)[-1] for line in r.stdout.splitlines() if line.endswith("/")}


def restore(jobs, gcs: str):
    """Download this selection's run dirs (incl. checkpoints and done markers) from GCS before starting. Run dirs
    already on local disk are kept as they are (they are at least as new as the mirror: a restarted VM)."""
    remote = gcs_list_dirs(gcs)
    for j in jobs:
        if j.name in remote and not Path(j.out).exists():
            Path(j.out).mkdir(parents=True, exist_ok=True)
            r = gcloud(["storage", "rsync", "-r", f"{gcs.rstrip('/')}/{j.name}", j.out])
            if r.returncode != 0:
                raise RuntimeError(f"restore of {j.name} from {gcs} failed:\n{r.stderr[-2000:]}")
            print(f"[sweep] restored {j.name} from GCS", flush=True)


class Mirror(threading.Thread):
    """Uploads the selection's existing run dirs every ``every`` seconds. Failures are printed (with gcloud's
    stderr) and retried next period; the mandatory uploads after each job / at the end raise instead."""

    def __init__(self, jobs, gcs: str, every: float):
        super().__init__(daemon=True)
        self.jobs, self.gcs, self.every = jobs, gcs.rstrip("/"), every
        self.stop_event = threading.Event()

    def run(self):
        while not self.stop_event.wait(self.every):
            self.sync_once()

    def sync_once(self):
        for j in self.jobs:
            if Path(j.out).exists():
                r = gcloud(["storage", "rsync", "-r", j.out, f"{self.gcs}/{j.name}"])
                if r.returncode != 0:
                    print(f"[sweep] WARN live mirror of {j.name} failed (retrying in {self.every:.0f}s):\n"
                          f"{r.stderr[-1000:]}", flush=True)


def log_tail(path: Path, n: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


# ---------------------------------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--parallel", type=int, default=1, help="jobs trained at once on this machine")
    ap.add_argument("--only", default=None, help="comma-separated run names (architectures) to keep")
    ap.add_argument("--shard", default=None, help="i/n: run only jobs i, i+n, ... of the spec's job order")
    ap.add_argument("--gcs", default=None, help="gs:// prefix to mirror run dirs to (default: spec `gcs`)")
    ap.add_argument("--restore", action="store_true", help="download this selection's run dirs from --gcs first")
    ap.add_argument("--mirror-every", type=float, default=0, help="upload run dirs every N seconds (0 = off)")
    ap.add_argument("--param", action="append", default=[], help="spec parameter name=value (repeatable)")
    ap.add_argument("--params-file", default=None, help="YAML of spec parameters (e.g. a report's best_params)")
    ap.add_argument("--allow-missing", action="store_true", help="skip runs whose config files do not exist")
    ap.add_argument("--allow-unfair", action="store_true", help="run even if scripts/search/fairness.py fails")
    ap.add_argument("--list", action="store_true", help="print the selected jobs and exit")
    ap.add_argument("--dry-run", action="store_true", help="print the trainer commands and exit")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    try:
        spec = Spec.load(a.spec, a.param, a.params_file)
        shard = parse_shard(a.shard)
        only = set(a.only.split(",")) if a.only else None
        jobs = spec.jobs(only=only, shard=shard, allow_missing=a.allow_missing)
    except SpecError as e:
        print(f"[sweep] spec error: {e}", flush=True)
        return 2
    if spec.missing:
        print(f"[sweep] WARNING: skipping runs with missing configs: {spec.missing}", flush=True)
    if a.list or a.dry_run:
        for j in jobs:
            print(f"{j.index:4d} {j.name:48s} {j.setting}" if a.list else
                  f"{sys.executable} " + " ".join(j.trainer_args()), flush=True)
        return 0
    gcs = a.gcs or spec.raw.get("gcs")

    from fairness import main as fairness_main  # apples-to-apples gate (writes <out_root>/fairness.md)

    fargs = [a.spec] + [f"--param={p}" for p in a.param] + (["--params-file", a.params_file] if a.params_file else [])
    if fairness_main(fargs + (["--allow-missing"] if a.allow_missing else [])) != 0 and not a.allow_unfair:
        print("[sweep] fairness check failed; refusing to run (see fairness.md, or pass --allow-unfair)", flush=True)
        return 2
    out_root = Path(spec.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_manifest.json").write_text(json.dumps(spec.manifest(a.allow_missing), indent=1))
    if gcs and a.restore:
        restore(jobs, gcs)
    status_path = out_root / status_filename(shard, only)
    status = {j.name: "pending" for j in jobs}
    lock = threading.Lock()
    # Post-run diagnostics run in this process; Triton's autotuner (fused Mamba kernels) is not thread-safe, so two
    # runs finishing together must not run their diagnostics concurrently ('NoneType' object is not a mapping).
    diag_lock = threading.Lock()
    texts = None

    def save_status():
        with lock:
            tmp = status_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"spec": spec.name, "shard": a.shard, "only": a.only,
                                       "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                       "runs": status}, indent=1))
            tmp.replace(status_path)

    failed = []  # first failure stops new runs from starting; running ones finish
    mirror = None
    if gcs and a.mirror_every > 0:
        mirror = Mirror(jobs, gcs, a.mirror_every)
        mirror.start()

    def run(job):
        nonlocal texts
        out = Path(job.out)
        done = out / "done.json"
        if done.exists():
            status[job.name] = "diverged" if json.loads(done.read_text()).get("diverged") else "done"
            return
        if failed:
            status[job.name] = "skipped (earlier failure)"
            return
        cmd = [sys.executable, *job.trainer_args()]
        print("[sweep] start", job.name, " ".join(cmd[3:]), flush=True)
        status[job.name] = "running"
        save_status()
        t0 = time.time()
        out.mkdir(parents=True, exist_ok=True)
        (out / "sweep_job.json").write_text(json.dumps({"spec": spec.name, **job.to_json()}, indent=1))
        log_path = out / "stdout.log"
        with open(log_path, "a") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if rc != 0:
            status[job.name] = f"failed rc={rc}"
            failed.append(job.name)
            save_status()
            print(f"[sweep] FAILED {job.name} rc={rc}; last lines of {log_path}:\n{log_tail(log_path)}", flush=True)
            if gcs:
                gcs_sync(job.out, f"{gcs.rstrip('/')}/{job.name}")
            return
        result = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        if (out / "diverged.json").exists():
            result["diverged"] = True  # a result (e.g. LR too high), no diagnostics on a NaN model
        else:
            with diag_lock:
                if texts is None:
                    texts = diag_texts()
                diag = run_diag(job.out, texts)
            if diag:
                (out / "diag.json").write_text(json.dumps(diag, indent=1))
        done.write_text(json.dumps(result))
        status[job.name] = "diverged" if result.get("diverged") else "done"
        save_status()
        print("[sweep] end", job.name, result, flush=True)
        if gcs:
            gcs_sync(job.out, f"{gcs.rstrip('/')}/{job.name}")

    try:
        with ThreadPoolExecutor(max_workers=a.parallel) as ex:
            list(ex.map(run, jobs))  # re-raises the first exception from any worker
    finally:
        if mirror is not None:
            mirror.stop_event.set()
        save_status()
    if gcs:
        for name in ("fairness.md", "fairness.json", "sweep_manifest.json", status_path.name):
            r = gcloud(["storage", "cp", str(out_root / name), f"{gcs.rstrip('/')}/{name}"])
            if r.returncode != 0:
                raise RuntimeError(f"upload of {name} to {gcs} failed:\n{r.stderr[-2000:]}")
    print("[sweep] finished:", json.dumps(status), flush=True)
    if failed:
        print(f"[sweep] FAILED runs: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
