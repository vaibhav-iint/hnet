"""Run an experiment sweep (many short training runs) on one machine, then chunking diagnostics per run.

    python scripts/search/sweep.py configs/search/exp002_tiny2m.yaml [--parallel 3] [--only name1,name2]
        [--gcs gs://bucket/runs/exp002] [--dry-run]

Spec file::

    name: exp002_tiny2m
    out_root: runs/search/exp002_tiny2m
    seeds: [0, 1]
    set: [train.train_tokens=5e7]            # applied to every run
    runs:
      - {name: bpe, config: configs/runs/tiny_bpe_2m.yaml}
      - {name: end_mlp, config: configs/runs/tiny_hnet2_2m.yaml, set: [model.dc.router=end_mlp]}

Each run trains with ``python -m hnet.train.train`` into ``<out_root>/<name>_s<seed>``; finished runs (a
``done.json`` marker) are skipped, so the sweep can be re-launched after a preemption. After training, H-Net runs get
``diag.json`` (hnet.analysis.chunks on FineWeb-Edu val documents) and every run dir is mirrored to ``--gcs`` if given.
``<out_root>/sweep_status.json`` records the state of every run. The sweep first runs ``fairness.py`` (same data,
bytes, FLOPs/byte, optimizer, schedule, eval for every run) and refuses to start if it fails.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _sys.path:  # allow `python scripts/...` from anywhere without PYTHONPATH
    _sys.path.insert(0, str(REPO_ROOT))


import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def expand(spec: dict, only=None):
    jobs = []
    for r in spec["runs"]:
        if only and r["name"] not in only:
            continue
        for seed in r.get("seeds", spec.get("seeds", [0])):
            out = Path(spec["out_root"]) / f"{r['name']}_s{seed}"
            sets = list(spec.get("set", [])) + list(r.get("set", [])) + [f"train.seed={seed}", f"train.out_dir={out}"]
            jobs.append(dict(name=f"{r['name']}_s{seed}", config=r["config"], sets=sets, out=str(out)))
    return jobs


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


def gcs_sync(local: str, remote: str):
    """Mirror a directory to GCS; a failed upload is an error (results must not silently stay local)."""
    r = subprocess.run(["gcloud", "storage", "rsync", "-r", local, remote], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage rsync {local} -> {remote} failed:\n{r.stderr[-2000:]}")


def log_tail(path: Path, n: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:]) if path.exists() else "(no log)"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--only", default=None, help="comma-separated run names (without seed suffix)")
    ap.add_argument("--gcs", default=None, help="gs:// prefix to mirror run dirs to")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-unfair", action="store_true", help="run even if scripts/search/fairness.py fails")
    a = ap.parse_args(argv)
    spec = yaml.safe_load(open(a.spec))
    from fairness import main as fairness_main  # apples-to-apples gate (writes <out_root>/fairness.md)

    if fairness_main([a.spec]) != 0 and not a.allow_unfair:
        print("[sweep] fairness check failed; refusing to run (see fairness.md, or pass --allow-unfair)", flush=True)
        return 2
    jobs = expand(spec, set(a.only.split(",")) if a.only else None)
    Path(spec["out_root"]).mkdir(parents=True, exist_ok=True)
    status_path = Path(spec["out_root"]) / "sweep_status.json"
    status = {j["name"]: "pending" for j in jobs}
    texts = None

    def save_status():
        status_path.write_text(json.dumps(status, indent=1))

    failed = []  # first failure stops new runs from starting; running ones finish

    def run(job):
        nonlocal texts
        done = Path(job["out"]) / "done.json"
        if done.exists():
            status[job["name"]] = "done"
            return
        if failed:
            status[job["name"]] = "skipped (earlier failure)"
            return
        cmd = [sys.executable, "-m", "hnet.train.train", "--config", job["config"]]
        for s in job["sets"]:
            cmd += ["--set", s]
        print("[sweep] start", job["name"], " ".join(cmd[3:]), flush=True)
        if a.dry_run:
            return
        status[job["name"]] = "running"
        save_status()
        t0 = time.time()
        Path(job["out"]).mkdir(parents=True, exist_ok=True)
        log_path = Path(job["out"]) / "stdout.log"
        with open(log_path, "a") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if rc != 0:
            status[job["name"]] = f"failed rc={rc}"
            failed.append(job["name"])
            save_status()
            print(f"[sweep] FAILED {job['name']} rc={rc}; last lines of {log_path}:\n{log_tail(log_path)}", flush=True)
            if a.gcs:
                gcs_sync(job["out"], f"{a.gcs.rstrip('/')}/{Path(job['out']).name}")
            return
        if texts is None:
            texts = diag_texts()
        diag = run_diag(job["out"], texts)
        if diag:
            (Path(job["out"]) / "diag.json").write_text(json.dumps(diag, indent=1))
        result = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        done.write_text(json.dumps(result))
        status[job["name"]] = "done"
        save_status()
        print("[sweep] end", job["name"], result, flush=True)
        if a.gcs:
            gcs_sync(job["out"], f"{a.gcs.rstrip('/')}/{Path(job['out']).name}")

    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        list(ex.map(run, jobs))  # re-raises the first exception from any worker
    save_status()
    if a.gcs:
        gcs_sync(str(Path(spec["out_root"])), a.gcs.rstrip("/"))
    print("[sweep] finished:", json.dumps(status), flush=True)
    if failed:
        print(f"[sweep] FAILED runs: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
