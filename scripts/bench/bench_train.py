"""Training-loop throughput benchmark: run a queue of short training runs, then summarize tokens/s, bytes/s, MFU.

    python scripts/bench/bench_train.py run configs/bench/plan_h100.yaml [--gpus 0] [--only a,b]
    python scripts/bench/bench_train.py summarize runs/bench/h100 [--skip 3] [--md out.md]

``run`` executes the plan (a ``scripts/ladder/run_queue.py`` queue file: runs share GPUs through ``gpu_frac``) and
then summarizes it. Every number comes from the trainer's own ``metrics.jsonl`` (the real training loop: data
loader, bf16 autocast, compile, optimizer, logging): the median over the log records after the first ``--skip``
(compile / autotune warm-up). ``group`` (a plan run key) marks runs that share one GPU at the same time; the summary
adds a row with their summed throughput. ``warmup_s`` is the wall time from the trainer start to the first log
record (compile cost, paid once per run).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("tokens_per_sec", "bytes_per_sec", "mfu", "step_time", "mem_gb", "data_wait_frac", "flops_per_byte")


def _run_queue():
    spec = importlib.util.spec_from_file_location("run_queue", ROOT / "scripts" / "ladder" / "run_queue.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_queue"] = mod
    spec.loader.exec_module(mod)
    return mod


def summarize_run(d: Path, skip: int) -> dict:
    recs = [json.loads(ln) for ln in (d / "metrics.jsonl").read_text().splitlines() if ln.strip()] \
        if (d / "metrics.jsonl").exists() else []
    job = json.loads((d / "queue_job.json").read_text()) if (d / "queue_job.json").exists() else {}
    meta = job.get("meta") or {}
    out = {"name": d.name, "group": meta.get("group"), "n_records": len(recs), "gpus": job.get("gpus")}
    done = d / "done.json"
    out["status"] = "done" if done.exists() else "failed/incomplete"
    use = recs[skip:] if len(recs) > skip else recs[-1:]
    for k in KEYS:
        vals = [r[k] for r in use if isinstance(r.get(k), (int, float))]
        out[k] = statistics.median(vals) if vals else None
    comp = [r.get("compression/s0") for r in use if r.get("compression/s0")]
    out["compression"] = statistics.median(comp) if comp else None
    log = d / "stdout.log"
    if log.exists():
        txt = log.read_text(errors="replace")
        for ln in txt.splitlines():
            if ln.startswith("[train] {"):
                info = json.loads(ln[len("[train] "):])
                out.update(params=info.get("params"), micro_batch=info.get("micro_batch"),
                           grad_accum=info.get("grad_accum"), seq_len=info.get("seq_len"))
        if out["status"] != "done":
            out["error"] = "\n".join(txt.splitlines()[-8:])
    if recs and len(recs) > 1 and recs[0].get("step_time"):
        out["warmup_s"] = round(recs[0]["step_time"] * recs[0]["step"], 1)
    return out


def summarize(root: str, skip: int = 3, md: str | None = None) -> list:
    rows = []
    for d in sorted(Path(root).iterdir()):
        if d.is_dir() and ((d / "queue_job.json").exists() or (d / "metrics.jsonl").exists()):
            rows.append(summarize_run(d, skip))
    groups = {}
    for r in rows:
        if r.get("group"):
            groups.setdefault(r["group"], []).append(r)
    for g, rs in groups.items():
        agg = {"name": f"[{g}] x{len(rs)} sum", "group": g, "status": "group"}
        for k in ("tokens_per_sec", "bytes_per_sec", "mfu"):
            vals = [r[k] for r in rs if r.get(k) is not None]
            agg[k] = sum(vals) if vals else None
        agg["mem_gb"] = sum(r["mem_gb"] for r in rs if r.get("mem_gb"))
        rows.append(agg)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30).stdout.strip().splitlines()[0]
    except Exception:
        gpu = "?"
    try:
        import torch

        tv = torch.__version__
    except Exception:
        tv = "?"
    lines = [f"GPU: {gpu}; torch {tv}; medians over log records after the first {skip}", "",
             "| run | params M | mb x accum | tok/s | bytes/s | MFU % | step s | mem GB | data wait % | comp | "
             "warmup s | status |", "|" + "---|" * 12]

    def f(v, fmt):
        return "-" if v is None else format(v, fmt)

    for r in rows:
        lines.append(
            f"| {r['name']} | {f(r.get('params') and r['params'] / 1e6, '.1f')} | "
            f"{r.get('micro_batch', '-')} x {r.get('grad_accum', '-')} | {f(r.get('tokens_per_sec'), ',.0f')} | "
            f"{f(r.get('bytes_per_sec'), ',.0f')} | {f(r.get('mfu') and 100 * r['mfu'], '.1f')} | "
            f"{f(r.get('step_time'), '.3f')} | {f(r.get('mem_gb'), '.1f')} | "
            f"{f(r.get('data_wait_frac') is not None and 100 * r['data_wait_frac'], '.1f')} | "
            f"{f(r.get('compression'), '.2f')} | {r.get('warmup_s', '-')} | {r['status']} |")
    text = "\n".join(lines)
    print(text, flush=True)
    for r in rows:
        if r.get("error"):
            print(f"\n--- {r['name']} (last log lines)\n{r['error']}", flush=True)
    (Path(root) / "summary.json").write_text(json.dumps({"gpu": gpu, "torch": tv, "skip": skip, "rows": rows},
                                                        indent=1, default=str))
    (Path(md) if md else Path(root) / "summary.md").write_text(text + "\n")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("plan")
    r.add_argument("--gpus")
    r.add_argument("--only", help="comma-separated run names")
    r.add_argument("--skip", type=int, default=3)
    r.add_argument("--set", action="append", default=[])
    s = sp.add_parser("summarize")
    s.add_argument("root")
    s.add_argument("--skip", type=int, default=3)
    s.add_argument("--md")
    a = ap.parse_args(argv)
    if a.cmd == "summarize":
        summarize(a.root, a.skip, a.md)
        return 0
    rq = _run_queue()
    out_root, jobs = rq.load_queue(a.plan, a.set)
    if a.only:
        keep = set(a.only.split(","))
        jobs = [j for j in jobs if j.name in keep]
    for j in jobs:
        if j.gpu_frac < 0:
            j.gpu_frac = 1.0
    gpus = [int(g) for g in a.gpus.split(",")] if a.gpus else rq.visible_gpus()
    Path(out_root).mkdir(parents=True, exist_ok=True)
    rc = rq.Runner(out_root, jobs, gpus, max_attempts=1, poll=2.0, log=lambda m: print(m, flush=True)).run()
    summarize(out_root, a.skip)
    return rc


if __name__ == "__main__":
    sys.exit(main())
