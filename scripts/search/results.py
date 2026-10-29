"""Load a sweep's run directories (local or pulled from GCS) and compute the statistics report.py tabulates.

    from results import load_sweep, pull
    sweep = load_sweep(spec, runs_dir)           # every job of the spec, whether or not its dir exists yet
    sweep.final("val/bpb", arch, point)          # {seed: final value} over finished seeds

Per job directory (``<runs_dir>/<job name>``): ``metrics.jsonl`` (train records), ``eval.jsonl`` (val bpb, task
metrics), ``done.json`` (finished; ``diverged: true`` if the trainer stopped on a non-finite loss), ``diag.json``
(chunk statistics, H-Nets), ``config.yaml``.

Statistics (seeds are the replicates):

* :func:`mean_ci`: mean and two-sided 95% Student-t interval over seeds;
* :func:`paired_ci`: mean difference over seeds present in both groups (same seed = same data order and init
  stream), with its t interval: the right interval for "variant vs baseline" because seed noise is shared;
* :func:`stable_points` + :func:`multiplier`: the blog's data multiplier (hnet.eval.stats.data_multiplier):
  baseline tokens needed to reach the candidate's val bpb / candidate tokens, both read in the stable phase
  (``lr_factor == 1``) so the WSD cooldown does not distort the comparison.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}

# Blog's improved H-Net, stage-1 chunks on English text (EXP-001, configs/reference/chunk_viewer.json).
BLOG_TARGET_S1 = {"bytes_per_chunk": 7.0, "pct_cut_after_space": 91.9, "pct_word_internal": 0.0, "pct_mid_utf8": 0.0}


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def mean_ci(values) -> dict:
    v = np.asarray([x for x in values if x is not None], dtype=float)
    n = len(v)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    m = float(v.mean())
    if n == 1 or not np.all(np.isfinite(v)):
        return {"mean": m, "lo": float("nan"), "hi": float("nan"), "n": n}
    h = t975(n - 1) * float(v.std(ddof=1)) / math.sqrt(n)
    return {"mean": m, "lo": m - h, "hi": m + h, "n": n}


def paired_ci(a: Dict[int, float], b: Dict[int, float]) -> dict:
    """a - b over common seeds."""
    seeds = sorted(set(a) & set(b))
    return {**mean_ci([a[s] - b[s] for s in seeds]), "seeds": seeds}


def read_jsonl(p: Path) -> List[dict]:
    if not p.exists():
        return []
    out = []
    lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            if i == len(lines) - 1:  # a run being written / killed mid-line
                continue
            raise
    return out


@dataclass
class RunData:
    job: object  # spec.Job
    dir: Path
    metrics: List[dict] = field(default_factory=list)
    evals: List[dict] = field(default_factory=list)
    done: Optional[dict] = None
    diag: Optional[dict] = None

    @property
    def exists(self) -> bool:
        return self.dir.exists()

    @property
    def finished(self) -> bool:
        return self.done is not None

    @property
    def diverged(self) -> bool:
        return bool(self.done and self.done.get("diverged")) or (self.dir / "diverged.json").exists()

    @property
    def state(self) -> str:
        if self.diverged:
            return "diverged"
        if self.finished:
            return "done"
        return "running" if self.metrics else ("started" if self.exists else "absent")

    def last(self, key: str, source: str = "eval"):
        for r in reversed(self.evals if source == "eval" else self.metrics):
            if key in r and r[key] is not None:
                return r[key]
        return None

    def final_eval(self) -> dict:
        """The last eval record (at the end of training for finished runs)."""
        return self.evals[-1] if self.evals else {}

    def lr_factor_at(self) -> Dict[int, float]:
        return {int(r["step"]): float(r["lr_factor"]) for r in self.metrics if "lr_factor" in r}


def load_run(job, runs_dir: Path) -> RunData:
    d = Path(runs_dir) / job.name
    rd = RunData(job=job, dir=d)
    if d.exists():
        rd.metrics = read_jsonl(d / "metrics.jsonl")
        rd.evals = read_jsonl(d / "eval.jsonl")
        if (d / "done.json").exists():
            rd.done = json.loads((d / "done.json").read_text())
        if (d / "diag.json").exists():
            rd.diag = json.loads((d / "diag.json").read_text())
    return rd


@dataclass
class Sweep:
    spec: object
    runs: List[RunData]

    def archs(self) -> List[str]:
        return [a for a in self.spec.run_names() if any(r.job.arch == a for r in self.runs)]

    def points(self, arch: str) -> List[dict]:
        out = []
        for r in self.runs:
            if r.job.arch == arch and r.job.point not in out:
                out.append(r.job.point)
        return out

    def select(self, arch: str, point: Optional[dict] = None) -> List[RunData]:
        return [r for r in self.runs if r.job.arch == arch and (point is None or r.job.point == point)]

    def final(self, key: str, arch: str, point: dict) -> Dict[int, float]:
        """{seed: value of ``key`` in the final eval record} over finished runs; diverged runs give +inf for
        val/bpb (worse than anything), are skipped for other keys."""
        out = {}
        for r in self.select(arch, point):
            if not r.finished:
                continue
            if r.diverged:
                if key == "val/bpb":
                    out[r.job.seed] = float("inf")
                continue
            v = r.final_eval().get(key)
            if v is not None:
                out[r.job.seed] = float(v)
        return out

    def counts(self) -> Dict[str, int]:
        c: Dict[str, int] = {}
        for r in self.runs:
            c[r.state] = c.get(r.state, 0) + 1
        return c


def load_sweep(spec, runs_dir) -> Sweep:
    jobs = spec.jobs(allow_missing=True)
    return Sweep(spec=spec, runs=[load_run(j, Path(runs_dir)) for j in jobs])


def pull(gcs: str, local: Path):
    """Mirror a sweep's GCS prefix locally, without checkpoints (read-only on GCS)."""
    env = None
    try:  # the isolated gcloud config of infra/gcp, when this runs on the dev box
        from infra.gcp.gce import gcloud_env

        env = gcloud_env()
    except ImportError:
        pass
    local.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["gcloud", "storage", "rsync", "-r", "--exclude", r".*/ckpt/.*", gcs.rstrip("/"), str(local)],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"gcloud storage rsync {gcs} -> {local} failed:\n{r.stderr[-2000:]}")


# ------------------------------------------------------------------------------------------- derived values
def soft_bpb(rec: dict) -> Optional[float]:
    """Average soft bpb over the suite's tasks (the suite average if logged, else the mean of task values)."""
    for k in ("downstream/averages/average/soft_bpb/value",):
        if k in rec:
            return float(rec[k])
    vals = [v for k, v in rec.items() if re.fullmatch(r"downstream/tasks/[^/]+/metrics/soft_bpb/value", k)]
    return float(np.mean(vals)) if vals else None


def acc_norm(rec: dict) -> Optional[float]:
    v = rec.get("downstream/averages/average/acc_norm_char/value")
    return None if v is None else 100.0 * float(v)


def bytes_per_position(rec: dict) -> List[float]:
    """[bytes per stage-1 position, per stage-2 position, ...] from a train record's realized compression."""
    out, c, s = [], 1.0, 0
    while f"compression/s{s}" in rec:
        c *= float(rec[f"compression/s{s}"])
        out.append(c)
        s += 1
    return out


def final_train_record(run: RunData) -> dict:
    return run.metrics[-1] if run.metrics else {}


def stable_points(run: RunData, key: str = "val/bpb"):
    """(tokens_eq, value) of eval records in the stable phase (lr_factor == 1 at that step)."""
    lf = run.lr_factor_at()
    pts = [(float(r["tokens_eq"]), float(r[key])) for r in run.evals
           if key in r and lf.get(int(r["step"])) == 1.0 and math.isfinite(float(r[key]))]
    return pts


def multiplier(base_runs: List[RunData], cand_runs: List[RunData], n_boot: int = 300) -> Optional[dict]:
    """Blog-style data multiplier of the candidate vs the baseline (stable phase vs stable phase), or None
    when there are too few stable points (needs >= 4 baseline points and one per candidate seed)."""
    from hnet.eval.stats import data_multiplier

    base = [stable_points(r) for r in base_runs if r.finished and not r.diverged]
    cand = [stable_points(r) for r in cand_runs if r.finished and not r.diverged]
    base = [b for b in base if len(b) >= 4]
    cand = [c for c in cand if c]
    if not base or not cand:
        return None
    n = min(len(b) for b in base)
    tokens = np.mean([[t for t, _ in b[:n]] for b in base], axis=0)
    values = np.mean([[v for _, v in b[:n]] for b in base], axis=0)
    c_tok = float(np.mean([c[-1][0] for c in cand]))
    c_vals = [c[-1][1] for c in cand]
    res = data_multiplier(tokens, values, c_tok, float(np.mean(c_vals)), method="powerlaw", increasing=False,
                          n_boot=n_boot, cand_samples=c_vals if len(c_vals) > 1 else None)
    res["cand_tokens"] = c_tok
    return res
