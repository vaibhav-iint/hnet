"""Scaling-ladder analysis: iso-FLOP frontiers, compute power laws, compute multipliers, loss-to-loss laws.

Reproduces the analysis of Cartesia's "From Tokens to Bytes" I/II (docs/BLOG_REPRO_PLAN.md) on our ladder runs;
the CLI with plots is ``scripts/analysis/fit_ladder.py``, the conventions are documented in docs/ANALYSIS_LADDER.md.

Data model
----------
A ``Run`` is one finished training run: architecture label (``bpe`` / ``hnet`` / ...), training FLOPs ``flops``
(integrated ``flops_per_byte x bytes`` from metrics.jsonl, or an override), parameters ``params``, bytes seen ``D``,
and a dict of losses ``metrics`` (all bits per byte, lower is better):

* ``val``              final validation bpb (eval.jsonl ``val/bpb``)
* ``src:<name>``       per-source bpb (``scripts/eval/per_source.py`` -> ``<run>/per_source.json``)
* ``group:<g>``        pooled bpb of a source group (english, code, math, multilingual, code_held_out, all, mix, ...)
* ``down:<task>``      downstream soft bpb of the gold continuation (``scripts/eval/downstream_bpb.py``), ``down:avg``

Pipeline per metric (``fit_metric``)
------------------------------------
1. Runs are grouped into compute budgets (nominal ``budget`` if given, else clustering of log10 C) and into cells
   (arch, budget, model size); seeds of a cell are averaged.
2. Iso-FLOP fit per (arch, budget): L = c0 + c1 x + c2 x^2 with x = ln N. Compute-optimal point x* = -c1 / 2 c2
   (``status="quadratic"``); if x* lies outside the sampled sizes it is clamped to the nearest edge
   (``"edge"``: the optimum is not bracketed, the loss is an upper bound); with < 3 sizes or c2 <= 0 the best
   observed cell is used (``"min_observed"``). D_opt from a log-linear fit of ln D vs ln N within the budget.
3. Power law of the compute-optimal losses: L(C) = E + A (C / C0)^-alpha with
   * ``free``   E fitted per architecture (profile likelihood over E, >= 3 budgets; exactly determined at 3),
   * ``shared`` one E shared by all architectures (same data entropy; 2 x 3 budgets -> 5 parameters / 6 points),
   * ``pure``   E = 0 (log-log linear, >= 2 budgets), ``fixed`` E given.
   ``auto``: >= 4 budgets -> free, 3 -> shared, 2 -> pure, 1 -> no law (degenerate, explained in ``notes``).
4. Compute multiplier of ``new`` (H-Net) over ``ref`` (BPE) at matched loss: m(L) = C_ref(L) / C_new(L), on the
   loss interval covered by both frontiers (else the new architecture's interval, flagged ``extrapolated``);
   reported as the geometric mean over a grid of that interval, at both ends and at each budget of ``new``.
   ``multiplier_interp`` is the law-free check (log-log interpolation of the ref frontier).
5. Bootstrap CIs (``n_boot``): ``seeds`` mode resamples seeds within each cell (needs replicate seeds);
   ``residual`` mode adds resampled iso-FLOP residuals (pooled per architecture, dof-corrected) to the fitted cell
   values. ``auto`` uses ``seeds`` when any cell has >= 2 seeds.

Loss-to-loss (``fit_loss_to_loss``): per architecture, over *all* runs (on and off the frontier), downstream bpb
y = K x^kappa of validation bpb x (log-log linear). The Part II multiplier: at H-Net's validation loss L (on its
compute frontier), H-Net's downstream loss is d = g_new(L); BPE reaches d at validation loss L' = g_ref^-1(d), which
costs C_ref(L') instead of C_ref(L): m_l2l = C_ref(L') / C_ref(L). The total compute -> downstream multiplier is
m_val(L) x m_l2l(L) (Part II Table: 1.28 x 1.61 = 2.06 for BPE).
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

C0 = 1e18  # reference compute for the power-law amplitude (numerics only)


# ====================================================================================================== runs
@dataclass
class Run:
    name: str
    arch: str
    flops: float  # training FLOPs
    params: float  # parameter count used for N (total parameters by default)
    D: Optional[float] = None  # training bytes
    seed: Optional[int] = None
    budget: Optional[float] = None  # nominal budget (FLOPs); None -> clustered from ``flops``
    metrics: Dict[str, float] = field(default_factory=dict)
    info: Dict = field(default_factory=dict)
    path: Optional[str] = None
    done: bool = True

    @staticmethod
    def from_dict(d: dict) -> "Run":
        keys = {"name", "arch", "flops", "params", "D", "seed", "budget", "metrics", "info", "path", "done"}
        return Run(**{k: v for k, v in d.items() if k in keys})

    def to_dict(self) -> dict:
        return {"name": self.name, "arch": self.arch, "flops": self.flops, "params": self.params, "D": self.D,
                "seed": self.seed, "budget": self.budget, "metrics": self.metrics, "path": self.path,
                "done": self.done}


def _read_jsonl(path: str) -> List[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line cut short by a hard kill
    return out


def integrated_flops(records: List[dict]) -> Tuple[Optional[float], Optional[float]]:
    """(training FLOPs, bytes seen) from metrics.jsonl: sum of flops_per_byte x bytes over the logged intervals
    (realized compression changes flops_per_byte over training for H-Nets)."""
    prev_b, tot, last_fpb, last_b = 0.0, 0.0, None, None
    for r in records:
        b, fpb = r.get("bytes_seen"), r.get("flops_per_byte")
        if b is None:
            continue
        if fpb is not None and math.isfinite(fpb):
            tot += float(fpb) * max(0.0, float(b) - prev_b)
            last_fpb = float(fpb)
        elif last_fpb is not None:
            tot += last_fpb * max(0.0, float(b) - prev_b)
        prev_b = float(b)
        last_b = float(b)
    if last_b is None or tot <= 0:
        return None, last_b
    return tot, last_b


def _downstream_from_eval(rec: dict) -> Dict[str, float]:
    """Trainer-logged paper7 soft bpb (lm-eval prompt boundary) from an eval.jsonl record."""
    out = {}
    pre = "downstream/tasks/"
    for k, v in rec.items():
        if not k.startswith(pre) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        task, _, rest = k[len(pre):].partition("/metrics/")
        if rest in ("soft_bpb/value", "bpb/value"):
            out[f"down:{task}"] = float(v)
    if out:
        out["down:avg"] = float(np.mean(list(out.values())))
    return out


def metrics_from_per_source(ps: dict) -> Tuple[Dict[str, float], dict]:
    """per_source.json -> ({src:<name>, group:<g>: bpb}, info with source metadata and compression)."""
    m, info = {}, {"sources": {}, "compression": {}}
    for name, s in ps.get("sources", {}).items():
        if s.get("bpb") is not None:
            m[f"src:{name}"] = float(s["bpb"])
        info["sources"][name] = {k: s.get(k) for k in ("group", "lang", "held_out", "n_bytes", "weight")}
        info["compression"][name] = {k: s.get(k) for k in ("bytes_per_main_position", "bytes_per_token",
                                                            "split_bytes_per_token")}
    for g, s in ps.get("groups", {}).items():
        if s.get("bpb") is not None:
            m[f"group:{g}"] = float(s["bpb"])
    info["settings"] = ps.get("settings", {})
    return m, info


def metrics_from_downstream(ds: dict) -> Dict[str, float]:
    m = {f"down:{t}": float(v["soft_bpb"]) for t, v in ds.get("tasks", {}).items() if v.get("soft_bpb") is not None}
    if ds.get("average", {}).get("soft_bpb") is not None:
        m["down:avg"] = float(ds["average"]["soft_bpb"])
    for name, v in ds.get("domains", {}).get("sources", {}).items():
        if v.get("bpb") is not None:
            m[f"dom:{name}"] = float(v["bpb"])
    return m


def default_arch(model_section: dict) -> str:
    t = model_section.get("type", "hnet")
    return {"dense": "bpe", "hnet": "hnet"}.get(t, t)


def load_run(path: str, arch: Optional[str] = None, param_key: str = "params",
             per_source: Optional[str] = None, downstream: Optional[str] = None) -> Run:
    """Read a trainer run directory (config.yaml, metrics.jsonl, eval.jsonl, done.json [, per_source.json,
    downstream_bpb.json, sweep_job.json, ladder.json]). Architecture label: ladder.json > ``arch`` > the sweep arm
    (sweep_job.json ``arch``) > model.type (dense -> bpe). Budget: ladder.json > sweep point ``budget`` > the spec
    name (``ladder_1e17_L05`` -> 1e17) > clustering in ``assign_budgets``. ``ladder.json`` may override any of
    flops/params/budget/arch/seed/D."""
    import yaml

    cfg = {}
    cp = os.path.join(path, "config.yaml")
    if os.path.exists(cp):
        with open(cp) as f:
            cfg = yaml.safe_load(f) or {}
    msec = cfg.get("model", {})
    params = params_ne = None
    try:
        from hnet.flops import count_params, count_params_nonembed
        from hnet.train.config import model_config_from_section

        mc = model_config_from_section(msec)
        params, params_ne = float(count_params(mc)), float(count_params_nonembed(mc))
    except Exception:  # custom/unknown model types: rely on ladder.json
        pass
    flops, D = integrated_flops(_read_jsonl(os.path.join(path, "metrics.jsonl")))
    ev = [r for r in _read_jsonl(os.path.join(path, "eval.jsonl")) if "val/bpb" in r]
    metrics, info = {}, {"down_source": None}
    if ev:
        metrics["val"] = float(ev[-1]["val/bpb"])
        info["val_step"] = ev[-1].get("step")
        dm = _downstream_from_eval(ev[-1])
        if dm:
            metrics.update(dm)
            info["down_source"] = "eval.jsonl (trainer, lm-eval prompt boundary)"
    psp = per_source or os.path.join(path, "per_source.json")
    if os.path.exists(psp):
        with open(psp) as f:
            pm, pinfo = metrics_from_per_source(json.load(f))
        metrics.update(pm)
        info.update(pinfo)
    dsp = downstream or os.path.join(path, "downstream_bpb.json")
    if os.path.exists(dsp):
        with open(dsp) as f:
            ds = json.load(f)
        metrics = {k: v for k, v in metrics.items() if not k.startswith("down:")}
        metrics.update(metrics_from_downstream(ds))
        info["down_source"] = f"downstream_bpb.json (boundary={ds.get('boundary')})"
    seed = cfg.get("train", {}).get("seed")
    budget = job_arch = None
    sj = os.path.join(path, "sweep_job.json")
    if os.path.exists(sj):
        with open(sj) as f:
            job = json.load(f)
        pt = job.get("point") or {}
        budget = pt.get("budget") or pt.get("flops_budget") or pt.get("flops")
        if budget is None:  # scripts/ladder/make_ladder.py specs are named ladder_<C>_L<depth>
            mt = re.search(r"(?:^|_)(\d+(?:\.\d+)?e[+-]?\d+)(?:_|$)", str(job.get("spec", "")))
            budget = float(mt.group(1)) if mt else None
        seed = job.get("seed", seed)
        job_arch = job.get("arch")  # the sweep arm (bpe / hnet / hnet_ws / ...)
    over = {}
    lj = os.path.join(path, "ladder.json")
    if os.path.exists(lj):
        with open(lj) as f:
            over = json.load(f)
    N = {"params": params, "params_nonembed": params_ne}.get(param_key, params)
    run = Run(name=os.path.relpath(path), arch=over.get("arch") or arch or job_arch or default_arch(msec),
              flops=over.get("flops", flops), params=over.get("params", N), D=over.get("D", D),
              seed=over.get("seed", seed), budget=over.get("budget", budget), metrics=metrics, info=info,
              path=path, done=os.path.exists(os.path.join(path, "done.json")))
    run.info.update({"params_total": params, "params_nonembed": params_ne})
    return run


def load_runs(specs: Iterable[Tuple[Optional[str], str]], include_unfinished: bool = False, **kw) -> List[Run]:
    """specs: (arch label or None, glob of run dirs). Directories without metrics are skipped."""
    out = []
    for arch, pattern in specs:
        for d in sorted(glob.glob(pattern)):
            if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "metrics.jsonl")):
                continue
            r = load_run(d, arch=arch, **kw)
            if (r.done or include_unfinished) and r.flops and r.params:
                out.append(r)
    return out


def assign_budgets(runs: List[Run], tol_dex: float = 0.15) -> Dict[str, float]:
    """Fill ``run.budget`` for runs without one: runs whose log10 FLOPs are within ``tol_dex`` of the previous run
    (sorted) share a budget, labelled by the geometric mean of their FLOPs. Returns {run name: budget}."""
    free = sorted([r for r in runs if r.budget is None], key=lambda r: r.flops)
    clusters, cur = [], []
    for r in free:
        if cur and math.log10(r.flops) - math.log10(cur[-1].flops) > tol_dex:
            clusters.append(cur)
            cur = []
        cur.append(r)
    if cur:
        clusters.append(cur)
    for c in clusters:
        b = float(10 ** np.mean([math.log10(r.flops) for r in c]))
        b = float(f"{b:.2g}")
        for r in c:
            r.budget = b
    for r in runs:
        r.budget = float(r.budget)
    return {r.name: r.budget for r in runs}


# ====================================================================================================== cells
@dataclass
class Cell:
    arch: str
    budget: float
    logN: float
    logD: float
    logC: float
    ys: np.ndarray  # one loss per seed
    y: float = 0.0  # mean (replaced by resampled values in the bootstrap)


def make_cells(runs: List[Run], metric: str) -> List[Cell]:
    groups: Dict[tuple, list] = {}
    for r in runs:
        v = r.metrics.get(metric)
        if v is None or not math.isfinite(v):
            continue
        key = (r.arch, r.budget, round(math.log(r.params), 3))
        groups.setdefault(key, []).append(r)
    cells = []
    for (arch, budget, _), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        ys = np.array([r.metrics[metric] for r in rs], dtype=float)
        D = [r.D for r in rs if r.D]
        cells.append(Cell(arch, budget, float(np.mean([math.log(r.params) for r in rs])),
                          float(np.mean(np.log(D))) if D else float("nan"),
                          float(np.mean([math.log(r.flops) for r in rs])), ys, float(ys.mean())))
    return cells


# ====================================================================================================== iso-FLOP
def fit_isoflop(x: np.ndarray, y: np.ndarray, logD: Optional[np.ndarray] = None) -> dict:
    """Quadratic in x = ln N. See module docstring for ``status``."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    i = int(np.argmin(y))
    res = {"n": len(x), "x_opt": float(x[i]), "L_opt": float(y[i]), "status": "min_observed", "coef": None,
           "resid": np.zeros(len(x)), "dof": 0}
    if len(np.unique(np.round(x, 6))) >= 3:
        xm = x.mean()
        c2, c1, c0 = np.polyfit(x - xm, y, 2)
        fit = c0 + c1 * (x - xm) + c2 * (x - xm) ** 2
        res["resid"] = y - fit
        res["dof"] = len(x) - 3
        res["coef"] = [float(c0), float(c1), float(c2), float(xm)]
        if c2 > 0:
            xs = -c1 / (2 * c2)
            status = "quadratic"
            lo, hi = x.min() - xm, x.max() - xm
            if xs < lo or xs > hi:
                xs = min(max(xs, lo), hi)
                status = "edge"
            res.update(x_opt=float(xs + xm), L_opt=float(c0 + c1 * xs + c2 * xs * xs), status=status)
    if logD is not None:
        d = np.asarray(logD, float)
        ok = np.isfinite(d)
        if ok.sum() >= 2 and len(np.unique(np.round(x[ok], 6))) >= 2:
            b, a = np.polyfit(x[ok], d[ok], 1)
            res["logD_opt"] = float(a + b * res["x_opt"])
        elif ok.any():
            res["logD_opt"] = float(d[ok][np.argmin(np.abs(x[ok] - res["x_opt"]))])
    return res


def frontier(cells: List[Cell], arch: str) -> List[dict]:
    """Compute-optimal point per budget of ``arch`` (sorted by budget)."""
    out = []
    for b in sorted({c.budget for c in cells if c.arch == arch}):
        cs = [c for c in cells if c.arch == arch and c.budget == b]
        f = fit_isoflop(np.array([c.logN for c in cs]), np.array([c.y for c in cs]),
                        np.array([c.logD for c in cs]))
        logC = float(np.mean([c.logC for c in cs]))
        out.append({"budget": b, "C": math.exp(logC), "L": f["L_opt"], "N_opt": math.exp(f["x_opt"]),
                    "D_opt": math.exp(f["logD_opt"]) if f.get("logD_opt") is not None and
                    math.isfinite(f["logD_opt"]) else None,
                    "status": f["status"], "n_sizes": f["n"], "coef": f["coef"], "_resid": f["resid"],
                    "_dof": f["dof"], "flops_spread_dex": float(np.ptp([c.logC for c in cs]) / math.log(10))})
    return out


# ====================================================================================================== power laws
@dataclass
class PowerLaw:
    E: float
    A: float
    alpha: float
    form: str
    rmse: float = 0.0
    n: int = 0
    C0: float = C0

    def loss(self, C):
        return self.E + self.A * (np.asarray(C, float) / self.C0) ** (-self.alpha)

    def compute(self, L):
        """Inverse: compute needed to reach loss L (inf where L <= E)."""
        L = np.asarray(L, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = (L - self.E) / self.A
            out = self.C0 * np.where(r > 0, r, np.nan) ** (-1.0 / self.alpha)
        return np.where(L > self.E, out, np.inf)

    def to_dict(self) -> dict:
        return {"E": self.E, "A": self.A, "alpha": self.alpha, "C0": self.C0, "form": self.form,
                "rmse": self.rmse, "n": self.n}


def _linfit_rows(x: np.ndarray, Y: np.ndarray):
    """Row-wise least squares Y[g, :] = a[g] + b[g] x. Returns a, b."""
    xm = x.mean()
    dx = x - xm
    vx = (dx * dx).sum()
    b = (Y * dx).sum(axis=1) / vx
    a = Y.mean(axis=1) - b * xm
    return a, b


def _profile_E(xs: List[np.ndarray], Ls: List[np.ndarray], Es: np.ndarray):
    """SSE (in L) of the per-array linear fits of ln(L - E) on x, for every E in ``Es`` (shared across arrays).
    Returns (sse[G], [(a[G], b[G]) per array])."""
    sse = np.zeros(len(Es))
    fits = []
    for x, L in zip(xs, Ls):
        Y = np.log(np.maximum(L[None, :] - Es[:, None], 1e-300))
        a, b = _linfit_rows(x, Y)
        pred = Es[:, None] + np.exp(a[:, None] + b[:, None] * x[None, :])
        e = ((pred - L[None, :]) ** 2).sum(axis=1)
        e = np.where(b < 0, e, np.inf)  # alpha = -b must be positive
        sse += e
        fits.append((a, b))
    return sse, fits


def _fit_E_profile(xs, Ls, n_grid: int = 240):
    Lmin = min(float(L.min()) for L in Ls)
    u = np.concatenate([[1.0], np.logspace(-5, 0, n_grid)[::-1]])
    Es = Lmin * (1.0 - u)  # from 0 up to just below the smallest loss
    sse, _ = _profile_E(xs, Ls, Es)
    if not np.isfinite(sse).any():
        return None
    k = int(np.nanargmin(sse))
    lo, hi = Es[max(k - 1, 0)], Es[min(k + 1, len(Es) - 1)]
    fine = np.linspace(lo, hi, 64)
    sse2, _ = _profile_E(xs, Ls, fine)
    if np.isfinite(sse2).any() and np.nanmin(sse2) <= sse[k]:
        return float(fine[int(np.nanargmin(sse2))])
    return float(Es[k])


def fit_power_law(C, L, E="free", C0: float = C0) -> Optional[PowerLaw]:
    """L(C) = E + A (C / C0)^-alpha. ``E``: "free" (profiled, needs >= 3 points), None/0 ("pure"), or a float."""
    C, L = np.asarray(C, float), np.asarray(L, float)
    x = np.log(C / C0)
    if len(C) < 2 or np.ptp(x) == 0:
        return None
    form = "fixed"
    if E == "free":
        if len(C) < 3:
            return None
        E = _fit_E_profile([x], [L])
        if E is None:
            return None
        form = "free"
    elif E is None or E == 0:
        E, form = 0.0, "pure"
    E = float(E)
    if (L <= E).any():
        return None
    a, b = _linfit_rows(x, np.log(L - E)[None, :])
    alpha = -float(b[0])
    if alpha <= 0:
        return None
    pl = PowerLaw(E, float(math.exp(a[0])), alpha, form, n=len(C), C0=C0)
    pl.rmse = float(np.sqrt(np.mean((pl.loss(C) - L) ** 2)))
    return pl


def fit_shared_E(data: Dict[str, Tuple[np.ndarray, np.ndarray]], C0: float = C0) -> Optional[Dict[str, PowerLaw]]:
    """Joint fit with one irreducible loss E for every architecture (each keeps its own A, alpha)."""
    archs = list(data)
    xs = [np.log(np.asarray(data[a][0], float) / C0) for a in archs]
    Ls = [np.asarray(data[a][1], float) for a in archs]
    if any(len(x) < 2 or np.ptp(x) == 0 for x in xs) or sum(len(x) for x in xs) < 2 * len(xs) + 1:
        return None
    E = _fit_E_profile(xs, Ls)
    if E is None:
        return None
    out = {}
    for a in archs:
        pl = fit_power_law(data[a][0], data[a][1], E=E if E > 0 else None, C0=C0)
        if pl is None:
            return None
        pl.form = "shared"
        out[a] = pl
    return out


def choose_form(n_budgets: Dict[str, int], form: str = "auto") -> Optional[str]:
    nb = min(n_budgets.values()) if n_budgets else 0
    if form != "auto":
        need = {"free": 3, "shared": 2, "pure": 2, "fixed": 2}[form]
        return form if nb >= need else None
    if nb >= 4:
        return "free"
    if nb == 3:
        return "shared" if len(n_budgets) >= 2 else "free"
    if nb == 2:
        return "pure"
    return None


def fit_laws(fronts: Dict[str, List[dict]], form: str, E_fixed: Optional[float] = None) -> Dict[str, PowerLaw]:
    data = {a: (np.array([p["C"] for p in f]), np.array([p["L"] for p in f])) for a, f in fronts.items()}
    if form == "shared":
        laws = fit_shared_E(data)
        if laws is not None:
            return laws
        form = "pure"  # degenerate shared fit -> fall back to pure power laws
    out = {}
    for a, (C, L) in data.items():
        E = {"free": "free", "pure": None, "fixed": E_fixed}[form]
        pl = fit_power_law(C, L, E=E)
        if pl is None and form == "free":
            pl = fit_power_law(C, L, E=None)
        if pl is not None:
            out[a] = pl
    return out


# ====================================================================================================== multipliers
def compute_multiplier(laws: Dict[str, PowerLaw], fronts: Dict[str, List[dict]], ref: str, new: str,
                       n_grid: int = 64) -> Optional[dict]:
    """m(L) = C_ref(L) / C_new(L) over the loss range covered by both frontiers (see module docstring)."""
    if ref not in laws or new not in laws:
        return None
    Lr = [p["L"] for p in fronts[ref]]
    Ln = [p["L"] for p in fronts[new]]
    lo, hi = max(min(Lr), min(Ln)), min(max(Lr), max(Ln))
    extrapolated = not lo < hi
    if extrapolated:
        lo, hi = min(Ln), max(Ln)
    grid = np.linspace(lo, hi, n_grid) if hi > lo else np.array([lo])
    m = laws[ref].compute(grid) / laws[new].compute(grid)
    ok = np.isfinite(m) & (m > 0)
    if not ok.any():
        return None
    per_budget = []
    for p in fronts[new]:
        cr = float(laws[ref].compute(p["L"]))
        per_budget.append({"budget": p["budget"], "L": p["L"], "C_new": p["C"], "C_ref": cr,
                           "multiplier": cr / p["C"] if math.isfinite(cr) else None})
    return {"multiplier": float(np.exp(np.mean(np.log(m[ok])))),
            "at_low_loss": float(m[0]) if ok[0] else None, "at_high_loss": float(m[-1]) if ok[-1] else None,
            "loss_range": [float(lo), float(hi)], "extrapolated": bool(extrapolated),
            "grid": {"L": grid.tolist(), "m": [float(v) if math.isfinite(v) else None for v in m]},
            "per_budget": per_budget}


def interp_multiplier(fronts: Dict[str, List[dict]], ref: str, new: str) -> Optional[dict]:
    """Law-free multiplier: C_ref at each of new's frontier losses by log-log interpolation of ref's frontier."""
    fr = sorted(fronts.get(ref, []), key=lambda p: p["L"])
    if len(fr) < 2 or not fronts.get(new):
        return None
    lL = np.log([p["L"] for p in fr])
    lC = np.log([p["C"] for p in fr])
    if np.ptp(lL) == 0:
        return None
    pts = []
    for p in fronts[new]:
        l = math.log(p["L"])
        inside = lL[0] <= l <= lL[-1]
        if inside:
            cr = float(np.exp(np.interp(l, lL, lC)))
        else:  # linear extrapolation from the nearest segment
            i = 0 if l < lL[0] else len(lL) - 2
            s = (lC[i + 1] - lC[i]) / (lL[i + 1] - lL[i])
            cr = float(np.exp(lC[i] + s * (l - lL[i])))
        pts.append({"budget": p["budget"], "multiplier": cr / p["C"], "inside": inside})
    ins = [q["multiplier"] for q in pts if q["inside"]]
    return {"multiplier": float(np.exp(np.mean(np.log(ins)))) if ins else None, "points": pts,
            "n_inside": len(ins)}


# ====================================================================================================== pipeline
def _pipeline(cells: List[Cell], archs: Sequence[str], ref: str, new: str, form: str, E_fixed=None):
    fronts = {a: frontier(cells, a) for a in archs}
    fronts = {a: f for a, f in fronts.items() if f}
    nb = {a: len(f) for a, f in fronts.items()}
    fm = choose_form(nb, form) if len(fronts) == len(archs) else None
    laws = fit_laws(fronts, fm, E_fixed) if fm else {}
    mult = compute_multiplier(laws, fronts, ref, new) if fm else None
    return fronts, laws, mult, fm


def _boot_mode(cells: List[Cell], mode: str) -> str:
    if mode != "auto":
        return mode
    return "seeds" if any(len(c.ys) >= 2 for c in cells) else "residual"


def _residuals(cells: List[Cell], fronts: Dict[str, List[dict]]) -> Dict[str, np.ndarray]:
    out = {}
    for a, f in fronts.items():
        rs = []
        for p in f:
            if p["_dof"] > 0:
                n = len(p["_resid"])
                rs.append(p["_resid"] * math.sqrt(n / p["_dof"]))
        out[a] = np.concatenate(rs) if rs else np.zeros(0)
    return out


def _fitted(cells: List[Cell], fronts: Dict[str, List[dict]]) -> np.ndarray:
    out = np.array([c.y for c in cells])
    for i, c in enumerate(cells):
        for p in fronts.get(c.arch, []):
            if p["budget"] == c.budget and p["coef"] is not None:
                c0, c1, c2, xm = p["coef"]
                out[i] = c0 + c1 * (c.logN - xm) + c2 * (c.logN - xm) ** 2
    return out


def _ci(vals: List[float]) -> Optional[dict]:
    v = np.array([x for x in vals if x is not None and math.isfinite(x) and x > 0])
    if len(v) < 10:
        return None
    lv = np.log(v)
    return {"lo": float(np.exp(np.percentile(lv, 2.5))), "hi": float(np.exp(np.percentile(lv, 97.5))),
            "median": float(np.exp(np.median(lv))), "n": int(len(v))}


def _strip(fronts):
    return {a: [{k: v for k, v in p.items() if not k.startswith("_")} for p in f] for a, f in fronts.items()}


def fit_metric(runs: List[Run], metric: str, ref: str = "bpe", new: str = "hnet", form: str = "auto",
               E_fixed: Optional[float] = None, n_boot: int = 1000, seed: int = 0, boot: str = "auto") -> dict:
    """Full pipeline (iso-FLOP -> power law -> multiplier, with bootstrap CI) for one loss metric."""
    archs = [ref, new]
    cells = [c for c in make_cells(runs, metric) if c.arch in archs]
    notes = []
    res = {"metric": metric, "notes": notes}
    for a in archs:
        if not any(c.arch == a for c in cells):
            notes.append(f"no '{a}' runs with metric '{metric}'")
    fronts, laws, mult, fm = _pipeline(cells, archs, ref, new, form, E_fixed)
    for a, f in fronts.items():
        for p in f:
            if p["status"] == "min_observed":
                notes.append(f"{a} @ {p['budget']:.3g} FLOPs: {p['n_sizes']} model size(s) -> compute-optimal loss "
                             "is the best observed run (iso-FLOP quadratic needs >= 3 sizes with a minimum)")
            elif p["status"] == "edge":
                notes.append(f"{a} @ {p['budget']:.3g} FLOPs: iso-FLOP minimum outside the sampled sizes "
                             "(clamped to the edge; extend the size grid)")
    nb = {a: len(f) for a, f in fronts.items()}
    if fm is None:
        notes.append(f"budgets per architecture {nb}: a compute power law (hence a compute multiplier) needs >= 2 "
                     "budgets per architecture; reporting compute-optimal losses only")
    res.update(form=fm, budgets_per_arch=nb, frontier=_strip(fronts),
               laws={a: l.to_dict() for a, l in laws.items()}, multiplier=mult,
               multiplier_interp=interp_multiplier(fronts, ref, new))
    # matched-budget loss gaps (useful even for a single budget)
    gaps = []
    for p in fronts.get(new, []):
        q = next((q for q in fronts.get(ref, []) if q["budget"] == p["budget"]), None)
        if q is not None:
            gaps.append({"budget": p["budget"], "L_ref": q["L"], "L_new": p["L"], "delta": p["L"] - q["L"],
                         "rel": p["L"] / q["L"] - 1})
    res["matched_budget_gap"] = gaps
    if mult is not None and n_boot > 0:
        mode = _boot_mode(cells, boot)
        rng = np.random.default_rng(seed)
        fitted = _fitted(cells, fronts)
        resid = _residuals(cells, fronts)
        if mode == "residual" and all(len(r) == 0 or np.allclose(r, 0) for r in resid.values()):
            notes.append("bootstrap skipped: no replicate seeds and no iso-FLOP residual degrees of freedom")
        else:
            vals, alphas = [], {a: [] for a in archs}
            for _ in range(n_boot):
                bc = []
                for i, c in enumerate(cells):
                    if mode == "seeds":
                        y = float(rng.choice(c.ys, size=len(c.ys), replace=True).mean())
                    else:
                        r = resid.get(c.arch)
                        y = float(fitted[i] + (rng.choice(r) if r is not None and len(r) else 0.0))
                    bc.append(Cell(c.arch, c.budget, c.logN, c.logD, c.logC, c.ys, y))
                _, bl, bm, _ = _pipeline(bc, archs, ref, new, fm, E_fixed)
                vals.append(bm["multiplier"] if bm else None)
                for a in archs:
                    if a in bl:
                        alphas[a].append(bl[a].alpha)
            ci = _ci(vals)
            mult["ci95"] = ci
            mult["bootstrap"] = {"mode": mode, "n_boot": n_boot,
                                 "n_valid": int(sum(v is not None for v in vals))}
            for a in archs:
                if a in res["laws"] and len(alphas[a]) >= 10:
                    res["laws"][a]["alpha_ci95"] = [float(np.percentile(alphas[a], 2.5)),
                                                    float(np.percentile(alphas[a], 97.5))]
    res["_laws"] = laws
    res["_fronts"] = fronts
    return res


# ====================================================================================================== N_opt, D_opt
def optimal_allocation(fronts: Dict[str, List[dict]]) -> Dict[str, dict]:
    """N_opt(C) = N0 (C/C0)^a and D_opt(C) = D0 (C/C0)^b per architecture (log-log fits over budgets)."""
    out = {}
    for arch, f in fronts.items():
        C = np.array([p["C"] for p in f])
        d = {"points": [{"C": p["C"], "N_opt": p["N_opt"], "D_opt": p["D_opt"], "status": p["status"]} for p in f]}
        if len(f) >= 2 and np.ptp(np.log(C)) > 0:
            x = np.log(C / C0)
            b, a = np.polyfit(x, np.log([p["N_opt"] for p in f]), 1)
            d["N_opt"] = {"N0": float(math.exp(a)), "exponent": float(b)}
            Ds = [p["D_opt"] for p in f]
            if all(v for v in Ds):
                b2, a2 = np.polyfit(x, np.log(Ds), 1)
                d["D_opt"] = {"D0": float(math.exp(a2)), "exponent": float(b2)}
        out[arch] = d
    return out


# ====================================================================================================== loss-to-loss
@dataclass
class L2L:
    K: float
    kappa: float
    rmse_rel: float
    n: int
    x_range: Tuple[float, float]

    def __call__(self, x):
        return self.K * np.asarray(x, float) ** self.kappa

    def inverse(self, y):
        return (np.asarray(y, float) / self.K) ** (1.0 / self.kappa)

    def to_dict(self):
        return {"K": self.K, "kappa": self.kappa, "rmse_rel": self.rmse_rel, "n": self.n,
                "x_range": list(self.x_range)}


def fit_l2l(x, y) -> Optional[L2L]:
    """Downstream loss y = K x^kappa of validation loss x (log-log linear least squares)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.ptp(np.log(x)) == 0:
        return None
    k, a = np.polyfit(np.log(x), np.log(y), 1)
    if k <= 0:
        return None
    g = L2L(float(math.exp(a)), float(k), 0.0, len(x), (float(x.min()), float(x.max())))
    g.rmse_rel = float(np.sqrt(np.mean((g(x) / y - 1) ** 2)))
    return g


def l2l_multiplier(law_ref: PowerLaw, g_ref: L2L, g_new: L2L, L_grid) -> dict:
    """Part II: m_l2l(L) = C_ref(L') / C_ref(L) with g_ref(L') = g_new(L)."""
    L = np.asarray(L_grid, float)
    Lp = g_ref.inverse(g_new(L))
    m = law_ref.compute(Lp) / law_ref.compute(L)
    ok = np.isfinite(m) & (m > 0)
    return {"multiplier": float(np.exp(np.mean(np.log(m[ok])))) if ok.any() else None,
            "grid": {"L": L.tolist(), "L_ref_matched": Lp.tolist(),
                     "m": [float(v) if math.isfinite(v) else None for v in m]}}


def fit_loss_to_loss(runs: List[Run], x_metric: str = "val", y_metric: str = "down:avg", ref: str = "bpe",
                     new: str = "hnet", val_fit: Optional[dict] = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Loss-to-loss laws per architecture and the Part II multipliers (needs ``val_fit`` = fit_metric(...,'val')
    for the compute -> validation law of ``ref`` and the frontier of ``new``)."""
    notes = []
    pts = {a: [(r.metrics[x_metric], r.metrics[y_metric]) for r in runs if r.arch == a and
               x_metric in r.metrics and y_metric in r.metrics] for a in (ref, new)}
    laws = {a: fit_l2l([p[0] for p in v], [p[1] for p in v]) for a, v in pts.items()}
    res = {"x": x_metric, "y": y_metric, "points": {a: v for a, v in pts.items()},
           "laws": {a: (g.to_dict() if g else None) for a, g in laws.items()}, "notes": notes}
    for a, g in laws.items():
        xs = [p[0] for p in pts[a]]
        spread = (max(xs) / min(xs) - 1) if xs and min(xs) > 0 else 0.0
        if g is None:
            why = ("needs >= 2 runs with distinct validation losses" if len(set(xs)) < 2 else
                   "fitted exponent <= 0: downstream loss does not increase with validation loss across these runs")
            notes.append(f"{a}: {len(pts[a])} run(s) with both '{x_metric}' and '{y_metric}': "
                         f"no loss-to-loss law ({why})")
        elif spread < 0.02:
            notes.append(f"{a}: validation losses span only {spread:.2%} (e.g. seeds of one configuration): the "
                         "loss-to-loss law is unconstrained; use all runs of the iso-FLOP sweeps")
    vl = (val_fit or {}).get("_laws", {})
    vf = (val_fit or {}).get("_fronts", {})
    if laws[ref] is None or laws[new] is None:
        return res
    if ref not in vl or new not in vl or not vf.get(new):
        notes.append("compute -> validation laws unavailable: loss-to-loss multiplier not computed")
        return res
    Ln = [p["L"] for p in vf[new]]
    lo = max(min(Ln), laws[ref].x_range[0], laws[new].x_range[0])
    hi = min(max(Ln), laws[ref].x_range[1], laws[new].x_range[1])
    extrap = not lo < hi
    if extrap:
        lo, hi = min(Ln), max(Ln)
        notes.append("H-Net frontier losses outside the range of the loss-to-loss fits: extrapolated")
    grid = np.linspace(lo, hi, 32) if hi > lo else np.array([lo])
    m_l2l = l2l_multiplier(vl[ref], laws[ref], laws[new], grid)
    m_val = vl[ref].compute(grid) / vl[new].compute(grid)
    tot = m_val * np.array([v if v is not None else np.nan for v in m_l2l["grid"]["m"]])
    ok = np.isfinite(tot) & (tot > 0)
    res.update(val_to_down_multiplier=m_l2l["multiplier"], grid=m_l2l["grid"], extrapolated=bool(extrap),
               compute_to_val_multiplier=float(np.exp(np.mean(np.log(m_val[np.isfinite(m_val) & (m_val > 0)]))))
               if np.isfinite(m_val).any() else None,
               compute_to_down_multiplier=float(np.exp(np.mean(np.log(tot[ok])))) if ok.any() else None,
               loss_range=[float(lo), float(hi)])
    if n_boot > 0:  # case bootstrap over runs for the loss-to-loss laws (validation law held fixed)
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(n_boot):
            g = {}
            for a in (ref, new):
                v = pts[a]
                idx = rng.integers(0, len(v), len(v))
                g[a] = fit_l2l([v[i][0] for i in idx], [v[i][1] for i in idx])
            if g[ref] is None or g[new] is None:
                bs.append(None)
                continue
            bs.append(l2l_multiplier(vl[ref], g[ref], g[new], grid)["multiplier"])
        res["val_to_down_ci95"] = _ci(bs)
    return res


# ====================================================================================================== whole ladder
def available_metrics(runs: List[Run], ref: str, new: str) -> List[str]:
    have = {a: set().union(*[set(r.metrics) for r in runs if r.arch == a]) if any(r.arch == a for r in runs)
            else set() for a in (ref, new)}
    common = have[ref] & have[new]

    def key(m):
        order = {"val": 0, "group": 1, "src": 2, "down": 3, "dom": 4}
        return (order.get(m.split(":")[0], 9), m)

    return sorted(common, key=key)


def source_meta(runs: List[Run]) -> Dict[str, dict]:
    meta = {}
    for r in runs:
        for k, v in r.info.get("sources", {}).items():
            meta.setdefault(k, v)
    return meta


def analyze_ladder(runs: List[Run], ref: str = "bpe", new: str = "hnet", metrics: Optional[List[str]] = None,
                   form: str = "auto", E_fixed: Optional[float] = None, n_boot: int = 1000, seed: int = 0,
                   boot: str = "auto", budget_tol_dex: float = 0.15, l2l_y: str = "down:avg") -> dict:
    """Everything fit_ladder.py reports. Private keys (``_laws``, ``_fronts``) are kept for plotting."""
    assign_budgets(runs, budget_tol_dex)
    notes = []
    budgets = sorted({r.budget for r in runs})
    if len(budgets) < 2:
        notes.append(f"single compute budget ({', '.join(f'{b:.3g}' for b in budgets)} FLOPs): compute "
                     "multipliers and power laws are undefined; only matched-budget loss gaps are reported. "
                     "Run >= 2 (preferably >= 3) budgets with >= 3 model sizes each.")
    metrics = metrics or available_metrics(runs, ref, new)
    out = {"ref": ref, "new": new, "budgets": budgets, "notes": notes,
           "runs": [r.to_dict() for r in runs], "metrics": {}}
    seq = {}
    for r in runs:
        s = r.info.get("settings", {})
        if s:
            seq[r.name] = (s.get("byte_seq_len"), s.get("token_seq_len"), s.get("split_tokenizer"))
    if len(set(seq.values())) > 1:
        notes.append(f"per-source windows differ between runs: {sorted(set(seq.values()))} -- per-source bpb is "
                     "not comparable across runs evaluated with different settings")
    for m in metrics:
        out["metrics"][m] = fit_metric(runs, m, ref, new, form, E_fixed, n_boot if m == "val" or
                                       m.startswith(("group:", "src:", "down:avg")) else 0, seed, boot)
    if "val" in out["metrics"]:
        out["allocation"] = optimal_allocation(out["metrics"]["val"]["_fronts"])
        out["loss_to_loss"] = fit_loss_to_loss(runs, "val", l2l_y, ref, new, out["metrics"]["val"],
                                               n_boot=n_boot, seed=seed)
    out["source_meta"] = source_meta(runs)
    return out


def public(res: dict) -> dict:
    """Drop private (``_``-prefixed) keys recursively for JSON output."""
    if isinstance(res, dict):
        return {k: public(v) for k, v in res.items() if not str(k).startswith("_")}
    if isinstance(res, list):
        return [public(v) for v in res]
    if isinstance(res, (np.floating, np.integer)):
        return res.item()
    if isinstance(res, np.ndarray):
        return res.tolist()
    return res
