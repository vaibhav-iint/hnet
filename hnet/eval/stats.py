"""Bootstrap confidence intervals, paired model comparisons and data-multiplier estimation.

* ``bootstrap(stat_fn, n, ...)``: percentile CI of any statistic of item indices (optionally stratified,
  e.g. by MMLU subject); ``mean_ci``, ``ratio_ci`` (bpb = sum nll / sum bytes) are thin wrappers.
* ``average_ci(tasks)``: CI of an average of per-task statistics, resampling each task's items independently.
* ``paired_bootstrap(a, b)``: two models scored on the *same* items; CI of the difference and a two-sided p-value.
* ``fit_curve`` / ``data_multiplier``: fit a smooth monotone baseline curve value(tokens) and return the baseline
  tokens needed to reach a candidate's value, divided by the candidate's tokens; CI by residual bootstrap of the
  baseline fit (plus the candidate's own sampling noise if bootstrap samples of its value are given).
  Methods: ``powerlaw`` (v = E + A t^-alpha; default for bpb/loss), ``quadratic_log`` (quadratic in log t, the
  fit used for SmolLM3's stage-1 points in our plan), ``isotonic`` (PAV in log t, linear interpolation; for noisy
  accuracies, no extrapolation).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


# ====================================================================== bootstrap
def _resample(rng, n, strata):
    if strata is None:
        return rng.integers(0, n, n)
    idx = []
    for s in strata:
        idx.append(s[rng.integers(0, len(s), len(s))])
    return np.concatenate(idx)


def _strata(groups):
    if groups is None:
        return None
    groups = np.asarray(groups, dtype=object)
    return [np.flatnonzero(groups == g) for g in dict.fromkeys(groups.tolist())]


def bootstrap(stat_fn: Callable[[np.ndarray], float], n: int, n_boot: int = 1000, seed: int = 0,
              alpha: float = 0.05, groups=None) -> Dict[str, float]:
    """Percentile bootstrap of ``stat_fn(indices)``. ``groups``: per-item labels for stratified resampling."""
    rng = np.random.default_rng(seed)
    strata = _strata(groups)
    point = float(stat_fn(np.arange(n)))
    if n == 0 or n_boot <= 0:
        return {"value": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    bs = np.array([stat_fn(_resample(rng, n, strata)) for _ in range(n_boot)], dtype=float)
    lo, hi = np.nanquantile(bs, [alpha / 2, 1 - alpha / 2])
    return {"value": point, "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(bs, ddof=1))}


def mean_ci(values, **kw):
    v = np.asarray(values, dtype=float)
    return bootstrap(lambda i: v[i].mean(), len(v), **kw)


def macro_mean_ci(values, groups, **kw):
    """Mean over groups of the per-group mean (lighteval MMLU average), stratified by group."""
    v = np.asarray(values, dtype=float)
    g = np.asarray(groups, dtype=object)

    def stat(i):
        gi = g[i]
        return float(np.mean([v[i][gi == k].mean() for k in dict.fromkeys(gi.tolist())]))

    return bootstrap(stat, len(v), groups=groups, **kw)


def ratio_ci(num, den, **kw):
    a, b = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return bootstrap(lambda i: a[i].sum() / b[i].sum(), len(a), **kw)


def average_ci(stats: Sequence[tuple], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """``stats``: list of (stat_fn(indices) -> float, n_items, groups-or-None[, weight]) per task.
    CI of the weighted mean of the task statistics; each task's items are resampled independently."""
    stats = [tuple(s) + (1.0,) * (4 - len(s)) for s in stats]
    w = np.array([s[3] for s in stats], dtype=float)
    w = w / w.sum()
    rng = np.random.default_rng(seed)
    point = float(np.dot(w, [f(np.arange(n)) for f, n, _, _ in stats]))
    if n_boot <= 0:
        return {"value": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan")}
    strata = [_strata(g) for _, _, g, _ in stats]
    bs = np.empty(n_boot)
    for b in range(n_boot):
        bs[b] = np.dot(w, [f(_resample(rng, n, s)) for (f, n, _, _), s in zip(stats, strata)])
    lo, hi = np.nanquantile(bs, [alpha / 2, 1 - alpha / 2])
    return {"value": point, "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(bs, ddof=1))}


def paired_bootstrap(a, b, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05, groups=None,
                     stat: Callable[[np.ndarray], float] = np.mean):
    """Difference stat(a) - stat(b) over the same items (a[i], b[i] from the same item). Returns the CI of the
    difference and a two-sided bootstrap p-value for 'no difference'."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape
    d = bootstrap(lambda i: stat(a[i]) - stat(b[i]), len(a), n_boot=n_boot, seed=seed, alpha=alpha, groups=groups)
    rng = np.random.default_rng(seed)
    strata = _strata(groups)
    bs = np.array([stat(a[i]) - stat(b[i]) for i in (_resample(rng, len(a), strata) for _ in range(n_boot))])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    d["p_value"] = float(min(1.0, p))
    return d


# ====================================================================== monotone curve fits
def _pav(y, w=None, increasing=True):
    """Pool-adjacent-violators isotonic regression."""
    y = np.asarray(y, dtype=float)
    if not increasing:
        return -_pav(-y, w, True)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi), wts.append(wi), cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)), wts.append(w1 + w2), cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


@dataclass
class Curve:
    method: str
    increasing: bool
    t_min: float
    t_max: float
    params: dict

    def predict(self, t):
        t = np.asarray(t, dtype=float)
        p = self.params
        if self.method == "powerlaw":
            return p["E"] + p["A"] * t ** (-p["alpha"])
        x = np.log(t)
        if self.method == "quadratic_log":
            return np.polyval(p["coef"], x)  # monotonicity is enforced in inverse()
        if self.method == "isotonic":
            return np.interp(x, p["x"], p["y"])
        raise ValueError(self.method)

    def inverse(self, v: float, extrapolate: float = 4.0) -> float:
        """Smallest t (on a log grid in [t_min, t_max * extrapolate]) where the curve reaches ``v``;
        nan if it never does. Isotonic fits never extrapolate."""
        hi = self.t_max * (1.0 if self.method == "isotonic" else extrapolate)
        grid = np.exp(np.linspace(math.log(self.t_min), math.log(hi), 4001))
        y = self.predict(grid)
        y = np.maximum.accumulate(y) if self.increasing else np.minimum.accumulate(y)
        s = y if self.increasing else -y
        target = v if self.increasing else -v
        if target < s[0] or target > s[-1]:
            return float("nan")
        k = int(np.searchsorted(s, target, side="left"))
        if k == 0:
            return float(grid[0])
        x0, x1, y0, y1 = math.log(grid[k - 1]), math.log(grid[k]), s[k - 1], s[k]
        f = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
        return float(math.exp(x0 + f * (x1 - x0)))


def fit_curve(tokens, values, method: str = "powerlaw", increasing: Optional[bool] = None) -> Curve:
    t = np.asarray(tokens, dtype=float)
    y = np.asarray(values, dtype=float)
    o = np.argsort(t)
    t, y = t[o], y[o]
    if increasing is None:
        increasing = bool(np.polyfit(np.log(t), y, 1)[0] > 0)
    if method == "powerlaw":
        best = None
        for alpha in np.exp(np.linspace(math.log(1e-3), math.log(3.0), 600)):
            X = np.stack([np.ones_like(t), t ** (-alpha)], 1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            sse = float(((X @ coef - y) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, alpha, coef)
        _, alpha, (E, A) = best
        return Curve("powerlaw", increasing, t[0], t[-1], {"E": float(E), "A": float(A), "alpha": float(alpha)})
    if method == "quadratic_log":
        x = np.log(t)
        coef = np.polyfit(x, y, 2)
        return Curve("quadratic_log", increasing, t[0], t[-1], {"coef": coef.tolist()})
    if method == "isotonic":
        x = np.log(t)
        return Curve("isotonic", increasing, t[0], t[-1], {"x": x.tolist(), "y": _pav(y, None, increasing).tolist()})
    raise ValueError(method)


def data_multiplier(base_tokens, base_values, cand_tokens: float, cand_value: float, method: str = "powerlaw",
                    increasing: Optional[bool] = None, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                    cand_samples: Optional[Sequence[float]] = None, extrapolate: float = 4.0) -> dict:
    """Baseline tokens needed to match ``cand_value``, divided by ``cand_tokens`` (blog / PLAN definition).

    ``increasing``: True for accuracies, False for bpb (inferred from the baseline trend if None).
    CI: refit the baseline on fitted + resampled residuals (and draw the candidate value from ``cand_samples``,
    e.g. its item-bootstrap replicates, when given). Bootstrap draws that never reach the value count as +inf."""
    t = np.asarray(base_tokens, dtype=float)
    y = np.asarray(base_values, dtype=float)
    curve = fit_curve(t, y, method, increasing)
    need = curve.inverse(cand_value, extrapolate)
    res = {"method": method, "increasing": curve.increasing, "tokens_needed": need,
           "multiplier": need / cand_tokens if np.isfinite(need) else float("nan"),
           "in_range": bool(np.isfinite(need) and need <= curve.t_max), "curve": curve.params}
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        fitted = curve.predict(t)
        resid = y - fitted
        ms = []
        for _ in range(n_boot):
            yb = fitted + rng.choice(resid, len(resid), replace=True)
            cb = fit_curve(t, yb, method, curve.increasing)
            v = cand_value if cand_samples is None else float(rng.choice(np.asarray(cand_samples)))
            nb = cb.inverse(v, extrapolate)
            ms.append(nb / cand_tokens if np.isfinite(nb) else np.inf)
        ms = np.array(ms)
        lo, hi = np.quantile(ms, [alpha / 2, 1 - alpha / 2])
        res.update({"ci_lo": float(lo), "ci_hi": float(hi), "frac_unreached": float(np.isinf(ms).mean())})
    return res
