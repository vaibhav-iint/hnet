"""Score tasks with any scorer and aggregate metrics with bootstrap CIs.

Per multiple-choice item (``MCItem``) we record the summed log-prob of each choice and derive
* ``acc``            argmax of the raw log-prob;
* ``acc_norm_char``  argmax of logprob / len(norm_text) (lm-eval ``acc_norm`` in paper7, lighteval
                     ``LogProbCharNorm`` in smollm3_cf; see each task module);
* ``acc_norm_token`` argmax of logprob / #continuation tokens (lighteval ``LogProbTokenNorm``; token scorers only);
* ``acc_norm_byte``  argmax of logprob / UTF-8 bytes of the continuation (lm-eval ``acc_bytes``);
* ``soft_bpb``       "soft accuracy": bits per byte of the gold continuation, aggregated as
                     sum(nll) / ln 2 / sum(bytes) over items (lower is better).
Ties go to the lowest index (``np.argmax``), gold may list several indices (correct if the prediction is in it).
LAMBADA-style ``LMItem``: ``acc`` = greedy match of the whole continuation, ``perplexity`` = exp(-mean logprob),
``bpb`` of the target word.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np

from hnet.eval import stats as st
from hnet.eval.perturb import MODES, PERTURBATIONS, perturb_mc_item, robustness_score
from hnet.eval.tasks import DERIVED, SUITE_AVERAGES, LMItem, MCItem, Task, get_task, suite_tasks

LN2 = math.log(2)
MC_METRICS = ("acc", "acc_norm_char", "acc_norm_token", "acc_norm_byte")


def score_items(scorer, items: Sequence, progress: bool = False) -> List[dict]:
    """Run all requests of ``items`` in one batched call; return one record per item."""
    reqs, spans = [], []
    for it in items:
        r = it.requests
        spans.append((len(reqs), len(reqs) + len(r)))
        reqs.extend(r)
    res = scorer.loglikelihood_detailed(reqs, progress=progress)
    token_unit = getattr(scorer, "unit", "token") == "token"
    recs = []
    for it, (a, b) in zip(items, spans):
        rr = res[a:b]
        lp = np.array([r.logprob for r in rr])
        nb = np.array([r.n_bytes for r in rr], dtype=float)
        nu = np.array([r.n_units for r in rr], dtype=float)
        if isinstance(it, LMItem):
            r = rr[0]
            recs.append({"doc_id": it.doc_id, "group": it.group, "lp": r.logprob, "greedy": r.greedy,
                         "n_bytes": r.n_bytes, "n_units": r.n_units, "acc": float(r.greedy)})
            continue
        nc = np.array([len(t) for t in it.norm_texts], dtype=float)
        preds = {
            "acc": int(np.argmax(lp)),
            "acc_norm_char": int(np.argmax(lp / np.maximum(nc, 1))),
            "acc_norm_byte": int(np.argmax(lp / np.maximum(nb, 1))),
        }
        if token_unit:
            preds["acc_norm_token"] = int(np.argmax(lp / np.maximum(nu, 1)))
        g = it.gold[0]
        rec = {"doc_id": it.doc_id, "group": it.group, "gold": list(it.gold), "lp": lp.tolist(),
               "n_bytes": nb.astype(int).tolist(), "n_units": nu.astype(int).tolist(), "pred": preds,
               "gold_nll_bits": float(-lp[g] / LN2), "gold_bytes": int(nb[g])}
        for m, p in preds.items():
            rec[m] = float(p in it.gold)
        recs.append(rec)
    return recs


def _stat_fns(task: Task, recs: List[dict]) -> Dict[str, tuple]:
    """metric -> (stat_fn(indices), n, groups) so that tasks can be bootstrapped alone or inside averages."""
    n = len(recs)
    groups = [r["group"] for r in recs] if task.macro_by_group else None
    out = {}
    if task.kind == "lm":
        acc = np.array([r["acc"] for r in recs])
        lp = np.array([r["lp"] for r in recs])
        nbits = -lp / LN2
        nby = np.array([r["n_bytes"] for r in recs], dtype=float)
        out["acc"] = (lambda i, v=acc: float(v[i].mean()), n, None)
        out["perplexity"] = (lambda i, v=lp: float(math.exp(-v[i].mean())), n, None)
        out["bpb"] = (lambda i: float(nbits[i].sum() / nby[i].sum()), n, None)
        return out
    gnb = np.array([r["gold_nll_bits"] for r in recs])
    gby = np.array([r["gold_bytes"] for r in recs], dtype=float)
    for m in MC_METRICS:
        if not recs or m not in recs[0]:
            continue
        v = np.array([r[m] for r in recs])
        if groups is not None:
            g = np.asarray(groups, dtype=object)

            def f(i, v=v, g=g):
                gi = g[i]
                return float(np.mean([v[i][gi == k].mean() for k in dict.fromkeys(gi.tolist())]))

            out[m] = (f, n, groups)
            out[m + "_micro"] = (lambda i, v=v: float(v[i].mean()), n, groups)
        else:
            out[m] = (lambda i, v=v: float(v[i].mean()), n, None)
    out["soft_bpb"] = (lambda i: float(gnb[i].sum() / gby[i].sum()), n, groups)
    return out


def summarize_task(task: Task, recs: List[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    fns = _stat_fns(task, recs)
    metrics = {m: st.bootstrap(f, n, n_boot=n_boot, seed=seed, groups=g) for m, (f, n, g) in fns.items()}
    return {"n": len(recs), "primary": task.primary, "metrics": metrics}


def evaluate_task(scorer, task: Task, limit: Optional[int] = None, n_boot: int = 1000, seed: int = 0,
                  progress: bool = False):
    items = task.items(limit)
    recs = score_items(scorer, items, progress=progress)
    return summarize_task(task, recs, n_boot, seed), recs


def _averages(suite: str, tasks: Dict[str, Task], recs: Dict[str, List[dict]], n_boot: int, seed: int) -> dict:
    """Suite averages and derived tasks (e.g. smollm3 ARC = mean(easy, challenge)), with joint bootstrap CIs."""
    fns = {name: _stat_fns(tasks[name], recs[name]) for name in tasks}
    derived = DERIVED.get(suite, {})
    out = {"derived": {}, "averages": {}}

    def parts(name, metric, w=1.0):
        if name in derived:
            subs = derived[name]
            return [p for s in subs for p in parts(s, metric, w / len(subs))]
        if name not in fns:
            return None
        m = metric if metric in fns[name] else tasks[name].primary
        f, n, g = fns[name][m]
        return [(f, n, g, w)]

    for name, subs in derived.items():
        if all(s in fns for s in subs):
            out["derived"][name] = {}
            for m in set(fns[subs[0]]) & set(fns[subs[1]]):
                out["derived"][name][m] = st.average_ci(parts(name, m), n_boot=n_boot, seed=seed)
    for avg_name, names in SUITE_AVERAGES.get(suite, {}).items():
        if not all(parts(n, "acc") is not None for n in names):
            continue
        metric_sets = {"primary": None, "acc_norm_char": "acc_norm_char", "acc_norm_token": "acc_norm_token",
                       "acc": "acc", "soft_bpb": "soft_bpb"}
        for label, metric in metric_sets.items():
            comps = []
            for n in names:
                if metric is None:
                    t = tasks[n] if n in tasks else tasks[derived[n][0]]
                    comps += parts(n, t.primary)
                else:
                    ps = parts(n, metric)
                    # LAMBADA has no acc_norm/soft_bpb: fall back to its primary metric for accuracy averages
                    if any(metric not in fns[k] for k in _leaves(n, derived)):
                        if metric in ("soft_bpb", "acc_norm_token"):
                            comps = None
                            break
                    comps += ps
            if comps:
                out["averages"][f"{avg_name}/{label}"] = st.average_ci(comps, n_boot=n_boot, seed=seed)
    return out


def _leaves(name, derived):
    return [l for s in derived[name] for l in _leaves(s, derived)] if name in derived else [name]


def evaluate_suite(scorer, suite: str, tasks: Optional[List[str]] = None, limit: Optional[int] = None,
                   n_boot: int = 1000, seed: int = 0, progress: bool = False, task_objs: Optional[List[Task]] = None):
    """-> (summary dict, {task: records}). ``task_objs`` overrides the registry (tests inject rows)."""
    if suite == "robustness":
        return evaluate_robustness(scorer, limit=limit, n_boot=n_boot, seed=seed, progress=progress,
                                   task=(task_objs or [None])[0])
    tl = task_objs if task_objs is not None else suite_tasks(suite, tasks)
    tmap = {t.name: t for t in tl}
    summary = {"suite": suite, "tasks": {}}
    all_recs = {}
    for t in tl:
        s, r = evaluate_task(scorer, t, limit, n_boot, seed, progress)
        summary["tasks"][t.name] = s
        all_recs[t.name] = r
    summary.update(_averages(suite, tmap, all_recs, n_boot, seed))
    return summary, all_recs


def evaluate_robustness(scorer, limit: Optional[int] = None, perturbations=PERTURBATIONS, modes=MODES,
                        n_boot: int = 1000, seed: int = 0, progress: bool = False, task: Optional[Task] = None,
                        metric: str = "acc_norm_char"):
    """HellaSwag (paper7 / lm-eval prompt, acc_norm) under the BLT perturbations, each applied to the prompt, the
    completion and both (BLT protocol: average over the three). Robustness score per H-Net App. D.1, computed
    on the average over perturbations (as in Table 3). CIs resample documents jointly across variants."""
    task = task or get_task("paper7/hellaswag")
    clean_items = task.items(limit)
    variants = {"clean": clean_items}
    for p in perturbations:
        for m in modes:
            variants[f"{p}/{m}"] = [perturb_mc_item(it, p, m, seed) for it in clean_items]
    flat = [it for v in variants.values() for it in v]
    recs = score_items(scorer, flat, progress=progress)
    n = len(clean_items)
    acc = {}
    recs_by = {}
    for k, name in enumerate(variants):
        rr = recs[k * n:(k + 1) * n]
        recs_by[name] = rr
        acc[name] = np.array([r[metric] for r in rr])
    pert_names = [p for p in perturbations]

    def per_pert(i, p):
        return float(np.mean([acc[f"{p}/{m}"][i].mean() for m in modes]))

    def avg(i):
        return float(np.mean([per_pert(i, p) for p in pert_names]))

    def score(i):
        return robustness_score(avg(i), float(acc["clean"][i].mean()))

    B = dict(n_boot=n_boot, seed=seed)
    summary = {"suite": "robustness", "metric": metric, "n": n, "modes": list(modes), "variants": {}, "perturbations": {}}
    for name in variants:
        summary["variants"][name] = st.bootstrap(lambda i, v=acc[name]: float(v[i].mean()), n, **B)
    for p in pert_names:
        summary["perturbations"][p] = st.bootstrap(lambda i, p=p: per_pert(i, p), n, **B)
    summary["clean"] = summary["variants"]["clean"]
    summary["average"] = st.bootstrap(avg, n, **B)
    summary["robustness_score"] = st.bootstrap(score, n, **B)
    return summary, recs_by
