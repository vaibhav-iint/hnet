"""Training-progress analysis over every run with a metrics.jsonl (metrics only, no checkpoints).

    python scripts/interp/phases.py [--quick]

For each H-Net run: derived mechanism series (R = dechunk_rms / residual_rms at stage 0, main-network gradient
share, margin, raw-score std), detected events (bypass opening, hierarchy dip and recovery, margin trough and
recovery, score-std peak, compression collapses, tau saturation, grad-norm and loss spikes), snapshot values at
steps 100 / 250 / 500, and the outcome (final val bpb, zero-main contribution at each eval). Then: correlations of
early quantities with outcomes across runs and within arms (seed lottery), a logistic fit for adoption, the
LR / optimizer dependence of the bypass scale, F/G drift, stage-1 behaviour of 2-stage runs.

Writes logbook/interp/data/phases_runs.json (per run), phases_summary.json (cross-run statistics) and figures
phase_timelines.png, phase_events.png, bypass_scale_vs_lr.png, early_predictors.png.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, dump, enumerate_runs, load_evals, load_metrics  # noqa: E402

SNAP = (100, 250, 500)


def _at(m: pd.DataFrame, col: str, step: int):
    if col not in m or m.empty:
        return None
    i = (m.step - step).abs().idxmin()
    if abs(int(m.step[i]) - step) > 15:
        return None
    v = m[col][i]
    return None if pd.isna(v) else float(v)


def _first(m, mask, after=0):
    s = m.step[mask & (m.step >= after)]
    return int(s.iloc[0]) if len(s) else None


def target_ratio(row, s=0):
    tr = row.get("target_ratios")
    if isinstance(tr, list) and len(tr) > s:
        return float(np.prod(tr[: s + 1])) / (float(np.prod(tr[:s])) if s else 1.0)
    return None


def events(m0: pd.DataFrame, row) -> dict:
    ev = {}
    m = m0
    if "dechunk_rms/s0" in m0 and "residual_rms/s0" in m0 and m0["dechunk_rms/s0"].notna().sum() > 3:
        m = m0.dropna(subset=["dechunk_rms/s0", "residual_rms/s0"]).reset_index(drop=True)
        dr, rr = m["dechunk_rms/s0"], m["residual_rms/s0"]
        R = dr / rr
        early = float(dr[m.step <= 60].max()) if (m.step <= 60).any() else float(dr.iloc[0])
        ev["dechunk_rms_early_max"] = early
        ev["t_bypass_half"] = _first(m, rr >= 0.5 * dr)  # bypass reaches half the hierarchy's rms
        ev["t_bypass_cross"] = _first(m, R < 1.0)  # bypass larger than the hierarchy
        w = m.step <= 600
        if w.any():
            i = dr[w].idxmin()
            ev["t_hier_dip"], ev["hier_dip_rms"] = int(m.step[i]), float(dr[i])
            ev["t_hier_recover"] = _first(m, dr >= 0.9 * early, after=int(m.step[i]))
        ev["R_final"] = float(R.iloc[-1])
        ev["R_min"] = float(R.min())
        ev["residual_rms_final"] = float(rr.iloc[-1])
        ev["dechunk_rms_final"] = float(dr.iloc[-1])
    m = m0
    if "p_margin/s0" in m0 and m0["p_margin/s0"].notna().sum() > 3:
        m = m0.dropna(subset=["p_margin/s0"]).reset_index(drop=True)
        pm = m["p_margin/s0"]
        w = (m.step >= 100) & (m.step <= 800)
        if w.any():
            i = pm[w].idxmin()
            ev["t_margin_trough"], ev["margin_trough"] = int(m.step[i]), float(pm[i])
            ev["t_margin_recover"] = _first(m, pm >= 0.3, after=int(m.step[i]))
        ev["margin_final"] = float(pm.iloc[-1])
        ev["t_margin_first_below_0.2"] = _first(m, pm < 0.2, after=30)
    m = m0
    if "score_std/s0" in m0 and m0["score_std/s0"].notna().sum() > 3:
        m = m0.dropna(subset=["score_std/s0"]).reset_index(drop=True)
        ss = m["score_std/s0"]
        i = ss.idxmax()
        ev["t_score_std_peak"], ev["score_std_peak"] = int(m.step[i]), float(ss[i])
        ev["score_std_final"] = float(ss.iloc[-1])
        ev["t_score_std_collapse"] = _first(m, ss < 0.2, after=50)
    m = m0
    for s in (0, 1):
        c = f"compression/s{s}"
        if c in m:
            tgt = target_ratio(row, s) or float(m[c].median())
            bad = (m[c] > 3 * tgt) | (m[c] < tgt / 3)
            ev[f"t_compression_collapse_s{s}"] = _first(m, bad, after=30)
            ev[f"n_compression_collapse_s{s}"] = int(bad[m.step >= 30].sum())
            ev[f"compression_final_s{s}"] = float(m[c].tail(10).mean())
            ev[f"compression_target_s{s}"] = tgt
        t = f"tau/s{s}"
        if t in m:
            ev[f"t_tau_saturated_s{s}"] = _first(m, m[t] >= 0.999)
    if "grad_norm" in m:
        g = m["grad_norm"]
        med = g.rolling(10, min_periods=5).median().shift(1)
        spike = (g > 5 * med) & (m.step >= 50)
        ev["grad_spikes"] = [int(s) for s in m.step[spike]]
    if "bpb" in m:
        b = m["bpb"]
        runmin = b.cummin().shift(1)
        sp = (b > runmin + 0.25) & (m.step >= 200)
        ev["loss_spikes"] = [int(s) for s in m.step[sp]]
    inner = "grad_norm/stage2" if "grad_norm/stage2" in m else "grad_norm/stage1"
    if inner in m and "grad_norm/stage0" in m:
        share = m[inner] / m["grad_norm/stage0"]
        ev["main_grad_share_10"] = _at(m.assign(_s=share), "_s", 10)
        ev["main_grad_share_50"] = _at(m.assign(_s=share), "_s", 50)
        ev["main_grad_share_final"] = float(share.tail(10).mean())
    if "F/s0" in m and "G/s0" in m:
        ev["FG_final_s0"] = [float(m["F/s0"].tail(10).mean()), float(m["G/s0"].tail(10).mean())]
        ev["G_minus_F_100_s0"] = (_at(m, "G/s0", 100) or 0) - (_at(m, "F/s0", 100) or 0)
        ev["G_minus_F_final_s0"] = float((m["G/s0"] - m["F/s0"]).tail(10).mean())
    if "F/s1" in m and "G/s1" in m:
        ev["FG_final_s1"] = [float(m["F/s1"].tail(10).mean()), float(m["G/s1"].tail(10).mean())]
        if "p_margin/s1" in m:
            ev["margin_final_s1"] = float(m["p_margin/s1"].tail(10).mean())
            ev["p_mid_final_s1"] = float(m["p_mid/s1"].tail(10).mean())
    return ev


def snapshots(m: pd.DataFrame) -> dict:
    out = {}
    m = m.copy()
    if "dechunk_rms/s0" in m:
        m["R"] = m["dechunk_rms/s0"] / m["residual_rms/s0"]
    inner = "grad_norm/stage2" if "grad_norm/stage2" in m else "grad_norm/stage1"
    if inner in m:
        m["main_grad_share"] = m[inner] / m["grad_norm/stage0"]
    for st in SNAP:
        for c in ("bpb", "R", "dechunk_rms/s0", "residual_rms/s0", "p_margin/s0", "p_mid/s0", "score_std/s0",
                  "chunk_vocab_ce/s0", "main_grad_share", "grad_norm", "compression/s0", "compression/s1",
                  "p_margin/s1"):
            v = _at(m, c, st)
            if v is not None:
                out[f"{c}@{st}"] = v
    return out


def outcome(e: pd.DataFrame) -> dict:
    if e.empty:
        return {}
    out = {"val_bpb_final": float(e["val/bpb"].iloc[-1]), "final_eval_step": int(e.step.iloc[-1])}
    if "val/zero_main_s0_bpb" in e:
        zm = e["val/zero_main_s0_bpb"] - e["val/bpb"]
        out["contrib_final"] = None if pd.isna(zm.iloc[-1]) else float(zm.iloc[-1])
        for st in (250, 500, 750, 1000):
            r = e[e.step == st]
            if len(r) and not pd.isna(r["val/zero_main_s0_bpb"].iloc[0]):
                out[f"contrib@{st}"] = float(r["val/zero_main_s0_bpb"].iloc[0] - r["val/bpb"].iloc[0])
    for st in (250, 500):
        r = e[e.step == st]
        if len(r):
            out[f"val_bpb@{st}"] = float(r["val/bpb"].iloc[0])
    return out


# ------------------------------------------------------------------------------------------------- statistics
def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 4:
        return None, len(x)
    rx, ry = pd.Series(x).rank().values, pd.Series(y).rank().values
    return float(np.corrcoef(rx, ry)[0, 1]), int(len(x))


def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return None, int(ok.sum())
    return float(np.corrcoef(x[ok], y[ok])[0, 1]), int(ok.sum())


def logistic_1d(x, y, l2=1e-3, iters=200):
    """Standardized 1-feature logistic regression by Newton steps; returns coef, intercept, leave-one-out acc."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 6 or len(set(y)) < 2:
        return None
    mu, sd = x.mean(), x.std() + 1e-9

    def fit(xs, ys):
        X = np.stack([np.ones_like(xs), (xs - mu) / sd], 1)
        w = np.zeros(2)
        for _ in range(iters):
            p = 1 / (1 + np.exp(-X @ w))
            g = X.T @ (p - ys) + l2 * w
            H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(2)
            w -= np.linalg.solve(H, g)
            w = np.clip(w, -30, 30)
        return w

    w = fit(x, y)
    hits = 0
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        wi = fit(x[mask], y[mask])
        pi = 1 / (1 + np.exp(-(wi[0] + wi[1] * (x[i] - mu) / sd)))
        hits += int((pi > 0.5) == (y[i] > 0.5))
    # best single threshold accuracy (descriptive)
    best = max(max(((x > t) == (y > 0.5)).mean(), ((x <= t) == (y > 0.5)).mean()) for t in np.unique(x))
    return {"coef_std": float(w[1]), "intercept": float(w[0]), "loo_acc": hits / len(x), "n": int(len(x)),
            "base_rate": float(y.mean()), "best_threshold_acc": float(best)}


# ----------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    runs = enumerate_runs()
    if a.quick:
        runs = runs[runs.exp_id.isin(["EXP-023", "EXP-024", "EXP-025", "EXP-027", "LADDER-1e16-L04"])]
    recs, series = [], {}
    for _, row in runs.iterrows():
        m = load_metrics(row.dir)
        if m.empty:
            continue
        e = load_evals(row.dir)
        rec = {k: (v if not isinstance(v, float) or not math.isnan(v) else None) for k, v in row.to_dict().items()}
        rec["steps_logged"] = int(m.step.max())
        rec.update(outcome(e))
        if row.kind == "hnet":
            rec.update(events(m, row))
            rec.update(snapshots(m))
            keep = [c for c in ("step", "bpb", "dechunk_rms/s0", "residual_rms/s0", "p_margin/s0", "score_std/s0",
                                "grad_norm/stage0", "grad_norm/stage1", "grad_norm/stage2", "compression/s0",
                                "compression/s1", "chunk_vocab_ce/s0", "p_margin/s1", "grad_norm") if c in m]
            series[rec["run_id"]] = m[keep]
        recs.append(rec)
    df = pd.DataFrame(recs)
    dump(recs, "phases_runs.json")
    summary = analyse(df)
    dump(summary, "phases_summary.json")
    figures(df, series)
    print(f"{len(df)} runs; summary keys: {list(summary)}")


def adoption_label(df):
    """1 if the final zero-main contribution >= 0.3 bpb (1-stage word-scale arms), NaN where not measured."""
    return np.where(df["contrib_final"].notna(), (df["contrib_final"] >= 0.3).astype(float), np.nan)


def analyse(df: pd.DataFrame) -> dict:
    out = {}
    # outcome statistics only on finished runs (tiny: final eval at >= step 1700; ladder: its own last step)
    fin = (df.scale != "tiny") | (df.final_eval_step.fillna(0) >= 1700)
    out["n_unfinished_excluded"] = int((~fin).sum())
    df = df[fin]
    h = df[(df.kind == "hnet")].copy()
    tiny = h[h.scale == "tiny"]
    out["n_runs"] = {"all": int(len(df)), "hnet": int(len(h)), "hnet_tiny": int(len(tiny)),
                     "with_contrib": int(tiny.contrib_final.notna().sum())}
    # --- phase timing table per experiment (medians)
    cols = ["t_bypass_half", "t_bypass_cross", "t_hier_dip", "hier_dip_rms", "t_hier_recover", "t_margin_trough",
            "margin_trough", "t_margin_recover", "t_score_std_peak", "score_std_peak", "R_final", "margin_final",
            "val_bpb_final", "contrib_final"]
    tab = h.groupby("exp_id")[[c for c in cols if c in h]].median(numeric_only=True).round(3)
    out["phase_table_by_exp"] = tab.reset_index().to_dict("records")
    # --- early predictors
    feats = [c for c in df.columns if "@" in c and not c.startswith(("val_bpb@", "contrib@"))]
    feats += ["contrib@250", "contrib@500", "val_bpb@250", "val_bpb@500", "t_margin_trough", "margin_trough",
              "score_std_peak", "t_hier_dip", "hier_dip_rms", "t_bypass_cross"]
    pred = {}
    pool = tiny[tiny.n_stages == 1].copy()
    pool["adopt"] = adoption_label(pool)
    # within-arm: demean by arm (seed lottery)
    for target in ("val_bpb_final", "contrib_final"):
        res = {}
        for f in feats:
            if f not in pool:
                continue
            r_all = spearman(pool[f], pool[target])
            g = pool[[f, target, "exp_id", "arm"]].dropna()
            g = g[g.groupby(["exp_id", "arm"])[f].transform("count") >= 2]
            if len(g) >= 6:
                dx = g[f] - g.groupby(["exp_id", "arm"])[f].transform("mean")
                dy = g[target] - g.groupby(["exp_id", "arm"])[target].transform("mean")
                r_w = pearson(dx, dy)
            else:
                r_w = (None, len(g))
            res[f] = {"spearman_all": r_all[0], "n_all": r_all[1], "pearson_within_arm": r_w[0], "n_within": r_w[1]}
        pred[target] = res
    out["predictors_1stage_tiny"] = pred
    logit = {}
    for f in feats:
        if f in pool:
            r = logistic_1d(pool[f], pool["adopt"])
            if r:
                logit[f] = r
    out["adoption_logistic_1stage_tiny"] = dict(sorted(logit.items(), key=lambda kv: -kv[1]["loo_acc"]))
    out["adoption_counts"] = {"adopted": int(np.nansum(pool.adopt)), "n": int(np.isfinite(pool.adopt).sum())}
    # --- bypass scale: optimizer and LR dependence, and does it track the hierarchy's contribution?
    hb = h[h.R_final.notna()].copy()
    hb["eff_lr"] = np.where(hb.optimizer == "muon", hb.muon_lr, hb.lr)
    byp = []
    for (opt, sc), g in hb.groupby(["optimizer", "scale"]):
        for lr, gg in g.groupby("eff_lr"):
            byp.append({"optimizer": opt, "scale": sc, "lr": float(lr), "n": int(len(gg)),
                        "residual_rms_final_median": float(gg.residual_rms_final.median()),
                        "dechunk_rms_final_median": float(gg.dechunk_rms_final.median()),
                        "R_final_median": float(gg.R_final.median()),
                        "contrib_final_median": None if gg.contrib_final.isna().all() else float(gg.contrib_final.median())})
    out["bypass_scale_by_optimizer_lr"] = byp
    hc = hb[hb.contrib_final.notna()]
    out["R_final_vs_contrib"] = {
        "spearman_all": spearman(hc.R_final, hc.contrib_final),
        "spearman_adamw": spearman(hc[hc.optimizer == "adamw"].R_final, hc[hc.optimizer == "adamw"].contrib_final),
        "spearman_muon": spearman(hc[hc.optimizer == "muon"].R_final, hc[hc.optimizer == "muon"].contrib_final),
        "log_residual_vs_log_lr_adamw_tiny": pearson(np.log(hb[(hb.optimizer == "adamw")].eff_lr),
                                                     np.log(hb[(hb.optimizer == "adamw")].residual_rms_final)),
        "log_residual_vs_log_lr_muon": pearson(np.log(hb[(hb.optimizer == "muon")].eff_lr),
                                               np.log(hb[(hb.optimizer == "muon")].residual_rms_final)),
    }
    # --- 2-stage: stage-1 behaviour
    two = h[h.n_stages == 2]
    s1 = two[[c for c in ("run_id", "margin_final_s1", "p_mid_final_s1", "FG_final_s1", "n_compression_collapse_s1",
                          "compression_final_s1", "compression_target_s1", "lr", "decision", "router",
                          "val_bpb_final") if c in two]]
    out["stage1_2stage"] = s1.to_dict("records")
    # --- collapses / spikes
    coll = h[[c for c in ("run_id", "t_compression_collapse_s0", "t_compression_collapse_s1", "t_tau_saturated_s0",
                          "t_tau_saturated_s1", "t_score_std_collapse", "grad_spikes", "loss_spikes", "val_bpb_final",
                          "contrib_final") if c in h]]
    coll = coll[coll.drop(columns=["run_id", "val_bpb_final", "contrib_final"]).apply(
        lambda r: any((isinstance(v, list) and len(v) > 0) or (not isinstance(v, list) and pd.notna(v)) for v in r), axis=1)]
    out["collapses_and_spikes"] = coll.to_dict("records")
    # grad spike preceding loss spike / compression collapse
    lead = []
    for _, r in h.iterrows():
        gs = r.get("grad_spikes") or []
        ls = r.get("loss_spikes") or []
        cc = [r.get("t_compression_collapse_s0"), r.get("t_compression_collapse_s1")]
        ev = [x for x in ls + [c for c in cc if c is not None and not pd.isna(c)]]
        if not ev:
            continue
        t_ev = min(ev)
        before = [g for g in gs if t_ev - 100 <= g <= t_ev]
        lead.append({"run_id": r.run_id, "t_event": int(t_ev), "grad_spike_within_100_before": bool(before),
                     "first_grad_spike": min(gs) if gs else None})
    out["grad_spike_precedes_event"] = {
        "n_events": len(lead), "n_preceded": int(sum(x["grad_spike_within_100_before"] for x in lead)), "rows": lead}
    # --- F/G drift
    fg = h[h.G_minus_F_final_s0.notna()] if "G_minus_F_final_s0" in h else h.iloc[:0]
    out["FG_drift_s0"] = [{"group": k, "n": int(len(g)), "G_minus_F_final_median": float(g.G_minus_F_final_s0.median()),
                           "G_minus_F_100_median": float(g.G_minus_F_100_s0.median())}
                          for k, g in fg.groupby(fg.decision.fillna("threshold"))]
    # --- Muon vs AdamW score scale (same arm family: EXP-024 prior vs EXP-025 prior_muon)
    sc = h[h.exp_id.isin(["EXP-024", "EXP-025", "LADDER-LR-1e16", "LADDER-LREXT-1e16"]) | h.exp_id.str.startswith("LADDER-1e16")]
    out["score_scale_by_optimizer"] = [{"exp_arm": f"{r.exp_id}/{r.arm}", "seed": r.seed, "optimizer": r.optimizer,
                                        "lr": r.muon_lr if r.optimizer == "muon" else r.lr,
                                        "score_std_peak": r.get("score_std_peak"), "score_std_final": r.get("score_std_final"),
                                        "margin_final": r.get("margin_final"), "R_final": r.get("R_final"),
                                        "val_bpb_final": r.get("val_bpb_final"), "contrib_final": r.get("contrib_final")}
                                       for _, r in sc.iterrows()]
    return out


def figures(df: pd.DataFrame, series: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1. phase timelines for the narrative's turning-point runs
    groups = {
        "EXP-023 vp (word s0/s2, suffix s1)": ["EXP-023/vp_s0", "EXP-023/vp_s1", "EXP-023/vp_s2"],
        "EXP-023 vp_ng (s0 not adopted)": ["EXP-023/vp_ng_s0", "EXP-023/vp_ng_s1", "EXP-023/vp_ng_s2"],
        "EXP-024 prior": ["EXP-024/prior_s0", "EXP-024/prior_s1", "EXP-024/prior_s2"],
        "EXP-025 prior_muon 0.025 / 0.05": ["EXP-025/prior_muon__mlr0.025_s0", "EXP-025/prior_muon__mlr0.05_s0",
                                             "EXP-025/prior_muon__mlr0.05_s1"],
        "EXP-027 cste s0 vs s2": ["EXP-027/cste_s0", "EXP-027/cste_s2", "EXP-027/rgate_s0"],
        "Ladder L04 1e16": ["LADDER-1e16-L04/hnet_s0", "LADDER-1e16-L04/hnet_ws_s0", "LADDER-LREXT-1e16/hnet__mlr0.0025_s0"],
    }
    panels = [("dechunk_rms/s0", "hierarchy rms"), ("residual_rms/s0", "bypass rms"), ("R", "R = hier/bypass"),
              ("p_margin/s0", "router margin"), ("score_std/s0", "raw score std"), ("main_grad_share", "main grad / stage-0 grad")]
    fig, axes = plt.subplots(len(groups), len(panels), figsize=(3.1 * len(panels), 2.3 * len(groups)), squeeze=False)
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    for gi, (gname, ids) in enumerate(groups.items()):
        for pi, (col, title) in enumerate(panels):
            ax = axes[gi, pi]
            for k, rid in enumerate(ids):
                if rid not in series:
                    continue
                m = series[rid].copy()
                if "dechunk_rms/s0" in m:
                    m["R"] = m["dechunk_rms/s0"] / m["residual_rms/s0"]
                inner = "grad_norm/stage2" if "grad_norm/stage2" in m else "grad_norm/stage1"
                if inner in m:
                    m["main_grad_share"] = m[inner] / m["grad_norm/stage0"]
                if col in m:
                    c = df.loc[df.run_id == rid, "contrib_final"]
                    lab = rid.split("/")[1] + (f" (c={c.iloc[0]:.2f})" if len(c) and pd.notna(c.iloc[0]) else "")
                    ax.plot(m.step, m[col], color=colors[k % 3], lw=1, label=lab)
            ax.set_xscale("log")
            if col in ("R", "residual_rms/s0", "score_std/s0", "main_grad_share"):
                ax.set_yscale("log")
            if gi == 0:
                ax.set_title(title, fontsize=9)
            if pi == 0:
                ax.set_ylabel(gname, fontsize=7)
                ax.legend(fontsize=5.5, loc="best")
            ax.tick_params(labelsize=6)
            for x in (60, 130, 500):
                ax.axvline(x, color="0.85", lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(FIG / "phase_timelines.png", dpi=110)
    plt.close(fig)

    # 2. event timing across runs (1-stage tiny with contributions)
    h = df[(df.kind == "hnet") & df.contrib_final.notna()].copy().sort_values("contrib_final")
    evs = [("t_bypass_cross", "bypass > hierarchy", "o"), ("t_hier_dip", "hierarchy rms dip", "v"),
           ("t_margin_trough", "margin trough", "s"), ("t_score_std_peak", "score-std peak", "^"),
           ("t_margin_recover", "margin recovers >= 0.3", "*")]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.11 * len(h))))
    for j, (c, lab, mk) in enumerate(evs):
        if c in h:
            ax.scatter(h[c], np.arange(len(h)), marker=mk, s=12, label=lab, alpha=0.8)
    ax.set_yticks(np.arange(len(h)))
    ax.set_yticklabels([f"{r} ({c:.2f})" for r, c in zip(h.run_id, h.contrib_final)], fontsize=4.5)
    ax.set_xscale("log")
    ax.set_xlabel("step (runs sorted by final zero-main contribution, bpb)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "phase_events.png", dpi=120)
    plt.close(fig)

    # 3. bypass scale vs LR, coloured by optimizer; and R_final vs contribution
    hb = df[(df.kind == "hnet") & df.R_final.notna()].copy()
    hb["eff_lr"] = np.where(hb.optimizer == "muon", hb.muon_lr, hb.lr)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for (opt, sc), g in hb.groupby(["optimizer", "scale"]):
        axes[0].scatter(g.eff_lr, g.residual_rms_final, s=12, label=f"{opt} {sc}", alpha=0.7)
        axes[1].scatter(g.eff_lr, g.dechunk_rms_final, s=12, label=f"{opt} {sc}", alpha=0.7)
        gc = g[g.contrib_final.notna()]
        axes[2].scatter(gc.R_final, gc.contrib_final, s=12, label=f"{opt} {sc}", alpha=0.7)
    for ax, t in zip(axes, ("bypass (residual) rms at end vs LR", "hierarchy (dechunk) rms at end vs LR",
                            "R_final vs hierarchy contribution (zero-main)")):
        ax.set_title(t, fontsize=9)
        ax.legend(fontsize=7)
    axes[0].set_xscale("log"); axes[0].set_yscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("peak LR (AdamW lr or Muon lr)"); axes[1].set_xlabel("peak LR")
    axes[2].set_xscale("log"); axes[2].set_xlabel("R_final = dechunk_rms / residual_rms"); axes[2].set_ylabel("bpb")
    fig.tight_layout()
    fig.savefig(FIG / "bypass_scale_vs_lr.png", dpi=120)
    plt.close(fig)

    # 4. early predictors (1-stage tiny): margin@250, score_std@250, R@250, contrib@250 vs final contrib
    p = df[(df.kind == "hnet") & (df.scale == "tiny") & (df.n_stages == 1) & df.contrib_final.notna()]
    fs = ["p_margin/s0@250", "score_std/s0@250", "R@250", "contrib@250", "chunk_vocab_ce/s0@500", "p_margin/s0@500"]
    fig, axes = plt.subplots(1, len(fs), figsize=(3 * len(fs), 3))
    for ax, f in zip(axes, fs):
        if f in p:
            sc = ax.scatter(p[f], p.contrib_final, c=p.prior_w.fillna(0) > 0, cmap="coolwarm", s=12)
            r, n = spearman(p[f], p.contrib_final)
            ax.set_title(f"{f}\nrho={r:.2f} n={n}" if r is not None else f, fontsize=8)
        ax.set_ylabel("final contribution (bpb)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "early_predictors.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
