"""When does a run's outcome become predictable from its mechanism metrics? (metrics only)

    python scripts/interp/predictability.py

Population: tiny 1-stage H-Nets with the chunk vocabulary (the word-scale line, EXP-018 / 023 / 024 / 025 / 026 /
027), split into (a) the prior-free AdamW arms, where the seed lottery lives, and (b) all of them. For every logged
step up to 1000 and each metric (hierarchy rms, bypass rms, R, margin, raw-score std, next-chunk CE, main-network
gradient share, train bpb), the rank correlation with the final zero-main contribution and the final val bpb,
across runs and within arms (demeaned by arm), and the AUC for adoption (final contribution >= 0.3 bpb).
Writes data/predictability.json and figures/predictability_vs_step.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIG, dump, enumerate_runs, load_evals, load_metrics  # noqa: E402

METRICS = ["dechunk_rms/s0", "residual_rms/s0", "R", "p_margin/s0", "score_std/s0", "chunk_vocab_ce/s0",
           "main_grad_share", "bpb"]


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    pos, neg = x[y > 0.5], x[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def rank_corr(x, y):
    x, y = pd.Series(x, dtype=float), pd.Series(y, dtype=float)
    ok = x.notna() & y.notna()
    if ok.sum() < 5:
        return None
    return float(np.corrcoef(x[ok].rank(), y[ok].rank())[0, 1])


def main(argv=None):
    runs = enumerate_runs()
    fam = runs[(runs.kind == "hnet") & (runs.scale == "tiny") & (runs.n_stages == 1) & runs.chunk_vocab]
    rows, mats = [], {}
    for _, r in fam.iterrows():
        m = load_metrics(r.dir)
        e = load_evals(r.dir)
        if m.empty or e.empty or "val/zero_main_s0_bpb" not in e or int(e.step.iloc[-1]) < 1700:
            continue
        m = m.set_index("step")
        m["R"] = m["dechunk_rms/s0"] / m["residual_rms/s0"]
        m["main_grad_share"] = m["grad_norm/stage1"] / m["grad_norm/stage0"]
        mats[r.run_id] = m.reindex(range(10, 1010, 10))
        rows.append({"run_id": r.run_id, "exp_id": r.exp_id, "arm": r.arm, "prior": r.prior_w > 0,
                     "optimizer": r.optimizer, "residual_mod": (r.residual_warmup > 0) or (r.residual_gate != "none"),
                     "val_bpb_final": float(e["val/bpb"].iloc[-1]),
                     "contrib_final": float(e["val/zero_main_s0_bpb"].iloc[-1] - e["val/bpb"].iloc[-1])})
    meta = pd.DataFrame(rows)
    meta["adopt"] = (meta.contrib_final >= 0.3).astype(float)
    pops = {"prior_free_adamw": meta[(~meta.prior) & (meta.optimizer == "adamw")],
            "prior_free_adamw_unmodified_residual": meta[(~meta.prior) & (meta.optimizer == "adamw") & (~meta.residual_mod)],
            "all_vocab_1stage": meta}
    out = {"population": meta.to_dict("records"), "curves": {}, "earliest": {}}
    steps = list(range(10, 1010, 10))
    for pname, pm in pops.items():
        cur = {}
        for met in METRICS:
            c = {"step": steps, "rho_contrib": [], "rho_bpb": [], "rho_within_bpb": [], "auc_adopt": []}
            for st in steps:
                x = np.array([mats[i].loc[st, met] if met in mats[i] else np.nan for i in pm.run_id])
                c["rho_contrib"].append(rank_corr(x, pm.contrib_final.values))
                c["rho_bpb"].append(rank_corr(x, pm.val_bpb_final.values))
                c["auc_adopt"].append(auc(x, pm.adopt.values))
                g = pm.assign(x=x).dropna(subset=["x"])
                g = g[g.groupby(["exp_id", "arm"]).x.transform("count") >= 2]
                if len(g) >= 6:
                    dx = g.x - g.groupby(["exp_id", "arm"]).x.transform("mean")
                    dy = g.val_bpb_final - g.groupby(["exp_id", "arm"]).val_bpb_final.transform("mean")
                    c["rho_within_bpb"].append(float(np.corrcoef(dx, dy)[0, 1]) if dx.std() > 0 else None)
                else:
                    c["rho_within_bpb"].append(None)
            cur[met] = c
            # earliest step from which |within-arm r| >= 0.5 holds for every later grid step up to 1000
            w = [abs(v) if v is not None else 0 for v in c["rho_within_bpb"]]
            first = None
            for k in range(len(steps)):
                if all(v >= 0.5 for v in w[k:]):
                    first = steps[k]
                    break
            a = [abs((v or 0.5) - 0.5) * 2 for v in c["auc_adopt"]]
            first_auc = next((steps[k] for k in range(len(steps)) if all(v >= 0.6 for v in a[k:])), None)
            out["earliest"].setdefault(pname, {})[met] = {"within_arm_|r|>=0.5_from": first,
                                                         "auc>=0.8_or<=0.2_from": first_auc,
                                                         "n": int(len(pm))}
        out["curves"][pname] = cur
    dump(out, "predictability.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    pm = "prior_free_adamw"
    for met in METRICS:
        c = out["curves"][pm][met]
        axes[0].plot(steps, [np.nan if v is None else v for v in c["rho_within_bpb"]], label=met, lw=1)
        axes[1].plot(steps, [np.nan if v is None else v for v in c["auc_adopt"]], label=met, lw=1)
        axes[2].plot(steps, [np.nan if v is None else v for v in c["rho_contrib"]], label=met, lw=1)
    n = int(len(pops[pm]))
    axes[0].set_title(f"within-arm r(metric@step, final val bpb), prior-free AdamW vocab runs, n={n}", fontsize=8)
    axes[1].set_title("AUC for adoption (final contribution >= 0.3)", fontsize=8)
    axes[2].set_title("Spearman(metric@step, final contribution)", fontsize=8)
    for ax in axes:
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_xlabel("step")
        ax.set_xscale("log")
    axes[1].axhline(0.5, color="0.7", lw=0.5)
    axes[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "predictability_vs_step.png", dpi=120)
    print(pd.DataFrame(out["earliest"]["prior_free_adamw"]).T)
    print(meta[["run_id", "val_bpb_final", "contrib_final"]].to_string())


if __name__ == "__main__":
    main()
