"""Figures and a summary table from data/ckpt/*.json (written by analyze_ckpt.py).

    python scripts/interp/figures_ckpt.py

figures/router_hist_by_class.png   standardized router score by class of the byte, word vs suffix seeds over checkpoints
figures/probe_accuracy.png         held-out probe information (bits) from x_enc / x / z, per checkpoint
figures/decoder_path_ablation.png  bits lost when each decoder path is mean-replaced, by class of the predicted byte
data/ckpt_summary.json             one row per checkpoint with the headline numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, FIG, dump  # noqa: E402

HIST_SET = ["exp023_vp_s0__0500", "exp023_vp_s0__1000", "exp018_h1_vocab_pred_s0__final", "exp024_prior_s0__final",
            "exp023_vp_s1__0500", "exp023_vp_s1__1000", "exp023_vp_s1__final", "exp018_h1_vocab_pred_s1__final"]
CLS_SHOW = ["space", "newline", "punct", "word_start", "word_mid", "word_end"]


def load():
    out = {}
    for p in sorted((DATA / "ckpt").glob("*.json")):
        out[p.stem] = json.loads(p.read_text())
    return out


def summary(R: dict) -> list:
    rows = []
    for name, d in R.items():
        pa, rt, tk, pr = d["paths"], d["router"], d["tokenizer"], d["probes"]
        b = pa["bpb"]
        we = rt.get("word_end_prediction", {})
        row = {"ckpt": name, "bpb": d["bpb"], "n_stages": d["n_stages"], "semantics": d["semantics"],
               "eval_bytes_per_chunk": rt["eval_bytes_per_chunk"],
               "gain_hier_zero": round(b["hier_zero"] - b["full"], 4), "gain_hier_mean": round(b["hier_mean"] - b["full"], 4),
               "gain_resid_zero": round(b["resid_zero"] - b["full"], 4), "gain_resid_mean": round(b["resid_mean"] - b["full"], 4),
               "lens_hier": b["lens_hier"], "lens_resid": b["lens_resid"],
               "rms_hier": round(pa["rms_hier"], 3), "rms_resid": round(pa["rms_resid"], 3), "cos": round(pa["cos_hier_resid"], 3),
               "p_margin": round(rt["p_margin"], 3), "ends_on_ws": tk["ends_on_whitespace"], "ws_hit": tk["whitespace_hit"],
               "consistency": tk["word_consistency"], "singletons": tk["singleton_frac"],
               "bnd_at_ws": we.get("boundary_types", {}).get("at_whitespace"),
               "bnd_letter_wordend": we.get("boundary_types", {}).get("letter_then_nonletter"),
               "bnd_intra_word": we.get("boundary_types", {}).get("letter_then_letter(intra_word)"),
               "auc_wordend_at_letters": we.get("auc_z_for_next_nonletter_at_letters"),
               "r2_ngram4": rt["ngram_r2"]["n4"]["r2"], "r2_ngram2": rt["ngram_r2"]["n2"]["r2"],
               "r2_ngram1": rt["ngram_r2"]["n1"]["r2"],
               "top_chunks": " | ".join(repr(c)[1:-1] for c, _ in tk["top_chunks"][:8])}
        for k in ("x_enc", "x", "z"):
            for t in ("identity_top63", "length", "first_byte", "next_first_byte", "prev_last_byte"):
                row[f"{k}:{t}"] = pr.get(k, {}).get(t, {}).get("info_bits")
            row[f"{k}:pr"] = round(pr.get(k, {}).get("eff_rank", {}).get("participation_ratio", float("nan")), 1)
        if "attention" in d:
            mo = [v["mean_offset"] for v in d["attention"].values()]
            row["attn_mean_offset"] = round(float(np.mean(mo)), 1)
            row["attn_off0_1"] = round(float(np.mean([v["off0-0"] + v["off1-1"] for v in d["attention"].values()])), 3)
            row["attn_entropy"] = round(float(np.mean([v["entropy_nats"] for v in d["attention"].values()])), 2)
        for c in ("space", "word_start", "word_mid", "word_end", "punct", "newline"):
            bc = pa["by_class"].get(c)
            if bc:
                row[f"hier_mean_gain:{c}"] = round(bc["hier_mean"] - bc["full"], 3)
                row[f"resid_mean_gain:{c}"] = round(bc["resid_mean"] - bc["full"], 3)
        rows.append(row)
    return rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = load()
    rows = summary(R)
    dump(rows, "ckpt_summary.json")
    # 1. router score histograms
    hs = [n for n in HIST_SET if n in R]
    if hs:
        fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), squeeze=False)
        for ax, n in zip(axes.flat, hs):
            h = R[n]["router"]["z_hist"]
            e = np.array(h["edges"])
            c = 0.5 * (e[1:] + e[:-1])
            for cl in CLS_SHOW:
                if cl in h["counts"]:
                    v = np.array(h["counts"][cl], float)
                    ax.plot(c, v / max(1, v.sum()), label=cl, lw=1.2)
            ax.axvline(0, color="k", lw=0.6)
            tk = R[n]["tokenizer"]
            ax.set_title(f"{n}\nbpb {R[n]['bpb']:.3f}, ends on ws {tk['ends_on_whitespace']:.2f}", fontsize=8)
            ax.set_xlabel("standardized score - threshold (z - z_tau)", fontsize=7)
            ax.tick_params(labelsize=6)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Router score by class of the candidate byte (boundary iff > 0). Top: word-tokenizer seeds; bottom: suffix seeds", fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG / "router_hist_by_class.png", dpi=115)
        plt.close(fig)
    # 2. probe information
    names = [r["ckpt"] for r in rows]
    tg = ["identity_top63", "length", "first_byte", "next_first_byte", "prev_last_byte"]
    fig, axes = plt.subplots(1, len(tg), figsize=(4 * len(tg), max(4, 0.28 * len(names))), squeeze=False)
    for ax, t in zip(axes[0], tg):
        y = np.arange(len(names))
        for k, (src, col) in enumerate((("x_enc", "#999"), ("x", "#1f77b4"), ("z", "#d62728"))):
            v = [r.get(f"{src}:{t}") or 0 for r in rows]
            ax.barh(y + 0.27 * (k - 1), v, 0.27, color=col, label=src)
        ax.set_yticks(y)
        ax.set_yticklabels(names if t == tg[0] else [], fontsize=6)
        ax.set_title(f"{t}: probe info (bits)", fontsize=8)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "probe_accuracy.png", dpi=115)
    plt.close(fig)
    # 3. decoder path ablations by class
    cls = ["space", "word_start", "word_mid", "word_end", "punct", "newline"]
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4, 0.3 * len(names))))
    for ax, key, title in ((axes[0], "hier_mean_gain", "hierarchy mean-replaced"), (axes[1], "resid_mean_gain", "bypass mean-replaced")):
        M = np.array([[r.get(f"{key}:{c}", np.nan) for c in cls] for r in rows], float)
        im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=np.nanpercentile(M, 95))
        ax.set_xticks(range(len(cls)))
        ax.set_xticklabels(cls, fontsize=7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=6)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5, color="w" if M[i, j] < np.nanpercentile(M, 60) else "k")
        ax.set_title(f"bits lost per predicted byte, {title}", fontsize=8)
        plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIG / "decoder_path_ablation.png", dpi=115)
    plt.close(fig)
    print(f"{len(rows)} checkpoints summarized")


if __name__ == "__main__":
    main()
