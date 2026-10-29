"""Dense early-training checkpoints (runs/gcp/probe/tiny_hnet2_dyn, paper 2-stage router, steps 1-400): when does
the decoder *functionally* switch from the hierarchy to the bypass? Ablation-based, not rms-based.

    python scripts/interp/early_dynamics.py [--docs 12] [--quick]

Per checkpoint: bpb, bits with the hierarchy zeroed / mean-replaced, with the bypass zeroed / mean-replaced, the
two paths' rms and cosine, stage-0 / stage-1 router p statistics and eval compression. Writes
data/early_dynamics.json and figures/early_dynamics.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import capture, decode  # noqa: E402
from common import FIG, ROOT, dump  # noqa: E402

from hnet.analysis.diagnose import load_docs  # noqa: E402
from hnet.eval.run import load_run_checkpoint  # noqa: E402

RUN = ROOT / "runs/gcp/probe/tiny_hnet2_dyn"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    docs = load_docs(4 if a.quick else a.docs, 2048)
    cks = sorted((RUN / "ckpt").glob("step_*"))
    if a.quick:
        cks = cks[::4]
    rows = []
    for ck in cks:
        if not (ck / "model.pt").exists():
            continue
        model, _, _ = load_run_checkpoint(str(ck))
        acc = {k: [] for k in ("full", "hier_zero", "hier_mean", "resid_zero", "resid_mean")}
        rms_h, rms_r, cos, p0, p1, m0, m1 = [], [], [], [], [], [], []
        for d in docs:
            c = capture(model, d)
            T = len(c["ids"])
            acc["full"].append(c["bits"])
            acc["hier_zero"].append(decode(model, c, hier=np.zeros_like(c["out"])))
            acc["hier_mean"].append(decode(model, c, hier=np.repeat(c["out"].mean(0, keepdims=True), T, 0)))
            acc["resid_zero"].append(decode(model, c, resid=np.zeros_like(c["residual"])))
            acc["resid_mean"].append(decode(model, c, resid=np.repeat(c["residual"].mean(0, keepdims=True), T, 0)))
            rms_h.append(c["out_rms"].mean())
            rms_r.append(c["res_rms"].mean())
            cos.append(c["cos_out_res"].mean())
            p0.append(c["p"][1:])
            m0.append(c["mask"][1:])
            if "inner_p" in c:
                p1.append(c["inner_p"][1:])
                m1.append(c["inner_mask"][1:])
        step = int(ck.name.split("_")[1])
        b = {k: float(np.nanmean(np.concatenate(v))) for k, v in acc.items()}
        P0, M0 = np.concatenate(p0), np.concatenate(m0)
        row = {"step": step, **{f"bpb_{k}": round(v, 4) for k, v in b.items()},
               "gain_hier_zero": round(b["hier_zero"] - b["full"], 4), "gain_hier_mean": round(b["hier_mean"] - b["full"], 4),
               "gain_resid_zero": round(b["resid_zero"] - b["full"], 4), "gain_resid_mean": round(b["resid_mean"] - b["full"], 4),
               "rms_hier": float(np.mean(rms_h)), "rms_resid": float(np.mean(rms_r)), "cos": float(np.mean(cos)),
               "s0_bytes_per_chunk": float(1 / max(1e-6, M0.mean())), "s0_p_mid": float(((P0 > 0.4) & (P0 < 0.6)).mean()),
               "s0_margin": float(np.abs(P0 - 0.5).mean())}
        if p1:
            P1, M1 = np.concatenate(p1), np.concatenate(m1)
            row.update({"s1_chunks_per_s0": float(1 / max(1e-6, M1.mean())),
                        "s1_p_mid": float(((P1 > 0.4) & (P1 < 0.6)).mean()), "s1_margin": float(np.abs(P1 - 0.5).mean())})
        rows.append(row)
        print(row["step"], row["bpb_full"], row["gain_hier_mean"], row["gain_resid_mean"], round(row["rms_hier"], 3),
              round(row["rms_resid"], 3))
        del model
    dump(rows, "early_dynamics.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for k, lab in (("gain_hier_zero", "hierarchy zeroed"), ("gain_hier_mean", "hierarchy mean-replaced"),
                   ("gain_resid_zero", "bypass zeroed"), ("gain_resid_mean", "bypass mean-replaced")):
        axes[0].plot(s, [r[k] for r in rows], marker=".", label=lab)
    axes[0].set_ylabel("bits/byte lost by the ablation")
    axes[0].set_yscale("symlog", linthresh=0.01)
    axes[0].legend(fontsize=7)
    axes[1].plot(s, [r["rms_hier"] for r in rows], label="hierarchy rms")
    axes[1].plot(s, [r["rms_resid"] for r in rows], label="bypass rms")
    axes[1].plot(s, [r["cos"] for r in rows], label="cos(hier, bypass)")
    axes[1].legend(fontsize=7)
    axes[2].plot(s, [r["s0_bytes_per_chunk"] for r in rows], label="stage-0 bytes/chunk")
    if "s1_chunks_per_s0" in rows[-1]:
        axes[2].plot(s, [r.get("s1_chunks_per_s0") for r in rows], label="stage-1 chunks per s0 chunk")
        axes[2].plot(s, [r.get("s1_p_mid") for r in rows], label="stage-1 p in (0.4,0.6)")
    axes[2].plot(s, [r["s0_p_mid"] for r in rows], label="stage-0 p in (0.4,0.6)")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("step")
    fig.suptitle("Dense early checkpoints, paper 2-stage H-Net (probe/tiny_hnet2_dyn): ablation-measured path use", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "early_dynamics.png", dpi=120)


if __name__ == "__main__":
    main()
