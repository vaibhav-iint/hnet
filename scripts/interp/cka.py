"""Cross-checkpoint similarity of byte-level representations (linear CKA) on the same validation documents.

    python scripts/interp/cka.py [--docs 24] [--quick]

For the 1-stage word-scale checkpoints (word-tokenizer and suffix-tokenizer seeds, the warm-started run, the
no-vocabulary baseline), at every byte: the encoder output (h_enc), the bypass (residual), the dechunked hierarchy
signal (out) and the decoder output (h_dec). Also the agreement of the two routers' boundary sets (Jaccard) and
the correlation of per-byte losses. Writes data/cka.json and figures/cka.png.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import capture  # noqa: E402
from common import FIG, dump  # noqa: E402

from hnet.analysis.diagnose import load_docs  # noqa: E402
from hnet.eval.run import load_run_checkpoint  # noqa: E402

CKPTS = {
    "W18 h1_vocab_pred s0 (word)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s0",
    "S18 h1_vocab_pred s1 (suffix)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s1",
    "W23 vp s0 @1000 (word)": "runs/gcp/exp023_word_vocab_trimmed/vp_s0/ckpt/step_0001000",
    "S23 vp s1 @1000 (suffix)": "runs/gcp/exp023_word_vocab_trimmed/vp_s1/ckpt/step_0001000",
    "S23 vp s1 final (suffix)": "runs/gcp/exp023_word_vocab_trimmed/vp_s1",
    "W23ng vp_ng s2 (word)": "runs/gcp/exp023_word_vocab_trimmed/vp_ng_s2",
    "P24 prior s0 (word, warm start)": "runs/gcp/exp024_adoption_curriculum/prior_s0",
    "Q18 h1_quant s0 (no vocab)": "runs/gcp/exp018_vocab_shapes/h1_quant_s0",
}
REPS = ("h_enc", "residual", "out", "h_dec")


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    xy = np.linalg.norm(Y.T @ X) ** 2
    return float(xy / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    ck = dict(list(CKPTS.items())[:4]) if a.quick else CKPTS
    docs = load_docs(8 if a.quick else a.docs, 2048, seed=1)
    reps, masks, bits = {}, {}, {}
    for name, path in ck.items():
        model, _, _ = load_run_checkpoint(path)
        caps = [capture(model, d) for d in docs]
        reps[name] = {r: np.concatenate([c[r] for c in caps])[1::2] for r in REPS}
        masks[name] = np.concatenate([c["mask"] for c in caps])
        bits[name] = np.concatenate([c["bits"] for c in caps])
        del model
    names = list(ck)
    out = {"names": names, "cka": {r: [[None] * len(names) for _ in names] for r in REPS},
           "boundary_jaccard": [[None] * len(names) for _ in names],
           "bits_corr": [[None] * len(names) for _ in names]}
    for i, j in itertools.product(range(len(names)), repeat=2):
        if j < i:
            continue
        for r in REPS:
            v = linear_cka(reps[names[i]][r], reps[names[j]][r])
            out["cka"][r][i][j] = out["cka"][r][j][i] = round(v, 4)
        mi, mj = masks[names[i]], masks[names[j]]
        jac = float((mi & mj).sum() / max(1, (mi | mj).sum()))
        out["boundary_jaccard"][i][j] = out["boundary_jaccard"][j][i] = round(jac, 4)
        bi, bj = bits[names[i]], bits[names[j]]
        ok = np.isfinite(bi) & np.isfinite(bj)
        out["bits_corr"][i][j] = out["bits_corr"][j][i] = round(float(np.corrcoef(bi[ok], bj[ok])[0, 1]), 4)
    dump(out, "cka.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mats = [(f"CKA {r}", np.array(out["cka"][r], float)) for r in REPS] + [
        ("boundary Jaccard", np.array(out["boundary_jaccard"], float))]
    fig, axes = plt.subplots(1, len(mats), figsize=(4.2 * len(mats), 4.6))
    short = [n.split(" ")[0] for n in names]
    for ax, (t, M) in zip(axes, mats):
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(short, rotation=90, fontsize=7)
        ax.set_yticklabels(short, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=5.5,
                        color="w" if M[i, j] < 0.6 else "k")
        ax.set_title(t, fontsize=9)
    fig.text(0.01, 0.01, " | ".join(names), fontsize=5.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "cka.png", dpi=120)
    print({r: np.round(np.array(out["cka"][r], float), 2).tolist() for r in REPS})


if __name__ == "__main__":
    main()
