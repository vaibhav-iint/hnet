"""Where does each H-Net lose (or win) against BPE on identical text? Token-aligned comparison.

    python scripts/interp/bpe_compare.py [--docs 48] [--quick]

Wraps ``hnet.analysis.diagnose.bpe_token_bits`` / ``compare_with_bpe`` (GPT-2 tokens; the H-Net's bits summed over
each token's bytes) and adds: the gap by token length, by token kind, and the H-Net's bits per byte at each
position of a "space+word" token (the space, first letter, ...), for the final checkpoints of the turning-point
runs against the tiny BPE baseline at the same LR (EXP-006 bpe_s0, lr 1.5e-3; no checkpoint of the LR-tuned BPE
exists). Writes data/bpe_compare.json and figures/bpe_gap.png.
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import capture  # noqa: E402
from common import FIG, dump  # noqa: E402

from hnet.analysis.diagnose import bpe_token_bits, compare_with_bpe, load_docs  # noqa: E402
from hnet.eval.run import load_run_checkpoint  # noqa: E402

BPE = "runs/gcp/exp006_close_gap_tiny/bpe_s0"
HNETS = {
    "EXP-024 prior s0 (word, warm start)": "runs/gcp/exp024_adoption_curriculum/prior_s0",
    "EXP-023 vp_ng s2 (word)": "runs/gcp/exp023_word_vocab_trimmed/vp_ng_s2",
    "EXP-018 h1_vocab_pred s0 (word)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s0",
    "EXP-018 h1_vocab_pred s1 (suffix)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s1",
    "EXP-023 vp s1 (suffix)": "runs/gcp/exp023_word_vocab_trimmed/vp_s1",
    "EXP-018 h1_quant s0 (no vocab)": "runs/gcp/exp018_vocab_shapes/h1_quant_s0",
    "EXP-006 hnet1 s0 (1-stage, threshold)": "runs/gcp/exp006_close_gap_tiny/hnet1_s0",
    "EXP-002 paper s0 (2-stage)": "runs/gcp/exp002_tiny2m/paper_s0",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=48)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    docs = [d for d in load_docs(a.docs if not a.quick else 8, 2048) if _valid_utf8(d)]
    bmodel, _, _ = load_run_checkpoint(BPE)
    toks = bpe_token_bits(bmodel, docs, "cpu")
    del bmodel
    out = {"bpe_checkpoint": BPE, "docs": len(docs), "models": {}}
    hn = dict(list(HNETS.items())[:2]) if a.quick else HNETS
    for name, path in hn.items():
        model, _, _ = load_run_checkpoint(path)
        per_doc = []
        for d in docs:
            c = capture(model, d, keep_hidden=False)
            per_doc.append(c["bits"][: len(d)].tolist())
        del model
        cmp = compare_with_bpe(per_doc, toks, docs)
        # gap by token length and per-position bits inside "space+word" tokens
        by_len = collections.defaultdict(lambda: [0.0, 0.0, 0])
        pos_bits = collections.defaultdict(lambda: [0.0, 0])
        for hb, tk, d in zip(per_doc, toks, docs):
            for s, n, bits in tk:
                L = min(n, 8)
                by_len[L][0] += sum(hb[s: s + n])
                by_len[L][1] += bits
                by_len[L][2] += n
                tok = d[s: s + n]
                if tok.startswith(b" ") and tok[1:].isalpha():
                    for k in range(n):
                        pos_bits[min(k, 7)][0] += hb[s + k]
                        pos_bits[min(k, 7)][1] += 1
        out["models"][name] = {
            "hnet_bpb_same_text": cmp["ALL"]["hnet_bits_per_byte"], "bpe_bpb_same_text": cmp["ALL"]["bpe_bits_per_byte"],
            "by_kind": cmp,
            "gap_by_token_len": {str(L): {"gap_bits_per_byte": round((h - b) / nb, 4), "share_bytes": nb}
                                 for L, (h, b, nb) in sorted(by_len.items())},
            "space_word_pos_bits": {("space" if k == 0 else f"letter{k}" + ("+" if k == 7 else "")):
                                    round(v / n, 4) for k, (v, n) in sorted(pos_bits.items())},
        }
        tot = sum(v["share_bytes"] for v in out["models"][name]["gap_by_token_len"].values())
        for v in out["models"][name]["gap_by_token_len"].values():
            v["share_bytes"] = round(v["share_bytes"] / tot, 4)
        print(name, cmp["ALL"]["hnet_bits_per_byte"], cmp["ALL"]["bpe_bits_per_byte"])
    dump(out, "bpe_compare.json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = ["space+word", "word_piece", "punct/other", "whitespace/newline", "number"]
    names = list(out["models"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        bk = out["models"][n]["by_kind"]
        vals = [bk.get(k, {}).get("gap_bits_per_byte", np.nan) for k in kinds]
        axes[0].bar(np.arange(len(kinds)) + i * w, vals, w, label=n)
        pb = out["models"][n]["space_word_pos_bits"]
        axes[1].plot(list(pb), list(pb.values()), marker="o", label=n, lw=1)
    axes[0].set_xticks(np.arange(len(kinds)) + 0.4)
    axes[0].set_xticklabels(kinds, fontsize=8)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_ylabel("H-Net - BPE, bits per byte (GPT-2 token kinds)")
    axes[0].legend(fontsize=6)
    axes[1].set_ylabel("H-Net bits per byte inside ' word' tokens")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "bpe_gap.png", dpi=120)


def _valid_utf8(d: bytes) -> bool:
    try:
        d.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


if __name__ == "__main__":
    main()
