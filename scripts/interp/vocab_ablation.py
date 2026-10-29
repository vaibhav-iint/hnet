"""How much of the hierarchy's value is the chunk vocabulary table itself? (inference-time ablation)

    python scripts/interp/vocab_ablation.py [--docs 24]

For every chunk-vocabulary checkpoint: bpb with the table rows added to the chunk vectors (as trained), with the
table output zeroed, with it replaced by its mean row, and with the whole hierarchy mean-replaced (for scale); by
class of the predicted byte. Writes data/vocab_ablation.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import CLASSES, capture, decode  # noqa: E402
from common import ROOT, dump  # noqa: E402

from hnet.analysis.diagnose import load_docs  # noqa: E402
from hnet.eval.run import load_run_checkpoint  # noqa: E402

L16 = "runs/gcp/hnet-ladder-1e16/search/ladder/1e16"
CKPTS = {
    "EXP-018 h1_vocab_pred s0 (word)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s0",
    "EXP-018 h1_vocab_pred s1 (suffix)": "runs/gcp/exp018_vocab_shapes/h1_vocab_pred_s1",
    "EXP-023 vp s0 @1000 (word)": "runs/gcp/exp023_word_vocab_trimmed/vp_s0/ckpt/step_0001000",
    "EXP-023 vp s1 final (suffix)": "runs/gcp/exp023_word_vocab_trimmed/vp_s1",
    "EXP-023 vp_ng s2 (word)": "runs/gcp/exp023_word_vocab_trimmed/vp_ng_s2",
    "EXP-024 prior s0 (word)": "runs/gcp/exp024_adoption_curriculum/prior_s0",
    "ladder L03 hnet (prior-free)": f"{L16}/L03/hnet_s0",
    "ladder L03 hnet_ws (warm start)": f"{L16}/L03/hnet_ws_s0",
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=24)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    docs = load_docs(6 if a.quick else a.docs, 2048, seed=2)
    out = {}
    for name, path in (list(CKPTS.items())[:2] if a.quick else CKPTS.items()):
        if not (ROOT / path).exists():
            continue
        model, _, _ = load_run_checkpoint(path)
        cv = model.backbone.chunk_vocab
        orig = cv.embed
        mean_row = (sum(t.weight.mean(0) for t in cv.tables)).detach()
        res = {k: [] for k in ("full", "vocab_zero", "vocab_mean", "hier_mean")}
        cls = []
        for d in docs:
            c = capture(model, d)
            res["full"].append(c["bits"])
            T = len(c["ids"])
            res["hier_mean"].append(decode(model, c, hier=np.repeat(c["out"].mean(0, keepdims=True), T, 0)))
            cls.append(np.roll(c["cls"], -1))
            cv.embed = lambda b, dt: torch.zeros(b.shape[1], cv.tables[0].weight.shape[1], dtype=dt)  # noqa: E731
            res["vocab_zero"].append(capture(model, d, keep_hidden=False)["bits"])
            cv.embed = lambda b, dt: mean_row.to(dt).expand(b.shape[1], -1).clone()  # noqa: E731
            res["vocab_mean"].append(capture(model, d, keep_hidden=False)["bits"])
            cv.embed = orig
        A = {k: np.concatenate(v) for k, v in res.items()}
        C = np.concatenate(cls)
        ok = np.isfinite(A["full"])
        row = {"bpb": {k: round(float(v[ok].mean()), 4) for k, v in A.items()}}
        row["by_class_gain_vocab_mean"] = {cn: round(float((A["vocab_mean"] - A["full"])[ok & (C == i)].mean()), 3)
                                          for i, cn in enumerate(CLASSES) if (ok & (C == i)).sum() > 20}
        out[name] = row
        print(name, row["bpb"], flush=True)
        del model
    dump(out, "vocab_ablation.json")


if __name__ == "__main__":
    main()
