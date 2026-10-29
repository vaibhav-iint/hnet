"""Is the word/suffix basin visible at initialization? (no training)

    python scripts/interp/init_basin.py [--docs 16]

The training data order is seed-independent (``shuffle_buffer: 0``), so a run's seed only changes its
initialization (and the sampled negatives of the next-chunk loss). For the 1-stage word-scale arms, rebuild each
run's model exactly as the trainer does (``torch.manual_seed(seed)``, build, ``init_weights``), run the router on
validation text, and measure where the top-1/6 of the scores sit at step 0: at whitespace (the word-tokenizer basin)
or at the last letter of a word / inside words (the suffix basin), plus the mean standardized score by byte class.
Compared with each run's measured outcome (final zero-main contribution, val bpb). Writes data/init_basin.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import CLASSES, byte_classes, capture  # noqa: E402
from common import ROOT, dump, enumerate_runs, load_evals  # noqa: E402

from hnet.analysis.diagnose import load_docs  # noqa: E402
from hnet.train.train import build_model_from_section  # noqa: E402

ARMS = [("EXP-018", "h1_vocab_pred"), ("EXP-023", "vp"), ("EXP-023", "vp_rwarm"), ("EXP-023", "vp_ng"),
        ("EXP-024", "temp4"), ("EXP-024", "prior"), ("EXP-027", "cste"), ("EXP-027", "cste_rgate"), ("EXP-027", "rgate"),
        ("EXP-018", "h1_quant")]


def init_stats(cfg: dict, seed: int, docs) -> dict:
    torch.manual_seed(seed)
    model, _ = build_model_from_section(cfg["model"])
    model.init_weights()
    model.eval()
    zs, cls, nxt = [], [], []
    for d in docs:
        c = capture(model, d, keep_hidden=False)
        zs.append(c["logit"][1:])
        ids = c["ids"].astype(np.int64)
        cls.append(c["cls"][1:])
        nxt.append(np.roll(ids, -1)[1:])
    s = np.concatenate(zs)
    cl = np.concatenate(cls)
    z = (s - s.mean()) / (s.std() + 1e-6)
    thr = np.quantile(z, 1 - 1 / 6)
    top = z > thr
    out = {"top_share_by_class": {c: round(float((top & (cl == i)).sum() / top.sum()), 4) for i, c in enumerate(CLASSES)},
           "z_mean_by_class": {c: round(float(z[cl == i].mean()), 3) for i, c in enumerate(CLASSES) if (cl == i).any()},
           "boundary_rate_by_class": {c: round(float(top[cl == i].mean()), 4) for i, c in enumerate(CLASSES) if (cl == i).any()}}
    out["space_minus_wordend_z"] = round(out["z_mean_by_class"].get("space", 0) - out["z_mean_by_class"].get("word_end", 0), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=16)
    a = ap.parse_args(argv)
    torch.set_grad_enabled(False)
    docs = load_docs(a.docs, 2048)
    runs = enumerate_runs()
    rows = []
    for exp, arm in ARMS:
        for _, r in runs[(runs.exp_id == exp) & (runs.arm == arm)].iterrows():
            cfgp = Path(r.dir) / "config.yaml"
            if not cfgp.exists():
                continue
            cfg = yaml.safe_load(open(cfgp))
            seed = int(cfg["train"]["seed"])
            e = load_evals(r.dir)
            st = init_stats(cfg, seed, docs)
            row = {"run_id": r.run_id, "seed": seed, **st}
            if not e.empty:
                row["val_bpb_final"] = float(e["val/bpb"].iloc[-1])
                row["final_step"] = int(e.step.iloc[-1])
                if "val/zero_main_s0_bpb" in e:
                    row["contrib_final"] = float(e["val/zero_main_s0_bpb"].iloc[-1] - e["val/bpb"].iloc[-1])
            rows.append(row)
            print(r.run_id, seed, row.get("val_bpb_final"), row.get("contrib_final"), st["top_share_by_class"]["space"],
                  st["top_share_by_class"]["word_end"], st["space_minus_wordend_z"], flush=True)
    # do runs that share a parameter layout and seed share an initial router? (same seed, same param set -> same init)
    dump(rows, "init_basin.json")


if __name__ == "__main__":
    main()
