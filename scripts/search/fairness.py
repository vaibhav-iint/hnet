"""Apples-to-apples check for a sweep: every run must see the same data and spend the same compute per byte.

    python scripts/search/fairness.py configs/search/exp002_tiny2m.yaml [--flops-tol 0.02] [--bytes-tol 0.02]

For each run (first seed) it composes the exact training config and reports:
  data          dataset + shards + window order (shuffle) + cross-window labels
  bytes/step    UTF-8 text bytes per optimizer step (token runs: tokens/step x measured bytes/token)
  total bytes   bytes over the whole run
  GFLOPs/byte   training FLOPs per byte at target compression (hnet.flops, SPEC convention)
  optimizer     name, peak LR, betas, weight decay, clip, schedule (+ warmup/decay fractions)
  precision / compile / val set / eval cadence / downstream suite

Held-equal fields must be identical across runs (FLOPs/byte and bytes within tolerance). Fields that are allowed to
differ by design (the varied factor, H-Net-only knobs such as LR multipliers) are listed but not enforced.
Exit code 1 if any held-equal field differs. Writes <out_root>/fairness.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep import expand  # noqa: E402

from hnet.data.loader import bytes_per_token  # noqa: E402
from hnet.flops import model_flops  # noqa: E402
from hnet.train.config import load_run_config  # noqa: E402
from hnet.train.train import build_model_from_section  # noqa: E402


def describe(cfg: dict) -> dict:
    model, mcfg = build_model_from_section(cfg["model"])
    d, t, e = cfg["data"], cfg["train"], cfg.get("eval", {})
    is_bytes = hasattr(mcfg, "n_stages")
    bpt = bytes_per_token(d.get("dataset", "finewebedu10B"))
    seq = int(d["seq_len"])
    windows = (int(d["batch_bytes"]) // seq) if is_bytes else (int(d["batch_tokens"]) // seq)
    bytes_step = windows * seq if is_bytes else windows * seq * bpt
    total_bytes = float(t["train_tokens"]) * bpt if t.get("train_tokens") else float(t.get("train_bytes", 0))
    fl = model_flops(mcfg, seq_len=seq) if is_bytes else model_flops(mcfg, seq_len=seq, bytes_per_token=bpt)
    sched = t.get("schedule", {})
    ds = e.get("downstream") or {}
    return {
        "kind": "bytes" if is_bytes else "tokens",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "data": f"{d.get('dataset')} shards={d.get('n_train_shards', 'all')} shuffle={d.get('shuffle_buffer', 0)} "
                f"xwin={d.get('cross_window_label', True)}",
        "bytes_per_step": round(bytes_step),
        "total_bytes_M": round(total_bytes / 1e6, 1),
        "gflops_per_byte": round(fl["flops_per_byte"] / 1e9, 5),
        "optimizer": f"{t.get('optimizer')} lr={t.get('lr')} betas={t.get('betas')} wd={t.get('weight_decay')} "
                     f"clip={t.get('grad_clip')}",
        "schedule": f"{sched.get('name')} warmup={sched.get('warmup')} decay={sched.get('decay')} "
                    f"shape={sched.get('decay_shape')}",
        "precision": f"{t.get('precision')} compile={t.get('compile')}",
        "val": f"{e.get('val')} {e.get('val_bytes')}B every={e.get('every')}",
        "downstream": f"{ds.get('suite')} every={ds.get('every')} limit={ds.get('limit')}" if ds else "none",
        # allowed to differ by design:
        "router": getattr(getattr(mcfg, "dc", None), "router", "-"),
        "dechunker": getattr(getattr(mcfg, "dc", None), "dechunker", "-"),
        "ratios": getattr(getattr(mcfg, "dc", None), "target_ratios", "-"),
        "alpha": getattr(getattr(mcfg, "dc", None), "ratio_loss_alpha", "-"),
        "lr_mult": getattr(mcfg, "lr_multiplier", "-"),
    }


HELD_EXACT = ["data", "optimizer", "schedule", "precision", "val", "downstream"]
HELD_TOL = {"bytes_per_step": "bytes_tol", "total_bytes_M": "bytes_tol", "gflops_per_byte": "flops_tol"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--flops-tol", type=float, default=0.02)
    ap.add_argument("--bytes-tol", type=float, default=0.02)
    a = ap.parse_args(argv)
    spec = yaml.safe_load(open(a.spec))
    rows = {}
    for j in expand(spec):
        name = j["name"].rsplit("_s", 1)[0]
        if name in rows:
            continue
        rows[name] = describe(load_run_config(j["config"], j["sets"]))
    ref_name = next(iter(rows))
    ref = rows[ref_name]
    problems = []
    for name, r in rows.items():
        for k in HELD_EXACT:
            if r[k] != ref[k]:
                problems.append(f"{name}: {k} differs: {r[k]!r} vs {ref_name} {ref[k]!r}")
        for k, tol in HELD_TOL.items():
            rel = abs(r[k] - ref[k]) / max(abs(ref[k]), 1e-12)
            if rel > getattr(a, tol):
                problems.append(f"{name}: {k} {r[k]} vs {ref[k]} ({100 * rel:+.1f}%)")
    cols = ["kind", "params_M", "bytes_per_step", "total_bytes_M", "gflops_per_byte", "router", "dechunker", "ratios",
            "alpha", "lr_mult"]
    lines = [f"# Fairness check: {spec['name']}", "",
             "Held equal across all runs: " + "; ".join(f"**{k}** = `{ref[k]}`" for k in HELD_EXACT), "",
             "| run | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in rows.items():
        lines.append(f"| {name} | " + " | ".join(str(r[c]) for c in cols) + " |")
    lines += ["", "**Problems:** " + ("none" if not problems else ""), *[f"- {p}" for p in problems]]
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(spec["out_root"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness.md").write_text(text)
    (out / "fairness.json").write_text(json.dumps({"rows": rows, "problems": problems}, indent=1))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
