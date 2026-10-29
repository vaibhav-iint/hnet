"""EXP-002 results table (run after the sweep; reads the runs pulled by exp002_progress.py).

    python logbook/exp002_summary.py   ->  logbook/tables/exp002_summary.md

Per variant (mean ± half-range over seeds):
  final val bpb and Δ vs BPE · realized bytes per position at each stage · stage-1 chunk statistics on English text
  (bytes/chunk, % cuts after a space, % word-internal cuts, % mid-UTF-8) next to the blog's improved-H-Net targets
  (EXP-001) · final paper7 soft bpb average and acc_norm average.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "runs" / "gcp" / "exp002_tiny2m"
OUT = ROOT / "logbook" / "tables" / "exp002_summary.md"
BLOG_TARGET_S1 = {"bytes_per_chunk": 7.0, "pct_cut_after_space": 91.9, "pct_word_internal": 0.0, "pct_mid_utf8": 0.0}
ORDER = ["bpe", "paper", "end_linear", "end_mlp", "end_cosine_shift", "paper_r42", "paper_nosmooth",
         "paper_nolrmult", "paper_alpha01"]
PAPER7 = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa"]


def read_jsonl(p):
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def last_with(recs, key):
    for r in reversed(recs):
        if key in r:
            return r[key]
    raise KeyError(f"no record with {key!r}")


def fmt(vals, digits=3):
    v = np.array(vals, dtype=float)
    return f"{v.mean():.{digits}f} ± {(v.max() - v.min()) / 2:.{digits}f}" if len(v) > 1 else f"{v[0]:.{digits}f}"


def main():
    per = defaultdict(lambda: defaultdict(list))
    for d in sorted(LOCAL.glob("*_s*")):
        if not (d / "done.json").exists():
            raise SystemExit(f"{d.name} has not finished (no done.json); run after the sweep completes")
        v = d.name.rsplit("_s", 1)[0]
        ev = read_jsonl(d / "eval.jsonl")
        tr = read_jsonl(d / "metrics.jsonl")
        per[v]["val_bpb"].append(last_with(ev, "val/bpb"))
        per[v]["soft_bpb"].append(np.mean([last_with(ev, f"downstream/tasks/{t}/metrics/soft_bpb/value") for t in PAPER7]))
        per[v]["acc_norm"].append(100 * last_with(ev, "downstream/averages/average/acc_norm_char/value"))
        if (d / "diag.json").exists():
            dg = json.loads((d / "diag.json").read_text())
            for k in BLOG_TARGET_S1:
                per[v][f"s1_{k}"].append(dg["s1"][k])
            per[v]["bpp_s0"].append(last_with(tr, "compression/s0"))
            per[v]["bpp_s1"].append(last_with(tr, "compression/s0") * last_with(tr, "compression/s1"))
    base = np.mean(per["bpe"]["val_bpb"])
    rows = ["| variant | seeds | val bpb | Δ vs BPE | bytes/pos s0 | bytes/pos s1 | s1 bytes/chunk | s1 % after space "
            "| s1 % word-internal | s1 % mid-UTF-8 | paper7 soft bpb | paper7 acc_norm |",
            "|---" * 12 + "|"]
    for v in ORDER:
        if v not in per:
            continue
        p = per[v]
        delta = np.mean(p["val_bpb"]) - base
        hn = "s1_bytes_per_chunk" in p
        rows.append(" | ".join([
            f"| {v}", str(len(p["val_bpb"])), fmt(p["val_bpb"]), "—" if v == "bpe" else f"{delta:+.3f}",
            fmt(p["bpp_s0"], 2) if hn else "—", fmt(p["bpp_s1"], 2) if hn else "—",
            fmt(p["s1_bytes_per_chunk"], 2) if hn else "—", fmt(p["s1_pct_cut_after_space"], 1) if hn else "—",
            fmt(p["s1_pct_word_internal"], 1) if hn else "—", fmt(p["s1_pct_mid_utf8"], 2) if hn else "—",
            fmt(p["soft_bpb"]), fmt(p["acc_norm"], 1)]) + " |")
    target = " | ".join(["| blog's improved H-Net (EXP-001, English)", "", "", "", "", "",
                         f"{BLOG_TARGET_S1['bytes_per_chunk']}", f"{BLOG_TARGET_S1['pct_cut_after_space']}",
                         f"{BLOG_TARGET_S1['pct_word_internal']}", f"{BLOG_TARGET_S1['pct_mid_utf8']}", "", ""]) + " |"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(rows + [target]) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
