#!/usr/bin/env python
"""Score evolve runs and append them to the population database (docs/EVOLVE.md).

    python scripts/evolve/score.py --gen 1 --runs <dir with <arm>_s<seed> run dirs> [--bpe 1.695] [--write]

Fitness = mean over finished seeds of (val bpb - BPE val bpb). BPE = --bpe, or the mean of the population's `bpe`
entries. Also records per-seed values, the zero-main probe, and stage-0 mechanism metrics at the end of training
(bypass/hierarchy rms ratio, router margin, undecided fraction, score std) and at step 300.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "logbook/evolve/population.jsonl")


def _jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def load_db():
    return _jsonl(DB)


def bpe_ref(db):
    v = [s for e in db if e["id"] == "bpe" for s in e.get("seeds", {}).values()]
    return statistics.mean(v) if v else None


def mech(metrics, step=None):
    rows = [m for m in metrics if step is None or m["step"] <= step]
    if not rows:
        return {}
    m = rows[-1]
    d, r = m.get("dechunk_rms/s0"), m.get("residual_rms/s0")
    return {k: v for k, v in {
        "step": m["step"], "bypass_ratio": (r / d) if d and r else None, "p_margin": m.get("p_margin/s0"),
        "p_mid": m.get("p_mid/s0"), "score_std": m.get("score_std/s0"), "compression": m.get("compression/s0"),
        "chunk_vocab_ce": m.get("chunk_vocab_ce/s0")}.items() if v is not None}


def score(gen, run_root, bpe):
    arms = defaultdict(dict)
    for d in sorted(glob.glob(os.path.join(run_root, "*_s[0-9]*"))):
        arm, seed = os.path.basename(d).rsplit("_s", 1)
        ev = [e for e in _jsonl(os.path.join(d, "eval.jsonl")) if "val/bpb" in e]
        if not ev or not os.path.exists(os.path.join(d, "done.json")):
            continue
        met = _jsonl(os.path.join(d, "metrics.jsonl"))
        arms[arm][seed] = {"bpb": ev[-1]["val/bpb"], "zero_main": ev[-1].get("val/zero_main_s0_bpb"),
                           "end": mech(met), "step300": mech(met, 300)}
    out = []
    for arm, seeds in arms.items():
        vals = [s["bpb"] for s in seeds.values()]
        e = {"id": arm.replace("_", "-") if arm.startswith("g") else arm, "gen": gen,
             "seeds": {k: round(v["bpb"], 4) for k, v in seeds.items()},
             "mean": round(statistics.mean(vals), 4), "sd": round(statistics.stdev(vals), 4) if len(vals) > 1 else None,
             "fitness": round(statistics.mean(vals) - bpe, 4) if bpe is not None and arm != "bpe" else None,
             "mechanism": {k: v["end"] for k, v in seeds.items()},
             "mechanism_step300": {k: v["step300"] for k, v in seeds.items()},
             "zero_main": {k: v["zero_main"] for k, v in seeds.items()}}
        cand = os.path.join(ROOT, "logbook/evolve/candidates", e["id"], "candidate.yaml")
        if os.path.exists(cand):
            import yaml
            c = yaml.safe_load(open(cand))
            e.update({k: c.get(k) for k in ("parents", "track", "hypothesis", "prediction", "branch")})
        out.append(e)
    return sorted(out, key=lambda e: (e["fitness"] is None, e["fitness"] if e["fitness"] is not None else 0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--bpe", type=float, default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    db = load_db()
    bpe = a.bpe if a.bpe is not None else bpe_ref(db)
    rows = score(a.gen, a.runs, bpe)
    print(f"BPE reference {bpe}")
    print(f"{'id':28s} {'mean':>7s} {'sd':>6s} {'fitness':>8s}  seeds")
    for e in rows:
        print(f"{e['id']:28s} {e['mean']:7.4f} {e['sd'] or 0:6.3f} {e['fitness'] if e['fitness'] is not None else float('nan'):8.4f}  {e['seeds']}")
    if a.write:
        known = {(e["id"], e["gen"]) for e in db}
        with open(DB, "a") as f:
            for e in rows:
                if (e["id"], e["gen"]) not in known:
                    f.write(json.dumps(e) + "\n")


if __name__ == "__main__":
    sys.exit(main())
