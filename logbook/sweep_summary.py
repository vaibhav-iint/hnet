"""Summary of finished runs across all fix sweeps: val bpb and the zero-main probes (mean ± sd over seeds).

    python logbook/sweep_summary.py [--all]   (--all includes unfinished runs at their latest eval, marked *)
"""

import argparse
import collections
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEPS = ["exp006_close_gap_tiny", "exp008_hierarchy_use", "exp009_oracle_chunks", "exp010_dynamics_and_mixers",
          "exp011_fixed_hourglass", "exp012_pool_chunker", "exp013_lr_allocation", "exp014_quantile_decision",
          "exp017_chunk_vocab", "exp018_vocab_shapes", "exp019_local_router", "exp020_router_consistency",
          "exp023_word_vocab_trimmed", "exp024_adoption_curriculum"]


def realized_gflops(d: str):
    """Training GFLOPs/byte at the realized compression of the run's last 100 logged steps (None for BPE)."""
    sys.path.insert(0, ROOT)
    import yaml
    from hnet.config import HNetConfig
    from hnet.flops import model_flops

    cfg = yaml.safe_load(open(d + "/config.yaml"))["model"]
    if cfg.get("type", "hnet") != "hnet":
        return None, None
    cfg = HNetConfig.from_dict({k: v for k, v in cfg.items() if k != "type"})
    rs = [json.loads(l) for l in open(d + "/metrics.jsonl") if l.strip()][-10:]
    n = cfg.n_stages
    pos = [st.mean(r[f"positions/s{s}"] for r in rs) for s in range(n + 1)]
    bpp = [pos[0] / p for p in pos]  # bytes per position of each stage
    budget = model_flops(cfg)["flops_per_byte"] / 1e9
    real = model_flops(cfg, bytes_per_position=bpp)["flops_per_byte"] / 1e9
    return real, real / budget - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    rows = collections.defaultdict(list)
    for e in SWEEPS:
        for d in sorted(glob.glob(f"{ROOT}/runs/gcp/{e}/*_s[0-9]")):
            done = os.path.exists(d + "/done.json")
            if not done and not a.all:
                continue
            if not os.path.exists(d + "/eval.jsonl"):
                continue
            ev = [r for r in (json.loads(l) for l in open(d + "/eval.jsonl") if l.strip()) if "val/bpb" in r]
            if not ev:
                continue
            last = ev[-1]
            name = os.path.basename(d).rsplit("_s", 1)[0]
            real, dev = realized_gflops(d)
            rows[(e, name)].append((last["val/bpb"], last.get("val/zero_main_s1_bpb"), last.get("val/zero_main_s0_bpb"),
                                    done, last["step"], dev))
    print(f"{'sweep':28s} {'arm':26s} n   val bpb          innermost-main   below-stage-0   realized FLOPs vs budget")
    for (e, name), v in sorted(rows.items()):
        b = [x[0] for x in v]
        z1 = [x[1] - x[0] for x in v if x[1] is not None]
        z0 = [x[2] - x[0] for x in v if x[2] is not None]
        sd = st.stdev(b) if len(b) > 1 else 0.0
        flag = "" if all(x[3] for x in v) else f" *(min step {min(x[4] for x in v)})"
        line = f"{e:28s} {name:26s} {len(b)}   {st.mean(b):.4f} ± {sd:.4f}"
        if z1:
            line += f"   {st.mean(z1):+.3f}        "
        else:
            line += "                 "
        line += f" {st.mean(z0):+.3f}" if z0 else "       "
        devs = [x[5] for x in v if x[5] is not None]
        if devs:
            m = st.mean(devs)
            line += f"        {m:+.1%}" + ("  <-- OVER BUDGET" if m > 0.02 else "")
        print(line + flag)


if __name__ == "__main__":
    main()
