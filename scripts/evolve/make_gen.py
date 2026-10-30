#!/usr/bin/env python
"""Build the sweep spec of one evolve generation (docs/EVOLVE.md) from its candidates.

    python scripts/evolve/make_gen.py --gen 1 logbook/evolve/candidates/g1-*/candidate.yaml [--controls bpe,hnet_ws]
        [--seeds 0,1] [--out configs/evolve/gen1.yaml]

The spec is the benchmark spec (configs/search/ladder_1e16_L03.yaml) with its runs replaced by one run per candidate
(the hnet arm's config + the candidate's model file and overrides) plus optional control arms copied verbatim.
A candidate with ``base: <arm>`` (e.g. ``base: bpe`` for a BPE-side control such as g2-bpe-match) is built from that
benchmark arm instead (its config, model, batch and train-length overrides); the default base is ``hnet``.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BENCH = os.path.join(ROOT, "configs/search/ladder_1e16_L03.yaml")


def build(gen: int, cand_paths, controls, seeds):
    bench = yaml.safe_load(open(BENCH))
    by_name = {r["name"]: r for r in bench["runs"]}
    spec = {k: v for k, v in bench.items() if k not in ("runs", "report")}
    name = f"evolve_g{gen}"
    spec.update(name=name, out_root=f"runs/search/{name}", gcs=f"gs://neat-dispatch-501718-a5-hnet/runs/{name}",
                seeds=list(seeds))
    spec["params"] = dict(spec.get("params") or {}, best_mlr={"bpe": 0.02})
    runs, labels = [], {}
    for c in controls:
        runs.append(copy.deepcopy(by_name[c]))
        labels[c] = f"control: {c}"
    for p in cand_paths:
        c = yaml.safe_load(open(p))
        arm = c.get("base", "hnet")
        assert arm in by_name, f"{p}: base {arm!r} is not an arm of the benchmark ({sorted(by_name)})"
        r = copy.deepcopy(by_name[arm])
        r["name"] = c["id"].replace("-", "_")
        if c.get("model"):
            r["model"] = c["model"]
        r["set"] = list(r.get("set") or []) + list(c.get("set") or [])
        runs.append(r)
        labels[r["name"]] = c.get("hypothesis", "").split(".")[0][:120]
    spec["runs"] = runs
    spec["report"] = {"baseline": "bpe" if "bpe" in controls else None, "labels": labels}
    return spec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("candidates", nargs="*")
    ap.add_argument("--controls", default="")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    controls = [c for c in a.controls.split(",") if c]
    spec = build(a.gen, a.candidates, controls, [int(s) for s in a.seeds.split(",")])
    out = a.out or os.path.join(ROOT, f"configs/evolve/gen{a.gen}.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# evolve generation {a.gen} (scripts/evolve/make_gen.py; benchmark: {os.path.relpath(BENCH, ROOT)})\n")
        yaml.safe_dump(spec, f, sort_keys=False, width=120)
    print(out, f"{len(spec['runs'])} runs x {len(spec['seeds'])} seeds")


if __name__ == "__main__":
    sys.exit(main())
