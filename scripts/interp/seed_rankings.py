"""Is the seed lottery noise or a deterministic function of the initialization? (metrics only)

    python scripts/interp/seed_rankings.py

Arms that share a parameter layout get the identical initialization for a given seed (verified in
data/init_basin.json: identical step-0 router statistics), and the data order is seed-independent
(``data.shuffle_buffer: 0``). If the lottery were noise, the seed ranking would be independent across arms of the
same layout. Writes data/seed_rankings.json: per arm the seed order by final val bpb, per layout family the
agreement, and the chance probability of the observed agreement.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import dump, enumerate_runs, load_evals  # noqa: E402

FAMILIES = {
    "no n-gram tables (1-stage vocab + next-chunk)": [("EXP-018", "h1_vocab_pred"), ("EXP-023", "vp"), ("EXP-023", "vp_rwarm")],
    "with n-gram tables (vp_ng layout)": [("EXP-023", "vp_ng"), ("EXP-024", "temp4"), ("EXP-027", "cste"),
                                          ("EXP-027", "cste_rgate"), ("EXP-027", "rgate"), ("EXP-024", "prior")],
}


def main():
    runs = enumerate_runs()
    out = {"families": {}}
    for fam, arms in FAMILIES.items():
        rows = []
        for exp, arm in arms:
            vals = {}
            for _, r in runs[(runs.exp_id == exp) & (runs.arm == arm)].iterrows():
                e = load_evals(r.dir)
                if e.empty or int(e.step.iloc[-1]) < 1700:
                    continue
                c = None
                if "val/zero_main_s0_bpb" in e:
                    c = float(e["val/zero_main_s0_bpb"].iloc[-1] - e["val/bpb"].iloc[-1])
                vals[int(r.seed)] = (float(e["val/bpb"].iloc[-1]), c)
            if len(vals) == 3:
                order = sorted(vals, key=lambda s: vals[s][0])
                rows.append({"arm": f"{exp}/{arm}", "bpb": {s: round(v[0], 4) for s, v in vals.items()},
                             "contrib": {s: None if v[1] is None else round(v[1], 3) for s, v in vals.items()},
                             "order_best_to_worst": order, "worst_seed": order[-1]})
        free = [r for r in rows if "prior" not in r["arm"]]
        worst = [r["worst_seed"] for r in free]
        orders = [tuple(r["order_best_to_worst"]) for r in free]
        k = len(free)
        p_worst = (1 / 3) ** (k - 1) if k and len(set(worst)) == 1 else None
        p_order = (1 / 6) ** (k - 1) if k and len(set(orders)) == 1 else None
        out["families"][fam] = {"arms": rows, "prior_free_arms": k, "same_worst_seed": len(set(worst)) == 1,
                                "same_full_order": len(set(orders)) == 1,
                                "p_same_worst_by_chance": p_worst, "p_same_order_by_chance": p_order}
    dump(out, "seed_rankings.json")
    for fam, v in out["families"].items():
        print(fam, v["prior_free_arms"], v["same_worst_seed"], v["same_full_order"], v["p_same_worst_by_chance"],
              v["p_same_order_by_chance"])
        for r in v["arms"]:
            print("   ", r["arm"], r["order_best_to_worst"], r["bpb"])


if __name__ == "__main__":
    main()
