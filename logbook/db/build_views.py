#!/usr/bin/env python3
"""Denormalized markdown views of the experiment database in logbook/db/*.csv (csv module only, no pandas).

    python logbook/db/build_views.py            # all three views
    python logbook/db/build_views.py big        # (a) one row per arm, settings pivoted, grouped by experiment category
    python logbook/db/build_views.py runs       # (b) runs per experiment
    python logbook/db/build_views.py findings   # (c) findings by category
    python logbook/db/build_views.py counts     # row counts per table
    python logbook/db/build_views.py verify     # re-read eval.jsonl for every run_dir in runs.csv and report drift

The schema (keys, foreign keys) is described in README.md next to this file.
"""
import csv
import glob
import json
import os
import sys
from collections import OrderedDict, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NULL = "NULL"
TABLES = ["experiments", "arms", "arm_settings", "runs", "arm_results", "references", "findings", "components"]
CATEGORY_ORDER = ["baselines & calibration", "router / decision rule", "hierarchy use & bypass", "chunk vocabulary & memory",
                  "curricula & priors", "optimizer & mixers", "ladder / scaling"]
# settings pivoted into columns of the big table (setting name in arm_settings.csv -> column header)
PIVOT = OrderedDict([("router", "router"), ("decision", "decision"), ("chunk_vocab", "chunk_vocab"), ("ngram_embed", "ngram"),
                     ("router_prior_weight", "prior w"), ("router_prior_frac", "prior frac"), ("optimizer", "optimizer"),
                     ("lr", "lr"), ("muon_lr", "muon lr"), ("residual_init", "resid init"), ("residual_warmup", "resid warmup"),
                     ("residual_gate", "resid gate"), ("residual_dropout", "resid drop"), ("chunk_ste", "chunk STE"),
                     ("router_consistency_weight", "consistency w"), ("router_gain_weight", "gain w"), ("mixer", "mixer")])


def load(name):
    with open(os.path.join(HERE, name + ".csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def db():
    return {t: load(t) for t in TABLES}


def md_escape(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(md_escape(c) for c in r) + " |")
    return "\n".join(out)


def short(v, n=28):
    v = str(v)
    if v == NULL or v == "":
        return ""
    if v.startswith("[{buckets"):  # chunk vocab: keep the informative part
        v = v.replace("buckets: ", "").replace("n_hash: ", "h").replace("max_len: ", "L").replace("pred_weight: ", "pw").replace(", n_neg: 1024", "").replace(", null", "+none")
        n = max(n, 34)
    if v.startswith("{ns:"):
        v = "n3-8 " + v.split("buckets: ")[-1].rstrip("}")
    return v if len(v) <= n else v[: n - 1] + "…"


def view_big(d):
    exps = {e["exp_id"]: e for e in d["experiments"]}
    settings = defaultdict(dict)
    for s in d["arm_settings"]:
        settings[(s["exp_id"], s["arm"])][s["setting"]] = s["value"]
    results = {(r["exp_id"], r["arm"]): r for r in d["arm_results"]}
    per_seed = defaultdict(list)
    for r in d["runs"]:
        if r["val_bpb_final"] != NULL:
            tag = "" if r["status"] == "done" else f"({r['status'][:4]})"
            per_seed[(r["exp_id"], r["arm"])].append((int(r["seed"]) if r["seed"] != NULL else 99, f"{r['val_bpb_final']}{tag}"))
    by_cat = defaultdict(list)
    for a in d["arms"]:
        by_cat[exps[a["exp_id"]]["category"]].append(a)
    out = ["# (a) The big table: one row per arm\n",
           "Settings are pivoted from `arm_settings.csv`; blank = not set for that arm (see README for the base configs). "
           "Per-seed values marked `(stop)`, `(runn)`, `(fail)`, `(void)` are not final and are excluded from mean ± sd. "
           "Δ BPE = mean − 1.634 (EXP-006 BPE) for tiny-scale arms; ladder arms show Δ vs their rung's BPE in the reference column.\n"]
    headers = ["exp", "arm", "label", "base"] + list(PIVOT.values()) + ["n", "mean ± sd", "per seed", "Δ BPE", "Δ ref (ref)", "verdict"]
    for cat in CATEGORY_ORDER + sorted(set(by_cat) - set(CATEGORY_ORDER)):
        arms = by_cat.get(cat)
        if not arms:
            continue
        out.append(f"\n## {cat}\n")
        rows = []
        for a in arms:
            k = (a["exp_id"], a["arm"])
            st, res = settings[k], results.get(k, {})
            seeds = ", ".join(v for _, v in sorted(per_seed.get(k, [])))
            mean = res.get("val_bpb_mean", NULL)
            msd = "" if mean == NULL else (f"{mean} ± {res['val_bpb_sd']}" if res.get("val_bpb_sd", NULL) != NULL else mean)
            dref = "" if res.get("delta_vs_reference", NULL) in (NULL, "") else f"{float(res['delta_vs_reference']):+.3f} ({res['reference']})"
            dbpe = "" if res.get("delta_vs_bpe", NULL) in (NULL, "") else f"{float(res['delta_vs_bpe']):+.3f}"
            rows.append([a["exp_id"], a["arm"], short(a["label"], 48), os.path.basename(a["base_config"]).replace(".yaml", "")]
                        + [short(st.get(s, "")) for s in PIVOT] + [res.get("n_seeds", ""), msd, seeds, dbpe, dref, res.get("verdict", "")])
        out.append(table(headers, rows))
    return "\n".join(out)


def view_runs(d):
    exps = {e["exp_id"]: e for e in d["experiments"]}
    by_exp = defaultdict(list)
    for r in d["runs"]:
        by_exp[r["exp_id"]].append(r)
    out = ["# (b) Runs per experiment\n"]
    for e in d["experiments"]:
        runs = by_exp.get(e["exp_id"])
        out.append(f"\n## {e['exp_id']} · {e['name']} · status {e['status']}\n")
        out.append(f"*{e['question']}*\n\n{e['outcome']}\n")
        if not runs:
            out.append("_no runs_\n")
            continue
        headers = ["arm", "seed", "status", "steps", "final step", "val bpb", "zero-main", "main contrib.", "train bpb", "adopted", "tokenizer", "source", "notes"]
        rows = [[r["arm"], r["seed"], r["status"], f"{r['steps_done']}/{r['steps_total']}", r["final_step"], r["val_bpb_final"], r["zero_main_bpb_final"],
                 r["main_contribution"], r["train_bpb_final"], r["adopted"], r["tokenizer_kind"], r["source"], short(r["notes"], 90)]
                for r in sorted(runs, key=lambda r: (r["arm"], r["seed"]))]
        out.append(table(headers, rows))
    return "\n".join(out)


def view_findings(d):
    by_cat = defaultdict(list)
    for f in d["findings"]:
        by_cat[f["category"]].append(f)
    out = ["# (c) Findings by category\n"]
    for cat in ["training-dynamics", "router-objective", "architecture", "optimizer", "data", "infra", "scaling"] + sorted(set(by_cat) - {"training-dynamics", "router-objective", "architecture", "optimizer", "data", "infra", "scaling"}):
        fs = by_cat.get(cat)
        if not fs:
            continue
        out.append(f"\n## {cat} ({len(fs)})\n")
        out.append(table(["id", "exp", "date", "statement", "evidence", "consequence", "work-log section"],
                         [[f["finding_id"], f["exp_id"], f["date"], f["statement"], f["evidence"], f["consequence"], short(f["worklog_section"], 60)] for f in fs]))
    return "\n".join(out)


def counts(d):
    return "\n".join(f"{t}.csv: {len(d[t])} rows" for t in TABLES)


def verify(d):
    """Re-read the last val/bpb of every run_dir in runs.csv and report drift (runs still in flight will differ)."""
    problems, checked = [], 0
    for r in d["runs"]:
        if r["run_dir"] == NULL or r["source"] != "eval_jsonl":
            continue
        path = r["run_dir"] if r["run_dir"].startswith("/") else os.path.join(REPO, r["run_dir"])
        ev = os.path.join(path, "eval.jsonl")
        if not os.path.isdir(path):
            problems.append(f"{r['exp_id']}/{r['arm']}_s{r['seed']}: run dir not found ({path})")
            continue
        last = None
        for line in (open(ev) if os.path.exists(ev) else []):
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("val/bpb") is not None:
                last = rec
        checked += 1
        now = NULL if last is None else f"{last['val/bpb']:.4f}"
        if now != r["val_bpb_final"]:
            problems.append(f"{r['exp_id']}/{r['arm']}_s{r['seed']}: runs.csv {r['val_bpb_final']} @ {r['final_step']} vs eval.jsonl {now} @ {last['step'] if last else None}")
    head = f"verified {checked} run directories; {len(problems)} differ from runs.csv"
    return "\n".join([head] + problems)


def main(argv):
    d = db()
    what = argv[1] if len(argv) > 1 else "all"
    if what == "counts":
        print(counts(d))
    elif what == "verify":
        print(verify(d))
    elif what == "big":
        print(view_big(d))
    elif what == "runs":
        print(view_runs(d))
    elif what == "findings":
        print(view_findings(d))
    else:
        print(view_big(d))
        print("\n\n")
        print(view_runs(d))
        print("\n\n")
        print(view_findings(d))


if __name__ == "__main__":
    main(sys.argv)
