"""Apples-to-apples gate for a sweep: every run must see the same data and spend the same compute per byte.

    python scripts/search/fairness.py SPEC [--mode strict|tuned] [--param name=value ...] [--params-file F]
        [--allow-missing] [--flops-tol 0.02] [--bytes-tol 0.02]

For every (architecture, grid point) of the spec (first seed) it composes the exact training config and reports:
  data          dataset + shards + window order (shuffle) + cross-window labels
  bytes/step    UTF-8 text bytes per optimizer step (token runs: tokens/step x measured bytes/token)
  total bytes   bytes over the whole run
  GFLOPs/byte   training FLOPs per byte at target compression (hnet.flops, SPEC convention)
  optimizer     name, peak LR, betas, weight decay, clip (+ the Muon section for muon), schedule
  precision / compile / val set / eval cadence / downstream suite

Modes (spec ``fairness:``, or ``--mode``):

* **strict** (default): every held-equal field is identical across all runs and grid points (FLOPs/byte and
  bytes within tolerance). A grid over a held field (e.g. ``train.lr``) therefore fails: use tuned.
* **tuned**: runs may differ in the swept hyperparameters, and the comparison is made at each architecture's best
  grid value (scripts/search/report.py). The gate requires (1) every architecture is swept over the SAME grid
  (same keys, same values) with the same seeds, and (2) all other held-equal fields are identical. Keys listed in
  ``fairness.per_arch`` may also differ between architectures (values tuned per architecture in an earlier tuned
  sweep, e.g. EXP-004's Adam LR from EXP-003); each such key must be set in every run's ``set`` from a spec
  parameter (``key=${param}``), so its provenance is in the spec, and must not also be a grid key.

Fields that may differ by design (the varied factor, H-Net-only knobs such as LR multipliers) are listed but not
enforced. Exit code 1 on any problem (or missing config files, unless ``--allow-missing``, which checks the other
runs and lists the missing ones). Writes ``<out_root>/fairness.md`` and ``fairness.json``.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _sys.path:  # allow `python scripts/...` from anywhere without PYTHONPATH
    _sys.path.insert(0, str(REPO_ROOT))
_HERE = str(_Path(__file__).resolve().parent)
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from spec import Spec, SpecError, compose_config  # noqa: E402

TUNED = "<tuned>"
HELD_EXACT = ["data", "optimizer", "schedule", "precision", "val", "downstream"]
HELD_TOL = {"bytes_per_step": "bytes_tol", "total_bytes_M": "bytes_tol", "gflops_per_byte": "flops_tol"}


def _mask(cfg: dict, keys) -> dict:
    out = copy.deepcopy(cfg)
    for k in keys:
        node, parts = out, k.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node[parts[-1]] = TUNED
    return out


def _optimizer_str(t: dict) -> str:
    s = f"{t.get('optimizer')} lr={t.get('lr')} betas={t.get('betas')} wd={t.get('weight_decay')} " \
        f"clip={t.get('grad_clip')}"
    if t.get("optimizer") == "muon":
        from hnet.optim.muon import MUON_DEFAULTS

        sec = t.get("muon") or {}
        if sec == TUNED:
            sec = {}
        mu = {**MUON_DEFAULTS, **sec}
        s = f"muon wd={t.get('weight_decay')} clip={t.get('grad_clip')} " + " ".join(
            f"{k}={list(v) if isinstance(v, tuple) else v}" for k, v in sorted(mu.items()))
    return s


def describe(cfg: dict, mask_keys=()) -> dict:
    """Held-equal and informational fields of one composed config. ``mask_keys`` (tuned mode) are replaced by
    ``<tuned>`` in the held-equal text fields; model-derived numbers use the real config."""
    from hnet.data.loader import bytes_per_token
    from hnet.flops import model_flops
    from hnet.train.train import build_model_from_section

    model, mcfg = build_model_from_section(cfg["model"])
    m = _mask(cfg, mask_keys)
    d, t, e = m["data"], m["train"], m.get("eval", {})
    is_bytes = hasattr(mcfg, "n_stages")
    bpt = bytes_per_token(cfg["data"].get("dataset", "finewebedu10B"))
    rd, rt = cfg["data"], cfg["train"]
    seq = int(rd["seq_len"])
    windows = (int(rd["batch_bytes"]) // seq) if is_bytes else (int(rd["batch_tokens"]) // seq)
    bytes_step = windows * seq if is_bytes else windows * seq * bpt
    total_bytes = float(rt["train_tokens"]) * bpt if rt.get("train_tokens") else float(rt.get("train_bytes") or 0)
    fl = model_flops(mcfg, seq_len=seq) if is_bytes else model_flops(mcfg, seq_len=seq, bytes_per_token=bpt)
    sched = t.get("schedule", {}) if isinstance(t.get("schedule"), dict) else {"name": t.get("schedule")}
    ds = e.get("downstream") or {}
    dc = getattr(mcfg, "dc", None)
    return {
        "kind": "bytes" if is_bytes else "tokens",
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "data": f"{d.get('dataset')} shards={d.get('n_train_shards', 'all')} shuffle={d.get('shuffle_buffer', 0)} "
                f"xwin={d.get('cross_window_label', True)}",
        "bytes_per_step": round(bytes_step),
        "total_bytes_M": round(total_bytes / 1e6, 1),
        "gflops_per_byte": round(fl["flops_per_byte"] / 1e9, 5),
        "optimizer": _optimizer_str(t),
        "schedule": f"{sched.get('name')} warmup={sched.get('warmup')} decay={sched.get('decay')} "
                    f"shape={sched.get('decay_shape')}",
        "precision": f"{t.get('precision')} compile={t.get('compile')}",
        "val": f"{e.get('val')} {e.get('val_bytes')}B every={e.get('every')}",
        "downstream": f"{ds.get('suite')} every={ds.get('every')} limit={ds.get('limit')}" if ds else "none",
        # allowed to differ by design:
        "router": getattr(dc, "router", "-"),
        "dechunker": getattr(dc, "dechunker", "-"),
        "ratios": getattr(dc, "target_ratios", "-"),
        "alpha": getattr(dc, "ratio_loss_alpha", "-"),
        "lr_mult": getattr(mcfg, "lr_multiplier", "-"),
        "lr": rt.get("lr") if rt.get("optimizer") != "muon" else (rt.get("muon") or {}).get("lr"),
    }


def check_tuned_structure(spec: Spec, per_arch) -> list:
    """Grid identity, seed identity and per-arch provenance (tuned mode)."""
    problems = []
    runs = spec.raw["runs"]
    grids = {r["name"]: spec.grid_of(r) for r in runs}
    ref_name = runs[0]["name"]
    ref = grids[ref_name]
    for name, g in grids.items():
        if list(g) != list(ref):
            problems.append(f"{name}: grid keys {list(g)} differ from {ref_name}'s {list(ref)}")
            continue
        for k in g:
            if sorted(map(str, g[k])) != sorted(map(str, ref[k])):
                problems.append(f"{name}: grid {k}={g[k]} differs from {ref_name}'s {ref[k]}")
    seeds = {r["name"]: [int(s) for s in r.get("seeds", spec.seeds)] for r in runs}
    for name, s in seeds.items():
        if s != seeds[ref_name]:
            problems.append(f"{name}: seeds {s} differ from {ref_name}'s {seeds[ref_name]} (paired comparisons)")
    for key in per_arch:
        if key in ref:
            problems.append(f"per_arch key {key} is also a grid key")
        for r in runs:
            refs = spec.param_refs(r)
            if key not in refs:
                problems.append(f"{r['name']}: per_arch key {key} must be set in the run's `set` as "
                                f"{key}=${{param}} (provenance); found {refs or 'no param refs'}")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--mode", choices=("strict", "tuned"), default=None, help="default: the spec's fairness mode")
    ap.add_argument("--param", action="append", default=[], help="spec parameter name=value (repeatable)")
    ap.add_argument("--params-file", default=None)
    ap.add_argument("--allow-missing", action="store_true", help="check the runs whose config files exist")
    ap.add_argument("--flops-tol", type=float, default=0.02)
    ap.add_argument("--bytes-tol", type=float, default=0.02)
    a = ap.parse_args(argv)
    try:
        spec = Spec.load(a.spec, a.param, a.params_file)
        mode = a.mode or spec.fairness_mode()
        per_arch = spec.per_arch_keys()
        jobs = spec.jobs(allow_missing=a.allow_missing)
    except SpecError as e:
        print(f"[fairness] {e}", flush=True)
        return 1
    problems = []
    if per_arch and mode != "tuned":
        problems.append(f"fairness.per_arch {per_arch} requires mode tuned")
    grid_keys = sorted({k for j in jobs for k in j.point})
    if mode == "tuned":
        problems += check_tuned_structure(spec, per_arch)
    mask = (grid_keys + per_arch) if mode == "tuned" else []
    rows, first = {}, {}
    for j in jobs:
        key = (j.arch, j.setting)
        if key in first:
            continue
        first[key] = j
        rows[key] = describe(compose_config(j), mask)
    (ref_arch, ref_set), ref = next(iter(rows.items()))
    for (arch, setting), r in rows.items():
        label = f"{arch} [{setting}]" if setting != "-" else arch
        for k in HELD_EXACT:
            if r[k] != ref[k]:
                hint = f" (the grid varies {grid_keys}: use fairness mode tuned)" if mode == "strict" and grid_keys \
                    else ""
                problems.append(f"{label}: {k} differs: {r[k]!r} vs {ref_arch} {ref[k]!r}{hint}")
        for k, tol in HELD_TOL.items():
            rel = abs(r[k] - ref[k]) / max(abs(ref[k]), 1e-12)
            if rel > getattr(a, tol):
                problems.append(f"{label}: {k} {r[k]} vs {ref[k]} ({100 * rel:+.1f}%)")
    cols = ["kind", "params_M", "bytes_per_step", "total_bytes_M", "gflops_per_byte", "lr", "router", "dechunker",
            "ratios", "alpha", "lr_mult"]
    lines = [f"# Fairness check: {spec.name}", "", f"Mode: **{mode}**" + (
        " (runs may differ only in the grid keys" + (f" and the per-architecture keys {per_arch}" if per_arch
                                                     else "") + "; compare each architecture at its best value)"
        if mode == "tuned" else " (all held-equal fields identical across every run)"), "",
        "Held equal across all runs: " + "; ".join(f"**{k}** = `{ref[k]}`" for k in HELD_EXACT), ""]
    if grid_keys:
        g0 = spec.grid_of(spec.raw["runs"][0])
        lines += ["Grid (identical for every architecture: " + ("checked" if mode == "tuned" else "not required")
                  + "): " + "; ".join(f"`{k}` ∈ {g0.get(k)}" for k in grid_keys), ""]
    if per_arch:
        lines += ["Tuned per architecture (from spec params" + (f"; {spec.raw.get('params_note')}"
                                                              if spec.raw.get("params_note") else "") + "):", ""]
        for r in spec.raw["runs"]:
            refs = spec.param_refs(r)
            lines.append(f"- {r['name']}: " + ", ".join(
                f"`{k}` = {spec.subst('${' + refs[k] + '}') if k in refs else '?'}" for k in per_arch))
        lines.append("")
    lines += ["| run | setting | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for (arch, setting), r in rows.items():
        lines.append(f"| {arch} | {setting} | " + " | ".join(str(r[c]) for c in cols) + " |")
    if spec.missing:
        lines += ["", "**NOT CHECKED (config files missing):** " + "; ".join(f"{r}: `{p}`" for r, p in spec.missing)]
    lines += ["", "**Problems:** " + ("none" if not problems else ""), *[f"- {p}" for p in problems]]
    text = "\n".join(lines) + "\n"
    print(text)
    out = Path(spec.out_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fairness.md").write_text(text)
    (out / "fairness.json").write_text(json.dumps(
        {"mode": mode, "rows": {f"{a_}|{s}": r for (a_, s), r in rows.items()}, "problems": problems,
         "missing": spec.missing, "grid_keys": grid_keys, "per_arch": per_arch}, indent=1, default=str))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
