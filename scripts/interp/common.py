"""Shared helpers for the mechanistic-interpretability pass (logbook/interp/).

* ``enumerate_runs()``: every run in ``logbook/db/runs.csv`` with a readable ``metrics.jsonl``, joined with the
  settings read from the run's own ``config.yaml`` (falling back to the local ``runs/gcp`` mirror for ladder runs).
* ``load_metrics(run_dir)`` / ``load_evals(run_dir)``: per-step training metrics and per-eval records as DataFrames.
* ``OUT``: logbook/interp; ``DATA``/``FIG`` its subdirectories.

Nothing here modifies the repository outside scripts/interp and logbook/interp.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("INTERP_OUT", ROOT / "logbook" / "interp"))  # run_all --quick writes to .../interp/quick
DATA = OUT / "data"
FIG = OUT / "figures"
LIVE = Path("/tmp/claude-0/-home-user-hnet-smollm3/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad/live")
for d in (DATA, FIG):
    d.mkdir(parents=True, exist_ok=True)


def _jsonl(path: Path) -> list:
    out = []
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def load_metrics(run_dir) -> pd.DataFrame:
    rows = _jsonl(Path(run_dir) / "metrics.jsonl")
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)
    return df


def load_evals(run_dir) -> pd.DataFrame:
    rows = [{k: v for k, v in r.items() if k == "step" or k.startswith("val/")} for r in _jsonl(Path(run_dir) / "eval.jsonl")]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "val/bpb" not in df:
        return pd.DataFrame()
    return df.dropna(subset=["val/bpb"]).drop_duplicates("step", keep="last").sort_values("step").reset_index(drop=True)


def _resolve_dir(run_dir: str) -> Optional[Path]:
    if not isinstance(run_dir, str):
        return None
    p = Path(run_dir)
    if not p.is_absolute():
        p = ROOT / p
    return p if (p / "metrics.jsonl").exists() else None


def _find_config(p: Path, exp: str, arm: str, seed) -> Optional[dict]:
    cands = [p / "config.yaml"]
    if exp.startswith("LADDER-1e16-"):
        rung = exp.split("-")[-1]
        cands.append(ROOT / f"runs/gcp/hnet-ladder-1e16/search/ladder/1e16/{rung}/{arm}_s{int(seed)}/config.yaml")
    if exp.startswith("LADDER-LR"):
        for base in ("hnet-ladder-lr", "hnet-ladder-lr2"):
            for c in (ROOT / "runs/gcp" / base).rglob(f"{arm}_s{int(seed)}/config.yaml"):
                cands.append(c)
    for c in cands:
        if c.exists():
            with open(c) as f:
                return yaml.safe_load(f)
    return None


def _get(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def run_features(cfg: Optional[dict], arm: str, exp: str) -> dict:
    """Flat settings of a run from its config (None-safe)."""
    m = _get(cfg, "model", default={}) or {}
    t = _get(cfg, "train", default={}) or {}
    dc = m.get("dc", {}) or {}
    kind = "bpe" if (m.get("type") in ("dense", "transformer", "gpt", "dense_modded") or arm.startswith("bpe")
                     or "bpe" == arm) else "hnet"
    if m.get("type") == "hnet":
        kind = "hnet"
    cv = dc.get("chunk_vocab")
    cv0 = cv[0] if isinstance(cv, list) and cv else (cv if isinstance(cv, dict) else None)
    lay = m.get("arch_layout")
    n_stages = None
    if kind == "hnet" and lay is not None:
        n_stages, x = 0, lay
        while isinstance(x, list) and len(x) == 3:
            n_stages += 1
            x = x[1]
    opt = t.get("optimizer")
    return {
        "kind": kind,
        "n_stages": n_stages,
        "router": dc.get("router"),
        "decision": dc.get("decision"),
        "optimizer": opt,
        "lr": t.get("lr"),
        "muon_lr": _get(t, "muon", "lr") if isinstance(t.get("muon"), dict) else t.get("muon_lr"),
        "ratio_alpha": dc.get("ratio_loss_alpha"),
        "chunk_vocab": cv0 is not None,
        "vocab_pred_w": (cv0 or {}).get("pred_weight", 0.0) if cv0 else 0.0,
        "ngram": bool(m.get("ngram_embed")),
        "prior_w": dc.get("router_prior_weight", 0.0) or 0.0,
        "residual_init": dc.get("residual_init", "zero"),
        "residual_warmup": dc.get("residual_warmup", 0.0) or 0.0,
        "residual_gate": dc.get("residual_gate", "none"),
        "residual_dropout": dc.get("residual_dropout", 0.0) or 0.0,
        "chunk_ste": bool(dc.get("chunk_ste", False)),
        "target_ratios": dc.get("target_ratios"),
        "temp": _get(dc, "decision_kwargs", "temp"),
        "d_model": m.get("d_model"),
        "scale": "ladder" if exp.startswith("LADDER") else "tiny",
    }


def enumerate_runs() -> pd.DataFrame:
    r = pd.read_csv(ROOT / "logbook/db/runs.csv")
    rows = []
    for _, x in r.iterrows():
        p = _resolve_dir(x.run_dir)
        if p is None:
            continue
        cfg = _find_config(p, x.exp_id, x.arm, x.seed if not (isinstance(x.seed, float) and math.isnan(x.seed)) else 0)
        f = run_features(cfg, x.arm, x.exp_id)
        if cfg is None and x.exp_id.startswith("LADDER"):
            f["kind"] = "bpe" if x.arm.startswith("bpe") else "hnet"
            f["optimizer"] = "muon"
            f["muon_lr"] = float(x.arm.split("mlr")[1]) if "mlr" in x.arm else 0.02
            f["n_stages"] = 1 if f["kind"] == "hnet" else None
            f["prior_w"] = 1.0 if x.arm.startswith("hnet_ws") else 0.0
        if x.exp_id.startswith("LADDER") and f.get("muon_lr") is None and "mlr" in x.arm:
            f["muon_lr"] = float(x.arm.split("mlr")[1])
        if x.exp_id.startswith("LADDER") and f.get("muon_lr") is None:
            f["muon_lr"] = 0.02
        rid = f"{x.exp_id}/{x.arm}_s{int(x.seed) if not pd.isna(x.seed) else 0}"
        rows.append({"run_id": rid, "exp_id": x.exp_id, "arm": x.arm, "seed": x.seed, "dir": str(p),
                     "status": x.status, "val_bpb_db": x.val_bpb_final, "zero_main_db": x.zero_main_bpb_final,
                     "adopted_db": x.adopted, "tokenizer_db": x.tokenizer_kind, **f})
    # live mirrors: runs that finished or appeared after the runs.csv snapshot (EXP-025/026/027, ladder)
    have = {(x["exp_id"], x["arm"], int(x["seed"]) if not pd.isna(x["seed"]) else 0) for x in rows}
    have_dirs = {x["dir"] for x in rows}
    live_map = {"exp025_muon": "EXP-025", "exp026_mamba_free": "EXP-026", "exp027_adoption_mechanism": "EXP-027",
                "ladder_lr_1e16": "LADDER-LR-1e16", "lrext_1e16": "LADDER-LREXT-1e16"}
    if LIVE.exists():
        for sub in sorted(LIVE.iterdir()):
            exp = live_map.get(sub.name)
            if exp is None and sub.name.startswith("ladder_1e1"):
                b, rung = sub.name.split("_")[1:3]
                exp = f"LADDER-{b}-{rung}"
            if exp is None:
                continue
            for rd in sorted(sub.glob("*_s[0-9]")):
                if not (rd / "metrics.jsonl").exists() or str(rd) in have_dirs:
                    continue
                arm, seed = rd.name.rsplit("_s", 1)
                if (exp, arm, int(seed)) in have:
                    continue
                cfg = _find_config(rd, exp, arm, int(seed))
                f = run_features(cfg, arm, exp)
                if cfg is None and exp.startswith("LADDER"):
                    f.update(kind="bpe" if arm.startswith("bpe") else "hnet", optimizer="muon",
                             muon_lr=float(arm.split("mlr")[1]) if "mlr" in arm else 0.02,
                             n_stages=None if arm.startswith("bpe") else 1, prior_w=1.0 if arm.startswith("hnet_ws") else 0.0)
                rows.append({"run_id": f"{exp}/{arm}_s{seed}", "exp_id": exp, "arm": arm, "seed": float(seed),
                             "dir": str(rd), "status": "live", "val_bpb_db": None, "zero_main_db": None,
                             "adopted_db": None, "tokenizer_db": None, **f})
    return pd.DataFrame(rows)


def json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def dump(obj, name: str):
    path = DATA / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=json_default)
    return path
