"""Evaluation CLI.

    python -m hnet.eval.run --model <spec> --suite paper7|smollm3_cf|robustness|bpb|all [--tasks a,b]
                            [--limit N] [--out results.json] [--items items.jsonl] [--n-boot 1000]

Model specs
  runs/a2                          our run dir -> latest ``ckpt/step_*`` (trainer layout, SPEC)
  runs/a2/ckpt/step_0010000        a specific checkpoint (``model.pt`` + ``meta.json`` with the run config)
  hf:gpt2 | hf:HuggingFaceTB/SmolLM2-135M@<revision>   transformers checkpoint (HFScorer)
  official:cartesia-ai/hnet_2stage_L                     goombalab/hnet weights via hnet.model.load_official_checkpoint
                                                         (config from configs/official/<name>.json)

``--suite bpb`` evaluates FineWeb-Edu val bpb (``hnet.data.valsets.standard_valset``; ``--valset`` picks the dataset)
for our byte/token models; HF models with other tokenizers use the decoded val documents (``bpb_of_texts``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

import torch

from hnet.eval.scorers import ByteModelScorer, HFScorer, TokenModelScorer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ALL_SUITES = ("paper7", "smollm3_cf", "robustness")


# ====================================================================== model loading
def resolve_checkpoint(path: str) -> str:
    if os.path.exists(os.path.join(path, "model.pt")):
        return path
    from hnet.train.checkpoint import latest_checkpoint

    ck = latest_checkpoint(path)
    if ck is None:
        raise FileNotFoundError(f"no checkpoint (model.pt or ckpt/step_*) under {path}")
    return ck


def load_run_checkpoint(path: str, device="cpu", dtype=torch.float32):
    """-> (model, model_cfg, meta). Trainer layout: ``model.pt`` (state_dict) + ``meta.json`` ({"config": run cfg})."""
    ck = resolve_checkpoint(path)
    with open(os.path.join(ck, "meta.json")) as f:
        meta = json.load(f)
    section = meta["config"]["model"]
    from hnet.train.train import build_model_from_section

    model, mcfg = build_model_from_section(section)
    state = torch.load(os.path.join(ck, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(device=device, dtype=dtype).eval()
    for m in model.modules():  # residual_proj stays fp32 (as in training and the official model; audit B3)
        if hasattr(m, "residual_proj"):
            m.residual_proj.float()
    meta["checkpoint"] = ck
    return model, mcfg, meta


def model_kind(mcfg) -> str:
    from hnet.config import DenseConfig, HNetConfig

    if isinstance(mcfg, HNetConfig):
        return "byte"
    if isinstance(mcfg, DenseConfig):
        return "token"
    return "byte" if getattr(mcfg, "vocab_size", 256) <= 256 else "token"


def load_scorer(spec: str, device=None, batch_tokens: Optional[int] = None, max_len: Optional[int] = None,
                dtype: Optional[str] = None, add_bos: Optional[bool] = None):
    """-> (scorer, info dict)."""
    device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    tdtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32, None: None}[dtype]
    kw = {}
    if batch_tokens:
        kw["batch_tokens"] = batch_tokens
    if max_len:
        kw["max_len"] = max_len
    if spec.startswith("hf:"):
        name, _, rev = spec[3:].partition("@")
        sc = HFScorer(name, revision=rev or None, dtype=tdtype, device=device, add_bos=add_bos, **kw)
        return sc, {"type": "hf", "name": name, "revision": rev or None, "kind": "token", "add_bos": sc.add_bos}
    if spec.startswith("official:"):
        from hnet.model import load_official_checkpoint

        repo = spec[len("official:"):]
        name = repo.rstrip("/").split("/")[-1].removesuffix(".pt")
        cfg_json = os.path.join(REPO_ROOT, "configs", "official", f"{name}.json")
        model = load_official_checkpoint(repo, cfg_json, dtype=tdtype or torch.float32, device=device)
        return ByteModelScorer(model, device=device, **kw), {"type": "official", "name": repo, "kind": "byte"}
    model, mcfg, meta = load_run_checkpoint(spec, device=device, dtype=tdtype or torch.float32)
    kind = model_kind(mcfg)
    if kind == "byte":
        sc = ByteModelScorer(model, device=device, **kw)
    else:
        sc = TokenModelScorer(model, device=device, add_bos=True if add_bos is None else add_bos, **kw)
    info = {"type": "run", "name": spec, "kind": kind, "checkpoint": meta["checkpoint"], "step": meta.get("step"),
            "tokens_eq": meta.get("tokens_eq"), "bytes_seen": meta.get("bytes_seen")}
    return sc, info


# ====================================================================== main
def _flatten(summary: dict) -> dict:
    """Compact {name: value} view for logs/tables."""
    out = {}
    suite = summary.get("suite")
    if suite == "robustness":
        out["robustness/clean"] = summary["clean"]["value"]
        for p, v in summary["perturbations"].items():
            out[f"robustness/{p}"] = v["value"]
        out["robustness/average"] = summary["average"]["value"]
        out["robustness/score"] = summary["robustness_score"]["value"]
        return out
    for t, s in summary.get("tasks", {}).items():
        for m, v in s["metrics"].items():
            out[f"{suite}/{t}/{m}"] = v["value"]
    for t, ms in summary.get("derived", {}).items():
        for m, v in ms.items():
            out[f"{suite}/{t}/{m}"] = v["value"]
    for a, v in summary.get("averages", {}).items():
        out[f"{suite}/{a}"] = v["value"]
    return out


def run_bpb(scorer, info, valset_name: str, n_bytes: int, max_batches=None):
    from hnet.data.valsets import standard_valset
    from hnet.eval.bpb import bpb_of_texts, valset_bpb, valset_texts

    vs = standard_valset(valset_name, n_bytes_target=n_bytes)
    if info["type"] in ("run", "official"):
        return valset_bpb(scorer.model, vs, info["kind"], device=scorer.device, max_batches=max_batches)
    if info["type"] == "hf" and info["name"] in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl",
                                                 "openai-community/gpt2"):
        return valset_bpb(scorer.model, vs, "token", device=scorer.device, max_batches=max_batches, hf=True)
    return bpb_of_texts(scorer, valset_texts(vs))


def evaluate_model(model, model_cfg=None, kind: str = "bytes", device=None, suites=("paper7",), limit: int = 500,
                   n_boot: int = 0, tasks=None, subset_seed: Optional[int] = 0, suite=None, **kw) -> dict:
    """Hook for the trainer (``eval.downstream: {suite|suites, tasks, limit, n_boot}``): flat {name: float} for a live
    model. ``limit`` items per task are a fixed seeded random subset (``subset_seed``), identical across checkpoints."""
    if suite is not None:
        suites = suite
    from hnet.eval.harness import evaluate_suite

    was_training = getattr(model, "training", False)
    k = "byte" if kind in ("bytes", "byte") else "token"
    out = {}
    if isinstance(tasks, str):
        tasks = tasks.split(",")
    try:
        sc = ByteModelScorer(model, device=device, **kw) if k == "byte" else TokenModelScorer(model, device=device, **kw)
        for s in (suites.split(",") if isinstance(suites, str) else suites):
            summary, _ = evaluate_suite(sc, s, tasks=tasks, limit=limit, n_boot=n_boot, subset_seed=subset_seed)
            out.update(_flatten(summary))
    finally:
        model.train(was_training)  # the scorer puts the model in eval mode
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", default="paper7", help="paper7|smollm3_cf|robustness|bpb|all (comma list ok)")
    ap.add_argument("--tasks", default=None, help="comma-separated task names within the suite")
    ap.add_argument("--limit", type=int, default=None, help="max items per task (first N, as lm-eval)")
    ap.add_argument("--subset-seed", type=int, default=None,
                    help="with --limit: a seeded random subset instead of the first N items")
    ap.add_argument("--out", default=None, help="results json")
    ap.add_argument("--items", default=None, help="per-item records jsonl (for paired bootstrap)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--batch-tokens", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--add-bos", type=int, default=None, choices=[0, 1], help="override BOS/EOT prefix (token models)")
    ap.add_argument("--valset", default="finewebedu10B")
    ap.add_argument("--val-bytes", type=int, default=10_000_000)
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)

    from hnet.eval.harness import evaluate_suite

    suites = ["paper7", "smollm3_cf", "robustness", "bpb"] if a.suite == "all" else a.suite.split(",")
    t0 = time.time()
    scorer, info = load_scorer(a.model, a.device, a.batch_tokens, a.max_len, a.dtype,
                               None if a.add_bos is None else bool(a.add_bos))
    results = {"model": a.model, "info": info, "limit": a.limit, "seed": a.seed, "suites": {}, "flat": {}}
    items_f = open(a.items, "w") if a.items else None
    for s in suites:
        ts = time.time()
        if s == "bpb":
            r = run_bpb(scorer, info, a.valset, a.val_bytes)
            results["suites"]["bpb"] = r
            results["flat"][f"bpb/{a.valset}"] = r["bpb"]
        else:
            tasks = a.tasks.split(",") if a.tasks else None
            summary, recs = evaluate_suite(scorer, s, tasks=tasks, limit=a.limit, n_boot=a.n_boot, seed=a.seed,
                                           progress=a.progress, subset_seed=a.subset_seed)
            summary["seconds"] = time.time() - ts
            results["suites"][s] = summary
            results["flat"].update(_flatten(summary))
            if items_f:
                for t, rr in recs.items():
                    for r in rr:
                        items_f.write(json.dumps({"model": a.model, "suite": s, "task": t, **r}) + "\n")
        print(f"[eval] {s} done in {time.time() - ts:.1f}s", file=sys.stderr)
    if items_f:
        items_f.close()
    results["seconds"] = time.time() - t0
    for k, v in results["flat"].items():
        print(f"{k:60s} {v:.4f}" if isinstance(v, float) else f"{k:60s} {v}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1, default=float)
    return results


if __name__ == "__main__":
    main()
