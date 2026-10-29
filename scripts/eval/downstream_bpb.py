#!/usr/bin/env python
"""Downstream "soft accuracy": bits per byte of the correct continuation on the paper7 tasks (+ Paloma-like domains).

    python scripts/eval/downstream_bpb.py --model runs/ladder/bpe_1e17_L8 [--limit 1000] [--domains paloma_like]
    python scripts/eval/downstream_bpb.py --model hf:gpt2 --boundary lmeval --out gpt2_lmeval.json

Writes ``<run>/downstream_bpb.json`` (or ``--out``); ``scripts/analysis/fit_ladder.py`` reads ``down:<task>`` and
``down:avg`` from it (loss-to-loss laws, downstream compute multipliers).

Metric (the blog's OLMES / Paloma bpb treatment)
-----------------------------------------------
Per item: -log2 p(gold continuation | context) / UTF-8 bytes of the continuation; per task pooled over items
(sum bits / sum bytes; ``hnet.eval.harness`` ``soft_bpb``; LAMBADA: the target word, ``bpb``); ``average`` = the
unweighted mean over the 7 tasks, with a bootstrap CI that resamples items within every task. Prompts are the
lm-eval paper7 prompts (``hnet.eval.tasks.lmeval``); trailing context whitespace moves into the continuation for
every model (``split_trailing_space``), so all models score the same bytes. By default only the gold choice is
scored (``--all-choices`` also scores the distractors and reports accuracies).

Prompt-boundary handling for BPE models (``--boundary``, default ``canonical``)
-----------------------------------------------------------------------------
* ``canonical``: tokenize context + continuation *as one string* (the model's canonical tokenization, i.e. what
  it saw in training), and score the tokens from the first token that contains any continuation byte. When the
  context/continuation boundary falls inside a token (``straddle``), that token is scored as part of the answer, so
  its context bytes are charged to the answer (a small pessimistic bias for BPE; the bpb denominator stays the
  continuation's bytes so it is the same for every model). The straddle rate is reported
  (``boundary_stats``); with GPT-2 pretokenization and the leading space moved into the continuation it is ~0.
* ``lmeval``: lm-eval ``_encode_pair``: ctx = enc(context), continuation = enc(context + continuation)[len(ctx):];
  the model input is ctx + that tail, which is *not* a canonical tokenization when enc(context) is not a prefix of
  the joint encoding (the prompt-boundary problem; ``boundary_stats.noncanonical`` counts these items).
Byte models (H-Net) have no boundary problem: the continuation bytes are scored exactly.
The blog (Part II) notes that BPE without pretokenization needs marginalization over tokenizations; we do not
marginalize (this repo's BPE baselines use GPT-2 pretokenization, for which canonical == lm-eval almost always).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
from typing import List, Optional

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hnet.eval.scorers import ByteModelScorer, HFScorer, TokenModelScorer, split_trailing_space  # noqa: E402
from hnet.eval.tasks.base import LMItem, MCItem  # noqa: E402

LN2 = math.log(2.0)
PAPER7 = ("lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande", "openbookqa")

# Ungated Paloma-like held-out domains (Paloma itself is gated on the HF hub). Evaluated with per_source.py.
PALOMA_LIKE = [
    {"name": "wikitext103", "group": "paloma_like", "doc_split": "wikitext", "max_bytes": 1_000_000,
     "paths": ["hf://Salesforce/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet"]},
    {"name": "c4_en", "group": "paloma_like", "max_bytes": 1_000_000,
     "paths": ["hf://allenai/c4/en/c4-validation.00000-of-00008.json.gz"]},
]


# ====================================================================================================== boundary
class CanonicalBoundaryMixin:
    """``encode_pair`` that scores the canonical tokenization of context + continuation (see module docstring)."""

    boundary = "canonical"

    def _init_boundary(self, boundary: str):
        assert boundary in ("canonical", "lmeval"), boundary
        self.boundary = boundary
        self.boundary_stats = {"pairs": 0, "straddle": 0, "noncanonical": 0}

    def _token_ends(self, text: str, ids: List[int]) -> np.ndarray:
        """Cumulative UTF-8 byte offset of the end of each token of ``ids`` = encoding of ``text``."""
        raise NotImplementedError

    def encode_pair(self, context, continuation):
        bs = self.boundary_stats
        bs["pairs"] += 1
        c2, x2 = split_trailing_space(context, continuation)
        if c2 != "":
            whole = self._enc(c2 + x2)
            ctx_lm = self._enc(c2)
            if ctx_lm != whole[:len(ctx_lm)]:
                bs["noncanonical"] += 1
        if self.boundary == "lmeval" or c2 == "":
            return super().encode_pair(context, continuation)
        nb = len(x2.encode("utf-8"))
        pre = [self.bos_id] if self.add_bos else []
        ends = self._token_ends(c2 + x2, whole)
        bnd = len(c2.encode("utf-8"))
        k = int(np.searchsorted(ends, bnd, side="right"))  # first token ending after the boundary
        start_k = int(ends[k - 1]) if k > 0 else 0
        if start_k < bnd:
            bs["straddle"] += 1
        ctx, cont = whole[:k], whole[k:]
        if not pre and not ctx:
            pre = [self.prefix_id]
        return pre + ctx, cont, nb


class CanonicalTokenScorer(CanonicalBoundaryMixin, TokenModelScorer):
    def __init__(self, model, boundary: str = "canonical", **kw):
        super().__init__(model, **kw)
        self._init_boundary(boundary)

    def _token_ends(self, text, ids):
        return np.cumsum([len(self.tok.decode_single_token_bytes(t)) for t in ids])


class CanonicalHFScorer(CanonicalBoundaryMixin, HFScorer):
    def __init__(self, name, boundary: str = "canonical", **kw):
        super().__init__(name, **kw)
        self._init_boundary(boundary)

    def _token_ends(self, text, ids):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert list(enc["input_ids"]) == list(ids)
        cb = np.zeros(len(text) + 1, dtype=np.int64)
        np.cumsum([len(ch.encode("utf-8")) for ch in text], out=cb[1:])
        return np.maximum.accumulate(np.array([cb[e] for _, e in enc["offset_mapping"]], dtype=np.int64))


# ====================================================================================================== tasks
def gold_only(item):
    """MC item reduced to its (first) gold choice: soft bpb needs only p(gold)."""
    if item is None or isinstance(item, LMItem):
        return item
    g = item.gold[0]
    return dataclasses.replace(item, contexts=[item.contexts[g]], continuations=[item.continuations[g]],
                               norm_texts=[item.norm_texts[g]], gold=[0],
                               choices=[item.choices[g]] if item.choices else None)


def paper7_tasks(names=None, all_choices: bool = False):
    from hnet.eval.tasks import get_task

    out = []
    for n in (names or PAPER7):
        t = get_task(f"paper7/{n}")
        if not all_choices and t.kind == "mc":
            build = t.build
            t = dataclasses.replace(t, build=lambda row, i, b=build: gold_only(b(row, i)))
        out.append(t)
    return out


def _item_arrays(task, recs):
    if task.kind == "lm":
        return np.array([-r["lp"] / LN2 for r in recs]), np.array([r["n_bytes"] for r in recs], float)
    return np.array([r["gold_nll_bits"] for r in recs]), np.array([r["gold_bytes"] for r in recs], float)


def average_ci(arrs, n_boot: int, seed: int = 0):
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    vals = np.zeros(n_boot)
    for bits, nb in arrs:
        idx = rng.integers(0, len(bits), size=(n_boot, len(bits)))
        vals += bits[idx].sum(1) / nb[idx].sum(1)
    vals /= len(arrs)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate_downstream(scorer, tasks=None, limit: Optional[int] = None, subset_seed: Optional[int] = 0,
                        n_boot: int = 1000, seed: int = 0, all_choices: bool = False, task_objs=None,
                        progress: bool = False) -> dict:
    from hnet.eval.harness import evaluate_suite

    tl = task_objs if task_objs is not None else paper7_tasks(tasks, all_choices)
    summary, recs = evaluate_suite(scorer, "paper7", task_objs=tl, limit=limit, n_boot=n_boot, seed=seed,
                                   subset_seed=subset_seed if (limit and subset_seed is not None and subset_seed >= 0)
                                   else None, progress=progress)
    out = {"tasks": {}}
    arrs = []
    for t in tl:
        s = summary["tasks"][t.name]
        key = "bpb" if t.kind == "lm" else "soft_bpb"
        v = s["metrics"][key]
        bits, nb = _item_arrays(t, recs[t.name])
        arrs.append((bits, nb))
        d = {"soft_bpb": v["value"], "ci95": [v["lo"], v["hi"]] if n_boot else None, "n": s["n"],
             "n_bytes": int(nb.sum())}
        if all_choices or t.kind == "lm":
            d["acc"] = s["metrics"]["acc"]["value"]
            if t.kind == "mc":
                d["acc_norm_char"] = s["metrics"]["acc_norm_char"]["value"]
        out["tasks"][t.name] = d
    vals = [d["soft_bpb"] for d in out["tasks"].values()]
    out["average"] = {"soft_bpb": float(np.mean(vals)), "ci95": average_ci(arrs, n_boot, seed),
                      "pooled_soft_bpb": float(sum(b.sum() for b, _ in arrs) / sum(n.sum() for _, n in arrs)),
                      "n_tasks": len(vals)}
    return out


# ====================================================================================================== CLI
def load_downstream_scorer(spec: str, boundary: str, device=None, dtype=None):
    from hnet.eval.run import load_run_checkpoint, model_kind

    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec.startswith("hf:"):
        name, _, rev = spec[3:].partition("@")
        sc = CanonicalHFScorer(name, boundary=boundary, revision=rev or None, device=device)
        return sc, {"type": "hf", "name": name, "kind": "token"}, None, None
    model, mcfg, meta = load_run_checkpoint(spec, device=device)
    kind = model_kind(mcfg)
    if kind == "byte":
        sc = ByteModelScorer(model, device=device)
    else:
        sc = CanonicalTokenScorer(model, boundary=boundary, device=device, add_bos=True)
    info = {"type": "run", "name": spec, "kind": kind, "checkpoint": meta["checkpoint"], "step": meta.get("step")}
    return sc, info, model, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default=None, help="comma list (default: the 7 paper7 tasks)")
    ap.add_argument("--limit", type=int, default=None, help="items per task (seeded random subset)")
    ap.add_argument("--subset-seed", type=int, default=0, help="with --limit: seeded random subset; -1 = first N items "
                    "(the trainer's eval.downstream convention)")
    ap.add_argument("--boundary", default="canonical", choices=["canonical", "lmeval"])
    ap.add_argument("--all-choices", action="store_true", help="also score distractors (accuracies)")
    ap.add_argument("--domains", default=None,
                    help="'paloma_like' (built-in ungated domains) or a per_source.py manifest")
    ap.add_argument("--domain-max-bytes", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: <run>/downstream_bpb.json")
    ap.add_argument("--progress", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    sc, info, model, meta = load_downstream_scorer(a.model, a.boundary, a.device)
    res = {"model": a.model, "info": info, "boundary": a.boundary if info["kind"] == "token" else "exact (bytes)",
           "limit": a.limit, "subset_seed": a.subset_seed if (a.limit and a.subset_seed >= 0) else None, "gold_only": not a.all_choices}
    res.update(evaluate_downstream(sc, a.tasks.split(",") if a.tasks else None, a.limit, a.subset_seed, a.n_boot,
                                   all_choices=a.all_choices, progress=a.progress))
    if hasattr(sc, "boundary_stats"):
        res["boundary_stats"] = dict(sc.boundary_stats)
    if a.domains:
        if model is None:
            print("[downstream_bpb] --domains needs a run checkpoint (per_source.py); skipped", file=sys.stderr)
        else:
            import importlib.util

            spec = importlib.util.spec_from_file_location("per_source", os.path.join(ROOT, "scripts", "eval",
                                                                                      "per_source.py"))
            ps = importlib.util.module_from_spec(spec)
            sys.modules["per_source"] = ps
            spec.loader.exec_module(ps)
            if a.domains == "paloma_like":
                import tempfile

                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump({"sources": PALOMA_LIKE}, f)
                sources = ps.load_manifest(f.name)
                os.unlink(f.name)
            else:
                sources = ps.load_manifest(a.domains)
            bsl, tsl = ps.run_seq_lens(meta)
            dom = ps.evaluate_sources(model, info["kind"], sources, device=sc.device,
                                      model_tokenizer=ps.model_tokenizer_of(meta), byte_seq_len=bsl or 8192,
                                      token_seq_len=tsl or 1792, max_bytes=a.domain_max_bytes or 1_000_000)
            vals = [s["bpb"] for s in dom["sources"].values() if s.get("bpb") is not None]
            dom["average_bpb"] = float(np.mean(vals)) if vals else None
            res["domains"] = dom
    res["seconds"] = time.time() - t0
    for t, d in res["tasks"].items():
        print(f"{t:16s} soft_bpb {d['soft_bpb']:.4f}  (n={d['n']})")
    print(f"{'average':16s} soft_bpb {res['average']['soft_bpb']:.4f}  ci95 {res['average']['ci95']}")
    if res.get("boundary_stats"):
        print(f"boundary: {res['boundary']} {res['boundary_stats']}")
    for n, s in res.get("domains", {}).get("sources", {}).items():
        print(f"[domain] {n:14s} bpb {s['bpb']:.4f}")
    out = a.out
    if out is None:
        ck = info.get("checkpoint")
        if ck is None:
            out = "downstream_bpb.json"
        else:
            out = os.path.join(os.path.dirname(os.path.dirname(ck)) if os.path.basename(os.path.dirname(ck)) == "ckpt"
                               else a.model, "downstream_bpb.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"-> {out} ({res['seconds']:.1f}s)", file=sys.stderr)
    return res


if __name__ == "__main__":
    main()
