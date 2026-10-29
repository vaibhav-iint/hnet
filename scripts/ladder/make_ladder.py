#!/usr/bin/env python
"""Compute-optimal scaling ladder (Cartesia "From Tokens to Bytes" Part I recipe): BPE Transformer rungs, their
1-stage H-Net conversions, and iso-FLOP sweep specs.

    python scripts/ladder/make_ladder.py                       # defaults: 4.3 bytes/token, vocab 32768
    python scripts/ladder/make_ladder.py --bytes-per-token 4.41 --vocab 32768 --dataset mix_v1
    python scripts/ladder/make_ladder.py --check               # exit 1 if the committed files are stale

Writes (and removes stale generated files of the same patterns):

* ``configs/ladder/models/{bpe,hnet,hnet_ws}_LNN.yaml``: model sections (+ ``matching:`` / ``ladder:`` metadata);
* ``configs/ladder/runs/ladder_{bpe,hnet}.yaml``: run files (data, Muon + WSD, eval); the spec swaps the model;
* ``configs/ladder/ladder.json``: every number below, plus the generator options (tests regenerate from them);
* ``configs/search/ladder_<C>_LNN.yaml``: one iso-FLOP spec per (budget, rung): bpe / hnet / hnet_ws at the same
  steps, batch and data (the fairness gate holds FLOPs/byte and bytes within 2% inside a spec, so rungs of one
  budget live in separate specs);
* ``configs/search/ladder_lr_<C>.yaml``: Muon-LR tuning sweep at the smallest budget (tuned fairness);
* ``docs/LADDER.md``: tables.

Recipe
------
BPE rung LNN (depth L): D = nearest multiple of 128 to 75 L (floor ``--d-floor`` 256); MHA with L heads of 128 (the
attention width 128 L may exceed D); SwiGLU width = nearest multiple of 128 to 3.6 D; global causal attention, full
RoPE, untied embeddings, vocab ``--vocab``.

H-Net rung: main network = the identical Transformer (same D, depth, heads, head_dim, MLP); 1 stage at target ratio
= bytes/token, pinned by the rank decider (``dc.decision: quantile``, ``end_mlp`` router); outer encoder/decoder =
Mamba-2 stacks at width d0. (d0, layers, d_state) are searched so that stage-0 FLOPs/byte (encoder + decoder +
router + residual_proj + byte LM head + chunk-vocab head) equal the BPE embedding + LM-head FLOPs/byte within
``--outer-tol`` (1%) and total training FLOPs/byte within ``--total-tol`` (1%); among those the most regular design
wins (d0 near clip(D/2, 192, 512), d_state near 128, symmetric, d0 a multiple of 64). The parameter budget left after
the outer network (BPE total - H-Net without tables) is filled with FLOP-free hashed tables at width d0: the chunk
vocabulary (``dc.chunk_vocab``, n_hash 2, ``--chunk-vocab-share`` of the rows) and byte n-gram tables
(``ngram_embed``, n = 3..8, the rest), bucket counts = primes. ``hnet_ws`` = ``hnet`` + the whitespace warm start
(``router_prior_weight`` 1.0 for the first 10% of steps).

FLOPs: hnet.flops SPEC convention (causal attention, LM head counted, embedding lookups free, backward = 2x).
Windows: ``--seq-tokens`` tokens <-> the nearest multiple of 64 to seq_tokens x bytes/token bytes, same number of
windows per step for both kinds, so a step covers the same text.

Iso-FLOP: for budget C and rung r, T = C / FLOPs_per_token(r); the Chinchilla prior N_opt has T = 20 N (N =
non-embedding backbone params). The ``--n-rungs`` rungs closest to the prior in log(T/N) are kept (rungs outside
[1, 400] tokens/param are never used). Steps = round(T / tokens_per_step); BPE trains steps x batch tokens, the
H-Nets steps x batch bytes (same steps, same data). Batch (windows of seq_tokens tokens) per budget: ``--batch``
(default 64 / 128 / 256 at 1e16 / 1e17 / 1e18: ~0.13 / 0.26 / 0.52M tokens, ~C^(1/3)).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from hnet.config import DenseConfig, HNetConfig  # noqa: E402
from hnet.flops import count_params, model_flops  # noqa: E402

GEN = "scripts/ladder/make_ladder.py"
MODELS_DIR = "configs/ladder/models"
RUNS_DIR = "configs/ladder/runs"
SEARCH_DIR = "configs/search"
DOC = "docs/LADDER.md"
MANIFEST = "configs/ladder/ladder.json"
NGRAM_NS = [3, 4, 5, 6, 7, 8]
VARIANTS = ("bpe", "hnet", "hnet_ws")

DEFAULTS = dict(
    bytes_per_token=4.3, vocab=32768, dataset="mix_v1", seq_tokens=2048, rungs=list(range(3, 13)),
    budgets=["1e16", "1e17", "1e18"], batch={"1e16": 64, "1e17": 128, "1e18": 256}, n_rungs=5,
    tokens_per_param=20.0, d_floor=256, outer_tol=0.01, total_tol=0.01, chunk_vocab_share=0.64,
    lr_grid=[0.01, 0.02, 0.04, 0.08], adam_lr=3.0e-3, gcs_root="gs://neat-dispatch-501718-a5-hnet/runs",
    micro_batch={"bpe": 16, "hnet": 8},
)


def nearest_multiple(x: float, m: int = 128) -> int:
    return int(math.floor(x / m + 0.5)) * m


def prev_prime(n: int) -> int:
    def is_prime(k):
        if k < 2:
            return False
        if k % 2 == 0:
            return k == 2
        r = int(math.isqrt(k))
        return all(k % p for p in range(3, r + 1, 2))

    while not is_prime(n):
        n -= 1
    return n


# ------------------------------------------------------------------------------------------ BPE rungs
def bpe_dims(L: int, d_floor: int = 256) -> dict:
    D = max(d_floor, nearest_multiple(75 * L))
    return dict(L=L, D=D, heads=L, head_dim=128, mlp=nearest_multiple(3.6 * D))


def bpe_model(L: int, vocab: int, d_floor: int = 256) -> dict:
    d = bpe_dims(L, d_floor)
    return dict(type="dense", variant="llama", vocab_size=vocab, d_model=d["D"], n_layers=L,
                d_intermediate=d["mlp"], num_heads=L, num_kv_heads=None, head_dim=128, rope_theta=10000.0,
                nope_every=0, tie_embeddings=False, norm_eps=1.0e-5, initializer_range=0.02)


def _cfg(model: dict):
    d = dict(model)
    kind = d.pop("type")
    return DenseConfig.from_dict(d) if kind == "dense" else HNetConfig.from_dict(d)


# ------------------------------------------------------------------------------------------ H-Net rungs
def hnet_model(bpe: dict, bpt: float, d0: int, n_enc: int, n_dec: int, d_state: int, cv_buckets: int,
               ng_buckets: int, prior: bool) -> dict:
    L, D = bpe["n_layers"], bpe["d_model"]
    return dict(
        type="hnet",
        arch_layout=[f"m{n_enc}", [f"T{L}"], f"m{n_dec}"],
        d_model=[d0, D],
        d_intermediate=[0, bpe["d_intermediate"]],
        vocab_size=256,
        tie_embeddings=False,
        ssm=dict(d_state=d_state, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256),
        # stage-0 attention fields are unused (Mamba-only outer network); the main network is the BPE rung's
        attn=dict(num_heads=[max(1, d0 // 64), L], num_kv_heads=None, head_dim=[None, 128],
                  rotary_emb_dim=[None, None], rope_theta=10000.0, window_size=[-1, -1], nope_every=0),
        dc=dict(target_ratios=[float(bpt)], ratio_loss_alpha=0.03, router="end_mlp", router_kwargs={},
                decision="quantile", chunker="select", dechunker="ema", ste=True, aux_loss="ratio",
                aux_loss_scope="local",
                chunk_vocab=[dict(buckets=int(cv_buckets), n_hash=2, max_len=16, pred_weight=0.5, n_neg=1024)],
                router_prior_weight=1.0 if prior else 0.0, router_prior_frac=0.1),
        ngram_embed=dict(ns=list(NGRAM_NS), buckets=int(ng_buckets)),
        norm_eps=1.0e-5,
        initializer_range=0.02,
        lr_multiplier="auto",
        lr_multiplier_n_gpt=float(bpt),
    )


def _stage0(cfg, seq_bytes):
    r = model_flops(cfg, seq_len=seq_bytes)
    return r["per_stage"][0]["flops_per_byte"], r["flops_per_byte"]


def match_outer(bpe: dict, bpt: float, seq_tokens: int, seq_bytes: int, outer_tol: float, total_tol: float) -> dict:
    """Search (d0, n_enc, n_dec, d_state); stage-0 FLOPs/byte are exactly linear in the layer count, so two
    evaluations per (d0, d_state) give every depth; the pick is re-evaluated in full."""
    bcfg = _cfg(bpe)
    rb = model_flops(bcfg, seq_len=seq_tokens, bytes_per_token=bpt)
    target = rb["per_stage"][0]["components"]["lm_head"]["flops_per_byte"] \
        + rb["per_stage"][0]["components"]["embedding"]["flops_per_byte"]
    total = rb["flops_per_byte"]
    D = bpe["d_model"]
    pref = min(D, max(192, min(512, D / 2)))
    # main-stage FLOPs/byte do not depend on the outer design
    probe = _cfg(hnet_model(bpe, bpt, min(D, 256), 1, 1, 128, 1024, 1024, False))
    main = model_flops(probe, seq_len=seq_bytes)["per_stage"][1]["flops_per_byte"]
    cands = []
    for d0 in range(128, D + 1, 32):
        for ds in range(64, 257, 16):
            f11, _ = _stage0(_cfg(hnet_model(bpe, bpt, d0, 1, 1, ds, 1024, 1024, False)), seq_bytes)
            f21, _ = _stage0(_cfg(hnet_model(bpe, bpt, d0, 2, 1, ds, 1024, 1024, False)), seq_bytes)
            layer, fixed = f21 - f11, f11 - 2 * (f21 - f11)
            for nd in range(1, 7):
                for ne in (nd, nd + 1):
                    s0 = fixed + (ne + nd) * layer
                    eo, et = s0 / target - 1, (s0 + main) / total - 1
                    if abs(eo) <= outer_tol and abs(et) <= total_tol:
                        cost = (abs(math.log2(d0 / pref)) + 0.3 * abs(math.log2(ds / 128)) + 0.3 * (ne != nd)
                                + 0.15 * (d0 % 64 != 0) + 0.5 * (nd < 2) + 0.1 * abs(nd - 2) + 5 * abs(eo))
                        cands.append((cost, d0, ne, nd, ds))
    if not cands:
        raise SystemExit(f"L{bpe['n_layers']:02d}: no outer design within {outer_tol:.1%} / {total_tol:.1%}; "
                         f"widen --outer-tol / --total-tol")
    _, d0, ne, nd, ds = min(cands)
    return dict(d0=d0, n_enc=ne, n_dec=nd, d_state=ds, target_outer=target, bpe_total=total)


def fill_tables(bpe: dict, bpt: float, outer: dict, share: float) -> dict:
    """Bucket counts so that H-Net total params = BPE total params (to < 0.1%)."""
    bcfg = _cfg(bpe)
    p_bpe = count_params(bcfg)
    base = dict(bpe=bpe, bpt=bpt, d0=outer["d0"], n_enc=outer["n_enc"], n_dec=outer["n_dec"],
                d_state=outer["d_state"], prior=False)
    d0 = outer["d0"]
    probe = count_params(_cfg(hnet_model(cv_buckets=1, ng_buckets=1, **base)))  # tables of one row each
    budget_rows = (p_bpe - (probe - (2 + len(NGRAM_NS)) * d0)) / d0
    if budget_rows <= 2 * 1024:
        raise SystemExit(f"L{bpe['n_layers']:02d}: outer network leaves no parameter budget for the tables")
    cv = prev_prime(int(share * budget_rows / 2))
    ng = prev_prime(int((budget_rows - 2 * cv) / len(NGRAM_NS)))
    return dict(cv_buckets=cv, ng_buckets=ng, bpe_params=p_bpe)


# ------------------------------------------------------------------------------------------ ladder
@dataclass
class Opts:
    bytes_per_token: float
    vocab: int
    dataset: str
    seq_tokens: int
    rungs: List[int]
    budgets: List[str]
    batch: Dict[str, int]
    n_rungs: int
    tokens_per_param: float
    d_floor: int
    outer_tol: float
    total_tol: float
    chunk_vocab_share: float
    lr_grid: List[float]
    adam_lr: float
    gcs_root: str
    micro_batch: Dict[str, int]

    @property
    def seq_bytes(self) -> int:
        return nearest_multiple(self.seq_tokens * self.bytes_per_token, 64)

    def to_json(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def rung_name(L: int) -> str:
    return f"L{L:02d}"


def build_rung(L: int, o: Opts) -> dict:
    bpe = bpe_model(L, o.vocab, o.d_floor)
    outer = match_outer(bpe, o.bytes_per_token, o.seq_tokens, o.seq_bytes, o.outer_tol, o.total_tol)
    tab = fill_tables(bpe, o.bytes_per_token, outer, o.chunk_vocab_share)
    kw = dict(d0=outer["d0"], n_enc=outer["n_enc"], n_dec=outer["n_dec"], d_state=outer["d_state"],
              cv_buckets=tab["cv_buckets"], ng_buckets=tab["ng_buckets"])
    hnet = hnet_model(bpe, o.bytes_per_token, prior=False, **kw)
    hnet_ws = hnet_model(bpe, o.bytes_per_token, prior=True, **kw)
    bcfg, hcfg = _cfg(bpe), _cfg(hnet)
    rb = model_flops(bcfg, seq_len=o.seq_tokens, bytes_per_token=o.bytes_per_token)
    rh = model_flops(hcfg, seq_len=o.seq_bytes)
    bc = rb["per_stage"][0]["components"]
    s0, sm = rh["per_stage"]
    c0 = s0["components"]
    outer_p = sum(c0[k]["params"] for k in ("encoder", "decoder", "router", "residual_proj"))
    table_p = c0["chunk_vocab"]["params"] + c0["ngram_embedding"]["params"]
    num = dict(
        bpe=dict(params=rb["params"], backbone=rb["params_nonembed"], emb_head=bc["embedding"]["params"]
                 + bc["lm_head"]["params"], flops_per_token=rb["flops_per_token"], flops_per_byte=rb["flops_per_byte"],
                 emb_head_flops_per_byte=bc["lm_head"]["flops_per_byte"] + bc["embedding"]["flops_per_byte"],
                 head_share=bc["lm_head"]["flops_per_byte"] / rb["flops_per_byte"]),
        hnet=dict(params=rh["params"], main=sm["params"], outer=outer_p, tables=table_p,
                  byte_emb_head=c0["embedding"]["params"] + c0["lm_head"]["params"],
                  flops_per_byte=rh["flops_per_byte"], stage0_flops_per_byte=s0["flops_per_byte"],
                  main_flops_per_byte=sm["flops_per_byte"],
                  byte_head_flops_per_byte=c0["lm_head"]["flops_per_byte"],
                  outer_share=s0["flops_per_byte"] / rh["flops_per_byte"],
                  byte_head_share=c0["lm_head"]["flops_per_byte"] / rh["flops_per_byte"],
                  flops_vs_bpe=rh["flops_per_byte"] / rb["flops_per_byte"] - 1,
                  outer_vs_emb_head=s0["flops_per_byte"] / (bc["lm_head"]["flops_per_byte"]
                                                            + bc["embedding"]["flops_per_byte"]) - 1,
                  params_vs_bpe=rh["params"] / rb["params"] - 1),
    )
    return dict(name=rung_name(L), L=L, dims=bpe_dims(L, o.d_floor), outer=kw, models=dict(bpe=bpe, hnet=hnet,
                hnet_ws=hnet_ws), numbers=num)


def budget_runs(C: float, rungs: List[dict], o: Opts, batch_windows: int) -> List[dict]:
    rows = []
    for r in rungs:
        n = r["numbers"]
        T = C / n["bpe"]["flops_per_token"]
        tpp = T / n["bpe"]["backbone"]
        rows.append(dict(rung=r["name"], tokens_prior=T, tokens_per_param=tpp,
                         dist=abs(math.log(tpp / o.tokens_per_param))))
    ok = [x for x in rows if 1.0 <= x["tokens_per_param"] <= 400.0]
    keep = {x["rung"] for x in sorted(ok, key=lambda x: x["dist"])[: o.n_rungs]}
    out = []
    by = {r["name"]: r for r in rungs}
    tok_step, byte_step = batch_windows * o.seq_tokens, batch_windows * o.seq_bytes
    for x in rows:
        if x["rung"] not in keep:
            continue
        n = by[x["rung"]]["numbers"]
        steps = max(1, round(x["tokens_prior"] / tok_step))
        tokens, nbytes = steps * tok_step, steps * byte_step
        out.append(dict(rung=x["rung"], steps=steps, batch_windows=batch_windows, tokens=tokens, bytes=nbytes,
                        tokens_per_param=tokens / n["bpe"]["backbone"],
                        flops_bpe=tokens * n["bpe"]["flops_per_token"], flops_hnet=nbytes * n["hnet"]["flops_per_byte"],
                        optimum_prior=x["dist"] == min(y["dist"] for y in rows if y["rung"] in keep)))
    return out


def build(o: Opts) -> dict:
    rungs = [build_rung(L, o) for L in o.rungs]
    budgets = {b: budget_runs(float(b), rungs, o, int(o.batch[b])) for b in o.budgets}
    return dict(options=o.to_json(), seq_bytes=o.seq_bytes, rungs=rungs, budgets=budgets)


# ------------------------------------------------------------------------------------------ emitters
HEADER = "# Generated by scripts/ladder/make_ladder.py -- do not edit by hand; rerun the generator (see docs/LADDER.md).\n"


class _Flow(list):
    pass


class _FlowMap(dict):
    pass


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(_Flow, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
_Dumper.add_representer(_FlowMap, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v.items(),
                                                                   flow_style=True))
_Dumper.add_representer(dict, lambda d, v: d.represent_mapping("tag:yaml.org,2002:map", v.items(), flow_style=False))
_Dumper.add_representer(list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=False))


def _styled(x):
    """Lists without mappings and small all-scalar mappings in flow style, everything else block style."""
    if isinstance(x, dict):
        vals = {k: _styled(v) for k, v in x.items()}
        flat = all(not isinstance(v, (dict, list)) or isinstance(v, _Flow) and all(
            not isinstance(e, (dict, list)) for e in v) for v in x.values())
        return _FlowMap(vals) if flat and len(x) <= 8 else vals
    if isinstance(x, list):
        vals = [_styled(v) for v in x]
        return vals if any(isinstance(v, dict) for v in x) else _Flow(vals)
    return x


def dump(obj) -> str:
    return yaml.dump(_styled(obj), Dumper=_Dumper, sort_keys=False, width=118)


def model_yaml(r: dict, variant: str, o: Opts) -> str:
    m = r["models"][variant]
    n = r["numbers"]
    if variant == "bpe":
        desc = (f"# {variant}_{r['name']}: BPE Transformer rung {r['name']} (D {m['d_model']}, {m['num_heads']} heads x 128,"
                f" SwiGLU {m['d_intermediate']}, vocab {m['vocab_size']}): {n['bpe']['params'] / 1e6:.1f}M params,"
                f" backbone {n['bpe']['backbone'] / 1e6:.1f}M.\n")
        matching = dict(seq_len_bytes=o.seq_bytes, seq_len_tokens=o.seq_tokens, bytes_per_token=o.bytes_per_token,
                        reference=None)
    else:
        ws = " + whitespace warm start (router_prior_weight 1.0, first 10% of steps)" if variant == "hnet_ws" else \
            " (prior-free)"
        desc = (f"# {variant}_{r['name']}: 1-stage H-Net conversion of bpe_{r['name']}{ws}. Main = the identical "
                f"Transformer;\n# outer m{r['outer']['n_enc']}/m{r['outer']['n_dec']} at d0 {r['outer']['d0']}, "
                f"d_state {r['outer']['d_state']}; FLOPs/byte {100 * n['hnet']['flops_vs_bpe']:+.2f}% and params "
                f"{100 * n['hnet']['params_vs_bpe']:+.2f}% vs bpe_{r['name']}.\n")
        matching = dict(seq_len_bytes=o.seq_bytes, seq_len_tokens=o.seq_tokens, bytes_per_token=o.bytes_per_token,
                        reference=f"bpe_{r['name']}")
    meta = dict(rung=r["name"], variant=variant, bytes_per_token=o.bytes_per_token, vocab=o.vocab)
    return HEADER + desc + dump(dict(model=m)) + "# Not part of the model (scripts/flops_report.py, tests).\n" + \
        dump(dict(matching=matching, ladder=meta))


def run_yaml(kind: str, o: Opts, first: str) -> str:
    hn = kind == "hnet"
    data = dict(dataset=o.dataset, bytes_per_token=o.bytes_per_token,
                seq_len=o.seq_bytes if hn else o.seq_tokens, micro_batch=o.micro_batch[kind])
    if hn:
        data["batch_bytes"] = 256 * o.seq_bytes
    else:
        data["batch_tokens"] = 256 * o.seq_tokens
    train = dict(out_dir=f"runs/ladder/{kind}", optimizer="muon",
                 muon=dict(adam_betas=[0.9, 0.95], adam_weight_decay=0.1),
                 schedule=dict(name="wsd", warmup=0.1, decay=0.2, decay_shape="inverse_sqrt", min_lr_ratio=0.0))
    if hn:
        train["train_bytes"] = 1.0e9
    else:
        train["train_tokens"] = 2.5e8
    ev = dict(val_byte_seq_len=o.seq_bytes, val_token_seq_len=o.seq_tokens)
    head = (f"# Ladder run file ({'H-Net, byte windows' if hn else 'BPE, token windows'}): {o.dataset}, "
            f"{o.seq_tokens} tokens <-> {o.seq_bytes} bytes per window ({o.bytes_per_token} bytes/token), Muon + WSD.\n"
            "# The model, batch, train length, eval cadence and LRs are set per job by configs/search/ladder_*.yaml.\n")
    body = dict(include=f"../../runs/{'_core' if hn else '_core_bpe'}.yaml",
                model=dict(include=f"../models/{kind}_{first}.yaml"), data=data, train=train, eval=ev)
    return HEADER + head + dump(body)


def _common_sets(o: Opts, steps: int, lr_ref: Optional[str]) -> List[str]:
    every = max(25, steps // 10)
    s = ["train.optimizer=muon"]
    if lr_ref:
        s.append(f"train.muon.lr={lr_ref}")
    s += ["train.muon.lr_embed=${adam_lr}", "train.muon.lr_head=${adam_lr}", "train.muon.lr_other=${adam_lr}",
          # Muon momentum ramp (default 300 steps) as a fixed fraction of the run, like the WSD phases
          f"train.muon.momentum_warmup_steps={min(300, max(1, steps // 10))}",
          f"eval.every={every}", f"train.ckpt_every={max(every, steps // 4)}"]
    return s


def _runs(r: dict, run: dict, o: Opts) -> List[dict]:
    W, steps = run["batch_windows"], run["steps"]
    out = []
    for v in VARIANTS:
        kind = "bpe" if v == "bpe" else "hnet"
        mb = min(W, o.micro_batch[kind])
        sets = [f"data.micro_batch={mb}"]
        if kind == "bpe":
            sets += [f"data.batch_tokens={W * o.seq_tokens}", f"train.train_tokens={run['tokens']}"]
        else:
            sets += [f"data.batch_bytes={W * o.seq_bytes}", f"train.train_bytes={run['bytes']}"]
        out.append(dict(name=v, config=f"{RUNS_DIR}/ladder_{kind}.yaml",
                        model=f"{MODELS_DIR}/{v}_{r['name']}.yaml", set=sets))
    return out


def iso_spec(b: str, run: dict, r: dict, o: Opts) -> str:
    name = f"ladder_{b}_{r['name']}"
    n = r["numbers"]
    head = (f"# Iso-FLOP ladder, budget {b} training FLOPs, rung {r['name']} (D {r['dims']['D']}): BPE vs its 1-stage "
            f"H-Net conversions\n# (prior-free / whitespace warm start) at the same steps ({run['steps']}) x batch "
            f"({run['batch_windows']} windows = {run['batch_windows'] * o.seq_tokens} tokens = "
            f"{run['batch_windows'] * o.seq_bytes} bytes) = the same data.\n"
            f"# BPE {run['tokens'] / 1e9:.3f}B tokens ({run['tokens_per_param']:.1f} tokens/backbone param), "
            f"{run['flops_bpe']:.3g} FLOPs; H-Net {run['bytes'] / 1e9:.3f}B bytes, {run['flops_hnet']:.3g} FLOPs "
            f"({100 * n['hnet']['flops_vs_bpe']:+.2f}% FLOPs/byte).\n"
            "# LRs: the blog transfers BPE's hyperparameters to the H-Net: every run uses BPE's best Muon LR from the\n"
            f"# tuning sweep (--params-file runs/search/ladder_lr_{o.budgets[0]}/report/best_params.yaml).\n")
    spec = dict(name=name, out_root=f"runs/search/ladder/{b}/{r['name']}", gcs=f"{o.gcs_root}/ladder/{b}/{r['name']}",
                seeds=[0], params={"best_mlr": {"bpe": None}, "adam_lr": o.adam_lr},
                params_note=(f"best_mlr.bpe = BPE's best Muon LR from ladder_lr_{o.budgets[0]} (report/best_params.yaml);"
                             " adam_lr = Adam-part LR (embeddings, heads, norms), fixed, not tuned yet"),
                fairness="strict", set=_common_sets(o, run["steps"], "${best_mlr.bpe}"),
                report=dict(baseline="bpe", labels={"bpe": f"BPE {r['name']}", "hnet": f"H-Net {r['name']} (prior-free)",
                                                    "hnet_ws": f"H-Net {r['name']} + whitespace warm start"}),
                runs=_runs(r, run, o))
    return HEADER + head + dump(spec)


def lr_spec(b: str, run: dict, r: dict, o: Opts) -> str:
    name = f"ladder_lr_{b}"
    head = (f"# Muon-LR tuning for the ladder at the smallest budget ({b} FLOPs), rung {r['name']} (the Chinchilla-prior "
            "optimum),\n# BPE and both H-Nets, same steps/batch/data as configs/search/ladder_"
            f"{b}_{r['name']}.yaml. report.py writes best_params.yaml\n# (best_mlr: {{bpe: ..., hnet: ..., hnet_ws: ...}})"
            " for the iso-FLOP specs' --params-file (they use best_mlr.bpe for all runs).\n")
    spec = dict(name=name, out_root=f"runs/search/{name}", gcs=f"{o.gcs_root}/{name}", seeds=[0],
                params={"adam_lr": o.adam_lr}, params_note="adam_lr = Adam-part LR (embeddings, heads, norms), fixed",
                fairness="tuned", set=_common_sets(o, run["steps"], None), grid={"train.muon.lr": list(o.lr_grid)},
                grid_alias={"train.muon.lr": "mlr"},
                report=dict(baseline="bpe", labels={"bpe": f"BPE {r['name']}", "hnet": f"H-Net {r['name']} (prior-free)",
                                                    "hnet_ws": f"H-Net {r['name']} + whitespace warm start"}),
                runs=_runs(r, run, o))
    return HEADER + head + dump(spec)


def _M(x):
    return f"{x / 1e6:.1f}"


def doc(lad: dict, o: Opts) -> str:
    rungs = lad["rungs"]
    L = ["# Compute-optimal ladder: BPE Transformer vs 1-stage H-Net (blog Part I recipe)", "",
         f"Generated by `{GEN}` (options: {o.bytes_per_token} bytes/token, vocab {o.vocab}, dataset `{o.dataset}`, "
         f"{o.seq_tokens} tokens <-> {lad['seq_bytes']} bytes per window). Do not edit by hand; rerun:", "",
         "```",
         f"python {GEN} --bytes-per-token <measured> --vocab {o.vocab} --dataset {o.dataset}",
         f"python {GEN} --check   # tests/test_ladder.py fails when the committed files are stale",
         "```", "",
         "## Recipe", "",
         "* **BPE rung LNN** (Cartesia Part I): depth L; D = nearest multiple of 128 to 75 L (floor 256: L03/L04);"
         " MHA with L heads of 128 (attention width 128 L, larger than D); SwiGLU width = nearest multiple of 128 to"
         f" 3.6 D; global causal attention, full RoPE (theta 1e4), untied embeddings, vocab {o.vocab}.",
         "* **H-Net rung** (`hnet`): main network = the identical Transformer (same D, depth, heads x head_dim, MLP;"
         f" runs at 1/{o.bytes_per_token} positions per byte = one per BPE token); 1 stage, target ratio = bytes/token,"
         " pinned by the rank decider (`dc.decision: quantile`) with the `end_mlp` router; Mamba-2 encoder/decoder at"
         " width d0 sized so stage-0 FLOPs/byte (encoder + decoder + router + residual_proj + byte LM head + chunk-vocab"
         " head) equal BPE's embedding + LM-head FLOPs/byte (within 1%); the parameter budget left after the outer"
         " network is filled with FLOP-free hashed tables at width d0: the chunk vocabulary (`dc.chunk_vocab`, 2 hashes,"
         f" {o.chunk_vocab_share:.0%} of the rows, next-chunk objective pred_weight 0.5 / 1024 negatives) and byte"
         f" n-gram tables (`ngram_embed`, n = {NGRAM_NS[0]}..{NGRAM_NS[-1]}). `router_prior_weight` 0.",
         "* **`hnet_ws`**: `hnet` + whitespace warm start (`router_prior_weight` 1.0, `router_prior_frac` 0.1)."
         " Same params and FLOPs.",
         "* **FLOPs**: `hnet.flops` SPEC convention: causal attention, LM head counted, embedding / table lookups free,"
         " backward = 2 x forward. Stage-0 FLOPs include the training-only chunk-vocab head.",
         "* **Training**: Muon (hidden matrices) + Adam (embeddings, tables, heads, norms, router), WSD 10% warmup /"
         " 20% 1-sqrt decay (`configs/runs/_core.yaml`), Muon momentum ramp 0.85 -> 0.95 over min(300, 10% of"
         " steps), bf16. Muon LR from the tuning sweep (blog: BPE's"
         " hyperparameters transferred to the H-Net), Adam-part LR"
         f" {o.adam_lr:g} (fixed). Same steps, windows per step and therefore the same text for all three"
         " architectures of a rung.", "",
         "## Rungs", "",
         "BPE: params = total, backbone = non-embedding (no embedding, no LM head). FLOPs are training FLOPs"
         " (3 x forward). H-Net: params vs BPE total; FLOPs/byte vs BPE; outer = stage 0 (encoder, decoder, router,"
         " residual_proj, byte head, chunk-vocab head) vs BPE embedding + head.", "",
         "| rung | D | heads x 128 | MLP | BPE params (M) | backbone (M) | emb+head (M) | BPE MFLOPs/token |"
         " BPE MFLOPs/byte | BPE head share |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rungs:
        d, n = r["dims"], r["numbers"]["bpe"]
        L.append(f"| {r['name']} | {d['D']} | {d['heads']} | {d['mlp']} | {_M(n['params'])} | {_M(n['backbone'])} |"
                 f" {_M(n['emb_head'])} | {n['flops_per_token'] / 1e6:.1f} | {n['flops_per_byte'] / 1e6:.2f} |"
                 f" {100 * n['head_share']:.1f}% |")
    L += ["", "| rung | H-Net layout | d0 | d_state | chunk-vocab buckets (x2) | n-gram buckets (x6) | params (M) |"
          " vs BPE | outer (M) | tables (M) | main (M) | MFLOPs/byte | vs BPE | outer vs emb+head |"
          " outer share | byte-head share |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rungs:
        h, ou = r["numbers"]["hnet"], r["outer"]
        L.append(f"| {r['name']} | m{ou['n_enc']} / T{r['L']} / m{ou['n_dec']} | {ou['d0']} | {ou['d_state']} |"
                 f" {ou['cv_buckets']} | {ou['ng_buckets']} | {_M(h['params'])} | {100 * h['params_vs_bpe']:+.2f}% |"
                 f" {_M(h['outer'])} | {_M(h['tables'])} | {_M(h['main'])} | {h['flops_per_byte'] / 1e6:.2f} |"
                 f" {100 * h['flops_vs_bpe']:+.2f}% | {100 * h['outer_vs_emb_head']:+.2f}% |"
                 f" {100 * h['outer_share']:.1f}% | {100 * h['byte_head_share']:.1f}% |")
    L += ["", "\"main\" includes the D - d0 padding vector; \"outer\" = encoder + decoder + router + residual_proj;"
          " byte embedding + head = 2 x 256 x d0. The main stage's FLOPs/byte differ from BPE's body FLOPs/byte only"
          f" by the attention context ({lad['seq_bytes']} / {o.bytes_per_token} = "
          f"{lad['seq_bytes'] / o.bytes_per_token:.0f} main positions vs {o.seq_tokens} tokens).", "",
          "## Iso-FLOP sweeps", "",
          f"Chinchilla prior: T = {o.tokens_per_param:g} N (N = backbone params); per budget the {o.n_rungs} rungs"
          " closest to the prior in log(T/N). Each (budget, rung) is one spec with three runs (bpe, hnet, hnet_ws);"
          " the fairness gate (`scripts/search/fairness.py`, strict) holds FLOPs/byte and bytes within 2% inside a"
          " spec. Steps = round(T / tokens per step); the BPE run trains steps x batch tokens, the H-Nets steps x"
          " batch bytes. Specs need `--params-file` from the LR sweep (or `--param best_mlr.bpe=...`).", ""]
    grand = 0.0
    for b, runs in lad["budgets"].items():
        tot = sum(x["flops_bpe"] + 2 * x["flops_hnet"] for x in runs)
        grand += tot
        W = runs[0]["batch_windows"] if runs else 0
        L += [f"### {b} FLOPs: {len(runs)} rungs x 3 = {3 * len(runs)} runs, {tot:.3g} FLOPs in total", "",
              f"Batch {W} windows = {W * o.seq_tokens / 1e6:.3f}M tokens = {W * lad['seq_bytes'] / 1e6:.3f}M bytes per"
              " step.", "",
              "| spec | rung | steps | BPE tokens (B) | H-Net bytes (B) | tokens / backbone param | BPE FLOPs |"
              " H-Net FLOPs | prior optimum |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for x in runs:
            L.append(f"| `ladder_{b}_{x['rung']}` | {x['rung']} | {x['steps']} | {x['tokens'] / 1e9:.3f} |"
                     f" {x['bytes'] / 1e9:.3f} | {x['tokens_per_param']:.1f} | {x['flops_bpe']:.3g} |"
                     f" {x['flops_hnet']:.3g} | {'yes' if x['optimum_prior'] else ''} |")
        L.append("")
    b0 = o.budgets[0]
    lr_run = next(x for x in lad["budgets"][b0] if x["optimum_prior"])
    lr_tot = len(o.lr_grid) * (lr_run["flops_bpe"] + 2 * lr_run["flops_hnet"])
    grand += lr_tot
    L += [f"### LR tuning: `ladder_lr_{b0}`", "",
          f"Rung {lr_run['rung']} at {b0} (the prior optimum), Muon LR grid {o.lr_grid} (tuned fairness), bpe / hnet /"
          f" hnet_ws: {3 * len(o.lr_grid)} runs, {lr_tot:.3g} FLOPs. `report.py` writes `best_params.yaml`"
          " (`best_mlr: {bpe, hnet, hnet_ws}`); the iso-FLOP specs read `best_mlr.bpe`.", "",
          f"**Everything: {grand:.3g} training FLOPs** (iso-FLOP sweeps + LR sweep, one seed).", "",
          "## Deviations from the blog and open issues", "",
          f"* Vocabulary {o.vocab} in-corpus BPE (blog: 128k, 4.57 bytes/token); bytes/token {o.bytes_per_token} is a"
          " placeholder until measured. The fairness gate reads the measured value from `hnet/data/bpt_stats.json`"
          f" (dataset `{o.dataset}`), so it fails until the dataset is measured and the ladder regenerated with that"
          " value (`--bytes-per-token`).",
          "* Rungs L03..L12 (blog: L08..L28); D floor 256 makes L03/L04 share D = 256.",
          "* Outer network design is not published: Mamba-2 stacks, width and d_state searched per rung for FLOP"
          " matching (so they vary irregularly between rungs); parameter matching uses chunk-vocabulary and n-gram"
          " tables (ours, not the blog's).",
          "* Matching uses target compression (ratio = bytes/token); the rank decider pins it, but realized"
          " bytes/position should be checked in the runs (`hnet.flops.model_flops(..., bytes_per_position=...)`).",
          "* Batch and LR scaling laws are not fitted (blog: fitted on selected rungs). Batch follows ~C^(1/3) from"
          " 0.5M tokens at 1e18; the Muon LR is tuned once at the smallest budget and transferred; the Adam-part LR is"
          " fixed. Small-rung / small-budget runs have few steps (see tables).",
          "* SPEC FLOPs convention (standard causal attention), not the blog's Fig. 8 convention (half-cost causal).",
          "* Data: the 70/20/5/5 mix and 32k tokenizer are built elsewhere; the BPE loader still needs the tokenizer's"
          " EOT id (`data.eot`) and byte lengths for bpb.", ""]
    return "\n".join(L)


def generate(o: Opts) -> Dict[str, str]:
    lad = build(o)
    files: Dict[str, str] = {}
    by = {r["name"]: r for r in lad["rungs"]}
    for r in lad["rungs"]:
        for v in VARIANTS:
            files[f"{MODELS_DIR}/{v}_{r['name']}.yaml"] = model_yaml(r, v, o)
    first = lad["rungs"][0]["name"]
    for kind in ("bpe", "hnet"):
        files[f"{RUNS_DIR}/ladder_{kind}.yaml"] = run_yaml(kind, o, first)
    for b, runs in lad["budgets"].items():
        for x in runs:
            files[f"{SEARCH_DIR}/ladder_{b}_{x['rung']}.yaml"] = iso_spec(b, x, by[x["rung"]], o)
    b0 = o.budgets[0]
    x0 = next(x for x in lad["budgets"][b0] if x["optimum_prior"])
    files[f"{SEARCH_DIR}/ladder_lr_{b0}.yaml"] = lr_spec(b0, x0, by[x0["rung"]], o)
    slim = dict(lad, rungs=[{k: v for k, v in r.items() if k != "models"} for r in lad["rungs"]])
    files[MANIFEST] = json.dumps(slim, indent=1) + "\n"
    files[DOC] = doc(lad, o)
    return files


GENERATED = [re.compile(r"^configs/ladder/models/(bpe|hnet|hnet_ws)_L\d+\.yaml$"),
             re.compile(r"^configs/ladder/runs/ladder_(bpe|hnet)\.yaml$"),
             re.compile(r"^configs/search/ladder_[A-Za-z0-9_.+-]+\.yaml$")]


def existing_generated(root: Path) -> List[str]:
    out = []
    for d in (MODELS_DIR, RUNS_DIR, SEARCH_DIR):
        p = root / d
        if p.exists():
            for f in sorted(p.iterdir()):
                rel = f"{d}/{f.name}"
                if any(g.match(rel) for g in GENERATED):
                    out.append(rel)
    return out


def opts_from(d: dict) -> Opts:
    return Opts(**{**DEFAULTS, **d})


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bytes-per-token", type=float, default=DEFAULTS["bytes_per_token"])
    ap.add_argument("--vocab", type=int, default=DEFAULTS["vocab"])
    ap.add_argument("--dataset", default=DEFAULTS["dataset"], help="data.dataset of the ladder runs")
    ap.add_argument("--seq-tokens", type=int, default=DEFAULTS["seq_tokens"])
    ap.add_argument("--rungs", default="3-12", help="depth range a-b (inclusive)")
    ap.add_argument("--budgets", nargs="+", default=DEFAULTS["budgets"])
    ap.add_argument("--batch", nargs="+", default=[f"{k}={v}" for k, v in DEFAULTS["batch"].items()],
                    help="windows per step per budget, e.g. 1e16=64 1e17=128 1e18=256")
    ap.add_argument("--n-rungs", type=int, default=DEFAULTS["n_rungs"])
    ap.add_argument("--tokens-per-param", type=float, default=DEFAULTS["tokens_per_param"])
    ap.add_argument("--adam-lr", type=float, default=DEFAULTS["adam_lr"])
    ap.add_argument("--lr-grid", type=float, nargs="+", default=DEFAULTS["lr_grid"])
    ap.add_argument("--root", default=str(REPO), help="repository root to write into")
    ap.add_argument("--check", action="store_true", help="compare with the files on disk; exit 1 if stale")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    lo, _, hi = a.rungs.partition("-")
    batch = {k: int(v) for k, _, v in (s.partition("=") for s in a.batch)}
    missing = [b for b in a.budgets if b not in batch]
    if missing:
        raise SystemExit(f"--batch has no entry for budgets {missing}")
    o = opts_from(dict(bytes_per_token=a.bytes_per_token, vocab=a.vocab, dataset=a.dataset, seq_tokens=a.seq_tokens,
                       rungs=list(range(int(lo), int(hi or lo) + 1)), budgets=list(a.budgets),
                       batch={b: batch[b] for b in a.budgets}, n_rungs=a.n_rungs, tokens_per_param=a.tokens_per_param,
                       adam_lr=a.adam_lr, lr_grid=list(a.lr_grid)))
    files = generate(o)
    root = Path(a.root)
    stale = [p for p in existing_generated(root) if p not in files]
    if a.check:
        bad = [p for p, t in files.items() if not (root / p).exists() or (root / p).read_text() != t] + stale
        for p in bad:
            print(f"[ladder] stale: {p}")
        return 1 if bad else 0
    for p in stale:
        (root / p).unlink()
        print(f"[ladder] removed stale {p}")
    for p, t in files.items():
        (root / p).parent.mkdir(parents=True, exist_ok=True)
        (root / p).write_text(t)
    print(f"[ladder] wrote {len(files)} files ({len([p for p in files if p.startswith(SEARCH_DIR)])} specs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
