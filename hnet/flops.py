"""Analytic parameter and FLOP counts for H-Nets (HNetConfig) and dense baselines (DenseConfig).

Everything is computed from the config alone; no model is instantiated.

    from hnet.flops import model_flops, count_params
    r = model_flops(cfg, bytes_per_position=[1.0, 3.0, 9.0], seq_len=8192, training=True)
    r["flops_per_byte"], r["per_stage"], r["params"], r["params_nonembed"]

Counting (paper App. B, Chinchilla-style, plus Mamba-2 terms), forward FLOPs per position:

* every matmul: 2 x in x out (QKV with GQA, attention out-proj, SwiGLU fc1/fc2, Mamba-2 in/out-proj,
  router q/k, residual_proj, LM head);
* attention quadratic part: logits 2*ctx*(h*hd), softmax 3*h*ctx, scores@V 2*ctx*(h*hd), where ctx is the
  number of keys a query sees on average: causal -> (L+1)/2, sliding window w (keys in [t-w, t]) -> the exact
  average of min(t+1, w+1); "full" convention -> L (or w+1);
* Mamba-2: SSD 2*3*d_inner*d_state, depthwise conv 2*conv_dim*d_conv, gating 5*d_model;
* SwiGLU gating 5*d_model;
* backward = 2 x forward (training=True multiplies by 3).

Norms, rotary and the EMA dechunker are ignored (O(d) per position). The embedding lookup is not counted
by default (blog / SPEC convention); App. B literally counts it as a 2*V*d matmul (``count_embedding``).

H-Net stages: stage s runs at 1 / bytes_per_position[s] positions per byte, with sequence length
seq_len / bytes_per_position[s]. Default bytes_per_position = cumulative target ratios
[1, N0, N0*N1, ...]; pass realized values once measured. FLOPs are attributed per stage the way the
Cartesia blog (Fig. 8) does: stage 0 holds embedding, LM head, E0/D0, router 0 and residual_proj 0; the main
stage holds the main network.

Calibration against published numbers lives in tests/test_flops.py and configs/MATCHING.md.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Optional, Sequence

from hnet.config import DenseConfig, HNetConfig

GIGA = 1e9


# ----------------------------------------------------------------------------- conventions
@dataclass(frozen=True)
class FlopsConvention:
    """Switches for the conventions that differ between sources."""

    attn: str = "causal"  # "causal": average over causal context; "full": every query sees L keys (App. B literal)
    attn_windows: bool = True  # honour sliding windows (False: treat SWA layers as global)
    count_embedding: bool = False  # App. B counts 2*V*d for the embedding; blog / SPEC do not
    count_lm_head: bool = True
    count_dc: bool = True  # router q/k projections of every H-Net stage
    count_residual_proj: bool = True  # residual_proj (d x d) of every H-Net stage (blog Fig. 8 omits it)
    mamba_out_proj: str = "exact"  # "exact": 2*d_inner*d ; "paper": App. B literal 2*d*d
    count_elementwise: bool = True  # softmax 3*h*ctx, gating 5*d
    attn_quadratic_scale: float = 1.0  # extra factor on the quadratic attention terms (calibration knob)
    backward_factor: float = 2.0


# SPEC / PLAN (used for all our matching): causal attention, embedding not counted, LM head counted.
SPEC = FlopsConvention()
# Paper Table 2 (eval GFLOPs/byte): App. B taken literally -- full (non-causal) attention, embedding counted.
PAPER_APPB = FlopsConvention(attn="full", count_embedding=True)
# Cartesia blog Fig. 8 (SmolLM3 vs 3.67B H-Net): causal attention counted at half the usual causal cost
# (equivalently an average context of L/4, e.g. document masking), residual_proj not counted.
BLOG_FIG8 = FlopsConvention(attn_quadratic_scale=0.5, count_residual_proj=False)
CONVENTIONS = {"spec": SPEC, "paper_appB": PAPER_APPB, "blog_fig8": BLOG_FIG8}


def _conv(convention) -> FlopsConvention:
    if convention is None:
        return SPEC
    if isinstance(convention, str):
        return CONVENTIONS[convention]
    if isinstance(convention, dict):
        return replace(SPEC, **convention)
    return convention


def round_up(x: int, m: int = 128) -> int:
    return (int(x) + m - 1) // m * m


# ----------------------------------------------------------------------------- layer helpers
_LAYER_RE = re.compile(r"([A-Za-z])(\d+)")


def parse_layout(code: str):
    """'T1m4' -> [('t', True), ('m', False) x4]: (mixer code lower-case, has_mlp)."""
    out = []
    for c, n in _LAYER_RE.findall(code):
        out += [(c.lower(), c.isupper())] * int(n)
    return out


def _mamba_dims(d: int, ssm) -> dict:
    di = ssm.expand * d
    nh = di // ssm.headdim
    gn = ssm.ngroups * ssm.d_state
    conv_dim = di + 2 * gn
    return dict(di=di, nh=nh, gn=gn, conv_dim=conv_dim)


def mamba2_params(d: int, ssm) -> int:
    m = _mamba_dims(d, ssm)
    in_proj = d * (2 * m["di"] + 2 * m["gn"] + m["nh"])
    conv = m["conv_dim"] * ssm.d_conv + m["conv_dim"]  # weight + bias
    per_head = 3 * m["nh"]  # dt_bias, A_log, D
    norm = m["di"]  # gated RMSNorm weight
    out_proj = m["di"] * d
    return in_proj + conv + per_head + norm + out_proj


def mamba2_flops(d: int, ssm, cv: FlopsConvention) -> float:
    m = _mamba_dims(d, ssm)
    f = 2 * d * (2 * m["di"] + 2 * m["gn"] + m["nh"])  # XZ + BC dt projections
    f += 2 * m["conv_dim"] * ssm.d_conv  # depthwise conv
    f += 2 * 3 * m["di"] * ssm.d_state  # SSD
    f += 2 * (m["di"] if cv.mamba_out_proj == "exact" else d) * d
    if cv.count_elementwise:
        f += 5 * d
    return f


def _attn_dims(d: int, a: dict) -> dict:
    nh = a["num_heads"]
    nkv = a.get("num_kv_heads") or nh
    hd = a.get("head_dim") or d // nh  # explicit head_dim: heads * head_dim may differ from d
    return dict(nh=nh, nkv=nkv, hd=hd, q=nh * hd, kv=nkv * hd)


def attn_params(d: int, a: dict) -> int:
    m = _attn_dims(d, a)
    p = d * (m["q"] + 2 * m["kv"]) + m["q"] * d
    if a.get("qkv_bias"):
        p += m["q"] + 2 * m["kv"]
    if a.get("out_bias"):
        p += d
    return p


def attn_context(L: float, window: int, cv: FlopsConvention) -> float:
    """Average number of keys per query for a sequence of L positions."""
    w1 = None if (window is None or window < 0 or not cv.attn_windows) else window + 1
    if cv.attn == "full":
        return L if w1 is None else min(L, w1)
    if cv.attn != "causal":
        raise ValueError(cv.attn)
    if w1 is None or w1 >= L:
        return (L + 1) / 2
    return (w1 * (w1 + 1) / 2 + (L - w1) * w1) / L


def attn_flops(d: int, a: dict, L: float, cv: FlopsConvention) -> float:
    m = _attn_dims(d, a)
    ctx = attn_context(L, a.get("window_size", -1), cv)
    f = 2 * d * (m["q"] + 2 * m["kv"]) + 2 * m["q"] * d
    quad = 2 * ctx * m["q"] * 2
    if cv.count_elementwise:
        quad += 3 * m["nh"] * ctx
    return f + quad * cv.attn_quadratic_scale


def mlp_params(d: int, f: int) -> int:
    return 3 * d * round_up(f) if f else 0


def mlp_flops(d: int, f: int, cv: FlopsConvention) -> float:
    if not f:
        return 0.0
    return 6 * d * round_up(f) + (5 * d if cv.count_elementwise else 0)


# Router parameter / FLOP counts. Routers are tiny and their implementations evolve, so the exact parameter count is
# taken from the registered implementation built on the meta device (no memory, no weights) when hnet.model is
# importable; FLOPs per position = 2 x params (all router params are matmul weights, biases are negligible).
# ROUTER_COUNTS is the fallback without hnet.model: name -> fn(d, kwargs) -> params.
def _mlp_router(default_hidden):
    def f(d, kw):
        h = kw.get("d_hidden") or default_hidden(d)
        return d * h + h + h + 1

    return f


ROUTER_COUNTS = {
    "cosine_lookahead": lambda d, kw: 2 * d * d,
    "end_cosine_shift": lambda d, kw: 2 * d * d,
    "causal_end_of_chunk": _mlp_router(lambda d: max(d // 4, 16)),
    "end_mlp": _mlp_router(lambda d: max(d // 4, 16)),
    "end_linear": lambda d, kw: d + 1,
    "fixed_pool": lambda d, kw: 0,
    "space_like": lambda d, kw: 0,
    "external": lambda d, kw: 0,
}
_ROUTER_CACHE: dict = {}


def _router_from_registry(name, d, kwargs, target_ratio):
    key = (name, d, repr(sorted((kwargs or {}).items())), target_ratio)
    if key not in _ROUTER_CACHE:
        import torch

        import hnet.model  # noqa: F401  (registers the built-in routers)
        from hnet.registry import build

        with torch.device("meta"):
            r = build("router", name, d, target_ratio=target_ratio, **(kwargs or {}))
        _ROUTER_CACHE[key] = sum(x.numel() for x in r.parameters())
    return _ROUTER_CACHE[key]


def _chunker_counts(name: str, d: int, kwargs: dict, target_ratio: float):
    """(params, forward FLOPs per outer position) of a chunker; ``select`` has neither."""
    import torch

    import hnet.model  # noqa: F401
    from hnet.registry import build

    with torch.device("meta"):
        c = build("chunker", name, d, **(kwargs or {}))
    if not isinstance(c, torch.nn.Module):
        return 0, 0.0
    fl = c.flops_per_position(d, target_ratio, **(kwargs or {}))
    return sum(x.numel() for x in c.parameters()), float(fl)


def _router_counts(name: str, d: int, kwargs: dict, target_ratio=None):
    p = _router_from_registry(name, d, kwargs, target_ratio)
    if p is None:
        p = ROUTER_COUNTS[name](d, kwargs or {}) if name in ROUTER_COUNTS else 2 * d * d
    return p, 2 * p


# ----------------------------------------------------------------------------- isotropic stacks
def mixer_counts(mixer: str, d: int, kwargs: dict, L: float, cv: FlopsConvention, cfg_ssm=None):
    """(params, forward FLOPs per position) of one sequence mixer.

    ``mamba2`` and ``attn`` use the closed forms above. Any other registered mixer must provide a classmethod
    ``flops_per_position(d_model, L, cv, **kwargs) -> float`` (forward FLOPs per position at sequence length L,
    honouring ``cv``); its parameter count is taken from the registered class built on the meta device.
    """
    if mixer == "mamba2":
        return mamba2_params(d, cfg_ssm), mamba2_flops(d, cfg_ssm, cv)
    if mixer == "attn":
        return attn_params(d, kwargs), attn_flops(d, kwargs, L, cv)
    import torch

    from hnet import registry

    cls = registry.get("mixer", mixer)
    hook = getattr(cls, "flops_per_position", None)
    if hook is None:
        raise NotImplementedError(f"mixer {mixer!r} has no flops_per_position classmethod")
    with torch.device("meta"):
        m = cls(d, **kwargs)
    return sum(x.numel() for x in m.parameters()), float(hook(d, L, cv, **kwargs))


def _block_counts(mixer, mlp, d, f, kwargs, L, cv, cfg_ssm=None):
    p, fl = mixer_counts(mixer, d, kwargs, L, cv, cfg_ssm)
    p += d  # norm1
    if mlp and f:
        p += d + mlp_params(d, f)  # norm2 + SwiGLU
        fl += mlp_flops(d, f, cv)
    return p, fl


def attn_res_counts(height: int, block: int, d: int, cv: FlopsConvention):
    """(params, forward FLOPs/position) of Block Attention Residuals on a stack of ``height`` sub-layers
    (hnet.model.isotropic): height + 1 depth-attention operators, each a query vector and an RMSNorm weight. Operator
    j (the output one is j = height) attends over 1 + ceil(j / block) sources; per source a 2d logit dot and a 2d
    weighted sum, plus the RMSNorm and softmax elementwise work when counted. One source is a pass-through."""
    if block <= 0:
        return 0, 0.0
    fl = 0.0
    for j in range(height + 1):
        n = 1 + -(-j // block)
        if n > 1:
            fl += 4 * d * n + ((4 * d + 3) * n if cv.count_elementwise else 0)
    return 2 * d * (height + 1), fl


def _iso(cfg: HNetConfig, s: int, code: str, L: float, cv: FlopsConvention):
    d, f = cfg.d_model[s], cfg.d_intermediate[s]
    from hnet.model.block import mixer_config_kwargs

    codes = cfg.mixer_codes_for_stage(s)
    p = d  # final RMSNorm of the isotropic stack
    fl = 0.0
    for c, mlp in parse_layout(code):
        mixer = codes.get(c, c)
        if mixer == "attn":
            kw = {**cfg.attn_for_stage(s), **cfg.mixer_cfg_for_stage("attn", s)}
        elif mixer == "mamba2":
            kw = {}
        else:
            kw = mixer_config_kwargs(mixer, cfg, s)
        bp, bf = _block_counts(mixer, mlp, d, f, kw, L, cv, cfg.ssm)
        p += bp
        fl += bf
    height = sum(2 if mlp and f else 1 for _, mlp in parse_layout(code))
    ap, af = attn_res_counts(height, cfg.attn_res_for_stage(s), d, cv)
    return p + ap, fl + af


# ----------------------------------------------------------------------------- H-Net
def default_bytes_per_position(cfg: HNetConfig) -> list:
    out = [1.0]
    for r in cfg.dc.target_ratios:
        out.append(out[-1] * float(r))
    return out


def _hnet(cfg: HNetConfig, bpp, seq_len, cv: FlopsConvention):
    n = cfg.n_stages
    V, d0 = cfg.vocab_size, cfg.d_model[0]
    stages = []
    for s in range(n + 1):
        d = cfg.d_model[s]
        L = seq_len / bpp[s]
        layout = cfg.stage_layout(s)
        comp_p, comp_f = {}, {}
        if s == 0:
            comp_p["embedding"] = V * d0
            comp_f["embedding"] = 2 * V * d0 if cv.count_embedding else 0.0
            if cfg.ngram_embed:  # hashed n-gram tables: lookups like the embedding, plus one add per n
                ns = list(cfg.ngram_embed["ns"])
                b = cfg.ngram_embed["buckets"]
                rows = sum([int(b)] * len(ns) if isinstance(b, int) else [int(x) for x in b])
                comp_p["ngram_embedding"] = rows * d0
                comp_f["ngram_embedding"] = ((2 * rows * d0 if cv.count_embedding else 0.0)
                                             + (len(ns) * d0 if cv.count_elementwise else 0.0))
            comp_p["lm_head"] = 0 if cfg.tie_embeddings else V * d0
            comp_f["lm_head"] = 2 * V * d0 if cv.count_lm_head else 0.0
            if cfg.match_model:  # in-context match model (hnet.model.match): lookups + adds + a k-way logit gate
                mm = {"orders": [4, 8, 16, 32], "enc": True, "dec": True, "gate": True, **cfg.match_model}
                k = len(mm["orders"])
                n_tab = int(bool(mm["enc"])) + int(bool(mm["dec"]))
                comp_p["match_model"] = n_tab * k * 257 * d0 + ((d0 + 1) * k if mm["gate"] else 0)
                comp_f["match_model"] = ((n_tab * k * d0 if cv.count_elementwise else 0.0)
                                         + ((2 * d0 * k + (3 * k if cv.count_elementwise else 0)) if mm["gate"] else 0.0))
            if cfg.prefix_head:  # prefix output dictionary (hnet.model.prefix_head): table lookups are free
                from hnet.model.prefix_head import PrefixHead

                ph = {"buckets": 65536, "rank": 64, "k": 16, "use_main": True, **cfg.prefix_head}
                r, k, um = int(ph["rank"]), int(ph["k"]), bool(ph["use_main"])
                comp_p["prefix_head"] = int(ph["buckets"]) * r + d0 * (r - 1) * (2 if um else 1)
                comp_f["prefix_head"] = (PrefixHead.flops_per_byte(d0, r, k, um, cv.count_elementwise)
                                         if cv.count_lm_head else 0.0)
            if cfg.dc.main_aux_weight > 0:  # training-only auxiliary head on the dechunked main output
                comp_p["main_aux_head"] = V * d0
                comp_f["main_aux_head"] = 2 * V * d0 if cv.count_lm_head else 0.0
            if cfg.trie_head:  # prefix-trie output vocabulary (hnet.model.trie_head), every byte, train and eval
                th = {"dim": 32, "n_hash": 2, **cfg.trie_head}
                dim, nh = int(th["dim"]), int(th["n_hash"])
                comp_p["trie_head"] = nh * int(th["rows"]) * dim + d0 * dim
                comp_f["trie_head"] = (2 * d0 * dim  # query projection
                                       + (2 * V * dim if cv.count_lm_head else 0.0)  # V candidate scores
                                       + (((nh - 1) * V * dim + V) if cv.count_elementwise else 0.0))  # row sums, add
            if cfg.dc.chunk_pred_k > 0:  # training-only chunk-level head, once per innermost position
                d_in = cfg.d_model[n - 1]
                comp_p["chunk_pred_head"] = cfg.dc.chunk_pred_k * V * d_in
                comp_f["chunk_pred_head"] = (2 * cfg.dc.chunk_pred_k * V * d_in / bpp[n]) if cv.count_lm_head else 0.0
        if s > 0 and d > cfg.d_model[s - 1]:
            comp_p["pad_dimension"] = d - cfg.d_model[s - 1]
            comp_f["pad_dimension"] = 0.0
        if len(layout) == 3:
            for name, code in (("encoder", layout[0]), ("decoder", layout[2])):
                comp_p[name], comp_f[name] = _iso(cfg, s, code, L, cv)
            dcs = cfg.dc.for_stage(s, n)
            rk = dcs.get("router_kwargs") or {}
            rp, rf = _router_counts(dcs["router"], d, rk, dcs["target_ratio"])
            comp_p["router"] = rp
            comp_f["router"] = rf if cv.count_dc else 0.0
            comp_p["residual_proj"] = d * d + d
            comp_f["residual_proj"] = 2 * d * d if cv.count_residual_proj else 0.0
            if cfg.dc.fusion == "balanced":  # two RMSNorm gains on the decoder input (norms: no counted FLOPs)
                comp_p["fusion"] = 2 * d
                comp_f["fusion"] = 0.0
            if dcs.get("chunk_vocab"):  # hashed chunk tables (lookups, like the embedding) + training-only head
                cvk = {"buckets": 65536, "n_hash": 2, "pred_weight": 0.0, "n_neg": 4096, **dcs["chunk_vocab"]}
                d_in = d  # the inner stage returns its output at this stage's width
                rows = int(cvk["buckets"]) * int(cvk["n_hash"])
                L_in = seq_len / bpp[s + 1]
                comp_p["chunk_vocab"] = rows * d
                comp_f["chunk_vocab"] = ((2 * rows * d * L_in / seq_len if cv.count_embedding else 0.0)
                                         + (int(cvk["n_hash"]) * d * L_in / seq_len if cv.count_elementwise else 0.0))
                n_suf = len(cvk.get("suffix_lens") or [])
                if n_suf:  # suffix keys: gate (d -> S matmul + length table) and S x n_hash extra rows, weighted
                    comp_p["chunk_vocab"] += d * n_suf + n_suf + (int(cvk.get("max_len", 16)) + 1) * n_suf
                    comp_f["chunk_vocab"] += (2 * d * n_suf + ((int(cvk["n_hash"]) * n_suf * d + 2 * n_suf * d
                                                               + 5 * n_suf) if cv.count_elementwise else 0.0)
                                              ) * L_in / seq_len
                if float(cvk["pred_weight"]) > 0:
                    comp_p["chunk_vocab_head"] = d_in * d
                    comp_f["chunk_vocab_head"] = ((2 * d_in * d + 2 * d * (int(cvk["n_neg"]) + 1)) * L_in / seq_len
                                                  if cv.count_lm_head else 0.0)
            cp, cf = _chunker_counts(dcs["chunker"], d, dcs.get("chunker_kwargs") or {}, dcs["target_ratio"])
            if cp:
                comp_p["chunker"] = cp
                comp_f["chunker"] = cf if cv.count_dc else 0.0
            K = int(dcs.get("dechunk_slots") or 0)
            if K > 0:  # offset-resolved read-back: K d x d maps, run once per inner (chunk) position
                comp_p["dechunk_slots"] = K * d * d
                comp_f["dechunk_slots"] = 2 * K * d * d * bpp[s] / bpp[s + 1]
        else:
            assert len(layout) == 1, layout
            comp_p["main"], comp_f["main"] = _iso(cfg, s, layout[0], L, cv)
        fpp = sum(comp_f.values())
        stages.append(
            dict(
                stage=s,
                name="main" if s == n else f"stage{s}",
                bytes_per_position=bpp[s],
                positions_per_byte=1.0 / bpp[s],
                seq_len=L,
                params=sum(comp_p.values()),
                flops_per_position=fpp,
                flops_per_byte=fpp / bpp[s],
                components={k: dict(params=comp_p[k], flops_per_byte=comp_f[k] / bpp[s]) for k in comp_p},
            )
        )
    return stages


# ----------------------------------------------------------------------------- dense
def _dense(cfg: DenseConfig, seq_len, cv: FlopsConvention):
    if cfg.variant not in ("llama",):
        # TODO(interface): modded-nanogpt variant (WS-G) has extra value embeddings etc.; counted as llama here.
        pass
    d, V = cfg.d_model, cfg.vocab_size
    a = dict(num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads or cfg.num_heads, window_size=-1,
             head_dim=cfg.head_dim)
    if cfg.mixer != "attn":  # same kwargs the model passes to the mixer
        a = dict(num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads, rope_theta=cfg.rope_theta,
                 nope_every=cfg.nope_every, **({"head_dim": cfg.head_dim} if cfg.head_dim is not None else {}),
                 **cfg.extra.get("attn", {}), **cfg.mixer_kwargs)
    layer_p, layer_f = _block_counts(cfg.mixer, True, d, cfg.d_intermediate, a, seq_len, cv)
    comp_p = {
        "embedding": V * d,
        "lm_head": 0 if cfg.tie_embeddings else V * d,
        "body": cfg.n_layers * layer_p + d,  # + final norm
    }
    comp_f = {
        "embedding": 2 * V * d if cv.count_embedding else 0.0,
        "lm_head": 2 * V * d if cv.count_lm_head else 0.0,
        "body": cfg.n_layers * layer_f,
    }
    ap, af = attn_res_counts(2 * cfg.n_layers if cfg.d_intermediate else cfg.n_layers, cfg.attn_res, d, cv)
    comp_p["body"] += ap
    comp_f["body"] += af
    if cfg.match_model:  # token match model (hnet.model.match.TokenMatchModel), counted like the byte one + P_s
        mm = {"orders": [1, 2, 4, 8], "enc": True, "dec": True, "gate": True, **cfg.match_model}
        k = len(mm["orders"])
        n_site = int(bool(mm["enc"])) + int(bool(mm["dec"]))
        # per site: lookups free; d x d projection; per order a gain multiply, a sum over orders, an offset add
        comp_p["match_model"] = n_site * (d * d + 2 * k * d + k * d) + ((d + 1) * k if mm["gate"] else 0)
        comp_f["match_model"] = (n_site * (2 * d * d + (3 * k * d if cv.count_elementwise else 0.0))
                                 + ((2 * d * k + (3 * k if cv.count_elementwise else 0)) if mm["gate"] else 0.0))
    return comp_p, comp_f


# ----------------------------------------------------------------------------- public API
def count_params(cfg) -> int:
    """Exact parameter count (== sum(p.numel() for p in build_model(cfg).parameters()))."""
    if isinstance(cfg, HNetConfig):
        return int(sum(s["params"] for s in _hnet(cfg, default_bytes_per_position(cfg), 8192, SPEC)))
    if isinstance(cfg, DenseConfig):
        return int(sum(_dense(cfg, 1, SPEC)[0].values()))
    raise TypeError(type(cfg))


def count_params_nonembed(cfg) -> int:
    """Parameters excluding token embedding and LM head (tiny for byte models)."""
    if isinstance(cfg, HNetConfig):
        V, d0 = cfg.vocab_size, cfg.d_model[0]
        return count_params(cfg) - V * d0 * (1 if cfg.tie_embeddings else 2)
    p = _dense(cfg, 1, SPEC)[0]
    return int(p["body"] + p.get("match_model", 0))


def model_flops(
    cfg,
    bytes_per_position: Optional[Sequence[float]] = None,
    seq_len: Optional[float] = None,
    bytes_per_token: Optional[float] = None,
    training: bool = True,
    convention=None,
) -> dict:
    """FLOPs per byte (and per token for dense models).

    HNetConfig: ``seq_len`` is in bytes (default 8192); ``bytes_per_position[s]`` = bytes per stage-s position
    (default cumulative target ratios; ``bytes_per_position[-1]`` is BPIC). ``bytes_per_token`` is ignored.
    DenseConfig: ``seq_len`` is in tokens (default 1792); ``bytes_per_token`` default 4.6 (GPT-2 on FineWeb-Edu).
    ``convention``: None/"spec" (causal attention, no embedding, LM head), "paper_appB", a dict of
    FlopsConvention overrides, or a FlopsConvention.

    Returns dict(flops_per_byte, flops_per_token (dense only, else None), per_stage, params, params_nonembed,
    bytes_per_position, seq_len, training, convention).
    """
    cv = _conv(convention)
    mult = (1.0 + cv.backward_factor) if training else 1.0
    if isinstance(cfg, HNetConfig):
        bpp = list(bytes_per_position) if bytes_per_position is not None else default_bytes_per_position(cfg)
        assert len(bpp) == cfg.n_stages + 1, f"bytes_per_position needs {cfg.n_stages + 1} entries"
        bpp = [float(x) for x in bpp]
        seq_len = 8192 if seq_len is None else seq_len
        stages = _hnet(cfg, bpp, seq_len, cv)
        for st in stages:
            st["flops_per_position"] *= mult
            st["flops_per_byte"] *= mult
            for c in st["components"].values():
                c["flops_per_byte"] *= mult
        params = sum(st["params"] for st in stages)
        V, d0 = cfg.vocab_size, cfg.d_model[0]
        return dict(
            flops_per_byte=sum(st["flops_per_byte"] for st in stages),
            flops_per_token=None,
            per_stage=stages,
            params=int(params),
            params_nonembed=int(params - V * d0 * (1 if cfg.tie_embeddings else 2)),
            bytes_per_position=bpp,
            seq_len=seq_len,
            training=training,
            convention=asdict(cv),
        )
    if isinstance(cfg, DenseConfig):
        bpt = 4.6 if bytes_per_token is None else float(bytes_per_token)
        seq_len = 1792 if seq_len is None else seq_len
        comp_p, comp_f = _dense(cfg, seq_len, cv)
        fpt = sum(comp_f.values()) * mult
        stage = dict(
            stage=0,
            name="dense",
            bytes_per_position=bpt,
            positions_per_byte=1.0 / bpt,
            seq_len=seq_len,
            params=int(sum(comp_p.values())),
            flops_per_position=fpt,
            flops_per_byte=fpt / bpt,
            components={k: dict(params=comp_p[k], flops_per_byte=comp_f[k] * mult / bpt) for k in comp_p},
        )
        return dict(
            flops_per_byte=fpt / bpt,
            flops_per_token=fpt,
            per_stage=[stage],
            params=int(sum(comp_p.values())),
            params_nonembed=int(comp_p["body"] + comp_p.get("match_model", 0)),
            bytes_per_position=[bpt],
            seq_len=seq_len,
            training=training,
            convention=asdict(cv),
        )
    raise TypeError(type(cfg))


# ----------------------------------------------------------------------------- matching helpers
def with_main_depth(cfg: HNetConfig, n_layers: int, code: str = "T") -> HNetConfig:
    """Copy of ``cfg`` whose main network is ``f"{code}{n_layers}"``."""
    d = copy.deepcopy(cfg.to_dict())
    layout = d["arch_layout"]
    node = layout
    for _ in range(cfg.n_stages):
        node = node[1]
    node[0] = f"{code}{n_layers}"
    return HNetConfig.from_dict(d)


def match_main_depth(cfg: HNetConfig, target_flops_per_byte: float, **kw) -> tuple:
    """Main-network depth whose training FLOPs/byte is closest to the target. Returns (n_layers, rel_err)."""
    best = None
    for n in range(1, 200):
        c = with_main_depth(cfg, n)
        f = model_flops(c, **kw)["flops_per_byte"]
        err = f / target_flops_per_byte - 1
        if best is None or abs(err) < abs(best[1]):
            best = (n, err)
        if err > 0.5:
            break
    return best


def summary_row(cfg, **kw) -> dict:
    r = model_flops(cfg, **kw)
    return dict(
        params=r["params"],
        params_nonembed=r["params_nonembed"],
        gflops_per_byte=r["flops_per_byte"] / GIGA,
        bytes_per_main_position=r["bytes_per_position"][-1],
    )


__all__ = [
    "FlopsConvention",
    "SPEC",
    "PAPER_APPB",
    "BLOG_FIG8",
    "CONVENTIONS",
    "model_flops",
    "count_params",
    "count_params_nonembed",
    "default_bytes_per_position",
    "match_main_depth",
    "with_main_depth",
    "round_up",
    "parse_layout",
    "attn_context",
]
