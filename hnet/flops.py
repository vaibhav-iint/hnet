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
    hd = d // nh
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


# Router parameter / FLOP counts by registry name: name -> fn(d, kwargs) -> (params, forward FLOPs per position).
# cosine_lookahead = paper (q/k projections, d x d each, no bias). MLP routers: fc1 d->h (+bias), fc2 h->1 (+bias).
# Names not listed here are counted by instantiating the registered router on the meta device (no memory);
# tests/test_flops.py::test_params_match_model checks every config against hnet.model.build_model.
def _mlp_router(d, kw):
    h = kw.get("d_hidden") or d
    return d * h + h + h + 1, 2 * (d * h + h)


ROUTER_COUNTS = {
    "cosine_lookahead": lambda d, kw: (2 * d * d, 4 * d * d),
    "end_cosine_shift": lambda d, kw: (2 * d * d, 4 * d * d),  # TODO(interface): assumed q/k like the paper
    "causal_end_of_chunk": _mlp_router,
    "end_mlp": _mlp_router,
    "end_linear": lambda d, kw: (d + 1, 2 * d),
    "fixed_pool": lambda d, kw: (0, 0),
    "space_like": lambda d, kw: (0, 0),
    "external": lambda d, kw: (0, 0),
}


def _router_counts(name: str, d: int, kwargs: dict, target_ratio=None):
    if name in ROUTER_COUNTS:
        return ROUTER_COUNTS[name](d, kwargs or {})
    try:  # exact count from the registered implementation, built on the meta device
        import torch

        from hnet.registry import build

        with torch.device("meta"):
            r = build("router", name, d, target_ratio=target_ratio, **(kwargs or {}))
        p = sum(x.numel() for x in r.parameters())
    except Exception:  # pragma: no cover - unknown router and no implementation available
        p = 2 * d * d
    return p, 2 * p


# ----------------------------------------------------------------------------- isotropic stacks
def _block_params(code, mlp, d, f, cfg_ssm, attn_cfg, mixer_codes) -> int:
    mixer = mixer_codes.get(code, code)
    p = d  # norm1
    if mixer == "mamba2":
        p += mamba2_params(d, cfg_ssm)
    elif mixer == "attn":
        p += attn_params(d, attn_cfg)
    else:
        raise NotImplementedError(f"no param count for mixer {mixer!r}")
    if mlp and f:
        p += d + mlp_params(d, f)  # norm2 + SwiGLU
    return p


def _block_flops(code, mlp, d, f, cfg_ssm, attn_cfg, mixer_codes, L, cv) -> float:
    mixer = mixer_codes.get(code, code)
    if mixer == "mamba2":
        fl = mamba2_flops(d, cfg_ssm, cv)
    elif mixer == "attn":
        fl = attn_flops(d, attn_cfg, L, cv)
    else:
        raise NotImplementedError(f"no FLOP count for mixer {mixer!r}")
    if mlp:
        fl += mlp_flops(d, f, cv)
    return fl


def _iso(cfg: HNetConfig, s: int, code: str, L: float, cv: FlopsConvention):
    d, f = cfg.d_model[s], cfg.d_intermediate[s]
    a = cfg.attn_for_stage(s)
    p = d  # final RMSNorm of the isotropic stack
    fl = 0.0
    for c, mlp in parse_layout(code):
        p += _block_params(c, mlp, d, f, cfg.ssm, a, cfg.mixer_codes)
        fl += _block_flops(c, mlp, d, f, cfg.ssm, a, cfg.mixer_codes, L, cv)
    return p, fl


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
            comp_p["lm_head"] = 0 if cfg.tie_embeddings else V * d0
            comp_f["lm_head"] = 2 * V * d0 if cv.count_lm_head else 0.0
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
    a = dict(num_heads=cfg.num_heads, num_kv_heads=cfg.num_kv_heads or cfg.num_heads, window_size=-1)
    layer_p = d + attn_params(d, a) + d + mlp_params(d, cfg.d_intermediate)
    layer_f = attn_flops(d, a, seq_len, cv) + mlp_flops(d, cfg.d_intermediate, cv)
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
    return int(p["body"])


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
            params_nonembed=int(comp_p["body"]),
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
