"""Causal multi-head / grouped-query attention mixer (registered as mixer ``"attn"``).

Parameter names follow goombalab/hnet ``CausalMHA``: a fused ``Wqkv`` whose output rows are ``[q | k | v]``
(q: num_heads * head_dim, k and v: num_kv_heads * head_dim each) and ``out_proj``.

Masking: position t attends to key s iff s and t are in the same segment (``ctx.seq_idx``), s <= t and, for a
sliding window w >= 0, t - s <= w (flash-attn ``window_size=(w, -1)`` convention: w + 1 keys including t).

RoPE: partial, non-interleaved (see ``hnet.model.rotary``) with positions restarting at every segment. Layers
with ``nope_every > 0 and (layer_idx + 1) % nope_every == 0`` use no positional encoding (NoPE).

Backends, chosen per call (see :func:`select_backend`):

* ``"flash"``: ``flash_attn_varlen_func`` (CUDA, fp16/bf16, flash_attn installed). Needs ``ctx.max_seqlen``
  on the host (already a Python int in ``SeqCtx``, no sync here).
* ``"flex"``: ``torch.nn.attention.flex_attention`` with a document/causal/window block mask (CUDA, compiled).
  The block mask is built once per stage and window and memoized in ``ctx.cache``.
* ``"sdpa"``: ``scaled_dot_product_attention`` with an explicit [T, T] boolean mask (CPU / fallback; O(T^2)
  memory, meant for tests and small inputs). Also memoized in ``ctx.cache``.

``FORCE_BACKEND`` (module attribute, or env ``HNET_ATTN_BACKEND``) overrides the automatic choice.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from ..rotary import apply_rotary, rotary_cos_sin

try:  # optional fast path; only a missing package falls back, a broken install raises
    from flash_attn import flash_attn_varlen_func as _flash_varlen
except ModuleNotFoundError as e:
    if e.name != "flash_attn":
        raise
    _flash_varlen = None

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # torch < 2.5
    create_block_mask = flex_attention = None

BACKENDS = ("flash", "flex", "sdpa")
FORCE_BACKEND: Optional[str] = os.environ.get("HNET_ATTN_BACKEND") or None
_compiled_flex = None


def available_backends(device: torch.device) -> list:
    """Backends usable for tensors on ``device`` (flash additionally needs fp16/bf16 inputs)."""
    out = ["sdpa"]
    if flex_attention is not None:
        out.append("flex")
    if device.type == "cuda" and _flash_varlen is not None:
        out.append("flash")
    return out


def select_backend(q: torch.Tensor) -> str:
    if FORCE_BACKEND is not None:
        assert FORCE_BACKEND in BACKENDS, FORCE_BACKEND
        return FORCE_BACKEND
    if q.is_cuda:
        if _flash_varlen is not None and q.dtype in (torch.float16, torch.bfloat16):
            return "flash"
        if flex_attention is not None:
            return "flex"
    return "sdpa"


def _flex_fn():
    """flex_attention compiled once on CUDA, with dynamic shapes (eager flex is a slow reference implementation)."""
    global _compiled_flex
    if _compiled_flex is None:
        _compiled_flex = torch.compile(flex_attention, dynamic=True)  # inner-stage lengths vary per step
    return _compiled_flex


def packed_block_mask(mask_mod, H, T: int, device: torch.device, block: int = 128, max_elems: int = 1 << 23):
    """BlockMask for a packed length-T sequence whose ``mask_mod`` indexes per-position tensors (segment ids etc.).

    Built without ``create_block_mask``: its inductor-compiled path (``_compile=True``) caused CUDA illegal memory
    accesses on per-step packed lengths (EXP-002), and its eager path materializes [H, T, T] intermediates (int64 for
    index lookups: >1 GiB at T ~ 12k). Here the block-level partial/full maps are evaluated in row chunks of at most
    ``max_elems`` pairs by broadcasting ``mask_mod`` over (h, q, kv) index tensors, so a ``mask_mod`` must use only
    broadcasting tensor ops (true for vmap-style masks). Indices are clamped to T - 1 and out-of-range pairs masked,
    so a ``mask_mod`` never reads past its per-position tensors (flex also evaluates it on a partial tail block). The
    bound is a tensor, not a Python int, so a new T does not recompile flex.
    """
    from torch.nn.attention.flex_attention import BlockMask

    last = torch.tensor(T - 1, device=device)

    def bounded(b, h, q_idx, kv_idx):
        inside = (q_idx <= last) & (kv_idx <= last)
        return mask_mod(b, h, torch.minimum(q_idx, last), torch.minimum(kv_idx, last)) & inside

    nb = -(-T // block)
    Tp, Hn = nb * block, H or 1
    rows = max(block, (max_elems // (Hn * Tp)) // block * block)
    b0 = torch.zeros((), dtype=torch.long, device=device)
    h = torch.arange(Hn, device=device).view(Hn, 1, 1)
    kv = torch.arange(Tp, device=device).view(1, 1, Tp)
    any_parts, all_parts = [], []
    for a in range(0, Tp, rows):
        q = torch.arange(a, min(a + rows, Tp), device=device).view(1, -1, 1)
        m = bounded(b0, h, q, kv).expand(Hn, q.shape[1], Tp).reshape(Hn, -1, block, nb, block)
        any_parts.append(m.any(dim=4).any(dim=2))
        all_parts.append(m.all(dim=4).all(dim=2))
    full = torch.cat(all_parts, dim=1)  # [Hn, nb, nb]
    partial = torch.cat(any_parts, dim=1) & ~full

    def ordered(blocks):  # flex layout: per q block, the count and the indices of the kv blocks (selected first)
        num = blocks.sum(-1, dtype=torch.int32)
        idx = torch.argsort(blocks.to(torch.int8), dim=-1, descending=True, stable=True).to(torch.int32)
        return num[None], idx[None]

    kv_num, kv_idx = ordered(partial)
    full_num, full_idx = ordered(full)
    return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=block, mask_mod=bounded,
                                    seq_lengths=(T, T))


def _block_mask(ctx: SeqCtx, window: int, device: torch.device):
    key = ("flex_mask", window)
    bm = ctx.cache.get(key)
    if bm is None:
        seq = ctx.seq_idx[0]

        def mask_mod(b, h, q_idx, kv_idx):
            keep = (seq[q_idx] == seq[kv_idx]) & (kv_idx <= q_idx)
            if window >= 0:
                keep = keep & (q_idx - kv_idx <= window)
            return keep

        T = ctx.total_len
        bm = packed_block_mask(mask_mod, None, T, device)
        ctx.cache[key] = bm
    return bm


def _dense_mask(ctx: SeqCtx, window: int) -> torch.Tensor:
    key = ("dense_mask", window)
    m = ctx.cache.get(key)
    if m is None:
        seq = ctx.seq_idx[0]
        t = torch.arange(seq.numel(), device=seq.device)
        diff = t[:, None] - t[None, :]
        m = (seq[:, None] == seq[None, :]) & (diff >= 0)
        if window >= 0:
            m = m & (diff <= window)
        ctx.cache[key] = m
    return m


def attention(q, k, v, ctx: SeqCtx, window: int, softmax_scale: float, backend: Optional[str] = None):
    """q [T, H, hd], k/v [T, Hkv, hd] -> [T, H, hd]. Causal, segment-masked, optional sliding window."""
    backend = backend or select_backend(q)
    if backend == "flash":
        return _flash_varlen(
            q, k, v, ctx.cu_seqlens, ctx.cu_seqlens, ctx.max_seqlen, ctx.max_seqlen,
            softmax_scale=softmax_scale, causal=True, window_size=(window, -1),
        )
    gqa = k.shape[1] != q.shape[1]
    qt, kt, vt = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))  # [1, H, T, hd]
    if backend == "flex":
        fn = _flex_fn() if q.is_cuda else flex_attention
        out = fn(qt, kt, vt, block_mask=_block_mask(ctx, window, q.device), scale=softmax_scale, enable_gqa=gqa)
    else:
        out = F.scaled_dot_product_attention(
            qt, kt, vt, attn_mask=_dense_mask(ctx, window), scale=softmax_scale, enable_gqa=gqa
        )
    return out.squeeze(0).transpose(0, 1)


@register("mixer", "attn")
class CausalAttention(nn.Module):
    """x [1, T, d_model] -> [1, T, d_model]; see the module docstring for masking, RoPE and backends."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        rotary_emb_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        window_size: int = -1,
        nope_every: int = 0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        softmax_scale: Optional[float] = None,
        layer_idx: int = 0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        num_kv_heads = num_kv_heads or num_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        self.d_model, self.num_heads, self.num_kv_heads = d_model, num_heads, num_kv_heads
        self.head_dim = d_model // num_heads
        self.layer_idx = layer_idx
        self.window_size = int(window_size)
        self.rope_theta = float(rope_theta)
        rot = self.head_dim if rotary_emb_dim is None else int(rotary_emb_dim)
        if nope_every and (layer_idx + 1) % nope_every == 0:
            rot = 0
        assert rot % 2 == 0 and rot <= self.head_dim, f"bad rotary_emb_dim {rot}"
        self.rotary_emb_dim = rot
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim ** -0.5
        self.Wqkv = nn.Linear(d_model, (num_heads + 2 * num_kv_heads) * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_bias)

    @classmethod
    def config_kwargs(cls, cfg, stage: int) -> dict:
        """Constructor kwargs for stage ``stage`` of an ``HNetConfig``."""
        return cfg.attn_for_stage(stage)

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        qkv = self.Wqkv(x[0])
        hq, hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = qkv.split((hq * hd, hk * hd, hk * hd), dim=-1)
        q, k, v = q.view(T, hq, hd), k.view(T, hk, hd), v.view(T, hk, hd)
        if self.rotary_emb_dim > 0:
            cos, sin = rotary_cos_sin(ctx, self.rotary_emb_dim, self.rope_theta)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = attention(q, k, v, ctx, self.window_size, self.softmax_scale)
        return self.out_proj(out.reshape(1, T, hq * hd))

    def extra_repr(self) -> str:
        return (
            f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
            f"rotary_dim={self.rotary_emb_dim}, window={self.window_size}"
        )
