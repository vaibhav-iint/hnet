"""Rotary position embedding (GPT-NeoX / non-interleaved layout, optionally partial).

Positions come from ``ctx.position_ids`` so they restart at every packed segment, which matches flash-attn's
varlen rotary used by goombalab/hnet. The first ``rotary_dim`` channels of every head are rotated, as pairs
``(i, i + rotary_dim/2)``; the remaining channels pass through (partial rotary, e.g. H-Net uses head_dim/2).
"""

from __future__ import annotations

from typing import Tuple

import torch

from .ctx import SeqCtx


def rotary_cos_sin(ctx: SeqCtx, rotary_dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``(cos, sin)`` of shape [T, rotary_dim/2] for the positions of ``ctx``; memoized in ``ctx.cache``
    so every layer of a stage shares one table."""
    key = ("rotary", rotary_dim, float(theta))
    hit = ctx.cache.get(key)
    if hit is None:
        pos = ctx.position_ids.float()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, rotary_dim, 2, device=pos.device, dtype=torch.float32) / rotary_dim)
        )
        freqs = pos[:, None] * inv_freq[None, :]
        hit = ctx.cache[key] = (freqs.cos(), freqs.sin())
    return hit


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, H, hd]; cos/sin [T, r/2] with r <= hd. Returns x rotated on its first r channels, in x.dtype
    (math in fp32)."""
    half = cos.shape[-1]
    r = 2 * half
    xf = x[..., :r].float()
    x1, x2 = xf[..., :half], xf[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    rot = torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)
    if r == x.shape[-1]:
        return rot
    return torch.cat((rot, x[..., r:]), dim=-1)
