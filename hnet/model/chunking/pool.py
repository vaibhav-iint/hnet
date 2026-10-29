"""Pooling downsampler, registered as chunker ``pool``: the chunk vector is a learned summary of ALL positions of the
chunk, not just the encoder state at its boundary.

Why: probes on trained tiny H-Nets (logbook, "boundary state, not a chunk summary") show the boundary state encodes
the boundary byte almost losslessly and the rest of the chunk barely (~1.5 of 4.8 bits about a 9-byte chunk's first
byte), so the main network never learns which word it is looking at. BLT solves the same problem with byte->patch
cross-attention; this is that idea inside the H-Net chunker.

    x_j = h[end_j] + proj(pool_j),   pool_j = mean_{t in chunk j} h_t                    (mode "mean")
                                     pool_j = sum_t softmax_t(q . Wk h_t / sqrt(D)) Wv h_t   (mode "attn")

``proj`` is zero-initialized, so at init the chunker equals ``select`` exactly and training can only add to it.

Requires "end" boundary semantics (a selected position is the LAST position of its chunk): under "start" semantics
the dechunker lets the first bytes of a chunk read its vector, which would leak the chunk's later bytes. Positions
after the last completed chunk of a segment belong to no chunk and are excluded from every pool (the dechunker
never reads an unfinished chunk either).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from hnet.registry import register

from ..ctx import SeqCtx
from .select import select_positions

MODES = ("mean", "attn")


@register("chunker", "pool")
class PoolChunker(nn.Module):
    requires_semantics = "end"

    def __init__(self, d_model: int, mode: str = "mean", init_std: float = 0.02, **_):
        super().__init__()
        assert mode in MODES, f"pool mode must be one of {MODES}, got {mode!r}"
        self.d_model, self.mode = d_model, mode
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.proj.weight._no_reinit = True
        if mode == "attn":
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.query = nn.Parameter(torch.zeros(d_model))
            for w in (self.k_proj.weight, self.v_proj.weight, self.query):
                nn.init.normal_(w, std=init_std)
                w._no_reinit = True

    @classmethod
    def flops_per_position(cls, d_model: int, target_ratio: float, mode: str = "mean", **_) -> float:
        """Forward FLOPs per OUTER position (proj runs once per chunk, i.e. on 1/N of the positions)."""
        f = 2 * d_model * d_model / target_ratio
        if mode == "attn":
            f += 4 * d_model * d_model + 4 * d_model  # k and v projections, score and weighted sum
        return float(f)

    def forward(self, h: torch.Tensor, boundary_mask: torch.Tensor, ctx: SeqCtx) -> Tuple[torch.Tensor, SeqCtx]:
        csum, idx, ctx_inner = select_positions(boundary_mask, ctx)
        M = idx.numel()
        chunk_id = csum - boundary_mask.long()  # end semantics: chunk j = (end_{j-1}, end_j]
        # positions after the last end of their segment belong to no (completed) chunk
        seg_last = ctx_inner.cu_seqlens[1:].long()[ctx.seq_idx[0].long()]  # chunks completed up to segment end
        valid = chunk_id < seg_last
        cid = torch.where(valid, chunk_id, torch.zeros_like(chunk_id))
        hf = h.float()
        if self.mode == "mean":
            w = valid.to(hf.dtype)
            count = torch.zeros(M, device=h.device, dtype=hf.dtype).index_add_(0, cid, w)
            pooled = torch.zeros(M, hf.shape[-1], device=h.device, dtype=hf.dtype).index_add_(0, cid, hf * w[:, None])
            pooled = pooled / count.clamp(min=1.0)[:, None]
        else:
            k = self.k_proj(h).float()
            v = self.v_proj(h).float()
            score = (k @ self.query.float()) / math.sqrt(self.d_model)
            score = torch.where(valid, score, torch.full_like(score, -1e4))
            smax = torch.full((M,), -1e4, device=h.device, dtype=score.dtype)
            smax = smax.scatter_reduce(0, cid, score, reduce="amax", include_self=True).detach()
            e = torch.exp(score - smax[cid]) * valid.to(score.dtype)
            z = torch.zeros(M, device=h.device, dtype=score.dtype).index_add_(0, cid, e)
            wgt = e / z.clamp(min=1e-20)[cid]
            pooled = torch.zeros(M, hf.shape[-1], device=h.device, dtype=hf.dtype).index_add_(0, cid, v * wgt[:, None])
        x = h.index_select(0, idx) + self.proj(pooled.to(h.dtype))
        return x, ctx_inner
