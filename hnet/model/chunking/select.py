"""Downsampler that keeps the selected positions (paper Sec. 2.2.1), registered as chunker ``select``."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx


@register("chunker", "select")
class SelectChunker:
    """``(h [T, D], boundary_mask [T] bool, ctx) -> (h_inner [M, D], ctx_inner)``.

    Inner segment s holds the selected positions of outer segment s, in order:
    ``next_cu_seqlens = [0, cumsum(mask)[cu_seqlens[1:] - 1]]``. ``ctx_inner.outer_index`` [M] maps inner positions
    back to outer ones.

    Host syncs: exactly one per call, reading (M, max inner segment length) together; M sizes the selection
    (``nonzero_static``) and max_seqlen is required by flash-attn.
    """

    def __call__(self, h: torch.Tensor, boundary_mask: torch.Tensor, ctx: SeqCtx) -> Tuple[torch.Tensor, SeqCtx]:
        csum = boundary_mask.long().cumsum(0)
        next_cu = F.pad(csum[ctx.cu_seqlens[1:].long() - 1], (1, 0))
        M, max_len = torch.stack((next_cu[-1], (next_cu[1:] - next_cu[:-1]).max())).tolist()
        idx = torch.nonzero_static(boundary_mask, size=M).squeeze(1)
        seq_idx = ctx.seq_idx[:, idx]
        position_ids = torch.arange(M, device=h.device) - next_cu[seq_idx[0].long()]
        ctx_inner = SeqCtx(next_cu.to(torch.int32), int(max_len), seq_idx, position_ids, outer_index=idx)
        return h.index_select(0, idx), ctx_inner
