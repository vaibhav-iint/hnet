"""Upsamplers back to the outer resolution (paper Sec. 2.2.2), registry kind ``dechunker``.

Contract: ``(h_inner [M, D], router_out, ctx, ctx_inner) -> h_outer [T, D]`` (fp32). Outer position t reads inner
position ``plug_back_idx[t] = cumsum(boundary_mask)[t] - 1``: the last selected position at or before t (eq. 8).
For "start" semantics that is the chunk containing t; for "end" semantics it is the latest chunk that ended at or
before t (see ``routing.base``): the same index serves both.
"""

from __future__ import annotations

import torch

from hnet.registry import register

from ..ctx import SeqCtx
from ..routing.base import SEMANTICS, RouterOutput
from .scan import linear_recurrence

P_MIN = 1e-4


def plug_back_index(router_out: RouterOutput) -> torch.Tensor:
    """int64 [T]: index of the inner position each outer position reads (valid for both boundary semantics)."""
    assert router_out.semantics in SEMANTICS, router_out.semantics
    return router_out.boundary_mask.long().cumsum(0) - 1


@register("dechunker", "repeat")
class RepeatDechunker:
    """No smoothing: ``out_t = z_{plug_back_idx[t]}`` (paper ablation "w/o smoothing")."""

    def __call__(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        return h_inner.float().index_select(0, plug_back_index(router_out))


@register("dechunker", "ema")
class EMADechunker:
    """Smoothing module (eq. 5) then upsampling:

        P_j = clamp(p at the j-th selected position, 1e-4, 1 - 1e-4)
        zbar_j = P_j z_j + (1 - P_j) zbar_{j-1}      (state reset at every inner segment start)
        out_t = zbar_{plug_back_idx[t]}

    computed in fp32 with :func:`linear_recurrence` (``block`` positions per matmul block).
    """

    def __init__(self, block: int = 64):
        self.block = block

    def __call__(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        P = router_out.p.clamp(P_MIN, 1 - P_MIN).index_select(0, ctx_inner.outer_index)
        zbar = linear_recurrence(P[:, None] * h_inner.float(), torch.log1p(-P), ctx_inner.position_ids == 0, self.block)
        return zbar.index_select(0, plug_back_index(router_out))
