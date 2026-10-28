"""Cosine routers: the paper routing module (eq. 4; goombalab/hnet ``RoutingModule``) and its end-semantics twin.

    p_t = clamp((1 - cos(W_q h_{t-1}, W_k h_t)) / 2, 0, 1),     p = 1 at every segment start.

* ``cosine_lookahead`` (paper default, "start" semantics): a selected t is the first byte of a chunk, so a chunk is
  closed only after the first byte of the next chunk has been seen (LH-Net).
* ``end_cosine_shift`` ("end" semantics): the same formula, but a selected t is the last byte of a chunk.

Projections are identity-initialized (official) and flagged ``_no_reinit`` so ``init_weights`` keeps them.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, router_output


@register("router", "cosine_lookahead")
class CosineLookaheadRouter(nn.Module):
    """``forward(h [T, D], ctx) -> RouterOutput``. Projections run in the ambient (autocast) dtype; the cosine
    and probabilities are computed in fp32."""

    boundary_semantics = "start"

    def __init__(self, d_model: int, target_ratio: Optional[float] = None):
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        q = F.normalize(self.q_proj_layer(h[:-1]).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(h[1:]).float(), dim=-1)
        p = ((1 - (q * k).sum(-1)) / 2).clamp(0.0, 1.0)
        return router_output(F.pad(p, (1, 0), value=1.0), ctx, self.boundary_semantics)


@register("router", "end_cosine_shift")
class EndCosineShiftRouter(CosineLookaheadRouter):
    boundary_semantics = "end"
