"""Causal end-of-chunk router (ours), registered as ``causal_end_of_chunk``.

Exact rule
----------
For every position t of a segment (positions 0..L-1):

    e_t = sigmoid(fc2(silu(fc1(h_t))))          probability that a chunk ENDS at t (decided from h_t only)
    e_0 = 1                                     the first position is a chunk of its own
    b_t = [e_t > 1/2]                           t is the LAST position of its chunk

Chunk j covers the positions after the previous end up to and including its end position E_j; its vector is
the encoder output at the end, x_j = h_{E_j} (the ``select`` chunker keeps the positions with b_t = 1, so with
this router it picks chunk *ends*). Since h is causal, x_j and every inner-network output z_j depend only on
positions <= E_j.

Dechunking is shifted by one chunk: position t reads chunk

    J(t) = (sum_{s<=t} b_s) - 1,

the most recent chunk that ended at or before t. That is the previous chunk unless t itself ends a chunk, in which
case it is the chunk just completed (its inner output, computed from positions <= t, predicts the next chunk,
which begins at t + 1). The first position is always an end, so J(t) >= 0. ``J`` is exactly the ``plug_back_idx =
cumsum(mask) - 1`` of the standard dechunkers, so ``select`` / ``ema`` / ``repeat`` are reused unchanged; the
shift comes from selecting ends instead of starts. The EMA smoothing over chunks, the STE confidence c_t (a
function of e_t) and the ratio loss are as in the paper with p := e.

Causality: b_t is a function of h_t; chunk j enters the inner network only through h_{E_j}; position t only
reads z_{J(t)} with E_{J(t)} <= t, and the EMA state at J(t) mixes chunks 0..J(t) only. Hence logits at t depend
on bytes <= t alone, and the decision to start a new chunk at t + 1 never sees byte t + 1 (no lookahead).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, router_output


@register("router", "causal_end_of_chunk")
class CausalEndOfChunkRouter(nn.Module):
    """``forward(h [T, D], ctx) -> RouterOutput`` with ``boundary_mask`` marking chunk ends.

    d_hidden: MLP width (default d_model). init_std: std of ``fc2.weight`` (small, so e_t starts near 1/2 with a
    random per-position sign, i.e. about half of the positions are ends, like the identity-initialized cosine
    router). init_bias: initial logit offset (negative -> fewer ends at init). Both layers are flagged
    ``_no_reinit``.
    """

    def __init__(
        self,
        d_model: int,
        target_ratio: Optional[float] = None,
        d_hidden: Optional[int] = None,
        init_std: float = 0.02,
        init_bias: float = 0.0,
    ):
        super().__init__()
        d_hidden = d_hidden or d_model
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        with torch.no_grad():
            nn.init.normal_(self.fc2.weight, std=init_std)
            self.fc2.bias.fill_(init_bias)
        self.fc1.weight._no_reinit = True
        self.fc2.weight._no_reinit = True

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        logit = self.fc2(F.silu(self.fc1(h))).squeeze(-1)
        return router_output(torch.sigmoid(logit.float()), ctx)
