"""Learned end-of-chunk routers ("end" semantics; see ``routing.base``): p_t is the probability that byte t is
the LAST byte of its chunk, decided from h_t alone.

* ``end_linear``: ``p_t = sigmoid(w . h_t + b)``.
* ``end_mlp`` (alias ``causal_end_of_chunk``): ``p_t = sigmoid(fc2(silu(fc1(h_t))))``.

Exact rule (both): with e_t := p_t, e_0 = 1 at every segment start, b_t = [e_t > 1/2]:

* chunk j spans the positions after the previous end up to its end E_j (b_{E_j} = 1), and its vector is the
  encoder output at the end, x_j = h_{E_j} (``select`` chunker);
* position t reads inner output J(t) = (sum_{s<=t} b_s) - 1: the most recent chunk that ended at or before t,
  i.e. the chunk containing t shifted back by one, except at an end byte, which reads its own just-completed
  chunk (that output summarizes bytes <= t and predicts the next chunk, starting at t + 1). The forced 1-byte
  chunk at the segment start guarantees J(t) >= 0.
* EMA smoothing over chunks, STE(c_t) with c_t = e_t or 1 - e_t, and the ratio loss are as in the paper.

Causality: b_t depends on h_t only; chunk j reaches the inner network only through h_{E_j}; position t reads
z_{J(t)} with E_{J(t)} <= t, and the EMA state at J(t) mixes chunks 0..J(t) only. So logits at t depend on bytes
<= t, and closing a chunk never looks at the next chunk's first byte.

Init: the output bias is logit(1/N) so mean p starts near the target selection rate 1/N; ``init_scale`` sets the
spread of the initial logits (std ~ init_scale for unit-RMS h). Weights are flagged ``_no_reinit``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, ratio_init_bias, router_output


def _init_head(head: nn.Linear, fan_in: int, init_scale: float, target_ratio):
    with torch.no_grad():
        nn.init.normal_(head.weight, std=init_scale / fan_in**0.5)
        head.bias.fill_(ratio_init_bias(target_ratio))
    head.weight._no_reinit = True


@register("router", "end_linear")
class EndLinearRouter(nn.Module):
    boundary_semantics = "end"

    def __init__(self, d_model: int, target_ratio: Optional[float] = None, init_scale: float = 0.5):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        _init_head(self.proj, d_model, init_scale, target_ratio)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        logit = self.proj(h).squeeze(-1).float()
        return router_output(torch.sigmoid(logit), ctx, self.boundary_semantics)


@register("router", "end_mlp")
@register("router", "causal_end_of_chunk")
class EndMLPRouter(nn.Module):
    boundary_semantics = "end"

    def __init__(
        self,
        d_model: int,
        target_ratio: Optional[float] = None,
        d_hidden: Optional[int] = None,
        init_scale: float = 0.5,
    ):
        super().__init__()
        d_hidden = d_hidden or max(d_model // 4, 16)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        self.fc1.weight._no_reinit = True
        _init_head(self.fc2, d_hidden, init_scale, target_ratio)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        logit = self.fc2(F.silu(self.fc1(h))).squeeze(-1).float()
        return router_output(torch.sigmoid(logit), ctx, self.boundary_semantics)
