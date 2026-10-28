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

Init: for unit-RMS h (the encoder ends with an RMSNorm) the initial logits are ~ N(logit(1/N), init_scale^2). With
the default init_scale = 1.5 both the selected fraction F = Phi(logit(1/N) / 1.5) and the mean probability G start
near the target rate 1/N (N = 3: F ~ 0.32, G ~ 0.37), so the inner stage starts at roughly the target length.
Weights are flagged ``_no_reinit``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, ratio_init_bias, router_output


# Mean and std of silu(x) for x ~ N(0, 1): the MLP head's input statistics at init.
_SILU_MEAN, _SILU_STD = 0.2068, 0.5598


def _init_head(head: nn.Linear, in_mean: float, in_std: float, init_scale: float, target_ratio):
    """Logits ~ N(logit(1/N), init_scale^2) for i.i.d. inputs with mean ``in_mean`` and std ``in_std``."""
    with torch.no_grad():
        nn.init.normal_(head.weight, std=init_scale / (in_std * head.in_features**0.5))
        head.bias.copy_(ratio_init_bias(target_ratio) - in_mean * head.weight.sum(1))  # meta-device safe
    head.weight._no_reinit = True


@register("router", "end_linear")
class EndLinearRouter(nn.Module):
    boundary_semantics = "end"

    def __init__(self, d_model: int, target_ratio: Optional[float] = None, init_scale: float = 1.5):
        super().__init__()
        self.proj = nn.Linear(d_model, 1)
        _init_head(self.proj, 0.0, 1.0, init_scale, target_ratio)

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
        init_scale: float = 1.5,
    ):
        super().__init__()
        d_hidden = d_hidden or max(d_model // 4, 16)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        with torch.no_grad():
            nn.init.normal_(self.fc1.weight, std=d_model**-0.5)  # unit-variance pre-activations
            self.fc1.bias.zero_()
        self.fc1.weight._no_reinit = True
        _init_head(self.fc2, _SILU_MEAN, _SILU_STD, init_scale, target_ratio)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        logit = self.fc2(F.silu(self.fc1(h))).squeeze(-1).float()
        return router_output(torch.sigmoid(logit), ctx, self.boundary_semantics)
