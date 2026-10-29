"""Boundary decision rules applied to a router's probabilities (HNetConfig.dc.decision).

``threshold`` (paper): a position is selected iff p > 0.5. The batch fraction selected is steered only indirectly,
through the ratio loss on the mean of p. When a stage's p's cluster near 0.5 (undecided router), a uniform shift of
the router's bias flips every position at once; at high LR this becomes a limit cycle between "select everything"
and "select nothing" (logbook, EXP-003).

``quantile``: a position is selected iff p > tau, where in training tau is the (1 - 1/N) quantile of p over the
batch (so exactly a 1/N fraction is selected, whatever the scale or offset of p), and a running estimate of tau is
kept for eval, like a BatchNorm statistic. p is recentred so that tau maps to 0.5: p' = sigmoid(logit(p) -
logit(tau)); the STE confidence and the EMA dechunker then see a probability that is 0.5 exactly at the decision
boundary. Under this rule the ratio loss is an exact constant (F = 1/N), i.e. inert: compression no longer needs
an auxiliary loss, and the router only has to learn the relative order of positions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ctx import SeqCtx
from .base import RouterOutput, router_output

DECISIONS = ("threshold", "quantile")
EPS = 1e-6


class QuantileDecider(nn.Module):
    def __init__(self, target_ratio: float, momentum: float = 0.99):
        super().__init__()
        assert target_ratio > 1, target_ratio
        self.q = 1.0 - 1.0 / float(target_ratio)
        self.momentum = momentum
        self.register_buffer("tau", torch.tensor(0.5))
        self.register_buffer("initialized", torch.tensor(False))

    def forward(self, r: RouterOutput, ctx: SeqCtx) -> RouterOutput:
        p = r.p
        free = ctx.position_ids != 0  # the first position of every segment is forced selected (as in router_output)
        if self.training:
            with torch.no_grad():
                tau = torch.quantile(p[free].float(), self.q) if free.any() else p.new_tensor(0.5)
                if bool(self.initialized):
                    self.tau.mul_(self.momentum).add_((1 - self.momentum) * tau)
                else:
                    self.tau.copy_(tau)
                    self.initialized.fill_(True)
        else:
            tau = self.tau
        tau = tau.clamp(EPS, 1 - EPS)
        logit = torch.logit(p.clamp(EPS, 1 - EPS)) - torch.logit(tau)
        p2 = torch.sigmoid(logit)
        out = router_output(p2, ctx, r.semantics)
        out.extra = dict(r.extra)
        out.extra["tau"] = tau.detach().float()
        return out
