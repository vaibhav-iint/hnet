"""Boundary decision rules applied to a router's scores (HNetConfig.dc.decision).

``threshold`` (paper): a position is selected iff p > 0.5. The batch fraction selected is steered only indirectly,
through the ratio loss on the mean of p. When a stage's p's cluster near 0.5 (undecided router), a uniform shift of
the router's bias flips every position at once; at high LR this becomes a limit cycle between "select everything"
and "select nothing" (logbook, EXP-003).

``quantile`` (rank decisions on normalized scores): exactly the top 1/N positions of the batch, by router score,
are selected, whatever the scale or offset of the scores. The score s (the router's logit, or logit(p) when the
router has none) is standardized over the batch, z = (s - mean) / std, with the statistics in the graph (as in
BatchNorm), and the confidence seen by the STE and the EMA dechunker is p' = sigmoid(temp (z - z_tau)) with z_tau
midway between the k-th and (k+1)-th largest z: 0.5 exactly at the decision boundary. Running estimates of mean, std
and z_tau serve eval, like BatchNorm statistics. Under this rule the ratio loss is an exact constant (F = 1/N),
i.e. inert, and the router only has to learn the relative order of positions.

Why normalize (EXP-014): with the threshold on p and the batch quantile detached, nothing anchors the scale of the
router's logits. The STE confidence gradient pushes selected logits up and unselected ones down forever, so in ~70
steps at lr 1.5e-3 the end_mlp router's logits exceed 17 and p saturates at exactly 1.0 in fp32 for more than 1/N
of the positions; the quantile then sits on the saturated value, every tied position recentres to p' = 0.5, the
strict ``>`` selects none, and the stage is stuck (compression ~500 with only forced boundaries). Standardizing the
scores removes the gradient component along a common shift or scale of s (it is a direction the forward pass does
not depend on), and ranking the fp32 scores instead of thresholding p removes the saturation ties.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..ctx import SeqCtx
from .base import RouterOutput, router_output

DECISIONS = ("threshold", "quantile")
EPS = 1e-6


class QuantileDecider(nn.Module):
    """``temp`` sets the confidence margins (std of temp * z is temp); ``normalize=False`` keeps the raw score
    scale (the EXP-014 behaviour, apart from the rank selection), for ablation."""

    def __init__(self, target_ratio: float, momentum: float = 0.99, temp: float = 2.0, normalize: bool = True):
        super().__init__()
        assert target_ratio > 1, target_ratio
        self.ratio = float(target_ratio)
        self.momentum = momentum
        self.temp = float(temp)
        self.normalize = bool(normalize)
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("z_tau", torch.tensor(0.0))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def tau(self) -> torch.Tensor:
        """Eval-time decision threshold in score (logit) space."""
        return self.mu + self.sigma * self.z_tau

    def _update(self, buf: torch.Tensor, value: torch.Tensor):
        if bool(self.initialized):
            buf.mul_(self.momentum).add_((1 - self.momentum) * value)
        else:
            buf.copy_(value)

    def forward(self, r: RouterOutput, ctx: SeqCtx) -> RouterOutput:
        s = (r.logit if r.logit is not None else torch.logit(r.p.clamp(EPS, 1 - EPS))).float()
        free = ctx.position_ids != 0  # the first position of every segment is forced selected (as in router_output)
        n_free = int(free.sum())
        if self.training and n_free >= 2:
            sf = s[free]
            if self.normalize:
                mu, sigma = sf.mean(), sf.std() + EPS  # in the graph: the gradient loses its shift/scale components
            else:
                mu, sigma = s.new_zeros(()), s.new_ones(())
            z = (s - mu) / sigma
            k = max(1, min(n_free - 1, int(round(n_free / self.ratio))))
            with torch.no_grad():
                zf = z[free]
                top = torch.topk(zf, k + 1)  # ties broken by rank, not by a comparison with the threshold
                z_tau = 0.5 * (top.values[k - 1] + top.values[k])
                mask = torch.zeros_like(free)
                mask[torch.nonzero(free).squeeze(1)[top.indices[:k]]] = True
                self._update(self.mu, mu.detach())
                self._update(self.sigma, sigma.detach())
                self._update(self.z_tau, z_tau)
                self.initialized.fill_(True)
        else:
            z = (s - self.mu) / self.sigma
            z_tau = self.z_tau
            mask = (z > z_tau) & free
        p2 = torch.sigmoid(self.temp * (z - z_tau.detach()))
        out = router_output(p2, ctx, r.semantics, logit=s, mask=mask)
        out.extra = dict(r.extra)
        with torch.no_grad():
            out.extra["tau"] = torch.sigmoid(self.tau).float()  # threshold in p space, for the logs
            out.extra["score_mu"] = self.mu.clone()
            out.extra["score_std"] = self.sigma.clone()  # raw-score scale: the drift diagnostic
        return out
