"""Boundary decision rules applied to a router's scores (HNetConfig.dc.decision).

``threshold`` (paper): a position is selected iff p > 0.5. The batch fraction selected is steered only indirectly,
through the ratio loss on the mean of p. When a stage's p's cluster near 0.5 (undecided router), a uniform shift of
the router's bias flips every position at once; at high LR this becomes a limit cycle between "select everything"
and "select nothing" (logbook, EXP-003).

``quantile`` (rank decisions on normalized scores): exactly the top 1/N positions of the batch, by router score,
are selected, whatever the scale or offset of the scores. The score s (the router's logit, or logit(p) when the
router has none) is standardized over the batch, z = (s - mean) / std with the mean and std taken over the central
98% of the scores (robust to a few huge scores; plain mean/std with ``robust=False``), with the statistics in the
graph (as in BatchNorm), and the confidence seen by the STE and the EMA dechunker is p' = sigmoid(temp (z - z_tau)) with z_tau
midway between the k-th and (k+1)-th largest z: 0.5 exactly at the decision boundary. Running estimates of mean, std
and z_tau serve eval, like BatchNorm statistics. Under this rule the ratio loss is an exact constant (F = 1/N),
i.e. inert, and the router only has to learn the relative order of positions.

Exploration (``explore`` > 0, training only; off by default): the k selected positions are drawn by Gumbel-top-k,
i.e. the top k of ``z + tau * g`` with g ~ Gumbel(0, 1) i.i.d., instead of the top k of z. Compression stays exact
(k positions), the confidence p' and every running statistic are computed from the clean scores as before, so a
position selected only because of the noise has p' < 0.5: a *tentative* boundary that the EMA dechunker weights by
p' < 0.5 and whose confidence the STE / EMA credit pushes up (useful) or down (useless). The noise scale is in
standardized units and anneals linearly, tau = explore * max(0, 1 - progress / explore_frac) (explore_frac = 0:
constant). Because z is standardized, the noise only reorders positions near the threshold: an undecided router
explores, a decided (bimodal) one is left alone. Why: the only counterfactual credit the router gets is the EMA's
"fresh chunk vs stale chunk" gradient, and it exists only at selected positions; with deterministic rank decisions a
router that fell into the suffix basin never tries a boundary after the space and can never learn that it would be
worth more (the hierarchy's gain is 6x larger at word starts in the word basin).

Why normalize (EXP-014): with the threshold on p and the batch quantile detached, nothing anchors the scale of the
router's logits. The STE confidence gradient pushes selected logits up and unselected ones down forever, so in ~70
steps at lr 1.5e-3 the end_mlp router's logits exceed 17 and p saturates at exactly 1.0 in fp32 for more than 1/N
of the positions; the quantile then sits on the saturated value, every tied position recentres to p' = 0.5, the
strict ``>`` selects none, and the stage is stuck (compression ~500 with only forced boundaries). Standardizing the
scores removes the gradient component along a common shift or scale of s (it is a direction the forward pass does
not depend on), and ranking the fp32 scores instead of thresholding p removes the saturation ties.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..ctx import SeqCtx
from .base import RouterOutput, router_output

DECISIONS = ("threshold", "quantile")
EPS = 1e-6


class QuantileDecider(nn.Module):
    """``temp`` sets the confidence margins (std of temp * z is temp); ``normalize=False`` keeps the raw score
    scale (the EXP-014 behaviour, apart from the rank selection), for ablation."""

    def __init__(self, target_ratio: float, momentum: float = 0.99, temp: float = 2.0, normalize: bool = True,
                 robust: bool = True, z_clip: float = 4.0, trim: float = 0.01, explore: float = 0.0,
                 explore_frac: float = 0.3):
        super().__init__()
        assert target_ratio > 1, target_ratio
        self.ratio = float(target_ratio)
        self.momentum = momentum
        self.temp = float(temp)
        self.normalize = bool(normalize)
        self.robust = bool(robust)  # trimmed mean / std (EXP-018: heavy-tailed scores; EXP-021: MAD collapses)
        self.trim = float(trim)  # fraction trimmed at each end before the mean / std
        self.z_clip = float(z_clip)
        self.explore = float(explore)  # Gumbel-top-k noise scale (standardized units) at the start of training
        self.explore_frac = float(explore_frac)  # anneals linearly to 0 at this fraction of training (0 = constant)
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

    def explore_tau(self, progress: float) -> float:
        """Gumbel noise scale at training progress ``progress`` in [0, 1]."""
        if self.explore <= 0:
            return 0.0
        if self.explore_frac <= 0:
            return self.explore
        return self.explore * max(0.0, 1.0 - float(progress) / self.explore_frac)

    def forward(self, r: RouterOutput, ctx: SeqCtx, progress: float = 1.0,
                z_bias: Optional[torch.Tensor] = None) -> RouterOutput:
        """``z_bias`` (optional, [T] fp32, standardized units; HNetConfig.copy_route) is added to the standardized
        score before the rank decision and the confidence; the score statistics (mean, std) stay those of the raw
        score, and the running z_tau tracks the biased scores, so training and evaluation see the same rule."""
        s = (r.logit if r.logit is not None else torch.logit(r.p.clamp(EPS, 1 - EPS))).float()
        free = ctx.position_ids != 0  # the first position of every segment is forced selected (as in router_output)
        n_free = int(free.sum())
        if self.training and n_free >= 2:
            sf = s[free]
            if self.normalize and self.robust:
                # in the graph: the gradient loses its shift/scale components. Robust: a few huge scores must not
                # shrink everyone else's confidence toward 0.5 (EXP-018 seed 1: std 15, margins 0.10, hierarchy
                # unused), so the statistics are taken over the central 1 - 2 trim of the scores. Not median / MAD:
                # with a word tokenizer 5/6 of the scores are near-identical "not a boundary" logits, the MAD is ~0
                # and 1 / sigma blows up the gradient (EXP-021 seed 0 diverged at step ~900).
                with torch.no_grad():
                    lo, hi = torch.quantile(sf, torch.tensor([self.trim, 1 - self.trim], device=sf.device))
                    keep = (sf >= lo) & (sf <= hi)
                st = sf[keep] if int(keep.sum()) >= 2 else sf
                mu = st.mean()
                sigma = st.std() + 1e-3
            elif self.normalize:
                mu, sigma = sf.mean(), sf.std() + EPS
            else:
                mu, sigma = s.new_zeros(()), s.new_ones(())
            z = (s - mu) / sigma
            if z_bias is not None:
                z = z + z_bias
            k = max(1, min(n_free - 1, int(round(n_free / self.ratio))))
            with torch.no_grad():
                zf = z[free]
                top = torch.topk(zf, k + 1)  # ties broken by rank, not by a comparison with the threshold
                z_tau = 0.5 * (top.values[k - 1] + top.values[k])
                mask = torch.zeros_like(free)
                tau_x = self.explore_tau(progress)
                if tau_x > 0:  # Gumbel-top-k: sample k positions without replacement, ~ softmax(z / tau)
                    u = torch.rand_like(zf).clamp_(1e-6, 1 - 1e-6)
                    chosen = torch.topk(zf - tau_x * torch.log(-torch.log(u)), k).indices
                    clean = torch.zeros_like(zf, dtype=torch.bool)
                    clean[top.indices[:k]] = True
                    swapped = 1.0 - clean[chosen].float().mean()  # share of the k boundaries moved by the noise
                else:
                    chosen = top.indices[:k]
                mask[torch.nonzero(free).squeeze(1)[chosen]] = True
                self._update(self.mu, mu.detach())
                self._update(self.sigma, sigma.detach())
                self._update(self.z_tau, z_tau)
                self.initialized.fill_(True)
        else:
            z = (s - self.mu) / self.sigma
            if z_bias is not None:
                z = z + z_bias
            z_tau = self.z_tau
            mask = (z > z_tau) & free
        zc = z - z_tau.detach()
        if self.z_clip > 0:
            zc = zc.clamp(-self.z_clip, self.z_clip)
        p2 = torch.sigmoid(self.temp * zc)
        out = router_output(p2, ctx, r.semantics, logit=s, mask=mask)
        out.extra = dict(r.extra)
        with torch.no_grad():
            out.extra["tau"] = torch.sigmoid(self.tau).float()  # threshold in p space, for the logs
            out.extra["score_mu"] = self.mu.clone()
            out.extra["score_std"] = self.sigma.clone()  # raw-score scale: the drift diagnostic
            if self.training and self.explore > 0 and n_free >= 2:
                out.extra["explore_tau"] = s.new_tensor(tau_x)
                out.extra["explore_swap"] = swapped if tau_x > 0 else s.new_zeros(())
        return out
