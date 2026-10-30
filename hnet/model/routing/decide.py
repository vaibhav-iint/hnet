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

Score features (``probe_rank`` > 0 with ``feat_point`` and/or ``feat_region``; off by default, stage 0 only): the rank
is taken over a learned mix of the router's standardized score and causal, prior-free features of the model's own
next-byte uncertainty, c_t = z_t + g_pt f_pt + g_rg f_rg, restandardized over the batch (in the graph, so a common
scale of c is not a direction the forward pass depends on: the gains g set relative weights only, and growing a gain
cannot saturate the confidences). The features come from a low-rank next-byte probe on the stage's DETACHED encoder
output (trained by its own CE, ``probe_loss``, on the next byte of the same segment): Ĥ_t = entropy of the probe's
p(x_{t+1} | x_<=t), detached. ``feat_point``: f = standardized Ĥ_t. Under "end" semantics a chunk should close at t
when byte t+1 is hard to predict from the bytes so far: after "word<space>" (the next word's first letter), not on
the last letter before the space (the suffix basin's guess; there the next byte, the space, is easy). ``feat_region``:
f = log of the causal mean of Ĥ over the last ``region_window`` bytes of the segment, centred (not divided by its
std: a log entropy-rate ratio, so the gain acts as an elasticity of the boundary density): a region-level difficulty
offset, so that the exact batch budget of k = n / N boundaries is spent where the text is hard (novel text, scripts
with high entropy per byte) and not where it is predictable (in-document copies the match model already predicts,
boilerplate). The batch total stays exactly k (training FLOPs unchanged); what varies is the compression per region
and per document. The gains are learned (Adam), init ``feat_point`` / ``feat_region``; ``feat_learn=False`` freezes
them. Eval: running statistics of Ĥ, of its region mean and of c serve the threshold, so decisions stay causal.

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
import torch.nn.functional as F

from ..ctx import SeqCtx
from .base import RouterOutput, router_output

DECISIONS = ("threshold", "quantile")
EPS = 1e-6
H_FLOOR = 0.25  # nats: floor of the std of the entropy features (an uninformative, near-constant probe gives f ~ 0)
F_CLIP = 3.0  # features are clipped to +-F_CLIP
C_FLOOR = 0.1  # floor of the std of the mixed score (standardized units)
R_EPS = 0.05  # nats added to the region mean entropy before the log (a predicted copy: H ~ 0)


def trimmed_stats(x: torch.Tensor, trim: float):
    """(mean, std) over the central 1 - 2 trim of x, in the graph (the trimming mask is not). Not median / MAD: with a
    word tokenizer 5/6 of the scores are near-identical, the MAD is ~0 and 1 / sigma blows up (EXP-021)."""
    with torch.no_grad():
        lo, hi = torch.quantile(x, torch.tensor([trim, 1 - trim], device=x.device, dtype=x.dtype))
        keep = (x >= lo) & (x <= hi)
    xt = x[keep] if int(keep.sum()) >= 2 else x
    return xt.mean(), xt.std() + 1e-3


def causal_window_mean(x: torch.Tensor, ctx: SeqCtx, window: int) -> torch.Tensor:
    """[T]: mean of x over the last ``window`` positions of the same segment, ending at t (inclusive). Shifted adds in
    a fixed order, so a packed batch gives exactly the values of its segments run separately."""
    pos = ctx.position_ids.reshape(-1)
    acc = x.clone()
    for j in range(1, window):
        sh = torch.zeros_like(x)
        sh[j:] = x[:-j]
        acc = acc + sh * (pos >= j).to(x.dtype)
    return acc / torch.clamp(pos + 1, max=window).to(x.dtype)


class NextByteProbe(nn.Module):
    """Low-rank next-byte probe on a stage's detached encoder output (``QuantileDecider`` score features). The name
    contains "router", so the Muon optimizer trains it with Adam (``hnet.optim.muon.classify_params``)."""

    def __init__(self, d_model: int, rank: int, vocab: int = 256):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab)
        with torch.no_grad():
            nn.init.normal_(self.down.weight, std=d_model ** -0.5)
            nn.init.normal_(self.up.weight, std=0.02)
            self.up.bias.zero_()
        for p in self.parameters():
            p._no_reinit = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=h.device.type, enabled=False):
            return self.up(self.down(h.detach().float()))


class QuantileDecider(nn.Module):
    """``temp`` sets the confidence margins (std of temp * z is temp); ``normalize=False`` keeps the raw score
    scale (the EXP-014 behaviour, apart from the rank selection), for ablation. ``probe_rank`` / ``feat_point`` /
    ``feat_region``: score features from a next-byte probe (module docstring); ``d_model`` is the stage width (passed
    by ``HNet``)."""

    def __init__(self, target_ratio: float, momentum: float = 0.99, temp: float = 2.0, normalize: bool = True,
                 robust: bool = True, z_clip: float = 4.0, trim: float = 0.01, explore: float = 0.0,
                 explore_frac: float = 0.3, d_model: int = None, probe_rank: int = 0, feat_point: float = 0.0,
                 feat_region: float = 0.0, region_window: int = 16, feat_learn: bool = True,
                 probe_weight: float = 1.0, router_w0: float = 1.0, router_ramp: float = 0.0,
                 score_floor: float = 0.0, teach: float = 0.0):
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
        # score features (off by default: no parameters, no buffers, and the forward pass is the plain rank decision)
        self.feat_names = [n for n, g in (("pt", feat_point), ("rg", feat_region)) if g != 0]
        self.router_probe = None
        self.probe_weight = float(probe_weight)
        self.region_window = int(region_window)
        # feature-led start (a curriculum on the ROUTER's weight in the mix, not on any prior): the router's standardized
        # score enters with weight router_w0, rising linearly to 1 at router_ramp of training (0 = always 1)
        self.router_w0, self.router_ramp = float(router_w0), float(router_ramp)
        # score_floor > 0: floor of the std used to standardize the router score. With a zero-initialised router head
        # (g3-route-02) the scores are near-constant at init; standardizing them by their own tiny std would (a) blow
        # any init preference up to full strength (the init lottery standardization was meant to avoid) and (b) scale
        # the gradient into the router by 1 / std (grad norm 34 at step 5 in the CPU smoke, clipping the whole model).
        # Under the floor a router with no opinion yet has no say, and the gradient gain is at most 1 / score_floor.
        self.score_floor = float(score_floor)
        # teach > 0 (needs feat_point and router_ramp > 0): during the feature-led start the router's OWN standardized
        # score is also trained (BCE, scale-free, weight teach * (1 - progress / router_ramp)) toward the top k of the
        # probe's entropy feature. Measured reason (CPU smokes, seed 2): with the router's weight in the mix at 0.05 the
        # decisions sat 57% on whitespace at step 35, yet the router's own top k stayed at 0% whitespace and took the
        # decisions back as its weight rose: the router's own gradient, not its init, chose the suffix basin.
        self.teach = float(teach)
        assert 0.0 < self.router_w0 <= 1.0 and self.router_ramp >= 0.0
        if self.feat_names or int(probe_rank) > 0:
            assert int(probe_rank) > 0 and d_model, "score features need probe_rank > 0 and d_model"
            assert self.feat_names, "probe_rank > 0 without feat_point / feat_region does nothing"
            assert self.normalize, "score features are mixed in standardized units (normalize=True)"
            assert self.region_window >= 1
            with torch.random.fork_rng(devices=[]):  # every other module's init is unchanged by the probe
                self.router_probe = NextByteProbe(int(d_model), int(probe_rank))
            init = torch.tensor([float(g) for g in (feat_point, feat_region) if g != 0])
            if feat_learn:
                self.feat_gain = nn.Parameter(init)
            else:
                self.register_buffer("feat_gain", init)
            self.register_buffer("feat_stats", torch.tensor([[0.0, 1.0]] * len(self.feat_names)))  # running mean, std
            self.register_buffer("comb_stats", torch.tensor([0.0, 1.0]))  # running mean, std of the mixed score

    @property
    def tau(self) -> torch.Tensor:
        """Eval-time decision threshold in score (logit) space (without score features)."""
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

    def router_weight(self, progress: float) -> float:
        """Weight of the router's own standardized score in the mixed score at training progress ``progress``."""
        if self.router_ramp <= 0 or self.router_w0 >= 1.0:
            return 1.0
        return self.router_w0 + (1.0 - self.router_w0) * min(1.0, float(progress) / self.router_ramp)

    def teach_weight(self, progress: float) -> float:
        """Weight of the router teacher at training progress ``progress`` (decays to 0 at router_ramp)."""
        if self.teach <= 0 or self.router_ramp <= 0 or "pt" not in self.feat_names:
            return 0.0
        return self.teach * max(0.0, 1.0 - float(progress) / self.router_ramp)

    def _features(self, h: torch.Tensor, ctx: SeqCtx, input_ids):
        """(raw features [n_feat, T] detached, probe CE or None, probe entropy [T] detached). The probe reads h
        detached; its CE (next byte of the same segment, training only) is its only training signal."""
        assert h is not None, "score features need the stage's encoder output (HNet passes h)"
        logp = torch.log_softmax(self.router_probe(h), dim=-1)  # fp32 [T, V]
        loss = None
        if self.training and input_ids is not None and torch.is_grad_enabled():
            ids = input_ids.reshape(-1).long()
            pos = ctx.position_ids.reshape(-1)
            tgt = torch.full_like(ids, -100)
            tgt[:-1] = torch.where(pos[1:] != 0, ids[1:], torch.full_like(ids[1:], -100))  # same segment only
            if bool((tgt != -100).any()):
                loss = F.nll_loss(logp, tgt, ignore_index=-100)
        with torch.no_grad():
            H = -(logp.exp() * logp).sum(-1)
            raw = torch.stack([H if n == "pt" else torch.log(causal_window_mean(H, ctx, self.region_window) + R_EPS)
                               for n in self.feat_names])
        return raw, loss, H

    def _standardize_features(self, raw: torch.Tensor, free: torch.Tensor, batch_stats: bool):
        """(centred, scaled, clipped features [n_feat, T], mean, scale): batch statistics over the free positions in
        training, running ones in eval. "pt": standardized, the std floored at H_FLOOR nats (a near-constant probe gives
        f ~ 0). "rg": the log of the region's mean entropy, centred but NOT divided by its std: a fixed unit (a log
        ratio of entropy rates), so the gain is an elasticity (density ~ entropy rate^g' with g' ~ 1.3 g near a 1/4
        quantile of a Gaussian score) and small but real differences between regions are not inflated to +-3 sigma.
        (Measured in the first smoke: with a standardized region feature, a 0.17-nat difference in mean entropy
        between English and code moved English to 13.8 bytes per chunk and code to 3.1 in training mode.)"""
        if batch_stats:
            rf = raw[:, free]
            m, sd = rf.mean(1), rf.std(1)
        else:
            m, sd = self.feat_stats[:, 0], self.feat_stats[:, 1]
        is_rg = torch.tensor([n == "rg" for n in self.feat_names], device=raw.device)
        sd = torch.where(is_rg, torch.ones_like(sd), sd)
        f = (raw - m[:, None]) / sd.clamp(min=H_FLOOR)[:, None]
        return f.clamp(-F_CLIP, F_CLIP), m, sd

    def forward(self, r: RouterOutput, ctx: SeqCtx, progress: float = 1.0, h: torch.Tensor = None,
                input_ids: torch.Tensor = None, z_bias: Optional[torch.Tensor] = None) -> RouterOutput:
        """``z_bias`` (optional, [T] fp32, standardized units; HNetConfig.copy_route) is added to the standardized
        score before the rank decision and the confidence; the score statistics (mean, std) stay those of the raw
        score, and the running z_tau tracks the biased scores, so training and evaluation see the same rule."""
        s = (r.logit if r.logit is not None else torch.logit(r.p.clamp(EPS, 1 - EPS))).float()
        free = ctx.position_ids != 0  # the first position of every segment is forced selected (as in router_output)
        n_free = int(free.sum())
        feats = self.router_probe is not None
        probe_loss = None
        tw = 0.0
        if feats:
            raw, probe_loss, H = self._features(h, ctx, input_ids)
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
                sigma = (st.std().clamp(min=self.score_floor) if self.score_floor > 0 else st.std()) + 1e-3
            elif self.normalize:
                mu, sigma = sf.mean(), sf.std() + EPS
            else:
                mu, sigma = s.new_zeros(()), s.new_ones(())
            z = (s - mu) / sigma
            if feats:  # learned mix with the probe's features, restandardized in the graph (scale anchored)
                z_router = z.detach()
                f, f_mu, f_sd = self._standardize_features(raw, free, True)
                c = self.router_weight(progress) * z + (self.feat_gain.float()[:, None] * f).sum(0)
                c_mu, c_sd = trimmed_stats(c[free], self.trim)
                c_sd = c_sd.clamp(min=C_FLOOR)  # both parts uninformative (init): no 1 / std gradient blow-up
                z_own, z = z, (c - c_mu) / c_sd
            if z_bias is not None:
                z = z + z_bias
            k = max(1, min(n_free - 1, int(round(n_free / self.ratio))))
            tw = self.teach_weight(progress) if feats else 0.0
            if tw > 0:  # the router's own standardized score is also taught the probe's top k (curriculum)
                with torch.no_grad():
                    fi = f[self.feat_names.index("pt")][free]
                    tgt = torch.zeros_like(fi)
                    tgt[torch.topk(fi, k).indices] = 1.0
                    q = torch.topk(z_own[free].detach(), k + 1).values[k - 1:].mean()
                with torch.autocast(device_type=s.device.type, enabled=False):
                    teach_loss = tw * F.binary_cross_entropy_with_logits(self.temp * (z_own[free] - q), tgt)
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
                if feats:
                    self._update(self.feat_stats, torch.stack([f_mu, f_sd], 1))
                    self._update(self.comb_stats, torch.stack([c_mu.detach(), c_sd.detach()]))
                    # has the router itself adopted the mixed ranking? (its own top k vs the decided top k)
                    own = torch.zeros_like(free)
                    own[torch.nonzero(free).squeeze(1)[torch.topk(z_router[free], k).indices]] = True
                    r.extra["router_agree"] = (own & mask).sum().float() / k
                    if input_ids is not None:  # diagnostic only, never an input: the router's own basin
                        ids_ = input_ids.reshape(-1)
                        ws = (ids_ == 0x20) | (ids_ == 0x0A) | (ids_ == 0x09)
                        r.extra["router_own_on_ws"] = (own & ws).sum().float() / k
                self.initialized.fill_(True)
        else:
            z = (s - self.mu) / self.sigma
            if feats:
                f, _, _ = self._standardize_features(raw, free, False)
                c = self.router_weight(progress) * z + (self.feat_gain.float()[:, None] * f).sum(0)
                z = (c - self.comb_stats[0]) / self.comb_stats[1]
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
        if feats:
            out.probe_loss = probe_loss
            if self.training and n_free >= 2 and tw > 0:
                out.teach_loss = teach_loss
                out.extra["router_teach"] = teach_loss.detach() / tw
        with torch.no_grad():
            out.extra["tau"] = torch.sigmoid(self.tau).float()  # threshold in p space, for the logs
            out.extra["score_mu"] = self.mu.clone()
            out.extra["score_std"] = self.sigma.clone()  # raw-score scale: the drift diagnostic
            if self.training and self.explore > 0 and n_free >= 2:
                out.extra["explore_tau"] = s.new_tensor(tau_x)
                out.extra["explore_swap"] = swapped if tau_x > 0 else s.new_zeros(())
            if feats:
                out.extra.update(self._feature_stats(H, out.boundary_mask, ctx, probe_loss))
                if self.router_ramp > 0:
                    out.extra["router_w"] = s.new_tensor(self.router_weight(progress))
        return out

    def _feature_stats(self, H: torch.Tensor, mask: torch.Tensor, ctx: SeqCtx, probe_loss) -> dict:
        """Detached diagnostics: probe entropy and CE, the gains, and how unevenly the budget is spent per segment:
        ``comp_seg_cv`` = coefficient of variation of bytes per chunk over segments of >= 64 bytes; ``comp_seg_corr``
        = correlation of a segment's chunks per byte with its mean probe entropy; ``attn_ctx_ratio`` = sum over
        segments of chunks^2 relative to the same total spread uniformly over the bytes, i.e. the main network's
        realized attention-context FLOPs relative to uniform compression (uncounted by hnet.flops)."""
        d = {"probe_entropy": H.mean()}
        if probe_loss is not None:
            d["probe_ce"] = probe_loss.detach()
        for i, n in enumerate(self.feat_names):
            d[f"feat_gain_{n}"] = self.feat_gain.detach()[i].float()
        seg = ctx.seq_idx.reshape(-1).long()
        n_seg = int(ctx.cu_seqlens.numel() - 1)
        nb = torch.zeros(n_seg, device=H.device).index_add_(0, seg, torch.ones_like(H))
        nc = torch.zeros(n_seg, device=H.device).index_add_(0, seg, mask.reshape(-1).float())
        hs = torch.zeros(n_seg, device=H.device).index_add_(0, seg, H.float()) / nb.clamp(min=1)
        big = nb >= 64
        if int(big.sum()) >= 2:
            rate = nc[big] / nb[big]  # chunks per byte
            bpc = nb[big] / nc[big].clamp(min=1)
            d["comp_seg_cv"] = bpc.std() / bpc.mean()
            a, b = rate - rate.mean(), hs[big] - hs[big].mean()
            d["comp_seg_corr"] = (a * b).sum() / ((a.pow(2).sum() * b.pow(2).sum()).sqrt() + 1e-12)
        uni = nb * (nc.sum() / nb.sum().clamp(min=1))
        d["attn_ctx_ratio"] = nc.pow(2).sum() / uni.pow(2).sum().clamp(min=1e-6)
        return d
