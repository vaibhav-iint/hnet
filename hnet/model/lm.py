"""Causal LM wrappers' shared I/O (packing, loss, outputs) and the byte-level ``HNetForCausalLM``.

Forward API (shared with the dense models, see SPEC "Model API")::

    out = model(input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True)

input_ids: LongTensor [T] (packed; segments given by ``cu_seqlens``, default one segment) or [B, L] (each row
one segment). labels: same shape, ``IGNORE_INDEX`` (-100) where there is no target.

Host syncs per forward: none here when ``max_seqlen`` is given (or input is [B, L]); otherwise one to compute it.
Each H-Net stage adds exactly one in its chunker (see ``SelectChunker``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from hnet.config import IGNORE_INDEX, HNetConfig

from .block import Block
from .ctx import SeqCtx
from .hnet import HNet
from .ngram import HashNgramEmbedding
from .routing.base import RouterOutput

# Max logits materialized at once by the chunked cross-entropy (return_logits=False): 2**25 fp32 = 128 MiB.
CE_CHUNK_ELEMENTS = 2**25


@dataclass
class CausalLMOutput:
    """logits: [T, V] | [B, L, V] | None. loss: mean CE over targets (None without labels). ce_sum: fp32 sum of CE.
    n_targets: number of targets. aux_loss: alpha * sum of per-stage aux losses (0 for dense models).
    stats: flat dict of detached scalars. bpred: per-stage ``RouterOutput`` (outer first)."""

    logits: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    ce_sum: Optional[torch.Tensor] = None
    n_targets: Optional[torch.Tensor] = None
    aux_loss: Optional[torch.Tensor] = None
    stats: Dict[str, torch.Tensor] = field(default_factory=dict)
    bpred: List[RouterOutput] = field(default_factory=list)


def _binary_entropy(p: torch.Tensor) -> torch.Tensor:
    p = p.float().clamp(1e-6, 1 - 1e-6)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def pack_inputs(
    input_ids: torch.Tensor, cu_seqlens: Optional[torch.Tensor], max_seqlen: Optional[int]
) -> Tuple[torch.Tensor, SeqCtx, Optional[Tuple[int, int]]]:
    """-> (ids [T], ctx, batch_shape or None). [B, L] input becomes B segments of length L."""
    if input_ids.dim() == 2:
        assert cu_seqlens is None, "pass either [B, L] input_ids or packed [T] input_ids with cu_seqlens"
        B, L = input_ids.shape
        cu = torch.arange(B + 1, device=input_ids.device, dtype=torch.int32) * L
        return input_ids.reshape(-1), SeqCtx.from_cu_seqlens(cu, L, B * L), (B, L)
    T = input_ids.shape[0]
    if cu_seqlens is None:
        cu_seqlens, max_seqlen = torch.arange(2, dtype=torch.int32, device=input_ids.device) * T, T
    return input_ids, SeqCtx.from_cu_seqlens(cu_seqlens, max_seqlen, T), None


def _ce_sum(h: torch.Tensor, head: nn.Linear, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(head(h).float(), labels, ignore_index=IGNORE_INDEX, reduction="sum")


def _ce_chunk(logits: torch.Tensor, labels: torch.Tensor, grad: bool):
    """(fp32 CE sum, d(CE sum)/d(logits) in logits' dtype or None) of one chunk of logits [n, V]."""
    lf = logits.float()
    lse = torch.logsumexp(lf, dim=-1)
    valid = labels != IGNORE_INDEX
    safe = torch.where(valid, labels, torch.zeros_like(labels))
    tgt = lf.gather(1, safe[:, None]).squeeze(1)
    loss = torch.where(valid, lse - tgt, torch.zeros_like(lse)).sum()
    if not grad:
        return loss, None
    g = torch.exp(lf - lse[:, None])
    g = g - F.one_hot(safe, lf.shape[1]).to(g.dtype)
    g = g * valid[:, None].to(g.dtype)
    return loss, g.to(logits.dtype)


_ce_chunk_compiled = None


def _ce_chunk_fn(device: torch.device):
    """``_ce_chunk`` fused by torch.compile on CUDA (one pass over the logits instead of ~8); eager elsewhere."""
    global _ce_chunk_compiled
    if device.type != "cuda" or os.environ.get("HNET_CE_COMPILE", "1") != "1":
        return _ce_chunk
    if _ce_chunk_compiled is None:
        _ce_chunk_compiled = torch.compile(_ce_chunk, dynamic=False)
    return _ce_chunk_compiled


class _ChunkedLinearCE(torch.autograd.Function):
    """sum CE(h @ W^T, labels) over row chunks. Each chunk's gradient is computed in the forward pass and kept (grad
    w.r.t. h: [T, D]; w.r.t. W: [V, D] fp32), so neither logits nor a recomputation (activation checkpointing) are
    needed: the LM head costs exactly its forward + backward FLOPs. Matmuls run in h's dtype (bf16 under autocast,
    as ``head(h)`` would), CE math in fp32."""

    @staticmethod
    def forward(ctx, h, weight, labels, rows: int):
        cdt = h.dtype
        w = weight.to(cdt)
        need_h, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        fn = _ce_chunk_fn(h.device)
        total = torch.zeros((), dtype=torch.float32, device=h.device)
        gh = torch.empty_like(h) if need_h else None
        gw = torch.zeros(weight.shape, dtype=torch.float32, device=h.device) if need_w else None
        for s in range(0, h.shape[0], rows):
            hs, ys = h[s:s + rows], labels[s:s + rows]
            loss, g = fn(hs @ w.T, ys, need_h or need_w)
            total = total + loss
            if need_h:
                gh[s:s + rows] = g @ w
            if need_w:
                gw.add_(g.T @ hs)
        ctx.save_for_backward(gh, gw)
        ctx.weight_dtype = weight.dtype
        return total

    @staticmethod
    def backward(ctx, go):
        gh, gw = ctx.saved_tensors
        return (None if gh is None else gh * go.to(gh.dtype),
                None if gw is None else (gw * go).to(ctx.weight_dtype), None, None)


def chunked_ce_sum(h: torch.Tensor, head: nn.Linear, labels: torch.Tensor) -> torch.Tensor:
    """fp32 sum of CE(head(h), labels) holding at most ~CE_CHUNK_ELEMENTS logits at once. With gradients, the
    chunks' backward runs in the forward pass (:class:`_ChunkedLinearCE`: no stored logits, no recompute)."""
    rows = max(1, CE_CHUNK_ELEMENTS // head.out_features)
    if h.shape[0] <= rows:
        return _ce_sum(h, head, labels)
    if head.bias is None and torch.is_grad_enabled() and (h.requires_grad or head.weight.requires_grad):
        return _ChunkedLinearCE.apply(h, head.weight, labels, rows)
    total = h.new_zeros((), dtype=torch.float32)
    for hs, ys in zip(h.split(rows), labels.split(rows)):
        total = total + checkpoint(_ce_sum, hs, head, ys, use_reentrant=False)
    return total


def lm_outputs(
    h: torch.Tensor,
    head: nn.Linear,
    labels: Optional[torch.Tensor],
    return_logits: bool,
    batch_shape: Optional[Tuple[int, int]],
    logit_bias: Optional[torch.Tensor] = None,
) -> CausalLMOutput:
    """Logits and/or CE from final hidden states h [T, D]. ``logit_bias`` [T, V] (fp32), if given, is added to the
    logits (HNetConfig.trie_head); the logits are then materialized in fp32."""
    out = CausalLMOutput()
    if logit_bias is not None:
        logits = head(h).float() + logit_bias
        if labels is not None:
            labels = labels.reshape(-1)
            out.ce_sum = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")
            out.n_targets = (labels != IGNORE_INDEX).sum()
            out.loss = out.ce_sum / out.n_targets.clamp(min=1)
        if return_logits:
            out.logits = logits.view(*batch_shape, -1) if batch_shape is not None else logits
        return out
    if return_logits:
        out.logits = head(h)
    if labels is not None:
        labels = labels.reshape(-1)
        if out.logits is not None:
            out.ce_sum = F.cross_entropy(out.logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="sum")
        else:
            out.ce_sum = chunked_ce_sum(h, head, labels)
        out.n_targets = (labels != IGNORE_INDEX).sum()
        out.loss = out.ce_sum / out.n_targets.clamp(min=1)
    if out.logits is not None and batch_shape is not None:
        out.logits = out.logits.view(*batch_shape, -1)
    return out


def logits_outputs(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    return_logits: bool,
    batch_shape: Optional[Tuple[int, int]],
) -> CausalLMOutput:
    """``lm_outputs`` for fully materialized logits [T, V] (heads with additive per-position terms, e.g. the match
    model's logit bonus; byte vocabularies only, so the logits are small)."""
    out = CausalLMOutput()
    if labels is not None:
        labels = labels.reshape(-1)
        out.ce_sum = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX, reduction="sum")
        out.n_targets = (labels != IGNORE_INDEX).sum()
        out.loss = out.ce_sum / out.n_targets.clamp(min=1)
    if return_logits:
        out.logits = logits.view(*batch_shape, -1) if batch_shape is not None else logits
    return out


class HNetForCausalLM(nn.Module):
    """Byte embeddings -> recursive ``HNet`` backbone -> lm_head (official parameter names)."""

    def __init__(self, cfg: HNetConfig):
        super().__init__()
        self.config = cfg
        d0 = cfg.d_model[0]
        self.embeddings = nn.Embedding(cfg.vocab_size, d0)
        self.ngram_embeddings = HashNgramEmbedding(d0, **cfg.ngram_embed) if cfg.ngram_embed else None
        self.backbone = HNet(cfg, stage_idx=0)
        self.lm_head = nn.Linear(d0, cfg.vocab_size, bias=False)
        self.main_aux_head = nn.Linear(d0, cfg.vocab_size, bias=False) if cfg.dc.main_aux_weight > 0 else None
        self.trie_head = None
        if cfg.trie_head:
            from .trie_head import TrieHead

            assert cfg.n_stages > 0, "trie_head needs a hierarchy (its prefixes are the stage-0 chunks in progress)"
            self.trie_head = TrieHead(d0, cfg.vocab_size, **cfg.trie_head)
        self.chunk_pred_head = None
        if cfg.dc.chunk_pred_k > 0:
            assert cfg.dc.chunk_pred_weight > 0 and cfg.n_stages > 0, "chunk_pred_k needs a weight and a hierarchy"
            d_in = cfg.d_model[cfg.n_stages - 1]  # the innermost stage returns its output at its input width
            self.chunk_pred_head = nn.Linear(d_in, cfg.dc.chunk_pred_k * cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        # running mean / var of the per-chunk hierarchy gain (dc.router_gain_weight); not persistent so the state
        # dict stays identical to the official / dense layouts (re-warms within a few steps after a resume)
        self.register_buffer("gain_mu", torch.zeros(()), persistent=False)
        self.register_buffer("gain_var", torch.ones(()), persistent=False)
        self._param_stage = {}
        for st in self.backbone.stages():
            prefix = "backbone" + ".main_network" * st.stage_idx + "."
            for n, _ in st.named_parameters():
                if st.is_innermost or not n.startswith("main_network."):
                    self._param_stage[prefix + n] = st.stage_idx
        # built under a forked RNG (its parameters are zero-initialized anyway), so every other module's
        # initialization, here and in init_weights(), is identical with and without the match model
        self.match_model = None
        if cfg.match_model:
            from .match import MatchModel

            assert cfg.vocab_size == 256 and cfg.n_stages > 0, "match_model: byte vocabulary and a hierarchy"
            with torch.random.fork_rng(devices=[]):
                self.match_model = MatchModel(d0, **cfg.match_model)

    @property
    def n_stages(self) -> int:
        return self.config.n_stages

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        boundaries: Optional[List[torch.Tensor]] = None,
    ) -> CausalLMOutput:
        """See module docstring. ``boundaries``: optional per-stage [T_s] tensors for the ``external`` router."""
        ids, ctx, batch_shape = pack_inputs(input_ids, cu_seqlens, max_seqlen)
        x = self.embeddings(ids)
        if self.ngram_embeddings is not None:
            x = x + self.ngram_embeddings(ids, ctx)
        cand, dec_add = None, None
        if self.match_model is not None:  # copy candidates from earlier in the same document (no gradient)
            cand = self.match_model.candidates(ids, ctx)
            if self.match_model.enc is not None:
                x = x + self.match_model.enc_embed(cand).to(x.dtype)
            if self.match_model.dec is not None:
                dec_add = self.match_model.dec_embed(cand)
        binfo = None
        vocab_stages = [st for st in self.backbone.stages() if getattr(st, "chunk_vocab", None) is not None]
        if vocab_stages:
            from .chunk_vocab import ByteInfo, hash_bank

            bank = hash_bank(ids, ctx, max(st.chunk_vocab.n_hash for st in vocab_stages),
                             max(st.chunk_vocab.max_len for st in vocab_stages))
            binfo = ByteInfo(bank, torch.arange(ids.numel(), device=ids.device))
        h, bpred = self.backbone(x, ctx, input_ids=ids, boundaries=boundaries, binfo=binfo, dec_add=dec_add)
        trie_bias = trie_len = None
        if self.trie_head is not None:
            r0 = bpred[0]
            assert r0.semantics == "end", "trie_head needs 'end' boundary semantics at stage 0"
            q = r0.main_out if self.trie_head.query == "main" else h
            trie_bias, trie_len = self.trie_head(q, ids, ctx, r0.boundary_mask)
        if self.match_model is not None and self.match_model.gate is not None:
            logits = self.lm_head(h).float() + self.match_model.logit_bonus(h, cand, self.config.vocab_size)
            if trie_bias is not None:
                logits = logits + trie_bias
            out = logits_outputs(logits, labels, return_logits, batch_shape)
        else:
            out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape, logit_bias=trie_bias)
        out.bpred = bpred
        aux_terms = self.backbone.aux_losses(bpred)
        out.aux_loss = self.config.dc.ratio_loss_alpha * sum(loss for loss, _ in aux_terms)
        dc = self.config.dc
        if dc.router_entropy_weight > 0:
            ent = sum(_binary_entropy(r.p).mean() for r in bpred) / len(bpred)
            out.aux_loss = out.aux_loss + dc.router_entropy_weight * ent
        if self.main_aux_head is not None and labels is not None:
            lab = labels.reshape(-1)
            n = (lab != IGNORE_INDEX).sum().clamp(min=1)
            main_ce = chunked_ce_sum(bpred[0].main_out.to(h.dtype), self.main_aux_head, lab) / n
            out.aux_loss = out.aux_loss + dc.main_aux_weight * main_ce
        if self.chunk_pred_head is not None and labels is not None:
            chunk_ce = self._chunk_pred_loss(bpred, ids, labels.reshape(-1), ctx)
            out.aux_loss = out.aux_loss + dc.chunk_pred_weight * chunk_ce
        vocab_ce = {}
        if labels is not None:
            for st, r in zip(self.backbone.stages(), bpred):
                cv = getattr(st, "chunk_vocab", None)
                if cv is not None and cv.pred_weight > 0:
                    vocab_ce[st.stage_idx] = cv.pred_loss(r.inner_out, r.vocab_tgt)
                    out.aux_loss = out.aux_loss + cv.pred_weight * vocab_ce[st.stage_idx]
        stats = {"positions/s0": ids.new_full((), ids.numel(), dtype=torch.float32)}
        if trie_bias is not None:
            with torch.no_grad():
                stats["trie/bonus_rms"] = trie_bias.detach().pow(2).mean().sqrt()
                stats["trie/prefix_len"] = trie_len.float().mean()
        stats.update({f"chunk_vocab_ce/s{s}": v.detach() for s, v in vocab_ce.items()})
        if dc.router_prior_weight > 0 and self.training and bpred and bpred[0].logit is not None:
            prog = self.backbone.progress
            scale = max(0.0, 1.0 - prog / max(dc.router_prior_frac, 1e-6))
            if scale > 0:
                tgt = ((ids == 0x20) | (ids == 0x0A)).float()
                prior = F.binary_cross_entropy_with_logits(bpred[0].logit.float(), tgt)
                out.aux_loss = out.aux_loss + dc.router_prior_weight * scale * prior
                stats["router_prior/s0"] = prior.detach()
        if dc.router_surprise_weight > 0 and self.training and bpred:
            frac = dc.router_surprise_frac
            scale = max(0.0, 1.0 - self.backbone.progress / frac) if frac > 0 else 1.0
            if scale > 0:
                surprise, sstats = self._router_surprise_loss(h, bpred[0], ids, ctx)
                out.aux_loss = out.aux_loss + dc.router_surprise_weight * scale * surprise
                stats.update(sstats)
        if dc.router_gain_weight > 0 and self.training and labels is not None and bpred:
            gain_loss, gstats = self._router_gain_loss(h, bpred[0], ctx, labels.reshape(-1))
            out.aux_loss = out.aux_loss + dc.router_gain_weight * gain_loss
            stats.update(gstats)
        if dc.router_consistency_weight > 0 and bpred and bpred[0].logit is not None:
            cons = self._router_consistency(bpred[0].logit, ids, ctx)
            out.aux_loss = out.aux_loss + dc.router_consistency_weight * cons
            stats["router_consistency/s0"] = cons.detach()
        if self.main_aux_head is not None and labels is not None:
            stats["main_aux_ce"] = main_ce.detach()
        if self.chunk_pred_head is not None and labels is not None:
            stats["chunk_pred_ce"] = chunk_ce.detach()
        for s, (r, (loss, st)) in enumerate(zip(bpred, aux_terms)):
            n_sel = r.boundary_mask.sum().float()
            stats[f"aux/s{s}"] = loss.detach()
            stats.update({f"{k}/s{s}": v for k, v in st.items()})
            stats[f"compression/s{s}"] = r.boundary_mask.numel() / n_sel
            with torch.no_grad():
                p = r.p.detach()
                stats[f"p_mid/s{s}"] = ((p > 0.4) & (p < 0.6)).float().mean()  # undecided boundary decisions
                stats[f"p_margin/s{s}"] = (p - 0.5).abs().mean()  # 0.5 = fully confident, 0 = coin flips
                stats.update({f"{k}/s{s}": v for k, v in r.extra.items()})
            stats[f"positions/s{s + 1}"] = n_sel
        out.stats = stats
        return out

    # ------------------------------------------------------------------ utilities
    def init_weights(self):
        """Official ``HNetForCausalLM.init_weights``: lm_head N(0, initializer_range), embeddings N(0, 1), then
        depth-scaled Linear init of every stage (``HNet.init_weights``)."""
        std = self.config.initializer_range
        nn.init.normal_(self.lm_head.weight, std=std)
        if self.main_aux_head is not None:
            nn.init.normal_(self.main_aux_head.weight, std=std)
        if self.chunk_pred_head is not None:
            nn.init.normal_(self.chunk_pred_head.weight, std=std)
        if self.trie_head is not None:
            self.trie_head.reset_parameters()
        nn.init.normal_(self.embeddings.weight, std=1.0)
        if self.ngram_embeddings is not None:
            self.ngram_embeddings.reset_parameters()
        self.backbone.init_weights(std)
        for st in self.backbone.stages():
            if getattr(st, "chunk_vocab", None) is not None:
                st.chunk_vocab.reset_parameters()
        if self.match_model is not None:
            self.match_model.reset_parameters()

    def _router_consistency(self, logit: torch.Tensor, ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """Within-group variance of the byte-level router logit over positions in the batch that end in the same
        ``router_consistency_n`` bytes, divided by the total variance of the logit (1 - R^2 of the n-gram as a
        predictor of the logit). Scale-free: under rank decisions the forward pass does not depend on the logit scale,
        so an unnormalized penalty is satisfied by shrinking all logits (EXP-020, first launch). Positions too close
        to a segment start are skipped."""
        n = self.config.dc.router_consistency_n
        h, valid = HashNgramEmbedding.hashes(ids, ctx.position_ids, n, salt=424242 + n)
        B = 1 << 20
        idx = torch.remainder(h, B)
        lg = logit.float()
        ones = torch.ones_like(lg) * valid
        tot = torch.zeros(B, device=lg.device, dtype=lg.dtype).index_add_(0, idx, lg * valid)
        cnt = torch.zeros(B, device=lg.device, dtype=lg.dtype).index_add_(0, idx, ones)
        mean = tot / cnt.clamp(min=1)
        dev = ((lg - mean[idx]) ** 2 * valid).sum()
        total = (((lg - (lg * valid).sum() / ones.sum().clamp(min=1)) ** 2) * valid).sum()
        return dev / (total + 1e-6)

    @torch.no_grad()
    def next_byte_entropy(self, h: torch.Tensor) -> torch.Tensor:
        """fp32 [T] entropy (nats) of the model's next-byte distribution softmax(lm_head(h_t)), in row chunks."""
        rows = max(1, CE_CHUNK_ELEMENTS // self.lm_head.out_features)
        out = []
        for hs in h.detach().split(rows):
            lg = self.lm_head(hs).float()
            logp = torch.log_softmax(lg, dim=-1)
            out.append(-(logp.exp() * logp).sum(-1))
        return torch.cat(out)

    def _router_surprise_loss(self, h: torch.Tensor, r: RouterOutput, ids: torch.Tensor, ctx: SeqCtx):
        """dc.router_surprise_weight: BCE of the stage-0 router's decided confidence p toward the top 1/N positions
        of the batch by the entropy rise dH_t = H_t - H_{t-1} of the model's own next-byte distribution (detached).
        Rank-based target (any distribution of H), scale-free prediction (p of the rank decider depends on the
        standardized score only). Returns (loss, detached stats)."""
        free = ctx.position_ids.reshape(-1) != 0  # segment starts are forced boundaries (router_output)
        n_free = int(free.sum())
        if n_free < 2:
            return h.new_zeros((), dtype=torch.float32), {}
        with torch.no_grad():
            H = self.next_byte_entropy(h)
            dH = H - torch.cat([H[:1], H[:-1]])  # within a segment for every free position
            k = max(1, min(n_free - 1, int(round(n_free / float(self.backbone.target_ratio)))))
            idx = torch.nonzero(free).squeeze(1)
            top = torch.topk(dH[idx], k).indices
            tgt = torch.zeros_like(H)
            tgt[idx[top]] = 1.0
        p = r.p.float().clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(p[free], tgt[free])
        with torch.no_grad():
            sel = r.boundary_mask.reshape(-1) & free
            ws = (ids.reshape(-1) == 0x20) | (ids.reshape(-1) == 0x0A)  # diagnostic only, never an input
            stats = {
                "router_surprise/s0": loss.detach(),
                "surprise_overlap/s0": (sel & (tgt > 0)).float().sum() / max(k, 1),  # router boundaries on the teacher's
                "surprise_ws/s0": ws[tgt > 0].float().mean(),  # teacher targets on whitespace (diagnostic)
                "entropy_mean/s0": H.mean(),
            }
        return loss, stats

    def _router_gain_loss(self, h: torch.Tensor, r: RouterOutput, ctx: SeqCtx, labels: torch.Tensor):
        """REINFORCE on the hierarchy's measured usefulness (dc.router_gain_weight): the stage-0 decoder and the LM
        head are rerun on the residual bypass alone; the per-byte gain CE_res - CE_full is averaged over the chunk
        covering each byte, standardized by running statistics, and the router's realized decisions inside each
        chunk are pushed toward (positive advantage) or away from (negative) their current confidence."""
        st = self.backbone
        D = h.shape[-1]
        with torch.no_grad():
            h_res = st.decoder(r.residual_in.to(h.dtype), ctx)[:, :D]
            ce_res = F.cross_entropy(self.lm_head(h_res).float(), labels, ignore_index=IGNORE_INDEX, reduction="none")
            ce_full = F.cross_entropy(self.lm_head(h.detach()).float(), labels, ignore_index=IGNORE_INDEX, reduction="none")
            valid = (labels != IGNORE_INDEX).float()
            gain = (ce_res - ce_full) * valid
            mask = r.boundary_mask.reshape(-1).long()
            cs = torch.cumsum(mask, 0)
            chunk = (cs - mask) if r.semantics == "end" else (cs - 1).clamp(min=0)  # chunk index covering each byte
            n_chunks = int(chunk.max()) + 1
            sums = torch.zeros(n_chunks, device=h.device).index_add_(0, chunk, gain)
            cnt = torch.zeros(n_chunks, device=h.device).index_add_(0, chunk, valid).clamp(min=1)
            mean_gain = sums / cnt
            ema = self.config.dc.router_gain_ema
            batch_mu = mean_gain.mean()
            batch_var = mean_gain.var(unbiased=False)
            self.gain_mu.mul_(ema).add_((1 - ema) * batch_mu)
            self.gain_var.mul_(ema).add_((1 - ema) * batch_var)
            adv = (mean_gain - self.gain_mu) / (self.gain_var.sqrt() + 1e-3)
            adv_t = adv[chunk] * valid
        logc = torch.log(r.selected_probs.reshape(-1).float().clamp(1e-4, 1 - 1e-4))
        loss = -(adv_t * logc).sum() / valid.sum().clamp(min=1)
        return loss, {"router_gain/s0": batch_mu.detach(), "router_gain_std/s0": batch_var.sqrt().detach()}

    def _chunk_pred_loss(self, bpred, ids: torch.Tensor, labels: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """Mean CE of predicting, from the innermost main network's output at each innermost position, the k bytes
        after that position's byte (targets outside the segment are ignored)."""
        k = self.config.dc.chunk_pred_k
        T = ids.numel()
        pos = torch.arange(T, device=ids.device)
        for r in bpred:  # byte position of every innermost position
            pos = pos[r.boundary_mask]
        z = bpred[-1].inner_out
        assert z.shape[0] == pos.numel(), (z.shape, pos.shape)
        seg_end = ctx.cu_seqlens[1:].long()[ctx.seq_idx[0, pos].long()]  # exclusive end of each position's segment
        tgt_pos = pos[:, None] + 1 + torch.arange(k, device=ids.device)[None, :]  # [M, k]
        valid = tgt_pos < seg_end[:, None]
        tgt = torch.where(valid, labels[(tgt_pos - 1).clamp(max=T - 1)], torch.full_like(tgt_pos, IGNORE_INDEX))
        logits = self.chunk_pred_head(z).float().reshape(-1, self.config.vocab_size)
        return F.cross_entropy(logits, tgt.reshape(-1), ignore_index=IGNORE_INDEX)

    def set_progress(self, progress: float):
        """Training progress in [0, 1] (called by the trainer before each step); drives dc.residual_warmup."""
        for st in self.backbone.stages():
            st.progress = float(progress)

    def stage_of_param(self, name: str) -> int:
        """Stage index owning parameter ``name`` (DDP ``module.`` / compile ``_orig_mod.`` prefixes allowed);
        embeddings and lm_head belong to stage 0."""
        name = name.replace("_orig_mod.", "").removeprefix("module.")
        if name.startswith(("embeddings.", "ngram_embeddings.", "lm_head.", "main_aux_head.", "chunk_pred_head.",
                            "trie_head.", "match_model.")):
            return 0
        return self._param_stage[name]

    def blocks(self) -> List[nn.Module]:
        """All residual blocks (activation-checkpointing / compile units), outer encoder first."""
        return [m for m in self.modules() if isinstance(m, Block)]
