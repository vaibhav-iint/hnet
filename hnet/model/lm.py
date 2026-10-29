"""Causal LM wrappers' shared I/O (packing, loss, outputs) and the byte-level ``HNetForCausalLM``.

Forward API (shared with the dense models, see SPEC "Model API")::

    out = model(input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True)

input_ids: LongTensor [T] (packed; segments given by ``cu_seqlens``, default one segment) or [B, L] (each row
one segment). labels: same shape, ``IGNORE_INDEX`` (-100) where there is no target.

Host syncs per forward: none here when ``max_seqlen`` is given (or input is [B, L]); otherwise one to compute it.
Each H-Net stage adds exactly one in its chunker (see ``SelectChunker``).
"""

from __future__ import annotations

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


def chunked_ce_sum(h: torch.Tensor, head: nn.Linear, labels: torch.Tensor) -> torch.Tensor:
    """fp32 sum of CE(head(h), labels) holding at most ~CE_CHUNK_ELEMENTS logits at once (chunks are
    recomputed in backward via activation checkpointing)."""
    rows = max(1, CE_CHUNK_ELEMENTS // head.out_features)
    if h.shape[0] <= rows:
        return _ce_sum(h, head, labels)
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
) -> CausalLMOutput:
    """Logits and/or CE from final hidden states h [T, D]."""
    out = CausalLMOutput()
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
        self.chunk_pred_head = None
        if cfg.dc.chunk_pred_k > 0:
            assert cfg.dc.chunk_pred_weight > 0 and cfg.n_stages > 0, "chunk_pred_k needs a weight and a hierarchy"
            d_in = cfg.d_model[cfg.n_stages - 1]  # the innermost stage returns its output at its input width
            self.chunk_pred_head = nn.Linear(d_in, cfg.dc.chunk_pred_k * cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        self._param_stage = {}
        for st in self.backbone.stages():
            prefix = "backbone" + ".main_network" * st.stage_idx + "."
            for n, _ in st.named_parameters():
                if st.is_innermost or not n.startswith("main_network."):
                    self._param_stage[prefix + n] = st.stage_idx

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
        binfo = None
        vocab_stages = [st for st in self.backbone.stages() if getattr(st, "chunk_vocab", None) is not None]
        if vocab_stages:
            from .chunk_vocab import ByteInfo, hash_bank

            bank = hash_bank(ids, ctx, max(st.chunk_vocab.n_hash for st in vocab_stages),
                             max(st.chunk_vocab.max_len for st in vocab_stages))
            binfo = ByteInfo(bank, torch.arange(ids.numel(), device=ids.device))
        h, bpred = self.backbone(x, ctx, input_ids=ids, boundaries=boundaries, binfo=binfo)
        out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape)
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
        stats.update({f"chunk_vocab_ce/s{s}": v.detach() for s, v in vocab_ce.items()})
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
        nn.init.normal_(self.embeddings.weight, std=1.0)
        if self.ngram_embeddings is not None:
            self.ngram_embeddings.reset_parameters()
        self.backbone.init_weights(std)
        for st in self.backbone.stages():
            if getattr(st, "chunk_vocab", None) is not None:
                st.chunk_vocab.reset_parameters()

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

    def set_progress(self, progress: float):
        """Training progress in [0, 1] (called by the trainer before each step); drives dc.residual_warmup."""
        for st in self.backbone.stages():
            st.progress = float(progress)

    def stage_of_param(self, name: str) -> int:
        """Stage index owning parameter ``name`` (DDP ``module.`` / compile ``_orig_mod.`` prefixes allowed);
        embeddings and lm_head belong to stage 0."""
        name = name.replace("_orig_mod.", "").removeprefix("module.")
        if name.startswith(("embeddings.", "ngram_embeddings.", "lm_head.", "main_aux_head.", "chunk_pred_head.")):
            return 0
        return self._param_stage[name]

    def blocks(self) -> List[nn.Module]:
        """All residual blocks (activation-checkpointing / compile units), outer encoder first."""
        return [m for m in self.modules() if isinstance(m, Block)]
