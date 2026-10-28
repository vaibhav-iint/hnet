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
        self.backbone = HNet(cfg, stage_idx=0)
        self.lm_head = nn.Linear(d0, cfg.vocab_size, bias=False)
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
        h, bpred = self.backbone(self.embeddings(ids), ctx, input_ids=ids, boundaries=boundaries)
        out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape)
        out.bpred = bpred
        aux_terms = self.backbone.aux_losses(bpred)
        out.aux_loss = self.config.dc.ratio_loss_alpha * sum(loss for loss, _ in aux_terms)
        stats = {"positions/s0": ids.new_full((), ids.numel(), dtype=torch.float32)}
        for s, (r, (loss, st)) in enumerate(zip(bpred, aux_terms)):
            n_sel = r.boundary_mask.sum().float()
            stats[f"aux/s{s}"] = loss.detach()
            stats.update({f"{k}/s{s}": v for k, v in st.items()})
            stats[f"compression/s{s}"] = r.boundary_mask.numel() / n_sel
            stats[f"positions/s{s + 1}"] = n_sel
        out.stats = stats
        return out

    # ------------------------------------------------------------------ utilities
    def init_weights(self):
        """Official ``HNetForCausalLM.init_weights``: lm_head N(0, initializer_range), embeddings N(0, 1), then
        depth-scaled Linear init of every stage (``HNet.init_weights``)."""
        std = self.config.initializer_range
        nn.init.normal_(self.lm_head.weight, std=std)
        nn.init.normal_(self.embeddings.weight, std=1.0)
        self.backbone.init_weights(std)

    def stage_of_param(self, name: str) -> int:
        """Stage index owning parameter ``name`` (DDP ``module.`` / compile ``_orig_mod.`` prefixes allowed);
        embeddings and lm_head belong to stage 0."""
        name = name.replace("_orig_mod.", "").removeprefix("module.")
        if name.startswith(("embeddings.", "lm_head.")):
            return 0
        return self._param_stage[name]

    def blocks(self) -> List[nn.Module]:
        """All residual blocks (activation-checkpointing / compile units), outer encoder first."""
        return [m for m in self.modules() if isinstance(m, Block)]
