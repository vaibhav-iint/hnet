"""A vocabulary for learned chunks (HNetConfig.dc.chunk_vocab): learned tokenization with a token embedding table.

A BPE model gets word identity for free: its embedding table and LM head are a FLOP-free memory of ~50k strings and
at small scale most of its parameters (tiny_bpe_2m: 12.9M of 15M). An H-Net has to compute word identity from bytes
with its FLOPs, and the logbook's mechanism study shows that at matched compute it cannot: 84% of the gap to BPE is
in the letters of words, the chunk vector entering the main network is the boundary byte's state rather than a
summary of the chunk, and the main network's output is less predictive than its input (no word-level signal).

This module treats the router's chunks as tokens:

* input side: the bytes of each chunk (capped at ``max_len``, the last ``max_len`` bytes for longer chunks) are hashed
  with ``n_hash`` independent rolling hashes into zero-initialized tables and the rows are added to the chunk vector
  that enters the main network (BLT-style hashed embeddings, but of data-dependent, learned chunks). Lookups cost no
  FLOPs, so the tables are matched to the BPE embedding in parameters, not FLOPs.
* output side (``pred_weight`` > 0): the main network's output at chunk j predicts the hash bucket (first hash) of
  chunk j + 1 of the same segment with a sampled softmax over the buckets present in the batch (``n_neg`` sampled
  candidates plus the positive), using the input table as the output embedding (tied). This is BPE's training
  objective over the learned segmentation: a direct chunk-level training signal for the main network, which the
  byte-level loss did not provide.

Requires "end" boundary semantics (a chunk's content is known at its boundary) and byte positions for every stage
position (``ByteInfo``); the hash bank is computed once per batch at the byte level and indexed by byte position, so
stage s > 0 chunks (spanning several stage-0 chunks) are hashed by their byte content as well.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ctx import SeqCtx
from .ngram import HashNgramEmbedding

IGNORE = -100


@dataclass
class ByteInfo:
    """Byte-level hash bank [n_hash, max_len, T_bytes] (int64) and the byte position of every position at the
    current stage [T_stage]."""

    bank: torch.Tensor
    byte_pos: torch.Tensor

    def select(self, idx: torch.Tensor) -> "ByteInfo":
        return ByteInfo(self.bank, self.byte_pos.index_select(0, idx))


def hash_bank(ids: torch.Tensor, ctx: SeqCtx, n_hash: int, max_len: int) -> torch.Tensor:
    """[n_hash, max_len, T]: hash k of the n bytes ending at t, for n = 1..max_len (n-1 = row index)."""
    ns = [n for _ in range(n_hash) for n in range(1, max_len + 1)]
    salts = [(k + 1) * 104729 + n * 7919 for k in range(n_hash) for n in range(1, max_len + 1)]
    h, _ = HashNgramEmbedding.multi_hashes(ids, ctx.position_ids, ns, salts)
    return h.view(n_hash, max_len, -1)


class ChunkVocab(nn.Module):
    def __init__(self, d_model: int, buckets: int = 65536, n_hash: int = 2, max_len: int = 16,
                 pred_weight: float = 0.0, n_neg: int = 4096):
        super().__init__()
        self.buckets, self.n_hash, self.max_len = int(buckets), int(n_hash), int(max_len)
        self.pred_weight, self.n_neg = float(pred_weight), int(n_neg)
        self.tables = nn.ModuleList(nn.Embedding(self.buckets, d_model) for _ in range(self.n_hash))
        # output side: project the inner output to the table width; logits = (W z) . E_0[c] / sqrt(d)
        self.pred_proj = nn.Linear(d_model, d_model, bias=False) if self.pred_weight > 0 else None
        self.scale = d_model ** -0.5
        self.reset_parameters()

    def reset_parameters(self):
        for t in self.tables:
            nn.init.zeros_(t.weight)
        if self.pred_proj is not None:
            nn.init.normal_(self.pred_proj.weight, std=self.pred_proj.in_features ** -0.5)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def chunk_ids(self, binfo: ByteInfo, ctx: SeqCtx, ctx_inner: SeqCtx):
        """Bucket ids [n_hash, M] of the chunks that end at the selected positions (``ctx_inner.outer_index``)."""
        idx = ctx_inner.outer_index
        byte_end = binfo.byte_pos.index_select(0, idx)  # byte position of each chunk's last byte
        first = ctx_inner.position_ids == 0
        seg_start = binfo.byte_pos.index_select(0, ctx.cu_seqlens[:-1].long()[ctx_inner.seq_idx[0].long()])
        prev_end = torch.roll(byte_end, 1)
        length = torch.where(first, byte_end - seg_start + 1, byte_end - prev_end).clamp(1, self.max_len)
        h = binfo.bank[:, length - 1, byte_end]  # [n_hash, M]
        return torch.remainder(h, self.buckets)

    def embed(self, bucket: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Summed table rows for bucket ids [n_hash, M] -> [M, d_model]."""
        out = None
        for k, t in enumerate(self.tables):
            e = t(bucket[k])
            out = e if out is None else out + e
        return out.to(dtype)

    @staticmethod
    def next_targets(bucket0: torch.Tensor, ctx_inner: SeqCtx) -> torch.Tensor:
        """For chunk j: the first-hash bucket of chunk j + 1 in the same segment, else IGNORE."""
        nxt = torch.roll(bucket0, -1)
        last = torch.roll(ctx_inner.position_ids == 0, -1)  # j is the last chunk of its segment
        last[-1] = True
        return torch.where(last, torch.full_like(nxt, IGNORE), nxt)

    def pred_loss(self, z: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Sampled-softmax next-chunk loss. z [M, d_model] inner outputs (the inner stage returns its input width),
        tgt [M] bucket ids or IGNORE."""
        valid = tgt != IGNORE
        if not bool(valid.any()):
            return z.new_zeros(())
        E = self.tables[0].weight
        q = self.pred_proj(z[valid].to(E.dtype)) * self.scale  # [m, d]
        t = tgt[valid]
        cand = torch.unique(t)
        if cand.numel() > self.n_neg:
            cand = cand[torch.randperm(cand.numel(), device=cand.device)[: self.n_neg]]
        pos = (q * E[t]).sum(-1, keepdim=True)  # [m, 1]
        neg = q @ E[cand].T  # [m, C]
        neg = neg.masked_fill(cand[None, :] == t[:, None], float("-inf"))  # the positive is in column 0 only
        logits = torch.cat([pos, neg], dim=1)
        return F.cross_entropy(logits.float(), torch.zeros(len(t), dtype=torch.long, device=z.device))
