"""Prefix output dictionary (HNetConfig.prefix_head): a BPE-style output embedding over the prefixes of the learned
chunks, scored against the context for the k most plausible next bytes only.

A BPE model's LM head is a string-keyed output memory: logit(w) = h . E_out[w], one vector per vocabulary string
(8.4M parameters at L03), and it costs 2 V d FLOPs per token (60% of BPE's compute at L03) because every string is
scored. A byte model's LM head is 256 vectors: it has no output memory for strings at all, and the existing H-Net's
15M table parameters are all on the input side. This module gives the byte model the output memory without the
FLOPs:

    logit_t(b) = W_b . h_t + q_t . K[key(prefix_t, b)]        for b in topk_k(W h_t)   (other bytes: W_b . h_t)

* prefix_t = the bytes of the current learned chunk so far (end semantics: the bytes after the last chunk end at or
  before t; empty right after a chunk end; the last ``max_len`` bytes for longer prefixes), so key(prefix_t, b) is
  the string "chunk so far + b": a node of the trie of the learned units. A word's probability factorizes over its
  bytes, each scored against its own trie node: a hashed hierarchical softmax whose tree is the learned
  segmentation (Morin & Bengio 2005 with the tree given by the tokenizer, at byte granularity).
* q_t = [(A h_t + B m_t) / sqrt(r - 1), c]: h_t = the final byte-level hidden state, m_t = the dechunked main-network
  output (the hierarchy's contribution before the residual is added) and c a constant, so the last table column is
  a context-free spelling prior per trie node and the rest is matched against the context. The m_t term is a direct,
  high-capacity path from the main network to the logits that does not pass through the residual sum or the
  decoder, where the bypass carries 5-40x the hierarchy's signal.
* Only the k bytes the base head ranks highest (detached top-k) are rescored: k lookups of r numbers per byte
  instead of 256, and a deterministic, causal function of the context at inference.

Zero-initialized table: a fresh model computes exactly the base logits. Causal: prefix_t, the boundary decisions,
h_t and m_t depend on bytes <= t. Packed = separate: a segment start is always a chunk end, so prefixes never cross
segments.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_P2 = 0x100000001B3  # FNV prime
_M2 = 0x9E3779B97F4A7C15 - (1 << 64)
_EMPTY = 0x51ED270B27E3A5  # key of the empty prefix


def chunk_prefix_len(boundary_mask: torch.Tensor) -> torch.Tensor:
    """int64 [T]: number of bytes of the current chunk up to and including t (0 where t ends a chunk), for end
    semantics with every segment start selected."""
    T = boundary_mask.shape[0]
    ar = torch.arange(T, device=boundary_mask.device)
    last_end = torch.cummax(torch.where(boundary_mask, ar, torch.full_like(ar, -1)), 0).values
    return torch.where(boundary_mask, torch.zeros_like(ar), ar - last_end)


class PrefixHead(nn.Module):
    def __init__(self, d_model: int, buckets: int = 65536, rank: int = 64, k: int = 16, max_len: int = 16,
                 use_main: bool = True, prior_scale: float = 4.0):
        super().__init__()
        assert rank >= 2 and k >= 1
        self.buckets, self.rank, self.k, self.max_len = int(buckets), int(rank), int(k), int(max_len)
        self.prior_scale = float(prior_scale)
        self.table = nn.Embedding(self.buckets, self.rank)
        self.q_dec = nn.Linear(d_model, self.rank - 1, bias=False)
        self.q_main = nn.Linear(d_model, self.rank - 1, bias=False) if use_main else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.table.weight)
        for lin in (self.q_dec, self.q_main):
            if lin is not None:
                nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)

    def prefix_hash(self, bank: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """int64 [T]: hash of the current chunk's prefix at every byte (bank [n_hash, >= max_len, T] from
        ``chunk_vocab.hash_bank``)."""
        n = chunk_prefix_len(boundary_mask).clamp(max=self.max_len)
        T = boundary_mask.shape[0]
        h = bank[0, (n - 1).clamp(min=0), torch.arange(T, device=bank.device)]
        h = torch.where(n > 0, h, torch.full_like(h, _EMPTY))
        return h * _P2 + n  # the (capped) length disambiguates prefixes longer than max_len

    def keys(self, phash: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """Table rows [T, k] of the strings prefix + b for the candidate bytes ``cand`` [T, k]."""
        x = (phash[:, None] * _P2 + cand + 1) * _M2
        x = x ^ (x >> 29)
        return torch.remainder(x, self.buckets)

    def forward(self, base_logits: torch.Tensor, h: torch.Tensor, main: torch.Tensor, phash: torch.Tensor):
        """base_logits [T, V] (fp32) -> corrected logits [T, V] fp32, and the candidate ids [T, k]."""
        cand = base_logits.detach().topk(self.k, dim=-1).indices  # [T, k]
        rows = self.table(self.keys(phash, cand))  # [T, k, r]
        q = self.q_dec(h.to(self.q_dec.weight.dtype))
        if self.q_main is not None:
            q = q + self.q_main(main.to(self.q_main.weight.dtype))
        q = torch.cat([q.float() * (self.rank - 1) ** -0.5,
                       q.new_full((q.shape[0], 1), self.prior_scale, dtype=torch.float32)], -1)
        score = torch.einsum("tkr,tr->tk", rows.float(), q)
        return base_logits.float().scatter_add(1, cand, score), cand

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def flops_per_byte(d_model: int, rank: int, k: int, use_main: bool, count_elementwise: bool = True) -> float:
        """Forward FLOPs per byte: the query projections, k r-dim dot products, the scatter-add."""
        f = 2 * d_model * (rank - 1) * (2 if use_main else 1) + 2 * k * rank
        return f + ((k + rank) if count_elementwise else 0)


__all__ = ["PrefixHead", "chunk_prefix_len"]
