"""Hashed byte n-gram embeddings for the byte-level stage (BLT-style, Pagnoni et al. 2024, Sec. 3.2.1).

For every byte position t and every n in ``ns``, the n bytes ending at t are hashed into a per-n table and the rows are
added to the byte embedding: ``e_t = E[b_t] + sum_n E_n[hash(b_{t-n+1..t}) mod buckets_n]``.

Why: a BPE model gets a large vocabulary embedding for free (a lookup costs no FLOPs), which at small scale is a big
share of its parameters. Cartesia's H-Net ladder matches the outer networks to the BPE embedding + LM head in both
parameters and FLOPs; parameter-rich, FLOP-free n-gram tables are one way to do that.

* Causal: only bytes at or before t are used. Document-aware: an n-gram that would cross the start of its segment
  (``ctx.position_ids[t] < n - 1``) contributes nothing.
* Tables are zero-initialized, so a fresh model computes exactly the plain byte embedding.
* The hash is a fixed multiplicative rolling hash over (byte + 1) in int64 (wrap-around arithmetic), with a
  different multiplier per n, reduced modulo the table size.
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn

from .ctx import SeqCtx

_MULT = 0x9E3779B97F4A7C15 - (1 << 64)  # golden-ratio constant as a signed int64
_PRIME = 1_000_003


class HashNgramEmbedding(nn.Module):
    def __init__(self, d_model: int, ns: Sequence[int], buckets: Union[int, Sequence[int]]):
        super().__init__()
        self.ns = [int(n) for n in ns]
        assert self.ns and all(n >= 2 for n in self.ns), "n-gram sizes must be >= 2 (n = 1 is the byte embedding)"
        bk = [int(buckets)] * len(self.ns) if isinstance(buckets, int) else [int(b) for b in buckets]
        assert len(bk) == len(self.ns), "one bucket count per n"
        self.buckets = bk
        self.tables = nn.ModuleList(nn.Embedding(b, d_model) for b in bk)
        self.reset_parameters()

    def reset_parameters(self):
        for t in self.tables:
            nn.init.zeros_(t.weight)

    @staticmethod
    def hashes(ids: torch.Tensor, position_ids: torch.Tensor, n: int, salt: int) -> tuple:
        """(int64 hash of the n bytes ending at each position, bool valid mask) for packed ids [T]."""
        T = ids.shape[0]
        h = torch.full_like(ids, salt, dtype=torch.int64)
        x = ids.long() + 1
        for k in range(n):  # byte t - k
            shifted = torch.zeros_like(x) if k >= T else torch.cat([x.new_zeros(k), x[: T - k]])
            h = (h * _PRIME) ^ shifted
        h = h * _MULT
        return h, position_ids >= n - 1

    def forward(self, ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """ids: int [T] (packed); returns the summed n-gram embeddings [T, d_model]."""
        assert ids.dim() == 1, "HashNgramEmbedding expects packed ids [T]"
        out = None
        for i, (n, table, b) in enumerate(zip(self.ns, self.tables, self.buckets)):
            h, valid = self.hashes(ids, ctx.position_ids, n, salt=n * 7919 + 1)
            idx = torch.remainder(h, b)
            e = table(idx) * valid.unsqueeze(-1).to(table.weight.dtype)
            out = e if out is None else out + e
        return out

    def num_params(self) -> int:
        return sum(b * t.embedding_dim for b, t in zip(self.buckets, self.tables))
