"""Chunk-level copying for the byte-level H-Net (evolve track "chunk-aware copying").

``ChunkCopy`` (HNetConfig.chunk_copy, g3-copy-01): a chunk pointer into the document's past for the main network.

For every stage-0 chunk j (end byte e_j, "end" semantics) and every order n in ``orders`` (bytes), find the most
recent earlier chunk j' < j of the same segment whose end byte e_j' is preceded by the same n bytes as e_j (an exact
index over the byte n-gram hashes, restricted to chunk ends), and take the longest order with a match. The chunk that
followed the earlier occurrence, j' + 1 (<= j, so causal), is the proposal for chunk j + 1: its main-network input
vector x_{j'+1} (encoder state at its end + chunk identity + suffix memory rows) is added to x_j through a
zero-initialised d x d projection, plus a per-order "match" / "no match" offset row (zero-init):

    x_j <- x_j + P x_{j'+1} [match] + O[order of the match, or "none"]

Why (the asymmetry it closes): the byte match model (hnet.model.match) gives the decoder the candidate *next byte*
per order and gives the main network, through the boundary byte's encoder state, at most the candidate first byte of
the next chunk. The token match model of the BPE control (g2-bpe-match) gives its whole layer stack the candidate
*next token*, a whole unit. With the pointer, the main network (most of the H-Net's parameters and FLOPs) sees the
whole next chunk, as a contextual vector of a variable-length learned unit (not a static type embedding), keyed by
the raw bytes before a learned boundary (whatever tokenization the earlier occurrence had).

Offline (whitespace proxy segmentation, mix_v1 val, logbook/evolve/candidates/g3-copy-01/offline.md): an order>=8
pointer exists for 28% of chunks (58% on code) and proposes the exact next chunk for 13% of chunks (42% on code).
At the byte level an offset-aligned pointer adds nothing over the byte index (0.06% of bytes), so there is no
byte-side term: the value of the pointer is what the main network does with it.

Cost: n-gram hashes and one stable sort per order over the M chunk ends (no FLOPs in the counting convention, like
every hash lookup), one d x d projection per chunk (counted in hnet.flops), d^2 + (k + 1) d parameters.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .ctx import SeqCtx
from .match import _SEG_MULT, previous_occurrence
from .ngram import HashNgramEmbedding


@torch.no_grad()
def chunk_pointers(ids: torch.Tensor, ctx: SeqCtx, ctx_inner: SeqCtx, orders: Sequence[int]
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """(src [M] long, order_idx [M] long): for chunk j the index j' + 1 of the chunk that followed the most recent
    earlier chunk end j' of the same segment preceded by the same n bytes (longest n of ``orders`` with a match;
    -1 if none) and the index of that order in ``orders`` (``len(orders)`` if none). ``ids`` / ``ctx`` are the
    byte-level ids and context, ``ctx_inner.outer_index`` the chunk ends."""
    ns = [int(n) for n in orders]
    H, V = HashNgramEmbedding.multi_hashes(ids, ctx.position_ids, ns, [n * 7368787 + 101 for n in ns])
    end = ctx_inner.outer_index
    seg = ctx.seq_idx.reshape(-1).long().index_select(0, end)
    M = end.numel()
    src = torch.full((M,), -1, dtype=torch.long, device=ids.device)
    oi = torch.full((M,), len(ns), dtype=torch.long, device=ids.device)
    for i in range(len(ns)):  # ascending orders: a longer match overrides
        key = H[i].index_select(0, end) ^ (seg * _SEG_MULT)
        prev = previous_occurrence(key, V[i].index_select(0, end))
        hit = prev >= 0
        src = torch.where(hit, prev + 1, src)
        oi = torch.where(hit, torch.full_like(oi, i), oi)
    return src, oi


class ChunkCopy(nn.Module):
    def __init__(self, d_model: int, orders: Sequence[int] = (8, 16, 32)):
        super().__init__()
        self.orders = sorted(int(n) for n in orders)
        assert self.orders and all(n >= 1 for n in self.orders)
        self.proj = nn.Linear(d_model, d_model, bias=False)  # a hidden matrix (Muon), zero-init
        self.offset = nn.Embedding(len(self.orders) + 1, d_model)  # per-order match row + the "none" row (last)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.offset.weight)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, ids: torch.Tensor, ctx: SeqCtx, ctx_inner: SeqCtx):
        """x [M, d] chunk vectors (main-network input) -> (x + pointer copy [M, d], detached stats dict)."""
        src, oi = chunk_pointers(ids, ctx, ctx_inner, self.orders)
        has = src >= 0
        v = x.index_select(0, src.clamp(min=0)) * has.unsqueeze(-1).to(x.dtype)
        ptr = self.proj(v.to(self.proj.weight.dtype))
        add = ptr + self.offset(oi)
        stats = {"chunk_copy_cov": has.float().mean()}
        with torch.no_grad():
            for i, n in enumerate(self.orders):
                stats[f"chunk_copy_o{n}"] = (oi == i).float().mean()
            stats["chunk_copy_add_rms"] = add.detach().float().pow(2).mean().sqrt()
            # the pointer term alone, over chunks that have a pointer (add_rms also holds the learned offset rows)
            n_ptr = has.sum().clamp(min=1)
            stats["chunk_copy_ptr_rms"] = (ptr.detach().float().pow(2).sum(-1) * has).sum().div(n_ptr * ptr.shape[-1]).sqrt()
        return x + add.to(x.dtype), stats
