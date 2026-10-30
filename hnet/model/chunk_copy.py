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
        add = self.proj(v.to(self.proj.weight.dtype)) + self.offset(oi)
        stats = {"chunk_copy_cov": has.float().mean()}
        with torch.no_grad():
            for i, n in enumerate(self.orders):
                stats[f"chunk_copy_o{n}"] = (oi == i).float().mean()
            stats["chunk_copy_add_rms"] = add.detach().float().pow(2).mean().sqrt()
        return x + add.to(x.dtype), stats


# ----------------------------------------------------------------------------- copy-aware compression (g3-copy-02)
N_RUN_BINS = 6  # consecutive verified copy bytes: 0, 1, 2-3, 4-7, 8-15, >= 16
N_STATES = 2 * N_RUN_BINS + 1  # (run bin, next-byte candidate exists) + "a verified copy just broke"


@torch.no_grad()
def copy_state(ids: torch.Tensor, cand: torch.Tensor, ctx: SeqCtx, orders: Sequence[int], min_order: int = 8,
               run_min: int = 4) -> dict:
    """Per byte t (all [T]; causal: bytes <= t and candidates made at <= t only), from the match model's candidates
    ``cand`` [k, T] (``cand[i, t]`` = order-``orders[i]`` candidate for x[t+1], NONE_ID if none):

    * ``run``: number of consecutive bytes ending at t that the longest order>=``min_order`` candidate predicted
      correctly (0 if x[t] was not predicted, or mispredicted);
    * ``has_next``: a longest order>=``min_order`` candidate for x[t+1] exists;
    * ``in_copy`` = run >= run_min and has_next (t sits inside a verified copy that is still going);
    * ``brk``: the copy broke at t (run(t-1) >= run_min, x[t] was predicted and the prediction was wrong);
    * ``state``: index into the learned residual table (run bin x has_next, or the break state).
    """
    from .match import NONE_ID

    rows = [i for i, n in enumerate(orders) if int(n) >= int(min_order)]
    assert rows, f"copy_route: no match order >= min_order {min_order} in {list(orders)}"
    T = ids.numel()
    longest = torch.full((T,), NONE_ID, dtype=torch.long, device=ids.device)
    for i in sorted(rows, key=lambda i: int(orders[i])):  # ascending: a longer order overrides
        longest = torch.where(cand[i] != NONE_ID, cand[i], longest)
    start = ctx.position_ids.reshape(-1) == 0
    prev = torch.cat([longest.new_full((1,), NONE_ID), longest[:-1]])  # the candidate made at t-1 for x[t]
    prev = torch.where(start, torch.full_like(prev, NONE_ID), prev)
    hit = prev == ids.long()
    ar = torch.arange(T, device=ids.device)
    last_miss = torch.cummax(torch.where(hit, torch.full_like(ar, -1), ar), 0).values
    run = ar - last_miss  # hit is False at every segment start, so runs never cross segments
    has_next = longest != NONE_ID
    run_prev = torch.cat([run.new_zeros(1), run[:-1]])
    brk = (run_prev >= run_min) & (prev != NONE_ID) & ~hit & ~start
    in_copy = (run >= run_min) & has_next
    rbin = torch.where(run > 0, torch.floor(torch.log2(run.clamp(min=1).float())).long() + 1, torch.zeros_like(run))
    rbin = rbin.clamp(max=N_RUN_BINS - 1)
    state = torch.where(brk, torch.full_like(rbin, N_STATES - 1), 2 * rbin + has_next.long())
    return {"run": run, "has_next": has_next, "in_copy": in_copy, "brk": brk, "state": state}


class CopyRoute(nn.Module):
    """Copy-aware compression (HNetConfig.copy_route, evolve g3-copy-02): a bias on the stage-0 rank decider's
    standardized score z (hnet.model.routing.decide.QuantileDecider, ``z_bias``)::

        b_t = -kappa [in_copy_t] + kappa_break [brk_t] + w[state_t]        (w: learned, zero-init; standardized units)

    Under rank decisions the batch keeps exactly T/N boundaries, so the ones removed from verified copies (where the
    byte match model already predicts the next bytes, 87% correct on the mix at run >= 4) move to novel text, where
    the main network is needed, and to the bytes where a copy breaks. FLOPs/byte are unchanged; the main network's
    positions are re-allocated by content, per document, which a fixed tokenizer cannot do. The bias is in
    standardized units, so it is scale-free like the decision it shifts (CLAUDE.md rule 4); w enters z directly, is
    learned through the decider's confidence p' (STE and EMA credit) and stops receiving gradient past the decider's
    z clip (bounded).
    """

    def __init__(self, orders: Sequence[int], kappa: float = 2.5, kappa_break: float = 1.0, run: int = 4,
                 min_order: int = 8, learn: bool = True):
        super().__init__()
        self.orders = [int(n) for n in orders]
        self.kappa, self.kappa_break = float(kappa), float(kappa_break)
        self.run_min, self.min_order = int(run), int(min_order)
        assert self.run_min >= 1
        assert any(n >= self.min_order for n in self.orders), (self.orders, self.min_order)
        self.resid = nn.Parameter(torch.zeros(N_STATES)) if learn else None  # 1-D: Adam, no weight decay

    def reset_parameters(self):
        if self.resid is not None:
            nn.init.zeros_(self.resid)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor, cand: torch.Tensor, ctx: SeqCtx):
        """-> (z_bias [T] fp32, the state dict of :func:`copy_state`)."""
        st = copy_state(ids, cand, ctx, self.orders, self.min_order, self.run_min)
        b = st["brk"].float() * self.kappa_break - st["in_copy"].float() * self.kappa
        if self.resid is not None:
            b = b + self.resid.float().index_select(0, st["state"])
        return b, st
