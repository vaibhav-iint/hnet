"""In-context match model (HNetConfig.match_model): exact copy candidates from earlier in the same document.

For every byte position t and every order n in ``orders``, find the most recent earlier position t' < t of the
same segment whose n bytes ending at t' equal the n bytes ending at t, and propose the byte that followed it,
``c_n(t) = x[t' + 1]``, as a candidate for x[t + 1] (the match model of PPM / LZ77 / PAQ-style compressors). No
match -> the "none" id 256. The candidates enter the model in three FLOP-light ways, all zero at init:

* ``enc``: a per-order embedding of the candidate byte is added to the byte embedding (the encoder, the router and
  the chunk vectors see "the next bytes are a copy");
* ``dec``: a second per-order embedding is added to the decoder input (next to the residual and the dechunked
  hierarchy);
* ``gate``: a direct logit bonus ``a_n(t) = w_n . h_t + b_n`` on byte c_n(t) (h_t = the final hidden state before the
  LM head), so trusting a match is a one-parameter decision whose confidence depends on the context.

Why: copying is the one capability a language model can have with no training at all. A small model trained on few
tokens has not yet formed induction heads, so it cannot exploit the repeats that make up a large share of code and
of long documents; an exact byte-level index can, from step 0, and it is tokenization-invariant (" foo", "(foo" and
".foo" all match on "foo"), which a token-level copy is not.

Causality: c_n(t) uses bytes <= t only (t' < t, so t' + 1 <= t). Document-aware: matches are restricted to the same
segment, and an n-gram that would cross the segment start matches nothing. Cost: one stable sort per order over the
int64 n-gram hashes of the batch (no FLOPs in the counting convention, like the n-gram table lookups) plus the adds
and the gate dot products (counted in hnet.flops).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from .ctx import SeqCtx
from .ngram import HashNgramEmbedding

NONE_ID = 256
_SEG_MULT = 0x2545F4914F6CDD1D  # odd 64-bit constant mixing the segment id into the key


def previous_occurrence(key: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """int64 [T]: for each position, the most recent earlier position with the same key (both valid), else -1.

    A stable sort by key keeps equal keys in position order, so each position's predecessor in the sorted order is
    its most recent earlier occurrence. Invalid positions get unique keys so they neither match nor break a chain.
    """
    T = key.shape[0]
    ar = torch.arange(T, device=key.device)
    k = torch.where(valid, key, torch.iinfo(torch.int64).min + ar)  # unique sentinels, sorted before everything
    order = torch.sort(k, stable=True).indices
    ks = k[order]
    same = (ks[1:] == ks[:-1]) & valid[order[1:]]
    prev = torch.full((T,), -1, dtype=torch.long, device=key.device)
    prev[order[1:]] = torch.where(same, order[:-1], torch.full_like(order[:-1], -1))
    return prev


@torch.no_grad()
def match_candidates(ids: torch.Tensor, ctx: SeqCtx, orders: Sequence[int], none_id: int = NONE_ID) -> torch.Tensor:
    """[len(orders), T] long: candidate next element per order (``none_id`` where there is no earlier match). Works
    for any integer ids (bytes, or BPE token ids for :class:`TokenMatchModel` with ``none_id = vocab_size``)."""
    ns = [int(n) for n in orders]
    H, V = HashNgramEmbedding.multi_hashes(ids, ctx.position_ids, ns, [n * 15485863 + 77 for n in ns])
    seg = ctx.seq_idx.reshape(-1).long()
    out = []
    for i in range(len(ns)):
        key = H[i] ^ (seg * _SEG_MULT)
        prev = previous_occurrence(key, V[i])
        cand = ids.long().index_select(0, (prev + 1).clamp(min=0, max=ids.numel() - 1))
        out.append(torch.where(prev >= 0, cand, torch.full_like(cand, none_id)))
    return torch.stack(out)


class MatchModel(nn.Module):
    def __init__(self, d_model: int, orders: Sequence[int] = (4, 8, 16, 32), enc: bool = True, dec: bool = True,
                 gate: bool = True):
        super().__init__()
        self.orders = [int(n) for n in orders]
        assert self.orders and all(n >= 1 for n in self.orders)
        k = len(self.orders)
        self.enc = nn.ModuleList(nn.Embedding(NONE_ID + 1, d_model) for _ in range(k)) if enc else None
        self.dec = nn.ModuleList(nn.Embedding(NONE_ID + 1, d_model) for _ in range(k)) if dec else None
        # the logit gate is an output head: named ``lm_head`` so the Muon optimizer puts it in the Adam head group
        self.lm_head = nn.Linear(d_model, k) if gate else None
        self.reset_parameters()

    @property
    def gate(self):
        return self.lm_head

    def reset_parameters(self):
        for tabs in (self.enc, self.dec):
            if tabs is not None:
                for t in tabs:
                    nn.init.zeros_(t.weight)
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def candidates(self, ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        return match_candidates(ids, ctx, self.orders)

    @staticmethod
    def _embed(tabs, cand: torch.Tensor) -> torch.Tensor:
        out = None
        for k, t in enumerate(tabs):
            e = t(cand[k])
            out = e if out is None else out + e
        return out

    def enc_embed(self, cand: torch.Tensor) -> torch.Tensor:
        return self._embed(self.enc, cand)

    def dec_embed(self, cand: torch.Tensor) -> torch.Tensor:
        return self._embed(self.dec, cand)

    def logit_bonus(self, h: torch.Tensor, cand: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """[T, vocab_size] fp32: a_n(t) added at byte c_n(t) for every order with a match (0 elsewhere)."""
        a = self.gate(h.to(self.gate.weight.dtype)).float()  # [T, k]
        bonus = torch.zeros(h.shape[0], vocab_size + 1, device=h.device, dtype=torch.float32)
        col = torch.where(cand.T == NONE_ID, torch.full_like(cand.T, vocab_size), cand.T.clamp(max=vocab_size))
        bonus = bonus.scatter_add(1, col, a)
        return bonus[:, :vocab_size]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class TokenMatchModel(nn.Module):
    """Token-level match model for the dense BPE baseline (``DenseConfig.match_model``): the fairness control for
    :class:`MatchModel` (logbook/evolve/candidates/g2-bpe-match).

    Same index as the byte model (:func:`match_candidates` over BPE token ids, ``none_id = vocab_size``): for every
    token position t and order n, the token that followed the most recent earlier occurrence, in the same segment,
    of the n tokens ending at t. Default orders (1, 2, 4, 8) tokens ~ the byte model's (4, 8, 16, 32) bytes at the
    benchmark's 4.145 bytes/token. The candidates enter the same three ways, all zero at init:

    * ``enc``: added to the token embedding (the byte model's encoder input);
    * ``dec``: added to the residual stream before block ``dec_layer`` (default the last block: the analogue of the
      byte model's decoder input, which sits after the main network and before the last byte-level layers);
    * ``gate``: the logit bonus ``a_n(t) = w_n . h_t + b_n`` on token c_n(t), ``h_t`` = final hidden state.

    Embedding of a candidate at site s (``enc`` / ``dec``)::

        e_s(t) = P_s( sum_n [c_n matched] (1 + g_{s,n}) * E[c_n(t)] ) + sum_n O_s[2n + [c_n matched]]

    E = the model's own input embedding (a learned per-order table over a 32k vocabulary would be 8.4M parameters
    per order and site, vs 257 x d for bytes), P_s a d x d projection per site (zero-init), g_{s,n} a per-order
    diagonal gain (zero-init, so the gain starts at 1; it receives gradient from the first step on, once P_s != 0)
    and O_s a 2k-row table holding, per order, a "no match" row (the byte tables' none row) and a "match" row.
    One projection per site instead of one per (site, order): per-order d x d projections would cost +2.6% params
    and +3.7% FLOPs of bpe_L03; the shared projection costs +0.7% / +1.0% and keeps order identity through g and O.
    """

    def __init__(self, d_model: int, vocab_size: int, n_layers: int, orders: Sequence[int] = (1, 2, 4, 8),
                 enc: bool = True, dec: bool = True, gate: bool = True, dec_layer: Optional[int] = None):
        super().__init__()
        self.orders = [int(n) for n in orders]
        assert self.orders and all(n >= 1 for n in self.orders)
        k, d = len(self.orders), d_model
        self.vocab_size = int(vocab_size)
        self.dec_layer = n_layers - 1 if dec_layer is None else int(dec_layer)
        assert 0 < self.dec_layer < n_layers, f"dec_layer must be in [1, {n_layers}) (0 is the enc site)"
        self.sites = [s for s, on in (("enc", enc), ("dec", dec)) if on]
        self.proj = nn.ModuleDict({s: nn.Linear(d, d, bias=False) for s in self.sites})  # hidden matrices (Muon)
        self.offset = nn.ModuleDict({s: nn.Embedding(2 * k, d) for s in self.sites})  # none / match row per order
        self.gain = nn.ParameterDict({s: nn.Parameter(torch.zeros(k * d)) for s in self.sites})  # 1-D: Adam, no wd
        # the logit gate is an output head: named ``lm_head`` so the Muon optimizer puts it in the Adam head group
        self.lm_head = nn.Linear(d, k) if gate else None
        self.reset_parameters()

    @property
    def gate(self):
        return self.lm_head

    def has(self, site: str) -> bool:
        return site in self.sites

    def reset_parameters(self):
        for s in self.sites:
            nn.init.zeros_(self.proj[s].weight)
            nn.init.zeros_(self.offset[s].weight)
            nn.init.zeros_(self.gain[s])
        if self.gate is not None:
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def candidates(self, ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """[k, T] long: candidate next token per order, ``vocab_size`` where there is no earlier match."""
        return match_candidates(ids, ctx, self.orders, none_id=self.vocab_size)

    def site_embed(self, site: str, cand: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        """[T, d] embedding of the candidates at ``site`` (see class docstring)."""
        d = embedding.embedding_dim
        gain = self.gain[site].view(len(self.orders), d)
        x = off = None
        for i in range(len(self.orders)):
            m = cand[i] != self.vocab_size
            e = embedding(torch.where(m, cand[i], torch.zeros_like(cand[i])))
            e = e * m.unsqueeze(-1).to(e.dtype) * (1.0 + gain[i]).to(e.dtype)
            o = self.offset[site](2 * i + m.long())
            x = e if x is None else x + e
            off = o if off is None else off + o
        proj = self.proj[site]
        return proj(x.to(proj.weight.dtype)).float() + off.float()

    def gate_values(self, h: torch.Tensor, cand: torch.Tensor):
        """(cols [T, k] long, vals [T, k] fp32): the bonus a_n(t) at column c_n(t); 0 (at column 0) without a match."""
        a = self.gate(h.to(self.gate.weight.dtype)).float()  # [T, k]
        m = cand.T != self.vocab_size
        cols = torch.where(m, cand.T, torch.zeros_like(cand.T))
        return cols, a * m.to(a.dtype)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
