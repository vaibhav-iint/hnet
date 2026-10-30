"""Prefix-trie output vocabulary (HNetConfig.trie_head): a FLOP-free, string-keyed output memory for a byte model.

What a BPE head gives its model that a byte head does not: a large output vocabulary. The main network predicts a
vector z and the head scores every whole token, ``logit(w) = z . E_out[w]``; the spelling of w is stored in its row.
A byte-level H-Net predicts the next chunk with its main network, and then the tiny byte decoder has to *spell* that
prediction from compute (the logbook: most of the gap to BPE is in the non-initial letters of words).

By the chain rule a softmax over strings factorizes byte by byte along a trie,
``P(w | z) = prod_t P(w_t | z, w_<t)``, and the factor for byte b after prefix p is a score of the *extended prefix*
p.b against z. This module adds that score to the byte logits with a hashed table of prefix rows:

    logit_t[b] += u_t . sum_k P_k[hash_k(prefix_t . b) mod rows]        u_t = W_u q_t,  q_t = query source

* ``prefix_t`` is the bytes of the chunk in progress: the bytes after the last chunk end at or before t ("end"
  boundary semantics, the stage-0 router's mask), so p is empty at a chunk end and the scores are then the next
  chunk's first byte. Prefixes longer than ``max_len`` use their last ``max_len`` bytes.
* ``query: main`` (default): q_t is the stage-0 dechunked main-network output, i.e. the main network's prediction of
  the next chunk (constant within a chunk up to the EMA): the main network scores spellings directly, the way a BPE
  Transformer scores tokens, without passing through the byte decoder or competing with the residual bypass.
  ``query: final``: q_t is the byte decoder's final state.
* Cost: lookups are free (as the embedding, SPEC convention); per byte one d0 x dim projection and V dot products of
  width dim, counted in ``hnet.flops`` (component ``trie_head``). Parameters: n_hash x rows x dim + d0 x dim.
* Tables are zero-initialized, so a fresh model has exactly the logits of the model without the head; the byte CE
  reaches the tables at step 0 (u != 0) and W_u from step 1 (once the rows are non-zero).
* Causal: prefix_t uses bytes <= t and boundary decisions at positions <= t; every segment starts with a forced
  boundary, so a prefix never crosses a segment start.

The scores are computed in row chunks under activation checkpointing (the gathered rows, [T, V, dim], are never
held for the whole batch).

Fast variant (``topk: k > 0``, generation 3): score only the k bytes the model's own logits rank highest at t
(``base_logits``, detached top-k: the LM head plus any other logit term, e.g. the match model's bonus), with the same
keys (prefix_t . b), tables and query:

    logit_t[b] += u_t . sum_k P_k[hash_k(prefix_t . b) mod rows]     for b in topk_k(base_logits_t), else 0

k gathered rows per table per byte instead of V (4x fewer at k = 64, V = 256), no activation checkpointing (the
[T, k, dim] rows fit), so no recomputed gather: the gather and, above all, the embedding backward over T x V indices
were what made the full head ~1.8x slower per step on an A100 (g1 runs: 1.88 vs 1.02 s mean step time). The
candidate set is a deterministic, causal function of the base logits at t (no extra state at inference); bytes
outside it get no trie score. With k = V it is the full head (same keys, same scores, up to summation order).
``topk: 0`` (default) keeps the full head bit-identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .ctx import SeqCtx
from .ngram import _MULT, _PRIME, HashNgramEmbedding

ELEMENTS_PER_CHUNK = 2**24  # max gathered row elements (rows x V x dim) materialized at once
_EMPTY_SALT = 0x5EED  # hash of the empty prefix (per table k: + k)


def _scores(u: torch.Tensor, H: torch.Tensor, rows: int, cand: torch.Tensor, *weights: torch.Tensor) -> torch.Tensor:
    """u [c, dim] fp32, H [n_hash, c] int64 prefix hashes -> [c, V] fp32 scores."""
    s = None
    for k, w in enumerate(weights):
        key = ((H[k][:, None] * _PRIME) ^ cand[None, :]) * _MULT
        e = F.embedding(torch.remainder(key, rows), w).float()  # [c, V, dim]
        s = e if s is None else s + e
    return torch.einsum("cvd,cd->cv", s, u)


class TrieHead(nn.Module):
    def __init__(self, d_query: int, vocab_size: int, rows: int, dim: int = 32, n_hash: int = 2, max_len: int = 16,
                 query: str = "main", topk: int = 0):
        super().__init__()
        assert query in ("main", "final"), query
        assert 0 <= int(topk) <= int(vocab_size), topk
        self.topk = int(topk)
        self.rows, self.dim, self.n_hash, self.max_len = int(rows), int(dim), int(n_hash), int(max_len)
        self.vocab_size, self.query = int(vocab_size), query
        self.tables = nn.ModuleList(nn.Embedding(self.rows, self.dim) for _ in range(self.n_hash))
        self.query_proj = nn.Linear(d_query, self.dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for t in self.tables:
            nn.init.zeros_(t.weight)
        nn.init.normal_(self.query_proj.weight, std=self.query_proj.in_features ** -0.5)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def prefix_hashes(self, ids: torch.Tensor, ctx: SeqCtx, boundary_mask: torch.Tensor):
        """([n_hash, T] int64 hash of the chunk-in-progress prefix at every byte, [T] prefix length (uncapped))."""
        T = ids.numel()
        pos = torch.arange(T, device=ids.device)
        last = torch.where(boundary_mask, pos, torch.full_like(pos, -1)).cummax(0).values  # last chunk end <= t
        length = pos - last  # 0 at a chunk end; segment starts are forced ends, so last >= segment start
        ns = [n for _ in range(self.n_hash) for n in range(1, self.max_len + 1)]
        salts = [(k + 1) * 15485863 + n * 104723 for k in range(self.n_hash) for n in range(1, self.max_len + 1)]
        bank, _ = HashNgramEmbedding.multi_hashes(ids, ctx.position_ids, ns, salts)
        bank = bank.view(self.n_hash, self.max_len, T)
        n_idx = (length.clamp(1, self.max_len) - 1)
        h = bank.gather(1, n_idx.view(1, 1, T).expand(self.n_hash, 1, T)).squeeze(1)  # [n_hash, T]
        empty = (torch.arange(self.n_hash, device=ids.device, dtype=torch.int64) + _EMPTY_SALT)[:, None]
        h = torch.where((length == 0)[None, :], empty.expand_as(h), h)
        return h, length

    def forward(self, q: torch.Tensor, ids: torch.Tensor, ctx: SeqCtx, boundary_mask: torch.Tensor):
        """q [T, d_query] query source -> ([T, V] fp32 logit bonus, [T] prefix length)."""
        H, length = self.prefix_hashes(ids, ctx, boundary_mask)
        u = self.query_proj(q.to(self.query_proj.weight.dtype) if not torch.is_autocast_enabled() else q).float()
        cand = torch.arange(self.vocab_size, device=ids.device, dtype=torch.int64) + 1
        weights = [t.weight for t in self.tables]
        rows = max(1, ELEMENTS_PER_CHUNK // (self.vocab_size * self.dim))
        grad = torch.is_grad_enabled() and (u.requires_grad or any(w.requires_grad for w in weights))
        outs = []
        for s in range(0, u.shape[0], rows):
            args = (u[s:s + rows], H[:, s:s + rows], self.rows, cand, *weights)
            outs.append(checkpoint(_scores, *args, use_reentrant=False) if grad else _scores(*args))
        return torch.cat(outs, 0), length

    def rescore(self, q: torch.Tensor, ids: torch.Tensor, ctx: SeqCtx, boundary_mask: torch.Tensor,
                base_logits: torch.Tensor):
        """Fast head (``topk`` > 0): q [T, d_query], base_logits [T, V] -> (cand [T, k] int64 = the detached top-k
        bytes of the base logits, vals [T, k] fp32 trie scores of prefix_t . cand, [T] prefix length). The caller adds
        ``vals`` at ``cand`` (``base_logits.scatter_add(1, cand, vals)``)."""
        assert self.topk > 0, "rescore needs topk > 0"
        H, length = self.prefix_hashes(ids, ctx, boundary_mask)
        cand = base_logits.detach().topk(self.topk, dim=-1).indices  # [T, k]
        u = self.query_proj(q.to(self.query_proj.weight.dtype) if not torch.is_autocast_enabled() else q).float()
        c1 = cand + 1  # the full head's candidate code (byte + 1)
        s = None
        for j, t in enumerate(self.tables):
            key = ((H[j][:, None] * _PRIME) ^ c1) * _MULT
            e = F.embedding(torch.remainder(key, self.rows), t.weight).float()  # [T, k, dim]
            s = e if s is None else s + e
        vals = torch.einsum("tkd,td->tk", s, u)
        return cand, vals, length
