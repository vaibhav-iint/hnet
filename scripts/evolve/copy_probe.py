#!/usr/bin/env python
"""Offline probe for evolve g3-copy-01 / g3-copy-02 on mix_v1 validation bytes (first 2 MB of each set, 8512-byte
windows), with a whitespace PROXY segmentation standing in for a word-basin router (analysis only; no model sees it):
chunks end at whitespace bytes, at most 12 bytes long.

    PYTHONPATH=. python scripts/evolve/copy_probe.py mix_v1_val mix_v1_val-code mix_v1_val-en

g3-copy-01 (chunk pointer): at every chunk end, the most recent earlier chunk end of the window with the same n bytes
before it; how often the chunk after it equals the actual next chunk, and what an offset-aligned byte pointer
anchored there would add over the byte match index (orders >= 8). g3-copy-02 (copy-aware compression): the share of
bytes inside a verified copy (run >= R correct longest order>=8 predictions and a candidate for the next byte), the
share of proxy boundaries inside such copies, and the novel-text compression if those boundaries moved elsewhere."""
import sys
import numpy as np, torch
from hnet.config import BOS_ID, EOS_ID
from hnet.data import shards as S
from hnet.model.ctx import SeqCtx
from hnet.model.match import match_candidates, previous_occurrence
from hnet.model.ngram import HashNgramEmbedding

WIN = 8512
root = "data"
NB = int(2_000_000)
for stem in sys.argv[1:]:
    by = np.asarray(S.read_byte_shard(f"{root}/mix_v1/bytes/{stem}_000000.bytes.bin"))[:NB].astype(np.int64)
    T = len(by)
    starts = np.concatenate([np.arange(0, T, WIN), np.flatnonzero(by == BOS_ID), np.flatnonzero(by == EOS_ID) + 1])
    cu = np.unique(np.concatenate([[0], starts[(starts > 0) & (starts < T)], [T]])).astype(np.int32)
    ctx = SeqCtx.from_cu_seqlens(torch.from_numpy(cu), None, T)
    ids = torch.from_numpy(by)
    C = match_candidates(ids, ctx, [4, 8, 16, 32]).numpy()
    nxt = np.concatenate([by[1:], [EOS_ID]])
    segstart = np.zeros(T, bool)
    segstart[cu[:-1]] = True
    tgt = ~np.concatenate([segstart[1:], [True]])
    lb = np.full(T, 256)
    for r in [1, 2, 3]:
        lb = np.where(C[r] != 256, C[r], lb)
    ok_b = (lb == nxt) & tgt
    ws = (by == 32) | (by == 10) | (by == 9)
    bnd = ws.copy()
    bnd[segstart] = True
    # cap chunk length at 12 (vectorised approximately: cut every 12th byte of a non-ws run)
    cs = np.cumsum(bnd)
    start_of = np.zeros(T, np.int64)
    last = np.maximum.accumulate(np.where(bnd, np.arange(T), 0))
    off = np.arange(T) - last
    bnd = bnd | ((off > 0) & (off % 12 == 0))
    E = np.flatnonzero(bnd)
    print(f"== {stem}: {T} bytes, proxy compression {T/len(E):.2f}")
    seg = ctx.seq_idx.reshape(-1).long()
    J = np.cumsum(bnd) - 1
    o = np.arange(T) - E[J]
    for n in [8, 16]:
        H, V = HashNgramEmbedding.multi_hashes(ids, ctx.position_ids, [n], [n * 31 + 7])
        Et = torch.from_numpy(E)
        key = (H[0] ^ (seg * 0x2545F4914F6CDD1D))[Et]
        prev = previous_occurrence(key, V[0][Et]).numpy()
        has = prev >= 0
        a = np.where(has[J], E[np.maximum(prev[J], 0)], -1)
        q = a + 1 + o
        cand = np.where(a >= 0, by[np.clip(q, 0, T - 1)], 256)
        eq = np.ones(T, bool)
        m = (a >= 0) & (o >= 1)
        eq[m] = by[m] == by[np.clip(a + o, 0, T - 1)][m]
        bad = np.cumsum(~eq)
        bad_since = bad - bad[E[J]]
        cand = np.where(bad_since == 0, cand, 256)
        okc = (cand == nxt) & tgt
        hc = (cand != 256) & tgt
        cov = hc.sum() / tgt.sum()
        acc = okc.sum() / max(hc.sum(), 1)
        jj = np.flatnonzero(has[:-1])[:100000]
        def chunk(i):
            return by[E[i] + 1:E[i + 1] + 1].tobytes() if i + 1 < len(E) else b""
        same = np.mean([chunk(prev[j]) == chunk(j) for j in jj]) if len(jj) else 0
        print(f"  chunk ptr o{n}: chunks with ptr {has.mean():.3f}, next chunk equal {same:.3f}; byte cov {cov:.3f} "
              f"acc {acc:.3f}; ptr ok & byte-idx not {(okc & ~ok_b).sum()/tgt.sum():.4f}; byte ok & ptr not "
              f"{(ok_b & ~okc).sum()/tgt.sum():.4f}; byte-idx ok {ok_b.sum()/tgt.sum():.3f}")
        f = bnd & tgt
        print(f"    first byte of next chunk: ptr ok {(okc & f).sum()/f.sum():.3f}  byte-idx ok {(ok_b & f).sum()/f.sum():.3f}"
              f"  either {((okc | ok_b) & f).sum()/f.sum():.3f}")
        # whole next chunk known at its boundary (the main network's view): ptr proposes the full chunk correctly
        print(f"    whole next chunk correct from ptr: {same * has[:-1].mean():.3f} of chunks")
    prevc = np.concatenate([[256], lb[:-1]])
    hit = (prevc == by) & ~segstart
    # run length of consecutive hits
    idx = np.arange(T)
    lastmiss = np.maximum.accumulate(np.where(~hit, idx, 0))
    runl = idx - lastmiss
    for R in [2, 4, 8]:
        inc = (runl >= R) & (lb != 256)
        brk = (np.concatenate([[0], runl[:-1]]) >= R) & ~hit & (prevc != 256)
        nb = bnd.sum()
        print(f"  R={R}: in_copy {inc.mean():.3f} of bytes; proxy boundaries inside {(bnd & inc).sum()/nb:.3f}; "
              f"breaks {brk.mean():.4f}; next byte ok | in_copy {ok_b[inc & tgt].mean():.3f}; novel-text "
              f"compression if copy spans hold no boundaries {(T - inc.sum())/len(E):.2f} (vs {T/len(E):.2f})")
