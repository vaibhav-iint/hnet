"""Offline segmentation probes (no model): is a candidate router *teacher* a tokenizer at all?

    python scripts/analysis/segmentation_probe.py --docs 256 --ratio 6 [--out probe.json]

Two candidates from the first-principles notes, scored with the chunk_stats metrics (whitespace alignment, hit rate,
consistency, types, unigram bits/byte) against the whitespace tokenizer on the same held-out text:

* ``entropy``: branching-entropy spikes (Harris 1955; BLT's patching rule). A hashed byte 5-gram model (counts from
  the first N training-pool docs) gives H_t = entropy of the next byte after position t; a chunk ends at the top
  1/ratio positions of H_t - H_{t-1} (end semantics: the last byte of a chunk is where the next byte is unpredictable).
  Also ``entropy_abs``: top 1/ratio of H_t itself.
* ``mdl``: unigram MDL (segmental marginal likelihood). Units = byte strings of length 1..K hashed into a table of
  log-probabilities; 10 EM iterations of the forward-backward algorithm over the text (the posterior expected unit
  counts re-estimate the table), with a per-unit length prior; the Viterbi segmentation is scored, and its chunk rate
  is reported (MDL does not pin the ratio).
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np

from hnet.analysis.chunk_stats import _alignment, _consistency, _segments, _stats, _whitespace_starts
from hnet.analysis.diagnose import load_docs

MASK = (1 << 61) - 1


def _hash(arr: np.ndarray, n: int, buckets: int) -> np.ndarray:
    """Rolling hash of the n bytes ending at each position (positions < n-1 hash the available prefix)."""
    h = np.zeros(len(arr), dtype=np.int64)
    for k in range(n):
        shifted = np.concatenate([np.full(k, 256, dtype=np.int64), arr[: len(arr) - k].astype(np.int64)]) if k else arr.astype(np.int64)
        h = (h * 1315423911 + shifted + 1) & MASK
    return h % buckets


def score_starts(il, starts, ws, chunks_out, align_out, cons_out):
    chunks_out += _segments(il, starts)
    a = _alignment(starts, ws)
    for i in range(3):
        align_out[i] += a[i]
    for w, c in _consistency(il, starts, ws).items():
        cons_out[w].update(c)


def report(name, chunks, align, cons, n_bytes, ws_chunks):
    st = _stats(chunks, n_bytes)
    cons_types = [w for w, c in cons.items() if sum(c.values()) >= 3]
    consistent = sum(1 for w in cons_types if max(cons[w].values()) == sum(cons[w].values()))
    st.update({
        "name": name,
        "boundaries_on_whitespace": align[0] / max(align[2], 1),
        "whitespace_boundaries_hit": align[1] / max(align[2], 1),
        "consistency": consistent / max(len(cons_types), 1),
        "n_consistency_types": len(cons_types),
        "top": collections.Counter(chunks).most_common(15),
    })
    return st


def entropy_model(train_docs, n: int, buckets: int):
    counts = np.zeros((buckets, 256), dtype=np.float32)
    for d in train_docs:
        arr = np.frombuffer(bytes(d), dtype=np.uint8).astype(np.int64)
        ctx = _hash(arr[:-1], n - 1, buckets)
        np.add.at(counts, (ctx, arr[1:]), 1.0)
    p = (counts + 0.1) / (counts + 0.1).sum(1, keepdims=True)
    H = -(p * np.log2(p)).sum(1)
    return H, n


def entropy_starts(il, H, n, buckets, ratio, absolute=False):
    arr = np.asarray(il, dtype=np.int64)
    ctx = _hash(arr, n - 1, buckets)
    h = H[ctx]  # entropy of the byte AFTER position t
    score = h if absolute else np.concatenate([[h[0]], np.diff(h)])
    k = max(1, int(len(il) // ratio))
    ends = np.sort(np.argpartition(-score, k - 1)[:k])
    starts = sorted({0} | {int(e) + 1 for e in ends if e + 1 < len(il)})
    return starts


def mdl_fit(docs, K: int, buckets: int, iters: int, length_prior: float):
    logq = np.full(buckets, -math.log(buckets), dtype=np.float64)
    hashes = []
    for d in docs:
        arr = np.asarray(d, dtype=np.int64)
        hashes.append(np.stack([_hash(arr, n, buckets) for n in range(1, K + 1)]))  # [K, T]: unit of length n ending at t
    for it in range(iters):
        exp_counts = np.zeros(buckets)
        total_nll = 0.0
        for arr, hk in zip(docs, hashes):
            T = len(arr)
            alpha = np.full(T + 1, -np.inf)
            alpha[0] = 0.0
            for t in range(1, T + 1):
                best = -np.inf
                for n in range(1, min(K, t) + 1):
                    v = alpha[t - n] + logq[hk[n - 1, t - 1]] - length_prior * n
                    best = np.logaddexp(best, v)
                alpha[t] = best
            beta = np.full(T + 1, -np.inf)
            beta[T] = 0.0
            for t in range(T - 1, -1, -1):
                best = -np.inf
                for n in range(1, min(K, T - t) + 1):
                    best = np.logaddexp(best, beta[t + n] + logq[hk[n - 1, t + n - 1]] - length_prior * n)
                beta[t] = best
            logZ = alpha[T]
            total_nll += -logZ
            for t in range(1, T + 1):
                for n in range(1, min(K, t) + 1):
                    b = hk[n - 1, t - 1]
                    w = math.exp(alpha[t - n] + logq[b] - length_prior * n + beta[t] - logZ)
                    exp_counts[b] += w
        logq = np.log((exp_counts + 1e-3) / (exp_counts + 1e-3).sum())
        n_bytes = sum(len(d) for d in docs)
        print(f"  EM iter {it}: {total_nll / n_bytes / math.log(2):.3f} bits/byte (unigram, with length prior)")
    return logq


def mdl_viterbi(il, logq, K, buckets, length_prior):
    arr = np.asarray(il, dtype=np.int64)
    hk = np.stack([_hash(arr, n, buckets) for n in range(1, K + 1)])
    T = len(arr)
    best = np.full(T + 1, -np.inf)
    back = np.zeros(T + 1, dtype=np.int64)
    best[0] = 0.0
    for t in range(1, T + 1):
        for n in range(1, min(K, t) + 1):
            v = best[t - n] + logq[hk[n - 1, t - 1]] - length_prior * n
            if v > best[t]:
                best[t], back[t] = v, n
    starts, t = [], T
    while t > 0:
        t -= back[t]
        starts.append(t)
    return sorted(starts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=256)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--train-docs", type=int, default=4000, help="docs for the n-gram counts / EM (from the same pool)")
    ap.add_argument("--ratio", type=float, default=6.0)
    ap.add_argument("--ngram", type=int, default=5)
    ap.add_argument("--buckets", type=int, default=1 << 20)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--em-iters", type=int, default=6)
    ap.add_argument("--em-docs", type=int, default=200)
    ap.add_argument("--length-prior", type=float, default=0.0)
    ap.add_argument("--skip-mdl", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    all_docs = load_docs(a.train_docs + a.docs, a.max_bytes)
    train, docs = all_docs[: a.train_docs], all_docs[a.train_docs:]
    docs = [list(d) for d in docs]
    n_bytes = sum(len(d) for d in docs)
    ws_chunks = []
    for il in docs:
        ws_chunks += _segments(il, _whitespace_starts(il))
    results = {}
    H, n = entropy_model(train, a.ngram, a.buckets)
    for name, absolute in (("entropy", False), ("entropy_abs", True)):
        chunks, align, cons = [], [0, 0, 0], collections.defaultdict(collections.Counter)
        for il in docs:
            score_starts(il, entropy_starts(il, H, n, a.buckets, a.ratio, absolute), _whitespace_starts(il), chunks, align, cons)
        results[name] = report(name, chunks, align, cons, n_bytes, ws_chunks)
    chunks, align, cons = [], [0, 0, 0], collections.defaultdict(collections.Counter)
    for il in docs:
        ws = _whitespace_starts(il)
        score_starts(il, ws, ws, chunks, align, cons)
    results["whitespace"] = report("whitespace", chunks, align, cons, n_bytes, ws_chunks)
    if not a.skip_mdl:
        print(f"MDL EM on {a.em_docs} docs, K={a.K}")
        logq = mdl_fit([list(d) for d in train[: a.em_docs]], a.K, a.buckets, a.em_iters, a.length_prior)
        chunks, align, cons = [], [0, 0, 0], collections.defaultdict(collections.Counter)
        for il in docs:
            score_starts(il, mdl_viterbi(il, logq, a.K, a.buckets, a.length_prior), _whitespace_starts(il), chunks, align, cons)
        results["mdl"] = report("mdl", chunks, align, cons, n_bytes, ws_chunks)
    keys = ["chunks", "mean_len", "types", "singleton_frac", "unigram_bits_per_byte", "boundaries_on_whitespace",
            "whitespace_boundaries_hit", "consistency"]
    print(f"{'segmentation':14s}" + "".join(f"{k[:14]:>15s}" for k in keys))
    for name, r in results.items():
        print(f"{name:14s}" + "".join(f"{r.get(k, float('nan')):15.3f}" if isinstance(r.get(k), float) else f"{str(r.get(k)):>15s}" for k in keys))
        print("   top:", " ".join(repr(bytes(c).decode('utf-8', 'replace')) for c, _ in r["top"][:12]))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
