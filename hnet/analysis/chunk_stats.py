"""What tokenizer did the router learn? Chunk statistics of a trained H-Net on held-out text.

    python -m hnet.analysis.chunk_stats RUN_DIR [--docs 256] [--max-bytes 2048] [--out stats.json]

Per stage (chunks = byte strings between consecutive boundaries, by the router's semantics), compared with a
whitespace tokenizer (chunk = the bytes up to and including each space/newline) on the same text:

* size: chunks, mean length, types (distinct strings), type/token ratio, singletons;
* vocabulary cost: unigram entropy of the chunk strings in bits per chunk and per byte (the cost of "spelling" the
  text with this vocabulary and no context: lower per byte = more of the text's regularity is inside the chunks);
* word alignment: fraction of boundaries that fall on a whitespace boundary, fraction of whitespace boundaries hit;
* consistency: for whitespace-delimited word types seen >= 3 times, the fraction whose occurrences are always
  segmented the same way (a vocabulary is only useful if the same string becomes the same token);
* the most frequent chunks.
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import torch

from hnet.analysis.diagnose import _pack, load_docs
from hnet.eval.run import load_run_checkpoint


def _segments(ids_list, starts):
    """Chunk byte strings from sorted chunk-start positions over one document (position 0 = BOS, excluded)."""
    out = []
    for a, b in zip(starts, starts[1:] + [len(ids_list)]):
        out.append(bytes(ids_list[a:b]))
    return out


def _starts_from_mask(mask, semantics):
    pos = torch.nonzero(mask).flatten().tolist()
    if semantics == "end":  # a selected position closes a chunk; the next chunk starts right after it
        starts = [p + 1 for p in pos if p + 1 < len(mask)]
    else:
        starts = pos
    return sorted(set([1] + [p for p in starts if p >= 1]))  # the BOS byte is not part of any chunk


def _whitespace_starts(ids_list):
    starts = [1]
    for i, b in enumerate(ids_list):
        if b in (0x20, 0x0A) and i + 1 < len(ids_list):
            starts.append(i + 1)
    return sorted(set(starts))


def _stats(all_chunks, n_bytes):
    cnt = collections.Counter(all_chunks)
    n = len(all_chunks)
    ent = -sum(c / n * math.log2(c / n) for c in cnt.values())
    return {"chunks": n, "mean_len": round(n_bytes / n, 3), "types": len(cnt),
            "type_token_ratio": round(len(cnt) / n, 4), "singleton_frac": round(sum(1 for c in cnt.values() if c == 1) / n, 4),
            "unigram_bits_per_chunk": round(ent, 3), "unigram_bits_per_byte": round(ent * n / n_bytes, 4),
            "top": [(k.decode("utf-8", "replace"), c) for k, c in cnt.most_common(25)]}


def _alignment(starts, ws_starts):
    s, w = set(starts), set(ws_starts)
    return len(s & w), len(s), len(w)


def _consistency(ids_list, starts, ws_starts):
    """Per whitespace-delimited word: the tuple of chunk boundaries inside it (relative offsets)."""
    s = set(starts)
    per_word = collections.defaultdict(collections.Counter)
    for a, b in zip(ws_starts, ws_starts[1:] + [len(ids_list)]):
        word = bytes(ids_list[a:b])
        inner = tuple(p - a for p in range(a + 1, b) if p in s)
        per_word[word][inner] += 1
    return per_word


@torch.no_grad()
def analyze(model, docs):
    stages = [st for st in model.backbone.stages() if not st.is_innermost]
    chunks = [[] for _ in stages]
    ws_chunks = []
    align = [[0, 0, 0] for _ in stages]
    cons = [collections.defaultdict(collections.Counter) for _ in stages]
    n_bytes = 0
    for d in docs:
        ids, cu, ml = _pack([d], "cpu")
        out = model(ids, cu_seqlens=cu, max_seqlen=ml, return_logits=False)
        il = ids.tolist()
        n_bytes += len(il)
        ws = _whitespace_starts(il)
        ws_chunks += _segments(il, ws)
        pos = torch.arange(len(il))
        for k, r in enumerate(out.bpred):
            mask_byte = torch.zeros(len(il), dtype=torch.bool)
            sel = pos[r.boundary_mask.reshape(-1).bool()]
            mask_byte[sel] = True
            starts = _starts_from_mask(mask_byte, getattr(r, "semantics", "start"))
            chunks[k] += _segments(il, starts)
            a = _alignment(starts, ws)
            for i in range(3):
                align[k][i] += a[i]
            for w, c in _consistency(il, starts, ws).items():
                cons[k][w].update(c)
            pos = sel
    rep = {"bytes": n_bytes, "whitespace": _stats(ws_chunks, n_bytes), "stages": []}
    for k in range(len(stages)):
        st = _stats(chunks[k], n_bytes)
        hit, n_s, n_w = align[k]
        st["boundaries_on_whitespace_frac"] = round(hit / max(n_s, 1), 4)
        st["whitespace_boundaries_hit_frac"] = round(hit / max(n_w, 1), 4)
        freq = {w: c for w, c in cons[k].items() if sum(c.values()) >= 3}
        st["word_types_seen_3plus"] = len(freq)
        st["words_segmented_consistently_frac"] = round(
            sum(1 for c in freq.values() if len(c) == 1) / max(len(freq), 1), 4)
        st["dominant_segmentation_share"] = round(
            sum(max(c.values()) / sum(c.values()) for c in freq.values()) / max(len(freq), 1), 4)
        rep["stages"].append(st)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--docs", type=int, default=256)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    model, mcfg, meta = load_run_checkpoint(a.run)
    rep = analyze(model, load_docs(a.docs, a.max_bytes))
    rep["checkpoint"] = meta["checkpoint"]
    txt = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
