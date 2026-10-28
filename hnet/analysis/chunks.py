"""Chunking diagnostics for trained H-Nets, comparable to the blog's chunk-viewer statistics (EXP-001).

For each text we run the model on ``[BOS] + utf8 bytes`` and recover the byte offsets where every stage starts a
new chunk, honouring each router's boundary semantics ("start": the boundary byte is the first byte of a chunk;
"end": it is the last byte). Stage s > 0 boundaries live on stage-(s-1) chunk positions and are mapped back to
bytes. Statistics per stage:

* bytes_per_chunk
* pct_mid_utf8        cuts that split a multi-byte UTF-8 character
* pct_word_internal   cuts between two ASCII alphanumerics
* pct_cut_after_space chunks that end right after whitespace ("word␣")
* pct_cut_before_space chunks that start with whitespace ("␣word")
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch

from hnet.config import BOS_ID


@torch.no_grad()
def chunk_cuts(model, text: str, device="cpu") -> List[List[int]]:
    """-> per stage, sorted byte offsets (into ``text.encode()``) where a new chunk starts (0 excluded)."""
    data = text.encode("utf-8")
    ids = torch.tensor([BOS_ID] + list(data), dtype=torch.long, device=device)
    T = ids.numel()
    cu = torch.tensor([0, T], dtype=torch.int32, device=device)
    out = model(ids, cu_seqlens=cu, max_seqlen=T, return_logits=False)
    positions = torch.arange(T, device=device)  # positions of the current stage's sequence, in byte coordinates
    cuts = []
    for r in out.bpred:
        mask = r.boundary_mask.reshape(-1).bool()
        b = positions[mask]
        semantics = getattr(r, "semantics", "start")
        starts = b + 1 if semantics == "end" else b  # byte position (with BOS at 0) where a new chunk begins
        text_cuts = (starts - 1).tolist()  # drop the BOS offset
        cuts.append(sorted(c for c in text_cuts if 0 < c < len(data)))
        positions = b  # next stage runs on this stage's selected positions
    return cuts


def summarize_cuts(texts: Sequence[str], cuts_per_text: Sequence[Sequence[int]]) -> Dict[str, float]:
    n_bytes = n_cuts = mid = wint = after = before = 0
    for text, cuts in zip(texts, cuts_per_text):
        data = text.encode("utf-8")
        n_bytes += len(data)
        for c in cuts:
            n_cuts += 1
            prev, nxt = data[c - 1], data[c]
            mid += 0x80 <= nxt < 0xC0
            after += prev in b" \n\t"
            before += nxt in b" \n\t"
            wint += prev < 128 and nxt < 128 and chr(prev).isalnum() and chr(nxt).isalnum()
    n_chunks = n_cuts + len(texts)
    pct = (lambda x: 100.0 * x / n_cuts) if n_cuts else (lambda x: float("nan"))
    return dict(bytes_per_chunk=n_bytes / max(n_chunks, 1), pct_mid_utf8=pct(mid), pct_word_internal=pct(wint),
                pct_cut_after_space=pct(after), pct_cut_before_space=pct(before), n_cuts=n_cuts)


def chunk_report(model, texts: Sequence[str], device="cpu") -> Dict[str, Dict[str, float]]:
    """-> {"s0": stats, "s1": stats, ...} over all texts."""
    per_text = [chunk_cuts(model, t, device) for t in texts]
    n_stages = len(per_text[0]) if per_text else 0
    return {f"s{s}": summarize_cuts(texts, [c[s] for c in per_text]) for s in range(n_stages)}


def render(text: str, cuts: Sequence[int], max_bytes: int = 120) -> str:
    """Segmentation of the first ``max_bytes`` bytes as ``a|b|c`` with ␣ for spaces."""
    data = text.encode("utf-8")[:max_bytes]
    parts, prev = [], 0
    for c in [c for c in cuts if c < len(data)] + [len(data)]:
        parts.append(data[prev:c].decode("utf-8", "replace").replace(" ", "␣").replace("\n", "⏎"))
        prev = c
    return "|".join(parts)
