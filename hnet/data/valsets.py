"""Fixed validation sets with identical documents (and identical scored bytes) for byte and token models.

Protocol ("whole-doc, aligned chunks"), used for every ValSet:
 1. Documents: the first complete documents of the val shard (both a leading <|endoftext|> and a following one
    inside the shard; the shard's leading/trailing fragments are skipped), in shard order starting at
    ``start_doc``, until their text reaches ``n_bytes_target`` UTF-8 bytes (whole docs only, so the last doc
    may overshoot). Empty docs are skipped.
 2. Chunks: each document's GPT-2 tokens are cut greedily, *at token boundaries*, into chunks such that the
    token segment ``1 + n_chunk_tokens <= token_seq_len`` and the byte segment ``1 + n_chunk_bytes <=
    byte_seq_len``. Both models therefore split a long doc at exactly the same text positions, with no
    context carried across chunks (each chunk is its own segment in both models).
 3. Segments: chunk c of a doc is fed as ``[prefix] + chunk`` where prefix is the doc start marker for c = 0
    (tokens: <|endoftext|>, bytes: BOS) and otherwise the last element (token / byte) of chunk c-1 (context
    only, not scored, so every text element is scored exactly once). Targets are the chunk's elements:
    tokens -> its tokens, bytes -> its bytes. The final "end of document" prediction (EOT / EOS) is NOT a target
    on either side (labels = IGNORE_INDEX), matching the SPEC bpb definition.
 4. Packing: segments are packed in order into fixed windows (token_seq_len / byte_seq_len), never splitting a
    segment (deterministic first-fit over the last 64 open windows); window tails are padding segments
    with IGNORE labels. ``windows_per_batch`` windows form one
    SPEC packed micro-batch.
Result: sum over targets of NLL / ln2 / n_bytes is the bpb on exactly the same text for both model types,
with ``n_bytes`` = number of byte targets = sum of UTF-8 lengths of token targets (both equal ``ValSet.n_bytes``).
Each batch carries ``score_mask`` (labels != IGNORE; for bytes this is exactly "target is a real byte 0..253")
and ``n_bytes`` (scored text bytes in that batch), so ``bpb = sum(out.ce_sum) / ln(2) / sum(batch["n_bytes"])``.

Speedrun target (modded-nanogpt): val loss = mean CE over the first 10,485,760 *targets* of the FineWeb val
shard, i.e. predictions of tokens[1 : 10_485_761] read as contiguous chunks; in the current code attention is
document-masked but every chunk start truncates context, and in older records it is a plain contiguous
window with attention crossing documents. That mean includes the ~1 per document <|endoftext|> targets (loss
counted, 0 bytes). We convert with ``target_bpb = loss * n_targets / ln2 / n_text_bytes(targets)``. This
assigns the EOT losses to the text bytes, so it is not identical to our bpb (which excludes doc-end
predictions and uses full doc context up to the chunk limits): it is the bpb-equivalent of "the same total
nats over the same text". See ``speedrun_target``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hnet.config import BOS_ID, IGNORE_INDEX
from hnet.data import shards as S
from hnet.data.loader import pack_windows

LN2 = math.log(2.0)
SPEEDRUN_VAL_TOKENS = 10_485_760


def complete_docs(tokens: np.ndarray, token_limit: int | None = None, eot_id: int = S.EOT_ID,
                  last_complete: bool = False):
    """(start, end) token spans [EOT, ..., last token] of docs that are complete inside tokens[:token_limit].

    ``last_complete``: the shard's last document is known to be complete (data-mix shards never split docs)."""
    n = len(tokens) if token_limit is None else min(len(tokens), token_limit)
    eot = np.flatnonzero(np.asarray(tokens[:n]) == eot_id)
    # a doc starting at eot[i] is complete iff the next EOT is inside the shard (eot[i+1] exists or tokens[n] is EOT)
    ends = list(eot[1:])
    if n < len(tokens) and tokens[n] == eot_id:
        ends.append(n)
    elif n == len(tokens) and last_complete and len(eot):
        ends.append(n)
    return [(int(a), int(b)) for a, b in zip(eot, ends)]


def chunk_doc(tok_bytes_len: np.ndarray, token_seq_len: int, byte_seq_len: int) -> list[tuple[int, int]]:
    """Greedy token-boundary chunks [a, b) of one doc's text tokens satisfying both segment limits."""
    n = len(tok_bytes_len)
    cb = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(tok_bytes_len, out=cb[1:])
    out, a = [], 0
    while a < n:
        # largest b with cb[b] - cb[a] <= byte_seq_len - 1 and b - a <= token_seq_len - 1
        b = int(np.searchsorted(cb, cb[a] + byte_seq_len - 1, side="right") - 1)
        b = min(b, a + token_seq_len - 1, n)
        assert b > a, "a single token is longer than the byte window"
        out.append((a, b))
        a = b
    return out


@dataclass
class _Seg:
    x: np.ndarray
    y: np.ndarray


def _pack(segs: list[_Seg], L: int, windows_per_batch: int, pad_id: int, weights=None, open_windows: int = 64):
    """Deterministic first-fit packing of segments into windows of length L (a segment never spans windows;
    only the last `open_windows` windows are candidates). Returns a list of batches
    (x[n,L], y[n,L], starts[n,L], n_bytes). Window tails are padding segments with IGNORE labels."""
    xs, ys, ss, used, nbytes = [], [], [], [], []
    for i, sg in enumerate(segs):
        m = len(sg.x)
        assert m <= L
        k = next((j for j in range(max(0, len(used) - open_windows), len(used)) if used[j] + m <= L), None)
        if k is None:
            xs.append(np.full(L, pad_id, np.int64))
            ys.append(np.full(L, IGNORE_INDEX, np.int64))
            ss.append(np.zeros(L, bool))
            used.append(0)
            nbytes.append(0)
            k = len(used) - 1
        u = used[k]
        xs[k][u : u + m] = sg.x
        ys[k][u : u + m] = sg.y
        ss[k][u] = True
        used[k] = u + m
        nbytes[k] += int(weights[i]) if weights is not None else 0
    for k in range(len(used)):
        if used[k] < L:
            ss[k][used[k]] = True  # padding tail is its own segment
    batches = []
    for k in range(0, len(used), windows_per_batch):
        sl = slice(k, k + windows_per_batch)
        batches.append((np.stack(xs[sl]), np.stack(ys[sl]), np.stack(ss[sl]), sum(nbytes[sl])))
    return batches


class ValSet:
    """Fixed val set over whole documents of a val token shard. See module docstring for the protocol.

    vs = ValSet("fwedu_10MB", 10_000_000, token_shard=".../finewebedu_val_000000.bin")
    for b in vs.byte_batches(): out = model(**b-inputs...); ce += out.ce_sum; nb += b["n_bytes"]
    bpb = ce / ln2 / nb   (identical nb for token_batches())
    """

    def __init__(self, name: str, n_bytes_target: int | None, token_shard, byte_seq_len: int = 8192,
                 token_seq_len: int = 1792, windows_per_batch: int = 8, start_doc: int = 0,
                 token_limit: int | None = None, tokenizer=None, last_complete: bool = False):
        """``tokenizer``: the shard's tokenizer (hnet.data.tokenizer; default GPT-2). ``last_complete``: the
        shard's last document is complete (true for data-mix shards)."""
        from hnet.data.tokenizer import load_tokenizer

        self.tok = load_tokenizer(tokenizer)
        self.name = name
        self.byte_seq_len, self.token_seq_len = int(byte_seq_len), int(token_seq_len)
        self.windows_per_batch = int(windows_per_batch)
        self.token_shard = str(token_shard)
        toks = S.read_token_shard(token_shard)
        lens = self.tok.table[2]
        docs = complete_docs(toks, token_limit, self.tok.eot_id, last_complete)[start_doc:]
        self.docs, self._doc_tokens = [], []
        total = 0
        for a, b in docs:
            if n_bytes_target is not None and total >= n_bytes_target:
                break
            t = np.asarray(toks[a + 1 : b], dtype=np.int64)
            if len(t) == 0:
                continue
            nb = int(lens[t].sum())
            self.docs.append((a, b))
            self._doc_tokens.append(t)
            total += nb
        self.n_docs = len(self.docs)
        self.n_bytes = total
        self.n_tokens = int(sum(len(t) for t in self._doc_tokens))
        assert n_bytes_target is None or total >= n_bytes_target, (
            f"val shard has only {total} bytes of complete docs < {n_bytes_target}")
        self._built = {}

    # ------------------------------------------------------------------ segments
    def _segments(self):
        from hnet.data.tokenizer import decode_with_table

        table = self.tok.table
        lens = table[2]
        tsegs, bsegs, tw = [], [], []
        for t in self._doc_tokens:
            bl = lens[t]
            doc_bytes = decode_with_table(t, table)  # exact UTF-8 text of the doc (no EOT inside)
            cb = np.zeros(len(t) + 1, dtype=np.int64)
            np.cumsum(bl, out=cb[1:])
            for c, (a, b) in enumerate(chunk_doc(bl, self.token_seq_len, self.byte_seq_len)):
                ct = t[a:b]
                cbytes = doc_bytes[cb[a] : cb[b]].astype(np.int64)
                tp = self.tok.eot_id if c == 0 else int(t[a - 1])
                bp = BOS_ID if c == 0 else int(doc_bytes[cb[a] - 1])
                tsegs.append(_Seg(np.concatenate([[tp], ct]), np.concatenate([ct, [IGNORE_INDEX]])))
                bsegs.append(_Seg(np.concatenate([[bp], cbytes]), np.concatenate([cbytes, [IGNORE_INDEX]])))
                tw.append(int(cb[b] - cb[a]))
        return tsegs, bsegs, tw

    def _build(self):
        if self._built:
            return
        tsegs, bsegs, tw = self._segments()
        self._built["tokens"] = _pack(tsegs, self.token_seq_len, self.windows_per_batch, pad_id=0, weights=tw)
        self._built["bytes"] = _pack(bsegs, self.byte_seq_len, self.windows_per_batch, pad_id=0, weights=tw)
        self.n_segments = len(tsegs)

    def batches(self, kind: str) -> list[dict]:
        self._build()
        out = []
        for x, y, s, nb in self._built[kind]:
            b = pack_windows(x, y, s, extra={"n_bytes": int(nb)})
            b["score_mask"] = b["labels"] != IGNORE_INDEX
            out.append(b)
        return out

    def byte_batches(self) -> list[dict]:
        return self.batches("bytes")

    def token_batches(self) -> list[dict]:
        return self.batches("tokens")

    def describe(self) -> dict:
        self._build()
        return dict(name=self.name, n_docs=self.n_docs, n_bytes=self.n_bytes, n_tokens=self.n_tokens,
                    n_segments=self.n_segments, bytes_per_token=self.n_bytes / max(1, self.n_tokens),
                    n_byte_batches=len(self._built["bytes"]), n_token_batches=len(self._built["tokens"]),
                    byte_seq_len=self.byte_seq_len, token_seq_len=self.token_seq_len,
                    first_doc_token_offset=self.docs[0][0] if self.docs else None)

    @staticmethod
    def bpb(ce_sum_nats: float, n_bytes: int) -> float:
        return float(ce_sum_nats) / LN2 / max(1, int(n_bytes))


def speedrun_target(token_shard, n_targets: int = SPEEDRUN_VAL_TOKENS, losses=(3.28, 2.92)) -> dict:
    """bpb equivalent of modded-nanogpt val losses (nats/token over the first n_targets targets).

    modded-nanogpt scores targets tokens[1 : n_targets + 1] (inputs tokens[0 : n_targets]); n_bytes is the
    UTF-8 length of those target tokens (EOT -> 0 bytes)."""
    toks = np.asarray(S.read_token_shard(token_shard)[1 : n_targets + 1], dtype=np.int64)
    assert len(toks) == n_targets
    n_bytes = int(S.token_byte_lengths(toks).sum())
    n_eot = int((toks == S.EOT_ID).sum())
    res = dict(n_targets=n_targets, n_bytes=n_bytes, n_eot_targets=n_eot,
               bytes_per_target=n_bytes / n_targets, bytes_per_text_token=n_bytes / (n_targets - n_eot))
    res["target_bpb"] = {str(l): l * n_targets / LN2 / n_bytes for l in losses}
    return res


def speedrun_valset(token_shard, **kw) -> ValSet:
    """Our-protocol ValSet over the whole docs inside the speedrun's first 10,485,761 val tokens."""
    return ValSet("speedrun_fineweb_val", None, token_shard, token_limit=SPEEDRUN_VAL_TOKENS + 1, **kw)


def standard_valset(dataset: str = "finewebedu10B", n_bytes_target: int | None = 10_000_000, root=None,
                    subset: str | None = None, **kw) -> ValSet:
    """The dataset's val set. Data mixes (hnet.data.mix): the combined val set (``subset=None`` or "mix"), or
    one of the per-source / per-language / per-category sets (``subset="code.python"``, ``"heldout"``, ...);
    a target larger than the set uses the whole set."""
    from hnet.data import mix
    from hnet.data.download import dataset_dir, shard_name

    if mix.is_mix(dataset, root):
        from hnet.data.tokenizer import dataset_tokenizer

        man = mix.load_manifest(dataset, root)
        name = subset or "mix"
        if name not in man["val_sets"]:
            raise KeyError(f"{dataset} has no val set {name!r}; available: {sorted(man['val_sets'])}")
        v = man["val_sets"][name]
        target = None if n_bytes_target is None or n_bytes_target >= v["bytes"] else n_bytes_target
        path = Path(dataset_dir(dataset, root)) / f"{v['stem']}.bin"
        return ValSet(f"{dataset}:{name}", target, path, tokenizer=dataset_tokenizer(dataset, root),
                      last_complete=True, **kw)
    assert subset in (None, "standard"), f"{dataset} has no val subsets"
    path = Path(dataset_dir(dataset, root)) / shard_name(dataset, "val", 0)
    return ValSet(f"{dataset}_val_{n_bytes_target // 1_000_000}MB", n_bytes_target, path, **kw)


__all__ = ["ValSet", "speedrun_target", "speedrun_valset", "standard_valset", "chunk_doc", "complete_docs",
           "SPEEDRUN_VAL_TOKENS", "LN2"]
