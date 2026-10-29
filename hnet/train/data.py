"""Data plumbing for the trainer: loader adapters, a built-in loader for tests, prefetching.

Trainer-side loader contract (``StepLoader``)
---------------------------------------------
``loader.get(step, micro) -> batch`` returns the packed micro-batch (SPEC "Packed micro-batch") of this
rank for optimizer step ``step`` and micro-step ``micro`` (0 <= micro < grad_accum), on CPU, plus

* ``n_bytes``  (int): UTF-8 text bytes covered by the micro-batch (BOS/EOS/<|endoftext|> excluded);
* ``n_tokens`` (int, token data only): number of positions.

Deterministic given (seed, step, micro, rank, world_size): resuming only needs the step. Within one step,
rank r / micro m reads global windows ``[t*W + (m*world + r)*mb, ... + mb)`` with ``W = accum*world*mb``,
so a 2-rank run with accumulation k sees exactly the windows of a 1-rank run with accumulation 2k.

``hnet.data`` (WS-C) loaders are adapted in :func:`build_train_loader`; the built-in ``synthetic`` and
``text`` sources exist so the trainer can be tested on CPU without downloaded shards.
"""

from __future__ import annotations

import os
import queue
import threading
from typing import Iterator, List, Optional

import numpy as np
import torch

from hnet.config import BOS_ID, EOS_ID, IGNORE_INDEX

GPT2_EOT = 50256


# ----------------------------------------------------------------------------- packing helpers
def pack_windows(windows: List[np.ndarray], seg_start_fn, next_labels: Optional[List[int]] = None) -> dict:
    """Concatenate windows into one packed micro-batch. ``seg_start_fn(w) -> bool[L]`` marks positions
    that begin a new segment (position 0 of every window always does). ``next_labels[i]`` (or IGNORE_INDEX)
    is the label of window i's last position (``cross_window_label``, same rule as hnet.data.loader)."""
    L = len(windows[0])
    ids = np.concatenate(windows).astype(np.int64)
    starts = np.concatenate([seg_start_fn(w) for w in windows])
    starts[::L] = True
    T = ids.shape[0]
    cu = np.flatnonzero(starts).astype(np.int32)
    cu = np.append(cu, T).astype(np.int32)
    labels = np.full(T, IGNORE_INDEX, dtype=np.int64)
    labels[:-1] = ids[1:]
    labels[cu[1:] - 1] = IGNORE_INDEX  # last position of every segment has no target
    if next_labels is not None:
        labels[L - 1::L] = next_labels
    lens = np.diff(cu)
    return {
        "input_ids": torch.from_numpy(ids),
        "labels": torch.from_numpy(labels),
        "cu_seqlens": torch.from_numpy(cu),
        "max_seqlen": int(lens.max()),
        "n_windows": len(windows),
    }


def byte_seg_starts(w: np.ndarray) -> np.ndarray:
    s = np.zeros(len(w), dtype=bool)
    s[1:] = w[:-1] == EOS_ID  # a new segment starts right after EOS
    return s


def token_seg_starts_fn(eot: int):
    def fn(w: np.ndarray) -> np.ndarray:
        return w == eot  # a new segment starts at <|endoftext|> (which precedes every document)

    return fn


# ----------------------------------------------------------------------------- built-in sources
_SYLL = ["ka", "to", "ri", "ne", "mo", "sa", "lu", "pe", "di", "go", "ba", "vi", "zu", "he", "fo", "ja"]


def synthetic_words(n_words: int = 200, seed: int = 1234) -> List[str]:
    rng = np.random.default_rng(seed)
    words = set()
    while len(words) < n_words:
        k = int(rng.integers(1, 4))
        words.add("".join(_SYLL[i] for i in rng.integers(0, len(_SYLL), k)))
    return sorted(words)


def synthetic_docs(n_docs: int, seed: int, n_words: int = 200, mean_len: int = 60) -> List[List[int]]:
    """Documents as word-id lists from a sparse first-order Markov chain (learnable structure)."""
    rng = np.random.default_rng(seed)
    trng = np.random.default_rng(999)  # transition table independent of the split seed
    succ = trng.integers(0, n_words, size=(n_words, 4))
    docs = []
    for _ in range(n_docs):
        n = int(rng.integers(mean_len // 2, mean_len * 3 // 2))
        w = int(rng.integers(0, n_words))
        doc = [w]
        for _ in range(n - 1):
            w = int(succ[w, rng.integers(0, 4)]) if rng.random() < 0.9 else int(rng.integers(0, n_words))
            doc.append(w)
        docs.append(doc)
    return docs


class Corpus:
    """A document list in both byte and token form (token form only for synthetic data)."""

    def __init__(self, byte_docs: List[bytes], token_docs: Optional[List[List[int]]] = None,
                 token_nbytes: Optional[np.ndarray] = None, eot: int = GPT2_EOT):
        self.byte_docs = byte_docs
        self.token_docs = token_docs
        self.token_nbytes = token_nbytes  # bytes per token id (eot -> 0)
        self.eot = eot

    @classmethod
    def synthetic(cls, n_docs: int, seed: int, eot: int = 255) -> "Corpus":
        words = synthetic_words()
        docs = synthetic_docs(n_docs, seed, n_words=len(words))
        byte_docs = [" ".join(words[i] for i in d).encode() for d in docs]
        # token id i = word i followed by a space ("word " has len+1 bytes; last word of a doc has no space)
        tok_docs = [list(d) for d in docs]
        nb = np.zeros(max(eot + 1, len(words)), dtype=np.int64)
        nb[: len(words)] = [len(w) + 1 for w in words]
        nb[eot] = 0
        return cls(byte_docs, tok_docs, nb, eot)

    @classmethod
    def from_text(cls, paths, sep: str = "\n\n", max_docs: Optional[int] = None) -> "Corpus":
        docs = []
        for p in [paths] if isinstance(paths, str) else paths:
            with open(p, encoding="utf-8", errors="replace") as f:
                docs += [d.strip().encode() for d in f.read().split(sep) if d.strip()]
        return cls(docs[:max_docs] if max_docs else docs)

    def byte_stream(self) -> np.ndarray:
        parts = []
        for d in self.byte_docs:
            parts += [np.array([BOS_ID], np.uint8), np.frombuffer(d, np.uint8), np.array([EOS_ID], np.uint8)]
        return np.concatenate(parts)

    def token_stream(self) -> np.ndarray:
        assert self.token_docs is not None, "token data needs a tokenized corpus (synthetic source or hnet.data)"
        parts = []
        for d in self.token_docs:
            parts += [np.array([self.eot], np.int64), np.asarray(d, np.int64)]
        return np.concatenate(parts)


class BuiltinLoader:
    """Windows cut from a concatenated stream (wraps around). See module docstring for the index scheme."""

    def __init__(self, stream: np.ndarray, kind: str, seq_len: int, micro_batch: int, grad_accum: int,
                 rank: int = 0, world_size: int = 1, token_nbytes: Optional[np.ndarray] = None,
                 eot: int = GPT2_EOT, offset: int = 0, cross_window_label: bool = True):
        assert len(stream) >= seq_len, "corpus shorter than one window"
        self.stream, self.kind, self.L = stream, kind, seq_len
        self.mb, self.accum, self.rank, self.world = micro_batch, grad_accum, rank, world_size
        self.n_win = len(stream) // seq_len
        self.token_nbytes, self.eot, self.offset = token_nbytes, eot, offset
        self.seg_fn = byte_seg_starts if kind == "bytes" else token_seg_starts_fn(eot)
        self.cross = cross_window_label

    def window(self, g: int) -> np.ndarray:
        i = (g + self.offset) % self.n_win
        return self.stream[i * self.L:(i + 1) * self.L]

    def get(self, step: int, micro: int) -> dict:
        W = self.accum * self.world * self.mb
        base = step * W + (micro * self.world + self.rank) * self.mb
        wins = [self.window(base + j) for j in range(self.mb)]
        nxt = None
        if self.cross:  # label of the last position = next stream element if it is in the same document
            nxt = []
            for j, w in enumerate(wins):
                i = (base + j + self.offset) % self.n_win
                n = int(self.stream[((i + 1) * self.L) % len(self.stream)])
                cur = int(w[-1])
                same = (n != self.eot) if self.kind == "tokens" else (cur != EOS_ID and n != BOS_ID)
                nxt.append(n if same else IGNORE_INDEX)
        b = pack_windows(wins, self.seg_fn, nxt)
        if self.kind == "bytes":
            ids = b["input_ids"]
            b["n_bytes"] = int((ids < BOS_ID).sum())
        else:
            b["n_bytes"] = int(self.token_nbytes[b["input_ids"].numpy()].sum())
            b["n_tokens"] = int(b["input_ids"].numel())
        return b

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, sd: dict):
        pass


def builtin_val_batches(corpus: Corpus, kind: str, batch_positions: int, max_docs: Optional[int] = None):
    """Whole documents packed into batches of <= batch_positions positions (one doc may exceed it)."""
    docs = corpus.byte_docs if kind == "bytes" else corpus.token_docs
    n = len(docs) if max_docs is None else min(max_docs, len(docs))
    cur: List[np.ndarray] = []
    cur_len = 0
    out = []

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        lens = [len(c) for c in cur]
        ids = np.concatenate(cur).astype(np.int64)
        cu = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        labels = np.full(len(ids), IGNORE_INDEX, np.int64)
        labels[:-1] = ids[1:]
        labels[cu[1:] - 1] = IGNORE_INDEX
        if kind == "bytes":
            nb = int((labels[(labels >= 0)] < BOS_ID).sum())
        else:
            tgt = labels[(labels >= 0) & (labels != corpus.eot)]
            nb = int(corpus.token_nbytes[tgt].sum())
        out.append({"input_ids": torch.from_numpy(ids), "labels": torch.from_numpy(labels),
                    "cu_seqlens": torch.from_numpy(cu), "max_seqlen": int(max(lens)), "n_bytes": nb})
        cur, cur_len = [], 0

    for i in range(n):
        if kind == "bytes":
            d = np.concatenate([[BOS_ID], np.frombuffer(docs[i], np.uint8), [EOS_ID]]).astype(np.int64)
        else:
            d = np.asarray([corpus.eot] + list(docs[i]), np.int64)
        if cur_len + len(d) > batch_positions:
            flush()
        cur.append(d)
        cur_len += len(d)
    flush()
    return out


# ----------------------------------------------------------------------------- factory
def data_kind(data_cfg: dict, model_cfg) -> str:
    k = data_cfg.get("kind")
    if k:
        return k
    vocab = getattr(model_cfg, "vocab_size", 256)
    return "bytes" if vocab <= 256 else "tokens"


def _builtin_corpus(data_cfg: dict, split: str, seed: int) -> Corpus:
    src = data_cfg["source"]
    if src == "synthetic":
        n = int(data_cfg.get("synthetic_docs", 400 if split == "train" else 40))
        return Corpus.synthetic(n, seed=(seed + 1) if split == "train" else 10_000 + seed,
                                eot=int(data_cfg.get("eot", 255)))
    if src == "text":
        paths = data_cfg.get(f"{split}_paths") or data_cfg.get("paths")
        return Corpus.from_text(paths, sep=data_cfg.get("doc_sep", "\n\n"))
    raise ValueError(src)


def _hnet_data():
    import hnet.data as hd

    return hd


def build_train_loader(data_cfg: dict, kind: str, grad_accum: int, rank: int, world_size: int, seed: int):
    src = data_cfg.get("source", "auto")
    if src in ("synthetic", "text"):
        corpus = _builtin_corpus(data_cfg, "train", seed)
        stream = corpus.byte_stream() if kind == "bytes" else corpus.token_stream()
        return BuiltinLoader(stream, kind, int(data_cfg["seq_len"]), int(data_cfg["micro_batch"]), grad_accum,
                             rank, world_size, corpus.token_nbytes, corpus.eot,
                             cross_window_label=bool(data_cfg.get("cross_window_label", True)))
    hd = _hnet_data()
    if hd is None:
        raise RuntimeError("data.source needs hnet.data (WS-C), which is not available; use synthetic|text")
    return HNetDataLoader(data_cfg, kind, grad_accum, rank, world_size, seed)


def _shard_paths(data_cfg: dict, kind: str, split: str = "train"):
    explicit = data_cfg.get("byte_paths" if kind == "bytes" else "token_paths") or data_cfg.get("paths")
    if explicit:
        return [str(p) for p in ([explicit] if isinstance(explicit, str) else explicit)]
    from hnet.data import download as D

    ds, root, n = data_cfg.get("dataset", "finewebedu10B"), data_cfg.get("root"), data_cfg.get("n_train_shards")
    paths = D.byte_shard_paths(ds, split, root, n) if kind == "bytes" else D.token_paths(ds, split, root, n)
    if not paths:
        raise FileNotFoundError(f"no {kind} {split} shards for {ds} under {D.dataset_dir(ds, root)}; "
                                f"run scripts/data/prepare.py")
    return paths


class HNetDataLoader:
    """``hnet.data.WindowLoader`` (WS-C) behind the ``get(step, micro)`` contract.

    WindowLoader's order: a micro-batch at ``consumed`` gives rank r the logical windows ``consumed + r*mb``;
    ``consumed`` advances by ``world*mb`` per micro-step, so step t / micro m is ``consumed=(t*accum+m)*world*mb``.
    Adds ``n_bytes`` (text bytes: real bytes for byte data, GPT-2 token byte lengths for token data).
    """

    def __init__(self, data_cfg, kind, grad_accum, rank, world_size, seed):
        from hnet.data import WindowLoader

        self.kind, self.accum, self.world = kind, grad_accum, world_size
        self.mb = int(data_cfg["micro_batch"])
        tok = dataset_tokenizer(data_cfg) if kind == "tokens" else None
        self.inner = WindowLoader(_shard_paths(data_cfg, kind), kind, int(data_cfg["seq_len"]), n_windows=self.mb,
                                  rank=rank, world_size=world_size, seed=seed,
                                  shuffle_buffer=int(data_cfg.get("shuffle_buffer", 0) or 0),
                                  cross_window_label=bool(data_cfg.get("cross_window_label", True)),
                                  pin_memory=False, prefetch=0,
                                  eot_id=default_eot(data_cfg) if kind == "tokens" else GPT2_EOT)
        if kind == "tokens":
            self._tbl = tok.byte_lengths

    def get(self, step: int, micro: int) -> dict:
        b = self.inner.build((step * self.accum + micro) * self.world * self.mb)
        ids = b["input_ids"].numpy()
        if self.kind == "bytes":
            b["n_bytes"] = int((ids < BOS_ID).sum())
        else:
            b["n_bytes"] = int(self._tbl(ids).sum())
            b["n_tokens"] = int(ids.size)
        return b

    def state_dict(self):
        return {"stream": self.inner.stream.fingerprint()}  # position is a function of the step

    def load_state_dict(self, sd):
        fp = sd.get("stream")
        if fp is not None and fp != self.inner.stream.fingerprint():
            raise AssertionError("training shard list differs from the checkpoint's")


def default_bytes_per_token(data_cfg: dict, model_cfg) -> float:
    """Text bytes per text token of the dataset's tokenizer (GPT-2, or a data mix's BPE): config, else WS-C's
    measured value (bpt_stats.json / the mix manifest), else model n_gpt, else 4.6."""
    if data_cfg.get("bytes_per_token"):
        return float(data_cfg["bytes_per_token"])
    if data_cfg.get("source") not in ("synthetic", "text"):
        return float(_hnet_data().bytes_per_token(data_cfg.get("dataset", "finewebedu10B"),
                                                  root=data_cfg.get("root")))
    return float(getattr(model_cfg, "lr_multiplier_n_gpt", None) or 4.6)


def dataset_tokenizer(data_cfg: dict):
    """hnet.data tokenizer (EOT id, per-id byte lengths) of the configured dataset's token shards."""
    from hnet.data.tokenizer import dataset_tokenizer as dt

    return dt(data_cfg.get("dataset", "finewebedu10B"), data_cfg.get("root"))


def default_eot(data_cfg: dict) -> int:
    """Document-start token id: ``data.eot`` if set, else the dataset tokenizer's (GPT-2 50256; data mixes 0)."""
    if data_cfg.get("eot") is not None:
        return int(data_cfg["eot"])
    if data_cfg.get("source", "hnet") in ("synthetic", "text"):
        return GPT2_EOT
    return int(dataset_tokenizer(data_cfg).eot_id)


def build_val_batches(data_cfg: dict, eval_cfg: dict, kind: str, seed: int):
    """List of whole-document val batches carrying ``n_bytes``; None if unavailable.

    For hnet.data the ValSet is built with the same (val_byte_seq_len, val_token_seq_len) for both model kinds
    so byte and token models score exactly the same text (see hnet/data/valsets.py)."""
    src = data_cfg.get("source", "auto")
    if src in ("synthetic", "text"):
        if src == "text" and not (data_cfg.get("val_paths")):
            return None
        corpus = _builtin_corpus(data_cfg, "val", seed)
        return builtin_val_batches(corpus, kind, int(eval_cfg.get("batch_bytes", 65536)), eval_cfg.get("max_docs"))
    hd = _hnet_data()
    if hd is None:
        return None
    kw = dict(byte_seq_len=int(eval_cfg.get("val_byte_seq_len", 8192)),
              token_seq_len=int(eval_cfg.get("val_token_seq_len", 1792)),
              windows_per_batch=int(eval_cfg.get("windows_per_batch", 8)))
    which = eval_cfg.get("val", "standard")
    ds = data_cfg.get("dataset", "finewebedu10B")
    vb = eval_cfg.get("val_bytes")
    if data_cfg.get("val_token_path"):  # explicit val token shard (e.g. hnet.data.synth fixtures)
        vs = hd.ValSet("custom_val", int(vb) if vb else None, data_cfg["val_token_path"],
                       tokenizer=dataset_tokenizer(data_cfg), **kw)
    elif which not in ("standard", "speedrun"):  # a data mix's named val set, e.g. eval.val: code.python
        vs = hd.standard_valset(ds, n_bytes_target=int(vb) if vb else None, root=data_cfg.get("root"),
                                subset=which, **kw)
    elif which == "speedrun":
        from hnet.data import download as D

        shard = D.dataset_dir(ds, data_cfg.get("root")) / D.shard_name(ds, "val", 0)
        vs = hd.speedrun_valset(shard, **kw)
    else:
        vs = hd.standard_valset(ds, n_bytes_target=int(vb or 10_000_000),
                                root=data_cfg.get("root"), **kw)
    return vs.batches(kind)


# ----------------------------------------------------------------------------- prefetch
class Prefetcher:
    """Produces ``loader.get(step, micro)`` in order on background threads; pins memory on CUDA."""

    def __init__(self, loader, start_step: int, grad_accum: int, pin: bool, depth: int = 4, threads: int = 1):
        self.loader, self.accum, self.pin = loader, grad_accum, pin
        self.step, self.micro = start_step, 0
        self.threads = threads
        self._q: "queue.Queue" = queue.Queue(maxsize=max(1, depth))
        self._stop = threading.Event()
        self._thread = None
        if threads > 0:
            self._thread = threading.Thread(target=self._run, args=(start_step,), daemon=True)
            self._thread.start()

    def _load(self, step, micro):
        b = self.loader.get(step, micro)
        if self.pin:
            for k in ("input_ids", "labels", "cu_seqlens"):
                b[k] = b[k].pin_memory()
        return b

    def _run(self, step):
        micro = 0
        try:
            while not self._stop.is_set():
                b = self._load(step, micro)
                while not self._stop.is_set():
                    try:
                        self._q.put((step, micro, b), timeout=0.1)
                        break
                    except queue.Full:
                        continue
                micro += 1
                if micro == self.accum:
                    step, micro = step + 1, 0
        except BaseException as e:  # surface loader errors in the main thread
            self._q.put(("error", e, None))

    def next(self) -> dict:
        if self._thread is None:
            b = self._load(self.step, self.micro)
        else:
            item = self._q.get()
            if item[0] == "error":
                raise item[1]
            s, m, b = item
            assert (s, m) == (self.step, self.micro), f"prefetch out of order: {(s, m)} vs {(self.step, self.micro)}"
        self.micro += 1
        if self.micro == self.accum:
            self.step, self.micro = self.step + 1, 0
        return b

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
