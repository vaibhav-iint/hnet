"""Deterministic, resumable fixed-window loaders for token and byte streams (SPEC "Packed micro-batch").

Stream: the concatenation of the given shards in the given order (token shards: uint16 GPT-2 ids, each doc
starts with <|endoftext|>; byte shards: uint8, docs are [BOS] + bytes + [EOS]).

Windows: window w covers stream positions [w*L, w*L + L) (inputs) and reads one extra element at w*L + L, so
the last input position of a window still gets its true next-element label (``cross_window_label=True``).
The number of windows per epoch is (stream_len - 1) // L.

Sharding / order: the loader keeps one integer of state, ``consumed`` = number of *global* windows handed out
so far (over all ranks). A micro-batch of ``n_windows`` per rank takes the logical windows
``consumed + rank * n_windows + [0, n_windows)`` and then advances ``consumed`` by ``world_size * n_windows``.
So rank batches at a step are disjoint and the order is independent of world size (elastic resume works).
Logical window g maps to physical window ``perm(g)``: identity, or (``shuffle_buffer = B > 0``) a seeded
permutation inside consecutive blocks of B windows (block seed = hash(seed, epoch, block)). After the last
window the stream wraps to a new epoch.

Segments (cu_seqlens): a segment starts at every window start and
  * tokens: at every <|endoftext|> (the first token of each doc);
  * bytes : at every BOS and right after every EOS.
Labels: ``labels[t] = next element`` except IGNORE_INDEX where the next element starts a new document
(tokens: next == <|endoftext|>; bytes: current == EOS or next == BOS). So labels never cross a document, and
predicting <|endoftext|> (the token model's doc start) is never trained; the byte model does learn to predict
EOS (it is excluded from bpb). With ``cross_window_label=False`` the last position of each window is also
IGNORE_INDEX (strict "within the packed segment" reading of the SPEC).
"""

from __future__ import annotations

import functools
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from hnet.config import BOS_ID, EOS_ID, IGNORE_INDEX
from hnet.data import shards as S


class ConcatStream:
    """Read-only view of several memory-mapped shards as one 1-D stream."""

    def __init__(self, paths, kind: str):
        assert kind in ("tokens", "bytes"), kind
        self.kind = kind
        self.paths = [str(p) for p in paths]
        assert self.paths, "no shards given"
        reader = S.read_token_shard if kind == "tokens" else S.read_byte_shard
        self.arrays = [reader(p) for p in self.paths]
        lens = np.array([len(a) for a in self.arrays], dtype=np.int64)
        self.offsets = np.zeros(len(lens) + 1, dtype=np.int64)
        np.cumsum(lens, out=self.offsets[1:])
        self.dtype = np.uint16 if kind == "tokens" else np.uint8

    def __len__(self):
        return int(self.offsets[-1])

    def read(self, start: int, length: int, out: np.ndarray | None = None) -> np.ndarray:
        out = np.empty(length, dtype=self.dtype) if out is None else out
        assert 0 <= start and start + length <= len(self)
        k = int(np.searchsorted(self.offsets, start, side="right") - 1)
        pos, filled = start, 0
        while filled < length:
            a = self.arrays[k]
            lo = pos - int(self.offsets[k])
            n = min(length - filled, len(a) - lo)
            out[filled : filled + n] = a[lo : lo + n]
            filled += n
            pos += n
            k += 1
        return out

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for p, n in zip(self.paths, np.diff(self.offsets)):
            h.update(f"{Path(p).name}:{int(n)};".encode())
        return h.hexdigest()[:16]


@functools.lru_cache(maxsize=8)
def _block_perm(seed: int, epoch: int, block: int, n: int) -> np.ndarray:
    ss = np.random.SeedSequence([seed & 0xFFFFFFFF, epoch, block, 0x484E4554])
    return np.random.default_rng(ss).permutation(n)


def segment_starts(x: np.ndarray, kind: str, eot_id: int = S.EOT_ID) -> np.ndarray:
    """Bool [n, L]: positions that start a segment (window starts + document starts)."""
    s = np.zeros(x.shape, dtype=bool)
    s[:, 0] = True
    if kind == "tokens":
        s |= x == eot_id
    else:
        s |= x == BOS_ID
        s[:, 1:] |= x[:, :-1] == EOS_ID
    return s


def make_labels(x: np.ndarray, nxt: np.ndarray, kind: str, cross_window_label: bool = True,
                eot_id: int = S.EOT_ID) -> np.ndarray:
    lab = nxt.astype(np.int64)
    if kind == "tokens":
        lab[nxt == eot_id] = IGNORE_INDEX
    else:
        lab[(x == EOS_ID) | (nxt == BOS_ID)] = IGNORE_INDEX
    if not cross_window_label:
        lab[:, -1] = IGNORE_INDEX
    return lab


def pack_windows(x: np.ndarray, labels: np.ndarray, starts: np.ndarray, pin_memory: bool = False,
                 extra: dict | None = None) -> dict:
    """[n, L] arrays -> the SPEC packed micro-batch dict (CPU tensors, optionally pinned)."""
    n, L = x.shape
    T = n * L
    st = np.flatnonzero(starts.reshape(-1))
    cu = np.empty(len(st) + 1, dtype=np.int32)
    cu[:-1] = st
    cu[-1] = T
    max_seqlen = int(np.diff(cu).max()) if len(cu) > 1 else 0
    if pin_memory:
        inp = torch.empty(T, dtype=torch.long, pin_memory=True)
        lab = torch.empty(T, dtype=torch.long, pin_memory=True)
        cut = torch.empty(len(cu), dtype=torch.int32, pin_memory=True)
        inp.numpy()[:] = x.reshape(-1)
        lab.numpy()[:] = labels.reshape(-1)
        cut.numpy()[:] = cu
    else:
        inp = torch.from_numpy(x.reshape(-1).astype(np.int64))
        lab = torch.from_numpy(np.ascontiguousarray(labels.reshape(-1), dtype=np.int64))
        cut = torch.from_numpy(cu)
    out = {"input_ids": inp, "labels": lab, "cu_seqlens": cut, "max_seqlen": max_seqlen, "n_windows": n}
    if extra:
        out.update(extra)
    return out


def batch_to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v) for k, v in batch.items()}


class WindowLoader:
    """Fixed-window packed micro-batch loader. See module docstring for the exact semantics.

    >>> ld = WindowLoader(paths, "bytes", seq_len=8192, n_windows=4, rank=r, world_size=W, seed=0)
    >>> batch = ld.next_batch()          # dict(input_ids, labels, cu_seqlens, max_seqlen, n_windows)
    >>> sd = ld.state_dict(); ld2.load_state_dict(sd)   # exact resume
    """

    def __init__(self, paths, kind: str, seq_len: int, n_windows: int = 1, rank: int = 0, world_size: int = 1,
                 seed: int = 0, shuffle_buffer: int = 0, cross_window_label: bool = True,
                 pin_memory: bool | None = None, prefetch: int = 2, max_epochs: int | None = None,
                 eot_id: int = S.EOT_ID):
        """``eot_id``: the token shards' document-start id (GPT-2 <|endoftext|> by default; a data mix's
        tokenizer has its own, see hnet.data.tokenizer.dataset_tokenizer)."""
        assert 0 <= rank < world_size
        self.eot_id = int(eot_id)
        self.stream = ConcatStream(paths, kind)
        self.kind = kind
        self.seq_len = int(seq_len)
        self.n_windows = int(n_windows)
        self.rank, self.world_size = int(rank), int(world_size)
        self.seed = int(seed)
        self.shuffle_buffer = int(shuffle_buffer)
        self.cross_window_label = cross_window_label
        self.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.max_epochs = max_epochs
        self.windows_per_epoch = (len(self.stream) - 1) // self.seq_len
        assert self.windows_per_epoch > 0, "stream shorter than one window"
        self.consumed = 0
        self.prefetch = int(prefetch)
        self._pool = ThreadPoolExecutor(1) if self.prefetch > 0 else None
        self._futures = {}  # consumed value -> future of the batch built at that position

    # ------------------------------------------------------------ order
    def physical_window(self, g: int) -> int:
        epoch, w = divmod(int(g), self.windows_per_epoch)
        if self.max_epochs is not None and epoch >= self.max_epochs:
            raise StopIteration(f"data exhausted after {self.max_epochs} epoch(s)")
        B = self.shuffle_buffer
        if B <= 1:
            return w
        block, j = divmod(w, B)
        n = min(B, self.windows_per_epoch - block * B)
        return block * B + int(_block_perm(self.seed, epoch, block, n)[j])

    def window_ids(self, consumed: int, n_windows: int | None = None, rank: int | None = None) -> list[int]:
        n = self.n_windows if n_windows is None else n_windows
        r = self.rank if rank is None else rank
        base = consumed + r * n
        return [self.physical_window(g) for g in range(base, base + n)]

    # ------------------------------------------------------------ build
    def read_windows(self, ids) -> np.ndarray:
        L = self.seq_len
        buf = np.empty((len(ids), L + 1), dtype=self.stream.dtype)
        for i, w in enumerate(ids):
            self.stream.read(w * L, L + 1, out=buf[i])
        return buf

    def build(self, consumed: int, n_windows: int | None = None) -> dict:
        n = self.n_windows if n_windows is None else n_windows
        buf = self.read_windows(self.window_ids(consumed, n)).astype(np.int64)
        x, nxt = buf[:, :-1], buf[:, 1:]
        labels = make_labels(x, nxt, self.kind, self.cross_window_label, self.eot_id)
        starts = segment_starts(x, self.kind, self.eot_id)
        return pack_windows(x, labels, starts, self.pin_memory)

    # ------------------------------------------------------------ iteration
    def _schedule(self, consumed: int):
        if self._pool is not None and consumed not in self._futures:
            self._futures[consumed] = self._pool.submit(self.build, consumed)

    def next_batch(self, n_windows: int | None = None) -> dict:
        """Next micro-batch for this rank. Passing a different n_windows changes the batch size from here on."""
        if n_windows is not None and n_windows != self.n_windows:
            self._drop_prefetch()
            self.n_windows = int(n_windows)
        c = self.consumed
        step_windows = self.world_size * self.n_windows
        if self._pool is not None:
            self._schedule(c)
            fut = self._futures.pop(c)
            for k in range(1, self.prefetch + 1):
                self._schedule(c + k * step_windows)
            batch = fut.result()
        else:
            batch = self.build(c)
        self.consumed = c + step_windows
        return batch

    def __iter__(self):
        while True:
            try:
                yield self.next_batch()
            except StopIteration:
                return

    def _drop_prefetch(self):
        for f in self._futures.values():
            f.cancel()
        self._futures.clear()

    # ------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {
            "consumed": int(self.consumed), "kind": self.kind, "seq_len": self.seq_len, "seed": self.seed,
            "shuffle_buffer": self.shuffle_buffer, "n_windows": self.n_windows, "world_size": self.world_size,
            "windows_per_epoch": self.windows_per_epoch, "stream": self.stream.fingerprint(),
            "epoch": self.consumed // self.windows_per_epoch,
        }

    def load_state_dict(self, sd: dict, strict: bool = True) -> None:
        for k in ("kind", "seq_len", "seed", "shuffle_buffer"):
            if strict:
                assert sd[k] == getattr(self, k), f"loader {k} mismatch: ckpt {sd[k]} vs {getattr(self, k)}"
        if strict and sd.get("stream") not in (None, self.stream.fingerprint()):
            raise AssertionError("loader shard list differs from the checkpoint's")
        self._drop_prefetch()
        self.consumed = int(sd["consumed"])

    def set_step(self, step: int, grad_accum: int = 1) -> None:
        """Position the loader at the start of optimizer step `step` (constant batch size runs)."""
        self._drop_prefetch()
        self.consumed = int(step) * int(grad_accum) * self.world_size * self.n_windows

    @property
    def epoch(self) -> float:
        return self.consumed / self.windows_per_epoch

    def close(self):
        self._drop_prefetch()
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


# ---------------------------------------------------------------- bytes-per-step matching
_STATS_PATH = Path(__file__).with_name("bpt_stats.json")


def load_bpt_stats(dataset: str | None = None, root=None) -> dict:
    """Bytes/token stats: a data mix's manifest.json (its own BPE, measured on its train shards), else the
    committed GPT-2 stats (written by scripts/data/measure_bpt.py)."""
    if dataset is not None:
        from hnet.data import mix

        if mix.is_mix(dataset, root):
            return mix.load_manifest(dataset, root)["train"]
    d = json.loads(_STATS_PATH.read_text()) if _STATS_PATH.exists() else {}
    return d if dataset is None else d[dataset]


def bytes_per_token(dataset: str = "finewebedu10B", stream: bool = False, root=None) -> float:
    """Measured UTF-8 bytes per text token (EOT excluded) of the dataset's tokenizer (GPT-2, or a data mix's
    BPE). `stream=True` gives the ratio of byte-stream positions to token-stream positions,
    ((bytes + BOS + EOS) / (tokens + EOT)), which is what equalizes the documents seen per step between a byte
    and a token loader."""
    st = load_bpt_stats(dataset, root)
    return float(st["stream_bytes_per_token"] if stream else st["bytes_per_token"])


def batch_bytes_per_step(kind: str, seq_len: int, n_windows: int, world_size: int = 1, grad_accum: int = 1,
                         bpt: float | None = None, dataset: str = "finewebedu10B") -> float:
    """Byte-stream positions (≈ UTF-8 bytes incl. one BOS+EOS per doc) consumed per optimizer step.
    For token loaders this is the expected value using the measured stream bytes/token."""
    positions = seq_len * n_windows * world_size * grad_accum
    if kind == "bytes":
        return float(positions)
    bpt = bpt if bpt is not None else bytes_per_token(dataset, stream=True)
    return positions * bpt


def matched_token_seq_len(byte_seq_len: int = 8192, bpt: float | None = None,
                          dataset: str = "finewebedu10B", multiple: int = 128) -> int:
    """Token window covering the same text as a byte window, rounded to `multiple` (8192 B -> 1792 tok)."""
    bpt = bpt if bpt is not None else bytes_per_token(dataset, stream=True)
    return max(multiple, int(round(byte_seq_len / bpt / multiple)) * multiple)


def windows_per_s(loader: WindowLoader, n_batches: int = 20) -> float:
    import time

    loader.next_batch()
    t0 = time.perf_counter()
    for _ in range(n_batches):
        loader.next_batch()
    return n_batches * loader.n_windows / (time.perf_counter() - t0)


__all__ = ["ConcatStream", "WindowLoader", "batch_to_device", "batch_bytes_per_step", "bytes_per_token",
           "matched_token_seq_len", "load_bpt_stats", "segment_starts", "make_labels", "pack_windows",
           "windows_per_s"]
