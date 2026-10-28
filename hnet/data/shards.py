"""Shard formats: modded-nanogpt GPT-2 token shards and our aligned byte shards.

Token shard (modded-nanogpt, read-only source of truth)
    header: 256 x int32 = [magic 20240520, version 1, ntok, 0, ...]   (1024 bytes)
    payload: ntok x uint16 GPT-2 token ids.
    Every document is preceded by <|endoftext|> (50256). Documents are *split across shard boundaries*
    (modded-nanogpt's fineweb.py puts the remainder of the doc that overflows shard k at the start of
    shard k+1, without a leading <|endoftext|>). So a shard may start with a fragment and end with a fragment.

Byte shard (ours), file ``<stem>.bytes.bin``
    header: 128 x int64 (1024 bytes) = [BYTE_MAGIC, BYTE_VERSION, nbytes, ndocs, ntok_source, flags, 0...]
    payload: nbytes x uint8.
    A token-shard document ``[EOT] + toks`` becomes ``[BOS_ID] + decode_bytes(toks) + [EOS_ID]``.
    * a leading fragment (shard does not start with EOT) gets no BOS;
    * the last document gets an EOS only if it is complete (the next shard starts with EOT, or the caller says
      so); otherwise its bytes continue in the next byte shard, exactly mirroring the token stream.
    Hence concatenating byte shards in order gives the same document sequence as concatenating token shards.

Doc index, file ``<stem>.idx.npy``: int64 [ndocs, 5], one row per document (or fragment) in shard order
    col 0 BYTE_OFF : offset of the doc's first element in the byte payload (its BOS if it has one)
    col 1 N_BYTES  : number of text bytes (BOS/EOS excluded)
    col 2 TOK_OFF  : offset of the doc's first element in the token payload (its EOT if it has one)
    col 3 N_TOKENS : number of text tokens (EOT excluded)
    col 4 FLAGS    : bit0 HAS_BOS (doc starts in this shard), bit1 HAS_EOS (doc ends in this shard)
    Text bytes of doc i = bytes[BYTE_OFF + HAS_BOS : ... + N_BYTES] = decode_bytes(tokens[TOK_OFF + HAS_BOS : ... + N_TOKENS]).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from hnet.config import BOS_ID, EOS_ID

# ---------------------------------------------------------------- token shards
TOKEN_MAGIC = 20240520
TOKEN_VERSION = 1
HEADER_BYTES = 1024
EOT_ID = 50256  # GPT-2 <|endoftext|>

# ---------------------------------------------------------------- byte shards
BYTE_MAGIC = 0x48_4E_45_54_42_59_54_45  # b"HNETBYTE" as a big int
BYTE_VERSION = 1
FLAG_LAST_COMPLETE = 1  # header flag: last doc of the shard has its EOS

# doc index columns
BYTE_OFF, N_BYTES, TOK_OFF, N_TOKENS, FLAGS = range(5)
HAS_BOS, HAS_EOS = 1, 2


def write_token_shard(path, tokens) -> None:
    tokens = np.asarray(tokens)
    assert tokens.ndim == 1 and len(tokens) < 2**31
    assert tokens.size == 0 or (tokens.min() >= 0 and tokens.max() < 2**16)
    header = np.zeros(256, dtype=np.int32)
    header[:3] = (TOKEN_MAGIC, TOKEN_VERSION, len(tokens))
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16, copy=False).tobytes())
    os.replace(tmp, path)


def token_shard_ntok(path) -> int:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == TOKEN_MAGIC, f"{path}: bad magic {header[0]}"
    assert header[1] == TOKEN_VERSION, f"{path}: unsupported version {header[1]}"
    ntok = int(header[2])
    expect = HEADER_BYTES + 2 * ntok
    size = os.path.getsize(path)
    assert size == expect, f"{path}: size {size} != header-implied {expect} (truncated download?)"
    return ntok


def read_token_shard(path) -> np.ndarray:
    """Memory-mapped uint16 view of a token shard payload."""
    ntok = token_shard_ntok(path)
    if ntok == 0:
        return np.zeros(0, dtype=np.uint16)
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HEADER_BYTES, shape=(ntok,))


def byte_paths(out_dir, stem: str):
    out_dir = Path(out_dir)
    return out_dir / f"{stem}.bytes.bin", out_dir / f"{stem}.idx.npy"


def write_byte_shard(path, data: np.ndarray, ndocs: int, ntok_source: int, last_complete: bool) -> None:
    data = np.asarray(data, dtype=np.uint8)
    header = np.zeros(128, dtype=np.int64)
    header[:6] = (BYTE_MAGIC, BYTE_VERSION, len(data), ndocs, ntok_source, FLAG_LAST_COMPLETE if last_complete else 0)
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header.tobytes())
        f.write(data.tobytes())
    os.replace(tmp, path)


def byte_shard_header(path) -> dict:
    h = np.fromfile(path, dtype=np.int64, count=128)
    assert h[0] == BYTE_MAGIC, f"{path}: not a byte shard"
    assert h[1] == BYTE_VERSION, f"{path}: unsupported byte-shard version {h[1]}"
    d = dict(nbytes=int(h[2]), ndocs=int(h[3]), ntok_source=int(h[4]), last_complete=bool(h[5] & FLAG_LAST_COMPLETE))
    size = os.path.getsize(path)
    assert size == HEADER_BYTES + d["nbytes"], f"{path}: size {size} does not match header"
    return d


def read_byte_shard(path) -> np.ndarray:
    """Memory-mapped uint8 view of a byte shard payload."""
    n = byte_shard_header(path)["nbytes"]
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    return np.memmap(path, dtype=np.uint8, mode="r", offset=HEADER_BYTES, shape=(n,))


def read_doc_index(path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


# ---------------------------------------------------------------- GPT-2 byte decoding
_VOCAB_CACHE = {}


def gpt2_vocab_table():
    """(flat uint8 bytes, int64 offsets[V+1], int64 lengths[V]) for GPT-2; EOT maps to 0 bytes."""
    if "gpt2" not in _VOCAB_CACHE:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        pieces = [enc.decode_single_token_bytes(t) for t in range(enc.n_vocab)]
        pieces[EOT_ID] = b""
        lens = np.array([len(p) for p in pieces], dtype=np.int64)
        offs = np.zeros(len(pieces) + 1, dtype=np.int64)
        np.cumsum(lens, out=offs[1:])
        flat = np.frombuffer(b"".join(pieces), dtype=np.uint8)
        _VOCAB_CACHE["gpt2"] = (flat, offs, lens)
    return _VOCAB_CACHE["gpt2"]


def token_byte_lengths(tokens: np.ndarray) -> np.ndarray:
    """UTF-8 byte length of each GPT-2 token (0 for <|endoftext|>)."""
    _, _, lens = gpt2_vocab_table()
    return lens[np.asarray(tokens, dtype=np.int64)]


def decode_tokens_to_stream(tokens: np.ndarray, table=None) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact GPT-2 byte decoding of a token stream, with EOT -> [EOS, BOS].

    Returns (bytes uint8, out_off int64[len(tokens)+1]) where out_off[i] is the output offset of token i.
    Equivalent to tiktoken ``decode_bytes`` per document (verified in tests and by prepare.py).
    """
    flat, offs, lens = table if table is not None else gpt2_vocab_table()
    tok = np.asarray(tokens, dtype=np.int64)
    # EOT emits two marker bytes; we handle it by giving it length 2 and patching afterwards.
    tl = lens[tok]
    is_eot = tok == EOT_ID
    tl = np.where(is_eot, 2, tl)
    out_off = np.zeros(len(tok) + 1, dtype=np.int64)
    np.cumsum(tl, out=out_off[1:])
    total = int(out_off[-1])
    # gather index into `flat`: for output byte j of token i: offs[tok[i]] + (j - out_off[i])
    src_start = offs[tok] - out_off[:-1]
    idx = np.repeat(src_start, tl) + np.arange(total, dtype=np.int64)
    eot_pos = out_off[:-1][is_eot]
    idx[eot_pos] = 0  # placeholder, overwritten below
    idx[eot_pos + 1] = 0
    out = flat[idx]
    out[eot_pos] = EOS_ID
    out[eot_pos + 1] = BOS_ID
    return out, out_off


def convert_token_array(tokens: np.ndarray, last_complete: bool, chunk_tokens: int = 1 << 20):
    """Token shard payload -> (byte payload uint8, doc index int64[ndocs, 5]).

    `last_complete`: whether the shard's last document ends in this shard (gets an EOS).
    Works `chunk_tokens` at a time, so peak memory is the output plus ~40 bytes per chunk token.

    Offsets: in the "full" decode where every EOT becomes [EOS, BOS], token i starts at
    F(i) = textbytes(tokens[:i]) + 2 * #EOT(tokens[:i]). The shard payload drops the first EOS when the
    shard starts with EOT (lead = 1; there is no previous doc to end) and appends an EOS when
    `last_complete`. So payload offset = F(i) - lead.
    """
    tokens = np.asarray(tokens)
    n = len(tokens)
    table = gpt2_vocab_table()
    lens = table[2]
    eot_pos = np.flatnonzero(tokens == EOT_ID).astype(np.int64)
    starts_with_eot = bool(n > 0 and tokens[0] == EOT_ID)
    lead = int(starts_with_eot)
    if n == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros((0, 5), dtype=np.int64)
    doc_start = eot_pos if starts_with_eot else np.concatenate([np.zeros(1, np.int64), eot_pos])
    n_docs = len(doc_start)
    text_total = 0
    for a in range(0, n, chunk_tokens):
        text_total += int(lens[np.asarray(tokens[a : a + chunk_tokens], dtype=np.int64)].sum())
    total = text_total + 2 * len(eot_pos) - lead + int(last_complete)
    out = np.empty(total, dtype=np.uint8)
    tb_start = np.zeros(n_docs + 1, dtype=np.int64)  # text bytes before each doc start (+ total at the end)
    running = 0
    for a in range(0, n, chunk_tokens):
        b = min(n, a + chunk_tokens)
        chunk = np.asarray(tokens[a:b], dtype=np.int64)
        cum = np.zeros(b - a + 1, dtype=np.int64)
        np.cumsum(lens[chunk], out=cum[1:])
        lo, hi = np.searchsorted(doc_start, [a, b])
        tb_start[lo:hi] = running + cum[doc_start[lo:hi] - a]
        piece, _ = decode_tokens_to_stream(chunk, table)
        off = running + 2 * int(np.searchsorted(eot_pos, a)) - lead
        if a == 0 and lead:
            piece, off = piece[1:], 0
        out[off : off + len(piece)] = piece
        running += int(cum[-1])
    tb_start[-1] = running
    if last_complete:
        out[-1] = EOS_ID

    idx = np.zeros((n_docs, 5), dtype=np.int64)
    has_bos = np.ones(n_docs, dtype=np.int64)
    if not starts_with_eot:
        has_bos[0] = 0
    has_eos = np.ones(n_docs, dtype=np.int64)
    has_eos[-1] = int(last_complete)
    doc_end = np.concatenate([doc_start[1:], [n]])
    n_eot_before = np.searchsorted(eot_pos, doc_start)  # EOTs strictly before each doc start
    idx[:, TOK_OFF] = doc_start
    idx[:, N_TOKENS] = doc_end - doc_start - has_bos
    idx[:, N_BYTES] = tb_start[1:] - tb_start[:-1]
    # BOS of a doc starting at EOT position p is at F(p) + 1 - lead; a leading fragment starts at 0.
    idx[:, BYTE_OFF] = tb_start[:-1] + 2 * n_eot_before + has_bos - lead
    idx[:, FLAGS] = has_bos * HAS_BOS + has_eos * HAS_EOS
    return out, idx


def doc_text_bytes(data: np.ndarray, idx_row) -> np.ndarray:
    off = int(idx_row[BYTE_OFF]) + int(idx_row[FLAGS] & HAS_BOS)
    return data[off : off + int(idx_row[N_BYTES])]


def doc_text_tokens(tokens: np.ndarray, idx_row) -> np.ndarray:
    off = int(idx_row[TOK_OFF]) + int(idx_row[FLAGS] & HAS_BOS)
    return tokens[off : off + int(idx_row[N_TOKENS])]


def convert_shard(token_path, out_dir, last_complete: bool | None = None, next_token_path=None, chunk_tokens=1 << 20):
    """Convert one token shard to a byte shard + doc index. Returns a stats dict.

    `last_complete`: if None, inferred from `next_token_path` (the next shard in stream order): complete iff it
    starts with EOT; if there is no next shard, the last doc is assumed complete.
    """
    token_path = Path(token_path)
    stem = token_path.name[: -len(".bin")] if token_path.name.endswith(".bin") else token_path.name
    bpath, ipath = byte_paths(out_dir, stem)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if last_complete is None:
        last_complete = True
        if next_token_path is not None and Path(next_token_path).exists():
            nxt = read_token_shard(next_token_path)
            last_complete = len(nxt) == 0 or int(nxt[0]) == EOT_ID
    tokens = read_token_shard(token_path)
    data, idx = convert_token_array(tokens, last_complete, chunk_tokens)
    write_byte_shard(bpath, data, len(idx), len(tokens), last_complete)
    tmp = str(ipath) + ".tmp.npy"
    np.save(tmp, idx)
    os.replace(tmp, ipath)
    return dict(
        shard=stem,
        ntok=int(len(tokens)),
        nbytes_stream=int(len(data)),
        n_text_bytes=int(idx[:, N_BYTES].sum()) if len(idx) else 0,
        n_text_tokens=int(idx[:, N_TOKENS].sum()) if len(idx) else 0,
        ndocs=int(len(idx)),
        n_marker_in_text=int(np.count_nonzero(data >= BOS_ID) - np.count_nonzero(idx[:, FLAGS] & HAS_BOS)
                             - np.count_nonzero(idx[:, FLAGS] & HAS_EOS)) if len(idx) else 0,
        last_complete=bool(last_complete),
    )
