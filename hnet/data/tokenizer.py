"""Tokenizers behind the shard formats: GPT-2 (modded-nanogpt shards) and in-corpus byte-level BPE (data mixes).

Every tokenizer used for token shards exposes the same small interface (``TokenizerInfo``):

* ``eot_id``      : the document-start token written before every document in token shards;
* ``vocab_size``  : number of ids (embedding rows needed; pad to a multiple of 128 in the model config);
* ``table``       : ``(flat uint8, offsets int64[V+1], lengths int64[V])`` exact UTF-8 bytes of every id (special
                    tokens map to 0 bytes), used for vectorized decoding and per-token byte counts;
* ``byte_lengths(ids)`` / ``decode_bytes(ids)``.

In-corpus BPE (``BPETokenizer``) is a HF ``tokenizers`` byte-level BPE. With ``pretokenize=True`` (default) the
pretokenizer is the Llama-3 / cl100k regex followed by ByteLevel (exactly the Llama-3 ``tokenizer.json``
pipeline); with ``pretokenize=False`` (BPE-NP, blog part II) only the ByteLevel byte->char map is applied, so
merges may cross word and line boundaries. The special token ``<|endoftext|>`` is id 0 and is never produced from
document text (``encode_special_tokens = True``: a literal "<|endoftext|>" in a document is encoded as ordinary
text), so every id 0 in a token shard is a document start.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np

EOT_TOKEN = "<|endoftext|>"
# Llama-3 (tiktoken cl100k-style with \p{N}{1,3}) pretokenization regex, as in Meta-Llama-3 tokenizer.json.
LLAMA3_PATTERN = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*"
                  r"|\s*[\r\n]+|\s+(?!\S)|\s+")


@functools.lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 / HF ByteLevel reversible byte -> printable unicode char map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


@functools.lru_cache(maxsize=1)
def unicode_to_bytes() -> dict[str, int]:
    return {c: b for b, c in bytes_to_unicode().items()}


def _table_from_pieces(pieces: list[bytes]):
    lens = np.array([len(p) for p in pieces], dtype=np.int64)
    offs = np.zeros(len(pieces) + 1, dtype=np.int64)
    np.cumsum(lens, out=offs[1:])
    flat = np.frombuffer(b"".join(pieces), dtype=np.uint8).copy()
    return flat, offs, lens


def decode_with_table(ids, table) -> np.ndarray:
    """Concatenated UTF-8 bytes of a 1-D id array (vectorized; special tokens decode to nothing)."""
    flat, offs, lens = table
    ids = np.asarray(ids, dtype=np.int64)
    tl = lens[ids]
    out_off = np.zeros(len(ids) + 1, dtype=np.int64)
    np.cumsum(tl, out=out_off[1:])
    idx = np.repeat(offs[ids] - out_off[:-1], tl) + np.arange(int(out_off[-1]), dtype=np.int64)
    return flat[idx]


class TokenizerInfo:
    name: str
    eot_id: int
    vocab_size: int

    @property
    def table(self):
        raise NotImplementedError

    def byte_lengths(self, ids) -> np.ndarray:
        return self.table[2][np.asarray(ids, dtype=np.int64)]

    def decode_bytes(self, ids) -> bytes:
        return decode_with_table(ids, self.table).tobytes()


class GPT2Tokenizer(TokenizerInfo):
    """The modded-nanogpt GPT-2 shards' tokenizer (tiktoken)."""

    name = "gpt2"
    eot_id = 50256
    vocab_size = 50257

    @property
    def table(self):
        from hnet.data.shards import gpt2_vocab_table

        return gpt2_vocab_table()

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return [np.asarray(t, dtype=np.int64) for t in enc.encode_ordinary_batch(texts)]


class BPETokenizer(TokenizerInfo):
    """Byte-level BPE from a HF ``tokenizers`` json (see module docstring)."""

    def __init__(self, path_or_json: str | Path, name: str | None = None):
        from tokenizers import Tokenizer

        p = Path(str(path_or_json)) if not str(path_or_json).lstrip().startswith("{") else None
        js = p.read_text() if p is not None else str(path_or_json)
        self.json = js
        self.sha256 = hashlib.sha256(js.encode()).hexdigest()
        self.tok = Tokenizer.from_str(js)
        self.tok.encode_special_tokens = True  # document text never yields special ids
        self.name = name or (p.stem if p is not None else "bpe")
        self.path = str(p) if p is not None else None
        eot = self.tok.token_to_id(EOT_TOKEN)
        assert eot is not None, f"{self.name}: no {EOT_TOKEN} token"
        self.eot_id = int(eot)
        self.vocab_size = int(self.tok.get_vocab_size(with_added_tokens=True))
        self.pretokenize = '"Split"' in js
        self._table = None

    @property
    def table(self):
        if self._table is None:
            u2b = unicode_to_bytes()
            vocab = self.tok.get_vocab(with_added_tokens=True)
            special = {t["content"] for t in json.loads(self.json).get("added_tokens", []) if t.get("special")}
            pieces = [b""] * self.vocab_size
            for s, i in vocab.items():
                pieces[i] = b"" if s in special else bytes(u2b[c] for c in s)
            self._table = _table_from_pieces(pieces)
        return self._table

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        encs = self.tok.encode_batch_fast(texts, add_special_tokens=False)
        return [np.asarray(e.ids, dtype=np.int64) for e in encs]

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]


def build_bpe_tokenizer(pretokenize: bool = True):
    """Untrained HF tokenizer with the Llama-3 pipeline (or ByteLevel only for BPE-NP)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    tok = Tokenizer(models.BPE(ignore_merges=False))
    bl = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=False)
    if pretokenize:
        tok.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Split(Regex(LLAMA3_PATTERN), behavior="isolated", invert=False), bl])
    else:
        tok.pre_tokenizer = bl
    tok.decoder = decoders.ByteLevel()
    return tok


def train_bpe(texts, vocab_size: int = 32768, pretokenize: bool = True, min_frequency: int = 2,
              max_token_length: int | None = None, length: int | None = None, show_progress: bool = False) -> str:
    """Train an in-corpus byte-level BPE on an iterable of strings; returns the tokenizer json string.

    ``vocab_size`` counts everything: <|endoftext|> (id 0) + the 256 byte symbols + merges."""
    from tokenizers import pre_tokenizers, trainers

    tok = build_bpe_tokenizer(pretokenize)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency, special_tokens=[EOT_TOKEN],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=show_progress,
                                  max_token_length=max_token_length)
    # A literal "<|endoftext|>" in training text could otherwise be merged into a token whose string equals the
    # special token (id 0) when merges may span it (BPE-NP), so hide it from the trainer.
    texts = (t.replace(EOT_TOKEN, "<|endoftext |>") if EOT_TOKEN in t else t for t in texts)
    tok.train_from_iterator(texts, trainer=trainer, length=length)
    js = tok.to_str()
    t = BPETokenizer(js)
    assert t.eot_id == 0, t.eot_id
    merged = {"".join(m) if isinstance(m, list) else m.replace(" ", "") for m in json.loads(js)["model"]["merges"]}
    assert EOT_TOKEN not in merged, "a merge produces the special token string"
    return js


def load_tokenizer(spec) -> TokenizerInfo:
    """``"gpt2"`` or a path to a tokenizers json."""
    if isinstance(spec, TokenizerInfo):
        return spec
    if spec in (None, "gpt2"):
        return GPT2Tokenizer()
    return _load_bpe_cached(str(Path(spec).resolve()))


@functools.lru_cache(maxsize=8)
def _load_bpe_cached(path: str) -> BPETokenizer:
    return BPETokenizer(path)


def dataset_tokenizer(dataset: str, root=None) -> TokenizerInfo:
    """Tokenizer of a dataset's token shards: the mix's own tokenizer.json for data mixes, else GPT-2."""
    from hnet.data import mix

    if mix.is_mix(dataset, root):
        return load_tokenizer(mix.mix_dir(dataset, root) / "tokenizer.json")
    return GPT2Tokenizer()


__all__ = ["BPETokenizer", "GPT2Tokenizer", "TokenizerInfo", "LLAMA3_PATTERN", "EOT_TOKEN", "train_bpe",
           "build_bpe_tokenizer", "load_tokenizer", "dataset_tokenizer", "decode_with_table", "bytes_to_unicode"]
