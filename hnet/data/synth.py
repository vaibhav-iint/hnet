"""Tiny synthetic datasets in the real on-disk formats, for tests and CPU smoke runs of other workstreams.

    from hnet.data.synth import make_synthetic_dataset
    info = make_synthetic_dataset(tmp_dir, n_docs=200, shard_tokens=5000)
    info["token_paths"], info["byte_paths"], info["val_token_path"], info["docs"]

Mirrors modded-nanogpt's fineweb.py sharding: every doc is [EOT] + encode_ordinary(text), shard 0 is val and
shards 1.. are train, and a doc that overflows a shard continues at the start of the next one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from hnet.data import shards as S

_WORDS = ["the", "of", "and", "data", "byte", "token", "model", "learning", "hierarchical", "chunk", "Paris",
          "naïve", "café", "Ünïcödé", "中文", "日本語", "😀", "🚀🚀", "—", "\n", "\n\n", "  ", "\t", "1234", "3.14",
          "don't", "it's", "<|endoftext|>", "\x00", "ÿþ", "Ω≈ç√", "é"]


def random_text(rng: np.random.Generator, n_words: int) -> str:
    w = rng.choice(len(_WORDS), size=n_words)
    seps = rng.choice([" ", "", " ", ", ", ". "], size=n_words)
    return "".join(_WORDS[i] + s for i, s in zip(w, seps))


def make_docs(n_docs: int, seed: int = 0, min_words: int = 1, max_words: int = 400) -> list[str]:
    rng = np.random.default_rng(seed)
    lens = rng.integers(min_words, max_words, size=n_docs)
    # a few long docs to exercise window/chunk splitting
    lens[rng.choice(n_docs, size=max(1, n_docs // 20), replace=False)] *= 10
    return [random_text(rng, int(n)) for n in lens]


def encode_docs(docs: list[str]) -> np.ndarray:
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    out = []
    for d in docs:
        out.append(S.EOT_ID)
        out.extend(enc.encode_ordinary(d))
    return np.array(out, dtype=np.uint16)


def make_synthetic_dataset(out_dir, n_docs: int = 200, shard_tokens: int = 5000, seed: int = 0,
                           prefix: str = "synth", convert: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = make_docs(n_docs, seed)
    stream = encode_docs(docs)
    token_paths = []
    for k, a in enumerate(range(0, len(stream), shard_tokens)):
        split = "val" if k == 0 else "train"
        p = out_dir / f"{prefix}_{split}_{k:06d}.bin"
        S.write_token_shard(p, stream[a : a + shard_tokens])
        token_paths.append(p)
    byte_paths, idx_paths = [], []
    if convert:
        bdir = out_dir / "bytes"
        for k, p in enumerate(token_paths):
            nxt = token_paths[k + 1] if k + 1 < len(token_paths) else None
            S.convert_shard(p, bdir, next_token_path=nxt)
            bp, ip = S.byte_paths(bdir, p.name[:-4])
            byte_paths.append(bp)
            idx_paths.append(ip)
    return dict(docs=docs, stream=stream, token_paths=token_paths, byte_paths=byte_paths, idx_paths=idx_paths,
                val_token_path=token_paths[0], train_token_paths=token_paths[1:], train_byte_paths=byte_paths[1:])


# ---------------------------------------------------------------------------------------------- data mixes
class InMemoryParquet:
    """Stand-in for hnet.data.hf_sources.ParquetReader over {file: [row group {column: list}]} (tests)."""

    def __init__(self, files: dict):
        self.files = files

    def list_files(self):
        return sorted(self.files)

    def num_row_groups(self, f):
        return len(self.files[f])

    def read(self, f, a, b, columns):
        return {c: sum((rg[c] for rg in self.files[f][a:b]), []) for c in columns}

    def prefetch(self, f):
        pass

    def release(self, f):
        pass

    def close(self):
        pass


SYNTH_MIX_SPEC = {
    "name": "synthmix", "seed": 0, "max_doc_bytes": 20000,
    "categories": {"en": 0.8, "code": 0.2},
    "sources": {
        "web": {"repo": "synth/web", "revision": "r0", "files": ["web/*"], "text": "text", "p_val": 0.1,
                "leaves": {"en.web": {"weight": 1}}},
        "code": {"repo": "synth/code", "revision": "r0", "files": ["code/*"], "text": "code", "select": "language",
                 "split_key": "repo_name", "p_val": 0.2,
                 "leaves": {"code.a": {"value": "A", "weight": 1}, "heldout.h": {"value": "H", "heldout": True}}},
    },
}


def make_synthetic_mix(root, dataset: str = "synthmix", train_bytes: float = 200_000, seed: int = 0,
                       vocab_size: int = 512) -> dict:
    """Fetch + tokenizer + assemble of a tiny two-source mix from in-memory 'parquet' (no network)."""
    from hnet.data import mix as M
    from hnet.data.tokenizer import train_bpe

    rng = np.random.default_rng(seed)
    files = {}
    for f in range(4):
        files[f"web/{f}"] = [{"text": [random_text(rng, int(rng.integers(5, 300))).encode() for _ in range(40)]}
                             for _ in range(4)]
        files[f"code/{f}"] = [{"code": [("def f(x):\n    return " + random_text(rng, int(rng.integers(3, 100))))
                                        .encode() for _ in range(40)],
                               "language": list(rng.choice(["A", "H"], 40, p=[0.8, 0.2])),
                               "repo_name": [f"r{int(i)}" for i in rng.integers(0, 50, 40)]} for _ in range(4)]
    spec = M.load_spec(SYNTH_MIX_SPEC)
    cfg = M.BuildConfig(dataset=dataset, train_bytes=train_bytes, shard_bytes=train_bytes / 4, val_leaf_bytes=8000,
                        val_group_bytes=8000, val_mix_bytes=20000, tokenizer_sample_bytes=train_bytes)
    M.fetch(spec, cfg, root, lambda src: InMemoryParquet(files), workers=1, log=lambda s: None)
    it, _, _ = M.tokenizer_sample(spec, cfg, root)
    tok_path = Path(root) / f"{dataset}_tok.json"
    tok_path.write_text(train_bpe(list(it), vocab_size=vocab_size))
    man = M.assemble(spec, cfg, tok_path, root, log=lambda s: None)
    return dict(spec=spec, cfg=cfg, manifest=man, tokenizer=tok_path)
