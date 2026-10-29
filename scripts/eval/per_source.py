#!/usr/bin/env python
"""Per-source bits per byte and compression of a trained checkpoint (BPE Transformer or byte H-Net).

    python scripts/eval/per_source.py --model runs/ladder/hnet_1e17_L8 --sources data/mix/val_sources.yaml
    python scripts/eval/per_source.py --model runs/ladder/bpe_1e17_L8  --source finewebedu=data/.../val.bytes.bin
    python scripts/eval/per_source.py --model RUN                       # default: FineWeb-Edu val (english)

Writes ``<run>/per_source.json`` (or ``--out``): per source and per group, bpb (+ bootstrap CI over documents),
bytes per main-network position (H-Net: realized compression on that source) and bytes per token (BPE).
``scripts/analysis/fit_ladder.py`` picks these up as metrics ``src:<name>`` and ``group:<group>``.

Sources manifest (YAML or JSON; a list, or ``{sources: [...]}``; relative paths resolve against the manifest's
directory, then the repo root)::

    sources:
      - {name: fineweb_edu, group: english, weight: 0.70, paths: [val/english/fineweb_edu.bytes.bin]}
      - {name: python, group: code, lang: python, paths: [val/code/python.jsonl.gz], text_field: content}
      - {name: haskell, group: code, lang: haskell, held_out: true, paths: [val/code_ood/haskell.jsonl]}
      - {name: wikitext103, group: paloma_like, doc_split: wikitext,
         paths: ["hf://Salesforce/wikitext/wikitext-103-raw-v1/test-00000-of-00001.parquet"]}

Per-source keys: ``name``, ``paths`` (or ``path``), ``group`` (english|code|math|multilingual|...), ``lang``,
``held_out`` (bool), ``weight`` (mixture weight, for the ``mix`` group), ``max_docs``, ``max_bytes`` (default
``--max-bytes``), ``text_field`` (jsonl/parquet; default text, then content), ``doc_split`` (``wikitext``).
File formats: our byte shards ``<stem>.bytes.bin`` (+ ``<stem>.idx.npy``; complete documents only),
modded-nanogpt GPT-2 token shards ``*.bin``, ``*.jsonl[.gz]`` / ``*.json.gz``, ``*.parquet``, ``*.txt`` (one doc),
and ``hf://<repo>/<file>[@revision]`` for any of those inside a HF dataset repo.

Protocol (identical text for every model; see docs/ANALYSIS_LADDER.md)
---------------------------------------------------------------------
* Documents: in file order, valid UTF-8 only, until the source's ``max_bytes`` (whole documents) / ``max_docs``.
* Windows: each document is cut greedily into windows at token boundaries of a fixed *split tokenizer*
  (``--split-tokenizer``, default GPT-2) that are also UTF-8 character starts, with at most ``byte_seq_len - 1``
  bytes and ``token_seq_len - 1`` split tokens per window (the ``hnet.data.valsets`` protocol; defaults from the
  run's ``eval.val_byte_seq_len`` / ``eval.val_token_seq_len``). Window c is scored given a one-element prefix:
  BOS / <|endoftext|> for c = 0, else the last byte / token of window c-1 (context only, never scored).
* Byte models: bits of every byte of the window. Token models: bits of every token of the window (the slice of
  the document's tokens when the model's tokenizer is the split tokenizer, else the window text re-encoded); the
  bpb denominator is the window's UTF-8 bytes in both cases. No end-of-document target.
* Compression (H-Net): positions entering each stage, counted from the routers' boundary masks in eval mode
  (running thresholds, so the ratio is free to vary per source). ``bytes_per_main_position`` = stage-0 positions
  (window bytes + prefix) / innermost positions, the training-log ``compression`` convention.
  BPE: ``bytes_per_token`` = window bytes / scored model tokens. ``split_bytes_per_token`` is the split
  tokenizer's ratio (reference, also for byte models).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from hnet.config import BOS_ID  # noqa: E402

LN2 = math.log(2.0)
DEFAULT_VAL = "data/finewebedu10B/bytes/finewebedu_val_000000.bytes.bin"


# ====================================================================================================== sources
@dataclass
class Source:
    name: str
    paths: List[str]
    group: str = "other"
    lang: Optional[str] = None
    held_out: bool = False
    weight: Optional[float] = None
    max_docs: Optional[int] = None
    max_bytes: Optional[int] = None
    text_field: Optional[str] = None
    doc_split: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def meta(self) -> dict:
        return {"group": self.group, "lang": self.lang, "held_out": self.held_out, "weight": self.weight,
                "paths": self.paths}


def _resolve(p: str, base: Optional[str]) -> str:
    if p.startswith("hf://") or os.path.isabs(p):
        return p
    for b in ([base] if base else []) + [os.getcwd(), ROOT]:
        q = os.path.join(b, p)
        if os.path.exists(q):
            return q
    return p


def load_manifest(path: str) -> List[Source]:
    with open(path) as f:
        txt = f.read()
    if path.endswith((".yaml", ".yml")):
        import yaml

        d = yaml.safe_load(txt)
    else:
        d = json.loads(txt)
    items = d["sources"] if isinstance(d, dict) else d
    base = os.path.dirname(os.path.abspath(path))
    out = []
    known = {"name", "paths", "path", "group", "lang", "held_out", "weight", "max_docs", "max_bytes", "text_field",
             "doc_split"}
    for it in items:
        paths = it.get("paths") or [it["path"]]
        if isinstance(paths, str):
            paths = [paths]
        out.append(Source(name=it["name"], paths=[_resolve(p, base) for p in paths], group=it.get("group", "other"),
                          lang=it.get("lang"), held_out=bool(it.get("held_out", False)), weight=it.get("weight"),
                          max_docs=it.get("max_docs"), max_bytes=it.get("max_bytes"),
                          text_field=it.get("text_field"), doc_split=it.get("doc_split"),
                          extra={k: v for k, v in it.items() if k not in known}))
    names = [s.name for s in out]
    assert len(set(names)) == len(names), f"duplicate source names in {path}"
    return out


def parse_source_arg(s: str) -> Source:
    """``NAME=PATH[,PATH...][@GROUP]`` (group optional, e.g. ``python=val/py.jsonl@code``)."""
    name, _, rest = s.partition("=")
    rest, _, group = rest.partition("@")
    return Source(name=name, paths=[_resolve(p, None) for p in rest.split(",")], group=group or "other")


def _open_local(path: str) -> str:
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    spec = path[len("hf://"):]
    spec, _, rev = spec.partition("@")
    parts = spec.split("/")
    repo, fn = "/".join(parts[:2]), "/".join(parts[2:])
    return hf_hub_download(repo, fn, repo_type="dataset", revision=rev or None)


def _text_of(row: dict, field_name: Optional[str]):
    for k in ([field_name] if field_name else []) + ["text", "content", "code"]:
        if k and k in row and isinstance(row[k], str):
            return row[k]
    return None


def iter_docs(path: str, text_field: Optional[str] = None) -> Iterator[bytes]:
    """Documents (UTF-8 bytes) of one file, in file order."""
    p = _open_local(path)
    if p.endswith(".bytes.bin"):
        from hnet.data import shards as S

        data = S.read_byte_shard(p)
        idx = S.read_doc_index(p[: -len(".bytes.bin")] + ".idx.npy")
        for row in idx:
            fl = int(row[S.FLAGS])
            if not (fl & S.HAS_BOS and fl & S.HAS_EOS):
                continue  # fragment of a document split across shards
            off = int(row[S.BYTE_OFF]) + 1
            yield bytes(data[off: off + int(row[S.N_BYTES])])
    elif p.endswith(".bin"):
        from hnet.data import shards as S
        from hnet.data.valsets import complete_docs

        toks = S.read_token_shard(p)
        for a, b in complete_docs(toks):
            t = np.asarray(toks[a + 1: b], dtype=np.int64)
            if len(t):
                yield S.decode_tokens_to_stream(t)[0].tobytes()
    elif p.endswith((".jsonl", ".jsonl.gz", ".json.gz", ".ndjson")):
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = _text_of(json.loads(line), text_field)
                    if t:
                        yield t.encode("utf-8")
    elif p.endswith(".parquet"):
        import pyarrow.parquet as pq

        for row in pq.read_table(p).to_pylist():
            t = _text_of(row, text_field)
            if t:
                yield t.encode("utf-8")
    else:  # plain text: one document
        with open(p, "rb") as f:
            yield f.read()


def _wikitext_articles(docs: Iterator[bytes]) -> Iterator[bytes]:
    """Wikitext parquet rows are lines; an article starts at a top-level heading line `` = Title = ``."""
    cur: List[bytes] = []
    for line in docs:
        s = line.strip()
        if s.startswith(b"= ") and s.endswith(b" =") and not s.startswith(b"= =") and cur:
            yield b"".join(cur).strip()
            cur = []
        cur.append(line)
    if cur:
        yield b"".join(cur).strip()


def load_source_docs(src: Source, default_max_bytes: Optional[int]) -> dict:
    max_bytes = src.max_bytes if src.max_bytes is not None else default_max_bytes
    docs, nb, skipped = [], 0, 0

    def stream():
        for p in src.paths:
            yield from iter_docs(p, src.text_field)

    it = _wikitext_articles(stream()) if src.doc_split == "wikitext" else stream()
    for d in it:
        if max_bytes is not None and nb >= max_bytes:
            break
        if src.max_docs is not None and len(docs) >= src.max_docs:
            break
        if not d:
            continue
        try:
            d.decode("utf-8")
        except UnicodeDecodeError:
            skipped += 1
            continue
        docs.append(d)
        nb += len(d)
    return {"docs": docs, "n_bytes": nb, "skipped_invalid_utf8": skipped}


# ====================================================================================================== tokenizers
class Tokenizer:
    """``gpt2`` / any tiktoken encoding name, or ``hf:<name or path>`` (fast tokenizer with offsets)."""

    def __init__(self, spec: str = "gpt2"):
        self.spec = spec
        if spec.startswith("hf:"):
            from transformers import AutoTokenizer

            self.hf = AutoTokenizer.from_pretrained(spec[3:])
            self.tt = None
            self.eot = self.hf.eos_token_id if self.hf.eos_token_id is not None else self.hf.bos_token_id
        else:
            import tiktoken

            self.tt = tiktoken.get_encoding(spec)
            self.hf = None
            self.eot = self.tt.eot_token

    def encode(self, text: str) -> List[int]:
        if self.tt is not None:
            return self.tt.encode_ordinary(text)
        return self.hf.encode(text, add_special_tokens=False)

    def encode_with_lens(self, text: str):
        """-> (ids, UTF-8 byte length of each token; sums to len(text.encode()))."""
        if self.tt is not None:
            ids = self.tt.encode_ordinary(text)
            return ids, np.array([len(self.tt.decode_single_token_bytes(t)) for t in ids], dtype=np.int64)
        enc = self.hf(text, add_special_tokens=False, return_offsets_mapping=True)
        cb = np.zeros(len(text) + 1, dtype=np.int64)
        np.cumsum([len(ch.encode("utf-8")) for ch in text], out=cb[1:])
        ends = np.array([cb[e] for _, e in enc["offset_mapping"]], dtype=np.int64)
        ends = np.maximum.accumulate(ends) if len(ends) else ends
        if len(ends):
            ends[-1] = cb[-1]
        return list(enc["input_ids"]), np.diff(np.concatenate([[0], ends]))


def make_windows(doc: bytes, ids, lens: np.ndarray, byte_seq_len: int, token_seq_len: int):
    """Greedy windows [(byte_a, byte_b, tok_a, tok_b)] at token boundaries that are UTF-8 character starts."""
    n = len(ids)
    cb = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(lens, out=cb[1:])
    assert cb[-1] == len(doc), "token byte lengths do not cover the document"
    ok = np.ones(n + 1, dtype=bool)
    arr = np.frombuffer(doc, dtype=np.uint8)
    inner = cb[1:-1]
    ok[1:-1] = (arr[inner] & 0xC0) != 0x80
    ok[1:-1] &= lens[1:] > 0  # zero-length tokens never start a window
    cand = np.flatnonzero(ok)
    out, a = [], 0
    while a < n:
        b_max = min(int(np.searchsorted(cb, cb[a] + byte_seq_len - 1, side="right") - 1), a + token_seq_len - 1, n)
        j = int(np.searchsorted(cand, b_max, side="right") - 1)
        b = int(cand[j]) if j >= 0 else a
        if b <= a:  # no admissible cut: fall back to the token boundary (rare: a char spanning many tokens)
            b = max(b_max, a + 1)
        out.append((int(cb[a]), int(cb[b]), a, b))
        a = b
    return out


# ====================================================================================================== scoring
@dataclass
class Segment:
    ids: List[int]  # prefix (1 element) + scored elements
    n_bytes: int  # UTF-8 bytes of the scored window
    src: int
    doc: int


def build_segments(sources_docs: List[List[bytes]], kind: str, split_tok: Tokenizer,
                   model_tok: Optional[Tokenizer], byte_seq_len: int, token_seq_len: int):
    """-> (segments, per-source split-tokenizer token counts)."""
    segs, split_tokens = [], []
    same = model_tok is not None and model_tok.spec == split_tok.spec
    for si, docs in enumerate(sources_docs):
        ntok = 0
        for di, doc in enumerate(docs):
            ids, lens = split_tok.encode_with_lens(doc.decode("utf-8"))
            ntok += len(ids)
            prev_tok_ids = None
            for c, (ba, bb, ta, tb) in enumerate(make_windows(doc, ids, lens, byte_seq_len, token_seq_len)):
                if kind == "byte":
                    pre = BOS_ID if c == 0 else doc[ba - 1]
                    segs.append(Segment([pre] + list(doc[ba:bb]), bb - ba, si, di))
                    continue
                if same:
                    toks = list(ids[ta:tb])
                    pre = model_tok.eot if c == 0 else int(ids[ta - 1])
                else:
                    toks = model_tok.encode(doc[ba:bb].decode("utf-8"))
                    pre = model_tok.eot if c == 0 else int(prev_tok_ids[-1])
                    prev_tok_ids = toks
                segs.append(Segment([pre] + toks, bb - ba, si, di))
        split_tokens.append(ntok)
    return segs, split_tokens


@torch.no_grad()
def score_segments(model, segs: List[Segment], device, batch_tokens: int, amp_dtype=None):
    """Packed forward passes. Returns (bits [n_seg], positions [n_seg, n_stages + 1] or None)."""
    n = len(segs)
    bits = np.zeros(n)
    pos_counts = None
    order = sorted(range(n), key=lambda i: -len(segs[i].ids))
    batches, cur, cost = [], [], 0
    for i in order:
        L = len(segs[i].ids)
        if cur and cost + L > batch_tokens:
            batches.append(cur)
            cur, cost = [], 0
        cur.append(i)
        cost += L
    if cur:
        batches.append(cur)
    amp = (torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.autocast(
        device.type, enabled=False))
    for b in batches:
        lens = [len(segs[i].ids) for i in b]
        ids = torch.tensor([t for i in b for t in segs[i].ids], dtype=torch.long, device=device)
        cu = torch.zeros(len(b) + 1, dtype=torch.int32)
        cu[1:] = torch.cumsum(torch.tensor(lens), 0)
        with amp:
            out = model(ids, cu_seqlens=cu.to(device), max_seqlen=max(lens), return_logits=True)
        lg = out.logits.reshape(-1, out.logits.shape[-1])
        for k, i in enumerate(b):
            a, e = int(cu[k]), int(cu[k + 1])
            nll = F.cross_entropy(lg[a:e - 1].float(), ids[a + 1:e], reduction="sum")
            bits[i] = float(nll) / LN2
        bpred = getattr(out, "bpred", None) or []
        if bpred:
            if pos_counts is None:
                pos_counts = np.zeros((n, len(bpred) + 1), dtype=np.int64)
            T = ids.numel()
            seg_of = torch.searchsorted(cu[1:].long(), torch.arange(T), right=True).numpy()
            pos_counts[b, 0] = lens
            pos = np.arange(T)
            for s, r in enumerate(bpred):
                m = r.boundary_mask.reshape(-1).bool().cpu().numpy()
                pos = pos[m]
                pos_counts[b, s + 1] = np.bincount(seg_of[pos], minlength=len(b))
    return bits, pos_counts


def _boot_ratio(num: np.ndarray, den: np.ndarray, n_boot: int, seed: int = 0):
    if n_boot <= 0 or len(num) < 2:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def evaluate_sources(model, kind: str, sources: List[Source], device=None, model_tokenizer: str = "gpt2",
                     split_tokenizer: str = "gpt2", byte_seq_len: int = 8192, token_seq_len: int = 1792,
                     max_bytes: Optional[int] = 1_000_000, batch_tokens: Optional[int] = None,
                     amp_dtype=None, n_boot: int = 200, progress: bool = True) -> dict:
    """Core evaluator (usable on a live model). ``kind``: "byte" | "token"."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if batch_tokens is None:
        batch_tokens = (16384 if kind == "byte" else 8192) if device.type == "cuda" else 4096
    batch_tokens = max(batch_tokens, byte_seq_len if kind == "byte" else token_seq_len)
    if amp_dtype is None and device.type == "cuda":
        amp_dtype = torch.bfloat16
    split_tok = Tokenizer(split_tokenizer)
    model_tok = Tokenizer(model_tokenizer) if kind == "token" else None
    if model_tok is not None and model_tok.spec == split_tok.spec:
        model_tok = split_tok
    t0 = time.time()
    loaded = [load_source_docs(s, max_bytes) for s in sources]
    segs, split_tokens = build_segments([l["docs"] for l in loaded], kind, split_tok, model_tok, byte_seq_len,
                                        token_seq_len)
    was_training = model.training
    model.eval()
    try:
        bits, pos = score_segments(model, segs, device, batch_tokens, amp_dtype)
    finally:
        model.train(was_training)
    res = {"sources": {}, "groups": {}}
    src_idx = np.array([s.src for s in segs])
    doc_idx = np.array([s.doc for s in segs])
    nbytes = np.array([s.n_bytes for s in segs], dtype=np.int64)
    nunits = np.array([len(s.ids) - 1 for s in segs], dtype=np.int64)
    for si, (src, ld) in enumerate(zip(sources, loaded)):
        m = src_idx == si
        r = {**src.meta(), "n_docs": len(ld["docs"]), "n_bytes": int(nbytes[m].sum()), "n_segments": int(m.sum()),
             "skipped_invalid_utf8": ld["skipped_invalid_utf8"], "bits": float(bits[m].sum())}
        if not m.any():
            r["bpb"] = None
            res["sources"][src.name] = r
            continue
        r["bpb"] = r["bits"] / r["n_bytes"]
        nd = len(ld["docs"])
        db = np.bincount(doc_idx[m], weights=bits[m], minlength=nd)
        dn = np.bincount(doc_idx[m], weights=nbytes[m], minlength=nd)
        r["bpb_ci95"] = _boot_ratio(db, dn, n_boot, seed=si)
        r["split_bytes_per_token"] = r["n_bytes"] / max(split_tokens[si], 1)
        if kind == "token":
            r["n_tokens"] = int(nunits[m].sum())
            r["bytes_per_token"] = r["n_bytes"] / max(r["n_tokens"], 1)
        if pos is not None:
            p = pos[m].sum(0)
            r["positions_by_stage"] = p.tolist()
            r["bytes_per_position_by_stage"] = [float(p[0] / max(q, 1)) for q in p]
            r["bytes_per_main_position"] = float(p[0] / max(p[-1], 1))
            r["text_bytes_per_main_position"] = float(r["n_bytes"] / max(p[-1], 1))
        res["sources"][src.name] = r
    res["groups"] = group_stats(res["sources"])
    res["seconds"] = time.time() - t0
    res["settings"] = {"kind": kind, "model_tokenizer": model_tokenizer if kind == "token" else None,
                       "split_tokenizer": split_tokenizer, "byte_seq_len": byte_seq_len,
                       "token_seq_len": token_seq_len, "max_bytes": max_bytes, "batch_tokens": batch_tokens}
    return res


def group_stats(sources: dict) -> dict:
    """Pooled (byte-weighted) bpb per group; held-out sources form ``<group>_held_out``; ``all`` pools every
    in-distribution source; ``mix`` weights in-distribution groups by the manifest's source weights."""
    groups: dict = {}

    def add(g, name, r):
        d = groups.setdefault(g, {"sources": [], "bits": 0.0, "n_bytes": 0, "positions": [0, 0], "tokens": 0})
        d["sources"].append(name)
        d["bits"] += r["bits"]
        d["n_bytes"] += r["n_bytes"]
        if r.get("positions_by_stage"):
            d["positions"][0] += r["positions_by_stage"][0]
            d["positions"][1] += r["positions_by_stage"][-1]
        d["tokens"] += r.get("n_tokens", 0)

    for name, r in sources.items():
        if r.get("bpb") is None:
            continue
        g = r["group"] + ("_held_out" if r.get("held_out") else "")
        add(g, name, r)
        if not r.get("held_out"):
            add("all", name, r)
    for d in groups.values():
        d["bpb"] = d["bits"] / max(d["n_bytes"], 1)
        if d["positions"][1]:
            d["bytes_per_main_position"] = d["positions"][0] / d["positions"][1]
        if d["tokens"]:
            d["bytes_per_token"] = d["n_bytes"] / d["tokens"]
        del d["positions"], d["tokens"]
    w = {n: r["weight"] for n, r in sources.items() if r.get("weight") and r.get("bpb") is not None
         and not r.get("held_out")}
    if w:
        tot = sum(w.values())
        groups["mix"] = {"sources": sorted(w), "bpb": sum(sources[n]["bpb"] * v / tot for n, v in w.items()),
                         "weights": {n: v / tot for n, v in w.items()}}
    return groups


# ====================================================================================================== CLI
def run_seq_lens(meta: dict):
    ev = (meta.get("config") or {}).get("eval") or {}
    return ev.get("val_byte_seq_len"), ev.get("val_token_seq_len")


def model_tokenizer_of(meta: dict) -> str:
    cfg = meta.get("config") or {}
    for sec in ("data", "model"):
        t = (cfg.get(sec) or {}).get("tokenizer")
        if t:
            return t
    return "gpt2"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="run dir or checkpoint dir (trainer layout)")
    ap.add_argument("--sources", default=None, help="sources manifest (yaml/json)")
    ap.add_argument("--source", action="append", default=[], help="NAME=PATH[,PATH][@GROUP] (repeatable)")
    ap.add_argument("--out", default=None, help="default: <run>/per_source.json")
    ap.add_argument("--max-bytes", type=int, default=1_000_000, help="per-source text bytes (whole docs)")
    ap.add_argument("--byte-seq-len", type=int, default=None)
    ap.add_argument("--token-seq-len", type=int, default=None)
    ap.add_argument("--split-tokenizer", default="gpt2")
    ap.add_argument("--tokenizer", default=None, help="BPE model tokenizer (default: config or gpt2)")
    ap.add_argument("--batch-tokens", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-boot", type=int, default=200)
    a = ap.parse_args(argv)

    from hnet.eval.run import load_run_checkpoint, model_kind

    device = torch.device(a.device) if a.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, mcfg, meta = load_run_checkpoint(a.model, device=device)
    kind = model_kind(mcfg)
    sources = load_manifest(a.sources) if a.sources else []
    sources += [parse_source_arg(s) for s in a.source]
    if not sources:
        sources = [Source("finewebedu", [os.path.join(ROOT, DEFAULT_VAL)], group="english")]
    bsl, tsl = run_seq_lens(meta)
    bsl = a.byte_seq_len or bsl or 8192
    tsl = a.token_seq_len or tsl or 1792
    tok = a.tokenizer or model_tokenizer_of(meta)
    res = evaluate_sources(model, kind, sources, device=device, model_tokenizer=tok,
                           split_tokenizer=a.split_tokenizer, byte_seq_len=bsl, token_seq_len=tsl,
                           max_bytes=a.max_bytes, batch_tokens=a.batch_tokens, n_boot=a.n_boot)
    res["run"] = {"model": a.model, "checkpoint": meta.get("checkpoint"), "step": meta.get("step"), "kind": kind,
                  "bytes_seen": meta.get("bytes_seen")}
    ck = meta.get("checkpoint") or a.model
    run_dir = os.path.dirname(os.path.dirname(ck)) if os.path.basename(os.path.dirname(ck)) == "ckpt" else a.model
    out = a.out or os.path.join(run_dir, "per_source.json")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    w = max(len(n) for n in res["sources"])
    print(f"{'source':{w}s} {'group':14s} {'bytes':>9s} {'bpb':>7s} {'B/pos':>7s} {'B/tok':>7s}")
    for n, r in res["sources"].items():
        g = r["group"] + ("*" if r["held_out"] else "")
        bpp = r.get("bytes_per_main_position") or r.get("bytes_per_token") or float("nan")
        print(f"{n:{w}s} {g:14s} {r['n_bytes']:9d} {r['bpb'] if r['bpb'] is not None else float('nan'):7.4f} "
              f"{bpp:7.3f} {r.get('split_bytes_per_token', float('nan')):7.3f}")
    for g, r in res["groups"].items():
        print(f"[group] {g:20s} bpb {r['bpb']:.4f}")
    print(f"-> {out} ({res['seconds']:.1f}s)", file=sys.stderr)
    return res


if __name__ == "__main__":
    main()
