"""Download modded-nanogpt GPT-2 token shards from HF and convert them to aligned byte shards.

Layout (``data_root`` defaults to ``$HNET_DATA`` or ``<repo>/data``)::

    <data_root>/<dataset>/<prefix>_{val,train}_XXXXXX.bin          token shards (as on HF)
    <data_root>/<dataset>/bytes/<prefix>_{val,train}_XXXXXX.bytes.bin   byte shards
    <data_root>/<dataset>/bytes/<prefix>_{val,train}_XXXXXX.idx.npy     doc index
    <data_root>/<dataset>/bytes/convert_report.json                   conversion + verification stats

Stream order (the order modded-nanogpt generated shards): val_000000, train_000001, train_000002, ...
Documents are split across consecutive shards, so the "last doc complete?" flag of shard k is read from the
first token of shard k+1 (locally if present, else via an 8-byte HTTP range request).
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hnet.data import shards as S

DATASETS = {
    # name: (HF repo, file prefix, number of train shards; each train shard is 100M tokens)
    "finewebedu10B": ("kjj0/finewebedu10B-gpt2", "finewebedu", 99),
    "fineweb10B": ("kjj0/fineweb10B-gpt2", "fineweb", 103),
    "fineweb100B": ("kjj0/fineweb100B-gpt2", "fineweb", 1030),
}


def data_root() -> Path:
    env = os.environ.get("HNET_DATA")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data"


def shard_name(dataset: str, split: str, i: int) -> str:
    prefix = DATASETS[dataset][1]
    return f"{prefix}_{split}_{i:06d}.bin"


def stream_order(dataset: str, n_train: int) -> list[str]:
    """Token-shard filenames in generation order (val first)."""
    return [shard_name(dataset, "val", 0)] + [shard_name(dataset, "train", i) for i in range(1, n_train + 1)]


def dataset_dir(dataset: str, root=None) -> Path:
    return Path(root or data_root()) / dataset


def bytes_dir(dataset: str, root=None) -> Path:
    return dataset_dir(dataset, root) / "bytes"


def token_paths(dataset: str, split: str, root=None, n=None) -> list[Path]:
    """Existing local token shards of a split, sorted (train: optionally the first n)."""
    d = dataset_dir(dataset, root)
    paths = sorted(d.glob(f"{DATASETS[dataset][1]}_{split}_*.bin"))
    return paths[:n] if n is not None else paths


def byte_shard_paths(dataset: str, split: str, root=None, n=None) -> list[Path]:
    d = bytes_dir(dataset, root)
    paths = sorted(d.glob(f"{DATASETS[dataset][1]}_{split}_*.bytes.bin"))
    return paths[:n] if n is not None else paths


def _shard_ok(path: Path) -> bool:
    try:
        S.token_shard_ntok(path)
        return True
    except (AssertionError, OSError):
        return False


def download_shard(dataset: str, fname: str, root=None, retries: int = 5) -> Path:
    """Resume-safe download (huggingface_hub resumes .incomplete files); verifies size against the header."""
    from huggingface_hub import hf_hub_download

    repo = DATASETS[dataset][0]
    local_dir = dataset_dir(dataset, root)
    path = local_dir / fname
    if path.exists() and _shard_ok(path):
        return path
    last = None
    for attempt in range(retries):
        try:
            hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=str(local_dir),
                            force_download=path.exists())  # a present-but-bad file is re-fetched
            if _shard_ok(path):
                return path
            last = RuntimeError(f"{path} failed header/size check")
        except Exception as e:  # network hiccup: retry with backoff
            last = e
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"download of {repo}/{fname} failed: {last}")


def download(dataset: str, n_train: int, root=None, workers: int = 8) -> list[Path]:
    """Download the val shard and the first n_train train shards (parallel threads)."""
    from concurrent.futures import ThreadPoolExecutor

    names = stream_order(dataset, n_train)
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda f: download_shard(dataset, f, root), names))


def first_token(dataset: str, fname: str, root=None) -> int | None:
    """First token of a shard: local file if present, else an HTTP range request. None if unavailable."""
    path = dataset_dir(dataset, root) / fname
    if path.exists() and _shard_ok(path):
        t = S.read_token_shard(path)
        return int(t[0]) if len(t) else None
    import requests
    from huggingface_hub import hf_hub_url

    url = hf_hub_url(DATASETS[dataset][0], fname, repo_type="dataset")
    r = requests.get(url, headers={"Range": f"bytes={S.HEADER_BYTES}-{S.HEADER_BYTES + 1}"}, timeout=30)
    r.raise_for_status()
    if len(r.content) < 2:
        raise RuntimeError(f"range request for the first token of {fname} returned {len(r.content)} bytes")
    return int(np.frombuffer(r.content[:2], dtype=np.uint16)[0])


# ---------------------------------------------------------------- conversion
def _verify_sample(token_path, byte_path, idx_path, n_sample: int, seed: int) -> dict:
    """tiktoken decode_bytes(doc tokens) must equal our bytes (hard); re-encode match rate is informational."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    toks = S.read_token_shard(token_path)
    data = S.read_byte_shard(byte_path)
    idx = S.read_doc_index(idx_path)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(idx), size=min(n_sample, len(idx)), replace=False) if len(idx) else []
    # always include the first and last rows (fragments) in the decode check
    sel = sorted(set(int(i) for i in sel) | ({0, len(idx) - 1} if len(idx) else set()))
    n_dec_ok = n_reenc_ok = n_reenc = 0
    for i in sel:
        tt = S.doc_text_tokens(toks, idx[i]).astype(np.int64).tolist()
        bb = bytes(S.doc_text_bytes(data, idx[i]))
        n_dec_ok += enc.decode_bytes(tt) == bb
        if idx[i, S.FLAGS] == S.HAS_BOS | S.HAS_EOS:  # re-encode only whole documents
            n_reenc += 1
            try:
                n_reenc_ok += enc.encode_ordinary(bb.decode("utf-8")) == tt
            except UnicodeDecodeError:
                pass
    return dict(n_checked=len(sel), decode_exact=n_dec_ok, n_reencode=n_reenc, reencode_match=n_reenc_ok)


def _convert_job(args):
    token_path, out_dir, last_complete, verify_n, seed = args
    t0 = time.time()
    st = S.convert_shard(token_path, out_dir, last_complete=last_complete)
    st["convert_s"] = time.time() - t0
    if verify_n:
        bpath, ipath = S.byte_paths(out_dir, st["shard"])
        st["verify"] = _verify_sample(token_path, bpath, ipath, verify_n, seed)
    return st


def convert_dataset(dataset: str, root=None, workers: int | None = None, verify_n: int = 64, overwrite=False,
                    n_train: int | None = None, only: list[str] | None = None) -> dict:
    """Convert every local token shard (in stream order) to byte shards with a process pool."""
    d = dataset_dir(dataset, root)
    out_dir = bytes_dir(dataset, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    present = sorted(p.name for p in d.glob(f"{DATASETS[dataset][1]}_*.bin"))
    n_all = DATASETS[dataset][2]
    order = stream_order(dataset, n_all)
    jobs = []
    for k, fname in enumerate(order):
        if fname not in present or (only and fname not in only):
            continue
        if n_train is not None and k > n_train:
            continue
        bpath, ipath = S.byte_paths(out_dir, fname[:-4])
        if not overwrite and bpath.exists() and ipath.exists():
            continue
        if k + 1 < len(order):
            ft = first_token(dataset, order[k + 1], root)
            last_complete = True if ft is None else ft == S.EOT_ID
        else:
            last_complete = True  # final shard of the dataset
        jobs.append((str(d / fname), str(out_dir), last_complete, verify_n, 1234 + k))
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    t0 = time.time()
    results = []
    if jobs:
        with get_context("spawn").Pool(workers) as pool:  # spawn: no inherited memmaps/threads
            for st in pool.imap_unordered(_convert_job, jobs):
                results.append(st)
                v = st.get("verify", {})
                print(f"[convert] {st['shard']}: {st['ntok']:,} tok -> {st['nbytes_stream']:,} B "
                      f"({st['convert_s']:.1f}s) decode_exact={v.get('decode_exact')}/{v.get('n_checked')} "
                      f"reencode={v.get('reencode_match')}/{v.get('n_reencode')}", flush=True)
    wall = time.time() - t0
    ntok = sum(r["ntok"] for r in results)
    rep = dict(dataset=dataset, n_shards=len(results), ntok=ntok, wall_s=wall,
               tok_per_s=ntok / wall if wall > 0 else None, workers=workers,
               shards=sorted(results, key=lambda r: r["shard"]))
    bad = [r["shard"] for r in results if r.get("verify") and r["verify"]["decode_exact"] != r["verify"]["n_checked"]]
    rep["decode_failures"] = bad
    rpath = out_dir / "convert_report.json"
    old = json.loads(rpath.read_text()) if rpath.exists() else {}
    old.setdefault("runs", []).append({k: v for k, v in rep.items() if k != "shards"})
    shards_old = {s["shard"]: s for s in old.get("shards", [])}
    shards_old.update({s["shard"]: s for s in results})
    old["shards"] = sorted(shards_old.values(), key=lambda s: s["shard"])
    rpath.write_text(json.dumps(old, indent=1))
    if bad:
        raise RuntimeError(f"byte decode mismatch vs tiktoken in shards {bad}")
    return rep
