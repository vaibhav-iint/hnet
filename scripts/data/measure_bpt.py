#!/usr/bin/env python
"""Measure UTF-8 bytes per GPT-2 token for a dataset (overall + per-document distribution).

  python scripts/data/measure_bpt.py --dataset finewebedu10B [--split val|train|all] [--max-shards N]

Uses the byte-shard doc indices when present (exact, instant), else the token shards (exact via the GPT-2 byte
table). Writes <data_root>/stats/<dataset>_<split>_bpt.json and merges the headline numbers into
hnet/data/bpt_stats.json (committed; read by hnet.data.loader.bytes_per_token).

Definitions: bytes_per_token = text bytes / text tokens (<|endoftext|> excluded on both sides);
stream_bytes_per_token = byte-stream positions / token-stream positions = (bytes + BOS + EOS) / (tokens + EOT),
the ratio that equalizes documents per step between a byte window and a token window.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hnet.data import download as D  # noqa: E402
from hnet.data import shards as S  # noqa: E402


def shard_doc_stats(token_path: Path, idx_path: Path | None):
    if idx_path is not None and idx_path.exists():
        idx = np.asarray(S.read_doc_index(idx_path))
        stream_bytes = S.byte_shard_header(str(idx_path).replace(".idx.npy", ".bytes.bin"))["nbytes"]
        stream_tokens = S.token_shard_ntok(token_path)
        return idx, stream_bytes, stream_tokens
    toks = S.read_token_shard(token_path)
    _, idx = S.convert_token_array(toks, last_complete=True)  # index only; stream count below
    stream_bytes = int(idx[:, S.N_BYTES].sum() + np.count_nonzero(idx[:, S.FLAGS] & S.HAS_BOS)
                       + np.count_nonzero(idx[:, S.FLAGS] & S.HAS_EOS))
    return idx, stream_bytes, len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="finewebedu10B", choices=sorted(D.DATASETS))
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--no-commit", action="store_true", help="do not update hnet/data/bpt_stats.json")
    a = ap.parse_args()
    splits = ["val", "train"] if a.split == "all" else [a.split]
    tpaths = [p for s in splits for p in D.token_paths(a.dataset, s, a.root)][: a.max_shards]
    assert tpaths, f"no {a.split} token shards for {a.dataset} under {D.dataset_dir(a.dataset, a.root)}"
    rows, per_shard = [], []
    sb = st = 0
    for tp in tpaths:
        ip = D.bytes_dir(a.dataset, a.root) / (tp.name[:-4] + ".idx.npy")
        idx, b, t = shard_doc_stats(tp, ip)
        sb += b
        st += t
        rows.append(idx)
        per_shard.append(dict(shard=tp.name, bytes_per_token=float(idx[:, S.N_BYTES].sum() / idx[:, S.N_TOKENS].sum()),
                              stream_bytes_per_token=b / t, n_docs=int(len(idx))))
        print(f"{tp.name}: {per_shard[-1]['bytes_per_token']:.4f} B/tok", flush=True)
    idx = np.concatenate(rows)
    nb, nt = idx[:, S.N_BYTES].astype(np.float64), idx[:, S.N_TOKENS].astype(np.float64)
    whole = (idx[:, S.FLAGS] == (S.HAS_BOS | S.HAS_EOS)) & (nt > 0)
    doc_bpt = nb[whole] / nt[whole]
    q = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct = lambda x: {f"p{k}": float(v) for k, v in zip(q, np.percentile(x, q))}  # noqa: E731
    out = dict(
        dataset=a.dataset, split=a.split, n_shards=len(tpaths), n_docs=int(len(idx)), n_whole_docs=int(whole.sum()),
        text_bytes=int(nb.sum()), text_tokens=int(nt.sum()), stream_bytes=int(sb), stream_tokens=int(st),
        bytes_per_token=float(nb.sum() / nt.sum()), stream_bytes_per_token=float(sb / st),
        doc_bytes_per_token=dict(mean=float(doc_bpt.mean()), std=float(doc_bpt.std()), **pct(doc_bpt)),
        doc_tokens=dict(mean=float(nt[whole].mean()), **pct(nt[whole])),
        doc_bytes=dict(mean=float(nb[whole].mean()), **pct(nb[whole])),
        per_shard=per_shard,
    )
    sdir = Path(a.root or D.data_root()) / "stats"
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / f"{a.dataset}_{a.split}_bpt.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_shard"}, indent=1))
    print(f"wrote {path}")
    if not a.no_commit:
        cpath = Path(__file__).resolve().parents[2] / "hnet" / "data" / "bpt_stats.json"
        allst = json.loads(cpath.read_text()) if cpath.exists() else {}
        cur = allst.get(a.dataset, {})
        # train measurements take precedence over val-only ones
        if a.split != "val" or cur.get("split", "val") == "val":
            allst[a.dataset] = {k: out[k] for k in ("split", "n_shards", "text_bytes", "text_tokens",
                                                     "bytes_per_token", "stream_bytes_per_token")}
            allst[a.dataset]["doc_bytes_per_token_p50"] = out["doc_bytes_per_token"]["p50"]
            cpath.write_text(json.dumps(allst, indent=1) + "\n")
            print(f"updated {cpath}")


if __name__ == "__main__":
    main()
