#!/usr/bin/env python
"""Download modded-nanogpt GPT-2 shards and convert them to aligned byte shards.

Examples (GPU box, from the repo root):
  python scripts/data/prepare.py --dataset finewebedu10B --n-train 99          # full FineWeb-Edu 10B
  python scripts/data/prepare.py --dataset fineweb10B --n-train 103            # speedrun data
  python scripts/data/prepare.py --dataset finewebedu10B --n-train 0           # val shard only
  HNET_DATA=/mnt/data python scripts/data/prepare.py ...                        # other data root
Re-running is safe: present+valid shards are skipped, interrupted downloads resume, converted shards are skipped.
Afterwards run scripts/data/measure_bpt.py to (re)write bytes/token stats.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hnet.data import download as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="finewebedu10B", choices=sorted(D.DATASETS))
    ap.add_argument("--n-train", type=int, default=None, help="train shards (100M tokens each); default all")
    ap.add_argument("--root", default=None, help="data root (default $HNET_DATA or <repo>/data)")
    ap.add_argument("--download-workers", type=int, default=8)
    ap.add_argument("--workers", type=int, default=None, help="conversion processes (default ncpu-1)")
    ap.add_argument("--verify", type=int, default=64, help="docs per shard checked against tiktoken")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="re-convert existing byte shards")
    a = ap.parse_args()
    n_train = D.DATASETS[a.dataset][2] if a.n_train is None else a.n_train
    if not a.no_download:
        t0 = time.time()
        paths = D.download(a.dataset, n_train, a.root, workers=a.download_workers)
        gb = sum(os.path.getsize(p) for p in paths) / 1e9
        print(f"[download] {len(paths)} shards, {gb:.2f} GB in {time.time() - t0:.0f}s -> {D.dataset_dir(a.dataset, a.root)}")
    if not a.no_convert:
        rep = D.convert_dataset(a.dataset, a.root, workers=a.workers, verify_n=a.verify, overwrite=a.overwrite,
                                n_train=n_train)
        if rep["n_shards"]:
            print(f"[convert] {rep['n_shards']} shards, {rep['ntok']:,} tokens in {rep['wall_s']:.1f}s "
                  f"= {rep['tok_per_s'] / 1e6:.2f}M tok/s with {rep['workers']} workers")
        else:
            print("[convert] nothing to do")


if __name__ == "__main__":
    main()
