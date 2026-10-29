#!/usr/bin/env python
"""Build a data mix (default mix_v1: 70% FineWeb-Edu / 20% code / 5% math / 5% multilingual, by bytes) as aligned
byte shards + in-corpus BPE token shards + per-source val sets. See docs/DATA_MIX.md and hnet/data/mix.py.

Examples (from the repo root):
  # local end-to-end test build (~50 MB of train text, a few hundred MB of HF reads, streamed)
  python scripts/data/build_mix.py --preset small
  # full build (~25 GB of train text) on a machine with fast internet: whole parquet files, deleted after use
  python scripts/data/build_mix.py --preset full --fetch-mode download --fetch-workers 12 --workers 8
  # (re)train the in-corpus tokenizer on the first 1 GB of the train stream, then build with it
  python scripts/data/build_mix.py --preset full --stages fetch,tokenizer,assemble --tokenizer-out configs/tokenizers/mix_v1_bpe32k.json
  # BPE-NP (no pretokenization, blog part II) tokenizer + a token-shard set built with it
  python scripts/data/build_mix.py --preset full --stages fetch,tokenizer,assemble --no-pretokenize \\
      --tokenizer-out configs/tokenizers/mix_v1_bpe32k_np.json --dataset mix_v1_np
  python scripts/data/build_mix.py --preset small --report              # print the manifest's bytes/token table
Re-running is safe: fetched units, finished shards and the tokenizer are reused; an interrupted run resumes.
Data root: --root, else $HNET_DATA, else <repo>/data. Pools (raw fetched docs, shared by all builds of a spec and
seed) live in <root>/mix_pools/; --delete-pools removes them after a successful assemble.
"""
import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from hnet.data import mix as M  # noqa: E402


def report(man: dict) -> str:
    lines = [f"{man['dataset']}: {man['train']['bytes'] / 1e6:,.1f} MB train text, {man['train']['tokens']:,} tokens, "
             f"{man['train']['docs']:,} docs, {man['train']['n_shards']} shards; bytes/token "
             f"{man['train']['bytes_per_token']:.3f} (stream {man['train']['stream_bytes_per_token']:.3f}); "
             f"tokenizer vocab {man['tokenizer']['vocab_size']} pretokenize={man['tokenizer']['pretokenize']}",
             f"{'leaf':24s} {'train MB':>10s} {'share':>7s} {'target':>7s} {'tok share':>9s} {'B/tok':>6s}"]
    for n, v in man["leaves"].items():
        t = v["train"]
        if v["heldout"]:
            continue
        lines.append(f"{n:24s} {t['bytes'] / 1e6:10.2f} {100 * t['byte_share']:6.2f}% {100 * t['target_share']:6.2f}%"
                     f" {100 * t['token_share']:8.2f}% {t['bytes_per_token'] or 0:6.3f}")
    for c, v in man["categories"].items():
        if v["bytes"]:
            lines.append(f"{'[' + c + ']':24s} {v['bytes'] / 1e6:10.2f} {100 * v['byte_share']:6.2f}% {'':7s} "
                         f"{100 * v['token_share']:8.2f}% {v['bytes_per_token']:6.3f}")
    lines.append(f"{'val set':24s} {'MB':>10s} {'docs':>7s} {'B/tok':>6s}")
    for n, v in man["val_sets"].items():
        lines.append(f"{n:24s} {v['bytes'] / 1e6:10.3f} {v['docs']:7d} {v['bytes_per_token']:6.3f}")
    for w in man.get("warnings", []):
        lines.append(f"WARNING {w}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="mix_v1", help="spec name or yaml path (configs/data/<name>.spec.yaml)")
    ap.add_argument("--preset", default="small", help="size preset in the spec (small | full)")
    ap.add_argument("--dataset", default=None, help="output dataset name (default: the preset's)")
    ap.add_argument("--root", default=None, help="data root (default $HNET_DATA or <repo>/data)")
    ap.add_argument("--seed", type=int, default=None, help="split/order seed (default: the spec's)")
    ap.add_argument("--stages", default="fetch,assemble", help="comma list of fetch, tokenizer, assemble")
    ap.add_argument("--train-bytes", type=float, default=None, help="override the preset's train text bytes")
    ap.add_argument("--shard-bytes", type=float, default=None)
    ap.add_argument("--val-leaf-bytes", type=float, default=None)
    ap.add_argument("--val-group-bytes", type=float, default=None)
    ap.add_argument("--val-mix-bytes", type=float, default=None)
    ap.add_argument("--sources", default=None, help="fetch only these sources (comma list)")
    ap.add_argument("--fetch-mode", default="stream", choices=["stream", "download"])
    ap.add_argument("--fetch-workers", type=int, default=6, help="sources fetched concurrently")
    ap.add_argument("--keep-downloads", action="store_true", help="download mode: keep parquet files")
    ap.add_argument("--tokenizer", default=None, help="tokenizer json for assemble (default: the spec's)")
    ap.add_argument("--tokenizer-out", default=None, help="tokenizer stage output (default: --tokenizer path)")
    ap.add_argument("--tokenizer-sample-bytes", type=float, default=None)
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--no-pretokenize", action="store_true", help="BPE-NP: no regex pretokenization")
    ap.add_argument("--workers", type=int, default=1, help="assemble processes (shards in parallel)")
    ap.add_argument("--threads", type=int, default=None, help="tokenizer threads per assemble process")
    ap.add_argument("--delete-pools", action="store_true", help="remove the fetched pools after assembling")
    ap.add_argument("--report", action="store_true", help="only print the manifest's summary")
    a = ap.parse_args()

    spec = M.load_spec(a.spec, seed=a.seed)
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    assert set(stages) <= {"fetch", "tokenizer", "assemble"}, stages
    cfg = M.preset(spec, a.preset, dataset=a.dataset, train_bytes=a.train_bytes, shard_bytes=a.shard_bytes,
                   val_leaf_bytes=a.val_leaf_bytes, val_group_bytes=a.val_group_bytes, val_mix_bytes=a.val_mix_bytes,
                   tokenizer_sample_bytes=a.tokenizer_sample_bytes)
    if "tokenizer" not in stages:
        cfg.tokenizer_sample_bytes = 0.0
    if a.report:
        print(report(M.load_manifest(cfg.dataset, a.root)))
        return
    tok_path = a.tokenizer or (str(M.REPO_ROOT / spec.tokenizer) if spec.tokenizer else None)
    t0 = time.time()
    if "fetch" in stages:
        from hnet.data.hf_sources import ParquetReader

        cache = os.path.join(a.root or str(M.data_root()), "hf_cache")

        def factory(src):
            return ParquetReader(src.repo, src.revision, mode=a.fetch_mode, cache_dir=cache, keep=a.keep_downloads)

        needs = M.leaf_needs(spec, cfg)
        print(f"[fetch] {spec.name} seed {spec.seed} -> {M.pool_dir(spec, a.root)}; quotas (MB train/val): "
              + ", ".join(f"{k}={v[0] / 1e6:.1f}/{v[1] / 1e6:.2f}" for k, v in needs.items()), flush=True)
        states = M.fetch(spec, cfg, a.root, factory, workers=a.fetch_workers,
                         sources=a.sources.split(",") if a.sources else None,
                         log=lambda s: print(s, flush=True))
        for name, st in states.items():
            print(f"[fetch] {name}: {st['units']} units, {st['rows']:,} rows, pool MB (train/val) "
                  + ", ".join(f"{k}={v[0] / 1e6:.1f}/{v[1] / 1e6:.2f}" for k, v in st["bytes"].items()), flush=True)
        print(f"[fetch] done in {time.time() - t0:.0f}s", flush=True)
    if "tokenizer" in stages:
        out = a.tokenizer_out or tok_path
        info = M.train_mix_tokenizer(spec, cfg, out, a.root, vocab_size=a.vocab_size,
                                     pretokenize=not a.no_pretokenize, min_frequency=a.min_frequency,
                                     log=lambda s: print(s, flush=True))
        print(json.dumps(info))
        tok_path = out
    if "assemble" in stages:
        if not tok_path or not os.path.exists(tok_path):
            sys.exit(f"tokenizer {tok_path} not found: add the tokenizer stage (--stages fetch,tokenizer,assemble)")
        man = M.assemble(spec, cfg, tok_path, a.root, workers=a.workers, threads=a.threads,
                         log=lambda s: print(s, flush=True))
        print(report(man))
        if a.delete_pools:
            shutil.rmtree(M.pool_dir(spec, a.root), ignore_errors=True)
            print(f"[assemble] removed {M.pool_dir(spec, a.root)}")
    print(f"[build_mix] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
