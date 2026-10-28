"""Print the bits-per-byte equivalents of the speedrun targets (3.28 nats/token: GPT-2 small track; 2.92: medium).

    python scripts/speedrun/target.py [--shard data/fineweb10B/fineweb_val_000000.bin] [--json]

modded-nanogpt's val loss is the mean CE over the first 10,485,760 targets of the FineWeb val shard, including
the ~1 per document <|endoftext|> targets (0 text bytes). Following ``hnet.data.valsets.speedrun_target`` (WS-C),
``target_bpb = loss * n_targets / ln 2 / n_text_bytes(targets)``: the same total nats spread over the same text.
Our bpb excludes doc-end predictions, so the line "if EOT targets cost nothing" brackets the ambiguity (the
true EOT loss is somewhere in between, and the gap is ~0.15%).

Downloads the val shard (~200 MB) with WS-C's downloader if it is missing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hnet.data import download, speedrun_target  # noqa: E402

TRACKS = {3.28: "GPT-2 small track (track 1)", 2.92: "GPT-2 medium track (track 2)"}


def val_shard_path(path: str | None) -> str:
    if path:
        return path
    name = download.shard_name("fineweb10B", "val", 0)
    default = download.dataset_dir("fineweb10B") / name
    if not default.exists():
        download.download_shard("fineweb10B", name)
    return str(default)


def compute(shard: str, losses=(3.28, 2.92)) -> dict:
    res = speedrun_target(shard, losses=losses)
    n, n_eot, nb = res["n_targets"], res["n_eot_targets"], res["n_bytes"]
    res["target_bpb_eot_free"] = {str(l): l * (n - n_eot) / math.log(2) / nb for l in losses}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shard", help="FineWeb val token shard (default: data/fineweb10B/fineweb_val_000000.bin)")
    ap.add_argument("--losses", type=float, nargs="+", default=[3.28, 2.92])
    ap.add_argument("--json", action="store_true", help="print the raw result as json")
    args = ap.parse_args(argv)
    res = compute(val_shard_path(args.shard), tuple(args.losses))
    if args.json:
        print(json.dumps(res, indent=1))
        return
    print(f"targets {res['n_targets']:,} (EOT {res['n_eot_targets']:,}), text bytes {res['n_bytes']:,}, "
          f"bytes/target {res['bytes_per_target']:.4f}")
    for l in args.losses:
        print(f"{l:.2f} nats/token -> {res['target_bpb'][str(l)]:.4f} bpb "
              f"(if EOT targets cost nothing: {res['target_bpb_eot_free'][str(l)]:.4f})  {TRACKS.get(l, '')}")


if __name__ == "__main__":
    main()
