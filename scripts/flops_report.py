#!/usr/bin/env python
"""Print the params / FLOPs matching table for configs/models/*.yaml (see configs/MATCHING.md).

Each yaml has a ``model:`` section (hnet.config.model_config_from_yaml) and a ``matching:`` section
(seq_len_bytes, seq_len_tokens, bytes_per_token, reference). Re-run with measured numbers once the trainer has them:

    python scripts/flops_report.py                                  # target ratios, 4.6 bytes/token
    python scripts/flops_report.py --bytes-per-token 4.75           # measured GPT-2 bytes/token on our data
    python scripts/flops_report.py --bytes-per-position hnet2_small=1,3.4,8.1 --bytes-per-position 1,5.2
    python scripts/flops_report.py --rematch                        # main depth that would re-match the reference

``--bytes-per-position`` takes ``[name=]b0,b1,...`` (bytes per stage-s position, cumulative; last = BPIC); without
``name=`` it applies to every H-Net with that many stages.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from hnet.config import DenseConfig, HNetConfig, load_yaml, model_config_from_yaml  # noqa: E402
from hnet.flops import CONVENTIONS, model_flops, with_main_depth  # noqa: E402


def parse_bpp(items):
    named, by_len = {}, {}
    for it in items or []:
        name, _, vals = it.rpartition("=")
        v = [float(x) for x in vals.split(",")]
        if name:
            named[name] = v
        else:
            by_len[len(v)] = v
    return named, by_len


def flops_for(cfg, m, name, args, named, by_len, **extra):
    conv = CONVENTIONS[args.convention]
    if isinstance(cfg, DenseConfig):
        bpt = args.bytes_per_token or m.get("bytes_per_token", 4.6)
        return model_flops(cfg, seq_len=m.get("seq_len_tokens", 1792), bytes_per_token=bpt, convention=conv)
    bpp = named.get(name) or by_len.get(cfg.n_stages + 1)
    bpp = extra.get("bpp", bpp)
    return model_flops(cfg, bytes_per_position=bpp, seq_len=args.seq_len_bytes or m.get("seq_len_bytes", 8192),
                       convention=conv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="yaml files (default: configs/models/*.yaml)")
    ap.add_argument("--bytes-per-token", type=float, default=None, help="override bytes/token for BPE models")
    ap.add_argument("--bytes-per-position", action="append", default=[], metavar="[NAME=]B0,B1,..")
    ap.add_argument("--seq-len-bytes", type=int, default=None, help="override H-Net byte sequence length")
    ap.add_argument("--convention", default="spec", choices=sorted(CONVENTIONS))
    ap.add_argument("--rematch", action="store_true", help="also print the main depth that best re-matches")
    args = ap.parse_args(argv)

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    paths = args.paths or sorted(glob.glob(os.path.join(root, "configs", "models", "*.yaml")))
    named, by_len = parse_bpp(args.bytes_per_position)

    rows, ref_fpb = [], {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        cfg = model_config_from_yaml(p)
        m = load_yaml(p).get("matching", {}) or {}
        r = flops_for(cfg, m, name, args, named, by_len)
        rows.append((name, cfg, m, r))
        ref_fpb[name] = r["flops_per_byte"]

    hdr = ("| config | layout | params (M) | non-emb (M) | train GFLOPs/byte | bytes/main pos | per-stage GFLOPs/byte"
           " | vs ref |")
    print(hdr)
    print("|---|---|---:|---:|---:|---:|---|---:|")
    for name, cfg, m, r in rows:
        if isinstance(cfg, HNetConfig):
            layout = "/".join(_flat(cfg.arch_layout))
        else:
            layout = f"T{cfg.n_layers} d{cfg.d_model}"
        ref = m.get("reference")
        vs = ""
        if ref and ref in ref_fpb:
            vs = f"{100 * (r['flops_per_byte'] / ref_fpb[ref] - 1):+.1f}%"
        stages = ", ".join(f"{s['flops_per_byte'] / 1e9:.4f}" for s in r["per_stage"])
        print(f"| {name} | {layout} | {r['params'] / 1e6:.1f} | {r['params_nonembed'] / 1e6:.1f} | "
              f"{r['flops_per_byte'] / 1e9:.4f} | {r['bytes_per_position'][-1]:.2f} | {stages} | {vs} |")

    if args.rematch:
        print()
        for name, cfg, m, r in rows:
            ref = m.get("reference")
            if not isinstance(cfg, HNetConfig) or not ref or ref not in ref_fpb:
                continue
            best = None
            for n in range(1, 160):
                c = with_main_depth(cfg, n)
                f = flops_for(c, m, name, args, named, by_len)["flops_per_byte"]
                e = f / ref_fpb[ref] - 1
                if best is None or abs(e) < abs(best[1]):
                    best = (n, e)
            print(f"{name}: main T{best[0]} -> {100 * best[1]:+.1f}% vs {ref}")


def _flat(layout):
    out = []
    for x in layout:
        if isinstance(x, list):
            out.extend(_flat(x))
        else:
            out.append(x)
    return out


if __name__ == "__main__":
    main()
