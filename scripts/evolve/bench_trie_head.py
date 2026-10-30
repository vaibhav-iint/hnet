"""CPU wall-clock of the trie output head alone (forward + backward to tables, query projection and query source):
the full head (every byte scored) against the top-k head, on the same real mix_v1 bytes and boundaries.

    OMP_NUM_THREADS=2 PYTHONPATH=. python scripts/evolve/bench_trie_head.py --bytes 8512 --ks 16 32 64

Projection to the A100: the g1 runs measured hnet_ws at 1.51 s/step and g1-mem-01 (full head) at 3.26 s/step, i.e.
1.75 s/step of head overhead; a head that costs a fraction x of the full head's time projects to 1.51 + 1.75 x.
"""

from __future__ import annotations

import argparse
import statistics
import time

import numpy as np
import torch

from hnet.model.ctx import SeqCtx
from hnet.model.trie_head import TrieHead

VAL = "data/mix_v1/bytes/mix_v1_val_000000.bytes.bin"
GPU_BASE, GPU_FULL = 1.51, 3.26  # s/step on an A100, hnet_ws and g1-mem-01 (logbook g1 runs, step_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bytes", type=int, default=8512)
    ap.add_argument("--ks", type=int, nargs="+", default=[16, 32, 64])
    ap.add_argument("--rounds", type=int, default=7)
    args = ap.parse_args()
    raw = np.fromfile(VAL, dtype=np.uint8)
    T = args.bytes
    ids = torch.from_numpy(raw[1_000_000:1_000_000 + T].astype(np.int64))
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, T], dtype=torch.int32))
    mask = (ids == 0x20) | (ids == 0x0A)  # word-like chunk ends (end semantics), as in a warm-started model
    mask[0] = True
    torch.manual_seed(0)
    q = torch.randn(T, 192, requires_grad=True)
    base = torch.randn(T, 256)
    heads = {"full": TrieHead(192, 256, rows=77383, dim=32, n_hash=2, max_len=16)}
    for k in args.ks:
        heads[f"top{k}"] = TrieHead(192, 256, rows=77383, dim=32, n_hash=2, max_len=16, topk=k)
    for h in heads.values():
        for t in h.tables:
            torch.nn.init.normal_(t.weight, std=0.02)

    def run(name, h):
        if h.topk:
            cand, vals, _ = h.rescore(q, ids, ctx, mask, base)
            logits = base.scatter_add(1, cand, vals)
        else:
            logits = base + h(q, ids, ctx, mask)[0]
        loss = torch.nn.functional.cross_entropy(logits, ids)
        loss.backward()

    times = {n: [] for n in heads}
    for n, h in heads.items():
        run(n, h)
    for _ in range(args.rounds):
        for n, h in heads.items():
            t0 = time.perf_counter()
            run(n, h)
            times[n].append(time.perf_counter() - t0)
    full = statistics.median(times["full"])
    print(f"trie head alone, {T} bytes, torch threads {torch.get_num_threads()}")
    for n in heads:
        med = statistics.median(times[n])
        x = med / full
        print(f"{n:6s} {med * 1e3:8.1f} ms  x{x:.3f} of full  -> projected A100 step {GPU_BASE + (GPU_FULL - GPU_BASE) * x:.2f}"
              f" s = x{(GPU_BASE + (GPU_FULL - GPU_BASE) * x) / GPU_BASE:.3f} of hnet_ws")


if __name__ == "__main__":
    main()
