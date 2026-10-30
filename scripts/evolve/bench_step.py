"""CPU wall-clock step time of several model configs on the same batch (forward + backward + AdamW step).

    OMP_NUM_THREADS=2 PYTHONPATH=. python scripts/evolve/bench_step.py configs/ladder/models/hnet_ws_L03.yaml \\
        configs/evolve/models/g1-mem-01.yaml configs/evolve/models/g3-mem-01.yaml --bytes 8512 --rounds 5

The arms are timed interleaved (round r runs one step of every arm) so a shared machine's load drifts affect them
alike; the report is the median per arm and its ratio to the first arm. The batch is real mix_v1 validation text.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model

VAL = "data/mix_v1/bytes/mix_v1_val_000000.bytes.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--bytes", type=int, default=8512)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--offset", type=int, default=1_000_000)
    args = ap.parse_args()
    raw = np.fromfile(VAL, dtype=np.uint8)
    n = args.batch * args.bytes
    ids = torch.from_numpy(raw[args.offset:args.offset + n].astype(np.int64)).view(args.batch, args.bytes)
    arms = []
    for path in args.configs:
        torch.manual_seed(0)
        cfg = model_config_from_yaml(path)
        m = build_model(cfg)
        m.init_weights()
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        arms.append((Path(path).stem, m, opt, []))

    def step(m, opt):
        out = m(ids, labels=ids, return_logits=False)
        opt.zero_grad(set_to_none=True)
        (out.loss + out.aux_loss).backward()
        opt.step()
        return out

    for _ in range(args.warmup):
        for _, m, opt, _ in arms:
            step(m, opt)
    for _ in range(args.rounds):
        for name, m, opt, times in arms:
            t0 = time.perf_counter()
            step(m, opt)
            times.append(time.perf_counter() - t0)
    ref = statistics.median(arms[0][3])
    print(f"batch {args.batch} x {args.bytes} bytes, {args.rounds} rounds, torch threads {torch.get_num_threads()}")
    for name, m, _, times in arms:
        med = statistics.median(times)
        params = sum(p.numel() for p in m.parameters())
        print(f"{name:28s} params {params:>11,d}  step {med:7.3f} s (min {min(times):.3f}, max {max(times):.3f})"
              f"  x{med / ref:.3f}")


if __name__ == "__main__":
    main()
