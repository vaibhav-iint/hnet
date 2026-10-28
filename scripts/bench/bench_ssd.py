"""Benchmark forward+backward of the Mamba2 mixer (pure-torch SSD, and the mamba_ssm kernels on CUDA).

    OMP_NUM_THREADS=4 python scripts/bench/bench_ssd.py --device cpu --small
    python scripts/bench/bench_ssd.py --device cuda            # both paths: kernels and HNET torch fallback

Prints tokens/s (forward+backward) and peak memory per shape. CPU peak memory is the process peak RSS above the
pre-run level (Linux /proc, reset per measurement); CUDA uses torch.cuda.max_memory_allocated.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from hnet.model import ssd  # noqa: E402
from hnet.model.ctx import SeqCtx  # noqa: E402
from hnet.model.mixers.mamba2 import Mamba2  # noqa: E402

FULL = dict(d_model=[256, 768], seqlen=[2048, 8192], d_state=[64, 128])
SMALL = dict(d_model=[256], seqlen=[512, 2048], d_state=[64])


def _rss_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field):
                return int(line.split()[1])
    return 0


def _reset_peak(device):
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated()
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")  # reset VmHWM (peak RSS) to the current RSS
    except OSError:
        pass
    return _rss_kb("VmRSS:") * 1024


def _peak(device, base) -> float:
    if device == "cuda":
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 2**20
    return (_rss_kb("VmHWM:") * 1024 - base) / 2**20


def bench(d_model, seqlen, d_state, device, dtype, chunk_size, n_seg, iters, warmup):
    torch.manual_seed(0)
    m = Mamba2(d_model, d_state=d_state, chunk_size=chunk_size, device=device)
    x = torch.randn(1, seqlen, d_model, device=device, requires_grad=True)
    cu = torch.linspace(0, seqlen, n_seg + 1, device=device).round().to(torch.int32)
    ctx = SeqCtx.from_cu_seqlens(cu) if n_seg > 1 else None
    autocast = torch.autocast(device_type=device, dtype=dtype, enabled=dtype != torch.float32)

    def step():
        with autocast:
            y = m(x, ctx)
        y.float().square().mean().backward()

    for _ in range(warmup):
        step()
    base = _reset_peak(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return seqlen / dt, dt * 1e3, _peak(device, base)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--small", action="store_true", help="small CPU-friendly shapes")
    ap.add_argument("--dtype", default=None, choices=["fp32", "bf16"], help="default: bf16 on cuda, fp32 on cpu")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--segments", type=int, default=4, help="documents packed in the sequence (seq_idx)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype or ("bf16" if args.device == "cuda" else "fp32")]
    shapes = SMALL if args.small else FULL
    paths = ["kernel", "torch"] if args.device == "cuda" else ["torch"]
    print(f"device={args.device} dtype={dtype} chunk={args.chunk_size} segments={args.segments} "
          f"threads={torch.get_num_threads()}")
    print(f"{'path':>6} {'d_model':>7} {'T':>6} {'d_state':>7} {'ms/iter':>9} {'tok/s':>10} {'peak MiB':>9}")
    for path in paths:
        ssd.FORCE_TORCH = path == "torch"
        for d_model, seqlen, d_state in itertools.product(shapes["d_model"], shapes["seqlen"], shapes["d_state"]):
            tps, ms, mem = bench(d_model, seqlen, d_state, args.device, dtype, args.chunk_size, args.segments,
                                 args.iters, args.warmup)
            print(f"{path:>6} {d_model:>7} {seqlen:>6} {d_state:>7} {ms:>9.1f} {tps:>10.0f} {mem:>9.0f}")


if __name__ == "__main__":
    main()
