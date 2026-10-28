"""Forward+backward throughput and peak memory of a few model configs.

    python scripts/quick/bench_model.py                     # CPU, small configs
    python scripts/quick/bench_model.py --device cuda --seq-len 8192 --batch 4 --amp bf16

Reports bytes (or tokens) per second over ``--steps`` timed steps after ``--warmup`` steps, and peak memory
(CUDA: max allocated; CPU: process max RSS, a coarse upper bound).
"""

from __future__ import annotations

import argparse
import contextlib
import resource
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from hnet.config import DenseConfig, HNetConfig, model_config_from_yaml  # noqa: E402
from hnet.model import build_model  # noqa: E402


def configs(args):
    tiny = model_config_from_yaml(str(ROOT / "configs/tiny_cpu.yaml"))
    end = model_config_from_yaml(str(ROOT / "configs/tiny_cpu.yaml"))
    end.dc.router = "end_mlp"
    small = HNetConfig.from_dict(dict(
        arch_layout=["m4", ["T1m4", ["T8"], "m4T1"], "m4"],
        d_model=[256, 384, 512],
        d_intermediate=[0, 1024, 1408],
        ssm=dict(chunk_size=256, headdim=64, d_state=64),
        attn=dict(num_heads=[4, 6, 8], rotary_emb_dim=[32, 32, 32], window_size=[1023, 1023, -1]),
        dc=dict(target_ratios=[3.0, 3.0]),
    ))
    dense = DenseConfig(vocab_size=50304, d_model=512, n_layers=8, d_intermediate=1408, num_heads=8)
    out = {"hnet_tiny_cpu": tiny, "hnet_tiny_cpu_end_mlp": end, "hnet_2stage_small": small, "dense_llama_small": dense}
    return {k: v for k, v in out.items() if not args.only or k in args.only}


def bench(name, cfg, args):
    device = torch.device(args.device)
    torch.manual_seed(0)
    model = build_model(cfg).to(device)
    model.init_weights()
    model.train()
    is_dense = isinstance(cfg, DenseConfig)
    L = args.seq_len // 4 if is_dense else args.seq_len  # ~4.5 bytes per BPE token: compare at ~equal bytes
    V = cfg.vocab_size
    amp = (
        torch.autocast(device.type, dtype=torch.bfloat16) if args.amp == "bf16" else contextlib.nullcontext()
    )

    def step():
        ids = torch.randint(0, min(V, 256) if not is_dense else V, (args.batch * L,), device=device)
        cu = torch.arange(args.batch + 1, device=device, dtype=torch.int32) * L
        with amp:
            out = model(ids, cu_seqlens=cu, max_seqlen=L, labels=ids, return_logits=False)
        (out.loss + out.aux_loss).backward()
        model.zero_grad(set_to_none=True)

    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        mem = torch.cuda.max_memory_allocated() / 2**30
    else:
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    dt = (time.perf_counter() - t0) / args.steps
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    unit = "tok" if is_dense else "B"
    print(f"{name:24s} params {n_params:7.2f}M  {args.batch}x{L:<6d} {unit:3s} fwd+bwd {dt * 1e3:8.1f} ms/step "
          f"{args.batch * L / dt:10.0f} {unit}/s  peak mem {mem:6.2f} GB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--amp", choices=["none", "bf16"], default="none")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--only", nargs="*", help="config names to run")
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    for name, cfg in configs(args).items():
        bench(name, cfg, args)


if __name__ == "__main__":
    main()
