"""Training metrics: device-side accumulation, cross-rank reduction, MFU, jsonl/stdout/wandb sinks.

Metric keys written to ``<out>/metrics.jsonl`` (one flat dict per log step):

* ``step`` (completed optimizer steps), ``tokens_eq`` (BPE-token equivalents seen: tokens for token data,
  bytes / bytes_per_token for byte data, or the loader's exact ``n_tokens`` when given), ``bytes_seen``
  (UTF-8 text bytes, exact from the loader), ``lr`` (base LR x schedule; stage s trains at lr x lambda^s);
* ``loss`` = ce + aux_loss (the optimized objective), ``ce`` (mean CE per target, nats), ``aux_loss``,
  ``bpb`` (training bpb: byte data ce/ln2 over all targets incl. EOS; token data sum CE/ln2/text bytes);
  these are averaged over every micro-step and rank since the previous log step;
* ``grad_norm`` (pre-clip global norm), ``grad_norm/stage{s}``;
* model stats (``compression/s{s}``, ``F/s{s}``, ``G/s{s}``, ``aux/s{s}``, ...) averaged over the log step's
  micro-steps and ranks;
* ``flops_per_byte`` (training FLOPs/byte at realized compression, hnet.flops), ``mfu``, ``bytes_per_sec``,
  ``tokens_per_sec``, ``mem_gb`` (peak allocated since last log), ``step_time`` (s/step over the interval).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional

import torch

# bf16 dense peak FLOP/s per GPU. Checked in order (longest/most specific first).
PEAK_FLOPS = [
    ("H200", 989e12),
    ("H100", 989e12),
    ("GH200", 989e12),
    ("A100", 312e12),
    ("L40S", 362e12),
    ("RTX 4090", 165e12),
    ("4090", 165e12),
    ("RTX 3090", 71e12),
    ("3090", 71e12),
    ("A10", 125e12),
    ("L4", 121e12),
]


def peak_flops(device_name: Optional[str] = None) -> Optional[float]:
    """Peak bf16 FLOP/s for a CUDA device name; None on CPU / unknown GPUs (MFU is then skipped)."""
    if device_name is None:
        if not torch.cuda.is_available():
            return None
        device_name = torch.cuda.get_device_name()
    env = os.environ.get("HNET_PEAK_FLOPS")
    if env:
        return float(env)
    for key, v in PEAK_FLOPS:
        if key in device_name:
            if key == "A10" and "A100" in device_name:
                continue
            if key == "L4" and "L40" in device_name:
                continue
            return v
    return None


class StatAccumulator:
    """Sums tensors on device without host syncs; reduced and read only at log steps."""

    def __init__(self):
        self.sums: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}

    def add(self, key: str, value, n: int = 1):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().reshape(())
        if key in self.sums:
            self.sums[key] = self.sums[key] + value
            self.counts[key] += n
        else:
            self.sums[key] = value if isinstance(value, torch.Tensor) else torch.tensor(float(value))
            self.counts[key] = n

    def add_dict(self, d: dict):
        for k, v in (d or {}).items():
            if isinstance(v, torch.Tensor) and v.numel() != 1:
                continue
            if isinstance(v, (bool,)) or v is None:
                continue
            self.add(k, v)

    def reduce_means(self, device, world_size: int) -> Dict[str, float]:
        """Mean over everything added (all ranks); one all-reduce + one host sync."""
        if not self.sums:
            return {}
        keys = sorted(self.sums)
        vals = torch.stack([self.sums[k].to(device, torch.float32) for k in keys])
        cnts = torch.tensor([float(self.counts[k]) for k in keys], device=device)
        buf = torch.cat([vals, cnts])
        if world_size > 1:
            torch.distributed.all_reduce(buf)
        buf = buf.cpu().tolist()
        n = len(keys)
        return {k: buf[i] / max(buf[n + i], 1e-12) for i, k in enumerate(keys)}

    def clear(self):
        self.sums.clear()
        self.counts.clear()


def grad_norms_by_stage(stage_params: Dict[int, List[torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Per-stage gradient L2 norms with foreach kernels (device tensors; no sync)."""
    out = {}
    for s, ps in sorted(stage_params.items()):
        gs = [p.grad for p in ps if p.grad is not None]
        if not gs:
            continue
        norms = torch._foreach_norm(gs, 2.0)
        out[f"grad_norm/stage{s}"] = torch.linalg.vector_norm(torch.stack([n.float() for n in norms]), 2.0)
    return out


def bytes_per_position_from_stats(stats: Dict[str, float], n_stages: int) -> List[float]:
    """bytes_per_position[s] = prod_{i<s} compression/s{i} (stage 0 = 1 byte/position)."""
    bpp = [1.0]
    for s in range(n_stages):
        c = stats.get(f"compression/s{s}")
        if c is None or not math.isfinite(c) or c <= 0:
            return None
        bpp.append(bpp[-1] * c)
    return bpp


class FlopsEstimator:
    """Training FLOPs per byte at the realized compression, via hnet.flops.model_flops.

    Falls back to ``6 * params_per_stage / bytes_per_position`` (TODO(interface): only until hnet.flops exists).
    """

    def __init__(self, model_cfg, model, seq_len: int, kind: str):
        self.cfg, self.seq_len, self.kind = model_cfg, seq_len, kind
        self.n_stages = getattr(model_cfg, "n_stages", 0) if hasattr(model_cfg, "arch_layout") else 0
        self._mf = None
        from hnet.config import DenseConfig, HNetConfig

        if isinstance(model_cfg, (HNetConfig, DenseConfig)):
            from hnet.flops import model_flops

            self._mf = model_flops
        from hnet.optim.groups import classify_params

        self._stage_params = {}
        for d in classify_params(model).values():
            if d["embedding"]:  # lookups are not counted (App. B)
                continue
            s = d["stage"]
            self._stage_params[s] = self._stage_params.get(s, 0) + d["param"].numel()
        self._cache = {}

    def flops_per_byte(self, stats: Dict[str, float], bytes_per_token: Optional[float]) -> Optional[float]:
        if self.kind == "tokens":
            if not bytes_per_token:
                return None
            key = ("tok", round(bytes_per_token, 3))
            if key not in self._cache:
                self._cache[key] = self._compute(None, bytes_per_token)
            return self._cache[key]
        bpp = bytes_per_position_from_stats(stats, self.n_stages) if self.n_stages else [1.0]
        if bpp is None:
            return None
        return self._compute(bpp, None)

    def _compute(self, bpp, bpt):
        if self._mf is not None:
            r = self._mf(self.cfg, bytes_per_position=bpp, seq_len=self.seq_len, bytes_per_token=bpt, training=True)
            return float(r["flops_per_byte"])
        if bpt is not None:
            return 6.0 * sum(self._stage_params.values()) / bpt
        return sum(6.0 * n / bpp[min(s, len(bpp) - 1)] for s, n in self._stage_params.items())


def _truncate_after(path: str, step: int):
    with open(path) as f:
        recs = [json.loads(line) for line in f if line.strip()]
    with open(path, "w") as f:
        for r in recs:
            if r["step"] <= step:
                f.write(json.dumps(r) + "\n")


class MetricsLogger:
    """Rank-0 sink: stdout, ``metrics.jsonl`` / ``eval.jsonl``, optional wandb."""

    PRINT_KEYS = ["loss", "ce", "bpb", "aux_loss", "grad_norm", "lr", "tokens_eq", "bytes_per_sec", "mfu",
                  "step_time", "mem_gb"]

    def __init__(self, out_dir: str, enabled: bool, wandb_project: Optional[str] = None, run_name=None,
                 config: Optional[dict] = None, resume_step: Optional[int] = None):
        self.enabled = enabled
        self.wandb = None
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        paths = [os.path.join(out_dir, n) for n in ("metrics.jsonl", "eval.jsonl")]
        if resume_step is not None:  # drop records logged after the checkpoint so resumed steps are not duplicated
            for p in paths:
                if os.path.exists(p):
                    _truncate_after(p, resume_step)
        mode = "w" if resume_step is None else "a"
        self.f, self.fe = (open(p, mode) for p in paths)
        if wandb_project and os.environ.get("WANDB_API_KEY"):  # opted in: a failing wandb is an error
            import wandb

            self.wandb = wandb.init(project=wandb_project, name=run_name, config=config, resume="allow", dir=out_dir)

    def log(self, rec: dict, kind: str = "train"):
        if not self.enabled:
            return
        rec = {k: (float(v) if isinstance(v, torch.Tensor) else v) for k, v in rec.items()}
        f = self.f if kind == "train" else self.fe
        f.write(json.dumps(rec) + "\n")
        f.flush()
        if self.wandb is not None:
            self.wandb.log({(k if kind == "train" else f"eval/{k}"): v for k, v in rec.items()
                            if isinstance(v, (int, float))}, step=rec.get("step"))
        if kind == "train":
            parts = [f"step {rec['step']}"]
            for k in self.PRINT_KEYS:
                v = rec.get(k)
                if v is None:
                    continue
                parts.append(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}")
            comp = [f"{rec[k]:.2f}" for k in sorted(rec) if k.startswith("compression/")]
            if comp:
                parts.append("comp " + "/".join(comp))
            print(" | ".join(parts), flush=True)
        else:
            print("[eval] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in rec.items()), flush=True)

    def close(self):
        if not self.enabled:
            return
        self.f.close()
        self.fe.close()
        if self.wandb is not None:
            self.wandb.finish()


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        t = time.perf_counter()
        dt, self.t0 = t - self.t0, t
        return dt
