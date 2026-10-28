"""Model package: H-Net (byte-level, dynamic chunking) and dense baselines.

Importing this package registers every built-in mixer, router, chunker, dechunker, aux loss and dense model.

    from hnet.model import build_model, load_official_checkpoint
    model = build_model(cfg)        # HNetConfig -> HNetForCausalLM ; DenseConfig -> registry "dense"[cfg.variant]
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

import torch

from hnet import registry
from hnet.config import DenseConfig, HNetConfig

from . import chunking, dense, losses, routing  # noqa: F401  (registration)
from .ctx import SeqCtx
from .hnet import HNet
from .lm import CausalLMOutput, HNetForCausalLM
from .mixers import attn, mamba2  # noqa: F401  (registration)
from .routing import RouterOutput

__all__ = [
    "CausalLMOutput",
    "HNet",
    "HNetForCausalLM",
    "RouterOutput",
    "SeqCtx",
    "build_model",
    "load_official_checkpoint",
]


def build_model(cfg):
    """``HNetConfig`` -> ``HNetForCausalLM``; ``DenseConfig`` -> the registered dense model ``cfg.variant``.
    Parameters are initialized by the module constructors; call ``model.init_weights()`` for the official init."""
    if isinstance(cfg, HNetConfig):
        return HNetForCausalLM(cfg)
    if isinstance(cfg, DenseConfig):
        return registry.build("dense", cfg.variant, cfg)
    raise TypeError(f"unsupported model config {type(cfg).__name__}")


def load_official_checkpoint(
    repo_or_path: str,
    config_json,
    dtype: torch.dtype = torch.float32,
    device="cpu",
    target_ratios: Optional[Sequence[float]] = None,
) -> HNetForCausalLM:
    """Load a goombalab/hnet checkpoint (e.g. ``"cartesia-ai/hnet_2stage_L"`` or a local ``.pt``) into our model.

    config_json: path or dict of the official config (``configs/official/*.json``). The model is built on the meta
    device and the (memory-mapped) state dict is assigned with ``strict=True``, so peak memory is about one copy of
    the weights in ``dtype``. ``residual_proj`` stays fp32 as in the official model.
    """
    cfg = HNetConfig.from_official_json(config_json, target_ratios)
    path = repo_or_path
    if not os.path.isfile(path):
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_or_path, f"{repo_or_path.rstrip('/').split('/')[-1]}.pt")
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = {
        k: v.to(device=device, dtype=torch.float32 if "residual_proj" in k else dtype) for k, v in state.items()
    }
    with torch.device("meta"):
        model = HNetForCausalLM(cfg)
    model.load_state_dict(state, strict=True, assign=True)
    leftover = [n for n, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if leftover:
        raise RuntimeError(f"tensors not provided by the checkpoint: {leftover[:5]}")
    return model.eval()
