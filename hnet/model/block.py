"""Pre-norm residual block with the residual stream kept in fp32 (goombalab/hnet ``Block``).

    residual = residual + hidden            (fp32; the first block starts from residual = hidden)
    hidden   = mixer(norm1(residual))
    if mlp:  residual = residual + hidden;  hidden = mlp(norm2(residual))
    return hidden, residual

The block returns its *un-added* output; the next block (or the final norm of ``Isotropic``) adds it to the
residual, exactly like flash-attn's fused ``RMSNorm(prenorm=True, residual_in_fp32=True)``.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn

from hnet import registry

from .ctx import SeqCtx
from .mlp import SwiGLU
from .norm import RMSNorm


def mixer_config_kwargs(name: str, cfg, stage: int) -> dict:
    """Constructor kwargs of mixer ``name`` for stage ``stage`` of an ``HNetConfig``.

    A mixer class may define ``config_kwargs(cfg, stage) -> dict``; otherwise ``mamba2`` reads ``cfg.ssm`` and
    any other mixer gets no extra kwargs.
    """
    cls = registry.get("mixer", name)
    hook = getattr(cls, "config_kwargs", None)
    if hook is not None:
        return hook(cfg, stage)
    if name == "mamba2":
        return asdict(cfg.ssm)
    return {}


class Block(nn.Module):
    """One residual block. ``forward(hidden [1,T,D], residual [1,T,D] fp32 | None, ctx) -> (hidden, residual)``."""

    def __init__(
        self,
        d_model: int,
        mixer: str,
        mixer_kwargs: dict,
        d_intermediate: int = 0,
        layer_idx: int = 0,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model, eps=norm_eps)
        self.mixer = registry.build("mixer", mixer, d_model, layer_idx=layer_idx, **mixer_kwargs)
        if d_intermediate > 0:
            self.norm2 = RMSNorm(d_model, eps=norm_eps)
            self.mlp = SwiGLU(d_model, d_intermediate)
        else:
            self.mlp = None

    def forward(
        self, hidden: torch.Tensor, residual: Optional[torch.Tensor], ctx: SeqCtx
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = hidden.float() if residual is None else residual + hidden.float()
        hidden = self.mixer(self.norm1(residual, out_dtype=hidden.dtype), ctx)
        if self.mlp is not None:
            residual = residual + hidden.float()
            hidden = self.mlp(self.norm2(residual, out_dtype=hidden.dtype))
        return hidden, residual
