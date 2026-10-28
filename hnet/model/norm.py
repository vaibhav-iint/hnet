"""RMSNorm with fp32 compute (matches flash-attn's Triton RMSNorm used by goombalab/hnet)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """``y = x / sqrt(mean(x^2) + eps) * weight``, computed in fp32.

    x: [..., D] any float dtype. The output dtype is ``out_dtype`` if given, else ``x.dtype``. Blocks pass the fp32
    residual stream as ``x`` and the incoming activation dtype as ``out_dtype`` (flash-attn's ``prenorm`` semantics).
    """

    def __init__(self, d: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d, device=device, dtype=dtype))

    def reset_parameters(self):
        nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor, out_dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return y.to(out_dtype or x.dtype)

    def extra_repr(self) -> str:
        return f"{self.weight.shape[0]}, eps={self.eps}"
