"""SwiGLU feed-forward (parameter names and gate order as goombalab/hnet ``SwiGLU``)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def round_intermediate(d_intermediate: int, multiple_of: int = 128) -> int:
    """Official convention: round the hidden width up to a multiple of 128."""
    return (d_intermediate + multiple_of - 1) // multiple_of * multiple_of


class SwiGLU(nn.Module):
    """``y, gate = fc1(x).chunk(2, -1); out = fc2(silu(gate) * y)``.

    fc1: [2 * d_int, d_model] with rows ``[y | gate]``; fc2: [d_model, d_int]; ``d_int`` rounded up to 128.
    x: [..., d_model] -> [..., d_model].
    """

    def __init__(self, d_model: int, d_intermediate: int, bias: bool = False, multiple_of: int = 128):
        super().__init__()
        d_int = round_intermediate(d_intermediate, multiple_of)
        self.d_intermediate = d_int
        self.fc1 = nn.Linear(d_model, 2 * d_int, bias=bias)
        self.fc2 = nn.Linear(d_int, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * y)
