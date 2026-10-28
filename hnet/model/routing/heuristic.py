"""Non-learned routers (hard 0/1 probabilities, so the STE confidence is 1 and EMA smoothing is a no-op).

Like ``cosine_lookahead`` they select chunk *starts* (the first position of every segment is always selected):

* ``fixed_pool``: a chunk starts every ``k`` positions of a segment (``k`` defaults to round(target_ratio)).
* ``space_like``: SpaceByte rule. A byte is "spacelike" unless it is an ASCII letter/digit or a UTF-8
  continuation byte (0x80-0xBF); a chunk starts at t when byte t-1 is spacelike and byte t is not (start of a
  word). Needs the extra ``input_ids`` [T] (the H-Net forwards the stage's byte ids, gathered for inner stages).
* ``external``: boundaries given as the extra ``boundaries`` [T] (bool, or probabilities in [0, 1]).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, router_output


@register("router", "fixed_pool")
class FixedPoolRouter(nn.Module):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None, k: Optional[int] = None):
        super().__init__()
        self.k = int(k if k is not None else round(target_ratio))
        assert self.k >= 1

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        return router_output((ctx.position_ids % self.k == 0).float(), ctx)

    def extra_repr(self) -> str:
        return f"k={self.k}"


def _byte_table() -> torch.Tensor:
    word = torch.zeros(256, dtype=torch.bool)
    for lo, hi in ((ord("0"), ord("9")), (ord("A"), ord("Z")), (ord("a"), ord("z")), (0x80, 0xBF)):
        word[lo : hi + 1] = True
    return ~word


@register("router", "space_like")
class SpaceLikeRouter(nn.Module):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None):
        super().__init__()
        self.register_buffer("spacelike", _byte_table(), persistent=False)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, input_ids: Optional[torch.Tensor] = None, **_) -> RouterOutput:
        assert input_ids is not None, "space_like router needs input_ids"
        sl = self.spacelike[input_ids.long()]
        start = torch.zeros_like(sl)
        start[1:] = sl[:-1] & ~sl[1:]
        return router_output(start.float(), ctx)


@register("router", "external")
class ExternalRouter(nn.Module):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None):
        super().__init__()

    def forward(self, h: torch.Tensor, ctx: SeqCtx, boundaries: Optional[torch.Tensor] = None, **_) -> RouterOutput:
        assert boundaries is not None, "external router needs boundaries"
        return router_output(boundaries.float(), ctx)
