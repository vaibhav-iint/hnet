"""Non-learned routers with hard 0/1 probabilities (STE confidence 1; EMA smoothing is a no-op).

Each takes ``semantics="start" | "end"`` (see ``routing.base``); the first position of a segment is always
selected.

* ``fixed_pool``: chunks of ``k`` positions (``k`` defaults to round(target_ratio)). "start": select positions
  with ``pos % k == 0``; "end": select ``pos % k == k - 1`` (plus the forced 1-byte chunk at ``pos == 0``).
* ``space_like``: SpaceByte rule. A byte is "spacelike" unless it is an ASCII letter/digit or a UTF-8
  continuation byte (0x80-0xBF). "start" (default): select t when byte t-1 is spacelike and byte t is not (first
  byte of a word); "end": select t when byte t is spacelike and byte t-1 is not (the space closing a word,
  SpaceByte's own placement). Needs the extra ``input_ids`` [T] (inner stages receive the gathered ids).
* ``external``: boundaries given as the extra ``boundaries`` [T] (bool, or probabilities in [0, 1]).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hnet.registry import register

from ..ctx import SeqCtx
from .base import SEMANTICS, RouterOutput, router_output


class _HardRouter(nn.Module):
    def __init__(self, semantics: str = "start"):
        super().__init__()
        assert semantics in SEMANTICS, semantics
        self.boundary_semantics = semantics

    def extra_repr(self) -> str:
        return f"semantics={self.boundary_semantics}"


@register("router", "fixed_pool")
class FixedPoolRouter(_HardRouter):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None, k: Optional[int] = None, semantics="start"):
        super().__init__(semantics)
        self.k = int(k if k is not None else round(target_ratio))
        assert self.k >= 1

    def forward(self, h: torch.Tensor, ctx: SeqCtx, **_) -> RouterOutput:
        phase = 0 if self.boundary_semantics == "start" else self.k - 1
        return router_output((ctx.position_ids % self.k == phase).float(), ctx, self.boundary_semantics)


def _spacelike_table() -> torch.Tensor:
    word = torch.zeros(256, dtype=torch.bool)
    for lo, hi in ((ord("0"), ord("9")), (ord("A"), ord("Z")), (ord("a"), ord("z")), (0x80, 0xBF)):
        word[lo : hi + 1] = True
    return ~word


@register("router", "space_like")
class SpaceLikeRouter(_HardRouter):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None, semantics="start"):
        super().__init__(semantics)
        self.register_buffer("spacelike", _spacelike_table(), persistent=False)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, input_ids: Optional[torch.Tensor] = None, **_) -> RouterOutput:
        assert input_ids is not None, "space_like router needs input_ids"
        sl = self.spacelike[input_ids.long()]
        sel = torch.zeros_like(sl)
        if self.boundary_semantics == "start":
            sel[1:] = sl[:-1] & ~sl[1:]
        else:
            sel[1:] = ~sl[:-1] & sl[1:]
        return router_output(sel.float(), ctx, self.boundary_semantics)


@register("router", "external")
class ExternalRouter(_HardRouter):
    def __init__(self, d_model: int, target_ratio: Optional[float] = None, semantics="start"):
        super().__init__(semantics)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, boundaries: Optional[torch.Tensor] = None, **_) -> RouterOutput:
        assert boundaries is not None, "external router needs boundaries"
        return router_output(boundaries.float(), ctx, self.boundary_semantics)
