"""Chunkers (``(h, boundary_mask, ctx) -> (h_inner, ctx_inner)``) and dechunkers
(``(h_inner, router_out, ctx, ctx_inner) -> h_outer``)."""

from . import dechunk, pool, select  # noqa: F401  (registration)
from .dechunk import plug_back_index
from .scan import linear_recurrence

__all__ = ["linear_recurrence", "plug_back_index"]
