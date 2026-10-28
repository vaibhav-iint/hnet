"""Boundary predictors (registry kind ``router``). Contract: ``forward(h [T, D], ctx, **extras) -> RouterOutput``;
constructor ``(d_model, target_ratio=None, **router_kwargs)``; class/instance attribute ``boundary_semantics``
("start" | "end", see ``routing.base``)."""

from . import cosine, end, heuristic  # noqa: F401  (registration)
from .base import SEMANTICS, RouterOutput, router_output

__all__ = ["SEMANTICS", "RouterOutput", "router_output"]
