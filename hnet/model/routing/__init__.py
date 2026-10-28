"""Boundary predictors (registry kind ``router``). Contract: ``forward(h [T, D], ctx, **extras) -> RouterOutput``;
constructor ``(d_model, target_ratio=None, **router_kwargs)``."""

from . import causal, cosine, heuristic  # noqa: F401  (registration)
from .base import RouterOutput, router_output

__all__ = ["RouterOutput", "router_output"]
