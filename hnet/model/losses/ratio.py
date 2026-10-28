"""Ratio (load-balancing) loss, paper eq. 10 / goombalab ``load_balancing_loss``; registered as aux_loss ``ratio``."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.distributed as dist

from hnet.registry import register

from ..routing.base import RouterOutput


def _rank_mean(x: torch.Tensor, differentiable: bool) -> torch.Tensor:
    """Mean of a scalar over ranks (every rank sees the same number of positions per step)."""
    if differentiable:
        from torch.distributed.nn.functional import all_reduce

        x = all_reduce(x)
    else:
        x = x.clone()
        dist.all_reduce(x)
    return x / dist.get_world_size()


@register("aux_loss", "ratio")
def ratio_loss(router_out: RouterOutput, target_ratio: float, scope: str = "local") -> Tuple[torch.Tensor, Dict]:
    """``N/(N-1) * ((N-1) F G + (1-F)(1-G))`` with F = mean(boundary_mask), G = mean(p) (gradient through G only).

    scope "local": F, G over this device's positions (official). "global": F, G averaged over all ranks
    (all-reduce; G keeps its gradient) when torch.distributed is initialized, else same as "local".
    Returns (loss [], {"F": F, "G": G}) with detached stats. Minimum 1 at F = G = 1/N.
    """
    N = float(target_ratio)
    assert N > 1, "target_ratio must be > 1"
    F_ = router_out.boundary_mask.float().mean()
    G = router_out.p.float().mean()
    if scope == "global" and dist.is_available() and dist.is_initialized():
        F_ = _rank_mean(F_.detach(), differentiable=False)
        G = _rank_mean(G, differentiable=True)
    elif scope not in ("local", "global"):
        raise ValueError(f"unknown aux_loss scope {scope!r}")
    loss = N / (N - 1) * ((N - 1) * F_ * G + (1 - F_) * (1 - G))
    return loss, {"F": F_.detach(), "G": G.detach()}
