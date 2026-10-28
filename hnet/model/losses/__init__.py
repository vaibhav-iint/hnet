"""Auxiliary losses on router outputs (registry kind ``aux_loss``): ``(router_out, target_ratio, scope) ->
(loss, stats)``."""

from .ratio import ratio_loss

__all__ = ["ratio_loss"]
