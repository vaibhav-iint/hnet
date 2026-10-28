"""Optimizers, parameter groups, per-stage LR multipliers and LR schedules.

Importing this package registers ``("optimizer", "adamw")`` and ``("schedule", "wsd"|"constant"|"cosine")``.
``hnet.optim.muon`` (WS-G) is imported by :mod:`hnet.registry`, not here.
"""

from hnet.optim import adamw, schedules  # noqa: F401  (registration)
from hnet.optim.build import MultiOptimizer, build_optimizer, make_param_groups_fn, set_lr
from hnet.optim.groups import (
    auto_lr_multipliers,
    classify_params,
    param_groups,
    params_by_stage,
    resolve_lr_multipliers,
)
from hnet.optim.schedules import build_schedule

__all__ = [
    "MultiOptimizer",
    "auto_lr_multipliers",
    "build_optimizer",
    "build_schedule",
    "classify_params",
    "make_param_groups_fn",
    "param_groups",
    "params_by_stage",
    "resolve_lr_multipliers",
    "set_lr",
]
