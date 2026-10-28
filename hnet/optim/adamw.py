"""AdamW (registered ``("optimizer", "adamw")``) and the optimizer-builder convention.

Optimizer builders registered under kind ``"optimizer"`` have the signature

    builder(model, train_cfg: dict, param_groups_fn) -> torch.optim.Optimizer

* ``train_cfg`` is the run's ``train:`` section (``lr``, ``betas``, ``eps``, ``weight_decay``, ...).
* ``param_groups_fn(model, filter_fn=None, extra=None, weight_decay=None) -> list[dict]`` returns
  groups with weight-decay exclusions and per-stage ``lr_mult`` already applied
  (see :func:`hnet.optim.groups.param_groups`).

The trainer sets, every step, for every group:
``group["lr"] = group.get("base_lr", train.lr) * group.get("lr_mult", 1.0) * schedule(step)``.
A builder that needs a different base LR for some groups (e.g. Muon's matrix LR) stores it as
``group["base_lr"]``.
"""

from __future__ import annotations

import torch

from hnet.registry import register


def _fused_ok(model) -> bool:
    try:
        p = next(model.parameters())
    except StopIteration:
        return False
    return p.is_cuda


@register("optimizer", "adamw")
def build_adamw(model, train_cfg: dict, param_groups_fn):
    groups = param_groups_fn(model, weight_decay=float(train_cfg.get("weight_decay", 0.1)))
    lr = float(train_cfg.get("lr", 6e-4))
    for g in groups:
        g["lr"] = g.get("base_lr", lr) * g.get("lr_mult", 1.0)
    betas = tuple(float(b) for b in train_cfg.get("betas", (0.9, 0.95)))
    eps = float(train_cfg.get("eps", 1e-8))
    fused = train_cfg.get("fused", None)
    if fused is None:
        fused = _fused_ok(model)
    kw = {"fused": True} if fused else {"foreach": True}
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, **kw)
