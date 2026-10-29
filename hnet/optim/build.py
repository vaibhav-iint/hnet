"""Entry points used by the trainer: :func:`build_optimizer`, :func:`set_lr`.

Muon hook (WS-G): if ``train.optimizer == "muon"`` the trainer calls

    hnet.optim.muon.build_muon_optimizer(model, train_cfg, param_groups_fn)

with the same contract as every optimizer builder (see :mod:`hnet.optim.adamw`). It may return a single
optimizer or a list/tuple of optimizers (e.g. Muon for matrices + AdamW for embeddings/head/scalars);
a list is wrapped in :class:`MultiOptimizer`. If ``build_muon_optimizer`` is absent, the registry entry
``("optimizer", "muon")`` is used with the same signature.
"""

from __future__ import annotations

import functools
from typing import List, Sequence

import torch

from hnet.optim.groups import param_groups, resolve_lr_multipliers
from hnet.registry import get


class MultiOptimizer:
    """Steps several optimizers as one (minimal Optimizer-like surface used by the trainer)."""

    def __init__(self, optimizers: Sequence[torch.optim.Optimizer]):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self) -> List[dict]:
        return [g for o in self.optimizers for g in o.param_groups]

    def step(self):
        for o in self.optimizers:
            o.step()

    def zero_grad(self, set_to_none: bool = True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {"optimizers": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["optimizers"]):
            o.load_state_dict(s)


def make_param_groups_fn(model_cfg, train_cfg: dict):
    """Bind lr multipliers and wd options from the configs into a ``param_groups_fn``."""
    mults = train_cfg.get("lr_multiplier_override", None)
    if mults is None:
        mults = resolve_lr_multipliers(model_cfg)
    wd_emb = bool(train_cfg.get("wd_embeddings", True))
    no_wd_1d = bool(train_cfg.get("no_wd_1d", True))
    default_wd = float(train_cfg.get("weight_decay", 0.1))

    def fn(model, filter_fn=None, extra=None, weight_decay=None):
        return param_groups(
            model,
            weight_decay=default_wd if weight_decay is None else weight_decay,
            lr_multipliers=mults,
            wd_embeddings=wd_emb,
            no_wd_1d=no_wd_1d,
            filter_fn=filter_fn,
            extra=extra,
        )

    fn.lr_multipliers = mults
    # H-Net-only: optimizer kinds the multipliers apply to (read by the Muon builder; None = all kinds)
    fn.lr_mult_kinds = getattr(model_cfg, "lr_multiplier_kinds", None)
    assert fn.lr_mult_kinds is None or train_cfg.get("optimizer") == "muon", "lr_multiplier_kinds needs Muon"
    return fn


def build_optimizer(model, model_cfg, train_cfg: dict):
    name = train_cfg.get("optimizer", "adamw")
    pg_fn = make_param_groups_fn(model_cfg, train_cfg)
    if name == "muon":
        import importlib

        mod = importlib.import_module("hnet.optim.muon")
        builder = getattr(mod, "build_muon_optimizer", None) or get("optimizer", "muon")
    else:
        builder = get("optimizer", name)
    opt = builder(model, train_cfg, pg_fn)
    if isinstance(opt, (list, tuple)):
        opt = MultiOptimizer(opt)
    lr = float(train_cfg.get("lr", 6e-4))
    for g in opt.param_groups:
        g.setdefault("lr_mult", 1.0)
        g.setdefault("base_lr", g.get("base_lr", lr))
    return opt


def set_lr(optimizer, factor: float) -> float:
    """Apply the schedule factor; returns the LR of a multiplier-1 group (for logging)."""
    base = None
    for g in optimizer.param_groups:
        g["lr"] = g["base_lr"] * g["lr_mult"] * factor
        if base is None:
            base = g["base_lr"] * factor
    return base if base is not None else 0.0
