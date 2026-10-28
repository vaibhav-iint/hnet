"""Learning-rate schedules, registered under kind ``"schedule"``.

A schedule is a callable ``step -> factor`` (factor multiplies ``base_lr * lr_mult`` of every group).
It is stateless: resuming only needs the step, so checkpoints store nothing for it.

Build from a config with :func:`build_schedule`::

    schedule: {name: wsd, warmup: 0.1, decay: 0.2, decay_shape: inverse_sqrt, min_lr_ratio: 0.0}
    schedule: constant            # == {name: constant, warmup: 0}
    schedule: {name: cosine, warmup: 2000, min_lr_ratio: 0.1}

``warmup`` / ``decay`` are fractions of ``total_steps`` when < 1, else absolute step counts.
Steps are 0-based: the update at step t uses ``factor(t)``; linear warmup reaches 1.0 at t = warmup-1
(so step 0 already has a non-zero LR, ``1/warmup``).

WSD decay shapes (``progress`` p in (0, 1] over the decay window, ``r = min_lr_ratio``):
* ``linear``:       1 - p
* ``inverse_sqrt``: 1 - sqrt(p)    (Hagele et al. 2024 "1-sqrt"; the H-Net paper's "inverse-square-root decay")
* ``cosine``:       0.5 (1 + cos(pi p))
* ``none``:         1              (no decay: constant after warmup; useful for open-ended runs)
and the factor is ``r + (1 - r) * shape(p)``.
"""

from __future__ import annotations

import math
from typing import Union

from hnet.registry import get, register

_SHAPES = {
    "linear": lambda p: 1.0 - p,
    "inverse_sqrt": lambda p: 1.0 - math.sqrt(p),
    "1-sqrt": lambda p: 1.0 - math.sqrt(p),
    "cosine": lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    "none": lambda p: 1.0,
}


def _n(x, total: int) -> int:
    if x is None:
        return 0
    x = float(x)
    if 0 < x < 1:
        return int(round(x * total))
    return int(x)


def _warm(step: int, warmup: int) -> float:
    return min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0


@register("schedule", "wsd")
class WSD:
    """Warmup-stable-decay: linear warmup, constant, then decay over the last ``decay`` steps."""

    def __init__(self, total_steps: int, warmup=0.1, decay=0.2, decay_shape: str = "inverse_sqrt",
                 min_lr_ratio: float = 0.0):
        assert decay_shape in _SHAPES, f"decay_shape must be one of {sorted(_SHAPES)}"
        self.total = int(total_steps)
        self.warmup = _n(warmup, self.total)
        self.decay = _n(decay, self.total) if decay_shape != "none" else 0
        assert self.warmup + self.decay <= self.total, "warmup + decay longer than the run"
        self.shape = _SHAPES[decay_shape]
        self.r = float(min_lr_ratio)

    def __call__(self, step: int) -> float:
        start = self.total - self.decay
        if self.decay > 0 and step >= start:
            p = min(1.0, (step - start + 1) / self.decay)
            return self.r + (1.0 - self.r) * self.shape(p)
        return _warm(step, self.warmup)


@register("schedule", "constant")
class Constant:
    def __init__(self, total_steps: int = 0, warmup=0, **_unused):
        self.warmup = _n(warmup, int(total_steps))

    def __call__(self, step: int) -> float:
        return _warm(step, self.warmup)


@register("schedule", "cosine")
class Cosine:
    """Linear warmup then cosine from 1 to ``min_lr_ratio`` at the last step."""

    def __init__(self, total_steps: int, warmup=0.0, min_lr_ratio: float = 0.1):
        self.total = int(total_steps)
        self.warmup = _n(warmup, self.total)
        self.r = float(min_lr_ratio)

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return _warm(step, self.warmup)
        span = max(1, self.total - self.warmup)
        p = min(1.0, (step - self.warmup + 1) / span)
        return self.r + (1.0 - self.r) * 0.5 * (1.0 + math.cos(math.pi * p))


def build_schedule(cfg: Union[str, dict, None], total_steps: int):
    """``cfg`` is a registry name or ``{name: ..., **kwargs}`` (``min_lr`` is accepted for ``min_lr_ratio``)."""
    if cfg is None:
        cfg = "constant"
    if isinstance(cfg, str):
        cfg = {"name": cfg}
    cfg = dict(cfg)
    name = cfg.pop("name", "wsd")
    if "min_lr" in cfg:
        cfg["min_lr_ratio"] = cfg.pop("min_lr")
    return get("schedule", name)(total_steps, **cfg)
