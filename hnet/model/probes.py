"""Interventions for analysing trained H-Nets (used by the trainer's eval probes and hnet.analysis.diagnose)."""

from __future__ import annotations

import contextlib

import torch


@contextlib.contextmanager
def zero_main(stage):
    """Within the context, ``stage.main_network`` returns zeros (inner router outputs are kept), so the stage's
    decoder sees only the residual path. Measures how much the hierarchy below ``stage`` contributes."""
    orig = stage.main_network.forward

    def fwd(x, ctx, **kw):
        z, preds = orig(x, ctx, **kw)
        return torch.zeros_like(z), preds

    stage.main_network.forward = fwd
    try:
        yield
    finally:
        del stage.main_network.forward  # restore the class method
