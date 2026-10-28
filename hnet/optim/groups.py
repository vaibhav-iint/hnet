"""Parameter grouping: weight-decay exclusion and per-stage learning-rate multipliers.

Learning-rate modulation (H-Net paper, App. C, Eq. 11)
-----------------------------------------------------
Every parameter of stage ``s`` (s = 0 outermost/byte level, s = S main network) trains with
``lr = base_lr * lambda^s * schedule(step)``, where

    lambda^s = sqrt( N_GPT * (prod_{i=s}^{S} N^i / prod_{i=0}^{S} N^i) * D^S / D^s ),   N^S = 1
             = sqrt( N_GPT * D^S / D^s / prod_{i<s} N^i )

* ``N^i``  = target downsampling ratio of stage i (``cfg.dc.target_ratios[i]``), N^S = 1;
* ``D^s``  = hidden width of stage s (``cfg.d_model[s]``);
* ``N_GPT`` = average bytes per BPE token of the training data (4.6 for GPT-2 on FineWeb-Edu;
  ``cfg.lr_multiplier_n_gpt``).

The two factors are "LR ~ sqrt(batch size)" (stage s sees N_GPT / prod_{i<s} N^i times as many
positions per step as a BPE model) and "LR ~ 1/sqrt(width)" (muP-style, relative to the main width).
Example (paper's 2-stage XL, (3,3)-DC, D = 1024/1536/2048, N_GPT = 4.6):
lambda = [sqrt(9.2), sqrt(4.6/3 * 4/3), sqrt(4.6/9)] = [3.033, 1.430, 0.715].
The paper also notes hand-set [2.0, 1.5, 1.0] works well; pass a list for that.

``cfg.lr_multiplier``: ``None`` -> all 1.0; a list (outer first, len n_stages+1); ``"auto"`` -> Eq. 11.
Embeddings and the LM head belong to stage 0 (``model.stage_of_param``), as in goombalab/hnet.

Weight decay is not applied to: biases, normalization weights, any parameter with ``ndim < 2``
(``no_wd_1d``, default on; covers Mamba's ``A_log``/``D``/``dt_bias``), parameters flagged
``p._no_weight_decay = True`` or ``p._optim = {"weight_decay": 0.0}`` (mamba_ssm/goombalab convention),
and optionally embeddings (``wd_embeddings: false``).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch import nn


def auto_lr_multipliers(d_model: Sequence[int], target_ratios: Sequence[float], n_gpt: float = 4.6) -> List[float]:
    """Paper App. C Eq. 11. ``d_model`` has n_stages+1 entries, ``target_ratios`` n_stages entries."""
    S = len(d_model) - 1
    assert len(target_ratios) == S, f"need {S} target ratios, got {len(target_ratios)}"
    N = list(map(float, target_ratios)) + [1.0]  # N^S = 1
    out = []
    for s in range(S + 1):
        num = math.prod(N[s:])  # prod_{i=s}^{S} N^i
        den = math.prod(N)  # prod_{i=0}^{S} N^i
        out.append(math.sqrt(n_gpt * num / den * d_model[S] / d_model[s]))
    return out


def resolve_lr_multipliers(model_cfg, n_stages: Optional[int] = None) -> List[float]:
    """Per-stage multipliers (outer first) from a model config. Dense configs -> [1.0]."""
    if model_cfg is None or not hasattr(model_cfg, "lr_multiplier"):
        return [1.0] * ((n_stages or 0) + 1)
    n = model_cfg.n_stages + 1
    lm = model_cfg.lr_multiplier
    if lm is None or lm is False:
        return [1.0] * n
    if isinstance(lm, str):
        assert lm == "auto", f"lr_multiplier must be a list, null or 'auto', got {lm!r}"
        return auto_lr_multipliers(model_cfg.d_model, model_cfg.dc.target_ratios, model_cfg.lr_multiplier_n_gpt)
    if isinstance(lm, (int, float)):
        return [float(lm)] * n
    lm = [float(x) for x in lm]
    assert len(lm) == n, f"lr_multiplier: expected {n} values (outer first), got {len(lm)}"
    return lm


_NORM_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)


def _is_norm_module(m: nn.Module) -> bool:
    return isinstance(m, _NORM_TYPES) or "norm" in type(m).__name__.lower() or isinstance(m, getattr(nn, "RMSNorm", ()))


def classify_params(model: nn.Module, wd_embeddings: bool = True, no_wd_1d: bool = True) -> Dict[str, dict]:
    """name -> {"param", "stage", "decay": bool} for every trainable parameter (shared params once)."""
    stage_fn: Callable[[str], int] = getattr(model, "stage_of_param", None) or (lambda name: 0)
    out: Dict[str, dict] = {}
    seen = set()
    for mname, mod in model.named_modules():
        is_norm = _is_norm_module(mod)
        is_emb = isinstance(mod, nn.Embedding)
        for pname, p in mod.named_parameters(recurse=False):
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            name = f"{mname}.{pname}" if mname else pname
            decay = True
            if pname.endswith("bias") or is_norm or ".norm" in name or name.startswith("norm"):
                decay = False
            if no_wd_1d and p.ndim < 2:
                decay = False
            if getattr(p, "_no_weight_decay", False):
                decay = False
            optim_flags = getattr(p, "_optim", None)
            if isinstance(optim_flags, dict) and optim_flags.get("weight_decay", None) == 0.0:
                decay = False
            if is_emb and not wd_embeddings:
                decay = False
            out[name] = {"param": p, "stage": int(stage_fn(name)), "decay": decay, "embedding": is_emb}
    return out


def param_groups(
    model: nn.Module,
    weight_decay: float = 0.0,
    lr_multipliers: Optional[Sequence[float]] = None,
    wd_embeddings: bool = True,
    no_wd_1d: bool = True,
    filter_fn: Optional[Callable[[str, torch.Tensor], bool]] = None,
    extra: Optional[dict] = None,
) -> List[dict]:
    """Build optimizer param groups keyed by (stage, decay).

    Each group has ``params``, ``weight_decay``, ``lr_mult`` (the stage's multiplier; the trainer sets
    ``lr = base_lr * lr_mult * schedule``), ``stage`` and ``names``. ``filter_fn(name, p)`` restricts
    the params considered (used by Muon to split matrices from the rest). ``extra`` is merged into
    every group (e.g. per-group betas).
    """
    info = classify_params(model, wd_embeddings=wd_embeddings, no_wd_1d=no_wd_1d)
    mults = list(lr_multipliers) if lr_multipliers is not None else None
    groups: Dict[tuple, dict] = {}
    for name, d in info.items():
        p = d["param"]
        if filter_fn is not None and not filter_fn(name, p):
            continue
        s = d["stage"]
        if mults is None:
            m = 1.0
        else:
            assert s < len(mults), f"param {name} has stage {s} but only {len(mults)} lr multipliers"
            m = float(mults[s])
        key = (s, d["decay"])
        if key not in groups:
            groups[key] = {
                "params": [],
                "names": [],
                "weight_decay": float(weight_decay) if d["decay"] else 0.0,
                "lr_mult": m,
                "stage": s,
                **(extra or {}),
            }
        groups[key]["params"].append(p)
        groups[key]["names"].append(name)
    return [groups[k] for k in sorted(groups)]


def params_by_stage(model: nn.Module) -> Dict[int, List[torch.Tensor]]:
    """stage -> list of trainable params (for per-stage grad norms)."""
    out: Dict[int, List[torch.Tensor]] = {}
    for name, d in classify_params(model).items():
        out.setdefault(d["stage"], []).append(d["param"])
    return out
