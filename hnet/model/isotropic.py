"""A stack of residual blocks followed by a final RMSNorm (goombalab/hnet ``Isotropic``).

Layout strings are sequences of ``<code><count>``, e.g. ``"T1m4"``. The lower-cased code is looked up in
``mixer_codes`` (e.g. ``{"m": "mamba2", "t": "attn"}``) to get a mixer registry name; an upper-case code adds a
SwiGLU MLP to each of its blocks. Adding a mixer is therefore a registry entry plus a ``mixer_codes`` entry.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import torch
import torch.nn as nn

from .block import Block
from .ctx import SeqCtx
from .norm import RMSNorm

_TOKEN = re.compile(r"([A-Za-z])(\d+)")


def parse_layout(layout: str, mixer_codes: dict) -> List[Tuple[str, bool]]:
    """``"T1m4"`` -> ``[("attn", True), ("mamba2", False) x4]`` (mixer name, has_mlp) per layer."""
    if re.fullmatch(r"(?:[A-Za-z]\d+)*", layout) is None:
        raise ValueError(f"bad layout {layout!r}; expected e.g. 'm4' or 'T1m4'")
    out = []
    for code, n in _TOKEN.findall(layout):
        if code.lower() not in mixer_codes:
            raise KeyError(f"layout code {code!r} not in mixer_codes {sorted(mixer_codes)}")
        out += [(mixer_codes[code.lower()], code.isupper())] * int(n)
    return out


class Isotropic(nn.Module):
    """``forward(h [T, D] or [1, T, D], ctx) -> same shape``.

    ``mixer_kwargs`` maps mixer registry name -> constructor kwargs. ``layer_idx`` counts from 0 inside this
    stack (as official; it drives e.g. NoPE layer selection). ``height`` = number of residual additions (one per
    mixer, one per MLP); it sets the depth-scaled init in ``init_linear_weights``.
    """

    def __init__(
        self,
        layout: str,
        d_model: int,
        d_intermediate: int,
        mixer_codes: dict,
        mixer_kwargs: dict,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        spec = parse_layout(layout, mixer_codes)
        if any(mlp for _, mlp in spec):
            assert d_intermediate > 0, f"layout {layout!r} has MLP blocks but d_intermediate == 0"
        self.layers = nn.ModuleList(
            Block(
                d_model,
                name,
                mixer_kwargs.get(name, {}),
                d_intermediate=d_intermediate if mlp else 0,
                layer_idx=i,
                norm_eps=norm_eps,
            )
            for i, (name, mlp) in enumerate(spec)
        )
        self.height = sum(2 if mlp else 1 for _, mlp in spec)
        self.rmsnorm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, h: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        squeeze = h.dim() == 2
        hidden, residual = (h.unsqueeze(0) if squeeze else h), None
        for layer in self.layers:
            hidden, residual = layer(hidden, residual, ctx)
        residual = hidden.float() if residual is None else residual + hidden.float()
        out = self.rmsnorm(residual, out_dtype=hidden.dtype)
        return out.squeeze(0) if squeeze else out

    def init_linear_weights(self, std: float, n_residuals: int):
        """Official ``_init_weights`` for this stack: every ``nn.Linear`` weight not flagged ``_no_reinit`` gets
        N(0, std); ``out_proj``/``fc2`` (the residual writers) get N(0, std / sqrt(n_residuals))."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                scale = n_residuals ** -0.5 if ("out_proj" in name or "fc2" in name) else 1.0
                nn.init.normal_(m.weight, mean=0.0, std=std * scale)
