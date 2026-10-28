"""Recursive H-Net stage (goombalab/hnet ``HNet``; parameter names match the official checkpoints).

Stage s with layout ``[enc, inner, dec]`` maps h [T, D^{s-1}] -> [T, D^{s-1}]:

    h        = cat(h, pad_dimension)                      (if D^s > D^{s-1}; learned, shared by all positions)
    h        = encoder(h)
    residual = residual_proj(h)                           (fp32 Linear, zero-init weight)
    r        = router(h, ctx, **extras)                   -> RouterOutput ("start" or "end" boundary semantics)
    x, ctx'  = chunker(h, r.boundary_mask, ctx)          (keeps the selected positions)
    z, preds = main_network(x, ctx', **extras')           (HNet stage s+1, or Isotropic if innermost)
    h        = dechunker(z, r, ctx, ctx') * STE(r.selected_probs) + residual      (fp32)
    h        = decoder(h)[:, :D^{s-1}]

A layout ``[main]`` is an innermost stage (pad, Isotropic, truncate). Router / chunker / dechunker / STE are
chosen per stage from ``cfg.dc.for_stage(s)`` through the registry. Extras forwarded to routers: ``input_ids``
(this stage's byte ids, gathered for inner stages) and ``boundaries`` (list, one [T_s] tensor per stage, for the
``external`` router).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from hnet import registry

from .block import mixer_config_kwargs
from .ctx import SeqCtx
from .isotropic import Isotropic
from .routing.base import RouterOutput


class STE(torch.autograd.Function):
    """Forward: ones; backward: identity (paper eq. 7: c + stopgrad(1 - c))."""

    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False)


class HNet(nn.Module):
    def __init__(self, cfg, stage_idx: int = 0):
        super().__init__()
        self.stage_idx = stage_idx
        self.d_model = d = cfg.d_model[stage_idx]
        layout = cfg.stage_layout(stage_idx)
        self.is_innermost = len(layout) == 1
        assert len(layout) in (1, 3), f"stage {stage_idx}: layout must be [main] or [enc, inner, dec], got {layout}"

        mixer_kwargs = {name: mixer_config_kwargs(name, cfg, stage_idx) for name in set(cfg.mixer_codes.values())}

        def iso(lay):
            return Isotropic(lay, d, cfg.d_intermediate[stage_idx], cfg.mixer_codes, mixer_kwargs, cfg.norm_eps)

        if self.is_innermost:
            self.main_network = iso(layout[0])
        else:
            n_stages = cfg.n_stages
            dc = cfg.dc.for_stage(stage_idx, n_stages)
            self.target_ratio = dc["target_ratio"]
            self.aux_loss_name = dc["aux_loss"]
            self.aux_loss_scope = cfg.dc.aux_loss_scope
            self.use_ste = bool(dc["ste"])
            self.encoder = iso(layout[0])
            self.main_network = HNet(cfg, stage_idx + 1)
            self.decoder = iso(layout[2])
            self.routing_module = registry.build(
                "router", dc["router"], d, target_ratio=dc["target_ratio"], **(dc["router_kwargs"] or {})
            )
            self.chunk_layer = registry.build("chunker", dc["chunker"])
            self.dechunk_layer = registry.build("dechunker", dc["dechunker"])
            self.residual_proj = nn.Linear(d, d, dtype=torch.float32)
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True

        d_prev = cfg.d_model[stage_idx - 1] if stage_idx > 0 else d
        self.pad_dimension = nn.Parameter(torch.zeros(d - d_prev)) if d > d_prev else None

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        h: torch.Tensor,
        ctx: SeqCtx,
        input_ids: Optional[torch.Tensor] = None,
        boundaries: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[RouterOutput]]:
        """h [T, D_in] -> ([T, D_in], router outputs of this and all inner stages, outer first)."""
        D_in = h.shape[-1]
        if self.pad_dimension is not None:
            h = torch.cat((h, self.pad_dimension.to(h.dtype).expand(h.shape[0], -1)), dim=-1)
        if self.is_innermost:
            return self.main_network(h, ctx)[:, :D_in], []

        h = self.encoder(h, ctx)
        with _no_autocast(h.device):
            residual = self.residual_proj(h.float())
        r = self.routing_module(h, ctx, input_ids=input_ids, boundaries=boundaries[0] if boundaries else None)
        x, ctx_inner = self.chunk_layer(h, r.boundary_mask, ctx)
        inner_ids = input_ids.index_select(0, ctx_inner.outer_index) if input_ids is not None else None
        inner_boundaries = boundaries[1:] if boundaries else None
        z, inner_preds = self.main_network(x, ctx_inner, input_ids=inner_ids, boundaries=inner_boundaries)
        out = self.dechunk_layer(z, r, ctx, ctx_inner)
        if self.use_ste:
            out = out * STE.apply(r.selected_probs)
        h = (out + residual).to(h.dtype)
        h = self.decoder(h, ctx)
        return h[:, :D_in], [r, *inner_preds]

    # ------------------------------------------------------------------ helpers
    def stages(self) -> List["HNet"]:
        """This stage and all inner stages, outer first."""
        return [self] if self.is_innermost else [self, *self.main_network.stages()]

    def aux_losses(self, preds: List[RouterOutput]):
        """[(loss, stats)] for this and inner stages, given the router outputs returned by ``forward``."""
        return [
            registry.get("aux_loss", st.aux_loss_name)(r, st.target_ratio, st.aux_loss_scope)
            for st, r in zip(self.stages(), preds)
        ]

    def init_weights(self, std: float, parent_residuals: int = 0):
        """Official ``HNet._init_weights``: depth-scaled normal init of encoder/decoder/main Linear weights, where
        the residual count accumulates the heights of all enclosing encoders and decoders."""
        n = parent_residuals
        if self.is_innermost:
            n += self.main_network.height
            self.main_network.init_linear_weights(std, n)
            return
        n += self.encoder.height + self.decoder.height
        self.encoder.init_linear_weights(std, n)
        self.decoder.init_linear_weights(std, n)
        self.main_network.init_weights(std, n)
