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

        codes = cfg.mixer_codes_for_stage(stage_idx)
        mixer_kwargs = {name: mixer_config_kwargs(name, cfg, stage_idx) for name in set(codes.values())}

        def iso(lay):
            return Isotropic(lay, d, cfg.d_intermediate[stage_idx], codes, mixer_kwargs, cfg.norm_eps,
                             attn_res=cfg.attn_res_for_stage(stage_idx))

        if self.is_innermost:
            self.main_network = iso(layout[0])
        else:
            n_stages = cfg.n_stages
            dc = cfg.dc.for_stage(stage_idx, n_stages)
            self.target_ratio = dc["target_ratio"]
            self.aux_loss_name = dc["aux_loss"]
            self.aux_loss_scope = cfg.dc.aux_loss_scope
            self.use_ste = bool(dc["ste"])
            self.residual_dropout = float(dc["residual_dropout"] or 0.0)
            self.residual_warmup = float(cfg.dc.residual_warmup or 0.0)
            self.progress = 1.0  # training progress in [0, 1], set by HNetForCausalLM.set_progress
            assert cfg.dc.residual_gate in ("none", "margin"), cfg.dc.residual_gate
            self.residual_gate = cfg.dc.residual_gate
            self.chunk_ste = bool(cfg.dc.chunk_ste)
            self.encoder = iso(layout[0])
            self.main_network = HNet(cfg, stage_idx + 1)
            self.decoder = iso(layout[2])
            self.routing_module = registry.build(
                "router", dc["router"], d, target_ratio=dc["target_ratio"], **(dc["router_kwargs"] or {})
            )
            from .routing.decide import DECISIONS, QuantileDecider

            assert dc["decision"] in DECISIONS, dc["decision"]
            self.decider = (QuantileDecider(dc["target_ratio"], **(dc["decision_kwargs"] or {}))
                            if dc["decision"] == "quantile" else None)
            self.chunk_layer = registry.build("chunker", dc["chunker"], d, **(dc["chunker_kwargs"] or {}))
            need = getattr(self.chunk_layer, "requires_semantics", None)
            have = self.routing_module.boundary_semantics
            assert need is None or need == have, (
                f"chunker {dc['chunker']!r} requires {need!r} boundary semantics, router {dc['router']!r} has {have!r}")
            self.dechunk_layer = registry.build("dechunker", dc["dechunker"])
            self.chunk_vocab = None
            if dc["chunk_vocab"]:
                from .chunk_vocab import ChunkVocab

                assert have == "end", f"dc.chunk_vocab needs 'end' boundary semantics, router {dc['router']!r} has {have!r}"
                self.chunk_vocab = ChunkVocab(d, **dc["chunk_vocab"])  # the inner stage returns width d
            self.residual_proj = nn.Linear(d, d, dtype=torch.float32)
            assert cfg.dc.residual_init in ("zero", "identity"), cfg.dc.residual_init
            if cfg.dc.residual_init == "identity":
                nn.init.eye_(self.residual_proj.weight)
            else:
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
        binfo=None,
    ) -> Tuple[torch.Tensor, List[RouterOutput]]:
        """h [T, D_in] -> ([T, D_in], router outputs of this and all inner stages, outer first)."""
        D_in = h.shape[-1]
        if self.pad_dimension is not None:
            h = torch.cat((h, self.pad_dimension.to(h.dtype).expand(h.shape[0], -1)), dim=-1)
        if self.is_innermost:
            return self.main_network(h, ctx)[:, :D_in], []

        h = self.encoder(h, ctx)
        with _no_autocast(h.device):
            residual = self.residual_proj(h.to(self.residual_proj.weight.dtype)).float()  # fp32 in training
        r = self.routing_module(h, ctx, input_ids=input_ids, boundaries=boundaries[0] if boundaries else None)
        if self.decider is not None:
            r = self.decider(r, ctx)
        x, ctx_inner = self.chunk_layer(h, r.boundary_mask, ctx)
        if self.chunk_ste:  # chunk-level losses (next-chunk CE, main_aux) reach the router through the chunk vectors
            x = x * STE.apply(r.selected_probs.index_select(0, ctx_inner.outer_index)).to(x.dtype)
        inner_ids = input_ids.index_select(0, ctx_inner.outer_index) if input_ids is not None else None
        inner_boundaries = boundaries[1:] if boundaries else None
        inner_binfo = binfo.select(ctx_inner.outer_index) if binfo is not None else None
        if self.chunk_vocab is not None:  # the chunk's identity (hashed content) joins its boundary state
            assert binfo is not None, "dc.chunk_vocab needs the byte hash bank (HNetForCausalLM builds it)"
            bucket = self.chunk_vocab.chunk_ids(binfo, ctx, ctx_inner)
            x_bound = x
            x = x + self.chunk_vocab.embed(bucket, x.dtype)
            if self.chunk_vocab.suffix_lens:  # rows of every suffix ending at the chunk end (consistent keys)
                e_suf, gate = self.chunk_vocab.suffix_embed(x_bound, binfo, ctx, ctx_inner)
                x = x + e_suf
                with torch.no_grad():  # mean gate over the looked-up (unmasked) suffixes
                    g = gate.detach()
                    used = (g > 0).float()
                    r.extra["suffix_gate"] = g.sum() / used.sum().clamp(min=1)
                    for i, n in enumerate(self.chunk_vocab.suffix_lens):
                        r.extra[f"suffix_gate_n{n}"] = g[:, i].sum() / used[:, i].sum().clamp(min=1)
            r.vocab_tgt = self.chunk_vocab.next_targets(bucket[0], ctx_inner)
        z, inner_preds = self.main_network(x, ctx_inner, input_ids=inner_ids, boundaries=inner_boundaries,
                                           binfo=inner_binfo)
        out = self.dechunk_layer(z, r, ctx, ctx_inner)
        if self.use_ste:
            out = out * STE.apply(r.selected_probs)
        with torch.no_grad():  # how strongly the hierarchy below speaks to the decoder, vs the residual path
            r.extra["dechunk_rms"] = out.detach().float().pow(2).mean().sqrt()
            r.extra["residual_rms"] = residual.detach().pow(2).mean().sqrt()
        r.main_out = out  # with grad: read by an optional auxiliary head (HNetConfig.dc.main_aux_weight)
        r.inner_out = z  # with grad: the main network's output at this stage's inner resolution [M, D_inner]
        if self.residual_warmup > 0 and self.progress < self.residual_warmup:
            residual = residual * (self.progress / self.residual_warmup)
        if self.residual_gate == "margin":  # the bypass opens only once the router has decided
            margin = (2 * (r.p.detach().float() - 0.5).abs().mean()).clamp(0, 1)
            residual = residual * margin
            r.extra["residual_gate"] = margin
        if self.training and self.residual_dropout > 0:  # drop the residual path for whole segments
            n_seg = ctx.cu_seqlens.numel() - 1
            keep = (torch.rand(n_seg, device=residual.device) >= self.residual_dropout).to(residual.dtype)
            residual = residual * keep[ctx.seq_idx.reshape(-1).long()].unsqueeze(-1)
        r.residual_in = residual.detach()  # read by HNetForCausalLM for the router gain objective (stage 0)
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
