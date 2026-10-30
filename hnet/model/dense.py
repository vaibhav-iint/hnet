"""Token-level Llama-style Transformer baseline, registered as dense ``"llama"`` (paper's BPE Transformer).

Pre-norm blocks (RMSNorm, fp32 residual stream), attention with RoPE (optional GQA, NoPE every k layers), SwiGLU
MLP, final RMSNorm, untied or tied lm_head. Built from the same ``Isotropic`` / ``Block`` / ``CausalAttention``
modules as the H-Net, so it shares document masking and the forward/output API of ``HNetForCausalLM``.
``cfg.extra["attn"]`` may override attention kwargs (e.g. ``rotary_emb_dim``, ``window_size``, ``qkv_bias``).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from hnet.config import IGNORE_INDEX, DenseConfig
from hnet.registry import register

from .block import Block
from .isotropic import Isotropic
from .lm import CausalLMOutput, lm_outputs, lm_outputs_sparse_bonus, pack_inputs


@register("dense", "llama")
class LlamaLM(nn.Module):
    n_stages = 0

    def __init__(self, cfg: DenseConfig):
        super().__init__()
        self.config = cfg
        attn = dict(
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            rope_theta=cfg.rope_theta,
            nope_every=cfg.nope_every,
        )
        if cfg.head_dim is not None:  # only when set: mixers without the knob keep their signature
            attn["head_dim"] = cfg.head_dim
        attn.update(cfg.extra.get("attn", {}))
        mixer_kwargs = {**attn, **cfg.mixer_kwargs}  # attention-style mixers share the head/RoPE settings
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = Isotropic(
            f"T{cfg.n_layers}", cfg.d_model, cfg.d_intermediate, {"t": cfg.mixer}, {cfg.mixer: mixer_kwargs},
            cfg.norm_eps, attn_res=cfg.attn_res,
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight
        # in-context match model (hnet.model.match.TokenMatchModel), built last under a forked RNG (its parameters
        # are zero-initialized), so every other module's initialization is identical with and without it
        self.match_model = None
        if cfg.match_model:
            from .match import TokenMatchModel

            assert cfg.attn_res == 0, "match_model is not combined with attn_res"
            with torch.random.fork_rng(devices=[]):
                self.match_model = TokenMatchModel(cfg.d_model, cfg.vocab_size, cfg.n_layers, **cfg.match_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        ids, ctx, batch_shape = pack_inputs(input_ids, cu_seqlens, max_seqlen)
        if self.match_model is not None:
            return self._forward_match(ids, ctx, batch_shape, labels, return_logits)
        h = self.backbone(self.embeddings(ids), ctx)
        out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape)
        out.aux_loss = torch.zeros((), device=ids.device)
        out.stats = {"positions/s0": ids.new_full((), ids.numel(), dtype=torch.float32)}
        return out

    def _forward_match(self, ids, ctx, batch_shape, labels, return_logits) -> CausalLMOutput:
        """``forward`` with the token match model: copy candidates (no gradient) enter at the embedding, before
        block ``dec_layer`` and as a gated logit bonus (hnet.model.match.TokenMatchModel)."""
        mm = self.match_model
        cand = mm.candidates(ids, ctx)
        x = self.embeddings(ids)
        if mm.has("enc"):
            x = x + mm.site_embed("enc", cand, self.embeddings).to(x.dtype)
        inject = (mm.dec_layer, mm.site_embed("dec", cand, self.embeddings)) if mm.has("dec") else None
        h = self.backbone(x, ctx, inject=inject)
        if mm.gate is not None:
            cols, vals = mm.gate_values(h, cand)
            out = lm_outputs_sparse_bonus(h, self.lm_head, labels, return_logits, batch_shape, cols, vals)
        else:
            out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape)
        out.aux_loss = torch.zeros((), device=ids.device)
        out.stats = {"positions/s0": ids.new_full((), ids.numel(), dtype=torch.float32)}
        with torch.no_grad():  # mechanism metrics: match coverage / accuracy per order, gate biases
            hit = cand != mm.vocab_size
            lab = labels.reshape(-1) if labels is not None else None
            for i, n in enumerate(mm.orders):
                out.stats[f"match_cov/o{n}"] = hit[i].float().mean()
                if lab is not None:
                    ok = hit[i] & (lab != IGNORE_INDEX)
                    out.stats[f"match_acc/o{n}"] = ((cand[i] == lab) & ok).sum().float() / ok.sum().clamp(min=1)
                if mm.gate is not None:
                    out.stats[f"match_gate_b/o{n}"] = mm.gate.bias[i].detach().float()
        return out

    def init_weights(self):
        """Depth-scaled init as the H-Net (residual writers out_proj/fc2 get std / sqrt(2 * n_layers)).
        Embeddings: N(0, 1) when untied (as the H-Net), N(0, initializer_range) when tied to the lm_head."""
        std = self.config.initializer_range
        nn.init.normal_(self.lm_head.weight, std=std)
        nn.init.normal_(self.embeddings.weight, std=std if self.config.tie_embeddings else 1.0)
        self.backbone.init_linear_weights(std, self.backbone.height)
        if self.match_model is not None:
            self.match_model.reset_parameters()

    def stage_of_param(self, name: str) -> int:
        return 0

    def blocks(self) -> List[nn.Module]:
        return [m for m in self.modules() if isinstance(m, Block)]
