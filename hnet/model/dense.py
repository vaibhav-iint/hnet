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

from hnet.config import DenseConfig
from hnet.registry import register

from .block import Block
from .isotropic import Isotropic
from .lm import CausalLMOutput, lm_outputs, pack_inputs


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
        attn.update(cfg.extra.get("attn", {}))
        mixer_kwargs = {**attn, **cfg.mixer_kwargs}  # attention-style mixers share the head/RoPE settings
        self.embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = Isotropic(
            f"T{cfg.n_layers}", cfg.d_model, cfg.d_intermediate, {"t": cfg.mixer}, {cfg.mixer: mixer_kwargs},
            cfg.norm_eps,
        )
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        labels: Optional[torch.Tensor] = None,
        return_logits: bool = True,
    ) -> CausalLMOutput:
        ids, ctx, batch_shape = pack_inputs(input_ids, cu_seqlens, max_seqlen)
        h = self.backbone(self.embeddings(ids), ctx)
        out = lm_outputs(h, self.lm_head, labels, return_logits, batch_shape)
        out.aux_loss = torch.zeros((), device=ids.device)
        out.stats = {"positions/s0": ids.new_full((), ids.numel(), dtype=torch.float32)}
        return out

    def init_weights(self):
        """Depth-scaled init as the H-Net (residual writers out_proj/fc2 get std / sqrt(2 * n_layers)).
        Embeddings: N(0, 1) when untied (as the H-Net), N(0, initializer_range) when tied to the lm_head."""
        std = self.config.initializer_range
        nn.init.normal_(self.lm_head.weight, std=std)
        nn.init.normal_(self.embeddings.weight, std=std if self.config.tie_embeddings else 1.0)
        self.backbone.init_linear_weights(std, self.backbone.height)

    def stage_of_param(self, name: str) -> int:
        return 0

    def blocks(self) -> List[nn.Module]:
        return [m for m in self.modules() if isinstance(m, Block)]
