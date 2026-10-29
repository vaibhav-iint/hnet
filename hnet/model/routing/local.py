"""Context-free routers: the boundary score at byte t is a function of the last ``k`` input bytes only, so the same
string is always segmented the same way (like BPE), whatever the context.

Why (logbook, tokenizer statistics): the paper's contextual routers place boundaries by context, and at the second
stage half of the chunks are strings that occur once; a word seen several times is segmented the same way only
23–53% of the time, so the main network sees mostly unique "tokens" and cannot learn identities. A tokenizer needs
*types*, i.e. consistency, which a context-free score gives by construction.

``local_mlp``: ``logit_t = MLP(concat(E[b_{t-k+1}], ..., E[b_t]))`` with its own byte embedding table (small), "end"
semantics (the chunk ends at t). Bytes before the segment start are a padding id. Only ``input_ids`` is used: the
encoder output h is ignored, so the router gets no gradient through h and cannot co-adapt with the encoder.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from .base import RouterOutput, ratio_init_bias, router_output

PAD = 256


@register("router", "local_mlp")
class LocalMLPRouter(nn.Module):
    boundary_semantics = "end"

    def __init__(self, d_model: int, target_ratio: Optional[float] = None, k: int = 6, d_embed: int = 32,
                 d_hidden: int = 128, init_scale: float = 1.5):
        super().__init__()
        self.k = int(k)
        self.embed = nn.Embedding(257, d_embed)  # 256 bytes + PAD
        self.fc1 = nn.Linear(self.k * d_embed, d_hidden)
        self.fc2 = nn.Linear(d_hidden, 1)
        with torch.no_grad():
            nn.init.normal_(self.embed.weight, std=1.0)
            nn.init.normal_(self.fc1.weight, std=(self.k * d_embed) ** -0.5)
            self.fc1.bias.zero_()
            nn.init.normal_(self.fc2.weight, std=init_scale / (0.56 * d_hidden**0.5))
            self.fc2.bias.copy_(ratio_init_bias(target_ratio) - 0.21 * self.fc2.weight.sum())  # meta-device safe
        for p in self.parameters():
            p._no_reinit = True

    def window(self, input_ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """[T, k] the last k bytes ending at each position, PAD before the segment start."""
        T = input_ids.shape[0]
        pos = ctx.position_ids.long()
        cols = []
        for j in range(self.k - 1, -1, -1):  # byte t - j, oldest first
            shifted = torch.cat([input_ids.new_full((j,), PAD), input_ids[: T - j]]) if j else input_ids
            cols.append(torch.where(pos >= j, shifted, torch.full_like(shifted, PAD)))
        return torch.stack(cols, dim=1)

    def forward(self, h: torch.Tensor, ctx: SeqCtx, input_ids: Optional[torch.Tensor] = None, **_) -> RouterOutput:
        assert input_ids is not None, "local_mlp needs input_ids (byte-level stage, or inner ids of the stage)"
        x = self.embed(self.window(input_ids, ctx)).reshape(input_ids.shape[0], -1)
        logit = self.fc2(F.silu(self.fc1(x))).squeeze(-1).float()
        return router_output(torch.sigmoid(logit), ctx, self.boundary_semantics, logit=logit)
