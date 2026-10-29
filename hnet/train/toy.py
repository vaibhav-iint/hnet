"""Tiny stand-in model implementing the SPEC model API, for trainer tests only (``model.type: custom``,
``model.factory: hnet.train.toy:build``). Not a research model.

Two fake "stages" (first half of the blocks = stage 0 with embeddings/head, second half = stage 1) so
per-stage LR multipliers and grad norms are exercised; emits H-Net-like ``stats`` and a small aux loss.
Each block mixes the current and previous position (reset at segment starts) through an MLP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from hnet.config import IGNORE_INDEX


@dataclass
class ToyOut:
    logits: torch.Tensor = None
    loss: torch.Tensor = None
    ce_sum: torch.Tensor = None
    n_targets: torch.Tensor = None
    aux_loss: torch.Tensor = None
    stats: dict = field(default_factory=dict)


class ToyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(2 * d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, h, starts):
        x = self.norm(h)
        prev = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
        prev = prev.masked_fill(starts[:, None], 0.0)
        return h + self.fc2(F.gelu(self.fc1(torch.cat([x, prev], -1))))


class ToyLM(nn.Module):
    def __init__(self, vocab=256, d=32, n_layers=2, aux_alpha=0.01, sleep_s=0.0):
        super().__init__()
        self.sleep_s = sleep_s  # artificial delay per training forward (kill/resume tests need a slow run)
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList(ToyBlock(d) for _ in range(n_layers))
        self.norm_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab, bias=False)
        self.gate = nn.Linear(d, 1)
        self.aux_alpha = aux_alpha
        self.n_layers = n_layers

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def set_progress(self, frac: float):
        """Called by the trainer every step with step / total_steps (records the history for tests)."""
        self.progress_log = getattr(self, "progress_log", []) + [frac]

    def stage_of_param(self, name: str) -> int:
        if name.startswith("layers."):
            return 0 if int(name.split(".")[1]) < self.n_layers // 2 else 1
        return 1 if name.startswith("gate") else 0

    def blocks(self):
        return list(self.layers)

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True):
        if self.sleep_s and self.training:
            import time

            time.sleep(self.sleep_s)
        flat = input_ids.reshape(-1)
        T = flat.numel()
        starts = torch.zeros(T, dtype=torch.bool, device=flat.device)
        if cu_seqlens is not None:
            starts[cu_seqlens[:-1].long()] = True
        else:
            starts[0] = True
        h = self.emb(flat)
        for blk in self.layers:
            h = blk(h, starts)
        h = self.norm_f(h)
        p = torch.sigmoid(self.gate(h)).squeeze(-1)
        logits = self.lm_head(h).float()
        out = ToyOut(logits=logits if return_logits else None)
        F_frac = (p > 0.5).float().mean()
        G = p.mean()
        out.aux_loss = self.aux_alpha * (G - 1 / 3) ** 2 * 3
        out.stats = {"F/s0": F_frac.detach(), "G/s0": G.detach(), "compression/s0": torch.tensor(3.0),
                     "aux/s0": out.aux_loss.detach()}
        if labels is not None:
            lab = labels.reshape(-1)
            ce = F.cross_entropy(logits, lab, ignore_index=IGNORE_INDEX, reduction="sum")
            n = (lab != IGNORE_INDEX).sum()
            out.ce_sum, out.n_targets = ce, n
            out.loss = ce / n.clamp_min(1)
        return out


def build(section: dict):
    vocab = int(section.get("vocab_size", 256))
    model = ToyLM(vocab=vocab, d=int(section.get("d_model", 32)), n_layers=int(section.get("n_layers", 2)),
                  sleep_s=float(section.get("sleep_ms", 0)) / 1000.0)
    cfg = SimpleNamespace(vocab_size=vocab, lr_multiplier_n_gpt=4.6)
    return model, cfg
