"""First-byte chunk refresh (HNetConfig.refresh; evolve g5-compute-01/02). Off by default.

Under "end" boundary semantics the main network's output z_c for the chunk ending at byte b_c is computed before the
next chunk's first byte x_{b_c+1} is known, and the bytes b_c+1 .. b_{c+1}-1 of the next chunk are then predicted by
the byte decoder alone, from a deep state that does not know which word started. In a trained H-Net (g1-wild-01,
mix_v1 val; logbook/evolve/candidates/g5-compute-01/probe.txt) those bytes carry 58% of the bits and 62% of the main
network's (mean-ablation) information, yet the deep state is never refreshed for them. BPE has no such positions: a
token is predicted whole, from one deep state, and there is no "first byte revealed" moment inside it.

The refresh is a query-only extra position per chunk, placed one byte after the chunk end:

    s_c   = zfull_c + W_up RMSNorm(e_{b_c+1})          zfull: the main network's full-width output [M, D1]
                                                       e: the stage-0 encoder output (knows bytes <= b_c+1)
    y     = Iso(s)[:, :D0]                             ``layout`` (e.g. "T1"), stage-1 widths, own final norm;
                                                       causal attention among the refreshed chunks of a segment
    r_c   = g * y_c                                    g [D0]: zero-init gain (the parent's output at init)
    add_t = r_{j(t)} if t is not a boundary else 0     j(t) = plug_back_index (the chunk t reads)

``add`` [T, D0] (fp32) joins the stage-0 decoder input. Causal: r_c uses bytes <= b_c+1 and reaches only bytes
t >= b_c+1 of the region of chunk c (t not a boundary => t > b_c).

``capacity`` < 1 (g5-compute-02): only that fraction of the chunks is refreshed (Iso runs on the packed sub-sequence
of the selected chunks: the FLOPs scale with the capacity). The chunks are ranked by the *surprise* of the revealed
first byte, -log q(x_{b_c+1} | zfull_c), under a low-rank next-byte probe on the DETACHED main output (trained by its
own CE, ``probe_loss``, which reaches the probe only): refresh where the deep state's guess of the next word was
wrong. Training: exactly the top round(capacity * M) chunks of the batch (exact FLOPs). Eval: surprise above a running
threshold (EMA of the training threshold), a per-chunk decision, so packed == separate and decoding is causal.
Chunks whose region is empty or crosses a segment end are never selected (they have no byte to refresh).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import mixer_config_kwargs
from .ctx import SeqCtx
from .isotropic import Isotropic
from .norm import RMSNorm


class ChunkRefresh(nn.Module):
    def __init__(self, cfg, layout: str = "T1", capacity: float = 1.0, probe_rank: int = 32,
                 probe_weight: float = 1.0, momentum: float = 0.99, gain_init: float = 0.0):
        super().__init__()
        assert cfg.n_stages == 1, "refresh: a 1-stage H-Net (the main network is stage 1)"
        assert 0.0 < capacity <= 1.0, capacity
        d0, d1 = cfg.d_model[0], cfg.d_model[1]
        self.d0, self.d1 = d0, d1
        self.capacity = float(capacity)
        self.momentum = float(momentum)
        self.probe_weight = float(probe_weight)
        codes = cfg.mixer_codes_for_stage(1)
        mixer_kwargs = {name: mixer_config_kwargs(name, cfg, 1) for name in set(codes.values())}
        self.norm = RMSNorm(d0, eps=cfg.norm_eps)
        self.up = nn.Linear(d0, d1, bias=False)
        self.layers = Isotropic(layout, d1, cfg.d_intermediate[1], codes, mixer_kwargs, cfg.norm_eps)
        self.gain = nn.Parameter(torch.full((d0,), float(gain_init)))
        self.gain._no_weight_decay = True
        self.probe = None
        if self.capacity < 1.0:
            V = cfg.vocab_size
            self.probe = nn.Sequential(nn.Linear(d1, probe_rank, bias=False), nn.Linear(probe_rank, V))
            self.register_buffer("thr", torch.zeros(()))  # running surprise threshold (eval decisions)
            self.register_buffer("thr_init", torch.zeros((), dtype=torch.bool))

    def reset_parameters(self, std: float):
        nn.init.normal_(self.up.weight, std=std)
        self.layers.init_linear_weights(std, self.layers.height)
        if self.probe is not None:
            nn.init.normal_(self.probe[0].weight, std=std)
            nn.init.normal_(self.probe[1].weight, std=std)
            nn.init.zeros_(self.probe[1].bias)

    # ------------------------------------------------------------------ selection
    def _select(self, zfull: torch.Tensor, nxt: torch.Tensor, valid: torch.Tensor):
        """(sorted chunk indices to refresh [K], probe CE or None, stats). ``valid``: chunks with a region."""
        M = zfull.shape[0]
        stats = {}
        if self.probe is None:
            return torch.arange(M, device=zfull.device), None, stats
        logits = self.probe(zfull.detach().to(self.probe[0].weight.dtype)).float()
        ce = F.cross_entropy(logits, nxt, reduction="none")  # nats; the probe's own loss and the surprise
        probe_loss = (ce * valid).sum() / valid.sum().clamp(min=1) if self.training else None
        sur = torch.where(valid, ce.detach(), torch.full_like(ce, -float("inf")))
        if self.training:
            k = max(1, min(M, int(round(self.capacity * M))))
            top = torch.topk(sur, k).indices
            with torch.no_grad():
                kth = sur[top].min()
                if torch.isfinite(kth):
                    if not bool(self.thr_init):
                        self.thr.copy_(kth)
                        self.thr_init.fill_(True)
                    else:
                        self.thr.mul_(self.momentum).add_((1 - self.momentum) * kth)
            sel = torch.sort(top).values
        else:
            sel = torch.nonzero(sur > self.thr).squeeze(1)
        with torch.no_grad():
            stats["refresh/probe_ce"] = (ce.detach() * valid).sum() / valid.sum().clamp(min=1)
            stats["refresh/thr"] = self.thr.clone()
        return sel, probe_loss, stats

    # ------------------------------------------------------------------ forward
    def forward(self, zfull: torch.Tensor, h_enc: torch.Tensor, input_ids: torch.Tensor, boundary_mask: torch.Tensor,
                ctx: SeqCtx, ctx_inner: SeqCtx):
        """-> (add [T, D0] fp32, probe_loss or None, detached stats)."""
        T, M = h_enc.shape[0], zfull.shape[0]
        pos = ctx_inner.outer_index  # byte position of every chunk end [M]
        qpos = (pos + 1).clamp(max=T - 1)
        seq = ctx.seq_idx.reshape(-1)
        # the region of chunk c is bytes b_c+1 .. b_{c+1}-1: empty if b_c+1 is a boundary, the segment's or batch's end
        valid = (pos + 1 < T) & (seq[qpos] == seq[pos]) & ~boundary_mask.reshape(-1)[qpos]
        nxt = input_ids.reshape(-1)[qpos] if input_ids is not None else torch.zeros_like(pos)
        sel, probe_loss, stats = self._select(zfull, nxt, valid)
        K = int(sel.numel())
        dt = zfull.dtype
        add_inner = torch.zeros(M, self.d0, device=h_enc.device, dtype=torch.float32)
        if K > 0:
            e = self.norm(h_enc.index_select(0, qpos.index_select(0, sel)), out_dtype=h_enc.dtype)
            s = zfull.index_select(0, sel).float() + self.up(e.to(self.up.weight.dtype)).float()
            if K == M:
                sub = ctx_inner
            else:  # packed sub-sequence of the selected chunks (segment structure kept; empty segments allowed)
                seg = ctx_inner.seq_idx.reshape(-1).index_select(0, sel).long()
                counts = torch.bincount(seg, minlength=ctx_inner.n_segments)
                cu = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)
                sub = SeqCtx.from_cu_seqlens(cu, int(counts.max()), K)
            y = self.layers(s.to(dt), sub)[:, : self.d0]
            add_inner = add_inner.index_copy(0, sel, self.gain.float() * y.float())
        j = boundary_mask.reshape(-1).long().cumsum(0) - 1  # plug_back_index
        keep = (~boundary_mask.reshape(-1)).to(torch.float32)[:, None]
        add = add_inner.index_select(0, j.clamp(min=0)) * keep
        with torch.no_grad():
            stats["refresh/frac"] = torch.tensor(K / max(M, 1), device=h_enc.device)
            stats["refresh/gain_rms"] = self.gain.detach().float().pow(2).mean().sqrt()
            stats["refresh/add_rms"] = add.detach().pow(2).mean().sqrt()
            stats["refresh/valid_frac"] = valid.float().mean()
        return add, probe_loss, stats
