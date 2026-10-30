"""Product-key memory layer in the main network (HNetConfig.main_memory; DenseConfig.memory for the BPE control).

Why (evolve g5-output-01): the H-Net's bits concentrate on the first byte of a chunk (on the g1-wild-01 checkpoint:
24% of the bytes, 42% of the bits, 2.7 bits each), and offline probes on that checkpoint found them limited neither by
the readout (a linear or MLP correction head on the frozen main-network output and decoder state, fitted on training
windows, gains <= 0.056 bits per chunk-start byte, <= 0.013 bpb) nor by in-context copying (a chunk-keyed match at
chunk starts: +0.0005 bpb) nor by n-gram statistics (count-based output memories keyed by the previous chunk or by
the last 4-12 bytes: <= 0.002 bpb). What predicts the next chunk is the main network's state, the only part of the
model that runs at chunk level, and it has 3.2M of the 20.0M parameters: the FLOP-free tables (15M) all sit at the
byte level (n-gram tables, chunk table, trie). A product-key memory (Lample et al. 2019; "Memory Layers at Scale",
Berges et al. 2024) puts lookup parameters inside the main network at a cost of a few dot products per chunk.

For an input x [N, d] (the main network's residual stream before layer ``layer``):

    q        = W_q (RMSNorm(x) - mu)               [N, 2 * k_dim], split into q1, q2 (k_dim each)
    s1, s2   = q1 K1^T, q2 K2^T                     sub-key scores [N, n_keys] each
    top-k of s1 and of s2, the k x k sums s1[i] + s2[j], their top-k -> slots i * n_keys + j, scores
    out      = sum_slots softmax(scores) * V[slot]  V: n_keys^2 x d values (an nn.Embedding: Adam, like the tables)

and the stream continues as x + out. mu is a running mean (EMA, momentum ``center``) of RMSNorm(x) over earlier
training steps, a persistent buffer: the residual stream shares a large common direction (mean cosine 0.29 between
chunk states of the trained g1-wild-01 main network), and without centering a fresh memory reads 4510 distinct slots
of 9216 over 3773 chunks x 32 reads on that stream, 8119 with it (Lample et al. use BatchNorm on the query for the same
reason; the EMA keeps the forward pass free of within-batch coupling). Values are zero-initialized, so a fresh model computes exactly what the model
without the memory computes; the LM loss reaches V from step 0 (softmax weights > 0), W_q and K once V != 0.
Causal and packing-invariant: position-wise (no mixing across positions). Cost per position (hnet.flops, component
``main_memory``): 2 d (2 k_dim) for the query, 2 * 2 n_keys k_dim for the sub-key scores, k^2 sums, softmax over k,
2 k d for the weighted value sum. Parameters: n_keys^2 d + 2 n_keys k_dim + 2 k_dim d + d.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class ProductKeyMemory(nn.Module):
    def __init__(self, d_model: int, n_keys: int = 96, k_dim: int = 32, topk: int = 32, layer: int = 2,
                 center: float = 0.99, norm_eps: float = 1e-5):
        super().__init__()
        self.d_model, self.n_keys, self.k_dim, self.topk = int(d_model), int(n_keys), int(k_dim), int(topk)
        self.layer = int(layer)
        assert 1 <= self.topk <= self.n_keys, (topk, n_keys)
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.query = nn.Linear(d_model, 2 * self.k_dim, bias=False)
        self.keys = nn.Parameter(torch.empty(2, self.n_keys, self.k_dim))
        self.values = nn.Embedding(self.n_keys ** 2, d_model)
        # running mean of the normalized input, subtracted before the query projection (``center`` = EMA momentum,
        # 0 = off). Persistent: it shapes which slots are read, so a resume must not reset it.
        self.center = float(center)
        assert 0.0 <= self.center < 1.0, center
        self.register_buffer("x_mean", torch.zeros(d_model), persistent=self.center > 0)
        self.last_stats = {}
        self.reset_parameters()

    def reset_parameters(self):
        self.norm.reset_parameters()
        nn.init.normal_(self.query.weight, std=self.d_model ** -0.5)
        b = self.k_dim ** -0.5
        nn.init.uniform_(self.keys, -b, b)
        nn.init.zeros_(self.values.weight)
        self.x_mean.zero_()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def flops_per_position(d: int, n_keys: int, k_dim: int, topk: int, elementwise: bool = True,
                           center: bool = True) -> float:
        f = 2 * d * 2 * k_dim + 2 * 2 * n_keys * k_dim + 2 * topk * d
        if elementwise:
            f += topk * topk + 3 * topk + (d if center else 0)
        return float(f)

    def lookup(self, x: torch.Tensor):
        """x [N, d] -> (slot ids [N, k] int64, softmax weights [N, k] fp32)."""
        xn = self.norm(x, out_dtype=torch.float32)
        if self.center > 0:
            mu = self.x_mean.clone()  # this step reads the mean of earlier steps only: no coupling inside a batch
            if self.training:
                with torch.no_grad():
                    self.x_mean.mul_(self.center).add_((1 - self.center) * xn.detach().mean(0))
            xn = xn - mu
        q = self.query(xn).float().view(-1, 2, self.k_dim)
        keys = self.keys.float()
        s1 = q[:, 0] @ keys[0].T  # [N, n]
        s2 = q[:, 1] @ keys[1].T
        k = self.topk
        t1, i1 = s1.topk(k, dim=-1)
        t2, i2 = s2.topk(k, dim=-1)
        sums = (t1[:, :, None] + t2[:, None, :]).reshape(-1, k * k)
        slots = (i1[:, :, None] * self.n_keys + i2[:, None, :]).reshape(-1, k * k)
        top, j = sums.topk(k, dim=-1)
        return slots.gather(1, j), torch.softmax(top, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., d] (any float dtype; the fp32 residual stream in the Isotropic stack) -> [..., d] fp32."""
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        idx, w = self.lookup(xf)
        out = F.embedding_bag(idx, self.values.weight.float(), per_sample_weights=w, mode="sum")
        with torch.no_grad():  # mechanism metrics: how much of the memory is addressed, how peaked the read is
            n_slots = self.n_keys ** 2
            used = torch.zeros(n_slots, device=idx.device, dtype=torch.bool)
            used[idx.reshape(-1)] = True
            self.last_stats = {
                "memory/slots_used": used.float().mean(),  # share of the n^2 slots read in this batch
                "memory/top_w": w[:, 0].mean() if w.shape[1] > 0 else w.new_zeros(()),  # weight of the best slot
                "memory/out_rms": out.detach().pow(2).mean().sqrt(),
                "memory/in_rms": xf.detach().float().pow(2).mean().sqrt(),
            }
        return out.reshape(shape)
