"""modded-nanogpt-style GPT: the speedrun token baseline S1 ("dense" registry name ``"modded"``).

A port of NanoGPT speedrun record #18 (2025-01-04, "lower logit softcap from 30 to 15", 3.4 min on 8xH100,
1390 steps x 524,288 tokens). See docs/SPEEDRUN.md for why this record and what was kept or dropped.

Architecture (defaults reproduce the record at GPT-2 small size: 12 layers, d=768, 6 heads of 128):

* token embedding, ``x0 = rms_norm(embed(ids))``; every block mixes ``x <- l0 * x + l1 * x0`` (learned, init 1, 0);
* pre-norm blocks with weightless RMSNorm; attention = separate q/k/v/out projections, QK-norm, half-truncated
  RoPE (base period 1024, half of each head's dims rotate, half are position-free), causal document masking plus
  a block-wise sliding window (128-token blocks) that grows from 128 to 1792 tokens during training;
* "value embeddings": ``n_value_embeds`` extra embedding tables whose output is mixed into the attention values
  ``v <- l0 * v + l1 * ve`` of the first and the last ``n_value_embeds`` layers (pattern 012...012);
* layers listed in ``skip_attn_layers`` have no attention (record: layer 7);
* MLP 4x with ReLU^2; the attention output projection, MLP output projection and LM head are zero-initialized;
* U-net skips: the output of encoder layer i (first half) is added, with a learned weight, to the input of
  decoder layer n_dec-1-i;
* untied LM head, logits soft-capped with ``cap * tanh(logits / cap)`` (cap 15), CE in fp32.

Knobs live in ``DenseConfig.extra`` (see ``DEFAULT_EXTRA``); ``d_intermediate`` must be set (record: 4 * d_model).

Packed input follows SPEC: ``input_ids[T]`` with ``cu_seqlens`` (segments never attend to each other; RoPE
positions restart per segment, which is equivalent to the record's absolute positions because RoPE is
relative), or ``input_ids[B, L]`` with one segment per row.

Attention backends: FlexAttention with a hand-built block-sparse mask (record code) on CUDA, dense-mask SDPA on
CPU. The window schedule needs the training progress: trainers call ``model.set_progress(step / total_steps)``
before each step (``getattr(model, "set_progress", None)``); without it the final (largest) window is used, which
is also what the record evaluates with.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from hnet.config import IGNORE_INDEX, DenseConfig
from hnet.model.ctx import SeqCtx
from hnet.model.lm import CE_CHUNK_ELEMENTS, CausalLMOutput
from hnet.registry import register

BLOCK_SIZE = 128  # FlexAttention block size; the sliding window is counted in these blocks

DEFAULT_EXTRA = dict(
    n_value_embeds=3,  # value-embedding tables, used by the first and last n layers
    skip_attn_layers=[7],  # layers without attention
    unet_skips=True,  # U-net skip connections between the two halves
    logit_softcap=15.0,  # 0 disables
    rope_base=1024.0,  # longest RoPE wavelength is 2*pi*rope_base tokens
    rope_frac=0.5,  # fraction of each head's dims that are rotated (rest: no position)
    window_tokens=[128, 1856],  # sliding window at progress 0 and 1; blocks = int(tokens // 128)
    attn_backend="auto",  # auto | flex | sdpa
)


def rms_norm(x: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),), eps=eps)


class HalfRotary(nn.Module):
    """RoPE on a fraction of the head dims ("half-truncated RoPE", record #17); the remaining dims get frequency 0."""

    def __init__(self, head_dim: int, base: float, frac: float):
        super().__init__()
        n_pairs = head_dim // 2
        n_rot = int(round(n_pairs * frac))
        freq = (1.0 / base) ** torch.linspace(0, 1, steps=n_rot, dtype=torch.float32)
        self.register_buffer("freq", torch.cat([freq, freq.new_zeros(n_pairs - n_rot)]), persistent=False)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x [T, H, Dh], pos [T] -> rotated x (pairs are (i, i + Dh/2), as in the record)."""
        theta = pos.to(torch.float32)[:, None] * self.freq[None]
        cos, sin = theta.cos()[:, None, :], theta.sin()[:, None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x2 * cos - x1 * sin), dim=-1).type_as(x)


class AttnMask:
    """Causal + same-segment + block-wise sliding-window mask, built once per forward and shared by all layers.

    Query q may attend to key k iff ``k <= q``, both lie in the same segment and ``q // 128 - k // 128 < n_blocks``.
    ``flex``: a BlockMask built from per-block segment ranges (record code; O((T/128)^2) memory, T % 128 == 0).
    ``sdpa``: a dense [T, T] boolean mask (fallback for CPU and tests).
    """

    def __init__(self, seq_idx: torch.Tensor, n_blocks: torch.Tensor, backend: str):
        self.backend = backend
        T = seq_idx.numel()
        if backend == "sdpa":
            pos = torch.arange(T, device=seq_idx.device)
            blk = pos // BLOCK_SIZE
            self.dense = (
                (pos[:, None] >= pos[None, :])
                & (seq_idx[:, None] == seq_idx[None, :])
                & (blk[:, None] - blk[None, :] < n_blocks)
            )
        else:
            self.block_mask = _flex_block_mask(seq_idx, n_blocks)

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q, k, v [1, H, T, Dh] -> [1, H, T, Dh]."""
        if self.backend == "sdpa":
            return F.scaled_dot_product_attention(q, k, v, attn_mask=self.dense)
        return _flex_attention()(q, k, v, block_mask=self.block_mask)


def _flex_block_mask(seq_idx: torch.Tensor, n_blocks: torch.Tensor):
    from torch.nn.attention.flex_attention import BlockMask

    T = seq_idx.numel()
    assert T % BLOCK_SIZE == 0, "flex backend needs T % 128 == 0 (forward pads)"
    nb = T // BLOCK_SIZE
    docs = seq_idx
    docs_low = docs.view(nb, BLOCK_SIZE)[:, 0].contiguous()
    docs_high = docs.view(nb, BLOCK_SIZE)[:, -1].contiguous()

    def document_causal_window(b, h, q_idx, kv_idx):
        # The window is also enforced by the block structure below (record); repeating it here keeps uncompiled
        # flex_attention, which ignores block sparsity and evaluates only mask_mod, exact.
        in_window = q_idx // BLOCK_SIZE - kv_idx // BLOCK_SIZE < n_blocks
        return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx]) & in_window

    def dense_to_ordered(dense: torch.Tensor):
        num = dense.sum(dim=-1, dtype=torch.int32)
        idx = dense.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
        return num[None, None].contiguous(), idx[None, None].contiguous()

    blk = torch.arange(nb, dtype=torch.int32, device=docs.device)
    q_blk, kv_blk = blk[:, None], blk[None, :]
    window = q_blk - kv_blk < n_blocks
    any_doc = (docs_low[:, None] <= docs_high[None, :]) & (docs_low[None, :] <= docs_high[:, None])
    one_doc = (docs_low[:, None] == docs_high[None, :]) & (docs_low[None, :] == docs_high[:, None])
    partial = (q_blk >= kv_blk) & window & any_doc
    full = (q_blk > kv_blk) & window & one_doc  # whole block visible: no mask_mod evaluation needed
    kv_num, kv_idx = dense_to_ordered(partial & ~full)
    full_num, full_idx = dense_to_ordered(full)
    return BlockMask.from_kv_blocks(kv_num, kv_idx, full_num, full_idx, BLOCK_SIZE=BLOCK_SIZE, mask_mod=document_causal_window)


_FLEX = None


def _flex_attention():
    """flex_attention, compiled once on first use (uncompiled it runs a slow O(T^2) reference)."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX = torch.compile(flex_attention, dynamic=False) if torch.cuda.is_available() else flex_attention
    return _FLEX


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, rotary: HalfRotary, eps: float):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads, self.eps, self.rotary = num_heads, eps, rotary
        self.c_q = nn.Linear(dim, dim, bias=False)
        self.c_k = nn.Linear(dim, dim, bias=False)
        self.c_v = nn.Linear(dim, dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))  # weights of v and of the value embedding

    def forward(self, x, ve, mask: AttnMask, pos):
        T, H = x.size(0), self.num_heads
        q, k, v = self.c_q(x).view(T, H, -1), self.c_k(x).view(T, H, -1), self.c_v(x).view(T, H, -1)
        dt = q.dtype
        v = self.lambdas[0] * v if ve is None else self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        q, k = rms_norm(q, self.eps), rms_norm(k, self.eps)
        q, k = self.rotary(q, pos), self.rotary(k, pos)
        y = mask.attend(*(t.to(dt).transpose(0, 1).unsqueeze(0) for t in (q, k, v)))
        return self.c_proj(y.squeeze(0).transpose(0, 1).reshape(T, -1))


class ReluSquaredMLP(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.c_fc = nn.Linear(dim, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class ModdedBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden, rotary, eps, use_attn: bool):
        super().__init__()
        self.eps = eps
        self.attn = CausalSelfAttention(dim, num_heads, rotary, eps) if use_attn else None
        self.mlp = ReluSquaredMLP(dim, hidden)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))  # weights of x and of x0

    def forward(self, x, x0, ve, mask: AttnMask, pos):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(rms_norm(x, self.eps), ve, mask, pos)
        return x + self.mlp(rms_norm(x, self.eps))


@register("dense", "modded")
class ModdedGPT(nn.Module):
    def __init__(self, cfg: DenseConfig):
        super().__init__()
        unknown = set(cfg.extra) - set(DEFAULT_EXTRA)
        assert not unknown, f"unknown modded extra keys: {sorted(unknown)}"
        x = {**DEFAULT_EXTRA, **cfg.extra}
        assert not cfg.tie_embeddings, "the modded variant uses an untied (zero-initialized) LM head"
        assert cfg.num_kv_heads in (None, cfg.num_heads) and cfg.nope_every == 0, "modded uses MHA with RoPE everywhere"
        assert x["attn_backend"] in ("auto", "flex", "sdpa")
        n_ve, L = int(x["n_value_embeds"]), cfg.n_layers
        assert 2 * n_ve <= L, "value embeddings go to the first and last n_value_embeds layers"
        self.cfg, self.extra = cfg, x
        self.eps = cfg.norm_eps
        self.softcap = float(x["logit_softcap"])

        D = cfg.d_model
        rotary = HalfRotary(D // cfg.num_heads, x["rope_base"], x["rope_frac"])
        self.embed = nn.Embedding(cfg.vocab_size, D)
        self.value_embeds = nn.ModuleList([nn.Embedding(cfg.vocab_size, D) for _ in range(n_ve)])
        skip_attn = set(x["skip_attn_layers"])
        self.layers = nn.ModuleList(
            [ModdedBlock(D, cfg.num_heads, cfg.d_intermediate, rotary, self.eps, i not in skip_attn) for i in range(L)]
        )
        self.lm_head = nn.Linear(D, cfg.vocab_size, bias=False)
        # U-net skips: layer i < n_skip saves its output, layer L - 1 - i adds it back (weighted) to its input.
        self.n_skip = L // 2 if x["unet_skips"] else 0
        self.skip_weights = nn.Parameter(torch.ones(self.n_skip)) if self.n_skip else None
        # Value-embedding table used by each layer (None = none): 0..n-1 at the start, 0..n-1 again at the end.
        self.ve_index = [i if i < n_ve else (i - (L - n_ve) if i >= L - n_ve else None) for i in range(L)]
        # Sliding window in 128-token blocks; a tensor so that changing it does not recompile under torch.compile.
        self.register_buffer("window_blocks", torch.tensor(0, dtype=torch.int32), persistent=False)
        self.set_progress(1.0)
        self.init_weights()

    # ------------------------------------------------------------------ SPEC API
    def init_weights(self):
        """The record's init: PyTorch defaults (Linear: U(+-1/sqrt(fan_in)); Embedding: N(0, 1)), zero output
        projections and LM head, mixing scalars at their identity values."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                m.reset_parameters()
        for b in self.layers:
            nn.init.zeros_(b.mlp.c_proj.weight)
            b.lambdas.data.copy_(torch.tensor([1.0, 0.0]))
            if b.attn is not None:
                nn.init.zeros_(b.attn.c_proj.weight)
                b.attn.lambdas.data.fill_(0.5)
        nn.init.zeros_(self.lm_head.weight)
        if self.skip_weights is not None:
            nn.init.ones_(self.skip_weights)

    def set_progress(self, progress: float):
        """Set the training progress in [0, 1]; grows the attention window as in the record."""
        lo, hi = self.extra["window_tokens"]
        p = min(max(float(progress), 0.0), 1.0)
        self.window_blocks.fill_(max(1, int(((1 - p) * lo + p * hi) // BLOCK_SIZE)))

    def stage_of_param(self, name: str) -> int:
        return 0

    def blocks(self) -> list:
        return list(self.layers)

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True) -> CausalLMOutput:
        """SPEC forward. With ``return_logits=False`` (the trainer) the CE is computed in checkpointed chunks of
        ~CE_CHUNK_ELEMENTS logits, so the fp32 [T, 50304] logits of a 64K-token window are never materialized."""
        shape = input_ids.shape
        if input_ids.dim() == 2:  # [B, L]: one segment per row
            B, L = shape
            cu_seqlens = torch.arange(B + 1, device=input_ids.device, dtype=torch.int32) * L
            max_seqlen = L
        elif cu_seqlens is None:
            cu_seqlens = torch.tensor([0, shape[0]], device=input_ids.device, dtype=torch.int32)
        ids = input_ids.reshape(-1)
        h = self._backbone(ids, SeqCtx.from_cu_seqlens(cu_seqlens, max_seqlen))

        out = CausalLMOutput(aux_loss=torch.zeros((), device=ids.device))
        logits = self._logits(h) if return_logits else None
        if labels is not None:
            labels = labels.reshape(-1)
            if logits is not None:
                out.ce_sum = F.cross_entropy(logits, labels, ignore_index=IGNORE_INDEX, reduction="sum")
            else:
                rows = max(1, CE_CHUNK_ELEMENTS // self.cfg.vocab_size)
                out.ce_sum = sum(checkpoint(self._ce_sum, hs, ys, use_reentrant=False)
                                 for hs, ys in zip(h.split(rows), labels.split(rows)))
            out.n_targets = (labels != IGNORE_INDEX).sum()
            out.loss = out.ce_sum / out.n_targets.clamp_min(1)
        if logits is not None:
            out.logits = logits.view(*shape, -1)
        return out

    # ------------------------------------------------------------------ internals
    def _logits(self, h: torch.Tensor) -> torch.Tensor:
        """Soft-capped fp32 logits."""
        logits = self.lm_head(h).float()
        return self.softcap * torch.tanh(logits / self.softcap) if self.softcap > 0 else logits

    def _ce_sum(self, h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self._logits(h), labels, ignore_index=IGNORE_INDEX, reduction="sum")

    def _backend(self, device) -> str:
        b = self.extra["attn_backend"]
        return ("flex" if device.type == "cuda" else "sdpa") if b == "auto" else b

    def _backbone(self, ids: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """ids [T] -> final hidden states [T, D]."""
        T = ids.numel()
        seq_idx, pos = ctx.seq_idx[0], ctx.position_ids
        backend = self._backend(ids.device)
        pad = (-T) % BLOCK_SIZE if backend == "flex" else 0
        if pad:  # padding positions form their own segment and are dropped at the end
            ids = F.pad(ids, (0, pad))
            seq_idx = F.pad(seq_idx, (0, pad), value=ctx.n_segments)
            pos = F.pad(pos, (0, pad))
        mask = AttnMask(seq_idx, self.window_blocks, backend)

        x0 = rms_norm(self.embed(ids), self.eps)
        ves = [emb(ids) for emb in self.value_embeds]
        x, skips = x0, []
        first_dec = len(self.layers) - self.n_skip
        for i, block in enumerate(self.layers):
            if i >= first_dec:
                x = x + self.skip_weights[i - first_dec] * skips.pop()
            ve = ves[self.ve_index[i]] if self.ve_index[i] is not None else None
            x = block(x, x0, ve, mask, pos)
            if i < self.n_skip:
                skips.append(x)
        return rms_norm(x[:T], self.eps)
