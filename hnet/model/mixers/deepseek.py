"""DeepSeek attention variants as registered sequence mixers (WS-I1): ``mla``, ``nsa``, ``dsa``.

All three take ``x [1, T, D]`` packed and a :class:`~hnet.model.ctx.SeqCtx`, and return ``[1, T, D]``. They honour
document masking (no attention across segments of ``ctx.cu_seqlens``; RoPE positions and every block structure
restart at each segment), accept the same attention keys as ``attn`` (``num_heads, num_kv_heads, rotary_emb_dim,
rope_theta, window_size, nope_every, qkv_bias, out_bias, softmax_scale, layer_idx``) plus their own, and provide
``config_kwargs(cfg, stage)`` (starts from ``cfg.attn_for_stage(stage)``) and ``flops_per_position(d_model, L, cv,
**kwargs)`` (forward FLOPs per position; ``hnet.flops`` multiplies by ``1 + backward_factor`` exactly as for
``attn_flops``). Use them per stage (``mixer_codes: [{m: mamba2, t: attn}, {m: mamba2, t: attn}, {m: mamba2,
t: mla}]`` plus ``mixer_cfg: {mla: {kv_lora_rank: 64}}``) or in the dense baseline (``mixer: mla``,
``mixer_kwargs: {...}``). Full write-up, tiny-scale settings and deviations: ``docs/ATTENTION_DEEPSEEK.md``.

Notation: T packed positions, H query heads, Hk key/value heads (GQA groups), G = H / Hk, t a query position and s
a key position *within their segment*.

``mla``: Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434, Sec. 2.1.2-2.1.3, Eqs. 9-19 / App. C Eqs.
37-47; DeepSeek-V3, arXiv 2412.19437, Sec. 2.1.1, Eqs. 1-11, which adds RMSNorm on both latents)::

    c^Q_t  = RMSNorm(W^DQ h_t)                       (q_lora_rank > 0; else q_t = W^Q h_t, full-rank query)
    q^C_t  = W^UQ c^Q_t,  q^R_t = RoPE(W^QR c^Q_t)  (per head: qk_nope_head_dim + qk_rope_head_dim)
    c^KV_t = RMSNorm(W^DKV h_t)                      (kv_lora_rank: the only per-token cache besides k^R)
    k^C_t  = W^UK c^KV_t,  v^C_t = W^UV c^KV_t       (per head: qk_nope_head_dim, v_head_dim)
    k^R_t  = RoPE(W^KR h_t)                           (one decoupled RoPE key shared by all heads)
    o_t,i  = sum_{s<=t} softmax_s( [q^C;q^R]_t,i . [k^C_s,i; k^R_s] / sqrt(d_nope + d_rope) ) v^C_s,i
    u_t    = W^O [o_t,1; ...; o_t,H]

This is the training-time (non-absorbed) form: keys/values are materialized per head and fed to the standard
flash / flex / SDPA kernels of ``attn`` (flash pads v to the qk head dim when they differ). Absorbed inference form
(not implemented, decoding only): ``q^C_t,i . k^C_s,i = (W^UK_i^T q^C_t,i) . c^KV_s`` and ``W^O_i W^UV_i c^KV``,
so W^UK folds into the query side and W^UV into W^O; each query head then attends over the shared cache
``[c^KV_s ; k^R_s]`` of size kv_lora_rank + qk_rope_head_dim per token (MQA over a latent, "MQA mode" in the V3.2
report). Fused weights ``wkv_a`` = [W^DKV; W^KR], ``wkv_b`` = [W^UK; W^UV] per head (V3 inference naming).
Defaults (paper ratios, d_h = d_model / num_heads): qk_nope_head_dim = d_h, qk_rope_head_dim = rotary_emb_dim or
d_h / 2, v_head_dim = d_h, kv_lora_rank = 4 d_h (V2: d_c = 4 d_h), q_lora_rank = 0. ``num_kv_heads`` is accepted
and ignored (MLA's cache saving comes from the latent, not from KV-head sharing). NoPE layers (``nope_every``) keep
the rope channels (so the parameter count does not depend on depth) but do not rotate them.

``nsa``: Native Sparse Attention (Yuan et al., arXiv 2502.11089, Sec. 3): per query three branches, Eq. (5)::

    o*_t = g^cmp_t Attn(q_t, K^cmp_t, V^cmp_t) + g^slc_t Attn(q_t, K^slc_t, V^slc_t) + g^win_t Attn(q_t, K^win_t, V^win_t)

* compression, Eq. (7): blocks of ``compress_block`` (l) keys with stride ``compress_stride`` (d); block i covers
  s in [i d, i d + l) and is visible to t iff i d + l - 1 <= t (complete at or before t; queries with t < l - 1 see
  none and that branch contributes 0). phi = mean of the block + MLP(vec(block + intra-block position embedding)),
  separately for keys and values and shared by the KV heads;
* selection, Eqs. (8)-(12): p^cmp_t = softmax(q_t K^cmp_t) (the compression branch's own probabilities), mapped to
  selection blocks of ``select_block`` (l') by spatial overlap (Eq. 9), summed over the G heads of a GQA group
  (Eq. 10), top-``select_n`` blocks per group (Eq. 11; ties go to the more recent block) with the first
  ``n_init_blocks`` and the ``n_local_blocks`` most recent blocks (the current, partially filled one included)
  always selected; tokens of the chosen blocks are attended causally (s <= t), Eq. (12);
* sliding window: keys s in [t - sliding_window, t];
* gates g^c_t = sigmoid(W_g h_t) per head and branch; by default independent K/V projections per branch
  (``separate_kv``) as in the paper.

Small-scale defaults (contexts ~200-2000 positions; the paper uses l=32, d=16, l'=64, n=16, w=512 at 8k-64k):
l = 8, d = 4, l' = 8, n = 8 (1 initial + 2 local + 5 ranked; 64 selected keys), w = 32. Selection is therefore
active (strictly sparse) for every query with t >= 64. Eq. (9) read literally sums p^cmp over indices
``l'/d j - m - n``, i.e. compression blocks that mostly precede selection block j; we use the overlap form the text
describes, ``p^slc[j] = sum_i p^cmp[i] |[i d, i d + l) ∩ [j l', (j+1) l')| / d`` (equal multiplicities when d | l,
d | l'). A global ``window_size >= 0`` (the shared attention key) additionally limits every branch to keys in
[t - window_size, t].

``dsa``: DeepSeek Sparse Attention (DeepSeek-V3.2-Exp tech report, Sec. 1, Eqs. 1-2; training Sec. 2.1, Eqs. 3-4;
reference code deepseek-ai/DeepSeek-V3.2-Exp ``inference/model.py`` class ``Indexer``)::

    I_t,s = sum_j w^I_t,j ReLU(q^I_t,j . k^I_s)          (Eq. 1; H^I small heads, w^I scaled by (H^I d^I)^-1/2)
    u_t   = Attn(h_t, {c_s : I_t,s in Top-k(I_t,:)})      (Eq. 2; causal, same segment, shared by all heads;
                                                            ties go to the more recent key)
    L^I   = sum_t KL(p_t,S_t || softmax(I_t,S_t))          (Eq. 4; p_t = L1-normalized sum over heads of the main
                                                            attention probabilities, S_t the selected set)

The base attention is MLA (``base="mla"``, as V3.2) or plain GQA attention (``base="attn"``); the indexer query
comes from the MLA query latent c^Q when q_lora_rank > 0 (as V3.2), else from h_t; its key is
LayerNorm(W^K_I h_s), RoPE on the first ``index_rope_dim`` channels of q^I and k^I. Following the paper the
indexer input is detached (``detach_indexer_input``): the indexer learns only from L^I and the main model only from
the LM loss. Small-scale defaults: H^I = 4, d^I = 32, k = ``index_topk`` = 64.

**Indexer training signal.** Each training forward (grad enabled, ``index_loss != "off"``) stores the KL term
(per-token mean of Eq. 4 times ``index_loss_coef``) as ``mixer.aux_loss`` and its detached value as
``mixer.index_kl``. The trainer adds only ``out.aux_loss`` of the model (``hnet/model/lm.py``, ``dense.py``), which
does not know about mixer losses, so by default (``index_loss="inject"``) the gradient of the KL is injected into
the backward pass by an identity autograd function on the mixer output: the indexer receives exactly
d(coef * L^I)/d(theta_I) whenever the LM loss is back-propagated, without any trainer change. Since the indexer
input is detached, no other parameter is affected. Caveat: the injected gradient is not divided by grad
accumulation or multiplied by an fp16 loss scale (bf16 needs none); with Adam this only rescales the indexer's
share of the global grad norm (set ``index_loss_coef = 1 / grad_accum`` to match exactly). With
``index_loss="attr"`` nothing is injected and the caller must add ``mixer_aux_loss(model)`` (sum of every mixer's
``aux_loss``) to the loss it back-propagates, scaled like ``out.aux_loss``.

Precision: logits, index scores and softmaxes are kept in fp32 (``.float()`` after each einsum, which bf16
autocast would otherwise downcast); probabilities @ values follow the autocast dtype.

Backends. ``mla`` uses ``attn.attention`` (flash / flex / SDPA, chosen automatically). ``nsa``'s window branch uses
it too; its compression branch is pure PyTorch on a padded per-segment layout (it needs the probabilities); its
selection branch and ``dsa``'s top-k attention use either ``"gather"`` (pure PyTorch reference: gathers the
selected keys per query, O(T x selected x Hk x head_dim) memory) or ``"flex"`` (``flex_attention`` with a
data-dependent ``mask_mod``; compiled on CUDA, eager on CPU). Auto: flex on CUDA, gather otherwise; override with
``FORCE_SPARSE_BACKEND`` / env ``HNET_SPARSE_BACKEND``. The DSA indexer is scored in chunks of queries (a Python
loop over query chunks; keys limited to the last ``max_seqlen`` positions) to bound memory.
"""

from __future__ import annotations

import functools
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from ..norm import RMSNorm
from ..rotary import apply_rotary, rotary_cos_sin
from . import attn as _attn
from .attn import CausalAttention

SPARSE_BACKENDS = ("gather", "flex")
FORCE_SPARSE_BACKEND: Optional[str] = os.environ.get("HNET_SPARSE_BACKEND") or None
INDEX_CHUNK_ELEMS = 1 << 24  # max fp32 elements of one DSA indexer score chunk (64 MB)


def select_sparse_backend(q: torch.Tensor) -> str:
    if FORCE_SPARSE_BACKEND is not None:
        assert FORCE_SPARSE_BACKEND in SPARSE_BACKENDS, FORCE_SPARSE_BACKEND
        return FORCE_SPARSE_BACKEND
    if q.is_cuda and _attn.flex_attention is not None:
        return "flex"
    return "gather"


def mixer_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum of the ``aux_loss`` of every mixer that has one (DSA indexer KL); for ``index_loss="attr"``."""
    terms = [m.aux_loss for m in model.modules() if isinstance(getattr(m, "aux_loss", None), torch.Tensor)]
    if not terms:
        return torch.zeros(())
    return torch.stack([t.float() for t in terms]).sum()


# ----------------------------------------------------------------------------- shared helpers
def _is_nope(layer_idx: int, nope_every: int) -> bool:
    return bool(nope_every) and (layer_idx + 1) % nope_every == 0


def _layout(ctx: SeqCtx):
    """(seg [T], pos [T], starts [B], B, Lmax): flat <-> padded per-segment indexing, memoized in ``ctx.cache``."""
    key = ("deepseek_layout",)
    hit = ctx.cache.get(key)
    if hit is None:
        hit = (
            ctx.seq_idx[0].long(),
            ctx.position_ids.long(),
            ctx.cu_seqlens[:-1].long(),
            ctx.n_segments,
            max(int(ctx.max_seqlen), 1),
        )
        ctx.cache[key] = hit
    return hit


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over the last dim restricted to ``mask``; rows with no allowed entry give all zeros (no NaN)."""
    scores = scores.masked_fill(~mask, float("-inf"))
    m = scores.amax(-1, keepdim=True)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m)).detach()
    p = (scores - m).exp()
    return p / p.sum(-1, keepdim=True).clamp_min(torch.finfo(p.dtype).tiny)


def _topk_recent(score: torch.Tensor, k: int):
    """Top-k over the last dim; ties go to the *later* index (most recent key / block), so the choice does not
    depend on how documents are packed (ReLU index scores tie at exactly 0). Returns (values, indices)."""
    n = score.shape[-1]
    vals, order = torch.sort(score.flip(-1), dim=-1, descending=True, stable=True)
    return vals[..., :k], (n - 1) - order[..., :k]


def _dense_attention(q, k, v, ctx: SeqCtx, window: int, scale: float) -> torch.Tensor:
    """``attn.attention`` with automatic backend; flash-attn needs equal qk / v head dims, so pad the smaller."""
    backend = _attn.select_backend(q)
    dq, dv = q.shape[-1], v.shape[-1]
    if backend == "flash" and dq != dv:
        D = max(dq, dv)
        q, k, v = F.pad(q, (0, D - dq)), F.pad(k, (0, D - dq)), F.pad(v, (0, D - dv))
        return _attn.attention(q, k, v, ctx, window, scale, backend)[..., :dv]
    return _attn.attention(q, k, v, ctx, window, scale, backend)


def _gather_attend(q, k, v, idx, valid, scale: float):
    """Sparse attention by gathering. q [T,H,dq], k [T,Hk,dq], v [T,Hk,dv], idx/valid [T,Hk,S] (flat key index per
    query and KV head). Returns (out [T,H,dv] in v.dtype, probs [T,H,S] fp32)."""
    T, H, dq = q.shape
    Hk, S = k.shape[1], idx.shape[-1]
    heads = torch.arange(Hk, device=q.device)[None, :, None]
    kg, vg = k[idx, heads], v[idx, heads]  # [T, Hk, S, d]
    s = torch.einsum("tagd,tasd->tags", q.view(T, Hk, H // Hk, dq).float(), kg.float()).float() * scale
    p = _masked_softmax(s, valid[:, :, None, :])
    o = torch.einsum("tags,tasd->tagd", p, vg.float())
    return o.reshape(T, H, -1).to(v.dtype), p.reshape(T, H, S)


def _flex_attend(q, k, v, mask_mod, scale: float, per_head: bool) -> torch.Tensor:
    """flex_attention over the packed sequence with a data-dependent ``mask_mod`` (rebuilt every call)."""
    if _attn.flex_attention is None:
        raise RuntimeError("flex backend requested but torch.nn.attention.flex_attention is unavailable")
    T, H = q.shape[0], q.shape[1]
    bm = _attn.packed_block_mask(mask_mod, H if per_head else None, T, q.device)
    fn = _attn._flex_fn() if q.is_cuda else _attn.flex_attention
    qt, kt, vt = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
    out = fn(qt, kt, vt, block_mask=bm, scale=scale, enable_gqa=k.shape[1] != H)
    return out.squeeze(0).transpose(0, 1)


class _InjectAuxGrad(torch.autograd.Function):
    """Identity on ``out``; in backward also sends d(aux)=1 into ``aux``'s graph (adds grad of aux to its params)."""

    @staticmethod
    def forward(fctx, out, aux):
        fctx.aux_dtype = aux.dtype
        return out.view_as(out)

    @staticmethod
    def backward(fctx, grad):
        return grad, torch.ones((), dtype=fctx.aux_dtype, device=grad.device)


# ----------------------------------------------------------------------------- FLOPs helpers
def _avg_ctx(N: int, f, cv) -> float:
    """Average of f(t) over t in [0, N) (causal) or f(N - 1) ("full" convention: every query sees the full L)."""
    if cv.attn == "full":
        return float(f(N - 1))
    if cv.attn != "causal":
        raise ValueError(cv.attn)
    return sum(f(t) for t in range(N)) / N


def _n_positions(L: float) -> int:
    return max(1, int(round(L)))


def _win_cap(window: int, cv) -> float:
    return math.inf if (window is None or window < 0 or not cv.attn_windows) else window + 1


def _quad(ctx_keys: float, heads: int, dqk: int, dv: int, cv) -> float:
    """Logits 2 ctx H dqk + probs@V 2 ctx H dv (+ softmax 3 H ctx), times the convention's quadratic scale."""
    f = 2 * ctx_keys * heads * dqk + 2 * ctx_keys * heads * dv
    if cv.count_elementwise:
        f += 3 * heads * ctx_keys
    return f * cv.attn_quadratic_scale


def _meta(cls, d_model, kwargs):
    with torch.device("meta"):
        return cls(d_model, **kwargs)


# ============================================================================= MLA
@register("mixer", "mla")
class MultiHeadLatentAttention(nn.Module):
    """Multi-head Latent Attention (see module docstring). x [1,T,D] -> [1,T,D]."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,  # accepted for interface compatibility; MLA has no KV-head grouping
        rotary_emb_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        window_size: int = -1,
        nope_every: int = 0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        softmax_scale: Optional[float] = None,
        layer_idx: int = 0,
        kv_lora_rank: Optional[int] = None,
        q_lora_rank: int = 0,
        qk_nope_head_dim: Optional[int] = None,
        qk_rope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        hd = d_model // num_heads
        self.d_model, self.num_heads, self.layer_idx = d_model, num_heads, layer_idx
        self.qk_nope_head_dim = hd if qk_nope_head_dim is None else int(qk_nope_head_dim)
        rope = (hd // 2 if rotary_emb_dim is None else int(rotary_emb_dim)) if qk_rope_head_dim is None else int(
            qk_rope_head_dim)
        assert rope % 2 == 0, f"qk_rope_head_dim must be even, got {rope}"
        self.qk_rope_head_dim = rope
        self.qk_head_dim = self.qk_nope_head_dim + rope
        assert self.qk_head_dim > 0
        self.v_head_dim = hd if v_head_dim is None else int(v_head_dim)
        self.kv_lora_rank = 4 * hd if kv_lora_rank is None else int(kv_lora_rank)
        self.q_lora_rank = int(q_lora_rank or 0)
        self.window_size = int(window_size)
        self.rope_theta = float(rope_theta)
        self.use_rope = rope > 0 and not _is_nope(layer_idx, nope_every)
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.qk_head_dim ** -0.5
        H = num_heads
        if self.q_lora_rank > 0:
            self.wq_a = nn.Linear(d_model, self.q_lora_rank, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, eps=norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, H * self.qk_head_dim, bias=qkv_bias)
        else:
            self.wq = nn.Linear(d_model, H * self.qk_head_dim, bias=qkv_bias)
        self.wkv_a = nn.Linear(d_model, self.kv_lora_rank + rope, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=norm_eps)
        self.wkv_b = nn.Linear(self.kv_lora_rank, H * (self.qk_nope_head_dim + self.v_head_dim), bias=qkv_bias)
        self.out_proj = nn.Linear(H * self.v_head_dim, d_model, bias=out_bias)

    @classmethod
    def config_kwargs(cls, cfg, stage: int) -> dict:
        return cfg.attn_for_stage(stage)

    def project(self, x2: torch.Tensor, ctx: SeqCtx):
        """x2 [T, D] -> (q [T,H,dqk], k [T,H,dqk], v [T,H,dv], query latent c^Q [T, q_lora_rank] or None)."""
        T, H = x2.shape[0], self.num_heads
        nope, rope = self.qk_nope_head_dim, self.qk_rope_head_dim
        if self.q_lora_rank > 0:
            cq = self.q_norm(self.wq_a(x2))
            q = self.wq_b(cq)
        else:
            cq, q = None, self.wq(x2)
        q = q.view(T, H, self.qk_head_dim)
        ckv, k_r = self.wkv_a(x2).split((self.kv_lora_rank, rope), dim=-1)
        kv = self.wkv_b(self.kv_norm(ckv)).view(T, H, nope + self.v_head_dim)
        k_c, v = kv.split((nope, self.v_head_dim), dim=-1)
        k_r = k_r.view(T, 1, rope)
        if self.use_rope:
            cos, sin = rotary_cos_sin(ctx, rope, self.rope_theta)
            q = torch.cat((q[..., :nope], apply_rotary(q[..., nope:], cos, sin)), dim=-1)
            k_r = apply_rotary(k_r, cos, sin)
        k = torch.cat((k_c, k_r.expand(T, H, rope)), dim=-1)
        return q, k, v, cq

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        q, k, v, _ = self.project(x[0], ctx)
        out = _dense_attention(q, k, v, ctx, self.window_size, self.softmax_scale)
        return self.out_proj(out.reshape(1, T, -1))

    def proj_flops(self) -> float:
        """Forward matmul FLOPs per position of the projections (no attention)."""
        H, d = self.num_heads, self.d_model
        if self.q_lora_rank > 0:
            f = 2 * d * self.q_lora_rank + 2 * self.q_lora_rank * H * self.qk_head_dim
        else:
            f = 2 * d * H * self.qk_head_dim
        f += 2 * d * (self.kv_lora_rank + self.qk_rope_head_dim)
        f += 2 * self.kv_lora_rank * H * (self.qk_nope_head_dim + self.v_head_dim)
        f += 2 * H * self.v_head_dim * d
        return f

    @classmethod
    def flops_per_position(cls, d_model: int, L: float, cv, **kwargs) -> float:
        m = _meta(cls, d_model, kwargs)
        cap = _win_cap(m.window_size, cv)
        ctx_keys = _avg_ctx(_n_positions(L), lambda t: min(t + 1, cap), cv)
        return m.proj_flops() + _quad(ctx_keys, m.num_heads, m.qk_head_dim, m.v_head_dim, cv)

    def extra_repr(self) -> str:
        return (f"heads={self.num_heads}, q_lora={self.q_lora_rank}, kv_lora={self.kv_lora_rank}, "
                f"qk_nope={self.qk_nope_head_dim}, qk_rope={self.qk_rope_head_dim}, v={self.v_head_dim}, "
                f"rope={self.use_rope}, window={self.window_size}")


# ============================================================================= NSA
class _Compressor(nn.Module):
    """phi of NSA Eq. (7): block [.., l, hd] -> [.., hd] = mean(block) + MLP(vec(block + pos_emb))."""

    def __init__(self, block: int, head_dim: int, hidden: int):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(block, head_dim))
        self.fc_in = nn.Linear(block * head_dim, hidden)
        self.fc_out = nn.Linear(hidden, head_dim)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        h = (blocks + self.pos_emb.to(blocks.dtype)).flatten(-2)
        return blocks.mean(-2) + self.fc_out(F.silu(self.fc_in(h)))


@register("mixer", "nsa")
class NativeSparseAttention(nn.Module):
    """Native Sparse Attention (see module docstring). x [1,T,D] -> [1,T,D]."""

    BRANCHES = ("cmp", "slc", "win")

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        rotary_emb_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        window_size: int = -1,
        nope_every: int = 0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        softmax_scale: Optional[float] = None,
        layer_idx: int = 0,
        head_dim: Optional[int] = None,
        compress_block: int = 8,
        compress_stride: int = 4,
        compress_hidden: Optional[int] = None,
        select_block: int = 8,
        select_n: int = 8,
        n_init_blocks: int = 1,
        n_local_blocks: int = 2,
        sliding_window: int = 32,
        separate_kv: bool = True,
    ):
        super().__init__()
        num_kv_heads = num_kv_heads or num_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"
        hd = d_model // num_heads if head_dim is None else int(head_dim)
        self.d_model, self.num_heads, self.num_kv_heads, self.head_dim = d_model, num_heads, num_kv_heads, hd
        self.layer_idx = layer_idx
        rot = hd if rotary_emb_dim is None else int(rotary_emb_dim)
        if _is_nope(layer_idx, nope_every):
            rot = 0
        assert rot % 2 == 0 and rot <= hd, f"bad rotary_emb_dim {rot}"
        self.rotary_emb_dim, self.rope_theta = rot, float(rope_theta)
        self.window_size = int(window_size)
        self.l, self.d, self.ls = int(compress_block), int(compress_stride), int(select_block)
        assert self.l >= 1 and self.d >= 1 and self.ls >= 1
        self.n, self.n_init, self.n_local = int(select_n), int(n_init_blocks), int(n_local_blocks)
        assert self.n_init + self.n_local <= self.n, "forced (initial + local) blocks exceed select_n"
        self.sliding_window = int(sliding_window)
        assert self.sliding_window >= 0
        self.separate_kv = bool(separate_kv)
        self.softmax_scale = softmax_scale if softmax_scale is not None else hd ** -0.5
        self.n_kv_sets = 3 if self.separate_kv else 1
        H, Hk = num_heads, num_kv_heads
        self.Wqkv = nn.Linear(d_model, (H + 2 * Hk * self.n_kv_sets) * hd, bias=qkv_bias)
        hidden = hd if compress_hidden is None else int(compress_hidden)
        self.compress_k = _Compressor(self.l, hd, hidden)
        self.compress_v = _Compressor(self.l, hd, hidden)
        self.gate_proj = nn.Linear(d_model, 3 * H)
        nn.init.zeros_(self.gate_proj.bias)
        self.out_proj = nn.Linear(H * hd, d_model, bias=out_bias)

    @classmethod
    def config_kwargs(cls, cfg, stage: int) -> dict:
        return cfg.attn_for_stage(stage)

    # ------------------------------------------------------------------ pieces
    def _overlap(self, n_cmp: int, n_slc: int, device) -> torch.Tensor:
        """M [n_cmp, n_slc]: |cmp block i ∩ slc block j| / d (Eq. 9 in overlap form)."""
        i = torch.arange(n_cmp, device=device, dtype=torch.float32)[:, None] * self.d
        j = torch.arange(n_slc, device=device, dtype=torch.float32)[None, :] * self.ls
        lo, hi = torch.maximum(i, j), torch.minimum(i + self.l, j + self.ls)
        return (hi - lo).clamp_min(0) / self.d

    def _compressed(self, q, k, v, ctx: SeqCtx):
        """Compression branch. Returns (out [T,H,hd], importance [B,Hk,Lmax,n_cmp] (detached, Eq. 10) or None)."""
        seg, pos, _, B, Lmax = _layout(ctx)
        T, H, hd = q.shape
        Hk = k.shape[1]
        G = H // Hk
        n_cmp = (Lmax - self.l) // self.d + 1 if Lmax >= self.l else 0
        if n_cmp <= 0:
            return q.new_zeros(T, H, hd), None
        pad = lambda x: x.new_zeros(B, Lmax, *x.shape[1:]).index_put((seg, pos), x)  # noqa: E731
        kp, vp, qp = pad(k), pad(v), pad(q)
        blocks = lambda x: x.unfold(1, self.l, self.d).permute(0, 1, 2, 4, 3)  # [B, n_cmp, Hk, l, hd]  # noqa: E731
        kc, vc = self.compress_k(blocks(kp)), self.compress_v(blocks(vp))  # [B, n_cmp, Hk, hd]
        t = torch.arange(Lmax, device=q.device)[:, None]
        start = torch.arange(n_cmp, device=q.device)[None, :] * self.d
        ok = start + self.l - 1 <= t  # [Lmax, n_cmp]: complete at or before t (Eq. 7)
        if self.window_size >= 0:
            ok = ok & (start >= t - self.window_size)
        s = torch.einsum("blagd,bnad->bagln", qp.view(B, Lmax, Hk, G, hd).float(), kc.float()).float()
        s = s * self.softmax_scale
        p = _masked_softmax(s, ok)  # [B, Hk, G, Lmax, n_cmp], Eq. (8)
        o = torch.einsum("bagln,bnad->blagd", p, vc.float()).reshape(B, Lmax, H, hd)
        return o[seg, pos].to(v.dtype), p.detach().sum(2)

    def _select(self, importance, ctx: SeqCtx, device):
        """Top-n block selection (Eqs. 9-11). Returns (blocks [B,Hk,Lmax,n] long, block_valid [B,Hk,Lmax,n])."""
        _, _, _, B, Lmax = _layout(ctx)
        Hk = self.num_kv_heads
        n_slc = -(-Lmax // self.ls)
        if importance is None:
            score = torch.zeros(B, Hk, Lmax, n_slc, device=device)
        else:
            score = importance @ self._overlap(importance.shape[-1], n_slc, device)
        t = torch.arange(Lmax, device=device)[:, None]
        j = torch.arange(n_slc, device=device)[None, :]
        cur = t // self.ls
        avail = j <= cur
        if self.window_size >= 0:
            avail = avail & ((j + 1) * self.ls - 1 >= t - self.window_size)
        forced = avail & ((j < self.n_init) | (j > cur - self.n_local))
        score = score.masked_fill(~avail, float("-inf")).masked_fill(forced, float("inf"))
        vals, blocks = _topk_recent(score, min(self.n, n_slc))
        return blocks, vals > float("-inf")

    def _selected(self, q, k, v, blocks, bvalid, ctx: SeqCtx, backend: str):
        seg, pos, starts, B, Lmax = _layout(ctx)
        T, H, hd = q.shape
        Hk = k.shape[1]
        G = H // Hk
        W = self.window_size
        if backend == "flex":
            n_slc = -(-Lmax // self.ls)
            sel = torch.zeros(B, Hk, Lmax, n_slc, dtype=torch.bool, device=q.device)
            sel.scatter_(-1, blocks, bvalid)
            seq = seg
            ls = self.ls

            def mask_mod(b, h, q_idx, kv_idx):
                keep = (seq[q_idx] == seq[kv_idx]) & (kv_idx <= q_idx)
                keep = keep & sel[seq[q_idx], h // G, pos[q_idx], pos[kv_idx] // ls]
                if W >= 0:
                    keep = keep & (q_idx - kv_idx <= W)
                return keep

            return _flex_attend(q, k, v, mask_mod, self.softmax_scale, per_head=True)
        bf = blocks.permute(0, 2, 1, 3)[seg, pos]  # [T, Hk, n]
        vf = bvalid.permute(0, 2, 1, 3)[seg, pos]
        tok = (bf[..., None] * self.ls + torch.arange(self.ls, device=q.device)).flatten(-2)  # [T, Hk, n*l']
        p = pos[:, None, None]
        valid = vf[..., None].expand(*vf.shape, self.ls).flatten(-2) & (tok <= p)
        if W >= 0:
            valid = valid & (p - tok <= W)
        idx = torch.where(valid, starts[seg][:, None, None] + tok, torch.zeros_like(tok))
        return _gather_attend(q, k, v, idx, valid, self.softmax_scale)[0]

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        H, Hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        sizes = (H * hd,) + (Hk * hd,) * (2 * self.n_kv_sets)
        parts = self.Wqkv(x[0]).split(sizes, dim=-1)
        q = parts[0].view(T, H, hd)
        kv = [p.view(T, Hk, hd) for p in parts[1:]]
        if self.rotary_emb_dim > 0:
            cos, sin = rotary_cos_sin(ctx, self.rotary_emb_dim, self.rope_theta)
            q = apply_rotary(q, cos, sin)
            kv = [apply_rotary(t, cos, sin) if i % 2 == 0 else t for i, t in enumerate(kv)]
        if self.separate_kv:
            (k_c, v_c), (k_s, v_s), (k_w, v_w) = kv[0:2], kv[2:4], kv[4:6]
        else:
            k_c = k_s = k_w = kv[0]
            v_c = v_s = v_w = kv[1]
        o_cmp, importance = self._compressed(q, k_c, v_c, ctx)
        blocks, bvalid = self._select(importance, ctx, x.device)
        o_slc = self._selected(q, k_s, v_s, blocks, bvalid, ctx, select_sparse_backend(q))
        w = self.sliding_window if self.window_size < 0 else min(self.sliding_window, self.window_size)
        o_win = _dense_attention(q, k_w, v_w, ctx, w, self.softmax_scale)
        g = torch.sigmoid(self.gate_proj(x[0]).float()).view(T, 3, H, 1)
        out = g[:, 0] * o_cmp.float() + g[:, 1] * o_slc.float() + g[:, 2] * o_win.float()
        return self.out_proj(out.to(x.dtype).reshape(1, T, H * hd))

    # ------------------------------------------------------------------ FLOPs
    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def _ctx_counts(N, l, d, ls, n, n_local, win, W, attn_conv):
        """(compressed blocks, selected keys, window keys) per query, averaged per the convention."""
        cv = type("C", (), {"attn": attn_conv})

        def n_cmp(t):
            hi = (t - l + 1) // d + 1 if t >= l - 1 else 0  # blocks complete at or before t
            if math.isinf(W):
                return hi
            lo = max(0, -(-(t - (W - 1)) // d))  # first block starting within [t - W + 1, t]
            return max(0, hi - lo)

        def n_slc(t):
            if (t // ls) + 1 <= n:
                keys = t + 1
            else:
                keys = (n - 1) * ls + (t % ls) + 1
            return min(keys, t + 1, W)

        return (_avg_ctx(N, n_cmp, cv), _avg_ctx(N, n_slc, cv), _avg_ctx(N, lambda t: min(t + 1, win + 1, W), cv))

    @classmethod
    def flops_per_position(cls, d_model: int, L: float, cv, **kwargs) -> float:
        m = _meta(cls, d_model, kwargs)
        H, Hk, hd, d = m.num_heads, m.num_kv_heads, m.head_dim, d_model
        hidden = m.compress_k.fc_in.out_features
        f = 2 * d * (H + 2 * Hk * m.n_kv_sets) * hd  # Wqkv
        f += 2 * d * 3 * H + 2 * H * hd * d  # gates, out_proj
        f += 2 * Hk * (2 * m.l * hd * hidden + 2 * hidden * hd) / m.d  # phi for K and V, one block per d positions
        W = _win_cap(m.window_size, cv)
        c_cmp, c_slc, c_win = cls._ctx_counts(_n_positions(L), m.l, m.d, m.ls, m.n, m.n_local, m.sliding_window,
                                              W, cv.attn)
        f += sum(_quad(c, H, hd, hd, cv) for c in (c_cmp, c_slc, c_win))
        if cv.count_elementwise:
            f += 2 * 3 * H * hd  # gated sum of the three branches
        return f

    def extra_repr(self) -> str:
        return (f"heads={self.num_heads}, kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
                f"cmp=(l={self.l}, d={self.d}), slc=(l'={self.ls}, n={self.n}, init={self.n_init}, "
                f"local={self.n_local}), win={self.sliding_window}, separate_kv={self.separate_kv}, "
                f"rotary_dim={self.rotary_emb_dim}, window={self.window_size}")


# ============================================================================= DSA
class LightningIndexer(nn.Module):
    """DSA lightning indexer (V3.2-Exp Eq. 1; reference ``Indexer``): returns (q^I [T,Hi,di], k^I [T,di],
    w^I [T,Hi]) with w^I already scaled by (Hi * di)^-1/2, so I_t,s = sum_j w_t,j relu(q_t,j . k_s)."""

    def __init__(self, d_model: int, q_in: int, n_heads: int, head_dim: int, rope_dim: int, rope_theta: float):
        super().__init__()
        assert rope_dim % 2 == 0 and rope_dim <= head_dim
        self.n_heads, self.head_dim, self.rope_dim, self.rope_theta = n_heads, head_dim, rope_dim, rope_theta
        self.wq = nn.Linear(q_in, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(d_model, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim)
        self.weights_proj = nn.Linear(d_model, n_heads, bias=False)

    def forward(self, x2, q_in, ctx: SeqCtx, use_rope: bool):
        T = x2.shape[0]
        q = self.wq(q_in).view(T, self.n_heads, self.head_dim)
        k = self.k_norm(self.wk(x2)).view(T, 1, self.head_dim)
        if use_rope and self.rope_dim > 0:
            cos, sin = rotary_cos_sin(ctx, self.rope_dim, self.rope_theta)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        w = self.weights_proj(x2).float() * (self.n_heads * self.head_dim) ** -0.5
        return q, k[:, 0], w

    @staticmethod
    def score(q, k, w):
        """q [..,Hi,di], k [..,S,di] (aligned leading dims), w [..,Hi] -> I [..,S] (fp32)."""
        s = torch.einsum("...hd,...sd->...hs", q.float(), k.float()).float().relu()
        return torch.einsum("...hs,...h->...s", s, w.float()).float()


def _attn_project(mod: CausalAttention, x2: torch.Tensor, ctx: SeqCtx):
    """q/k/v of the plain ``attn`` mixer (same math as ``CausalAttention.forward``)."""
    T = x2.shape[0]
    hq, hk, hd = mod.num_heads, mod.num_kv_heads, mod.head_dim
    q, k, v = mod.Wqkv(x2).split((hq * hd, hk * hd, hk * hd), dim=-1)
    q, k, v = q.view(T, hq, hd), k.view(T, hk, hd), v.view(T, hk, hd)
    if mod.rotary_emb_dim > 0:
        cos, sin = rotary_cos_sin(ctx, mod.rotary_emb_dim, mod.rope_theta)
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
    return q, k, v, None


@register("mixer", "dsa")
class DeepSeekSparseAttention(nn.Module):
    """DeepSeek Sparse Attention (see module docstring). x [1,T,D] -> [1,T,D].

    ``base_kwargs`` (e.g. kv_lora_rank, q_lora_rank, qk_nope_head_dim, qk_rope_head_dim, v_head_dim) go to the base
    mixer; unknown keys raise there."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        rotary_emb_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        window_size: int = -1,
        nope_every: int = 0,
        qkv_bias: bool = False,
        out_bias: bool = False,
        softmax_scale: Optional[float] = None,
        layer_idx: int = 0,
        base: str = "mla",
        index_topk: int = 64,
        index_n_heads: int = 4,
        index_head_dim: int = 32,
        index_rope_dim: Optional[int] = None,
        index_loss: str = "inject",
        index_loss_coef: float = 1.0,
        detach_indexer_input: bool = True,
        **base_kwargs,
    ):
        super().__init__()
        common = dict(num_heads=num_heads, num_kv_heads=num_kv_heads, rotary_emb_dim=rotary_emb_dim,
                      rope_theta=rope_theta, window_size=window_size, nope_every=nope_every, qkv_bias=qkv_bias,
                      out_bias=out_bias, softmax_scale=softmax_scale, layer_idx=layer_idx)
        if base == "mla":
            self.attn = MultiHeadLatentAttention(d_model, **common, **base_kwargs)
            q_in = self.attn.q_lora_rank or d_model
            use_rope = self.attn.use_rope
        elif base == "attn":
            self.attn = CausalAttention(d_model, **common, **base_kwargs)
            q_in = d_model
            use_rope = self.attn.rotary_emb_dim > 0
        else:
            raise ValueError(f"dsa base must be 'mla' or 'attn', got {base!r}")
        assert index_loss in ("inject", "attr", "off"), index_loss
        self.base, self.d_model, self.num_heads, self.layer_idx = base, d_model, num_heads, layer_idx
        self.index_topk = int(index_topk)
        assert self.index_topk >= 1
        self.index_loss, self.index_loss_coef = index_loss, float(index_loss_coef)
        self.detach_indexer_input = bool(detach_indexer_input)
        self.window_size = int(window_size)
        self.use_rope = use_rope
        rope = index_head_dim // 2 if index_rope_dim is None else int(index_rope_dim)
        self.indexer = LightningIndexer(d_model, q_in, int(index_n_heads), int(index_head_dim), rope,
                                        float(rope_theta))
        self.aux_loss: Optional[torch.Tensor] = None
        self.index_kl: Optional[torch.Tensor] = None

    @classmethod
    def config_kwargs(cls, cfg, stage: int) -> dict:
        return cfg.attn_for_stage(stage)

    # ------------------------------------------------------------------ pieces
    def _topk(self, iq, ik, iw, ctx: SeqCtx, K: int):
        """Top-K keys per query by index score (no grad). Returns (idx [T,K] flat, valid [T,K] bool).

        Python loop over query chunks of C positions; each chunk scores only keys in [a - max_seqlen + 1, b)."""
        T = iq.shape[0]
        seg = ctx.seq_idx[0]
        Lm, W = max(int(ctx.max_seqlen), 1), self.window_size
        span_max = Lm if W < 0 else min(Lm, W + 1)
        # chunk <= span_max so a chunk scores <= 2 * span_max keys; <= INDEX_CHUNK_ELEMS fp32 scores per chunk
        C = max(1, min(T, span_max, INDEX_CHUNK_ELEMS // (iq.shape[1] * 2 * span_max)))
        idx = torch.zeros(T, K, dtype=torch.long, device=iq.device)
        valid = torch.zeros(T, K, dtype=torch.bool, device=iq.device)
        for a in range(0, T, C):
            b = min(T, a + C)
            lo = max(0, a - span_max + 1)
            s = torch.einsum("chd,sd->chs", iq[a:b].float(), ik[lo:b].float()).float().relu_()
            s = torch.einsum("chs,ch->cs", s, iw[a:b]).float()  # Eq. 1
            qp = torch.arange(a, b, device=iq.device)[:, None]
            kp = torch.arange(lo, b, device=iq.device)[None, :]
            ok = (seg[a:b, None] == seg[None, lo:b]) & (kp <= qp)
            if W >= 0:
                ok = ok & (qp - kp <= W)
            vals, top = _topk_recent(s.masked_fill(~ok, float("-inf")), min(K, b - lo))
            kk = top.shape[-1]
            idx[a:b, :kk] = top + lo
            valid[a:b, :kk] = vals > float("-inf")
        return torch.where(valid, idx, torch.zeros_like(idx)), valid

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        x2 = x[0]
        if self.base == "mla":
            q, k, v, cq = self.attn.project(x2, ctx)
        else:
            q, k, v, cq = _attn_project(self.attn, x2, ctx)
        q_in = x2 if cq is None else cq
        xi, qi = (x2.detach(), q_in.detach()) if self.detach_indexer_input else (x2, q_in)
        iq, ik, iw = self.indexer(xi, qi, ctx, self.use_rope)
        Lm = max(int(ctx.max_seqlen), 1)
        K = min(self.index_topk, Lm if self.window_size < 0 else min(Lm, self.window_size + 1))
        with torch.no_grad():
            idx, valid = self._topk(iq, ik, iw, ctx, K)
        scale = self.attn.softmax_scale
        Hk = k.shape[1]
        idx_h, valid_h = idx[:, None, :].expand(T, Hk, K), valid[:, None, :].expand(T, Hk, K)
        need_aux = self.index_loss != "off" and self.training and torch.is_grad_enabled()
        backend = select_sparse_backend(q)
        if backend == "flex":
            seg, pos, _, B, Lmax = _layout(ctx)
            # scatter_add: invalid slots all point at key 0 and must not overwrite a valid hit there
            sel = torch.zeros(T, Lmax, dtype=torch.int32, device=x.device)
            sel = sel.scatter_add_(-1, pos[idx], valid.to(torch.int32)) > 0
            seq = seg

            def mask_mod(b, h, q_idx, kv_idx):
                return (seq[q_idx] == seq[kv_idx]) & (kv_idx <= q_idx) & sel[q_idx, pos[kv_idx]]

            out = _flex_attend(q, k, v, mask_mod, scale, per_head=False)
            probs = None
            if need_aux:
                with torch.no_grad():
                    probs = _gather_attend(q, k, v, idx_h, valid_h, scale)[1]
        else:
            out, probs = _gather_attend(q, k, v, idx_h, valid_h, scale)
        if need_aux:
            target = probs.detach().sum(1) / probs.shape[1]  # Eq. 4: head-summed, L1-normalized, [T, K]
            logits = LightningIndexer.score(iq, ik[idx], iw)  # I_t,S_t with grad to the indexer
            logq = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1).masked_fill(~valid, 0.0)
            kl = (torch.xlogy(target, target) - target * logq).sum(-1).mean()
            aux = kl * self.index_loss_coef
            self.index_kl = kl.detach()
            if self.index_loss == "inject":  # gradient already delivered; keep only the value (no graph)
                out = _InjectAuxGrad.apply(out, aux)
                self.aux_loss = aux.detach()
            else:
                self.aux_loss = aux
        else:
            self.aux_loss = None
        return self.attn.out_proj(out.reshape(1, T, -1))

    # ------------------------------------------------------------------ FLOPs
    @classmethod
    def flops_per_position(cls, d_model: int, L: float, cv, **kwargs) -> float:
        m = _meta(cls, d_model, kwargs)
        a = m.attn
        N = _n_positions(L)
        W = _win_cap(m.window_size, cv)
        if m.base == "mla":
            proj, dqk, dv, H = a.proj_flops(), a.qk_head_dim, a.v_head_dim, a.num_heads
        else:
            H, Hk, hd = a.num_heads, a.num_kv_heads, a.head_dim
            proj, dqk, dv = 2 * d_model * (H + 2 * Hk) * hd + 2 * H * hd * d_model, hd, hd
        ix = m.indexer
        Hi, di = ix.n_heads, ix.head_dim
        f = proj + 2 * ix.wq.in_features * Hi * di + 2 * d_model * di + 2 * d_model * Hi  # indexer projections
        ctx_all = _avg_ctx(N, lambda t: min(t + 1, W), cv)
        f += (2 * di * Hi + (3 * Hi if cv.count_elementwise else 0)) * ctx_all * cv.attn_quadratic_scale  # Eq. 1
        ctx_sel = _avg_ctx(N, lambda t: min(t + 1, W, m.index_topk), cv)
        return f + _quad(ctx_sel, H, dqk, dv, cv)

    def extra_repr(self) -> str:
        return (f"base={self.base}, topk={self.index_topk}, index_heads={self.indexer.n_heads}, "
                f"index_dim={self.indexer.head_dim}, index_loss={self.index_loss}x{self.index_loss_coef}")
