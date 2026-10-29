"""Kimi (Moonshot AI) attention variants and a close relative, as registered sequence mixers.

=================  =====================================================================================
mixer name         method
=================  =====================================================================================
``kda``            Kimi Delta Attention (Kimi Linear, arXiv 2510.26692): gated delta rule with channel-wise
                   forget gates, short convs on q/k/v, L2-normalized q/k, sigmoid output gate + RMSNorm.
``moba``           Mixture of Block Attention (Kimi, arXiv 2502.13189): every query attends to its own
                   block (causally) plus the top-k-1 past KV blocks chosen by q . mean-pooled block keys.
``gated_attn``     Gated attention (Qwen, arXiv 2505.06708, best variant "G1"): head-specific elementwise
                   sigmoid gate on the SDPA output, ``Y' = Y * sigmoid(X W_theta)``.
=================  =====================================================================================

All three drop into both the H-Net (per-stage ``HNetConfig.mixer_codes``, e.g. ``{"m": "mamba2", "t": "kda"}``
for the main stage only) and the dense baseline (``DenseConfig.mixer``). Every constructor takes
``(d_model, layer_idx=0, **attention kwargs)`` with the same attention keys as the reference ``attn`` mixer
(``num_heads, num_kv_heads, rotary_emb_dim, rope_theta, window_size, nope_every, qkv_bias, out_bias,
softmax_scale``) plus its own keys; keys a method has no use for are accepted and ignored explicitly (listed in each
class docstring). ``config_kwargs(cfg, stage)`` returns the stage's attention settings (``cfg.attn_for_stage``);
``cfg.mixer_cfg[name]`` adds the method's own knobs. ``flops_per_position(d_model, L, cv, **kwargs)`` returns
forward FLOPs per position in the ``hnet.flops`` conventions (the caller applies the backward factor).

Kimi Linear's hybrid (3 KDA : 1 full-attention layer, the full-attention layers without RoPE) is a configuration
pattern, not a mixer: e.g. main layout ``"K3T1K2T1"`` with ``mixer_codes`` ``{"k": "kda", "t": "attn"}`` and
``mixer_cfg: {"attn": [{}, {}, {"rotary_emb_dim": 0}]}`` (see configs/models/tiny_hnet2_2m_kda_hybrid.yaml).

Document masking: every mixer reads ``ctx`` (``seq_idx`` / ``position_ids`` / ``cu_seqlens``); no state,
convolution window, block or attention edge crosses a segment boundary.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.registry import register

from ..ctx import SeqCtx
from ..rotary import apply_rotary, rotary_cos_sin
from ..ssd import use_kernel
from .attn import CausalAttention, attention, create_block_mask, flex_attention, _flex_fn
from .conv import causal_conv1d

try:  # optional fast path; only a missing `fla` package falls back, a broken install raises
    from fla.ops.kda import chunk_kda as _fla_chunk_kda
except ModuleNotFoundError as e:
    if e.name != "fla":
        raise
    _fla_chunk_kda = None
else:  # Triton kernel is opaque to torch.compile: run it eagerly inside compiled graphs
    _fla_chunk_kda = torch.compiler.disable(_fla_chunk_kda)

__all__ = [
    "KimiDeltaAttention",
    "MoBA",
    "GatedAttention",
    "kda_chunk_torch",
    "kda_recurrent_reference",
    "moba_block_selection",
]

# Attention keys every mixer accepts (the reference ``attn`` constructor's keys besides d_model / layer_idx).
ATTN_KEYS = ("num_heads", "num_kv_heads", "rotary_emb_dim", "rope_theta", "window_size", "nope_every", "qkv_bias",
             "out_bias", "softmax_scale")


def _check_ignored(name: str, kwargs: dict, ignored: tuple):
    unknown = sorted(set(kwargs) - set(ignored))
    if unknown:
        raise TypeError(f"{name}: unexpected constructor arguments {unknown}")


def _no_autocast(t: torch.Tensor):
    return torch.autocast(device_type=t.device.type, enabled=False)


# =============================================================================================== KDA
def _decayed_scores(x: torch.Tensor, y: torch.Tensor, G: torch.Tensor, sub: int) -> torch.Tensor:
    """out[..., r, i] = sum_d x[r, d] y[i, d] exp(G[r, d] - G[i, d]) for i <= r (other entries: finite garbage).

    x, y, G: [..., C, K] with G non-increasing along C (cumulative log-decay). Overflow-free two-level scheme
    (the secondary chunking of the fla kernels): inside a sub-chunk of ``sub`` positions the [sub, sub, K] decay
    tensor is formed explicitly; for a source sub-chunk a and a later row r the decay factorizes through the
    reference point G_ref = G[end of a]:  exp(G_r - G_i) = exp(G_r - G_ref) * exp(G_ref - G_i), both factors <= 1.
    """
    *lead, C, K = x.shape
    ns = C // sub
    xs, ys, Gs = (t.reshape(*lead, ns, sub, K) for t in (x, y, G))
    # Diagonal sub-blocks, explicit pairwise decay [.., ns, r, i, K].
    D = (Gs[..., :, None, :] - Gs[..., None, :, :]).clamp(max=0.0).exp()
    diag = torch.einsum("...rk,...ik,...rik->...ri", xs, ys, D)  # [.., ns, sub, sub]
    # Off-diagonal sub-blocks through the reference point of the source sub-chunk.
    Gref = Gs[..., -1, :]  # [.., ns, K]
    y_hat = ys * (Gref[..., None, :] - Gs).exp()  # [.., ns(a), sub(i), K], exponent <= 0
    x_hat = x[..., None, :, :] * (G[..., None, :, :] - Gref[..., :, None, :]).clamp(max=0.0).exp()  # [.., a, C, K]
    off = torch.einsum("...ark,...aik->...rai", x_hat, y_hat).reshape(*lead, C, C)
    eye = torch.eye(ns, dtype=torch.bool, device=x.device)
    diag_full = (diag[..., :, :, None, :] * eye[:, None, :, None].to(diag.dtype)).reshape(*lead, C, C)
    blockdiag = eye[:, None, :, None].expand(ns, sub, ns, sub).reshape(C, C)
    return torch.where(blockdiag, diag_full, off)


def kda_chunk_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    seq_idx: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    sub_chunk: int = 16,
) -> torch.Tensor:
    """Chunkwise-parallel KDA (gated delta rule with channel-wise decay), pure PyTorch.

    Recurrence (Kimi Linear Eq. 1; per head, state S_t in R^{K x V}, alpha_t = exp(g_t) in (0, 1]^K):

        S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,      o_t = scale * S_t^T q_t

    with S reset to 0 at every segment start (``seq_idx`` changes). Chunk form (chunk of C positions, entry state
    S_0, Gamma_r = cumprod alpha inside the chunk, G = log Gamma): writing S_r = Diag(alpha_r) S_{r-1} + k_r u_r^T,
    the "pseudo values" u satisfy the unit-lower-triangular system (WY / UT transform, Kimi Linear Eqs. 6-8)

        (I + A) U = Diag(beta) (V - (Gamma * K) S_0),     A_{ri} = beta_r k_r^T Diag(Gamma_r / Gamma_i) k_i  (i < r)

    so U = U' - W S_0 with U' = T Diag(beta) V, W = T Diag(beta) (Gamma * K), T = (I + A)^{-1}. Then

        O     = scale * ((Gamma * Q) S_0 + P U),           P_{ri} = q_r^T Diag(Gamma_r / Gamma_i) k_i   (i <= r)
        S_C   = Diag(Gamma_C) S_0 + ((Gamma_C / Gamma) * K)^T U.

    Segment masking: A and P keep same-segment pairs only, the entry state only reaches positions of the segment
    that was running at the chunk start, and only positions of the chunk's last segment write the exit state.

    Args: q, k [T, H, K] (already L2-normalized), v [T, H, V], g [T, H, K] log-decay (<= 0), beta [T, H],
    seq_idx [T] (or [1, T]) segment ids. Returns o [T, H, V] in v's dtype. Computes in fp32 (fp64 for fp64 inputs)
    with autocast disabled. Python loop over chunks only (the matrix-valued chunk transition is sequential).
    """
    with _no_autocast(q):
        return _kda_chunk_torch(q, k, v, g, beta, scale, seq_idx, chunk_size, sub_chunk)


def _kda_chunk_torch(q, k, v, g, beta, scale, seq_idx, chunk_size, sub_chunk):
    T, H, K = q.shape
    V = v.shape[-1]
    f = torch.float64 if q.dtype == torch.float64 else torch.float32
    C = int(chunk_size)
    sub = min(int(sub_chunk), C)
    assert C % sub == 0, "chunk_size must be a multiple of sub_chunk"
    pad = (-T) % C
    N = (T + pad) // C

    def hm(t):  # [T, H, *] -> [H, N, C, *] (head-major, padded with zeros)
        t = t.to(f)
        if pad:
            t = F.pad(t, (0, 0) * (t.dim() - 1) + (0, pad))
        return t.reshape(N, C, H, *t.shape[2:]).movedim(2, 0)

    q, k, v, g = hm(q * scale), hm(k), hm(v), hm(g)
    beta = hm(beta.unsqueeze(-1)).squeeze(-1)  # [H, N, C]
    G = g.cumsum(dim=-2)

    if seq_idx is None:
        seg = torch.zeros(N, C, dtype=torch.long, device=q.device)
    else:
        seg = seq_idx.reshape(-1).to(torch.long)
        if pad:
            seg = torch.cat([seg, seg[-1:].expand(pad)])
        seg = seg.view(N, C)
    seg_entry = torch.cat([seg[:1, 0], seg[:-1, -1]])  # segment running when the chunk starts
    m_in = (seg == seg_entry[:, None]).to(f)  # [N, C]: reached by the entry state
    m_out = (seg == seg[:, -1:]).to(f)  # [N, C]: writes the exit state
    same = seg[:, :, None] == seg[:, None, :]
    pos = torch.arange(C, device=q.device)
    strict = same & (pos[:, None] > pos[None, :])
    incl = same & (pos[:, None] >= pos[None, :])

    A = (_decayed_scores(k, k, G, sub) * beta[..., None]).masked_fill(~strict, 0.0)
    eye = torch.eye(C, dtype=f, device=q.device)
    rhs = torch.cat([v * beta[..., None], k * G.exp() * m_in[..., None] * beta[..., None]], dim=-1)
    X = torch.linalg.solve_triangular(eye + A, rhs, upper=False, unitriangular=True)
    U1, W = X[..., :V], X[..., V:]
    P = _decayed_scores(q, k, G, sub).masked_fill(~incl, 0.0)
    Qg = q * G.exp() * m_in[..., None]
    G_last = G[..., -1:, :]
    Kg = k * (G_last - G).exp() * m_out[..., None]
    d_last = G_last.squeeze(-2).exp() * m_in[:, -1][None, :, None]  # [H, N, K]

    S = q.new_zeros(H, K, V)
    outs = []
    for n in range(N):
        U = U1[:, n] - W[:, n] @ S
        outs.append(Qg[:, n] @ S + P[:, n] @ U)
        S = d_last[:, n, :, None] * S + Kg[:, n].transpose(-1, -2) @ U
    o = torch.stack(outs, dim=1)  # [H, N, C, V]
    return o.movedim(0, 2).reshape(N * C, H, V)[:T].to(v.dtype)


def kda_recurrent_reference(q, k, v, g, beta, scale, seq_idx=None):
    """Step-by-step KDA recurrence (tests only; per-token Python loop). Same arguments as :func:`kda_chunk_torch`."""
    T, H, K = q.shape
    f = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v, g, beta = (t.to(f) for t in (q, k, v, g, beta))
    seg = None if seq_idx is None else seq_idx.reshape(-1).tolist()
    S = q.new_zeros(H, K, v.shape[-1])
    out = []
    for t in range(T):
        if seg is not None and t > 0 and seg[t] != seg[t - 1]:
            S = torch.zeros_like(S)
        S = g[t].exp()[..., None] * S  # Diag(alpha_t) S_{t-1}
        pred = torch.einsum("hk,hkv->hv", k[t], S)  # k_t^T Diag(alpha_t) S_{t-1}
        S = S + beta[t][:, None, None] * k[t][..., None] * (v[t] - pred)[:, None, :]
        out.append(scale * torch.einsum("hk,hkv->hv", q[t], S))
    return torch.stack(out).to(v.dtype)


class _HeadRMSNormGated(nn.Module):
    """Per-head ``RMSNorm(o) * weight * sigmoid(gate)`` (fla ``FusedRMSNormGated(head_v_dim, 'sigmoid')``)."""

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        of = o.float()
        y = of * torch.rsqrt(of.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(gate.float())).to(o.dtype)


@register("mixer", "kda")
class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention (Kimi Linear, arXiv 2510.26692, Sec. 3; parameterization of fla ``KimiDeltaAttention``).

    Per head (H heads, key dim K = ``head_dim``, value dim V = ``expand_v * head_dim``), Kimi Linear Eqs. 1, 9-10:

        q_t, k_t, v_t = SiLU(ShortConv(W_{q,k,v} x_t));   q_t, k_t <- L2Norm(q_t), L2Norm(k_t)
        g_t   = -exp(A_log) * softplus(W_f^up W_f^down x_t + dt_bias)  in R^K      (alpha_t = exp(g_t))
        beta_t = sigmoid(W_b x_t)                         (x2 if ``allow_neg_eigval``)
        S_t   = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,   o_t = S_t^T q_t / sqrt(K)
        y_t   = W_o ( RMSNorm(o_t) * sigmoid(W_g^up W_g^down x_t) )                 (per-head RMSNorm, output gate)

    Low-rank gate projections have rank V (fla). Init (fla): A_log = log U[1, 16], dt_bias = softplus^{-1}(dt) with
    dt log-uniform in [1e-3, 1e-1]; output-gate up-projection has a bias.

    Execution: CUDA + ``fla`` (``fla.ops.kda.chunk_kda``, varlen via ``cu_seqlens``) unless ``HNET_FORCE_TORCH=1``;
    otherwise :func:`kda_chunk_torch`. The short convolutions use :func:`hnet.model.mixers.conv.causal_conv1d`
    (``causal_conv1d`` kernel when available), which resets at segment starts like the state.

    Ignored attention keys (KDA has no RoPE, no GQA, no window and no softmax): ``num_kv_heads, rotary_emb_dim,
    rope_theta, window_size, nope_every, qkv_bias, out_bias, softmax_scale``. Own keys: ``head_dim`` (default
    d_model // num_heads), ``expand_v``, ``conv_size``, ``conv_bias``, ``gate_rank`` (default V), ``chunk_size``,
    ``sub_chunk``, ``allow_neg_eigval``, ``norm_eps``.
    """

    IGNORED = ("num_kv_heads", "rotary_emb_dim", "rope_theta", "window_size", "nope_every", "qkv_bias", "out_bias",
               "softmax_scale")

    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        layer_idx: int = 0,
        head_dim: Optional[int] = None,
        expand_v: float = 1.0,
        conv_size: int = 4,
        conv_bias: bool = False,
        gate_rank: Optional[int] = None,
        chunk_size: int = 64,
        sub_chunk: int = 16,
        allow_neg_eigval: bool = False,
        norm_eps: float = 1e-5,
        **ignored,
    ):
        super().__init__()
        _check_ignored("kda", ignored, self.IGNORED)
        dims = self._dims(d_model, num_heads, head_dim, expand_v, gate_rank)
        self.d_model, self.layer_idx = d_model, layer_idx
        self.num_heads, self.head_k_dim, self.head_v_dim = num_heads, dims["hk"], dims["hv"]
        self.key_dim, self.value_dim, self.gate_rank = dims["key"], dims["value"], dims["rank"]
        self.chunk_size, self.sub_chunk = int(chunk_size), int(sub_chunk)
        self.allow_neg_eigval = bool(allow_neg_eigval)
        self.scale = self.head_k_dim ** -0.5

        self.q_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.value_dim, bias=False)

        def conv(c):
            return nn.Conv1d(c, c, conv_size, groups=c, padding=conv_size - 1, bias=conv_bias)

        self.q_conv1d, self.k_conv1d, self.v_conv1d = conv(self.key_dim), conv(self.key_dim), conv(self.value_dim)
        self.f_proj = nn.Sequential(nn.Linear(d_model, self.gate_rank, bias=False),
                                    nn.Linear(self.gate_rank, self.key_dim, bias=False))
        self.b_proj = nn.Linear(d_model, num_heads, bias=False)
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(self.key_dim, dtype=torch.float32))
        for p in (self.A_log, self.dt_bias):
            p._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(d_model, self.gate_rank, bias=False),
                                    nn.Linear(self.gate_rank, self.value_dim, bias=True))
        self.o_norm = _HeadRMSNormGated(self.head_v_dim, eps=norm_eps)
        self.out_proj = nn.Linear(self.value_dim, d_model, bias=False)
        self.reset_kda_parameters()

    @staticmethod
    def _dims(d_model, num_heads, head_dim=None, expand_v=1.0, gate_rank=None) -> dict:
        hk = int(head_dim) if head_dim else d_model // num_heads
        assert head_dim or d_model % num_heads == 0, "d_model must be divisible by num_heads (or pass head_dim)"
        hv = int(round(hk * expand_v))
        assert math.isclose(hk * expand_v, hv), f"expand_v={expand_v} gives a non-integer value head dim"
        rank = int(gate_rank) if gate_rank else hv
        return dict(hk=hk, hv=hv, key=num_heads * hk, value=num_heads * hv, rank=rank, h=num_heads)

    @torch.no_grad()
    def reset_kda_parameters(self):
        """fla init of A_log (log U[1, 16]) and dt_bias (inverse softplus of log-uniform dt in [1e-3, 1e-1])."""
        if self.A_log.is_meta:
            return
        self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
        dt = torch.exp(torch.rand_like(self.dt_bias) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt = dt.clamp(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    @classmethod
    def config_kwargs(cls, cfg, stage: int) -> dict:
        return cfg.attn_for_stage(stage)

    @classmethod
    def flops_per_position(cls, d_model, L, cv, num_heads=4, head_dim=None, expand_v=1.0, conv_size=4,
                           gate_rank=None, **kwargs) -> float:
        """Forward FLOPs per position. Matmuls 2*in*out (q, k, v, low-rank f/g, b, out); depthwise convs
        2*conv_size*channels; the recurrence in its per-token form (like the Mamba-2 SSD term): decay, k^T S,
        rank-1 update and readout, 2*4*H*K*V; elementwise (if counted): gate + output gate/norm 5*(key + value).
        Independent of L (linear-time)."""
        m = cls._dims(d_model, num_heads, head_dim, expand_v, gate_rank)
        f = 2 * d_model * (2 * m["key"] + m["value"])  # q, k, v
        f += 2 * (d_model * m["rank"] + m["rank"] * m["key"])  # f_proj
        f += 2 * (d_model * m["rank"] + m["rank"] * m["value"])  # g_proj
        f += 2 * d_model * m["h"]  # b_proj
        f += 2 * m["value"] * d_model  # out_proj
        f += 2 * conv_size * (2 * m["key"] + m["value"])
        f += 2 * 4 * m["h"] * m["hk"] * m["hv"]
        if cv.count_elementwise:
            f += 5 * (m["key"] + m["value"])
        return float(f)

    def _conv(self, conv: nn.Conv1d, x: torch.Tensor, seq_idx) -> torch.Tensor:
        return causal_conv1d(x, conv.weight.squeeze(1), conv.bias, seq_idx, activation="silu")

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[0] == 1, "KDA expects packed x [1, T, D]"
        T = x.shape[1]
        seq_idx = ctx.seq_idx.reshape(1, T).to(torch.int32)
        H, K, V = self.num_heads, self.head_k_dim, self.head_v_dim
        q = self._conv(self.q_conv1d, self.q_proj(x), seq_idx)[0].view(T, H, K)
        k = self._conv(self.k_conv1d, self.k_proj(x), seq_idx)[0].view(T, H, K)
        v = self._conv(self.v_conv1d, self.v_proj(x), seq_idx)[0].view(T, H, V)
        q = q * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6).to(q.dtype)  # fla l2norm, eps 1e-6
        k = k * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6).to(k.dtype)
        f = self.f_proj(x[0]).float().view(T, H, K)
        g = -self.A_log.float().exp()[:, None] * F.softplus(f + self.dt_bias.float().view(H, K))
        beta = torch.sigmoid(self.b_proj(x[0]).float())
        if self.allow_neg_eigval:
            beta = 2 * beta
        if use_kernel(x, _fla_chunk_kda):
            cu = torch.unique_consecutive(ctx.cu_seqlens.to(torch.long))  # fla rejects empty segments
            o, _ = _fla_chunk_kda(q=q[None], k=k[None], v=v[None], g=g[None], beta=beta[None].to(q.dtype),
                                  scale=self.scale, cu_seqlens=cu)
            o = o[0]
        else:
            o = kda_chunk_torch(q, k, v, g, beta, self.scale, ctx.seq_idx, self.chunk_size, self.sub_chunk)
        o = self.o_norm(o, self.g_proj(x[0]).view(T, H, V))
        return self.out_proj(o.reshape(1, T, H * V))

    def extra_repr(self) -> str:
        return (f"heads={self.num_heads}, head_k_dim={self.head_k_dim}, head_v_dim={self.head_v_dim}, "
                f"gate_rank={self.gate_rank}, chunk={self.chunk_size}/{self.sub_chunk}")


# =============================================================================================== MoBA
@torch.no_grad()
def moba_block_selection(q: torch.Tensor, k: torch.Tensor, ctx: SeqCtx, block_size: int, top_k: int):
    """MoBA gate (arXiv 2502.13189, Eqs. 3-6): which KV blocks each (head, query) attends to.

    Blocks are ``block_size`` consecutive positions of one segment, aligned to the segment start (block j of a
    segment = positions [j*B, (j+1)*B)). For query t in block b(t):

        s_{t,i} = q_t . mean_pool(K[I_i])               for past blocks i < b(t) of the same segment
        g_{t,i} = 1  iff  i == b(t)  or  s_{t,i} in TopK({s_{t,j}}, top_k - 1)

    (the current block is always selected and counted in ``top_k``; future blocks / other segments are never
    candidates). The gate has no parameters and passes no gradient (top-k selection).

    q [T, H, hd], k [T, Hkv, hd] (after RoPE). Returns ``(sel [H, T, nb] bool, gid [T] long)`` where ``gid`` is
    the global block id of every position and ``nb = T // block_size + n_segments`` (an upper bound on the
    number of blocks, so there is no host sync).
    """
    T, H, hd = q.shape
    Hkv = k.shape[1]
    B = int(block_size)
    pos = ctx.position_ids.to(torch.long)
    seq = ctx.seq_idx.reshape(-1).to(torch.long)
    gid = (pos % B == 0).to(torch.long).cumsum(0) - 1
    nb = T // B + ctx.n_segments
    kf = k.float()
    ksum = kf.new_zeros(nb, Hkv, hd).index_add_(0, gid, kf)
    cnt = kf.new_zeros(nb).index_add_(0, gid, torch.ones_like(gid, dtype=kf.dtype))
    kbar = ksum / cnt.clamp(min=1.0)[:, None, None]
    block_seg = torch.full((nb,), -1, dtype=torch.long, device=q.device).scatter_(0, gid, seq)
    sel = torch.zeros(H, T, nb, dtype=torch.bool, device=q.device)
    n_pick = min(int(top_k) - 1, nb)
    if n_pick > 0:
        kbar = kbar.repeat_interleave(H // Hkv, dim=1)  # [nb, H, hd]
        scores = torch.einsum("thd,nhd->htn", q.float(), kbar)
        j = torch.arange(nb, device=q.device)
        cand = (block_seg[None, :] == seq[:, None]) & (j[None, :] < gid[:, None])  # [T, nb]
        scores = scores.masked_fill(~cand, float("-inf"))
        idx = scores.topk(n_pick, dim=-1).indices  # [H, T, n_pick]
        sel.scatter_(-1, idx, cand.expand(H, T, nb).gather(-1, idx))
    sel.scatter_(-1, gid.view(1, T, 1).expand(H, T, 1), True)  # current block
    return sel, gid


@register("mixer", "moba")
class MoBA(CausalAttention):
    """Mixture of Block Attention (Kimi, arXiv 2502.13189), on top of the reference ``attn`` mixer.

    Same projections (``Wqkv``, ``out_proj``), GQA and RoPE/NoPE as ``attn``; the softmax attention of query t
    runs over the keys of the blocks selected by :func:`moba_block_selection` (Eq. 2:
    ``MoBA(q, K, V) = Softmax(q K[I]^T) V[I]``, I = union of selected blocks), causally masked inside the current
    block and never crossing a segment. ``window_size >= 0`` additionally restricts keys to [t - w, t].
    With ``top_k >= number of blocks`` it equals causal attention exactly.

    Own keys: ``block_size`` (default 32), ``top_k`` (default 3, counts the current block): at 448-2048 positions
    a query sees at most 96 keys (vs 224-1024 on average for causal attention). Paper: block 512, top-k 3 at 8K.

    Backends: CUDA -> flex_attention with a per-(head, query) block-selection ``mask_mod``; otherwise SDPA with an
    explicit [H, T, T] boolean mask (O(H T^2) memory; tests / small inputs). ``backend=`` forces one.
    """

    def __init__(self, d_model: int, num_heads: int, layer_idx: int = 0, block_size: int = 32, top_k: int = 3,
                 backend: Optional[str] = None, **attn_kwargs):
        _check_ignored("moba", attn_kwargs, ATTN_KEYS)
        super().__init__(d_model, num_heads, layer_idx=layer_idx, **attn_kwargs)
        assert block_size >= 1 and top_k >= 1
        assert backend in (None, "sdpa", "flex"), backend
        self.block_size, self.top_k, self.backend = int(block_size), int(top_k), backend

    @classmethod
    def flops_per_position(cls, d_model, L, cv, num_heads=4, num_kv_heads=None, window_size=-1, block_size=32,
                           top_k=3, **kwargs) -> float:
        """Projections as ``attn``; attention terms (logits 2*ctx*q, scores@V 2*ctx*q, softmax 3*h*ctx) with ctx the
        exact average number of attended keys over a causal sequence of L positions (cv.attn == "full": every
        query sees min(L, top_k*block_size)); gate scores 2*q*n_cand with n_cand the average number of past
        candidate blocks; block mean-pooling ~kv dim. A window caps ctx at w + 1 (upper bound)."""
        from hnet.flops import _attn_dims

        m = _attn_dims(d_model, dict(num_heads=num_heads, num_kv_heads=num_kv_heads))
        B, k = int(block_size), int(top_k)
        n = max(int(round(L)), 1)
        if cv.attn == "full":
            ctx, cand = min(n, k * B), n / B
        else:
            tot_ctx = tot_cand = 0.0
            for b in range((n + B - 1) // B):
                nb_ = min(B, n - b * B)
                tot_ctx += nb_ * (nb_ + 1) / 2 + min(b, k - 1) * B * nb_
                tot_cand += b * nb_
            ctx, cand = tot_ctx / n, tot_cand / n
        if window_size is not None and window_size >= 0 and cv.attn_windows:
            ctx = min(ctx, window_size + 1)
        f = 2 * d_model * (m["q"] + 2 * m["kv"]) + 2 * m["q"] * d_model
        quad = 2 * ctx * m["q"] * 2 + (3 * m["nh"] * ctx if cv.count_elementwise else 0)
        gate = 2 * m["q"] * cand + m["kv"]
        return float(f + quad * cv.attn_quadratic_scale + gate)

    def _attend(self, q, k, v, ctx: SeqCtx) -> torch.Tensor:
        T, H, _ = q.shape
        sel, gid = moba_block_selection(q, k, ctx, self.block_size, self.top_k)
        seq = ctx.seq_idx.reshape(-1)
        w = self.window_size
        backend = self.backend or ("flex" if q.is_cuda and flex_attention is not None else "sdpa")
        gqa = k.shape[1] != H
        qt, kt, vt = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
        if backend == "flex":
            def mask_mod(b, h, qi, ki):
                keep = sel[h, qi, gid[ki]] & (ki <= qi) & (seq[qi] == seq[ki])
                if w >= 0:
                    keep = keep & (qi - ki <= w)
                return keep

            bm = create_block_mask(mask_mod, None, H, T, T, device=q.device, _compile=q.is_cuda)
            fn = _flex_fn() if q.is_cuda else flex_attention
            out = fn(qt, kt, vt, block_mask=bm, scale=self.softmax_scale, enable_gqa=gqa)
        else:
            t = torch.arange(T, device=q.device)
            diff = t[:, None] - t[None, :]
            base = (seq[:, None] == seq[None, :]) & (diff >= 0)
            if w >= 0:
                base = base & (diff <= w)
            mask = sel[:, :, gid] & base  # [H, T, T]
            out = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask.unsqueeze(0), scale=self.softmax_scale,
                                                 enable_gqa=gqa)
        return out.squeeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        hq, hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = self.Wqkv(x[0]).split((hq * hd, hk * hd, hk * hd), dim=-1)
        q, k, v = q.view(T, hq, hd), k.view(T, hk, hd), v.view(T, hk, hd)
        if self.rotary_emb_dim > 0:
            cos, sin = rotary_cos_sin(ctx, self.rotary_emb_dim, self.rope_theta)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = self._attend(q, k, v, ctx)
        return self.out_proj(out.reshape(1, T, hq * hd))

    def extra_repr(self) -> str:
        return super().extra_repr() + f", block_size={self.block_size}, top_k={self.top_k}"


# =============================================================================================== gated attention
@register("mixer", "gated_attn")
class GatedAttention(CausalAttention):
    """Gated attention (Qiu et al., Qwen, "Gated Attention for Large Language Models", arXiv 2505.06708).

    The paper's best variant (G1): a head-specific, elementwise, multiplicative sigmoid gate applied to the SDPA
    output before the output projection, computed from the same (pre-normed) block input X as Q/K/V:

        Y   = SDPA(Q, K, V)                       (per head, [T, H, hd]; causal / GQA / RoPE as ``attn``)
        Y'  = Y * sigmoid(X W_theta)               W_theta: d_model -> H * hd ("elementwise"), d_model -> H ("headwise")
        out = Y' W_O

    Own keys: ``gate`` ("elementwise" | "headwise", default elementwise), ``gate_bias`` (default False, as the paper;
    a bias lets tests force the gate open). All attention keys are honoured exactly as in ``attn``.
    """

    def __init__(self, d_model: int, num_heads: int, layer_idx: int = 0, gate: str = "elementwise",
                 gate_bias: bool = False, **attn_kwargs):
        _check_ignored("gated_attn", attn_kwargs, ATTN_KEYS)
        super().__init__(d_model, num_heads, layer_idx=layer_idx, **attn_kwargs)
        assert gate in ("elementwise", "headwise"), gate
        self.gate = gate
        self.gate_proj = nn.Linear(d_model, num_heads * (self.head_dim if gate == "elementwise" else 1),
                                   bias=gate_bias)

    @classmethod
    def flops_per_position(cls, d_model, L, cv, num_heads=4, gate="elementwise", gate_bias=False, **kwargs) -> float:
        """``attn`` FLOPs + gate projection 2*d*n_gate + sigmoid-and-multiply 5*q (if elementwise ops counted)."""
        from hnet.flops import _attn_dims, attn_flops

        a = dict(kwargs, num_heads=num_heads)
        m = _attn_dims(d_model, a)
        n_gate = m["q"] if gate == "elementwise" else m["nh"]
        return float(attn_flops(d_model, a, L, cv) + 2 * d_model * n_gate + (5 * m["q"] if cv.count_elementwise
                                                                               else 0))

    def forward(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        T = x.shape[1]
        hq, hk, hd = self.num_heads, self.num_kv_heads, self.head_dim
        q, k, v = self.Wqkv(x[0]).split((hq * hd, hk * hd, hk * hd), dim=-1)
        q, k, v = q.view(T, hq, hd), k.view(T, hk, hd), v.view(T, hk, hd)
        if self.rotary_emb_dim > 0:
            cos, sin = rotary_cos_sin(ctx, self.rotary_emb_dim, self.rope_theta)
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        out = attention(q, k, v, ctx, self.window_size, self.softmax_scale)
        gate = torch.sigmoid(self.gate_proj(x[0]).float()).view(T, hq, -1)
        out = (out.float() * gate).to(out.dtype)
        return self.out_proj(out.reshape(1, T, hq * hd))

    def extra_repr(self) -> str:
        return super().extra_repr() + f", gate={self.gate}"
