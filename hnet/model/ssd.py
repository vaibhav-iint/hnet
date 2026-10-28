"""Mamba-2 SSD (state space duality) chunked scan.

``ssd_chunk_scan`` has the semantics of ``mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined``.
For every batch row and head ``h`` (with group ``g = h // (nheads // ngroups)``) it runs the recurrence

    dt_t  = clamp(softplus?(dt_t + dt_bias), *dt_limit)
    S_t   = exp(dt_t * A) * S_{t-1} + dt_t * x_t B_t^T          S_t: [headdim, d_state]
    y_t   = S_t C_t + D * x_t
    y_t  *= silu(z_t)                                             (only if z is given)

where the state is reset to zero wherever ``seq_idx`` changes (packed variable-length sequences: no information
flows between positions with different ``seq_idx``; segments must be contiguous, as for the kernel).

Dispatch: CUDA tensors use the mamba_ssm Triton kernel when it is importable and ``HNET_FORCE_TORCH != "1"``;
everything else uses :func:`ssd_chunk_scan_torch`, a pure-PyTorch implementation of the chunked SSD algorithm
(paper Listing 1): a quadratic "attention-like" form inside each chunk plus a linear recurrence over chunk states,
the latter in matrix form so there is no Python loop over chunks or positions. It is autograd-differentiable and
computes in fp32 (fp64 for fp64 inputs) regardless of the input dtype.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:  # optional fast path
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _mamba_chunk_scan_combined
except Exception:  # ImportError, or a broken triton/CUDA install
    _mamba_chunk_scan_combined = None

# Read once so the dispatch is a static Python branch (torch.compile friendly). Tests may override it.
FORCE_TORCH = os.environ.get("HNET_FORCE_TORCH", "0") == "1"

__all__ = ["ssd_chunk_scan", "ssd_chunk_scan_torch", "use_kernel"]


def use_kernel(t: torch.Tensor, kernel) -> bool:
    """True if a CUDA kernel (or None if not importable) should be used for tensor ``t``."""
    return kernel is not None and t.is_cuda and not FORCE_TORCH


def ssd_chunk_scan(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    seq_idx: Optional[torch.Tensor] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
):
    """Chunked SSD scan (mamba_chunk_scan_combined semantics).

    Args:
        x: [b, l, h, p] inputs.
        dt: [b, l, h] raw time steps (``dt_bias`` / softplus / ``dt_limit`` are applied here).
        A: [h] (negative) state decay rates.
        B, C: [b, l, g, n] input / output projections; ``h % g == 0`` and head h uses group ``h // (h_total // g)``.
        chunk_size: chunk length. Mathematically irrelevant; it trades memory for speed.
        D: [h] or [h, p] skip connection.
        z: [b, l, h, p] gate; the output is multiplied by ``silu(z)``.
        dt_bias: [h], added to dt before softplus.
        dt_softplus: apply softplus to ``dt + dt_bias``.
        seq_idx: [b, l] integer segment ids (contiguous runs); the state resets wherever it changes.
        dt_limit: (min, max) clamp applied to the processed dt.
        return_final_states: also return the state after the last position, [b, h, p, n].

    Returns:
        y [b, l, h, p] in x's dtype, or (y, final_states) if ``return_final_states``.
    """
    if use_kernel(x, _mamba_chunk_scan_combined):
        kwargs = {} if tuple(dt_limit) == (0.0, float("inf")) else {"dt_limit": tuple(dt_limit)}
        if seq_idx is not None:
            seq_idx = seq_idx.to(torch.int32).contiguous()
        return _mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus,
            seq_idx=seq_idx, return_final_states=return_final_states, **kwargs,
        )
    return ssd_chunk_scan_torch(
        x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=dt_softplus,
        seq_idx=seq_idx, dt_limit=dt_limit, return_final_states=return_final_states,
    )


def _segsum(a: torch.Tensor) -> torch.Tensor:
    """Stable segment sums: out[..., i, j] = sum_{k=j+1..i} a[..., k] for j <= i, and -inf for j > i.

    Each segment is summed directly (masked cumsum) instead of differencing one long cumsum, so there is no
    catastrophic cancellation. a: [..., T] -> [..., T, T]. Used on the (short) chunk axis only.
    """
    T = a.shape[-1]
    idx = torch.arange(T, device=a.device)
    strict = idx[:, None] > idx[None, :]
    seg = a[..., :, None].expand(*a.shape, T).masked_fill(~strict, 0).cumsum(dim=-2)
    return seg.masked_fill(idx[:, None] < idx[None, :], -math.inf)


def _decay_matrix(a: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """exp(a[..., l] - a[..., s]) for kept (l, s) pairs, exactly 0 where ``drop``. a: [..., Q] -> [..., Q, Q].

    The exponent is clamped to +-80 (fp32) / +-700 (fp64) before exp: this changes nothing representable
    (exp(-80) ~ 2e-35) but keeps exp off its very slow denormal/underflow path on CPU, and makes dropped
    (acausal, possibly large positive) entries finite before they are zeroed.
    """
    lim = 700.0 if a.dtype == torch.float64 else 80.0
    seg = a[..., :, None] - a[..., None, :]
    return seg.clamp_(-lim, lim).exp_().masked_fill_(drop, 0.0)


class _IntraChunk(torch.autograd.Function):
    """Diagonal blocks of SSD: y[l] = sum_s CB[l, s] exp(a[l] - a[s]) xdt[s] over kept pairs (s <= l, same segment).

    CB [b, g, c, Q, Q] (shared by the heads of a group), a [b, g, e, c, Q] (in-chunk cumulative log-decay),
    xdt [b, g, e, c, Q, p], drop: bool, broadcastable to [b, g, e, c, Q, Q]. Returns [b, g, e, c, Q, p].

    The [b, h, c, Q, Q] decay matrix is recomputed in backward instead of being saved, so autograd keeps no
    tensor larger than the inputs (this is what makes the pure-torch path memory-lean).
    """

    @staticmethod
    def forward(ctx, CB, a, xdt, drop):
        M = _decay_matrix(a, drop).mul_(CB.unsqueeze(2))
        ctx.save_for_backward(CB, a, xdt, drop)
        return M @ xdt

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gy):
        CB, a, xdt, drop = ctx.saved_tensors
        decay = _decay_matrix(a, drop)
        dM = (gy @ xdt.transpose(-1, -2)).mul_(decay)  # dL/dM * decay
        dCB = dM.sum(dim=2)
        dseg = dM.mul_(CB.unsqueeze(2))  # dL/d(a_l - a_s) = dL/dM * M
        da = dseg.sum(dim=-1) - dseg.sum(dim=-2)
        dxdt = decay.mul_(CB.unsqueeze(2)).transpose(-1, -2) @ gy
        return dCB, da, dxdt, None


def ssd_chunk_scan_torch(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    chunk_size: int,
    D: Optional[torch.Tensor] = None,
    z: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    dt_softplus: bool = False,
    seq_idx: Optional[torch.Tensor] = None,
    dt_limit: Tuple[float, float] = (0.0, float("inf")),
    return_final_states: bool = False,
):
    """Pure-PyTorch :func:`ssd_chunk_scan` (same arguments and results).

    Notation: b batch, c chunks, l/s positions in a chunk (Q = chunk_size), g groups, e heads per group, p headdim,
    n d_state. Internally everything is laid out head-major, [b, g, e, c, Q, ...], so each step is one batched
    matmul. Cost: O(L Q h (n + p)) for the intra-chunk blocks plus O((L/Q)^2 h p n) for the matrix-form chunk
    recurrence (negligible for Q >= 64 at L <= 16k). Transient memory peaks at a few [b, h, L/Q, Q, Q] tensors;
    autograd saves only O(L h (p + n)) per call.
    """
    bsz, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2:]
    assert nheads % ngroups == 0, "nheads must be a multiple of ngroups"
    assert dt.shape == (bsz, seqlen, nheads) and A.shape == (nheads,)
    assert B.shape == C.shape == (bsz, seqlen, ngroups, dstate)
    hpg = nheads // ngroups
    f = torch.float64 if x.dtype == torch.float64 else torch.float32

    # Time steps, fp32.
    dt = dt.to(f)
    if dt_bias is not None:
        dt = dt + dt_bias.to(f)
    if dt_softplus:
        dt = F.softplus(dt)  # threshold 20, like the kernel
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    # Pad to a multiple of chunk_size. Padded positions get dt = 0 (identity decay, no input) and inherit the last
    # segment id, so the final state is the state after the last real position.
    pad = (-seqlen) % chunk_size
    nchunks = (seqlen + pad) // chunk_size
    Q = chunk_size

    def head_major(t, inner):  # [b, L, g * e', *inner] -> [b, g, e', c, Q, *inner] (contiguous)
        if pad:
            t = F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad))
        t = t.reshape(bsz, nchunks, Q, ngroups, -1, *inner)
        return t.permute(0, 3, 4, 1, 2, *range(5, t.dim())).contiguous()

    xdt = head_major(x * dt[..., None], (headdim,))  # [b, g, e, c, Q, p], fp32 by type promotion
    Bh = head_major(B.to(f), (dstate,))  # [b, g, 1, c, Q, n]
    Ch = head_major(C.to(f), (dstate,))
    a = head_major(dt * A.to(f), ()).cumsum(dim=-1)  # [b, g, e, c, Q]: in-chunk cumulative log-decay
    a_last = a[..., -1]  # [b, g, e, c]: total log-decay of each chunk

    pos = torch.arange(Q, device=x.device)
    drop = pos[:, None] < pos[None, :]  # [l, s]: acausal pairs
    if seq_idx is not None:
        # Contiguous segments: position t sees s only if seq_idx[s] == seq_idx[t]; a chunk's state belongs to the
        # segment of the chunk's last position.
        sid = seq_idx.to(torch.long)
        if pad:
            sid = torch.cat([sid, sid[:, -1:].expand(bsz, pad)], dim=1)
        sid = sid.view(bsz, 1, 1, nchunks, Q)
        sid_end = sid[..., -1]  # [b, 1, 1, c]
        sid_prev_end = torch.cat([sid_end[..., :1], sid_end[..., :-1]], dim=-1)
        drop = drop | (sid[..., :, None] != sid[..., None, :])  # [b, 1, 1, c, l, s]

    # 1) Intra-chunk (diagonal blocks).
    CB = Ch.squeeze(2) @ Bh.squeeze(2).transpose(-1, -2)  # [b, g, c, l, s]
    y = _IntraChunk.apply(CB, a, xdt, drop)

    # 2) Each chunk's own end state: S_c = sum_s exp(a_last - a_s) B_s^T xdt_s, stored as [n, p].
    decay_to_end = torch.exp(a_last[..., None] - a)
    if seq_idx is not None:
        decay_to_end = decay_to_end * (sid == sid_end[..., None])
    states = Bh.transpose(-1, -2) @ (xdt * decay_to_end[..., None])  # [b, g, e, c, n, p]

    # 3) Chunk recurrence in matrix form: row i is the state entering chunk i (row c is the final state) =
    #    sum_{j < i} exp(sum of chunk log-decays strictly between j and i) S_j.
    w = _segsum(F.pad(a_last, (1, 0)))[..., 1:]  # [b, g, e, c + 1, c]
    if seq_idx is not None:
        w = w.masked_fill(F.pad(sid_end[..., :, None] != sid_end[..., None, :], (0, 0, 1, 0), value=True), -math.inf)
    states = (torch.exp(w) @ states.flatten(-2)).unflatten(-1, (dstate, headdim))  # [b, g, e, c + 1, n, p]

    # 4) Incoming-state contribution (off-diagonal blocks): y_l += exp(a_l) C_l S_in.
    decay_in = torch.exp(a)
    if seq_idx is not None:
        decay_in = decay_in * (sid == sid_prev_end[..., None])
    y = torch.addcmul(y, Ch @ states[..., :-1, :, :], decay_in[..., None])

    # Back to [b, L, h, p]; skip connection and gate.
    y = y.permute(0, 3, 4, 1, 2, 5).reshape(bsz, nchunks * Q, nheads, headdim)[:, :seqlen]
    if D is not None:
        y = y + x * (D.to(f)[:, None] if D.dim() == 1 else D.to(f))
    if z is not None:
        y = y * F.silu(z.to(f))
    y = y.to(x.dtype)
    if return_final_states:
        final = states[..., -1, :, :].transpose(-1, -2).reshape(bsz, nheads, headdim, dstate)
        return y, final.to(C.dtype)
    return y
