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

    Notation: b batch, c chunks, l/s positions in a chunk (Q = chunk_size), g groups, e heads per group,
    p headdim, n d_state. Work is O(L Q (n + p) h) for the intra-chunk part plus O((L/Q)^2 h p n) for the
    matrix-form chunk recurrence; the largest intermediates are [b, c, h, Q, Q].
    """
    bsz, seqlen, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2:]
    assert nheads % ngroups == 0, "nheads must be a multiple of ngroups"
    assert dt.shape == (bsz, seqlen, nheads) and A.shape == (nheads,)
    assert B.shape == C.shape == (bsz, seqlen, ngroups, dstate)
    hpg = nheads // ngroups
    out_dtype = x.dtype
    f = torch.float64 if x.dtype == torch.float64 else torch.float32

    # Time steps and per-position log-decays, fp32.
    dt = dt.to(f)
    if dt_bias is not None:
        dt = dt + dt_bias.to(f)
    if dt_softplus:
        dt = F.softplus(dt)  # threshold 20, like the kernel
    dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])

    # Pad to a multiple of chunk_size. Padded positions get dt = 0 (identity decay, no input), and inherit the
    # last segment id, so the final state is the state after the last real position.
    pad = (-seqlen) % chunk_size
    nchunks = (seqlen + pad) // chunk_size

    def chunked(t):  # [b, L, ...] -> [b, c, Q, ...]
        if pad:
            t = F.pad(t, (0, 0) * (t.dim() - 2) + (0, pad))
        return t.reshape(bsz, nchunks, chunk_size, *t.shape[2:])

    xc = chunked(x.to(f)).view(bsz, nchunks, chunk_size, ngroups, hpg, headdim)
    dtc = chunked(dt).view(bsz, nchunks, chunk_size, ngroups, hpg)
    Bc = chunked(B.to(f))
    Cc = chunked(C.to(f))
    xdt = xc * dtc[..., None]  # [b, c, l, g, e, p]
    # Cumulative log-decay inside each chunk, laid out [b, c, g, e, l].
    a_cs = (dtc * A.to(f).view(ngroups, hpg)).permute(0, 1, 3, 4, 2).cumsum(dim=-1)
    a_last = a_cs[..., -1]  # [b, c, g, e]: total log-decay of each chunk

    # Segment masks (None without seq_idx): same[b, c, l, s], at_end[b, c, l] (same segment as the chunk's last
    # position), from_prev[b, c, l] (same segment as the previous chunk's last position), chunk_eq[b, c, c'].
    if seq_idx is not None:
        sid = seq_idx.to(torch.long)
        if pad:
            sid = torch.cat([sid, sid[:, -1:].expand(bsz, pad)], dim=1)
        sid = sid.view(bsz, nchunks, chunk_size)
        sid_end = sid[:, :, -1]  # [b, c]
        same = sid[:, :, :, None] == sid[:, :, None, :]
        at_end = sid == sid_end[:, :, None]
        sid_prev_end = torch.cat([sid_end[:, :1], sid_end[:, :-1]], dim=1)
        from_prev = sid == sid_prev_end[:, :, None]
        chunk_eq = sid_end[:, :, None] == sid_end[:, None, :]

    # 1) Intra-chunk (diagonal blocks): y_l = sum_{s<=l} (C_l . B_s) exp(a_cs_l - a_cs_s) dt_s x_s.
    pos = torch.arange(chunk_size, device=x.device)
    keep = pos[:, None] >= pos[None, :]  # [l, s] causal
    if seq_idx is not None:
        keep = keep & same  # [b, c, l, s]
        keep = keep[:, :, None, None]  # [b, c, 1, 1, l, s]
    CB = torch.einsum("bclgn,bcsgn->bcgls", Cc, Bc)  # shared by the heads of a group
    decay = a_cs[..., :, None] - a_cs[..., None, :]  # [b, c, g, e, l, s]
    decay = torch.exp(decay.masked_fill(~keep, -math.inf))
    y = torch.einsum("bcgels,bcsgep->bclgep", CB[:, :, :, None] * decay, xdt)

    # 2) State of each chunk (from its own inputs only): S_c = sum_s exp(a_last - a_cs_s) dt_s x_s B_s^T.
    decay_to_end = torch.exp(a_last[..., None] - a_cs)  # [b, c, g, e, l]
    if seq_idx is not None:
        decay_to_end = decay_to_end * at_end[:, :, None, None]
    states = torch.einsum("bclgep,bclgn->bcgepn", xdt * decay_to_end.permute(0, 1, 4, 2, 3)[..., None], Bc)

    # 3) Inter-chunk recurrence in matrix form. Row i of `w` gives the state entering chunk i (row nchunks =
    #    final state): sum over earlier chunks j of exp(sum of chunk log-decays strictly between them) S_j.
    w = _segsum(F.pad(a_last.permute(0, 2, 3, 1), (1, 0)))[..., 1:]  # [b, g, e, c+1, c]
    if seq_idx is not None:
        w = w.masked_fill(~F.pad(chunk_eq, (0, 0, 1, 0))[:, None, None], -math.inf)
    states = torch.einsum("bgeij,bjgepn->bigepn", torch.exp(w), states)  # [b, c+1, g, e, p, n]

    # 4) Contribution of the incoming state (off-diagonal blocks): y_l += exp(a_cs_l) C_l . S_in.
    decay_in = torch.exp(a_cs)  # [b, c, g, e, l]
    if seq_idx is not None:
        decay_in = decay_in * from_prev[:, :, None, None]
    y_off = torch.einsum("bclgn,bcgepn->bclgep", Cc, states[:, :-1])
    y = y + y_off * decay_in.permute(0, 1, 4, 2, 3)[..., None]

    y = y.reshape(bsz, nchunks * chunk_size, nheads, headdim)[:, :seqlen]
    if D is not None:
        Df = D.to(f)
        y = y + x.to(f) * (Df[:, None] if Df.dim() == 1 else Df)
    if z is not None:
        y = y * F.silu(z.to(f))
    y = y.to(out_dtype)
    if return_final_states:
        final = states[:, -1].reshape(bsz, nheads, headdim, dstate).to(C.dtype)
        return y, final
    return y
