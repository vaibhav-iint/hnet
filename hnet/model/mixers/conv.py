"""Depthwise causal 1-D convolution that respects packed-sequence boundaries.

``causal_conv1d(x, weight, bias, seq_idx, activation)`` computes, per channel d and position t,

    out[t, d] = act(bias[d] + sum_{k=0..K-1} weight[d, K-1-k] * x[t-k, d] * [seq_idx[t-k] == seq_idx[t]])

i.e. a causal convolution whose window never reaches into a previous segment (zero padding at every segment
start). CUDA tensors use ``causal_conv1d_fn`` from the ``causal_conv1d`` package when it is importable (and
``HNET_FORCE_TORCH != "1"``); otherwise a pure-PyTorch K-tap loop is used.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from hnet.model.ssd import use_kernel

try:  # optional fast path; only a missing package falls back, a broken install raises
    from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn
except ModuleNotFoundError as e:
    if e.name != "causal_conv1d":
        raise
    _causal_conv1d_fn = None
else:  # CUDA kernel is opaque to torch.compile: run it eagerly inside compiled graphs
    _causal_conv1d_fn = torch.compiler.disable(_causal_conv1d_fn)

__all__ = ["causal_conv1d", "causal_conv1d_torch"]


def causal_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """x [b, l, d] (channel-last), weight [d, K], bias [d], seq_idx [b, l] -> [b, l, d] in x's dtype."""
    if use_kernel(x, _causal_conv1d_fn):
        if seq_idx is not None:
            seq_idx = seq_idx.to(torch.int32).contiguous()
        out = _causal_conv1d_fn(x.transpose(1, 2), weight, bias, seq_idx=seq_idx, activation=activation)
        return out.transpose(1, 2)
    return causal_conv1d_torch(x, weight, bias, seq_idx, activation)


def causal_conv1d_torch(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    seq_idx: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
) -> torch.Tensor:
    """Pure-PyTorch :func:`causal_conv1d`. Accumulates in fp32 (like the kernel) and returns x's dtype."""
    assert activation in (None, "silu", "swish")
    bsz, seqlen, _ = x.shape
    width = weight.shape[-1]
    f = torch.float64 if x.dtype == torch.float64 else torch.float32
    w = weight.to(f)
    xp = F.pad(x, (0, 0, width - 1, 0))  # zero history before position 0
    sp = None
    if seq_idx is not None:
        sp = F.pad(seq_idx.to(torch.long), (width - 1, 0), value=-1)
    out = None
    for k in range(width):  # tap k looks k steps back and uses weight[:, width - 1 - k]
        xs = xp[:, width - 1 - k: width - 1 - k + seqlen]
        term = xs.to(f) * w[:, width - 1 - k]
        if sp is not None and k > 0:
            term = term * (sp[:, width - 1 - k: width - 1 - k + seqlen] == seq_idx)[..., None]
        out = term if out is None else out + term
    if bias is not None:
        out = out + bias.to(f)
    if activation is not None:
        out = F.silu(out)
    return out.to(x.dtype)
