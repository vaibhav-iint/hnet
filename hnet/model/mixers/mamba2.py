"""Mamba-2 mixer (registered as mixer ``"mamba2"``).

Parameters, shapes and initialization match ``mamba_ssm.modules.mamba2.Mamba2`` with its defaults
(``rmsnorm=True, norm_before_gate=False, D_has_hdim=False, bias=False, conv_bias=True, d_ssm=None``), so
goombalab/hnet (``cartesia-ai/hnet_*``) checkpoints load with ``load_state_dict``:

    in_proj.weight  [2*d_inner + 2*ngroups*d_state + nheads, d_model]   rows: [z | x | B | C | dt]
    conv1d.weight   [conv_dim, 1, d_conv], conv1d.bias [conv_dim]         conv_dim = d_inner + 2*ngroups*d_state
    dt_bias, A_log, D  [nheads]
    norm.weight     [d_inner]   gated RMSNorm: rmsnorm(y * silu(z)) per group of d_inner // ngroups channels
    out_proj.weight [d_model, d_inner]

Forward: ``x -> in_proj -> (z, xBC, dt)``; ``xBC -> causal depthwise conv -> silu -> (x, B, C)``;
``y = SSD(x, dt, A=-exp(A_log), B, C, D, dt_bias, softplus)``; ``out_proj(norm(y, z))``.

Execution paths (chosen per call):
* CUDA with ``mamba_ssm`` and ``causal_conv1d`` (and ``HNET_FORCE_TORCH != "1"``): the fused, memory-efficient
  ``mamba_split_conv1d_scan_combined`` (the official default path);
* otherwise: :func:`hnet.model.mixers.conv.causal_conv1d` + :func:`hnet.model.ssd.ssd_chunk_scan` + gated RMSNorm,
  each of which uses its own kernel when available and pure PyTorch otherwise.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hnet.model.ctx import SeqCtx
from hnet.model.mixers.conv import _causal_conv1d_fn, causal_conv1d
from hnet.model.ssd import ssd_chunk_scan, use_kernel
from hnet.registry import register

try:  # optional fast paths
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as _rmsnorm_gated_fn
    from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined as _split_conv1d_scan
except Exception:
    _rmsnorm_gated_fn = _split_conv1d_scan = None

__all__ = ["Mamba2", "RMSNormGated"]


class RMSNormGated(nn.Module):
    """``rmsnorm(x * silu(z)) * weight`` (``norm_before_gate=False``), normalized over groups of ``group_size``
    channels, computed in fp32 and returned in x's dtype. Same parameters and semantics as
    ``mamba_ssm.ops.triton.layernorm_gated.RMSNorm``; uses its Triton kernel on CUDA when available."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, group_size: Optional[int] = None, device=None,
                 dtype=None):
        super().__init__()
        self.eps = eps
        self.group_size = hidden_size if group_size is None else group_size
        assert hidden_size % self.group_size == 0
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor, z: Optional[torch.Tensor] = None) -> torch.Tensor:
        if use_kernel(x, _rmsnorm_gated_fn):
            return _rmsnorm_gated_fn(x, self.weight, None, z=z, eps=self.eps, group_size=self.group_size,
                                     norm_before_gate=False)
        f = torch.float64 if x.dtype == torch.float64 else torch.float32
        h = x.to(f)
        if z is not None:
            h = h * F.silu(z.to(f))
        h = h.unflatten(-1, (-1, self.group_size))
        h = h * torch.rsqrt(h.square().mean(dim=-1, keepdim=True) + self.eps)
        return (h.flatten(-2) * self.weight.to(f)).to(x.dtype)


@register("mixer", "mamba2")
class Mamba2(nn.Module):
    """Mamba-2 block mixer.

    ``forward(x, ctx=None)``: x is [1, T, D] packed (``ctx.seq_idx`` [1, T] marks the segments; neither the
    convolution nor the SSM state crosses a segment boundary) or [B, L, D] with ``ctx=None`` (each row one
    sequence). Returns [*, *, D] in x's dtype. The SSD scan runs in fp32 internally.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        layer_idx: Optional[int] = None,
        device=None,
        dtype=None,
        A_init_range=(1, 16),
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit=(0.0, float("inf")),
        **ignored,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % headdim == 0, "d_inner must be a multiple of headdim"
        self.nheads = self.d_inner // headdim
        assert self.nheads % ngroups == 0, "nheads must be a multiple of ngroups"
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.dt_limit = tuple(dt_limit)
        self.A_init_range = tuple(A_init_range)
        self.dt_init = (dt_min, dt_max, dt_init_floor)

        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.in_proj = nn.Linear(d_model, self.d_inner + conv_dim + self.nheads, bias=False, **factory)
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1, bias=True,
                                **factory)
        self.dt_bias = nn.Parameter(torch.empty(self.nheads, **factory))
        self.A_log = nn.Parameter(torch.empty(self.nheads, **factory))
        self.D = nn.Parameter(torch.empty(self.nheads, device=device))  # fp32, as in mamba_ssm
        for p in (self.dt_bias, self.A_log, self.D):
            p._no_weight_decay = True
        self.norm = RMSNormGated(self.d_inner, eps=1e-5, group_size=self.d_inner // ngroups, **factory)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False, **factory)
        self.reset_ssm_parameters()

    @torch.no_grad()
    def reset_ssm_parameters(self):
        """(Re)initialize dt_bias, A_log and D exactly as mamba_ssm does (e.g. after building on the meta device).

        dt ~ exp(U[log dt_min, log dt_max]) clamped to >= dt_init_floor, dt_bias = softplus^{-1}(dt);
        A ~ U[A_init_range], A_log = log A; D = 1.
        """
        dt_min, dt_max, dt_floor = self.dt_init
        dev = self.dt_bias.device
        dt = torch.exp(torch.rand(self.nheads, device=dev) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_floor)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log.copy_(torch.empty(self.nheads, device=dev).uniform_(*self.A_init_range).log())
        self.D.fill_(1.0)

    def _seq_idx(self, x: torch.Tensor, ctx: Optional[SeqCtx]) -> Optional[torch.Tensor]:
        if ctx is None or ctx.seq_idx is None:
            return None
        seq_idx = ctx.seq_idx
        if seq_idx.dim() == 1:
            seq_idx = seq_idx.unsqueeze(0)
        assert seq_idx.shape[-1] == x.shape[1], "ctx.seq_idx must cover every position of x"
        return seq_idx.to(torch.int32).expand(x.shape[0], -1).contiguous()

    def forward(self, x: torch.Tensor, ctx: Optional[SeqCtx] = None) -> torch.Tensor:
        assert x.dim() == 3, "Mamba2 expects x of shape [B, L, D] (packed: [1, T, D])"
        seq_idx = self._seq_idx(x, ctx)
        zxbcdt = self.in_proj(x)
        A = -torch.exp(self.A_log.float())
        dt_limit = {} if self.dt_limit == (0.0, float("inf")) else {"dt_limit": self.dt_limit}

        if use_kernel(x, _split_conv1d_scan) and _causal_conv1d_fn is not None:
            return _split_conv1d_scan(
                zxbcdt, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.dt_bias, A, D=self.D,
                chunk_size=self.chunk_size, seq_idx=seq_idx, activation="silu",
                rmsnorm_weight=self.norm.weight, rmsnorm_eps=self.norm.eps,
                outproj_weight=self.out_proj.weight, outproj_bias=self.out_proj.bias,
                headdim=self.headdim, ngroups=self.ngroups, norm_before_gate=False, **dt_limit,
            )

        bsz, seqlen, _ = x.shape
        gn = self.ngroups * self.d_state
        z, xBC, dt = torch.split(zxbcdt, [self.d_inner, self.d_inner + 2 * gn, self.nheads], dim=-1)
        xBC = causal_conv1d(xBC, self.conv1d.weight.squeeze(1), self.conv1d.bias, seq_idx, activation="silu")
        xs, B, C = torch.split(xBC, [self.d_inner, gn, gn], dim=-1)
        y = ssd_chunk_scan(
            xs.unflatten(-1, (self.nheads, self.headdim)),
            dt,
            A,
            B.unflatten(-1, (self.ngroups, self.d_state)),
            C.unflatten(-1, (self.ngroups, self.d_state)),
            self.chunk_size,
            D=self.D,
            dt_bias=self.dt_bias,
            dt_softplus=True,
            seq_idx=seq_idx,
            **dt_limit,
        )
        y = self.norm(y.reshape(bsz, seqlen, self.d_inner), z)
        return self.out_proj(y)
