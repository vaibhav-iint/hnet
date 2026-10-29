"""Upsamplers back to the outer resolution (paper Sec. 2.2.2), registry kind ``dechunker``.

Contract: ``(h_inner [M, D], router_out, ctx, ctx_inner) -> h_outer [T, D]`` (fp32); ``smooth(...)`` returns the
inner-rate [M, D] fp32 vectors that ``__call__`` upsamples (used by the offset-resolved read-back, ``dc.dechunk_slots``). Outer position t reads inner
position ``plug_back_idx[t] = cumsum(boundary_mask)[t] - 1``: the last selected position at or before t (eq. 8).
For "start" semantics that is the chunk containing t; for "end" semantics it is the latest chunk that ended at or
before t (see ``routing.base``): the same index serves both.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hnet.registry import register

from ..ctx import SeqCtx
from ..routing.base import SEMANTICS, RouterOutput
from .scan import linear_recurrence

P_MIN = 1e-4


def plug_back_index(router_out: RouterOutput) -> torch.Tensor:
    """int64 [T]: index of the inner position each outer position reads (valid for both boundary semantics)."""
    assert router_out.semantics in SEMANTICS, router_out.semantics
    return router_out.boundary_mask.long().cumsum(0) - 1


@register("dechunker", "repeat")
class RepeatDechunker:
    """No smoothing: ``out_t = z_{plug_back_idx[t]}`` (paper ablation "w/o smoothing")."""

    def smooth(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        """[M, D] fp32 inner-rate vectors before upsampling (here: the inner outputs themselves)."""
        return h_inner.float()

    def __call__(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        return self.smooth(h_inner, router_out, ctx, ctx_inner).index_select(0, plug_back_index(router_out))


@register("dechunker", "ema")
class EMADechunker:
    """Smoothing module (eq. 5) then upsampling:

        P_j = clamp(p at the j-th selected position, 1e-4, 1 - 1e-4)
        zbar_j = P_j z_j + (1 - P_j) zbar_{j-1}      (state reset at every inner segment start)
        out_t = zbar_{plug_back_idx[t]}

    computed in fp32 with :func:`linear_recurrence` (``block`` positions per matmul block).
    """

    def __init__(self, block: int = 64):
        self.block = block

    def smooth(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        """zbar [M, D] fp32: the smoothed inner vectors before upsampling."""
        P = router_out.p.clamp(P_MIN, 1 - P_MIN).index_select(0, ctx_inner.outer_index)
        return linear_recurrence(P[:, None] * h_inner.float(), torch.log1p(-P), ctx_inner.position_ids == 0, self.block)

    def __call__(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        return self.smooth(h_inner, router_out, ctx, ctx_inner).index_select(0, plug_back_index(router_out))


class SlotReadback(nn.Module):
    """Offset-resolved read-back (``HNetConfig.dc.dechunk_slots`` = K), wrapping a dechunker with ``smooth``:

        zbar = base.smooth(h_inner)                     [M, D] fp32 (EMA-smoothed chunk vectors)
        out_t = W_{min(k_t, K-1)} zbar_{j(t)}           j(t) = plug_back_index[t], k_t = t - outer position of j(t)

    Under "end" semantics k_t is the number of bytes of the next chunk byte t has already seen (the boundary byte
    itself has k = 0). The K maps (``slot_proj`` [K*D, D], rows k*D:(k+1)*D = W_k) are identity-initialized without
    consuming RNG, so at init the output equals ``base(...)`` exactly; they run once per inner position and each byte
    gathers its row. Causal: j(t) and k_t depend only on boundary decisions at or before t. Logs ``slot_div`` (mean
    ||W_k - W_0|| / ||W_0||, k >= 1) and ``slot_last_frac`` (share of bytes in the shared last slot) to
    ``router_out.extra``.
    """

    def __init__(self, d_model: int, n_slots: int, base):
        super().__init__()
        assert n_slots >= 1 and hasattr(base, "smooth"), (n_slots, type(base))
        self.n_slots, self.base = int(n_slots), base
        self.slot_proj = nn.Parameter(torch.eye(d_model, dtype=torch.float32).repeat(self.n_slots, 1))

    def forward(self, h_inner: torch.Tensor, router_out: RouterOutput, ctx: SeqCtx, ctx_inner: SeqCtx):
        K, D = self.n_slots, h_inner.shape[-1]
        zbar = self.base.smooth(h_inner, router_out, ctx, ctx_inner)  # [M, D] fp32
        with torch.autocast(device_type=zbar.device.type, enabled=False):
            y = torch.nn.functional.linear(zbar, self.slot_proj.float())  # [M, K * D]
        plug = plug_back_index(router_out)
        offset = torch.arange(plug.numel(), device=plug.device) - ctx_inner.outer_index.index_select(0, plug)
        slot = offset.clamp(max=K - 1)
        with torch.no_grad():
            W = self.slot_proj.detach().float().view(K, D, D)
            router_out.extra["slot_div"] = ((W[1:] - W[:1]).pow(2).sum((1, 2)).sqrt()
                                            / W[0].pow(2).sum().sqrt()).mean() if K > 1 else W.new_zeros(())
            router_out.extra["slot_last_frac"] = (slot == K - 1).float().mean()
        return y.view(-1, D).index_select(0, plug * K + slot)
