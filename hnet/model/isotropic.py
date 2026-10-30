"""A stack of residual blocks followed by a final RMSNorm (goombalab/hnet ``Isotropic``).

Layout strings are sequences of ``<code><count>``, e.g. ``"T1m4"``. The lower-cased code is looked up in
``mixer_codes`` (e.g. ``{"m": "mamba2", "t": "attn"}``) to get a mixer registry name; an upper-case code adds a
SwiGLU MLP to each of its blocks. Adding a mixer is therefore a registry entry plus a ``mixer_codes`` entry.

Optional **Attention Residuals** (Kimi Team, "Attention Residuals", arXiv 2603.15031; github.com/MoonshotAI/
Attention-Residuals), off by default (``attn_res=0``: standard pre-norm residuals, unchanged behaviour and
parameters). With ``attn_res = S > 0`` the fixed residual sum is replaced by softmax attention over depth. Every
mixer and every MLP is one "layer" (sub-layer) l with its own pseudo-query w_l in R^d (zero-init) and RMSNorm:

    h_l   = sum_i alpha_{i->l} v_i,    alpha_{i->l} = softmax_i( w_l^T RMSNorm(v_i) )        (Eqs. 2-4)
    out_l = f_l(norm_l(h_l))                                                    (the block's own pre-norm)

Block AttnRes (Eqs. 5-6): sub-layers are grouped into blocks of S; the sources v_i are b_0 = the stack input
(the "embedding"), the completed block sums b_1..b_{n-1} (b_n = sum of its sub-layer outputs) and, after the first
sub-layer of a block, the partial sum of the current block. S = 1 is Full AttnRes (every sub-layer output is a
source). The stack output is one more AttnRes (own pseudo-query) over all block representations, followed by the
final RMSNorm. Sources are kept in fp32 like the residual stream.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import torch
import torch.nn as nn

from .block import Block
from .ctx import SeqCtx
from .norm import RMSNorm

_TOKEN = re.compile(r"([A-Za-z])(\d+)")


def parse_layout(layout: str, mixer_codes: dict) -> List[Tuple[str, bool]]:
    """``"T1m4"`` -> ``[("attn", True), ("mamba2", False) x4]`` (mixer name, has_mlp) per layer."""
    if re.fullmatch(r"(?:[A-Za-z]\d+)*", layout) is None:
        raise ValueError(f"bad layout {layout!r}; expected e.g. 'm4' or 'T1m4'")
    out = []
    for code, n in _TOKEN.findall(layout):
        if code.lower() not in mixer_codes:
            raise KeyError(f"layout code {code!r} not in mixer_codes {sorted(mixer_codes)}")
        out += [(mixer_codes[code.lower()], code.isupper())] * int(n)
    return out


class DepthAttention(nn.Module):
    """One AttnRes operator: ``sum_i softmax_i(w^T RMSNorm(v_i)) v_i`` over a list of [1, T, D] sources (fp32).
    ``query`` (w) is zero-initialized, so the initial weights are uniform (paper Sec. 5)."""

    def __init__(self, d_model: int, norm_eps: float = 1e-5):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d_model))
        self.query._no_weight_decay = True
        self.norm = RMSNorm(d_model, eps=norm_eps)

    def forward(self, sources: List[torch.Tensor]) -> torch.Tensor:
        if len(sources) == 1:
            return sources[0]
        V = torch.stack(sources)  # [N, 1, T, D] fp32
        logits = (self.norm(V, out_dtype=torch.float32) * self.query.float()).sum(-1)
        return (logits.softmax(0).unsqueeze(-1) * V).sum(0)


class Isotropic(nn.Module):
    """``forward(h [T, D] or [1, T, D], ctx) -> same shape``.

    ``mixer_kwargs`` maps mixer registry name -> constructor kwargs. ``layer_idx`` counts from 0 inside this
    stack (as official; it drives e.g. NoPE layer selection). ``height`` = number of residual additions (one per
    mixer, one per MLP); it sets the depth-scaled init in ``init_linear_weights``. ``attn_res`` (default 0 = off):
    Block Attention Residuals with blocks of ``attn_res`` sub-layers (1 = Full AttnRes); see the module docstring.
    """

    def __init__(
        self,
        layout: str,
        d_model: int,
        d_intermediate: int,
        mixer_codes: dict,
        mixer_kwargs: dict,
        norm_eps: float = 1e-5,
        attn_res: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        spec = parse_layout(layout, mixer_codes)
        if any(mlp for _, mlp in spec):
            assert d_intermediate > 0, f"layout {layout!r} has MLP blocks but d_intermediate == 0"
        self.layers = nn.ModuleList(
            Block(
                d_model,
                name,
                mixer_kwargs.get(name, {}),
                d_intermediate=d_intermediate if mlp else 0,
                layer_idx=i,
                norm_eps=norm_eps,
            )
            for i, (name, mlp) in enumerate(spec)
        )
        self.height = sum(2 if mlp else 1 for _, mlp in spec)
        self.rmsnorm = RMSNorm(d_model, eps=norm_eps)
        self.attn_res_block = int(attn_res)
        if self.attn_res_block > 0:  # one operator per sub-layer + one for the output
            self.attn_res = nn.ModuleList(DepthAttention(d_model, norm_eps) for _ in range(self.height + 1))

    def forward(self, h: torch.Tensor, ctx: SeqCtx, inject=None, memory=None) -> torch.Tensor:
        """``inject``: optional ``(i, x)``, x [T, D] (or [1, T, D]) added to the residual stream (fp32) right before
        layer i, 0 < i < len(layers) (DenseConfig.match_model's decoder-side candidates). ``memory``: optional
        ``(i, f)``, 0 < i <= len(layers): the residual stream s right before layer i (after the last layer for
        i = len(layers)) becomes s + f(s) (HNetConfig.main_memory / DenseConfig.memory; f returns fp32)."""
        if memory is not None:
            assert self.attn_res_block == 0, "memory is not combined with attn_res"
            return self._forward_memory(h, ctx, *memory, inject=inject)
        squeeze = h.dim() == 2
        for layer in self.layers:  # shared per-stage tables (rotary, masks) built eagerly, outside compiled blocks
            prep = getattr(layer.mixer, "prepare_ctx", None)
            if prep is not None:
                prep(ctx)
        if self.attn_res_block > 0:
            assert inject is None, "inject is not supported with attn_res"
            out = self._forward_attn_res(h.unsqueeze(0) if squeeze else h, ctx)
            return out.squeeze(0) if squeeze else out
        hidden, residual = (h.unsqueeze(0) if squeeze else h), None
        if inject is not None:
            at, extra = inject
            assert 0 < at < len(self.layers), at
            for i, layer in enumerate(self.layers):
                if i == at:  # residual + hidden is what layer i normalizes: add to the fp32 residual stream
                    residual = residual + extra.float().reshape(residual.shape)
                hidden, residual = layer(hidden, residual, ctx)
        else:
            for layer in self.layers:
                hidden, residual = layer(hidden, residual, ctx)
        residual = hidden.float() if residual is None else residual + hidden.float()
        out = self.rmsnorm(residual, out_dtype=hidden.dtype)
        return out.squeeze(0) if squeeze else out

    def _forward_memory(self, h: torch.Tensor, ctx: SeqCtx, at: int, fn, inject=None) -> torch.Tensor:
        """``forward`` with ``memory = (at, fn)`` (and optionally ``inject``, applied first at its layer)."""
        assert 0 < at <= len(self.layers), at
        for layer in self.layers:
            prep = getattr(layer.mixer, "prepare_ctx", None)
            if prep is not None:
                prep(ctx)
        squeeze = h.dim() == 2
        hidden, residual = (h.unsqueeze(0) if squeeze else h), None
        if inject is not None:
            assert 0 < inject[0] < len(self.layers), inject[0]
        for i, layer in enumerate(self.layers):
            if inject is not None and i == inject[0]:
                residual = residual + inject[1].float().reshape(residual.shape)
            if i == at:  # the stream before layer i is residual + hidden; the block adds hidden itself
                residual = residual + fn(residual + hidden.float())
            hidden, residual = layer(hidden, residual, ctx)
        residual = hidden.float() if residual is None else residual + hidden.float()
        if at == len(self.layers):
            residual = residual + fn(residual)
        out = self.rmsnorm(residual, out_dtype=hidden.dtype)
        return out.squeeze(0) if squeeze else out

    def _forward_attn_res(self, x: torch.Tensor, ctx: SeqCtx) -> torch.Tensor:
        """Block AttnRes forward (Eqs. 2-6). Runs each Block's norms / mixer / MLP directly (not ``Block.forward``)."""
        dtype = x.dtype
        blocks, partial, j = [x.float()], None, 0
        for layer in self.layers:
            subs = [(layer.norm1, lambda u, m=layer.mixer: m(u, ctx))]
            if layer.mlp is not None:
                subs.append((layer.norm2, layer.mlp))
            for norm, fn in subs:
                h = self.attn_res[j](blocks if partial is None else blocks + [partial])
                out = fn(norm(h, out_dtype=dtype)).float()
                partial = out if partial is None else partial + out
                j += 1
                if j % self.attn_res_block == 0:
                    blocks.append(partial)
                    partial = None
        if partial is not None:
            blocks.append(partial)
        return self.rmsnorm(self.attn_res[j](blocks), out_dtype=dtype)

    def init_linear_weights(self, std: float, n_residuals: int):
        """Official ``_init_weights`` for this stack: every ``nn.Linear`` weight not flagged ``_no_reinit`` gets
        N(0, std); ``out_proj``/``fc2`` (the residual writers) get N(0, std / sqrt(n_residuals))."""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear) and not getattr(m.weight, "_no_reinit", False):
                scale = n_residuals ** -0.5 if ("out_proj" in name or "fc2" in name) else 1.0
                nn.init.normal_(m.weight, mean=0.0, std=std * scale)
