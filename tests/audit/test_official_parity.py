"""Audit: the default (paper) configuration reproduces the official goombalab/hnet dynamic-chunking forward AND
backward exactly.

The reference below is a line-by-line transcription of the official training-time code paths (packed mode,
``cu_seqlens``), written from reading ``hnet/modules/dc.py``, ``hnet/models/hnet.py`` and ``hnet/utils/train.py`` of
github.com/goombalab/hnet (the official code needs CUDA/Triton kernels, so it is not executed):

* ``RoutingModule.forward``: cos(q(h[:-1]), k(h[1:])), p = clamp((1 - cos)/2, 0, 1), F.pad(p, (1, 0), 1.0),
  ``p[cu_seqlens[:-1]] = 1``, ``boundary_prob = stack(1-p, p)``, ``mask = argmax == 1``,
  ``selected_probs = gather(argmax)``;
* ``ChunkLayer.forward``: ``h[mask]``, ``next_cu = pad(cumsum(mask)[cu[1:]-1], (1, 0))``;
* ``DeChunkLayer.forward``: p = clamp(p, 1e-4, 1-1e-4)[mask]; the Mamba-2 kernel with dt = log(1/(1-p)),
  x = h/dt, A = -1, B = p, C = 1 and seq_idx resets is exactly zbar_j = p_j z_j + (1-p_j) zbar_{j-1} (state 0 at an
  inner segment start), written here as an explicit Python loop; plug-back ``gather(cumsum(mask) - 1)``;
* ``HNet.forward``: pad_dimension, encoder, ``residual = residual_proj(h.float())`` (fp32, zero weight, live bias),
  router, chunk, main, dechunk, ``out * STE(selected_probs) + residual``, decoder, truncate;
* ``load_balancing_loss``: N/(N-1) * ((1-F)(1-G) + F G (N-1)), F = mean(mask), G = mean(p), per stage, summed with
  alpha (paper: L = L_AR + alpha * sum_s L_ratio^s, alpha = 0.03).

The isotropic stacks (Mamba-2 / attention / norms) are *our* modules in both paths, so this test isolates the
hierarchy / DC logic, its gradients, and the loss composition.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from hnet.config import IGNORE_INDEX, HNetConfig
from hnet.model import build_model
from hnet.model.ctx import SeqCtx

torch.set_num_threads(2)


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ones_like(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _ref_routing(rm, h, cu):
    hs = h.unsqueeze(0)
    cos = torch.einsum("bld,bld->bl", F.normalize(rm.q_proj_layer(hs[:, :-1]), dim=-1),
                       F.normalize(rm.k_proj_layer(hs[:, 1:]), dim=-1))
    bp = torch.clamp((1 - cos) / 2, min=0.0, max=1.0)
    bp = F.pad(bp, (1, 0), "constant", 1.0).squeeze(0)
    bp[cu[:-1].long()] = 1.0
    bp = torch.stack((1 - bp, bp), dim=-1)
    idx = torch.argmax(bp, dim=-1)
    mask = idx == 1
    sel = bp.gather(dim=-1, index=idx.unsqueeze(-1))
    return bp, mask, sel


def _ref_dechunk(z, mask, bp, next_cu):
    p = torch.clamp(bp[..., -1].float(), min=1e-4, max=1 - 1e-4)[mask]
    starts = set(next_cu[:-1].tolist())
    rows, state = [], None
    for j in range(z.shape[0]):
        cur = p[j] * z[j]
        state = cur if (j in starts or state is None) else cur + (1 - p[j]) * state
        rows.append(state)
    zbar = torch.stack(rows)
    plug = mask.long().cumsum(0) - 1
    return zbar.index_select(0, plug)


def _ref_stage(st, h, cu, preds):
    D = h.shape[-1]
    if st.pad_dimension is not None:
        h = torch.cat((h, st.pad_dimension.expand(h.shape[0], -1)), dim=-1)
    ctx = SeqCtx.from_cu_seqlens(cu.int())
    if st.is_innermost:
        return st.main_network(h, ctx)[:, :D]
    h = st.encoder(h, ctx)
    residual = st.residual_proj(h.to(st.residual_proj.weight.dtype))
    bp, mask, sel = _ref_routing(st.routing_module, h, cu)
    x = h[mask]
    next_cu = F.pad(mask.long().cumsum(0)[cu[1:].long() - 1], (1, 0))
    preds.append((bp, mask))
    z = _ref_stage(st.main_network, x, next_cu, preds)
    out = _ref_dechunk(z, mask, bp, next_cu)
    h = (out.to(residual.dtype) * _STE.apply(sel) + residual).to(h.dtype)
    h = st.decoder(h, ctx)
    return h[:, :D]


def _ref_load_balancing(bp, mask, N):
    F_ = mask.float().mean()
    G = bp[..., -1].float().mean()
    return ((1 - F_) * (1 - G) + F_ * G * (N - 1)) * N / (N - 1)


def _ref_loss(model, ids, cu, labels):
    preds = []
    h = model.embeddings(ids)
    h = _ref_stage(model.backbone, h, cu, preds)
    logits = model.lm_head(h)
    ce = F.cross_entropy(logits.float(), labels, ignore_index=IGNORE_INDEX)
    ratios = model.config.dc.target_ratios
    aux = sum(_ref_load_balancing(bp, m, float(N)) for (bp, m), N in zip(preds, ratios))
    return logits, ce, model.config.dc.ratio_loss_alpha * aux, preds


def _cfg(n_stages: int) -> HNetConfig:
    if n_stages == 2:
        d = dict(arch_layout=["m2", ["T1m1", ["T2"], "m1T1"], "m2"], d_model=[32, 48, 64],
                 d_intermediate=[0, 96, 128],
                 attn=dict(num_heads=[2, 2, 4], rotary_emb_dim=[8, 12, 8], window_size=[-1, 7, -1]),
                 dc=dict(target_ratios=[3.0, 3.0]))
    else:
        d = dict(arch_layout=["m2", ["T2"], "m2"], d_model=[32, 64], d_intermediate=[0, 128],
                 attn=dict(num_heads=[2, 4], rotary_emb_dim=[8, 8], window_size=[-1, -1]),
                 dc=dict(target_ratios=[6.0]))
    d["ssm"] = dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8)
    return HNetConfig.from_dict(d)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    lens = [37, 1, 70, 20]  # includes a 1-byte segment
    ids = torch.randint(0, 256, (sum(lens),), generator=g)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    labels = torch.full_like(ids, IGNORE_INDEX)
    labels[:-1] = ids[1:]
    labels[cu[1:].long() - 1] = IGNORE_INDEX
    return ids, cu, labels


@pytest.mark.parametrize("n_stages", [1, 2])
def test_default_config_matches_official_forward_and_backward(n_stages):
    torch.manual_seed(0)
    model = build_model(_cfg(n_stages))
    model.init_weights()
    # make the hierarchy non-trivial: random router projections and a non-zero residual projection
    with torch.no_grad():
        for st in model.backbone.stages():
            if not st.is_innermost:
                st.routing_module.q_proj_layer.weight.add_(0.3 * torch.randn_like(st.routing_module.q_proj_layer.weight))
                st.residual_proj.weight.normal_(0, 0.05)
    model.train()
    ids, cu, labels = _batch()

    out = model(ids, cu_seqlens=cu, max_seqlen=int((cu[1:] - cu[:-1]).max()), labels=labels)
    loss = out.loss + out.aux_loss
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    g_ours = torch.autograd.grad(loss, params, allow_unused=True)

    logits, ce, aux, preds = _ref_loss(model, ids, cu, labels)
    g_ref = torch.autograd.grad(ce + aux, params, allow_unused=True)

    for (bp, m), r in zip(preds, out.bpred):
        assert torch.equal(m, r.boundary_mask)
        torch.testing.assert_close(bp.to(r.boundary_prob.dtype), r.boundary_prob, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out.logits, logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.loss, ce, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(out.aux_loss.to(aux.dtype), aux, rtol=1e-5, atol=1e-6)
    for n, a, b in zip(names, g_ours, g_ref):
        if a is None or b is None:
            assert a is None and b is None or (a if a is not None else b).abs().max() == 0, n
            continue
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6, msg=lambda m, n=n: f"{n}: {m}")


def test_residual_proj_bias_is_live_as_in_official():
    """Official ``nn.Linear(d, d)`` keeps PyTorch's default bias (U(-1/sqrt d, 1/sqrt d)); only the weight is zeroed,
    and the released checkpoints contain ``residual_proj.bias``. Ours is the same, so at init the residual path is a
    constant (bias) vector, not exactly zero."""
    torch.manual_seed(0)
    model = build_model(_cfg(2))
    model.init_weights()
    for st in model.backbone.stages():
        if st.is_innermost:
            continue
        assert st.residual_proj.bias is not None
        assert torch.count_nonzero(st.residual_proj.weight) == 0
        assert st.residual_proj.bias.abs().max() > 0
        assert st.residual_proj.weight.dtype == torch.float32
