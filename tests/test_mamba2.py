"""Tests for the Mamba2 mixer (hnet.model.mixers.mamba2) and its causal conv helper."""

import pytest
import torch
import torch.nn.functional as F

from hnet.model import ssd
from hnet.model.ctx import SeqCtx
from hnet.model.mixers import conv as conv_mod
from hnet.model.mixers import mamba2 as mamba2_mod
from hnet.model.mixers.conv import causal_conv1d_torch
from hnet.model.mixers.mamba2 import Mamba2, RMSNormGated
from hnet.registry import _REGISTRY

torch.set_num_threads(1)


def tiny(**kw):
    torch.manual_seed(0)
    cfg = dict(d_model=16, d_state=8, d_conv=4, expand=2, headdim=8, ngroups=2, chunk_size=8)
    cfg.update(kw)
    return Mamba2(**cfg).double()


def test_registered():
    assert _REGISTRY["mixer"]["mamba2"] is Mamba2


def test_state_dict_matches_mamba_ssm_d1024():
    """Keys/shapes of mamba_ssm.modules.mamba2.Mamba2(d_model=1024) with defaults (d_state=128, expand=2,
    headdim=64, ngroups=1, d_conv=4): d_inner=2048, nheads=32, conv_dim=2048+2*128, in_proj rows 2*2048+2*128+32."""
    expected = {
        "dt_bias": (32,),
        "A_log": (32,),
        "D": (32,),
        "in_proj.weight": (4384, 1024),
        "conv1d.weight": (2304, 1, 4),
        "conv1d.bias": (2304,),
        "norm.weight": (2048,),
        "out_proj.weight": (1024, 2048),
    }
    m = Mamba2(1024, device="meta")
    assert {k: tuple(v.shape) for k, v in m.state_dict().items()} == expected
    assert m.D.dtype == torch.float32


def test_init_matches_mamba_ssm():
    m = Mamba2(64, d_state=16, headdim=4, layer_idx=3, rmsnorm=True, use_mem_eff_path=False)  # extras ignored
    dt = F.softplus(m.dt_bias.detach())
    assert dt.min() >= 1e-4 - 1e-7 and dt.max() <= 0.1 + 1e-6
    A = m.A_log.detach().exp()
    assert A.min() >= 1 and A.max() <= 16
    assert torch.equal(m.D.detach(), torch.ones(32)) and torch.equal(m.norm.weight.detach(), torch.ones(128))
    assert m.in_proj.bias is None and m.out_proj.bias is None and m.conv1d.bias is not None
    for p in (m.A_log, m.D, m.dt_bias):
        assert getattr(p, "_no_weight_decay", False)
    assert not hasattr(m.in_proj.weight, "_no_weight_decay")


def test_conv_matches_nn_conv1d_and_respects_segments():
    torch.manual_seed(0)
    conv = torch.nn.Conv1d(6, 6, 4, groups=6, padding=3).double()
    x = torch.randn(2, 13, 6, dtype=torch.float64)
    ref = F.silu(conv(x.transpose(1, 2))[..., :13]).transpose(1, 2)  # mamba_ssm's torch path
    out = causal_conv1d_torch(x, conv.weight.squeeze(1), conv.bias, None, "silu")
    torch.testing.assert_close(out, ref)
    seq_idx = torch.tensor([[0] * 5 + [1] * 8] * 2, dtype=torch.int32)
    out = causal_conv1d_torch(x, conv.weight.squeeze(1), conv.bias, seq_idx, "silu")
    torch.testing.assert_close(out[:, :5], ref[:, :5])
    alone = F.silu(conv(x[:, 5:].transpose(1, 2))[..., :8]).transpose(1, 2)
    torch.testing.assert_close(out[:, 5:], alone)


def test_gated_rmsnorm_reference():
    """Matches mamba_ssm's rms_norm_ref(..., group_size, norm_before_gate=False)."""
    torch.manual_seed(0)
    norm = RMSNormGated(12, eps=1e-5, group_size=4).double()
    norm.weight.data.uniform_(0.5, 1.5)
    x, z = torch.randn(3, 12, dtype=torch.float64), torch.randn(3, 12, dtype=torch.float64)
    h = (x * F.silu(z)).view(3, 3, 4)
    ref = (h / torch.sqrt(h.square().mean(-1, keepdim=True) + 1e-5)).view(3, 12) * norm.weight
    torch.testing.assert_close(norm(x, z), ref)


def test_forward_matches_unfused_reference():
    """Independent re-implementation of mamba_ssm's non-fused forward using nn.Conv1d and the naive recurrence."""
    from test_ssd import naive_scan  # tests/ is on sys.path under pytest (rootdir-relative import)

    m = tiny()
    x = torch.randn(2, 19, 16, dtype=torch.float64)
    z, xBC, dt = torch.split(m.in_proj(x), [32, 32 + 2 * 2 * 8, 4], dim=-1)
    xBC = F.silu(m.conv1d(xBC.transpose(1, 2))[..., :19].transpose(1, 2))
    xs, B, C = torch.split(xBC, [32, 16, 16], dim=-1)
    y, _ = naive_scan(xs.view(2, 19, 4, 8), dt, -m.A_log.exp(), B.view(2, 19, 2, 8), C.view(2, 19, 2, 8), D=m.D,
                      dt_bias=m.dt_bias, dt_softplus=True)
    ref = m.out_proj(m.norm(y.reshape(2, 19, 32), z))
    torch.testing.assert_close(m(x), ref)


def test_packed_equals_separate_outputs_and_grads():
    m = tiny()
    lens = [11, 6, 20]
    x = torch.randn(1, sum(lens), 16, dtype=torch.float64, requires_grad=True)
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 11, 17, 37], dtype=torch.int32))
    w = torch.randn(1, sum(lens), 16, dtype=torch.float64)

    y = m(x, ctx)
    (y * w).sum().backward()
    g_packed = {k: p.grad.clone() for k, p in m.named_parameters()}
    gx_packed = x.grad.clone()

    m.zero_grad()
    x.grad = None
    ys, s = [], 0
    for n in lens:
        ys.append(m(x[:, s:s + n]))
        s += n
    y_sep = torch.cat(ys, dim=1)
    (y_sep * w).sum().backward()
    torch.testing.assert_close(y, y_sep, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(gx_packed, x.grad, rtol=1e-10, atol=1e-10)
    for k, p in m.named_parameters():
        # rtol 1e-6: the A_log gradient is a sum with heavy cancellation; even the naive recurrence shows ~1e-8
        # relative packed-vs-separate differences in fp64.
        torch.testing.assert_close(g_packed[k], p.grad, rtol=1e-6, atol=1e-10, msg=k)


def test_batched_rows_equal_packed():
    m = tiny()
    x = torch.randn(3, 10, 16, dtype=torch.float64)
    packed = m(x.reshape(1, 30, 16), SeqCtx.from_cu_seqlens(torch.tensor([0, 10, 20, 30], dtype=torch.int32)))
    torch.testing.assert_close(m(x), packed.view(3, 10, 16), rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("packed", [False, True])
def test_causality(packed):
    m = tiny()
    T, k = 29, 13
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 9, T], dtype=torch.int32)) if packed else None
    x = torch.randn(1, T, 16, dtype=torch.float64, requires_grad=True)
    y = m(x, ctx)
    x2 = x.detach().clone()
    x2[:, k + 1:] += torch.randn_like(x2[:, k + 1:])
    y2 = m(x2, ctx)
    torch.testing.assert_close(y[:, :k + 1], y2[:, :k + 1], rtol=0, atol=0)
    # Jacobian view: output at k has zero gradient wrt any later input.
    (gx,) = torch.autograd.grad(y[:, k].sum(), x)
    assert gx[:, k + 1:].abs().max() == 0 and gx[:, k].abs().max() > 0


def test_float32_and_bfloat16_autocast_cpu():
    torch.manual_seed(0)
    m = Mamba2(32, d_state=16, headdim=16, chunk_size=16)
    x = torch.randn(2, 40, 32)
    y = m(x)
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        yb = m(x)
    assert yb.dtype == torch.bfloat16
    torch.testing.assert_close(yb.float(), y, rtol=5e-2, atol=5e-2)


# ----------------------------------------------------------------------------------------------- CUDA vs kernels
needs_kernels = pytest.mark.skipif(
    not torch.cuda.is_available() or mamba2_mod._split_conv1d_scan is None or conv_mod._causal_conv1d_fn is None,
    reason="needs CUDA + mamba_ssm + causal_conv1d",
)


@needs_kernels
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 3e-3), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("ngroups", [1, 2])
def test_kernel_path_agrees_with_torch_path(monkeypatch, dtype, tol, ngroups):
    torch.manual_seed(0)
    m = Mamba2(256, d_state=64, headdim=64, ngroups=ngroups, chunk_size=64, device="cuda")
    x = torch.randn(1, 700, 256, device="cuda")
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, 100, 333, 700], dtype=torch.int32, device="cuda"))
    outs, grads = [], []
    for force_torch in (False, True):
        monkeypatch.setattr(ssd, "FORCE_TORCH", force_torch)
        m.zero_grad()
        xi = x.clone().requires_grad_()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            y = m(xi, ctx)
        y.float().square().mean().backward()
        outs.append(y.float())
        grads.append({"x": xi.grad.float(), **{k: p.grad.float() for k, p in m.named_parameters()}})
    assert (outs[0] - outs[1]).abs().max() <= tol * outs[1].abs().max()
    for k in grads[1]:
        assert (grads[0][k] - grads[1][k]).abs().max() <= 5 * tol * grads[1][k].abs().max() + 1e-6, k
