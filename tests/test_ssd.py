"""Tests for hnet.model.ssd: pure-torch chunked SSD vs a naive sequential recurrence (and vs the Triton kernel on
CUDA)."""

import itertools
import random

import pytest
import torch
import torch.nn.functional as F

from hnet.model import ssd
from hnet.model.ssd import ssd_chunk_scan, ssd_chunk_scan_torch

torch.set_num_threads(1)
INF = float("inf")


def naive_scan(x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, seq_idx=None, dt_limit=(0.0, INF)):
    """Token-by-token recurrence S_t = exp(dt A) S_{t-1} + dt x_t B_t^T, y_t = S_t C_t (+ D x_t) (* silu z_t),
    with S reset to 0 wherever seq_idx changes. Returns (y, final_state)."""
    b, l, h, p = x.shape
    g, n = B.shape[-2:]
    if dt_bias is not None:
        dt = dt + dt_bias
    if dt_softplus:
        dt = F.softplus(dt)
    dt = dt.clamp(*dt_limit)
    Bh, Ch = B.repeat_interleave(h // g, dim=2), C.repeat_interleave(h // g, dim=2)
    S = x.new_zeros(b, h, p, n)
    ys = []
    for t in range(l):
        if seq_idx is not None and t > 0:
            S = S * (seq_idx[:, t] == seq_idx[:, t - 1]).to(x.dtype)[:, None, None, None]
        dA = torch.exp(dt[:, t] * A)[..., None, None]
        S = dA * S + (dt[:, t, :, None] * x[:, t])[..., None] * Bh[:, t, :, None, :]
        ys.append(torch.einsum("bhpn,bhn->bhp", S, Ch[:, t]))
    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + x * (D[:, None] if D.dim() == 1 else D)
    if z is not None:
        y = y * F.silu(z)
    return y, S


def random_seq_idx(b, l, n_seg, gen):
    """Contiguous segments with random cut points (ids increasing along each row)."""
    rows = []
    for _ in range(b):
        cuts = sorted(random.Random(int(torch.randint(0, 2**31, (1,), generator=gen))).sample(range(1, l), n_seg - 1))
        lens = [e - s for s, e in zip([0] + cuts, cuts + [l])]
        rows.append(torch.repeat_interleave(torch.arange(n_seg), torch.tensor(lens)))
    return torch.stack(rows).to(torch.int32)


def make_inputs(b, l, h, p, g, n, dtype=torch.float64, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=dtype).to(device)  # noqa: E731
    return dict(
        x=r(b, l, h, p),
        dt=r(b, l, h) - 1.0,
        A=-torch.exp(r(h)).to(device),
        B=r(b, l, g, n),
        C=r(b, l, g, n),
    ), gen


def _configs():
    rng = random.Random(1234)
    out = []
    for i in range(40):
        g = rng.choice([1, 2])
        out.append(dict(
            b=rng.choice([1, 2]), l=rng.choice([1, 5, 16, 23, 37, 64]), h=g * rng.choice([1, 2, 3]),
            p=rng.choice([1, 4]), g=g, n=rng.choice([1, 3, 8]), chunk=rng.choice([4, 8, 16]),
            use_D=rng.choice([None, "h", "hp"]), use_z=rng.random() < 0.5, use_bias=rng.random() < 0.5,
            softplus=rng.random() < 0.7, limit=rng.choice([(0.0, INF), (0.05, 0.5)]),
            n_seg=rng.choice([0, 1, 2, 4]), seed=i,
        ))
    return out


@pytest.mark.parametrize("cfg", _configs(), ids=lambda c: f"l{c['l']}h{c['h']}g{c['g']}q{c['chunk']}s{c['n_seg']}")
def test_matches_naive_recurrence(cfg):
    c = cfg
    t, gen = make_inputs(c["b"], c["l"], c["h"], c["p"], c["g"], c["n"], seed=c["seed"])
    kw = {}
    if c["use_D"] == "h":
        kw["D"] = torch.randn(c["h"], generator=gen, dtype=torch.float64)
    elif c["use_D"] == "hp":
        kw["D"] = torch.randn(c["h"], c["p"], generator=gen, dtype=torch.float64)
    if c["use_z"]:
        kw["z"] = torch.randn(c["b"], c["l"], c["h"], c["p"], generator=gen, dtype=torch.float64)
    if c["use_bias"]:
        kw["dt_bias"] = torch.randn(c["h"], generator=gen, dtype=torch.float64)
    kw["dt_softplus"] = c["softplus"]
    kw["dt_limit"] = c["limit"]
    if c["n_seg"] and c["l"] >= c["n_seg"]:
        kw["seq_idx"] = random_seq_idx(c["b"], c["l"], c["n_seg"], gen)
    y, S = ssd_chunk_scan_torch(**t, chunk_size=c["chunk"], return_final_states=True, **kw)
    y_ref, S_ref = naive_scan(**t, **kw)
    assert y.dtype == torch.float64 and y.shape == y_ref.shape
    torch.testing.assert_close(y, y_ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(S, S_ref, rtol=1e-9, atol=1e-9)


def test_gradients_match_naive():
    t, gen = make_inputs(2, 21, 4, 3, 2, 5, seed=7)
    D = torch.randn(4, generator=gen, dtype=torch.float64)
    z = torch.randn(2, 21, 4, 3, generator=gen, dtype=torch.float64)
    bias = torch.randn(4, generator=gen, dtype=torch.float64)
    seq_idx = random_seq_idx(2, 21, 3, gen)
    leaves = {**t, "D": D, "z": z, "dt_bias": bias}
    leaves = {k: v.clone().requires_grad_() for k, v in leaves.items()}
    w = torch.randn(2, 21, 4, 3, generator=gen, dtype=torch.float64)
    grads = []
    for fn in (lambda **a: ssd_chunk_scan_torch(chunk_size=8, **a), lambda **a: naive_scan(**a)[0]):
        y = fn(**leaves, dt_softplus=True, seq_idx=seq_idx, dt_limit=(0.01, 2.0))
        grads.append(torch.autograd.grad((y * w).sum(), list(leaves.values())))
    for name, a, b in zip(leaves, *grads):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-8, msg=name)


def test_gradcheck():
    t, gen = make_inputs(1, 11, 2, 2, 1, 3, seed=3)
    z = torch.randn(1, 11, 2, 2, generator=gen, dtype=torch.float64)
    D = torch.randn(2, generator=gen, dtype=torch.float64)
    bias = torch.randn(2, generator=gen, dtype=torch.float64)
    seq_idx = torch.tensor([[0] * 6 + [1] * 5], dtype=torch.int32)
    inputs = tuple(v.clone().requires_grad_() for v in (t["x"], t["dt"], t["A"], t["B"], t["C"], D, z, bias))

    def f(x, dt, A, B, C, D, z, bias):
        return ssd_chunk_scan_torch(x, dt, A, B, C, 4, D=D, z=z, dt_bias=bias, dt_softplus=True, seq_idx=seq_idx)

    assert torch.autograd.gradcheck(f, inputs)


def test_segments_do_not_interact():
    """Perturbing one segment changes nothing outside it (within a chunk and across chunks)."""
    t, gen = make_inputs(1, 40, 2, 2, 1, 4, seed=5)
    seq_idx = torch.tensor([[0] * 7 + [1] * 20 + [2] * 13], dtype=torch.int32)
    y0 = ssd_chunk_scan_torch(**t, chunk_size=16, seq_idx=seq_idx, dt_softplus=True)
    t2 = {k: v.clone() for k, v in t.items()}
    for k in ("x", "B", "dt"):
        t2[k][:, 7:27] += torch.randn_like(t2[k][:, 7:27])
    y1 = ssd_chunk_scan_torch(**t2, chunk_size=16, seq_idx=seq_idx, dt_softplus=True)
    # Exact before the segment; after it only fp64 rounding differs (the in-chunk cumsum of log-decays runs
    # through the perturbed positions, as in the kernel).
    torch.testing.assert_close(y0[:, :7], y1[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(y0[:, 27:], y1[:, 27:], rtol=0, atol=1e-13)
    assert not torch.allclose(y0[:, 7:27], y1[:, 7:27])
    # Packed == each segment alone.
    for s, e in ((0, 7), (7, 27), (27, 40)):
        alone = ssd_chunk_scan_torch(**{k: (v[:, s:e] if v.dim() > 1 else v) for k, v in t.items()}, chunk_size=16,
                                     dt_softplus=True)
        torch.testing.assert_close(y0[:, s:e], alone, rtol=1e-12, atol=1e-12)


def test_chunk_size_invariance_and_long_sequence_stability():
    """Large cumulative decays (sum dA ~ -1e4) stay finite and agree across chunk sizes in fp32."""
    t, gen = make_inputs(1, 512, 2, 4, 1, 8, dtype=torch.float32, seed=11)
    t["dt"] = t["dt"] + 4.0  # softplus(dt) ~ 3-5, A ~ -1..-7 -> very fast decay
    leaves = {k: v.requires_grad_() for k, v in t.items()}
    outs = []
    for q in (16, 64, 256):
        y = ssd_chunk_scan_torch(**leaves, chunk_size=q, dt_softplus=True)
        g = torch.autograd.grad(y.square().sum(), list(leaves.values()))
        assert torch.isfinite(y).all() and all(torch.isfinite(x).all() for x in g)
        outs.append(y)
    y_ref = naive_scan(**{k: v.detach().double() for k, v in t.items()}, dt_softplus=True)[0]
    for y in outs:
        torch.testing.assert_close(y.double(), y_ref, rtol=1e-4, atol=1e-4)


def test_low_precision_inputs():
    t, gen = make_inputs(2, 30, 4, 8, 2, 16, dtype=torch.float32, seed=2)
    y32 = ssd_chunk_scan_torch(**t, chunk_size=8, dt_softplus=True)
    tb = {k: (v.bfloat16() if k != "A" else v) for k, v in t.items()}
    yb = ssd_chunk_scan_torch(**tb, chunk_size=8, dt_softplus=True)
    assert yb.dtype == torch.bfloat16
    # Only the input rounding differs: fp32 computation of bf16-rounded inputs.
    y_round = ssd_chunk_scan_torch(**{k: v.float() for k, v in tb.items()}, chunk_size=8, dt_softplus=True)
    torch.testing.assert_close(yb.float(), y_round, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(yb.float(), y32, rtol=5e-2, atol=5e-2)


def test_autocast_bf16_backward():
    """Inside a bf16 autocast region the scan still runs (and differentiates) in fp32."""
    t, _ = make_inputs(1, 40, 4, 8, 2, 8, dtype=torch.float32, seed=4)
    leaves = {k: v.requires_grad_() for k, v in t.items()}
    y_ref = ssd_chunk_scan_torch(**leaves, chunk_size=16, dt_softplus=True)
    g_ref = torch.autograd.grad(y_ref.square().sum(), list(leaves.values()))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = ssd_chunk_scan_torch(**leaves, chunk_size=16, dt_softplus=True)
    g = torch.autograd.grad(y.square().sum(), list(leaves.values()))  # backward outside autocast, as recommended
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, y_ref)
    for a, b in zip(g, g_ref):
        torch.testing.assert_close(a, b)


def test_dispatch_uses_torch_on_cpu():
    t, _ = make_inputs(1, 9, 2, 2, 1, 2, seed=0)
    torch.testing.assert_close(ssd_chunk_scan(**t, chunk_size=4), ssd_chunk_scan_torch(**t, chunk_size=4))


# ----------------------------------------------------------------------------------------------- CUDA vs kernel
needs_kernel = pytest.mark.skipif(
    not torch.cuda.is_available() or ssd._mamba_chunk_scan_combined is None, reason="needs CUDA + mamba_ssm"
)


@needs_kernel
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 5e-3), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("with_seq_idx,ngroups,with_z", list(itertools.product([False, True], [1, 2], [False, True])))
def test_kernel_agreement(dtype, tol, with_seq_idx, ngroups, with_z):
    b, l, h, p, n = 2, 300, 8, 32, 64
    t, gen = make_inputs(b, l, h, p, ngroups, n, dtype=torch.float32, seed=0, device="cuda")
    t = {k: (v.to(dtype) if k != "A" else v) for k, v in t.items()}
    extra = {
        "D": torch.randn(h, generator=gen).cuda(),
        "dt_bias": torch.randn(h, generator=gen).cuda() - 2,
        "z": torch.randn(b, l, h, p, generator=gen).to("cuda", dtype) if with_z else None,
        "seq_idx": random_seq_idx(b, l, 3, gen).cuda() if with_seq_idx else None,
    }
    grads, outs = [], []
    for fn in (ssd._mamba_chunk_scan_combined, ssd_chunk_scan_torch):
        leaves = {k: v.clone().requires_grad_() for k, v in t.items()}
        lv = {k: (v.clone().requires_grad_() if (v is not None and v.is_floating_point()) else v)
              for k, v in extra.items()}
        y = fn(**leaves, chunk_size=64, dt_softplus=True, **lv)
        y.float().square().sum().backward()
        outs.append(y.float())
        grads.append({k: v.grad.float() for k, v in {**leaves, **lv}.items() if torch.is_tensor(v) and v.grad is not None})
    scale = outs[1].abs().max()
    assert (outs[0] - outs[1]).abs().max() <= tol * scale
    for k in grads[1]:
        gs = grads[1][k].abs().max()
        assert (grads[0][k] - grads[1][k]).abs().max() <= 5 * tol * gs + 1e-6, k
