"""Unit tests of the H-Net building blocks (CPU, tiny)."""

import math
import warnings

import pytest
import torch

from hnet import registry
from hnet.model import SeqCtx
from hnet.model.chunking import linear_recurrence, plug_back_index
from hnet.model.losses import ratio_loss
from hnet.model.mixers import attn as attn_mod
from hnet.model.mlp import SwiGLU
from hnet.model.norm import RMSNorm
from hnet.model.rotary import apply_rotary, rotary_cos_sin
from hnet.model.routing import RouterOutput, router_output

torch.set_num_threads(1)


def ctx_of(lengths):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


# ----------------------------------------------------------------------------- ctx / norm / rotary / mlp
def test_seqctx_positions_restart():
    c = ctx_of([3, 0, 4])
    assert c.seq_idx.tolist() == [[0, 0, 0, 2, 2, 2, 2]]
    assert c.position_ids.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert c.max_seqlen == 4


def test_rmsnorm_fp32_compute():
    n = RMSNorm(8, eps=1e-5)
    x = torch.randn(3, 8).bfloat16()
    y = n(x)
    ref = x.float() / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(n(x, out_dtype=torch.float32), ref)


def test_rotary_partial_and_restart():
    c = ctx_of([5, 5])
    cos, sin = rotary_cos_sin(c, 4, 10000.0)
    x = torch.randn(10, 2, 8)
    y = apply_rotary(x, cos, sin)
    assert torch.equal(y[..., 4:], x[..., 4:])  # partial: untouched channels
    assert torch.equal(y[0], x[0]) and torch.equal(y[5], x[5])  # position 0 of each segment: identity
    torch.testing.assert_close(y[:5], apply_rotary(x[:5], *rotary_cos_sin(ctx_of([5]), 4, 10000.0)))
    # Rotation of pair (i, i + r/2) by angle pos * theta^(-2i/r): channel 0 at position 1.
    ang = torch.tensor(1.0)
    torch.testing.assert_close(y[1, :, 0], x[1, :, 0] * ang.cos() - x[1, :, 2] * ang.sin())
    assert rotary_cos_sin(c, 4, 10000.0)[0] is cos  # memoized per stage


def test_swiglu_gate_order_and_rounding():
    m = SwiGLU(16, 100)
    assert m.d_intermediate == 128 and m.fc1.weight.shape == (256, 16)
    x = torch.randn(4, 16)
    y, gate = (x @ m.fc1.weight.T).chunk(2, -1)
    torch.testing.assert_close(m(x), (torch.nn.functional.silu(gate) * y) @ m.fc2.weight.T)


# ----------------------------------------------------------------------------- attention
def _naive_attention(q, k, v, ctx, window, scale):
    T, H, _ = q.shape
    k = k.repeat_interleave(H // k.shape[1], dim=1)
    v = v.repeat_interleave(H // v.shape[1], dim=1)
    out = torch.zeros_like(q)
    seq = ctx.seq_idx[0]
    for t in range(T):
        keys = [s for s in range(T) if seq[s] == seq[t] and s <= t and (window < 0 or t - s <= window)]
        w = torch.softmax(torch.einsum("hd,shd->hs", q[t], k[keys]) * scale, -1)
        out[t] = torch.einsum("hs,shd->hd", w, v[keys])
    return out


@pytest.mark.parametrize("window", [-1, 3])
@pytest.mark.parametrize("kv_heads", [4, 2])
def test_attention_backends_agree(window, kv_heads):
    torch.manual_seed(0)
    c = ctx_of([7, 1, 12])
    q, k, v = torch.randn(20, 4, 8), torch.randn(20, kv_heads, 8), torch.randn(20, kv_heads, 8)
    ref = _naive_attention(q, k, v, c, window, 8**-0.5)
    for backend in attn_mod.available_backends(q.device):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = attn_mod.attention(q, k, v, c, window, 8**-0.5, backend=backend)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5, msg=backend)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_attention_backends_agree_cuda():
    torch.manual_seed(0)
    lengths = [100, 3, 300]
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    c = SeqCtx.from_cu_seqlens(cu)
    q, k, v = (torch.randn(403, h, 64, device="cuda", dtype=torch.bfloat16) for h in (8, 2, 2))
    ref = attn_mod.attention(q, k, v, c, 63, 0.125, backend="sdpa").float()
    for backend in attn_mod.available_backends(q.device):
        out = attn_mod.attention(q, k, v, c, 63, 0.125, backend=backend).float()
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2, msg=backend)


def test_attention_module_names_and_nope():
    a = registry.build("mixer", "attn", 32, num_heads=4, num_kv_heads=2, rotary_emb_dim=4, layer_idx=0)
    assert a.Wqkv.weight.shape == ((4 + 2 * 2) * 8, 32) and a.rotary_emb_dim == 4
    nope = registry.build("mixer", "attn", 32, num_heads=4, nope_every=2, layer_idx=1)
    assert nope.rotary_emb_dim == 0
    c = ctx_of([5, 6])
    x = torch.randn(1, 11, 32)
    y = a(x, c)
    assert y.shape == x.shape
    torch.testing.assert_close(y[:, 5:], a(x[:, 5:], ctx_of([6])))  # documents are independent


# ----------------------------------------------------------------------------- scan / dechunk
def _naive_recurrence(u, a, reset):
    y, prev = torch.zeros_like(u), torch.zeros(u.shape[1], dtype=u.dtype)
    for i in range(u.shape[0]):
        prev = u[i] if reset[i] else a[i] * prev + u[i]
        y[i] = prev
    return y


@pytest.mark.parametrize("M,block", [(1, 4), (13, 4), (100, 4), (70, 64)])
def test_linear_recurrence_matches_loop(M, block):
    torch.manual_seed(M)
    u = torch.randn(M, 3, dtype=torch.float64)
    a = torch.rand(M, dtype=torch.float64).clamp(1e-4, 1 - 1e-4)
    reset = torch.rand(M) < 0.1
    reset[0] = True
    y = linear_recurrence(u, a.log(), reset, block)
    torch.testing.assert_close(y.double(), _naive_recurrence(u, a, reset), atol=1e-5, rtol=1e-5)


def test_linear_recurrence_grad():
    torch.manual_seed(0)
    u = torch.randn(20, 2, dtype=torch.float64, requires_grad=True)
    la = (torch.rand(20, dtype=torch.float64) * 0.9 + 0.05).log().requires_grad_()
    reset = torch.zeros(20, dtype=torch.bool)
    reset[[0, 7]] = True
    torch.autograd.gradcheck(lambda u, la: linear_recurrence(u, la, reset, 4), (u, la))


def test_ema_dechunker_definition():
    T = 9
    c = ctx_of([T])
    p = torch.tensor([1.0, 0.2, 0.9, 0.3, 0.6, 0.1, 0.8, 0.7, 0.4])
    r = router_output(p, c, "start")
    h_inner, c_inner = registry.build("chunker", "select")(torch.randn(T, 4), r.boundary_mask, c)
    out = registry.build("dechunker", "ema")(h_inner, r, c, c_inner)
    P = p[r.boundary_mask].clamp(1e-4, 1 - 1e-4)
    zbar, prev = [], torch.zeros(4)
    for j in range(len(P)):
        prev = P[j] * h_inner[j] + (1 - P[j]) * prev
        zbar.append(prev)
    torch.testing.assert_close(out, torch.stack(zbar)[plug_back_index(r)])


# ----------------------------------------------------------------------------- routers / chunking
@pytest.mark.parametrize("semantics", ["start", "end"])
def test_hand_built_boundaries(semantics):
    """Two segments; boundaries given explicitly. Same selection/plug-back code for both semantics."""
    c = ctx_of([6, 4])
    b = torch.tensor([0, 0, 1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.bool)  # position 0 / 6 forced to 1
    r = registry.build("router", "external", 4, semantics=semantics)(torch.zeros(10, 4), c, boundaries=b)
    assert r.semantics == semantics
    assert r.boundary_mask.tolist() == [1, 0, 1, 0, 1, 0, 1, 1, 0, 0]
    h = torch.arange(10.0)[:, None].repeat(1, 3)
    x, ci = registry.build("chunker", "select")(h, r.boundary_mask, c)
    assert x[:, 0].tolist() == [0, 2, 4, 6, 7]  # chunk vectors: first bytes ("start") or last bytes ("end")
    assert ci.cu_seqlens.tolist() == [0, 3, 5] and ci.max_seqlen == 3
    assert ci.position_ids.tolist() == [0, 1, 2, 0, 1] and ci.outer_index.tolist() == [0, 2, 4, 6, 7]
    pb = plug_back_index(r).tolist()
    assert pb == [0, 0, 1, 1, 2, 2, 3, 4, 4, 4]
    # "start": byte t reads the chunk containing it. "end": the latest chunk that ended at or before t, e.g.
    # bytes 1-2 form chunk "1..2" (ends at 2): byte 1 reads the BOS chunk 0, byte 2 reads its own chunk.
    out = registry.build("dechunker", "repeat")(x, r, c, ci)
    assert out[:, 0].tolist() == [0, 0, 2, 2, 4, 4, 6, 7, 7, 7]


def test_router_forced_first_and_semantics():
    c = ctx_of([4, 5])
    h = torch.randn(9, 16)
    for name in registry.names("router"):
        kw = {"input_ids": torch.randint(0, 256, (9,)), "boundaries": torch.zeros(9, dtype=torch.bool)}
        router = registry.build("router", name, 16, target_ratio=3.0)
        r = router(h, c, **kw)
        assert isinstance(r, RouterOutput) and r.semantics == router.boundary_semantics
        assert r.boundary_mask[[0, 4]].all() and (r.p[[0, 4]] == 1).all(), name
        assert r.boundary_prob.shape == (9, 2) and r.selected_probs.shape == (9, 1)
        torch.testing.assert_close(r.selected_probs[:, 0], torch.maximum(r.p, 1 - r.p))
    assert registry.get("router", "causal_end_of_chunk") is registry.get("router", "end_mlp")


def test_end_router_init_rate():
    torch.manual_seed(0)
    c = ctx_of([4000])
    h = torch.nn.functional.normalize(torch.randn(4000, 64), dim=-1) * 8  # unit RMS
    for name in ("end_linear", "end_mlp"):
        r = registry.build("router", name, 64, target_ratio=4.0)(h, c)
        # logits ~ N(logit(1/4), 1.5^2): F = Phi(-1.10 / 1.5) = 0.23, G = E[sigmoid] = 0.30
        assert abs(r.boundary_mask.float().mean().item() - 0.23) < 0.08, name
        assert abs(r.p.mean().item() - 0.30) < 0.06, name


def test_space_like_and_fixed_pool():
    text = b"ab cd, e"
    ids = torch.tensor(list(text))
    c = ctx_of([len(text)])
    h = torch.zeros(len(text), 4)
    start = registry.build("router", "space_like", 4)(h, c, input_ids=ids).boundary_mask
    end = registry.build("router", "space_like", 4, semantics="end")(h, c, input_ids=ids).boundary_mask
    assert [i for i, v in enumerate(start.tolist()) if v] == [0, 3, 7]  # word starts
    assert [i for i, v in enumerate(end.tolist()) if v] == [0, 2, 5]  # first non-word byte after a word
    fp = registry.build("router", "fixed_pool", 4, target_ratio=3.0)(h, c).boundary_mask
    assert [i for i, v in enumerate(fp.tolist()) if v] == [0, 3, 6]
    fe = registry.build("router", "fixed_pool", 4, k=3, semantics="end")(h, c).boundary_mask
    assert [i for i, v in enumerate(fe.tolist()) if v] == [0, 2, 5]


# ----------------------------------------------------------------------------- ratio loss
def _ro(mask, p):
    p = torch.tensor(p)
    return RouterOutput(torch.stack((1 - p, p), -1), torch.tensor(mask, dtype=torch.bool), p[:, None])


def test_ratio_loss_values():
    N = 4.0
    loss, st = ratio_loss(_ro([1, 0, 0, 0], [0.25] * 4), N)
    assert math.isclose(loss.item(), 1.0, rel_tol=1e-6)  # minimum at F = G = 1/N
    F_, G = 0.5, 0.3
    loss, st = ratio_loss(_ro([1, 1, 0, 0], [0.3] * 4), N)
    ref = N / (N - 1) * ((N - 1) * F_ * G + (1 - F_) * (1 - G))
    assert math.isclose(loss.item(), ref, rel_tol=1e-6)
    assert math.isclose(st["F"].item(), F_) and math.isclose(st["G"].item(), G, rel_tol=1e-6)
    # Official load_balancing_loss form.
    off = ((1 - F_) * (1 - G) + F_ * G * (N - 1)) * N / (N - 1)
    assert math.isclose(loss.item(), off, rel_tol=1e-6)
    # Gradient flows through G only.
    p = torch.full((4,), 0.3, requires_grad=True)
    r = RouterOutput(torch.stack((1 - p, p), -1), torch.tensor([1, 1, 0, 0], dtype=torch.bool), p[:, None])
    ratio_loss(r, N, "global")[0].backward()  # no process group -> local
    torch.testing.assert_close(p.grad, torch.full((4,), N / (N - 1) * ((N - 1) * F_ - (1 - F_)) / 4))
