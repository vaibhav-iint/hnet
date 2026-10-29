"""Kimi attention variants (KDA, MoBA), gated attention and Attention Residuals (CPU, tiny)."""

import os

import pytest
import torch

from hnet import registry
from hnet.config import DenseConfig, HNetConfig, load_yaml
from hnet.flops import SPEC, attn_flops, mixer_counts
from hnet.model import SeqCtx, build_model
from hnet.model.isotropic import Isotropic
from hnet.model.mixers import kimi
from hnet.model.mixers.kimi import kda_chunk_torch, kda_recurrent_reference, moba_block_selection

torch.set_num_threads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIXERS = ["kda", "moba", "gated_attn"]
D = 32
ATTN_KW = dict(num_heads=4, num_kv_heads=2, rotary_emb_dim=4, rope_theta=10000.0, window_size=-1, nope_every=0)
OWN_KW = {"kda": dict(chunk_size=16, sub_chunk=4), "moba": dict(block_size=4, top_k=2), "gated_attn": {}}


def ctx_of(lengths, device=None):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    return SeqCtx.from_cu_seqlens(cu)


def make_mixer(name, seed=0, dtype=torch.float32, **kw):
    torch.manual_seed(seed)
    m = registry.build("mixer", name, D, layer_idx=0, **{**ATTN_KW, **OWN_KW[name], **kw})
    return m.to(dtype)


def kda_inputs(T, H=2, K=8, V=6, seed=0, dtype=torch.float64, strong=True):
    g_ = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(T, H, K, generator=g_, dtype=dtype), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(T, H, K, generator=g_, dtype=dtype), dim=-1)
    v = torch.randn(T, H, V, generator=g_, dtype=dtype)
    g = -torch.rand(T, H, K, generator=g_, dtype=dtype) * 2
    if strong:  # very strong decays somewhere: exercises the overflow-free sub-chunk factorization
        g[::7] = g[::7] * 40
    beta = torch.rand(T, H, generator=g_, dtype=dtype)
    return q, k, v, g, beta


# ----------------------------------------------------------------------------- KDA core
@pytest.mark.parametrize("chunk,sub", [(8, 8), (16, 4), (32, 8), (64, 16)])
def test_kda_chunk_equals_recurrent_fp64(chunk, sub):
    lengths = [10, 0, 23, 16, 1, 30]  # empty segment, boundaries at/inside chunks, one-token segment
    ctx = ctx_of(lengths)
    args = kda_inputs(sum(lengths))
    ref = kda_recurrent_reference(*args, 0.37, ctx.seq_idx)
    out = kda_chunk_torch(*args, 0.37, ctx.seq_idx, chunk, sub)
    torch.testing.assert_close(out, ref, atol=1e-12, rtol=1e-10)


def test_kda_chunk_fp32_and_grad_match_reference():
    ctx = ctx_of([20, 25])
    a64 = [t.requires_grad_() for t in kda_inputs(45, strong=False)]
    a32 = [t.detach().float().requires_grad_() for t in a64]
    ref = kda_recurrent_reference(*a64, 0.5, ctx.seq_idx)
    out = kda_chunk_torch(*a32, 0.5, ctx.seq_idx, 16, 8)
    torch.testing.assert_close(out.double(), ref, atol=2e-5, rtol=1e-4)
    w = torch.randn_like(ref)
    g64 = torch.autograd.grad((ref * w).sum(), a64)
    g32 = torch.autograd.grad((out * w.float()).sum(), a32)
    for a, b in zip(g32, g64):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a.double(), b, atol=1e-4, rtol=1e-3)


def test_kda_state_semantics():
    """Delta rule with k = e_0, beta = 1, no decay: the state's row 0 is overwritten by the latest value."""
    T, H, K, V = 5, 1, 4, 3
    q = torch.zeros(T, H, K, dtype=torch.float64)
    q[..., 0] = 1
    k = q.clone()
    v = torch.randn(T, H, V, dtype=torch.float64)
    o = kda_chunk_torch(q, k, v, torch.zeros(T, H, K, dtype=torch.float64), torch.ones(T, H, dtype=torch.float64),
                        1.0, None, 4, 2)
    torch.testing.assert_close(o, v)


# ----------------------------------------------------------------------------- generic mixer properties
@pytest.mark.parametrize("name", MIXERS)
def test_causality(name):
    m = make_mixer(name)
    x = torch.randn(1, 40, D)
    ctx = ctx_of([40])
    y = m(x, ctx)
    x2 = x.clone()
    x2[:, 25] += 3.0
    y2 = m(x2, ctx)
    torch.testing.assert_close(y[:, :25], y2[:, :25], atol=1e-6, rtol=1e-5)
    assert (y[:, 25:] - y2[:, 25:]).abs().max() > 1e-4


@pytest.mark.parametrize("name", MIXERS)
def test_packed_equals_separate(name):
    m = make_mixer(name, dtype=torch.float64)
    lengths = [9, 0, 21, 14]
    x = torch.randn(1, sum(lengths), D, dtype=torch.float64)
    y = m(x, ctx_of(lengths))
    start = 0
    for n in lengths:
        if n:
            ys = m(x[:, start:start + n], ctx_of([n]))
            torch.testing.assert_close(y[:, start:start + n], ys, atol=1e-10, rtol=1e-8)
        start += n


@pytest.mark.parametrize("name", MIXERS)
def test_gradient_flow(name):
    m = make_mixer(name)
    x = torch.randn(1, 30, D, requires_grad=True)
    m(x, ctx_of([12, 18])).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
        if not n.endswith("bias"):
            assert p.grad.abs().sum() > 0, n


@pytest.mark.parametrize("name,kw", [
    ("kda", {}), ("kda", dict(expand_v=1.5, head_dim=12, gate_rank=5, conv_bias=True)),
    ("moba", {}), ("moba", dict(qkv_bias=True, out_bias=True)),
    ("gated_attn", {}), ("gated_attn", dict(gate="headwise", gate_bias=True)),
])
def test_meta_params_equal_flops_params(name, kw):
    kwargs = {**ATTN_KW, **OWN_KW[name], **kw}
    with torch.device("meta"):
        m = registry.build("mixer", name, 64, layer_idx=3, **kwargs)
    p, f = mixer_counts(name, 64, kwargs, 300.0, SPEC)
    assert p == sum(t.numel() for t in m.parameters())
    assert f > 0


def test_unknown_kwargs_rejected_and_attn_keys_ignored():
    with pytest.raises(TypeError):
        registry.build("mixer", "kda", D, num_heads=4, bogus=1)
    with pytest.raises(TypeError):
        registry.build("mixer", "moba", D, num_heads=4, bogus=1)
    m = registry.build("mixer", "kda", D, num_heads=4, rotary_emb_dim=8, window_size=5, nope_every=2, num_kv_heads=1,
                       qkv_bias=True, out_bias=True, softmax_scale=0.1)
    assert m.key_dim == D and m.head_k_dim == 8


# ----------------------------------------------------------------------------- MoBA
@pytest.mark.parametrize("window", [-1, 6])
def test_moba_topk_all_blocks_equals_attention(window):
    lengths = [13, 27]
    x = torch.randn(1, sum(lengths), D, dtype=torch.float64)
    ctx = ctx_of(lengths)
    m = make_mixer("moba", dtype=torch.float64, top_k=100, window_size=window)
    a = registry.build("mixer", "attn", D, layer_idx=0, **{**ATTN_KW, "window_size": window}).double()
    a.load_state_dict(m.state_dict())
    torch.testing.assert_close(m(x, ctx), a(x, ctx), atol=1e-12, rtol=1e-10)
    m.top_k = 2  # sparse: must differ
    assert (m(x, ctx) - a(x, ctx)).abs().max() > 1e-6


def test_moba_selection_rules():
    lengths = [11, 0, 18]
    ctx = ctx_of(lengths)
    T, H, B, k = sum(lengths), 3, 4, 3
    q, kk = torch.randn(T, H, 8), torch.randn(T, 1, 8)
    sel, gid = moba_block_selection(q, kk, ctx, B, k)
    pos, seq = ctx.position_ids, ctx.seq_idx[0].long()
    assert torch.equal(gid, torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
                                          5, 5, 5, 5, 6, 6, 6, 6, 7, 7]))
    for t in range(T):
        for h in range(H):
            js = sel[h, t].nonzero().view(-1).tolist()
            assert gid[t].item() in js  # current block always
            assert all(j <= gid[t] for j in js)  # no future block
            assert all(seq[gid == j][0] == seq[t] for j in js)  # same document
            n_past = int(pos[t]) // B
            assert len(js) == min(k, n_past + 1)
            if n_past >= k:  # the chosen past blocks are the top-scoring ones
                past = [j for j in range(int(gid[t]) - n_past, int(gid[t]))]
                scores = torch.stack([q[t, h] @ kk[gid == j, 0].mean(0) for j in past])
                best = {past[i] for i in scores.topk(k - 1).indices.tolist()}
                assert best == set(js) - {gid[t].item()}


@pytest.mark.filterwarnings("ignore:flex_attention called without torch.compile")
def test_moba_flex_equals_sdpa():
    if kimi.flex_attention is None:
        pytest.skip("flex_attention unavailable")
    m = make_mixer("moba", window_size=7)
    x, ctx = torch.randn(1, 37, D), ctx_of([15, 22])
    with torch.no_grad():
        y = m(x, ctx)
        m.backend = "flex"
        torch.testing.assert_close(m(x, ctx), y, atol=1e-5, rtol=1e-4)


def test_moba_flops_reduce_to_attention():
    kw = dict(num_heads=4, num_kv_heads=2, window_size=-1)
    L, B = 200.0, 16
    dense_like = kimi.MoBA.flops_per_position(64, L, SPEC, block_size=B, top_k=100, **kw)
    gate = 2 * 64 * sum(b * min(B, 200 - b * B) for b in range(13)) / 200 + 32
    assert dense_like - gate == pytest.approx(attn_flops(64, kw, L, SPEC))
    assert kimi.MoBA.flops_per_position(64, L, SPEC, block_size=B, top_k=2, **kw) < dense_like - gate


# ----------------------------------------------------------------------------- gated attention
@pytest.mark.parametrize("gate", ["elementwise", "headwise"])
def test_gated_attn_open_gate_equals_attn(gate):
    m = make_mixer("gated_attn", dtype=torch.float64, gate=gate, gate_bias=True)
    with torch.no_grad():
        m.gate_proj.weight.zero_()
        m.gate_proj.bias.fill_(60.0)  # sigmoid(60) == 1 in fp64 to ~1e-26
    a = registry.build("mixer", "attn", D, layer_idx=0, **ATTN_KW).double()
    a.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("gate_proj")})
    x, ctx = torch.randn(1, 25, D, dtype=torch.float64), ctx_of([10, 15])
    torch.testing.assert_close(m(x, ctx), a(x, ctx))
    assert m.gate_proj.out_features == (D if gate == "elementwise" else 4)


# ----------------------------------------------------------------------------- whole models
def tiny_hnet(main_codes, layout="T2", mixer_cfg=None):
    return HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T1", [layout], "T1"], "m1"],
        d_model=[32, 32, 48],
        d_intermediate=[0, 64, 96],
        ssm=dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8),
        attn=dict(num_heads=2, rotary_emb_dim=[8, 8, 12], window_size=[7, 7, -1]),
        dc=dict(target_ratios=[2.0, 2.0], router="fixed_pool", router_kwargs={"k": 2}),
        mixer_codes=[{"m": "mamba2", "t": "attn"}, {"m": "mamba2", "t": "attn"}, main_codes],
        mixer_cfg=mixer_cfg or {},
    ))


@pytest.mark.parametrize("name", MIXERS)
def test_tiny_hnet_forward_backward(name):
    cfg = tiny_hnet({"m": "mamba2", "t": name}, mixer_cfg={name: OWN_KW[name]})
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    assert any(type(b.mixer).__name__ != "CausalAttention" and registry.get("mixer", name) is type(b.mixer)
               for b in m.blocks())
    ids = torch.randint(0, 256, (2, 48))
    out = m(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    assert torch.isfinite(out.loss)
    assert all(p.grad is not None for n, p in m.named_parameters() if "main_network.main_network" in n)


def test_tiny_hnet_kimi_linear_hybrid_layout():
    cfg = tiny_hnet({"m": "mamba2", "t": "attn", "k": "kda"}, layout="K3T1",
                    mixer_cfg={"attn": [{}, {}, {"rotary_emb_dim": 0}], "kda": OWN_KW["kda"]})
    m = build_model(cfg)
    main = m.backbone.main_network.main_network.main_network
    kinds = [type(b.mixer).__name__ for b in main.layers]
    assert kinds == ["KimiDeltaAttention"] * 3 + ["CausalAttention"]
    assert main.layers[3].mixer.rotary_emb_dim == 0  # NoPE full attention, as Kimi Linear
    out = m(torch.randint(0, 256, (1, 40)), labels=torch.randint(0, 256, (1, 40)))
    out.loss.backward()


@pytest.mark.parametrize("name", MIXERS)
def test_tiny_dense_forward_backward(name):
    cfg = DenseConfig(vocab_size=300, d_model=D, n_layers=2, d_intermediate=64, num_heads=4, mixer=name,
                      mixer_kwargs=OWN_KW[name])
    m = build_model(cfg)
    m.init_weights()
    ids = torch.randint(0, 300, (2, 24))
    out = m(ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss) and all(p.grad is not None for p in m.parameters())
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(m(ids, labels=ids).loss)


# ----------------------------------------------------------------------------- configs
VARIANTS = ["tiny_bpe_2m_kda", "tiny_bpe_2m_moba", "tiny_bpe_2m_gated", "tiny_bpe_2m_gated_headwise",
            "tiny_hnet2_2m_kda", "tiny_hnet2_2m_kda_hybrid", "tiny_hnet2_2m_moba", "tiny_hnet2_2m_gated"]


@pytest.mark.parametrize("name", VARIANTS)
def test_variant_configs_differ_only_in_mixer(name):
    y = load_yaml(os.path.join(ROOT, "configs", "models", name + ".yaml"))
    base = load_yaml(os.path.join(ROOT, "configs", "models", y["matching"]["variant_of"] + ".yaml"))
    a, b = y["model"], base["model"]
    diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    allowed = {"mixer", "mixer_kwargs", "mixer_codes", "mixer_cfg"} | ({"arch_layout"} if "hybrid" in name else set())
    assert diff and diff <= allowed, diff
    assert y["matching"]["reference"] == "tiny_bpe_2m"  # FLOPs-matched (+-2%) in tests/test_flops.py


# ----------------------------------------------------------------------------- attention residuals
def _iso(attn_res, layout="T2m1T1", seed=0):
    torch.manual_seed(seed)
    m = Isotropic(layout, D, 64, {"t": "attn", "m": "mamba2"},
                  {"attn": dict(num_heads=4), "mamba2": dict(d_state=8, headdim=16, chunk_size=8)}, attn_res=attn_res)
    return m.double()


def _attn_res_reference(iso, x, ctx, S):
    """Eqs. 2-6 of arXiv 2603.15031 written directly: sub-layer l (0-based) in block n = l // S attends over
    b_0 = x, the completed block sums b_1..b_n and (if l is not first in its block) the partial sum of block n."""
    subs = []
    for layer in iso.layers:
        subs.append((layer.norm1, lambda u, mm=layer.mixer: mm(u, ctx)))
        if layer.mlp is not None:
            subs.append((layer.norm2, layer.mlp))
    outs = []

    def attend(op, srcs):
        V = torch.stack(srcs)
        kk = V / torch.sqrt(V.pow(2).mean(-1, keepdim=True) + op.norm.eps) * op.norm.weight
        a = torch.softmax((kk * op.query).sum(-1), dim=0)
        return (a[..., None] * V).sum(0)

    for l, (norm, fn) in enumerate(subs):
        n = l // S
        srcs = [x] + [sum(outs[j * S:(j + 1) * S]) for j in range(n)]
        if l % S:
            srcs.append(sum(outs[n * S:l]))
        outs.append(fn(norm(attend(iso.attn_res[l], srcs))))
    nb = (len(subs) + S - 1) // S
    final = attend(iso.attn_res[len(subs)], [x] + [sum(outs[j * S:(j + 1) * S]) for j in range(nb)])
    return iso.rmsnorm(final)


def test_attn_res_off_is_unchanged():
    m = _iso(0)
    assert not any("attn_res" in n for n, _ in m.named_parameters())
    ref = _iso(0)
    x, ctx = torch.randn(1, 12, D, dtype=torch.float64), ctx_of([12])
    torch.testing.assert_close(m(x, ctx), ref(x, ctx))


@pytest.mark.parametrize("S", [1, 2, 3, 100])
def test_attn_res_matches_paper_equations(S):
    m = _iso(S)
    assert len(m.attn_res) == m.height + 1 == 8  # T, T, m, T: 7 sub-layers + output
    with torch.no_grad():  # non-zero pseudo-queries / norm weights (zero-init gives uniform weights)
        for op in m.attn_res:
            op.query.normal_()
            op.norm.weight.uniform_(0.5, 1.5)
    x, ctx = torch.randn(1, 20, D, dtype=torch.float64), ctx_of([8, 12])
    # the stack keeps sources in fp32 like the residual stream, the reference runs in fp64
    torch.testing.assert_close(m(x, ctx), _attn_res_reference(m, x, ctx, S), atol=2e-5, rtol=1e-4)


def test_attn_res_zero_init_causal_packed_and_grads():
    m = _iso(2)
    assert all(torch.equal(op.query, torch.zeros(D, dtype=torch.float64)) for op in m.attn_res)
    lengths = [7, 13]
    x = torch.randn(1, 20, D, dtype=torch.float64, requires_grad=True)
    y = m(x, ctx_of(lengths))
    torch.testing.assert_close(y[:, 7:], m(x[:, 7:], ctx_of([13])), atol=1e-5, rtol=1e-4)  # per-position mixing
    y.square().sum().backward()
    assert all(op.query.grad is not None and op.query.grad.abs().sum() > 0 for op in m.attn_res[1:])


# ----------------------------------------------------------------------------- optional fast paths (CUDA)
@pytest.mark.skipif(not torch.cuda.is_available() or kimi._fla_chunk_kda is None, reason="needs CUDA + fla")
def test_kda_fla_parity():
    torch.manual_seed(0)
    m = registry.build("mixer", "kda", 128, num_heads=2).cuda()
    x, ctx = torch.randn(1, 300, 128, device="cuda"), ctx_of([100, 0, 200], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_fast = m(x, ctx)
    orig = kimi._fla_chunk_kda
    kimi._fla_chunk_kda = None
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y_ref = m(x, ctx)
    finally:
        kimi._fla_chunk_kda = orig
    torch.testing.assert_close(y_fast.float(), y_ref.float(), atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not torch.cuda.is_available() or kimi.flex_attention is None, reason="needs CUDA flex")
def test_moba_flex_cuda_parity():
    m = make_mixer("moba").cuda()
    x, ctx = torch.randn(1, 300, D, device="cuda"), ctx_of([100, 200], device="cuda")
    y = m(x, ctx)
    m.backend = "sdpa"
    torch.testing.assert_close(y, m(x, ctx), atol=2e-3, rtol=2e-3)
