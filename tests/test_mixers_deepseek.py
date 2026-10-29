"""DeepSeek attention mixers (mla, nsa, dsa): causality, packing, references, gradients, FLOPs, tiny models, configs.

CPU, tiny, single thread. CUDA parity tests are skipped without a GPU.
"""

import copy
import glob
import math
import os
import warnings

import pytest
import torch
import torch.nn.functional as F

from hnet import registry
from hnet.config import IGNORE_INDEX, DenseConfig, HNetConfig, load_yaml, model_config_from_yaml
from hnet.flops import SPEC, FlopsConvention, attn_flops, count_params, mixer_counts, model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.mixers import attn as attn_mod
from hnet.model.mixers import deepseek as ds
from hnet.model.rotary import apply_rotary, rotary_cos_sin

torch.set_num_threads(1)
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = 32

# Small block sizes so that every sparse path is active (strictly sparse) at T ~ 40-70.
NSA_SMALL = dict(compress_block=4, compress_stride=2, select_block=4, select_n=3, n_init_blocks=1, n_local_blocks=1,
                 sliding_window=5)
VARIANTS = {
    "mla": ("mla", dict(num_heads=4, kv_lora_rank=12, qk_nope_head_dim=6, qk_rope_head_dim=4, v_head_dim=8)),
    "mla_q_win": ("mla", dict(num_heads=4, q_lora_rank=10, kv_lora_rank=12, window_size=6)),
    "nsa": ("nsa", dict(num_heads=4, num_kv_heads=2, rotary_emb_dim=4, **NSA_SMALL)),
    "nsa_mha_shared_win": ("nsa", dict(num_heads=2, separate_kv=False, window_size=9, compress_block=4,
                                        compress_stride=4, select_block=8, select_n=2, n_local_blocks=1,
                                        sliding_window=3)),
    "dsa": ("dsa", dict(num_heads=4, q_lora_rank=10, kv_lora_rank=12, index_topk=6, index_n_heads=2,
                        index_head_dim=8)),
    "dsa_attn_gqa": ("dsa", dict(num_heads=4, num_kv_heads=2, base="attn", index_topk=5, index_n_heads=2,
                                 index_head_dim=8, rotary_emb_dim=4)),
}


def ctx_of(lengths, device=None):
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=device)
    return SeqCtx.from_cu_seqlens(cu)


def make(name, seed=0, **kw):
    torch.manual_seed(seed)
    m = registry.build("mixer", name, D, **kw)
    with torch.no_grad():  # non-trivial gates / LayerNorm / pos-emb so nothing is accidentally symmetric
        for n, p in m.named_parameters():
            if p.dim() == 1 or "pos_emb" in n:
                p.add_(0.3 * torch.randn_like(p))
    return m


@pytest.fixture(autouse=True)
def _quiet_flex():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # uncompiled flex_attention on CPU warns that it is slow
        yield


@pytest.fixture
def force_sparse_backend():
    old = ds.FORCE_SPARSE_BACKEND

    def set_(b):
        ds.FORCE_SPARSE_BACKEND = b

    yield set_
    ds.FORCE_SPARSE_BACKEND = old


# ----------------------------------------------------------------------------- generic properties
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_causality(variant):
    name, kw = VARIANTS[variant]
    m = make(name, **kw).eval()
    T = 70
    x = torch.randn(1, T, D)
    c = ctx_of([T])
    with torch.no_grad():
        base = m(x, c)
        for u in (3, 17, 41, 69):
            x2 = x.clone()
            x2[:, u:] = torch.randn(1, T - u, D)
            out = m(x2, c)
            torch.testing.assert_close(out[:, :u], base[:, :u], atol=1e-5, rtol=1e-5)
            assert not torch.allclose(out[:, u:], base[:, u:])


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_packed_equals_separate(variant):
    name, kw = VARIANTS[variant]
    m = make(name, **kw).eval()
    lengths = [23, 1, 40, 9, 0, 13]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        packed = m(x, ctx_of(lengths))
        parts, o = [], 0
        for n in lengths:
            if n:
                parts.append(m(x[:, o:o + n], ctx_of([n])))
            o += n
    torch.testing.assert_close(packed, torch.cat(parts, 1), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_gradient_flow(variant):
    name, kw = VARIANTS[variant]
    m = make(name, **kw).train()
    x = torch.randn(1, 50, D, requires_grad=True)
    m(x, ctx_of([30, 20])).square().sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    assert not missing, missing
    assert x.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_meta_params_match_flops_counts(variant):
    name, kw = VARIANTS[variant]
    m = registry.build("mixer", name, D, **kw)
    p, f = mixer_counts(name, D, kw, 64.0, SPEC)
    assert p == sum(x.numel() for x in m.parameters())
    assert f > 0


def test_registered_and_config_kwargs():
    assert {"mla", "nsa", "dsa"} <= set(registry.names("mixer"))
    cfg = tiny_hnet_cfg("mla")
    for name in ("mla", "nsa", "dsa"):
        assert registry.get("mixer", name).config_kwargs(cfg, 2) == cfg.attn_for_stage(2)
    with pytest.raises(TypeError):
        registry.build("mixer", "dsa", D, num_heads=4, base="attn", kv_lora_rank=8)  # MLA key on a plain base
    with pytest.raises(TypeError):
        registry.build("mixer", "mla", D, num_heads=4, not_a_key=1)


# ----------------------------------------------------------------------------- MLA
def test_mla_matches_naive_equations():
    """Eqs. (37)-(47) of DeepSeek-V2 App. C (+ V3 latent RMSNorms), written out with plain tensor ops."""
    m = make("mla", num_heads=2, q_lora_rank=6, kv_lora_rank=5, qk_nope_head_dim=4, qk_rope_head_dim=4,
             v_head_dim=3).eval()
    T, H = 9, 2
    x = torch.randn(1, T, D)
    c = ctx_of([T])
    rms = lambda z, w: z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-6) * w  # noqa: E731
    cq = rms(x[0] @ m.wq_a.weight.T, m.q_norm.weight)
    q = (cq @ m.wq_b.weight.T).view(T, H, 8)
    ckv, kr = (x[0] @ m.wkv_a.weight.T).split((5, 4), -1)
    kv = (rms(ckv, m.kv_norm.weight) @ m.wkv_b.weight.T).view(T, H, 7)
    kc, v = kv.split((4, 3), -1)
    cos, sin = rotary_cos_sin(c, 4, 10000.0)
    qr, kr = apply_rotary(q[..., 4:], cos, sin), apply_rotary(kr.view(T, 1, 4), cos, sin)
    logits = (torch.einsum("thd,shd->hts", q[..., :4], kc) + torch.einsum("thd,sd->hts", qr, kr[:, 0])) / math.sqrt(8)
    logits = logits.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
    o = torch.einsum("hts,shd->thd", logits.softmax(-1), v).reshape(T, H * 3)
    torch.testing.assert_close(m(x, c)[0], o @ m.out_proj.weight.T, atol=1e-5, rtol=1e-5)


def test_mla_nope_layer_keeps_params():
    a = registry.build("mixer", "mla", D, num_heads=4, nope_every=2, layer_idx=0)
    b = registry.build("mixer", "mla", D, num_heads=4, nope_every=2, layer_idx=1)
    assert a.use_rope and not b.use_rope
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


# ----------------------------------------------------------------------------- NSA
def _nsa_qkv(m, x, c):
    T, H, Hk, hd = x.shape[1], m.num_heads, m.num_kv_heads, m.head_dim
    parts = m.Wqkv(x[0]).split((H * hd,) + (Hk * hd,) * (2 * m.n_kv_sets), -1)
    q, kv = parts[0].view(T, H, hd), [p.view(T, Hk, hd) for p in parts[1:]]
    if m.rotary_emb_dim:
        cos, sin = rotary_cos_sin(c, m.rotary_emb_dim, m.rope_theta)
        q = apply_rotary(q, cos, sin)
        kv = [apply_rotary(t, cos, sin) if i % 2 == 0 else t for i, t in enumerate(kv)]
    return q, (kv * 3 if len(kv) == 2 else kv)


def _naive_nsa(m, x, lengths):
    """Per-query loops over the NSA definitions (Eqs. 5, 7-12) for every segment separately."""
    outs, o0 = [], 0
    H, Hk, hd = m.num_heads, m.num_kv_heads, m.head_dim
    G, sc = H // Hk, m.softmax_scale
    for L in lengths:
        xs = x[:, o0:o0 + L]
        o0 += L
        c = ctx_of([L])
        q, (kc_, vc_, ks, vs, kw, vw) = _nsa_qkv(m, xs, c)
        gate = torch.sigmoid(m.gate_proj(xs[0])).view(L, 3, H)
        out = torch.zeros(L, H, hd)
        n_cmp = (L - m.l) // m.d + 1 if L >= m.l else 0
        Kc = [m.compress_k(kc_[i * m.d:i * m.d + m.l].transpose(0, 1)) for i in range(n_cmp)]  # [Hk, hd] each
        Vc = [m.compress_v(vc_[i * m.d:i * m.d + m.l].transpose(0, 1)) for i in range(n_cmp)]
        n_slc = -(-L // m.ls)
        for t in range(L):
            W = m.window_size
            vis = [i for i in range(n_cmp) if i * m.d + m.l - 1 <= t and (W < 0 or i * m.d >= t - W)]
            for a in range(Hk):
                imp = torch.zeros(n_slc)
                for g in range(G):
                    h = a * G + g
                    if vis:
                        p = torch.softmax(torch.stack([q[t, h] @ Kc[i][a] for i in vis]) * sc, 0)
                        out[t, h] += gate[t, 0, h] * sum(p[n] * Vc[i][a] for n, i in enumerate(vis))
                        for n, i in enumerate(vis):
                            for j in range(n_slc):
                                ov = min(i * m.d + m.l, (j + 1) * m.ls) - max(i * m.d, j * m.ls)
                                imp[j] += p[n] * max(ov, 0) / m.d
                cur = t // m.ls
                avail = [j for j in range(n_slc) if j <= cur and (W < 0 or (j + 1) * m.ls - 1 >= t - W)]
                forced = [j for j in avail if j < m.n_init or j > cur - m.n_local]
                rest = sorted((j for j in avail if j not in forced), key=lambda j: (-float(imp[j]), -j))
                chosen = forced + rest[:m.n - len(forced)]
                keys = sorted(s for j in chosen for s in range(j * m.ls, (j + 1) * m.ls)
                              if s <= t and (W < 0 or t - s <= W))
                w = m.sliding_window if W < 0 else min(m.sliding_window, W)
                wkeys = list(range(max(0, t - w), t + 1))
                for g in range(G):
                    h = a * G + g
                    p = torch.softmax(torch.stack([q[t, h] @ ks[s, a] for s in keys]) * sc, 0)
                    out[t, h] += gate[t, 1, h] * (p[:, None] * vs[keys, a]).sum(0)
                    p = torch.softmax(torch.stack([q[t, h] @ kw[s, a] for s in wkeys]) * sc, 0)
                    out[t, h] += gate[t, 2, h] * (p[:, None] * vw[wkeys, a]).sum(0)
        outs.append(m.out_proj(out.reshape(L, H * hd)))
    return torch.cat(outs, 0)[None]


@pytest.mark.parametrize("variant", ["nsa", "nsa_mha_shared_win"])
def test_nsa_matches_naive_loops(variant, force_sparse_backend):
    name, kw = VARIANTS[variant]
    m = make(name, **kw).eval()
    lengths = [26, 3, 17]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        ref = _naive_nsa(m, x, lengths)
        for backend in ("gather", "flex"):
            force_sparse_backend(backend)
            torch.testing.assert_close(m(x, ctx_of(lengths)), ref, atol=2e-5, rtol=2e-5, msg=backend)


@pytest.mark.parametrize("branch", ["slc", "win"])
def test_nsa_reduces_to_dense_attention(branch):
    """Gates saturated to one branch; with every block selected the selection branch is dense causal attention,
    and the window branch is sliding-window attention: both equal the ``attn`` mixer with the same weights."""
    kw = dict(num_heads=4, num_kv_heads=2, rotary_emb_dim=8, compress_block=4, compress_stride=2,
              select_block=4, select_n=50, sliding_window=7)
    m = make("nsa", **kw).eval()
    hq, hk = 4 * 8, 2 * 8
    on = {"slc": 1, "win": 2}[branch]
    a = registry.build("mixer", "attn", D, num_heads=4, num_kv_heads=2, rotary_emb_dim=8,
                       window_size=-1 if branch == "slc" else 7)
    with torch.no_grad():
        m.gate_proj.weight.zero_()
        m.gate_proj.bias.copy_(torch.tensor([40.0 if i == on else -40.0 for i in range(3)]).repeat_interleave(4))
        W = m.Wqkv.weight
        k0 = hq + 2 * hk * on
        a.Wqkv.weight.copy_(torch.cat([W[:hq], W[k0:k0 + 2 * hk]]))
        a.out_proj.weight.copy_(m.out_proj.weight)
    lengths = [37, 5, 20]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        torch.testing.assert_close(m(x, ctx_of(lengths)), a(x, ctx_of(lengths)), atol=1e-5, rtol=1e-5)


def test_nsa_selection_is_sparse_with_defaults():
    m = registry.build("mixer", "nsa", 64, num_heads=4, num_kv_heads=2)
    assert m.n * m.ls == 64 and m.sliding_window == 32  # documented small-scale defaults
    c = ctx_of([200])
    imp = torch.rand(1, 2, 200, (200 - m.l) // m.d + 1)
    blocks, valid = m._select(imp, c, torch.device("cpu"))
    assert int(valid[0, 0, -1].sum()) == m.n < -(-200 // m.ls)  # strictly fewer blocks than available


# ----------------------------------------------------------------------------- DSA
def _copy_base(dsa, base_name, **kw):
    b = registry.build("mixer", base_name, D, **kw)
    b.load_state_dict({k[len("attn."):]: v for k, v in dsa.state_dict().items() if k.startswith("attn.")})
    return b


@pytest.mark.parametrize("base", ["mla", "attn"])
def test_dsa_topk_ge_context_equals_dense_base(base):
    kw = dict(num_heads=4, num_kv_heads=2, rotary_emb_dim=4) if base == "attn" else dict(num_heads=4, q_lora_rank=10)
    m = make("dsa", base=base, index_topk=64, index_n_heads=2, index_head_dim=8, **kw).eval()
    b = _copy_base(m, base, **kw).eval()
    lengths = [30, 2, 21]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        torch.testing.assert_close(m(x, ctx_of(lengths)), b(x, ctx_of(lengths)), atol=1e-5, rtol=1e-5)


def test_dsa_matches_naive_topk(force_sparse_backend):
    kw = dict(num_heads=4, num_kv_heads=2, rotary_emb_dim=4)
    m = make("dsa", base="attn", index_topk=5, index_n_heads=3, index_head_dim=8, window_size=11, **kw).eval()
    lengths = [25, 14]
    x = torch.randn(1, sum(lengths), D)
    a = m.attn
    outs, o0 = [], 0
    with torch.no_grad():
        for L in lengths:
            xs, c = x[:, o0:o0 + L], ctx_of([L])
            o0 += L
            q, k, v, _ = ds._attn_project(a, xs[0], c)
            iq, ik, iw = m.indexer(xs[0], xs[0], c, True)
            out = torch.zeros(L, 4, 8)
            for t in range(L):
                cand = [s for s in range(t + 1) if t - s <= 11]
                score = torch.stack([(iw[t] * (iq[t] @ ik[s]).relu()).sum() for s in cand])
                keys = sorted(cand, key=lambda s: (-float(score[cand.index(s)]), -s))[:5]  # ties: most recent
                for h in range(4):
                    p = torch.softmax(torch.stack([q[t, h] @ k[s, h // 2] for s in keys]) * a.softmax_scale, 0)
                    out[t, h] = (p[:, None] * v[keys, h // 2]).sum(0)
            outs.append(a.out_proj(out.reshape(L, 32)))
        ref = torch.cat(outs)[None]
        for backend in ("gather", "flex"):
            force_sparse_backend(backend)
            torch.testing.assert_close(m(x, ctx_of(lengths)), ref, atol=1e-5, rtol=1e-5, msg=backend)


def test_dsa_indexer_loss_inject_equals_explicit():
    """'inject' delivers exactly d(coef * KL)/d(indexer) without touching the main model's gradients."""
    name, kw = VARIANTS["dsa"]
    x = torch.randn(1, 40, D)
    c = ctx_of([25, 15])
    grads = {}
    for mode in ("inject", "attr", "off"):
        m = make(name, **{**kw, "index_loss": mode, "index_loss_coef": 0.7}).train()
        y = m(x, c)
        loss = y.square().sum()
        if mode == "attr":
            assert m.aux_loss.requires_grad
            loss = loss + ds.mixer_aux_loss(m)
        loss.backward()
        grads[mode] = {n: (p.grad.clone() if p.grad is not None else None) for n, p in m.named_parameters()}
        if mode != "off":
            kl = float(m.index_kl)
            assert kl >= 0 and math.isfinite(kl)
            torch.testing.assert_close(float(m.aux_loss), 0.7 * kl)
        else:
            assert m.aux_loss is None
    for n, g in grads["inject"].items():
        torch.testing.assert_close(g, grads["attr"][n], atol=1e-6, rtol=1e-5, msg=n)
        if n.startswith("indexer."):
            assert g.abs().sum() > 0 and grads["off"][n] is None
        else:
            torch.testing.assert_close(g, grads["off"][n], atol=1e-6, rtol=1e-5, msg=n)


def test_dsa_kl_is_zero_when_indexer_matches_attention():
    """Top-1: target and indexer distributions are both a point mass on the selected key, so KL = 0."""
    name, kw = VARIANTS["dsa"]
    m = make(name, **{**kw, "index_topk": 1}).train()
    x = torch.randn(1, 12, D, requires_grad=True)
    m(x, ctx_of([12])).sum().backward()
    assert float(m.index_kl) == pytest.approx(0.0, abs=1e-6)


def test_dsa_no_aux_in_eval_or_no_grad():
    name, kw = VARIANTS["dsa"]
    m = make(name, **kw).train()
    with torch.no_grad():
        m(torch.randn(1, 10, D), ctx_of([10]))
    assert m.aux_loss is None
    m.eval()
    m(torch.randn(1, 10, D), ctx_of([10]))
    assert m.aux_loss is None


def test_dsa_chunked_topk_matches_single_chunk(monkeypatch):
    name, kw = VARIANTS["dsa"]
    m = make(name, **kw).eval()
    lengths = [33, 4, 27]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        ref = m(x, ctx_of(lengths))
        monkeypatch.setattr(ds, "INDEX_CHUNK_ELEMS", 50)  # forces chunks of a few queries
        torch.testing.assert_close(m(x, ctx_of(lengths)), ref)


# ----------------------------------------------------------------------------- backends
@pytest.mark.parametrize("variant", ["nsa", "nsa_mha_shared_win", "dsa", "dsa_attn_gqa"])
def test_flex_matches_gather_forward(variant, force_sparse_backend):
    """CPU: eager flex_attention (forward only; flex has no CPU backward) vs the gather reference."""
    name, kw = VARIANTS[variant]
    m = make(name, **kw).eval()
    lengths = [31, 6, 19]
    x = torch.randn(1, sum(lengths), D)
    with torch.no_grad():
        force_sparse_backend("gather")
        ref = m(x, ctx_of(lengths))
        force_sparse_backend("flex")
        torch.testing.assert_close(m(x, ctx_of(lengths)), ref, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("variant", ["nsa", "dsa", "dsa_attn_gqa"])
def test_cuda_flex_matches_gather_with_grads(variant, force_sparse_backend):
    name, kw = VARIANTS[variant]
    lengths = [131, 6, 90]
    x = torch.randn(1, sum(lengths), D, device="cuda")
    res = {}
    for backend in ("gather", "flex"):
        force_sparse_backend(backend)
        m = make(name, **kw).cuda().train()
        xx = x.clone().requires_grad_(True)
        y = m(xx, ctx_of(lengths, "cuda"))
        y.square().sum().backward()
        res[backend] = (y.detach(), xx.grad, {n: p.grad for n, p in m.named_parameters()})
    torch.testing.assert_close(res["flex"][0], res["gather"][0], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(res["flex"][1], res["gather"][1], atol=2e-3, rtol=2e-3)
    for n, g in res["gather"][2].items():
        torch.testing.assert_close(res["flex"][2][n], g, atol=2e-3, rtol=2e-3, msg=n)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_cuda_fast_paths_match_reference(variant, force_sparse_backend):
    """CUDA bf16: automatic backends (flash/flex) vs the pure-PyTorch paths (SDPA + gather)."""
    name, kw = VARIANTS[variant]
    lengths = [300, 7, 150]
    torch.manual_seed(0)
    m = make(name, **kw).cuda().to(torch.bfloat16).eval()
    x = torch.randn(1, sum(lengths), D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        auto = m(x, SeqCtx.from_cu_seqlens(cu)).float()
        force_sparse_backend("gather")
        old = attn_mod.FORCE_BACKEND
        attn_mod.FORCE_BACKEND = "sdpa"
        try:
            ref = m(x, SeqCtx.from_cu_seqlens(cu)).float()
        finally:
            attn_mod.FORCE_BACKEND = old
    torch.testing.assert_close(auto, ref, atol=3e-2, rtol=3e-2)


# ----------------------------------------------------------------------------- FLOPs
def test_flops_conventions():
    L = 100.0
    a = dict(num_heads=4, num_kv_heads=2, window_size=-1)
    base = attn_flops(D, a, L, SPEC)
    # DSA on plain attention with top-k >= L = attention + indexer (projections + Eq. 1 scores over the causal ctx)
    f = ds.DeepSeekSparseAttention.flops_per_position(D, L, SPEC, base="attn", index_topk=1000, index_n_heads=2,
                                                      index_head_dim=8, **a)
    ctx = sum(t + 1 for t in range(100)) / 100
    indexer = 2 * D * 2 * 8 + 2 * D * 8 + 2 * D * 2 + (2 * 8 * 2 + 3 * 2) * ctx
    assert f == pytest.approx(base + indexer)
    # MLA with qk = v = head_dim and kv_lora_rank r: the projections are the only difference to MHA
    m = ds.MultiHeadLatentAttention.flops_per_position(D, L, SPEC, num_heads=4, qk_nope_head_dim=4,
                                                       qk_rope_head_dim=4, kv_lora_rank=16)
    proj = 2 * D * 32 + 2 * D * (16 + 4) + 2 * 16 * 4 * (4 + 8) + 2 * 32 * D
    quad = 4 * ctx * 32 + 3 * 4 * ctx
    assert m == pytest.approx(proj + quad)
    # sparsity: NSA / DSA attend to fewer keys than dense at long L
    long = 2000.0
    dense = ds.MultiHeadLatentAttention.flops_per_position(D, long, SPEC, num_heads=4)
    sparse = ds.DeepSeekSparseAttention.flops_per_position(D, long, SPEC, num_heads=4, index_n_heads=2,
                                                           index_head_dim=8)
    assert sparse < dense
    full = FlopsConvention(attn="full")
    nsa_full = ds.NativeSparseAttention.flops_per_position(D, long, full, num_heads=4)
    assert nsa_full < attn_flops(D, dict(num_heads=4), long, full)


def test_model_flops_training_multiplier_passes_through():
    """model_flops(training) = 3 x forward for the new mixers exactly as for attn (dense swap = n_layers x diff)."""
    cfg = tiny_dense_cfg("attn")
    kw = dict(kv_lora_rank=16)
    cfg2 = tiny_dense_cfg("mla", **kw)
    diff = model_flops(cfg2, seq_len=64)["flops_per_token"] - model_flops(cfg, seq_len=64)["flops_per_token"]
    a = dict(num_heads=4, num_kv_heads=None, rope_theta=10000.0, nope_every=0)
    per = ds.MultiHeadLatentAttention.flops_per_position(D, 64, SPEC, **a, **kw) - attn_flops(
        D, dict(num_heads=4, window_size=-1), 64, SPEC)
    assert diff == pytest.approx(3 * cfg.n_layers * per)


# ----------------------------------------------------------------------------- whole models
def tiny_hnet_cfg(name, main_kw=None):
    return HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T1m1", ["T2"], "m1T1"], "m1"],
        d_model=[32, 32, 48],
        d_intermediate=[0, 64, 96],
        ssm=dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8),
        attn=dict(num_heads=[2, 2, 4], rotary_emb_dim=[8, 8, 6], window_size=[7, 7, -1]),
        dc=dict(target_ratios=[2.0, 2.0]),
        mixer_codes=[{"m": "mamba2", "t": "attn"}, {"m": "mamba2", "t": "attn"}, {"m": "mamba2", "t": name}],
        mixer_cfg={name: [{}, {}, main_kw or {}]},
    ))


def tiny_dense_cfg(name, **kw):
    return DenseConfig(vocab_size=97, d_model=D, n_layers=2, d_intermediate=64, num_heads=4, mixer=name,
                       mixer_kwargs=kw)


MODEL_KW = {
    "mla": dict(kv_lora_rank=16, q_lora_rank=12, qk_nope_head_dim=6, qk_rope_head_dim=6),
    "nsa": dict(num_kv_heads=2, **NSA_SMALL),
    "dsa": dict(kv_lora_rank=16, q_lora_rank=12, index_topk=4, index_n_heads=2, index_head_dim=8),
}


@pytest.mark.parametrize("name", ["mla", "nsa", "dsa"])
@pytest.mark.parametrize("kind", ["hnet", "dense"])
def test_tiny_models_forward_backward(name, kind):
    torch.manual_seed(0)
    if kind == "hnet":
        cfg = tiny_hnet_cfg(name, MODEL_KW[name])
        ids = torch.randint(0, 254, (90,))
    else:
        kw = dict(MODEL_KW[name])
        kw.pop("num_kv_heads", None)
        cfg = tiny_dense_cfg(name, **kw)
        ids = torch.randint(0, 97, (90,))
    model = build_model(cfg)
    model.init_weights()
    model.train()
    assert count_params(cfg) == sum(p.numel() for p in model.parameters())
    mixers = [mm for mm in model.modules() if isinstance(mm, registry.get("mixer", name))]
    assert mixers, "variant mixer not built"
    cu = torch.tensor([0, 50, 90], dtype=torch.int32)
    labels = torch.cat([ids[1:], ids.new_tensor([IGNORE_INDEX])])
    labels[49] = IGNORE_INDEX
    out = model(ids, cu_seqlens=cu, labels=labels)
    (out.loss + out.aux_loss).backward()
    assert torch.isfinite(out.loss)
    dead = [n for n, p in model.named_parameters() if p.grad is None]
    assert not dead, dead[:5]
    # packed == separate documents through the whole model
    model.eval()
    with torch.no_grad():
        full = model(ids, cu_seqlens=cu).logits
        torch.testing.assert_close(full[50:], model(ids[50:]).logits, atol=1e-4, rtol=1e-4)


# ----------------------------------------------------------------------------- configs
VARIANT_CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "models", "tiny_*_2m_*.yaml")))
VARIANT_CONFIGS = [p for p in VARIANT_CONFIGS if p.rsplit("_", 1)[-1][:-5] in ("mla", "nsa", "dsa")]
MIXER_KEYS = {"mixer", "mixer_kwargs", "mixer_codes", "mixer_cfg", "num_kv_heads"}


def _flops_of(path):
    cfg = model_config_from_yaml(path)
    m = load_yaml(path)["matching"]
    if isinstance(cfg, HNetConfig):
        return cfg, model_flops(cfg, seq_len=m["seq_len_bytes"])["flops_per_byte"]
    return cfg, model_flops(cfg, seq_len=m["seq_len_tokens"], bytes_per_token=m["bytes_per_token"])["flops_per_byte"]


def test_variant_configs_exist():
    names = {os.path.basename(p)[:-5] for p in VARIANT_CONFIGS}
    assert names == {f"tiny_{k}_2m_{v}" for k in ("bpe", "hnet2") for v in ("mla", "nsa", "dsa")}


@pytest.mark.parametrize("path", VARIANT_CONFIGS, ids=lambda p: os.path.basename(p)[:-5])
def test_variant_configs_matched_and_minimal(path):
    y = load_yaml(path)
    base_name = y["matching"]["ablation_of"]
    base = load_yaml(os.path.join(ROOT, "configs", "models", base_name + ".yaml"))
    variant = os.path.basename(path)[:-5].rsplit("_", 1)[-1]
    # identical to the base except the attention-mixer fields
    strip = lambda d: {k: v for k, v in d.items() if k not in MIXER_KEYS}  # noqa: E731
    assert strip(y["model"]) == strip(base["model"])
    skip = ("ablation_of", "reference")
    assert {k: v for k, v in y["matching"].items() if k not in skip} == {
        k: v for k, v in base["matching"].items() if k not in skip}
    cfg, fpb = _flops_of(path)
    if isinstance(cfg, DenseConfig):
        assert cfg.mixer == variant
    else:
        assert cfg.mixer_codes_for_stage(cfg.n_stages)["t"] == variant
        assert all(cfg.mixer_codes_for_stage(s)["t"] == "attn" for s in range(cfg.n_stages))
    ref = y["matching"]["reference"]
    _, ref_fpb = _flops_of(os.path.join(ROOT, "configs", "models", ref + ".yaml"))
    _, base_fpb = _flops_of(os.path.join(ROOT, "configs", "models", base_name + ".yaml"))
    assert abs(fpb / ref_fpb - 1) <= 0.02
    assert abs(fpb / base_fpb - 1) <= 0.01
    model = build_model(cfg)  # builds (meta-free) and the param count matches the analytic count
    assert count_params(cfg) == sum(p.numel() for p in model.parameters())


def test_dsa_inject_survives_activation_checkpointing():
    """The trainer wraps blocks in non-reentrant checkpointing; injected indexer gradients must be unchanged."""
    from torch.utils.checkpoint import checkpoint

    name, kw = VARIANTS["dsa"]
    x = torch.randn(1, 40, D)
    c = ctx_of([25, 15])
    grads = []
    for ckpt in (False, True):
        m = make(name, **kw).train()
        y = checkpoint(m, x, c, use_reentrant=False) if ckpt else m(x, c)
        y.square().sum().backward()
        grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
    for n, g in grads[0].items():
        torch.testing.assert_close(grads[1][n], g, msg=n)
