"""Attention head_dim independent of d_model (needed by the ladder: heads = depth, head_dim 128 > D / heads):
CausalAttention shapes, H-Net / dense config plumbing, exact parameter counts and FLOPs, and the default unchanged."""

import pytest
import torch

from hnet.config import DenseConfig, HNetConfig
from hnet.flops import SPEC, attn_flops, attn_params, count_params, model_flops
from hnet.model import build_model
from hnet.model.mixers.attn import CausalAttention


def test_causal_attention_with_explicit_head_dim():
    m = CausalAttention(96, num_heads=3, head_dim=64, num_kv_heads=1)  # inner 192 > d_model 96, GQA
    assert m.Wqkv.weight.shape == (64 * (3 + 2), 96) and m.out_proj.weight.shape == (96, 192)
    assert m.rotary_emb_dim == 64 and m.softmax_scale == pytest.approx(64 ** -0.5)
    a = dict(num_heads=3, num_kv_heads=1, head_dim=64)
    assert sum(p.numel() for p in m.parameters()) == attn_params(96, a)
    # inner width 192: projections 2*96*(192+128) + 2*192*96, quadratic 4*ctx*192 (+3*h*ctx)
    ctx = (10 + 1) / 2
    assert attn_flops(96, a, 10, SPEC) == pytest.approx(2 * 96 * 320 + 2 * 192 * 96 + 4 * ctx * 192 + 9 * ctx)


def test_default_head_dim_is_unchanged():
    m = CausalAttention(96, num_heads=3)
    assert m.head_dim == 32 and m.out_proj.weight.shape == (96, 96)
    h = HNetConfig.from_dict(dict(arch_layout=["m1", ["T1"], "m1"], d_model=[64, 96], d_intermediate=[0, 256],
                                  attn=dict(num_heads=[2, 3]), dc=dict(target_ratios=[3.0])))
    assert "head_dim" not in h.attn_for_stage(1)  # mixers without the knob see the same kwargs as before
    with pytest.raises(AssertionError):
        CausalAttention(100, num_heads=3)


def test_dense_and_hnet_configs_count_and_run():
    torch.manual_seed(0)
    d = DenseConfig(vocab_size=300, d_model=64, n_layers=2, d_intermediate=128, num_heads=3, head_dim=32)
    md = build_model(d)
    assert sum(p.numel() for p in md.parameters()) == count_params(d)
    assert {b.mixer.head_dim for b in md.blocks()} == {32}
    ids = torch.randint(0, 300, (2, 16))
    out = md(ids, labels=ids)
    out.loss.backward()
    assert torch.isfinite(out.loss)
    base = model_flops(DenseConfig(vocab_size=300, d_model=64, n_layers=2, d_intermediate=128, num_heads=2),
                       seq_len=16)
    assert model_flops(d, seq_len=16)["flops_per_token"] > base["flops_per_token"]  # inner 96 > 64

    h = HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T2"], "m1"], d_model=[64, 96], d_intermediate=[0, 256], vocab_size=256,
        ssm=dict(d_state=16, headdim=32, chunk_size=64),
        attn=dict(num_heads=[2, 3], head_dim=[None, 64], rotary_emb_dim=[None, None]),
        dc=dict(target_ratios=[3.0], router="end_mlp", decision="quantile")))
    a = h.attn_for_stage(1)
    assert a["head_dim"] == 64 and a["rotary_emb_dim"] == 64
    mh = build_model(h)
    assert sum(p.numel() for p in mh.parameters()) == count_params(h)
    ids = torch.randint(0, 250, (1, 96))
    out = mh(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    assert torch.isfinite(out.loss)
