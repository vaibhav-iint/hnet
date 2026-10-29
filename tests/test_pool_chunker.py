"""Pooling chunker (hnet.model.chunking.pool): chunk vectors that summarize the whole chunk."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.chunking.pool import PoolChunker
from hnet.model.ctx import SeqCtx

CFG = "configs/models/tiny_hnet2_2m.yaml"


def _cfg(chunker="pool", mode="mean", router="end_mlp"):
    cfg = model_config_from_yaml(CFG)
    cfg.dc.router = router
    cfg.dc.chunker = chunker
    cfg.dc.chunker_kwargs = {"mode": mode} if chunker == "pool" else {}
    return cfg


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def test_mean_pool_by_hand():
    # two segments; ends (E) at: seg A = positions 0,2,4 (then a trailing unfinished chunk 5,6); seg B = 7,9
    mask = torch.tensor([1, 0, 1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)
    h = torch.arange(10, dtype=torch.float32)[:, None].repeat(1, 4)
    ch = PoolChunker(4, mode="mean")
    with torch.no_grad():
        ch.proj.weight.copy_(torch.eye(4))  # x = h[end] + mean(chunk)
    x, ctx_inner = ch(h, mask, _ctx([7, 3]))
    means = torch.tensor([0.0, 1.5, 3.5, 7.0, 8.5])  # chunks: {0}, {1,2}, {3,4}; trailing {5,6} dropped; {7}, {8,9}
    ends = torch.tensor([0.0, 2.0, 4.0, 7.0, 9.0])
    torch.testing.assert_close(x[:, 0], ends + means)
    assert ctx_inner.cu_seqlens.tolist() == [0, 3, 5]


def test_attn_pool_uniform_query_equals_mean_of_v():
    mask = torch.tensor([1, 0, 0, 1, 0, 1], dtype=torch.bool)
    torch.manual_seed(0)
    h = torch.randn(6, 8)
    ch = PoolChunker(8, mode="attn")
    with torch.no_grad():
        ch.query.zero_()  # uniform weights -> mean of v over the chunk
        ch.proj.weight.copy_(torch.eye(8))
    x, _ = ch(h, mask, _ctx([6]))
    v = ch.v_proj(h)
    expect = h[[0, 3, 5]] + torch.stack([v[0], v[1:4].mean(0), v[4:6].mean(0)])
    torch.testing.assert_close(x, expect)


@pytest.mark.parametrize("mode", ["mean", "attn"])
def test_zero_init_equals_select_and_trains(mode):
    torch.manual_seed(0)
    sel = build_model(_cfg("select"))
    torch.manual_seed(0)
    pool = build_model(_cfg("pool", mode))
    pool.load_state_dict(sel.state_dict(), strict=False)
    ids = torch.randint(0, 256, (2, 160))
    sel.eval(), pool.eval()
    torch.testing.assert_close(pool(ids, labels=ids).loss, sel(ids, labels=ids).loss)
    pool.train()
    out = pool(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    for st in pool.backbone.stages():
        if not st.is_innermost:
            assert st.chunk_layer.proj.weight.grad.abs().sum() > 0
    cfg = _cfg("pool", mode)
    assert count_params(cfg) == sum(p.numel() for p in pool.parameters())
    base = model_flops(_cfg("select"))["flops_per_byte"]
    extra = model_flops(cfg)["flops_per_byte"] / base - 1
    assert 0 < extra < (0.02 if mode == "mean" else 0.06)


def test_full_model_is_causal_with_pooling():
    torch.manual_seed(0)
    m = build_model(_cfg("pool", "attn"))
    with torch.no_grad():  # make pooling matter
        for st in m.backbone.stages():
            if not st.is_innermost:
                torch.nn.init.eye_(st.chunk_layer.proj.weight)
    m.eval()
    ids = torch.randint(0, 256, (1, 200))
    a = m(ids).logits[0]
    ids2 = ids.clone()
    ids2[0, 120:] = (ids2[0, 120:] + 7) % 254
    b = m(ids2).logits[0]
    torch.testing.assert_close(a[:120], b[:120])
    assert not torch.allclose(a[120:], b[120:])


def test_pool_refuses_start_semantics():
    with pytest.raises(AssertionError, match="requires 'end'"):
        build_model(_cfg("pool", "mean", router="cosine_lookahead"))
