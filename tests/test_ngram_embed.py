"""Hashed byte n-gram embeddings (hnet.model.ngram)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.ngram import HashNgramEmbedding

NG = {"ns": [3, 4, 5], "buckets": 257}


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def test_zero_init_is_identity_and_params_counted():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    base = count_params(cfg)
    cfg.ngram_embed = NG
    model = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in model.parameters())
    assert count_params(cfg) - base == 3 * 257 * cfg.d_model[0]
    ids = torch.randint(0, 256, (50,))
    assert torch.equal(model.ngram_embeddings(ids, _ctx([50])), torch.zeros(50, cfg.d_model[0]))
    assert model.stage_of_param("ngram_embeddings.tables.0.weight") == 0
    f0 = model_flops(model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml"))["flops_per_byte"]
    assert 0 < model_flops(cfg)["flops_per_byte"] - f0 < 1e-3 * f0  # only the adds


def test_causal_and_document_aware():
    torch.manual_seed(0)
    emb = HashNgramEmbedding(8, **NG)
    for t in emb.tables:
        torch.nn.init.normal_(t.weight)
    a = torch.randint(0, 256, (40,))
    out = emb(a, _ctx([40]))
    b = a.clone()
    b[30:] = (b[30:] + 1) % 256  # change the future: positions < 30 must not change
    torch.testing.assert_close(emb(b, _ctx([40]))[:30], out[:30])
    # a document starting at 20 sees no n-gram crossing its start: positions 20, 21 (n >= 3) get nothing from n=3..5
    two = emb(a, _ctx([20, 20]))
    assert torch.equal(two[20], torch.zeros(8)) and torch.equal(two[21], torch.zeros(8))
    # and position 22 of the 2nd doc gets exactly the n=3 row of its own 3 bytes, same as a fresh doc
    fresh = emb(a[20:], _ctx([20]))
    torch.testing.assert_close(two[20:], fresh)


def test_same_ngram_same_row():
    emb = HashNgramEmbedding(4, ns=[3], buckets=10007)
    torch.nn.init.normal_(emb.tables[0].weight)
    ids = torch.tensor([5, 6, 7, 1, 5, 6, 7])
    out = emb(ids, _ctx([7]))
    torch.testing.assert_close(out[2], out[6])
    assert not torch.allclose(out[2], out[3])


def test_trains():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.ngram_embed = NG
    torch.manual_seed(0)
    model = build_model(cfg)
    ids = torch.randint(0, 256, (1, 256))
    out = model(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    g = model.ngram_embeddings.tables[0].weight.grad
    assert g is not None and g.abs().sum() > 0
