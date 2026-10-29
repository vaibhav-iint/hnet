"""Hashed vocabulary for learned chunks (hnet.model.chunk_vocab)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.flops import model_flops
from hnet.model import build_model
from hnet.model.chunk_vocab import IGNORE, ByteInfo, ChunkVocab, hash_bank
from hnet.model.chunking.select import select_positions
from hnet.model.ctx import SeqCtx


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def _setup(text_segments, ends_per_segment):
    """Bytes of several segments (BOS prepended), with the given chunk-end offsets (end semantics)."""
    ids, mask, lengths = [], [], []
    for text, ends in zip(text_segments, ends_per_segment):
        b = [0] + list(text.encode())
        m = [False] * len(b)
        m[0] = True
        for e in ends:
            m[e] = True
        ids += b
        mask += m
        lengths.append(len(b))
    ids, mask = torch.tensor(ids), torch.tensor(mask)
    ctx = _ctx(lengths)
    _, idx, ctx_inner = select_positions(mask, ctx)
    binfo = ByteInfo(hash_bank(ids, ctx, 2, 16), torch.arange(ids.numel()))
    return ids, ctx, ctx_inner, binfo


def test_identical_chunk_strings_hash_to_the_same_bucket_and_differ_otherwise():
    # "the cat" / "the dog": chunks "the", " cat" | "the", " dog" (ends at byte offsets 3, 7 after BOS)
    ids, ctx, ctx_inner, binfo = _setup(["the cat", "the dog"], [[3, 7], [3, 7]])
    cv = ChunkVocab(8, buckets=1 << 20, n_hash=2)
    bucket = cv.chunk_ids(binfo, ctx, ctx_inner)
    assert bucket.shape == (2, 6)  # per segment: BOS chunk, "the", " cat"/" dog"
    b = bucket[0].tolist()
    assert b[0] == b[3] and b[1] == b[4]  # BOS chunk and "the" repeat across segments
    assert b[2] != b[5]  # " cat" vs " dog"
    assert len(set(b[:3])) == 3
    assert torch.equal(bucket[0], bucket[0]) and not torch.equal(bucket[0], bucket[1])  # independent hashes


def test_chunk_hash_depends_only_on_the_chunk_content():
    # the same chunk " cat" at different positions / after different prefixes
    ids, ctx, ctx_inner, binfo = _setup(["the cat", "a big cat"], [[3, 7], [1, 5, 9]])
    cv = ChunkVocab(8, buckets=1 << 20, n_hash=1)
    b = cv.chunk_ids(binfo, ctx, ctx_inner)[0].tolist()  # BOS, "the", " cat" | BOS, "a", " big", " cat"
    assert len(b) == 7
    assert b[2] == b[6]  # " cat" == " cat"
    assert b[1] != b[5]  # "the" vs " big"


def test_long_chunks_use_their_last_max_len_bytes():
    ids, ctx, ctx_inner, binfo = _setup(["x" * 30 + "abcd", "y" * 5 + "abcd"], [[34], [9]])
    cv = ChunkVocab(8, buckets=1 << 20, n_hash=1, max_len=4)
    b = cv.chunk_ids(binfo, ctx, ctx_inner)[0].tolist()
    assert b[1] == b[3]  # both chunks end in "abcd" and are capped at 4 bytes


def test_next_targets_stop_at_segment_ends():
    ids, ctx, ctx_inner, binfo = _setup(["the cat", "the dog"], [[3, 7], [3, 7]])
    cv = ChunkVocab(8, buckets=1 << 20, n_hash=1)
    bucket = cv.chunk_ids(binfo, ctx, ctx_inner)[0]
    tgt = ChunkVocab.next_targets(bucket, ctx_inner)
    assert tgt.tolist() == [bucket[1].item(), bucket[2].item(), IGNORE, bucket[4].item(), bucket[5].item(), IGNORE]


def test_embedding_is_zero_at_init_and_pred_loss_is_log_candidates():
    torch.manual_seed(0)
    ids, ctx, ctx_inner, binfo = _setup(["the cat sat", "the dog ran"], [[3, 7, 11], [3, 7, 11]])
    cv = ChunkVocab(16, buckets=4096, n_hash=2, pred_weight=1.0, n_neg=64)
    bucket = cv.chunk_ids(binfo, ctx, ctx_inner)
    assert torch.equal(cv.embed(bucket, torch.float32), torch.zeros(8, 16))
    tgt = ChunkVocab.next_targets(bucket[0], ctx_inner)
    z = torch.randn(8, 16)
    loss = cv.pred_loss(z, tgt)
    n_cand = len(torch.unique(tgt[tgt != IGNORE]))
    assert abs(float(loss) - torch.log(torch.tensor(float(n_cand)))) < 1e-5  # all logits zero: uniform
    loss.backward()
    assert cv.tables[0].weight.grad.abs().sum() > 0


def test_model_with_chunk_vocab_trains_and_counts_params_and_flops():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    base = model_flops(cfg)["flops_per_byte"]
    cfg.dc.chunk_vocab = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 1024}, None]
    rep = model_flops(cfg)
    comps = rep["per_stage"][0]["components"]
    assert comps["chunk_vocab"]["params"] == 2 * 4096 * cfg.d_model[0]
    assert 0 < (rep["flops_per_byte"] - base) / base < 0.03  # the sampled head is a small training-only cost
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    assert sum(p.numel() for p in m.parameters()) == rep["params"]
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    assert "chunk_vocab_ce/s0" in out.stats and torch.isfinite(out.stats["chunk_vocab_ce/s0"])
    (out.loss + out.aux_loss).backward()
    assert m.backbone.chunk_vocab.tables[1].weight.grad.abs().sum() > 0
    assert m.stage_of_param("backbone.chunk_vocab.tables.0.weight") == 0
    m.eval()
    assert torch.isfinite(m(ids, labels=ids).loss)
