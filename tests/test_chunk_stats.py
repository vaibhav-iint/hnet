"""Tokenizer statistics of a model's chunks (hnet.analysis.chunk_stats)."""

import torch

from hnet.analysis.chunk_stats import _consistency, _segments, _stats, _whitespace_starts, analyze
from hnet.config import model_config_from_yaml
from hnet.model import build_model


def test_whitespace_segmentation_and_stats():
    ids = [0] + list(b"the cat the dog")
    ws = _whitespace_starts(ids)
    segs = _segments(ids, ws)
    assert segs == [b"the ", b"cat ", b"the ", b"dog"]
    st = _stats(segs, len(ids))
    assert st["chunks"] == 4 and st["types"] == 3 and abs(st["unigram_bits_per_chunk"] - 1.5) < 1e-6


def test_consistency_counts_segmentations_per_word():
    ids = [0] + list(b"cat cat cat")
    ws = _whitespace_starts(ids)
    per_word = _consistency(ids, [1, 2, 5, 6, 9], ws)  # "cat " split as c|at twice, "cat" (last) unsplit
    assert dict(per_word[b"cat "]) == {(1,): 2}
    assert dict(per_word[b"cat"]) == {(): 1}


def test_analyze_runs_on_a_tiny_model():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    torch.manual_seed(0)
    m = build_model(cfg).eval()
    docs = [b"the quick brown fox jumps over the lazy dog. " * 4, b"a b c d e f g h i j k l m n o p"]
    rep = analyze(m, docs)
    assert rep["bytes"] == sum(len(d) + 1 for d in docs)
    assert len(rep["stages"]) == 2
    for st in rep["stages"]:
        assert 0 <= st["boundaries_on_whitespace_frac"] <= 1 and st["chunks"] > 0
