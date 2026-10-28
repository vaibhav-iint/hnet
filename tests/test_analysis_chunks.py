import torch

from hnet.analysis.chunks import chunk_cuts, chunk_report, render, summarize_cuts
from hnet.config import model_config_from_yaml
from hnet.model import build_model


def test_summarize_cuts_counts():
    s = summarize_cuts(["ab cd"], [[3]])  # "ab␣|cd" -> cut after a space
    assert s["n_cuts"] == 1 and s["pct_cut_after_space"] == 100.0 and s["bytes_per_chunk"] == 2.5
    assert summarize_cuts(["abcd"], [[2]])["pct_word_internal"] == 100.0
    assert summarize_cuts(["é"], [[1]])["pct_mid_utf8"] == 100.0


def test_chunk_cuts_on_tiny_model_both_semantics():
    torch.manual_seed(0)
    for router in ("cosine_lookahead", "end_mlp"):
        cfg = model_config_from_yaml("configs/tiny_cpu.yaml")
        cfg.dc.router = router
        model = build_model(cfg).eval()
        text = "Ever wondered if you could beat a bear in a race? The answers may surprise you."
        cuts = chunk_cuts(model, text)
        assert len(cuts) == cfg.n_stages
        n = len(text.encode())
        assert all(0 < c < n for c in cuts[0])
        assert set(cuts[1]) <= set(cuts[0])  # stage-1 chunks are unions of stage-0 chunks
        rep = chunk_report(model, [text])
        assert rep["s0"]["bytes_per_chunk"] <= rep["s1"]["bytes_per_chunk"]
        assert "|" in render(text, cuts[0]) or not cuts[0]
