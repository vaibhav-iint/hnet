"""The step-0 gradient audit (CLAUDE.md rules 1-2) sees the paths we know about and flags the known degeneracies."""

import torch

from hnet.analysis.grad_audit import ZERO, audit, degeneracies
from hnet.config import model_config_from_yaml


def _cfg(**dc):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.chunk_vocab = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]
    for k, v in dc.items():
        setattr(cfg.dc, k, v)
    return cfg


def test_audit_reports_the_known_paths_and_non_paths():
    ids = torch.randint(0, 256, (2, 256))
    table, metrics, groups = audit(_cfg(), ids)
    assert "byte_ce" in table and "next_chunk_ce/s0" in table and "ratio/s0" in table
    # the byte CE reaches the router through the STE; the next-chunk CE does not (computed before dechunking)
    assert table["byte_ce"]["s0.routing_module"] > 0
    assert table["next_chunk_ce/s0"]["s0.routing_module"] == ZERO
    # zero-initialised: the residual projection gets gradient (its input is live) but the innermost main network
    # gets no gradient from the next-chunk CE while the chunk table is zero
    assert table["next_chunk_ce/s0"]["s1.main_network"] == ZERO
    assert table["next_chunk_ce/s0"]["s0.chunk_vocab"] > 0
    assert "boundary_frac/s0" in metrics and 0 < metrics["boundary_frac/s0"] < 1
    warns = degeneracies(table, metrics, groups)
    assert any("sends the router no gradient" in w for w in warns)


def test_chunk_ste_opens_the_path_in_the_audit():
    ids = torch.randint(0, 256, (2, 256))
    table, metrics, groups = audit(_cfg(chunk_ste=True), ids, steps=3, lr=1e-2)  # a few steps move the table off 0
    assert table["next_chunk_ce/s0"]["s0.routing_module"] > 0
    assert not any("sends the router no gradient" in w for w in degeneracies(table, metrics, groups))
