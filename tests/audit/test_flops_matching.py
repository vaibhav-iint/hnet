"""Audit: are the "equal FLOPs" pairs equal at the compression the H-Nets actually *realize*?

``configs/MATCHING.md`` / ``scripts/search/fairness.py`` match H-Nets to their BPE partner at the *target*
compression (``hnet.flops`` default ``bytes_per_position`` = cumulative target ratios). With the paper's threshold
decision (p > 0.5) nothing pins the realized compression to the target, and the committed EXP-002 metrics show the
routers settle below target (more chunks), i.e. over the FLOPs budget. With the rank ("quantile") decider used by
the ladder the realized compression is pinned, so the ladder pairs hold at the realized compression too.
"""

from __future__ import annotations

import glob
import json
import os

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import model_flops
from hnet.model import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOL = 0.02  # MATCHING.md: "target: within +-2%"


def _p(*a):
    return os.path.join(ROOT, *a)


def test_ladder_pairs_match_within_1pct_at_target():
    man = json.load(open(_p("configs/ladder/ladder.json")))
    bpt, sb, st = man["options"]["bytes_per_token"], man["seq_bytes"], man["options"]["seq_tokens"]
    for f in sorted(glob.glob(_p("configs/ladder/models/bpe_L*.yaml"))):
        b = model_flops(model_config_from_yaml(f), seq_len=st, bytes_per_token=bpt)["flops_per_byte"]
        for v in ("hnet", "hnet_ws"):
            c = model_config_from_yaml(f.replace("bpe_", v + "_"))
            assert c.dc.decision == "quantile"  # realized compression pinned to the target in training
            h = model_flops(c, seq_len=sb)["flops_per_byte"]
            assert abs(h / b - 1) <= 0.01, (f, v, h / b - 1)


@pytest.mark.parametrize("name", ["tiny_hnet1_2m", "tiny_hnet2_2m", "tiny_hnet2_2m_ratios42"])
def test_tiny_pairs_match_at_target(name):
    b = model_flops(model_config_from_yaml(_p("configs/models/tiny_bpe_2m.yaml")), seq_len=448,
                    bytes_per_token=4.6)["flops_per_byte"]
    h = model_flops(model_config_from_yaml(_p(f"configs/models/{name}.yaml")), seq_len=2048)["flops_per_byte"]
    assert abs(h / b - 1) <= TOL, h / b - 1


def _realized(run_dir):
    recs = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    recs = [r for r in recs if r.get("flops_per_byte")]
    tail = recs[len(recs) // 2:]  # second half of training (past the early router collapse)
    return sum(r["flops_per_byte"] for r in tail) / len(tail)


EXP002 = _p("runs/gcp/exp002_tiny2m")


@pytest.mark.skipif(not os.path.isdir(EXP002), reason="EXP-002 run metrics not present")
@pytest.mark.parametrize("arm", ["paper_s0", "paper_s1", "end_cosine_shift_s0", "end_mlp_s1", "paper_nolrmult_s0"])
def test_threshold_router_runs_stay_within_budget_at_realized_compression(arm):
    """FAILS: realized training FLOPs/byte of the paper-router (threshold-decision) tiny runs are 7-12% above the BPE
    partner's, although the configs are matched within 2% at the target ratios (the gate checks only targets)."""
    bpe = _realized(os.path.join(EXP002, "bpe_s0"))
    h = _realized(os.path.join(EXP002, arm))
    assert abs(h / bpe - 1) <= TOL, f"{arm}: realized FLOPs/byte {h / bpe - 1:+.1%} vs BPE"


def test_quantile_decider_pins_realized_compression():
    cfg = model_config_from_yaml(_p("configs/models/tiny_hnet1_2m.yaml"))
    cfg.dc.router, cfg.dc.decision = "end_mlp", "quantile"
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    m.train()
    ids = torch.randint(0, 254, (8, 512))
    with torch.no_grad():
        out = m(ids)
    T, n_sel = out.bpred[0].boundary_mask.numel(), int(out.bpred[0].boundary_mask.sum())
    n_free = T - 8  # one forced boundary per row/segment
    assert n_sel == round(n_free / 6.0) + 8
