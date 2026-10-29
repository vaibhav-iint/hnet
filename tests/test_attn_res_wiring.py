"""Attention Residuals wired through configs, models and the FLOPs counter (hnet.model.isotropic)."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import SPEC, attn_res_counts, count_params, model_flops
from hnet.model import build_model

CFGS = ["configs/models/tiny_hnet2_2m.yaml", "configs/models/tiny_bpe_2m.yaml"]


def _with_attn_res(path, S):
    cfg = model_config_from_yaml(path)
    cfg.attn_res = S
    return cfg


@pytest.mark.parametrize("path", CFGS)
@pytest.mark.parametrize("S", [0, 1, 3])
def test_param_count_matches_model(path, S):
    cfg = _with_attn_res(path, S)
    with torch.device("meta"):
        model = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("path", CFGS)
def test_attn_res_adds_flops_only_when_on(path):
    off, on = (model_flops(_with_attn_res(path, S))["flops_per_byte"] for S in (0, 2))
    assert on > off and (on - off) / off < 0.05  # depth attention is a small overhead


def test_counts_by_hand():
    # height 4, blocks of 2: operators j = 0..4 see 1, 2, 2, 3, 3 sources (j = 0 is a pass-through)
    p, f = attn_res_counts(4, 2, 8, SPEC)
    assert p == 2 * 8 * 5
    per_src = 4 * 8 + ((4 * 8 + 3) if SPEC.count_elementwise else 0)
    assert f == per_src * (2 + 2 + 3 + 3)


def test_hnet_forward_with_attn_res():
    cfg = _with_attn_res(CFGS[0], [0, 0, 2])  # main network only
    torch.manual_seed(0)
    model = build_model(cfg)
    ids = torch.randint(0, 256, (1, 300))
    cu = torch.tensor([0, 120, 300], dtype=torch.int32)
    out = model(ids[0], cu_seqlens=cu, max_seqlen=180)
    assert torch.isfinite(out.logits).all()
