import math

import pytest
import torch
from torch import nn

from hnet.config import HNetConfig
from hnet.optim import auto_lr_multipliers, build_optimizer, param_groups, resolve_lr_multipliers, set_lr
from hnet.registry import names

torch.set_num_threads(1)


def test_auto_lr_multipliers_paper_xl_example():
    # Paper App. C Eq. 11, 2-stage XL: (3,3)-DC, D = 1024/1536/2048, N_GPT = 4.6.
    lam = auto_lr_multipliers([1024, 1536, 2048], [3.0, 3.0], 4.6)
    expect = [math.sqrt(4.6 * 2048 / 1024), math.sqrt(4.6 / 3 * 2048 / 1536), math.sqrt(4.6 / 9)]
    assert lam == pytest.approx(expect)
    assert lam == pytest.approx([3.0332, 1.4298, 0.7149], abs=1e-4)
    # literal form of Eq. 11: prod_{i=s}^S N^i / prod_{i=0}^S N^i with N^S = 1
    N = [3.0, 3.0, 1.0]
    for s in range(3):
        lit = math.sqrt(4.6 * math.prod(N[s:]) / math.prod(N) * 2048 / [1024, 1536, 2048][s])
        assert lam[s] == pytest.approx(lit)
    # 1-stage 6-DC with equal widths: [sqrt(4.6), sqrt(4.6/6)]
    assert auto_lr_multipliers([1024, 1024], [6.0]) == pytest.approx([math.sqrt(4.6), math.sqrt(4.6 / 6)])


def test_resolve_lr_multipliers_from_config():
    cfg = HNetConfig.from_official_json("configs/official/hnet_2stage_XL.json")
    cfg.lr_multiplier = "auto"
    assert resolve_lr_multipliers(cfg) == pytest.approx([3.0332, 1.4298, 0.7149], abs=1e-4)
    cfg.lr_multiplier = None
    assert resolve_lr_multipliers(cfg) == [1.0, 1.0, 1.0]
    cfg.lr_multiplier = [2.0, 1.5, 1.0]
    assert resolve_lr_multipliers(cfg) == [2.0, 1.5, 1.0]
    cfg.lr_multiplier = [2.0, 1.0]
    with pytest.raises(AssertionError):
        resolve_lr_multipliers(cfg)
    assert resolve_lr_multipliers(None) == [1.0]


class RMSNormLike(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(10, 4)
        self.enc = nn.Linear(4, 4)
        self.enc_norm = RMSNormLike(4)
        self.main = nn.Linear(4, 4, bias=False)
        self.A_log = nn.Parameter(torch.zeros(2, 2))
        self.A_log._no_weight_decay = True
        self.lm_head = nn.Linear(4, 10, bias=False)

    def stage_of_param(self, name):
        return 1 if name.startswith("main") else 0


def _groups_by_name(groups):
    return {n: (g["stage"], g["weight_decay"], g["lr_mult"]) for g in groups for n in g["names"]}


def test_param_groups_weight_decay_and_stages():
    m = Tiny()
    g = _groups_by_name(param_groups(m, weight_decay=0.1, lr_multipliers=[2.0, 0.5]))
    assert g["embeddings.weight"] == (0, 0.1, 2.0)
    assert g["enc.weight"] == (0, 0.1, 2.0)
    assert g["enc.bias"] == (0, 0.0, 2.0)
    assert g["enc_norm.weight"] == (0, 0.0, 2.0)
    assert g["A_log"] == (0, 0.0, 2.0)
    assert g["main.weight"] == (1, 0.1, 0.5)
    assert g["lm_head.weight"] == (0, 0.1, 2.0)
    g2 = _groups_by_name(param_groups(m, weight_decay=0.1, wd_embeddings=False))
    assert g2["embeddings.weight"][1] == 0.0 and g2["lm_head.weight"][1] == 0.1
    # every param exactly once
    n = sum(len(x["params"]) for x in param_groups(m, 0.1))
    assert n == len(list(m.parameters()))


def test_build_adamw_and_set_lr():
    assert "adamw" in names("optimizer") and {"wsd", "constant", "cosine"} <= set(names("schedule"))
    m = Tiny()
    opt = build_optimizer(m, None, {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.1,
                                    "lr_multiplier_override": [2.0, 0.5]})
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["betas"] == (0.9, 0.95)
    lr = set_lr(opt, 0.5)
    for gr in opt.param_groups:
        assert gr["lr"] == pytest.approx(1e-3 * gr["lr_mult"] * 0.5)
    assert lr == pytest.approx(5e-4)
    loss = m.lm_head(m.main(m.enc(m.embeddings(torch.tensor([1, 2]))))).sum()
    loss.backward()
    opt.step()
