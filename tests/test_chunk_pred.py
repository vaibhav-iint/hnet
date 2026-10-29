"""Chunk-level multi-byte prediction loss (HNetConfig.dc.chunk_pred_k / chunk_pred_weight)."""

import torch
import torch.nn.functional as F

from hnet.config import IGNORE_INDEX, model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model


def _cfg(k=2, w=0.5, router="fixed_pool", semantics="end"):
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = router
    cfg.dc.router_kwargs = {"semantics": semantics}
    cfg.dc.chunk_pred_k = k
    cfg.dc.chunk_pred_weight = w
    return cfg


def test_targets_by_hand_two_segments():
    torch.manual_seed(0)
    cfg = _cfg(k=3)
    m = build_model(cfg).eval()
    ids = torch.randint(0, 254, (2, 40))  # two segments of 40 bytes
    labels = torch.roll(ids, -1, dims=1)  # the trainer's convention: labels[t] = ids[t + 1]
    labels[:, -1] = IGNORE_INDEX
    out = m(ids, labels=labels)
    # recompute: innermost byte positions via the boundary masks, targets = the k bytes after each position within
    # its segment, logits from the head applied to the innermost main-network output
    flat = ids.reshape(-1)
    pos = torch.arange(80)
    for r in out.bpred:
        pos = pos[r.boundary_mask]
    z = out.bpred[-1].inner_out
    assert z.shape[0] == pos.numel()
    logits = m.chunk_pred_head(z).float().reshape(pos.numel(), 3, 256)
    tgt = torch.full((pos.numel(), 3), IGNORE_INDEX)
    for i, p in enumerate(pos.tolist()):
        seg_end = 40 if p < 40 else 80
        for j in range(3):
            q = p + 1 + j
            if q < seg_end:
                tgt[i, j] = flat[q]
    expect = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1), ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(out.stats["chunk_pred_ce"], expect)
    assert (tgt == IGNORE_INDEX).any()  # positions near a segment end lose some targets
    # the loss is part of aux_loss with its weight (ratio loss adds a constant under hard routers)
    assert out.aux_loss.item() > 0.5 * expect.item() - 1e-6


def test_chunk_loss_trains_the_main_network_and_is_counted():
    cfg = _cfg(k=4, w=0.3)
    torch.manual_seed(0)
    m = build_model(cfg)
    assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    base = model_flops(_cfg(k=0, w=0.0))["flops_per_byte"]
    extra = model_flops(cfg)["flops_per_byte"] / base - 1
    assert 0 < extra < 0.02, extra
    ids = torch.randint(0, 254, (2, 120))
    out = m(ids, labels=ids)
    out.aux_loss.backward()  # no LM loss: the only path to the main network is the chunk head
    g = sum(p.grad.abs().sum() for n, p in m.named_parameters()
            if n.startswith("backbone.main_network.main_network") and p.grad is not None)
    assert g > 0
    assert m.lm_head.weight.grad is None  # the LM head is untouched by the auxiliary loss


def test_works_with_learned_router_and_quantile():
    cfg = _cfg(k=2, w=0.3, router="end_mlp")
    cfg.dc.router_kwargs = {}
    cfg.dc.decision = "quantile"
    torch.manual_seed(0)
    m = build_model(cfg)
    ids = torch.randint(0, 254, (1, 200))
    out = m(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    assert torch.isfinite(out.stats["chunk_pred_ce"])
