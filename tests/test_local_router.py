"""Context-free (last-k-bytes) router: hnet.model.routing.local."""

import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.local import PAD, LocalMLPRouter


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def test_window_is_the_last_k_bytes_with_pad_before_the_segment_start():
    r = LocalMLPRouter(16, target_ratio=3.0, k=3)
    ids = torch.tensor([0, 10, 11, 12, 0, 20, 21])
    w = r.window(ids, _ctx([4, 3]))
    assert w.tolist() == [[PAD, PAD, 0], [PAD, 0, 10], [0, 10, 11], [10, 11, 12], [PAD, PAD, 0], [PAD, 0, 20], [0, 20, 21]]


def test_same_local_string_gives_the_same_score_whatever_the_context():
    torch.manual_seed(0)
    r = LocalMLPRouter(16, target_ratio=3.0, k=4)
    a = torch.tensor([0] + list(b"the big cat sat"))
    b = torch.tensor([0] + list(b"my old cat ran"))
    ca, cb = _ctx([len(a)]), _ctx([len(b)])
    h = torch.zeros(len(a), 16)
    la = r(h, ca, input_ids=a).logit
    lb = r(h[: len(b)], cb, input_ids=b).logit
    # " cat" ends at index 11 in a and 10 in b: identical 4-byte windows -> identical scores
    assert torch.allclose(la[11], lb[10])
    # the score does not depend on h at all
    la2 = r(torch.randn(len(a), 16), ca, input_ids=a).logit
    assert torch.allclose(la, la2)


def test_model_trains_with_local_router_and_rank_decisions():
    cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
    cfg.dc.router = "local_mlp"
    cfg.dc.router_kwargs = {"k": 6}
    cfg.dc.decision = "quantile"
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    for s in (0, 1):
        assert abs(float(out.stats[f"compression/s{s}"]) - 3.0) < 0.15
    g = [p.grad for n, p in m.named_parameters() if "routing_module" in n and p.grad is not None]
    assert g and sum(float(x.abs().sum()) for x in g) > 0  # the router learns through the STE confidence
