"""dc.router_surprise_weight: a prior-free tokenizer teacher from the model's own next-byte entropy rise.

The teacher is training-only: the forward pass (logits, byte loss) is bit-identical with the flag on or off, and the
default config leaves the auxiliary loss untouched. The term is scale-free under rank decisions, reaches only the
router and what feeds it (never the LM head, decoder or main network, because the entropy is detached), and pushes
the router's scores up at the positions where the next byte gets hard to predict.
"""

import pytest
import torch

from hnet.config import DCConfig, model_config_from_yaml
from hnet.model import build_model


def _cfg(**dc):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cfg.dc.chunk_vocab = [{"buckets": 512, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 64}]
    cfg.ngram_embed = {"ns": [3, 4], "buckets": 257}
    for k, v in dc.items():
        setattr(cfg.dc, k, v)
    return cfg


def _model(seed=0, **dc):
    torch.manual_seed(seed)
    m = build_model(_cfg(**dc))
    m.init_weights()
    return m


def _ids(seed=0, B=2, L=300):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (B, L), generator=g)


def test_default_is_off():
    assert DCConfig().router_surprise_weight == 0.0


@pytest.mark.parametrize("train", [True, False])
def test_forward_pass_bit_identical_with_flag(train):
    ids = _ids()
    outs = []
    for w in (0.0, 1.0):
        m = _model(router_surprise_weight=w).train(train)
        m.set_progress(0.0)
        torch.manual_seed(123)
        o = m(ids, labels=ids)
        outs.append((o, torch.get_rng_state()))
    (a, rng_a), (b, rng_b) = outs
    assert torch.equal(a.logits, b.logits) and torch.equal(a.loss, b.loss)
    assert torch.equal(rng_a, rng_b)  # the teacher draws no random numbers
    if train:
        extra = float((b.aux_loss - a.aux_loss).detach())
        assert abs(extra - float(b.stats["router_surprise/s0"])) < 1e-5
        assert "router_surprise/s0" not in a.stats
    else:  # training-only
        assert torch.equal(a.aux_loss, b.aux_loss) and "router_surprise/s0" not in b.stats


def test_curriculum_decays_to_zero():
    ids = _ids()
    m = _model(router_surprise_weight=1.0, router_surprise_frac=0.4).train()
    base = _model(router_surprise_weight=0.0).train()
    vals = {}
    for prog in (0.0, 0.2, 0.5):
        m.set_progress(prog)
        base.set_progress(prog)
        torch.manual_seed(1)  # the next-chunk loss samples negatives
        o = m(ids, labels=ids)
        torch.manual_seed(1)
        o0 = base(ids, labels=ids)
        vals[prog] = (float((o.aux_loss - o0.aux_loss).detach()), float(o.stats.get("router_surprise/s0", 0.0)))
    assert abs(vals[0.0][0] - vals[0.0][1]) < 1e-5
    assert abs(vals[0.2][0] - 0.5 * vals[0.2][1]) < 1e-5
    assert vals[0.5] == (0.0, 0.0)
    m2 = _model(router_surprise_weight=1.0, router_surprise_frac=0.0).train()  # frac 0: constant weight
    m2.set_progress(0.9)
    assert "router_surprise/s0" in m2(ids, labels=ids).stats


def test_teacher_targets_are_the_top_entropy_rises(monkeypatch):
    m = _model(router_surprise_weight=1.0).train()
    m.set_progress(0.0)
    ids = _ids(L=120)
    T = ids.numel()
    H = torch.zeros(T)
    rises = [10, 50, 130, 200]  # entropy jumps up at these positions (one per ~30 bytes)
    for t in rises:
        H[t:] += 1.0
    monkeypatch.setattr(m, "next_byte_entropy", lambda h: H)
    seen = {}
    orig = torch.nn.functional.binary_cross_entropy

    def spy(p, t, *a, **k):
        seen["tgt"] = t.clone()
        return orig(p, t, *a, **k)

    monkeypatch.setattr(torch.nn.functional, "binary_cross_entropy", spy)
    m(ids, labels=ids)
    free = torch.ones(T, dtype=torch.bool)
    free[0] = free[120] = False  # segment starts
    pos = torch.nonzero(free).squeeze(1)[seen["tgt"] > 0]
    for t in rises:
        assert t in pos.tolist()


def test_scale_free_under_rank_decisions():
    """Scaling the router's logits (fc2 weight and bias) leaves the standardized scores, hence the term, unchanged:
    the cheapest escape (shrink or blow up every score) does not exist."""
    ids = _ids()
    m = _model(router_surprise_weight=1.0).train()
    m.set_progress(0.0)
    a = float(m(ids, labels=ids).stats["router_surprise/s0"])
    fc2 = m.backbone.routing_module.fc2
    with torch.no_grad():
        fc2.weight.mul_(3.0)
        fc2.bias.mul_(3.0)
    m.backbone.decider.initialized.fill_(False)
    b = float(m(ids, labels=ids).stats["router_surprise/s0"])
    assert abs(a - b) < 1e-2 * abs(a)


def test_gradient_paths_router_and_encoder_only():
    ids = _ids()
    m = _model(router_surprise_weight=1.0).train()
    m.set_progress(0.0)
    captured = {}
    hook = m.backbone.register_forward_hook(lambda mod, args, res: captured.update(h=res[0]))  # returns None
    out = m(ids, labels=ids)
    hook.remove()
    from hnet.model.lm import pack_inputs

    _, ctx, _ = pack_inputs(ids, None, None)
    loss, _ = m._router_surprise_loss(captured["h"], out.bpred[0], ids.reshape(-1), ctx)

    def gnorm(prefix):
        ps = [p for n, p in m.named_parameters() if n.startswith(prefix)]
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        return sum(float(g.norm()) for g in gs if g is not None)

    assert gnorm("backbone.routing_module.") > 0
    assert gnorm("backbone.encoder.") > 0
    for silent in ("lm_head.", "backbone.decoder.", "backbone.main_network.", "backbone.residual_proj."):
        assert gnorm(silent) == 0.0, silent


def test_gradient_step_raises_scores_at_teacher_targets(monkeypatch):
    m = _model(router_surprise_weight=1.0).train()
    m.set_progress(0.0)
    ids = _ids(L=200)
    T = ids.numel()
    H = torch.zeros(T)
    tgt_pos = list(range(5, 200, 7)) + list(range(205, 400, 7))
    for t in tgt_pos:
        H[t] = 3.0  # a spike: rise at t, fall at t + 1
    monkeypatch.setattr(m, "next_byte_entropy", lambda h: H)
    out = m(ids, labels=ids)
    before = out.bpred[0].logit.detach().clone()
    router = [p for n, p in m.named_parameters() if "routing_module" in n]
    from hnet.model.lm import pack_inputs

    _, ctx, _ = pack_inputs(ids, None, None)
    loss, _ = m._router_surprise_loss(out.bpred[0].logit, out.bpred[0], ids.reshape(-1), ctx)
    grads = torch.autograd.grad(loss, router)
    with torch.no_grad():
        for p, g in zip(router, grads):
            p.sub_(0.5 * g)
    after = m(ids, labels=ids).bpred[0].logit.detach()
    on = torch.zeros(T, dtype=torch.bool)
    on[tgt_pos] = True
    d = after - before
    assert float(d[on].mean()) > float(d[~on].mean())
