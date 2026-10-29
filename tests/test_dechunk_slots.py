"""dc.dechunk_slots = K (offset-resolved read-back; candidate g1-arch-02).

    out_t = W_{min(k_t, K-1)} zbar_{j(t)},   k_t = t - (outer position of the inner position j(t) that byte t reads)
"""

import pytest
import torch

import test_invariances as inv
from hnet.config import model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.chunking.dechunk import plug_back_index

VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}]


def _cfg(**kw):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router, cfg.dc.decision, cfg.dc.chunk_vocab = "end_mlp", "quantile", VOCAB
    for k, v in kw.items():
        setattr(cfg.dc, k, v)
    return cfg


def _model(seed=0, **kw):
    torch.manual_seed(seed)
    m = build_model(_cfg(**kw))
    m.init_weights()
    return m


def _fwd_bwd(m, ids):
    out = m(ids, labels=ids)
    (out.loss + out.aux_loss).backward()
    return out.logits.detach(), {n: p.grad for n, p in m.named_parameters()}


def test_default_is_off_and_adds_nothing():
    """Default: no slot parameters and the forward / backward pass is the same as with dechunk_slots=0 given
    explicitly (bit-identical logits and gradients)."""
    assert _cfg().dc.dechunk_slots == 0
    ids = torch.randint(0, 256, (2, 300), generator=torch.Generator().manual_seed(1))
    m0, m1 = _model().train(), _model(dechunk_slots=0).train()
    assert not any("slot" in n for n, _ in m0.named_parameters())
    assert sorted(m0.state_dict()) == sorted(m1.state_dict())
    (l0, g0), (l1, g1) = _fwd_bwd(m0, ids), _fwd_bwd(m1, ids)
    assert torch.equal(l0, l1)
    assert all((g0[n] is None and g1[n] is None) or torch.equal(g0[n], g1[n]) for n in g0)


@pytest.mark.parametrize("dechunker", ["ema", "repeat"])
def test_identity_init_equals_the_plain_dechunker(dechunker):
    """At init every W_k is the identity: the model with K slots computes exactly what the K = 0 model computes (same
    RNG stream, the slot parameters consume none), and the slot parameters receive a gradient."""
    ids = torch.randint(0, 256, (2, 300), generator=torch.Generator().manual_seed(1))
    m0 = _model(dechunker=dechunker).train()
    m4 = _model(dechunker=dechunker, dechunk_slots=4).train()
    (l0, g0), (l4, g4) = _fwd_bwd(m0, ids), _fwd_bwd(m4, ids)
    assert torch.equal(l0, l4)
    for n, g in g0.items():
        assert (g is None and g4[n] is None) or torch.allclose(g, g4[n], rtol=1e-4, atol=1e-7), n
    assert float(g4["backbone.slot_proj"].abs().sum()) > 0


def test_each_byte_reads_the_slot_of_its_offset():
    """With distinct W_k, byte t's hierarchy input is W_{min(k_t, K-1)} zbar_{j(t)}, k_t = bytes since the boundary it
    reads (end semantics: the boundary byte itself has k = 0)."""
    K = 3
    m = _model(dechunk_slots=K).eval()
    D = m.backbone.d_model
    with torch.no_grad():
        for k in range(K):
            m.backbone.slot_proj[k * D:(k + 1) * D] = torch.eye(D) * (k + 2.0)
    seen = {}
    hook = m.backbone.decoder.register_forward_pre_hook(lambda mod, args: seen.__setitem__("h", args[0].detach()))
    ids = torch.randint(0, 256, (1, 400))
    out = m(ids)
    hook.remove()
    r = out.bpred[0]
    # the residual path is zero-init (weight) but has a live bias: subtract it to isolate the hierarchy's part
    hier = seen["h"].float() - m.backbone.residual_proj.bias.detach().float()
    plug = plug_back_index(r)
    sel = torch.nonzero(r.boundary_mask).squeeze(1)
    k = torch.arange(400) - sel[plug]
    assert int(k.max()) >= K  # the test text has chunks longer than K
    base = hier[sel[plug]] / 2.0  # the boundary byte itself reads slot 0 (scale 2)
    expect_scale = (k.clamp(max=K - 1) + 2.0)[:, None]
    assert torch.allclose(hier, base * expect_scale, rtol=1e-4, atol=1e-5)
    assert 0.0 <= float(r.extra["slot_last_frac"]) <= 1.0 and float(r.extra["slot_div"]) > 0


def test_gradient_paths():
    """Rule 1: byte CE reaches the slot maps, the main network, the router (through the EMA's P and the STE) and the
    encoder; the next-chunk CE (computed before dechunking) does not reach the slots."""
    ids = torch.randint(0, 256, (2, 300))
    m = _model(dechunk_slots=4).train()
    out = m(ids, labels=ids)
    named = dict(m.named_parameters())
    groups = {
        "slots": [named["backbone.slot_proj"]],
        "router": [p for n, p in named.items() if "routing_module" in n],
        "main": [p for n, p in named.items() if n.startswith("backbone.main_network.")],
        "encoder": [p for n, p in named.items() if n.startswith("backbone.encoder.")],
    }

    def norm(loss, ps):
        gs = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
        return sum(float(g.abs().sum()) for g in gs if g is not None)

    assert all(norm(out.loss, ps) > 0 for ps in groups.values())
    r = out.bpred[0]
    nxt = m.backbone.chunk_vocab.pred_loss(r.inner_out, r.vocab_tgt)
    assert norm(nxt, groups["slots"]) == 0.0


def test_flops_and_params_counted():
    cfg = _cfg(dechunk_slots=4)
    m = build_model(cfg)
    d = cfg.d_model[0]
    assert sum(p.numel() for p in m.parameters()) == count_params(cfg) == count_params(_cfg()) + 4 * d * d
    f0 = model_flops(_cfg(), seq_len=2048)["flops_per_byte"]
    f4 = model_flops(cfg, seq_len=2048)["flops_per_byte"]
    assert f4 - f0 == pytest.approx(3 * 2 * 4 * d * d / cfg.dc.target_ratios[0], rel=1e-9)  # training = 3x forward
    l03 = model_config_from_yaml("configs/evolve/models/g1-arch-02.yaml")
    ref = model_config_from_yaml("configs/ladder/models/hnet_L03.yaml")
    assert l03.dc.dechunk_slots == 4
    assert count_params(l03) <= 20_023_040  # bpe_L03
    rel = model_flops(l03, seq_len=8512)["flops_per_byte"] / model_flops(ref, seq_len=8512)["flops_per_byte"] - 1
    assert 0 < rel < 0.02


@pytest.mark.parametrize("ngram", [False, True])
def test_invariances_with_slots(monkeypatch, ngram):
    monkeypatch.setitem(inv.VARIANTS, "slots", {**inv.VARIANTS["end_mlp_quantile_vocab"], "dechunk_slots": 4})
    inv.test_future_bytes_do_not_change_earlier_logits("slots", ngram)
    inv.test_packed_segments_equal_separate_sequences("slots", ngram)
