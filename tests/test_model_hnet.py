"""Whole-model tests: causality, packing, parity, STE gradients, API (CPU, tiny)."""

import pytest
import torch

from hnet import registry
from hnet.config import IGNORE_INDEX, DenseConfig, HNetConfig
from hnet.model import build_model
from hnet.model import lm as lm_mod

torch.set_num_threads(1)

ROUTERS = ["cosine_lookahead", "end_cosine_shift", "end_linear", "end_mlp", "fixed_pool", "space_like"]


def tiny_cfg(router="cosine_lookahead", dechunker="ema", router_kwargs=None, **kw) -> HNetConfig:
    d = dict(
        arch_layout=["m1t1", ["T1m1", ["T2"], "m1T1"], "t1m1"],
        d_model=[32, 48, 64],
        d_intermediate=[0, 96, 128],
        ssm=dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8),
        attn=dict(num_heads=2, num_kv_heads=[2, 2, 1], rotary_emb_dim=[8, 12, 16], window_size=[5, 5, -1],
                  nope_every=[0, 0, 2]),
        dc=dict(target_ratios=[2.0, 2.0], router=router, router_kwargs=router_kwargs or {}, dechunker=dechunker),
    )
    d.update(kw)
    return HNetConfig.from_dict(d)


def make(cfg, seed=0):
    torch.manual_seed(seed)
    m = build_model(cfg)
    m.init_weights()
    return m.eval()


def text_ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    words = [b"the ", b"cat ", b"sat, ", b"on ", b"a ", b"mat. ", b"\xc3\xa9t\xc3\xa9 ", b"x1 "]
    out = b"".join(words[i] for i in torch.randint(0, len(words), (n,), generator=g).tolist())
    return torch.tensor(list(out[:n]))


@pytest.mark.parametrize("semantics", ["start", "end"])
@pytest.mark.parametrize("router", ROUTERS)
def test_causality(router, semantics):
    """Changing bytes >= u never changes logits < u (dynamic boundaries included)."""
    kw = {"semantics": semantics} if router in ("fixed_pool", "space_like") else {}
    if router not in ("fixed_pool", "space_like") and registry.get("router", router).boundary_semantics != semantics:
        pytest.skip("learned routers have fixed semantics")
    m = make(tiny_cfg(router, router_kwargs=kw))
    ids = text_ids(96, 1)
    with torch.no_grad():
        base = m(ids).logits
        for u in (5, 40, 77):
            pert = ids.clone()
            pert[u:] = text_ids(96 - u, 100 + u)
            out = m(pert)
            torch.testing.assert_close(out.logits[:u], base[:u], atol=1e-5, rtol=1e-5)
            assert torch.equal(out.bpred[0].boundary_mask[:u], m(ids).bpred[0].boundary_mask[:u])


def test_causality_external_both_semantics():
    for semantics in ("start", "end"):
        cfg = tiny_cfg(["external", "fixed_pool"], router_kwargs=[{"semantics": semantics}, {"semantics": semantics}])
        m = make(cfg)
        ids = text_ids(60, 2)
        b0 = torch.rand(60, generator=torch.Generator().manual_seed(0)) < 0.4
        pert, pb0 = ids.clone(), b0.clone()
        pert[30:] = 7
        pb0[30:] = ~pb0[30:]
        with torch.no_grad():
            base, out = m(ids, boundaries=[b0]), m(pert, boundaries=[pb0])
        assert out.bpred[0].semantics == semantics
        torch.testing.assert_close(out.logits[:30], base.logits[:30], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("router", ["cosine_lookahead", "end_mlp"])
def test_packed_equals_separate(router):
    m = make(tiny_cfg(router))
    docs = [text_ids(n, s) for n, s in ((37, 1), (5, 2), (58, 3))]
    ids = torch.cat(docs)
    cu = torch.tensor([0, 37, 42, 100], dtype=torch.int32)
    with torch.no_grad():
        packed = m(ids, cu_seqlens=cu, max_seqlen=58)
        sep = [m(d) for d in docs]
    torch.testing.assert_close(packed.logits, torch.cat([o.logits for o in sep]), atol=1e-5, rtol=1e-5)
    for s in range(2):
        assert torch.equal(packed.bpred[s].boundary_mask, torch.cat([o.bpred[s].boundary_mask for o in sep]))
    # [B, L] rows == packed segments of equal length.
    with torch.no_grad():
        batched = m(torch.stack([docs[0][:30], docs[2][:30]]))
        packed2 = m(torch.cat([docs[0][:30], docs[2][:30]]), cu_seqlens=torch.tensor([0, 30, 60]))
    torch.testing.assert_close(batched.logits.reshape(60, -1), packed2.logits)


def test_isotropic_dense_parity():
    """A 0-stage H-Net with attention-only layout == the llama dense model with the same weights."""
    dcfg = DenseConfig(vocab_size=256, d_model=32, n_layers=2, d_intermediate=64, num_heads=4, num_kv_heads=2,
                       rope_theta=500.0, nope_every=2)
    dense = make(dcfg)
    hcfg = HNetConfig.from_dict(dict(
        arch_layout=["T2"], d_model=[32], d_intermediate=[64],
        attn=dict(num_heads=4, num_kv_heads=2, rope_theta=500.0, nope_every=2), dc=dict(target_ratios=[]),
    ))
    hnet = make(hcfg)
    sd = {k.replace("backbone.", "backbone.main_network.", 1): v for k, v in dense.state_dict().items()}
    hnet.load_state_dict(sd, strict=True)
    ids = text_ids(40, 5)
    labels = torch.roll(ids, -1)
    labels[-1] = IGNORE_INDEX
    a, b = dense(ids, labels=labels), hnet(ids, labels=labels)
    torch.testing.assert_close(a.logits, b.logits)
    torch.testing.assert_close(a.loss, b.loss)
    assert b.bpred == [] and float(b.aux_loss) == 0.0 and float(a.aux_loss) == 0.0


@pytest.mark.parametrize("router", ["cosine_lookahead", "end_linear", "end_mlp"])
def test_ste_gradient_reaches_router(router):
    m = make(tiny_cfg(router)).train()
    ids = text_ids(64, 4)
    labels = torch.roll(ids, -1)
    out = m(ids, labels=labels)
    out.loss.backward()  # CE only: router gets gradients through STE(selected_probs) and the EMA
    for st in m.backbone.stages()[:-1]:
        grads = [p.grad for p in st.routing_module.parameters()]
        assert all(g is not None and g.abs().sum() > 0 for g in grads), router
    # Without STE and with "repeat" dechunking, CE gives the router no gradient.
    m2 = make(_no_ste_cfg(router)).train()
    m2(ids, labels=labels).loss.backward()
    assert all(p.grad is None for p in m2.backbone.routing_module.parameters())


def _no_ste_cfg(router):
    cfg = tiny_cfg(router, dechunker="repeat")
    cfg.dc.ste = False
    return cfg


def test_outputs_stats_and_losses():
    m = make(tiny_cfg())
    ids = text_ids(80, 6)
    labels = torch.roll(ids, -1)
    labels[[10, 79]] = IGNORE_INDEX
    out = m(ids, labels=labels)
    assert out.logits.shape == (80, 256) and int(out.n_targets) == 78
    ref = torch.nn.functional.cross_entropy(out.logits.float(), labels, ignore_index=IGNORE_INDEX)
    torch.testing.assert_close(out.loss, ref)
    no_logits = m(ids, labels=labels, return_logits=False)
    assert no_logits.logits is None
    torch.testing.assert_close(no_logits.ce_sum, out.ce_sum)
    for k in ("aux/s0", "F/s0", "G/s0", "compression/s0", "positions/s1", "aux/s1", "compression/s1", "positions/s2"):
        assert k in out.stats and out.stats[k].ndim == 0
    assert int(out.stats["positions/s1"]) == int(out.bpred[0].boundary_mask.sum())
    torch.testing.assert_close(out.aux_loss, 0.03 * (out.stats["aux/s0"] + out.stats["aux/s1"]))
    assert len(out.bpred) == 2 and out.bpred[1].boundary_mask.numel() == int(out.stats["positions/s1"])


def test_chunked_ce(monkeypatch):
    m = make(tiny_cfg())
    ids, labels = text_ids(50, 7), text_ids(50, 8)
    full = m(ids, labels=labels, return_logits=False)
    full.ce_sum.backward()
    g_full = m.lm_head.weight.grad.clone()
    m.zero_grad()
    monkeypatch.setattr(lm_mod, "CE_CHUNK_ELEMENTS", 256 * 7)  # 7 rows per chunk
    chunked = m(ids, labels=labels, return_logits=False)
    chunked.ce_sum.backward()
    torch.testing.assert_close(chunked.ce_sum, full.ce_sum)
    torch.testing.assert_close(m.lm_head.weight.grad, g_full)


def test_init_stage_map_blocks():
    cfg = tiny_cfg()
    m = make(cfg)
    assert m.stage_of_param("embeddings.weight") == 0 and m.stage_of_param("module.lm_head.weight") == 0
    assert m.stage_of_param("backbone.encoder.layers.0.mixer.in_proj.weight") == 0
    assert m.stage_of_param("backbone.routing_module.q_proj_layer.weight") == 0
    assert m.stage_of_param("backbone.main_network.pad_dimension") == 1
    assert m.stage_of_param("backbone.main_network.decoder.rmsnorm.weight") == 1
    assert m.stage_of_param("_orig_mod.backbone.main_network.main_network.main_network.layers.1.mlp.fc2.weight") == 2
    assert {m.stage_of_param(n) for n, _ in m.named_parameters()} == {0, 1, 2}
    assert len(m.blocks()) == 2 + 2 + 2 + 2 + 2
    # Depth-scaled init: innermost residual count = heights of enclosing enc/dec + main height.
    torch.manual_seed(0)
    big = build_model(tiny_cfg(d_model=[64, 64, 512], d_intermediate=[0, 128, 1024]))
    big.init_weights()
    n_res = (2 + 2) + (3 + 3) + 4
    fc2 = big.backbone.main_network.main_network.main_network.layers[0].mlp.fc2.weight
    fc1 = big.backbone.main_network.main_network.main_network.layers[0].mlp.fc1.weight
    assert abs(fc2.std().item() / (0.02 / n_res**0.5) - 1) < 0.05 and abs(fc1.std().item() / 0.02 - 1) < 0.05
    assert torch.equal(big.backbone.routing_module.q_proj_layer.weight, torch.eye(64))
    rp = big.backbone.residual_proj.weight
    assert rp.abs().sum() == 0 and rp.dtype == torch.float32
    assert abs(big.embeddings.weight.std().item() - 1.0) < 0.05


def test_bf16_autocast_runs():
    m = make(tiny_cfg())
    ids = text_ids(64, 9)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = m(ids, labels=torch.roll(ids, -1))
    (out.loss + out.aux_loss).backward()
    assert torch.isfinite(out.loss) and out.bpred[0].boundary_prob.dtype == torch.float32
    assert m.backbone.residual_proj.weight.grad is not None


def test_dense_api():
    m = make(DenseConfig(vocab_size=300, d_model=32, n_layers=2, d_intermediate=64, num_heads=4,
                         tie_embeddings=True))
    ids = torch.randint(0, 300, (2, 16))
    out = m(ids, labels=ids)
    assert out.logits.shape == (2, 16, 300) and m.lm_head.weight is m.embeddings.weight
    assert m.stage_of_param("embeddings.weight") == 0 and len(m.blocks()) == 2


@pytest.mark.parametrize("router", sorted(set(ROUTERS) | {"external"}))
def test_builds_on_meta_device(router):
    """FLOPs/param counting and checkpoint loading build models on the meta device."""
    with torch.device("meta"):
        m = build_model(tiny_cfg(router))
    assert all(p.is_meta for p in m.parameters())
