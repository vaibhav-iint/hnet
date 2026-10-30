"""Product-key memory in the main network (HNetConfig.main_memory) and in the BPE Transformer (DenseConfig.memory):
hnet/model/memory.py, evolve g5-output-01 and its BPE control g5-output-bpe-mem. Default path unchanged, zero values
== the model without the memory bit for bit, initialization of every other module unchanged, exact product-key
top-k, gradient paths (CLAUDE.md rules 1-2), parameter / FLOP accounting. Causality and packing: tests/
test_invariances.py (variant ``end_mlp_quantile_suffix_match_trie_topk_memory``) and the dense tests below."""

import pytest
import torch

from hnet.config import DenseConfig, model_config_from_yaml
from hnet.flops import count_params, model_flops
from hnet.model import build_model
from hnet.model.memory import ProductKeyMemory

MEM = {"n_keys": 8, "k_dim": 8, "topk": 4, "layer": 2}
TRIE = {"rows": 4099, "dim": 16, "n_hash": 2, "max_len": 8, "topk": 8}
MATCH = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256,
          "suffix_lens": [2, 3, 4, 6, 8, 12, 16]}]


def _hcfg(mem=None, stack=False):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    if stack:  # the g4-combo-02 stack: match model + suffix chunk memory + fast trie + read-back
        cfg.match_model = dict(MATCH)
        cfg.trie_head = dict(TRIE)
        cfg.dc.chunk_vocab = [dict(v) for v in VOCAB]
        cfg.dc.dechunk_slots = 2
    if mem is not None:
        cfg.main_memory = dict(mem)
    return cfg


def _dcfg(mem=None, match=False):
    return DenseConfig(vocab_size=300, d_model=32, n_layers=3, d_intermediate=64, num_heads=2,
                       match_model={"orders": [1, 2, 4], "dec_layer": 2} if match else {},
                       memory=dict(mem) if mem is not None else {})


def _build(cfg, seed=0):
    torch.manual_seed(seed)
    m = build_model(cfg)
    m.init_weights()
    return m


def _randomize(mem, std=0.5):
    torch.nn.init.normal_(mem.values.weight, std=std)


def test_default_configs_have_no_memory_and_unchanged_state_dicts():
    h = _build(_hcfg())
    assert h.backbone.stages()[-1].memory is None
    assert not any("memory" in k for k in h.state_dict())
    d = _build(_dcfg())
    assert d.memory is None and not any("memory" in k for k in d.state_dict())


@pytest.mark.parametrize("stack", [False, True])
def test_init_of_every_other_module_is_unchanged(stack):
    a = _build(_hcfg(stack=stack))
    b = _build(_hcfg(MEM, stack=stack))
    sb = b.state_dict()
    for k, v in a.state_dict().items():
        assert torch.equal(v, sb[k]), k
    assert sorted(set(sb) - set(a.state_dict())) == sorted(f"backbone.main_network.memory.{n}" for n in
                                                          ["norm.weight", "query.weight", "keys", "values.weight",
                                                           "x_mean"])


@pytest.mark.parametrize("stack", [False, True])
def test_zero_values_leave_the_hnet_unchanged_bit_for_bit(stack):
    a = _build(_hcfg(stack=stack))
    b = _build(_hcfg(MEM, stack=stack))
    ids = torch.randint(0, 256, (2, 200))
    for mode in ("train", "eval"):
        getattr(a, mode)(), getattr(b, mode)()
        torch.manual_seed(1)
        oa = a(ids, labels=ids)
        torch.manual_seed(1)
        ob = b(ids, labels=ids)
        assert torch.equal(oa.logits, ob.logits)
        assert torch.equal(oa.loss, ob.loss) and torch.equal(oa.aux_loss, ob.aux_loss)


@pytest.mark.parametrize("match", [False, True])
def test_zero_values_leave_the_dense_model_unchanged_bit_for_bit(match):
    a = _build(_dcfg(match=match))
    b = _build(_dcfg(MEM, match=match))
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k]), k
    ids = torch.randint(0, 300, (2, 64))
    oa, ob = a(ids, labels=ids), b(ids, labels=ids)
    assert torch.equal(oa.logits, ob.logits) and torch.equal(oa.loss, ob.loss)
    assert "memory/slots_used" in ob.stats


@pytest.mark.parametrize("layer", [1, 2, 3])
def test_memory_reads_the_stream_at_its_layer(layer):
    """With random values the output changes, and a memory after the last layer (layer = n_layers) works."""
    a = _build(_dcfg())
    b = _build(_dcfg({**MEM, "layer": layer}))
    _randomize(b.memory)
    ids = torch.randint(0, 300, (2, 64))
    assert not torch.allclose(a(ids).logits, b(ids).logits)


def test_layer_out_of_range_is_rejected():
    with pytest.raises(AssertionError):
        _build(_dcfg({**MEM, "layer": 4}))
    m = ProductKeyMemory(16, n_keys=4, k_dim=4, topk=2, layer=0)
    with pytest.raises(AssertionError):
        from hnet.model.isotropic import Isotropic

        iso = Isotropic("T2", 16, 32, {"t": "attn"}, {"attn": {"num_heads": 2}})
        from hnet.model.ctx import SeqCtx

        iso(torch.randn(10, 16), SeqCtx.from_cu_seqlens(torch.tensor([0, 10], dtype=torch.int32)), memory=(0, m))


def test_product_key_topk_is_the_exact_topk_over_all_slots():
    torch.manual_seed(0)
    m = ProductKeyMemory(24, n_keys=16, k_dim=8, topk=5).eval()
    x = torch.randn(50, 24)
    idx, w = m.lookup(x)
    q = m.query(m.norm(x, out_dtype=torch.float32)).view(-1, 2, 8)
    full = (q[:, 0] @ m.keys[0].T)[:, :, None] + (q[:, 1] @ m.keys[1].T)[:, None, :]  # [N, n, n]
    top, ref = full.reshape(50, -1).topk(5, dim=-1)
    assert torch.equal(idx, ref)
    assert torch.allclose(w, torch.softmax(top, -1), atol=1e-6)


def test_output_is_the_softmax_weighted_sum_of_values():
    torch.manual_seed(0)
    m = ProductKeyMemory(12, n_keys=6, k_dim=4, topk=3).eval()
    _randomize(m)
    x = torch.randn(7, 12)
    idx, w = m.lookup(x)
    ref = (w[..., None] * m.values.weight[idx]).sum(1)
    assert torch.allclose(m(x), ref, atol=1e-6)


def test_gradient_paths():
    """Rule 1: at init (values 0) the LM loss reaches the values only; once values != 0 it reaches the query, the
    sub-keys and the norm. The memory reaches everything upstream of it (the main network's lower layers, the chunk
    vectors, the encoder) once values != 0."""
    m = _build(_hcfg(MEM))
    mem = m.backbone.stages()[-1].memory
    ids = torch.randint(0, 256, (2, 160))
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, [mem.values.weight, mem.query.weight, mem.keys, mem.norm.weight],
                            allow_unused=True)
    assert g[0] is not None and g[0].abs().sum() > 0
    for x in g[1:]:
        assert x is None or x.abs().sum() == 0
    _randomize(mem, std=0.1)
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, [mem.values.weight, mem.query.weight, mem.keys, mem.norm.weight])
    assert all(x.abs().sum() > 0 for x in g)
    # only the read slots get value gradient
    idx = set()
    hook = {}
    mem.register_forward_hook(lambda mod, i, o: hook.__setitem__("x", i[0]))
    out = m(ids, labels=ids)
    with torch.no_grad():
        slots, _ = mem.lookup(hook["x"].reshape(-1, hook["x"].shape[-1]))
    idx = torch.unique(slots)
    gv = torch.autograd.grad(out.loss, mem.values.weight)[0]
    nz = torch.nonzero(gv.abs().sum(-1)).squeeze(1)
    assert set(nz.tolist()) <= set(idx.tolist())


def test_params_and_flops_match_the_built_models():
    for cfg in (_hcfg(MEM), _hcfg(MEM, stack=True), _dcfg(MEM), _dcfg(MEM, match=True)):
        m = build_model(cfg)
        assert count_params(cfg) == sum(p.numel() for p in m.parameters())
    h0, h1 = model_flops(_hcfg(), seq_len=2048), model_flops(_hcfg(MEM), seq_len=2048)
    comp = h1["per_stage"][-1]["components"]["main_memory"]
    d, n, kd, k = 256, MEM["n_keys"], MEM["k_dim"], MEM["topk"]
    fpp = 2 * d * 2 * kd + 2 * 2 * n * kd + 2 * k * d + k * k + 3 * k + d  # + d: the query centering
    bpp = h1["per_stage"][-1]["bytes_per_position"]
    assert comp["params"] == n * n * d + 2 * n * kd + 2 * kd * d + d
    assert abs(comp["flops_per_byte"] - 3 * fpp / bpp) < 1e-6  # training: forward + 2x backward
    assert abs(h1["flops_per_byte"] - h0["flops_per_byte"] - 3 * fpp / bpp) < 1e-3
    d0, d1 = model_flops(_dcfg(), seq_len=64), model_flops(_dcfg(MEM), seq_len=64)
    assert d1["flops_per_token"] > d0["flops_per_token"]


def test_candidate_configs_fit_the_budget():
    bpe = count_params(model_config_from_yaml("configs/ladder/models/bpe_L03.yaml"))
    ref = model_flops(model_config_from_yaml("configs/ladder/models/hnet_L03.yaml"), seq_len=8512)["flops_per_byte"]
    for name in ("g5-output-01", "g5-output-02"):
        cfg = model_config_from_yaml(f"configs/evolve/models/{name}.yaml")
        assert count_params(cfg) <= bpe, name
        assert model_flops(cfg, seq_len=8512)["flops_per_byte"] <= 1.02 * ref, name


def test_memory_stats_are_logged():
    m = _build(_hcfg(MEM))
    out = m(torch.randint(0, 256, (2, 160)), labels=torch.randint(0, 256, (2, 160)))
    for k in ("memory/slots_used", "memory/top_w", "memory/out_rms", "memory/in_rms"):
        assert k in out.stats
    assert 0 < float(out.stats["memory/slots_used"]) <= 1
    assert float(out.stats["memory/out_rms"]) == 0.0  # zero values at init


def _dense_trained():
    m = _build(_dcfg(MEM, match=True))
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 300, (2, 64))
    for _ in range(3):
        o = m(ids, labels=ids)
        opt.zero_grad()
        o.loss.backward()
        opt.step()
    assert m.memory.values.weight.abs().sum() > 0
    return m.eval()


def test_dense_memory_is_causal_and_packing_invariant():
    m = _dense_trained()
    ids = torch.randint(0, 300, (1, 80))
    alt = ids.clone()
    alt[0, 50:] = (alt[0, 50:] + 7) % 300
    with torch.no_grad():
        assert torch.allclose(m(ids).logits[0, :50], m(alt).logits[0, :50], atol=1e-5)
        a, b = torch.randint(0, 300, (1, 40)), torch.randint(0, 300, (1, 56))
        la, lb = m(a).logits[0], m(b).logits[0]
        lp = m(torch.cat([a[0], b[0]]), cu_seqlens=torch.tensor([0, 40, 96], dtype=torch.int32), max_seqlen=56).logits
    assert torch.allclose(lp[:40], la, atol=1e-5) and torch.allclose(lp[40:], lb, atol=1e-5)


def test_query_centering_reads_earlier_steps_only():
    """The running mean is updated after the read, in training mode only: a row's output does not depend on the other
    rows of its batch (no within-batch coupling), eval leaves the buffer alone, center 0 has no buffer at all."""
    torch.manual_seed(0)
    m = ProductKeyMemory(16, n_keys=6, k_dim=4, topk=3, center=0.9)
    _randomize(m)
    m.x_mean.normal_()
    x = torch.randn(10, 16)
    m.train()
    before = m.x_mean.clone()
    full = m(x)
    assert not torch.equal(m.x_mean, before)  # updated
    m.x_mean.copy_(before)
    part = m(x[:4])
    assert torch.allclose(full[:4], part, atol=1e-6)
    m.eval()
    mu = m.x_mean.clone()
    m(x)
    assert torch.equal(m.x_mean, mu)
    off = ProductKeyMemory(16, n_keys=6, k_dim=4, topk=3, center=0.0)
    assert "x_mean" not in off.state_dict()
    with pytest.raises(AssertionError):
        ProductKeyMemory(16, n_keys=6, k_dim=4, topk=3, center=1.0)


def test_centering_spreads_the_reads_of_a_stream_with_a_common_direction():
    """Rule 5 (ugly distributions): inputs sharing one large direction address few slots; the running mean removes it."""
    torch.manual_seed(0)
    common = 5 * torch.randn(1, 64)
    x = common + torch.randn(2000, 64)
    m = ProductKeyMemory(64, n_keys=16, k_dim=8, topk=4, center=0.5)
    with torch.no_grad():
        raw = torch.unique(m.lookup(x)[0]).numel()
        m.train()
        for _ in range(20):
            m.lookup(x)
        m.eval()
        cen = torch.unique(m.lookup(x)[0]).numel()
    assert cen > 1.5 * raw, (raw, cen)
