"""Fast prefix-trie output head (HNetConfig.trie_head.topk > 0, hnet.model.trie_head.TrieHead.rescore): the trie's
keys and query, scored only for the top-k bytes of the base logits (generation 3, g3-mem-01 / g3-mem-02)."""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.flops import model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.trie_head import TrieHead

TRIE = {"rows": 4099, "dim": 16, "n_hash": 2, "max_len": 8}
MATCH = {"orders": [2, 4, 8], "enc": True, "dec": True, "gate": True}
SUFFIX_VOCAB = [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256,
                 "suffix_lens": [2, 3, 4, 6, 8, 12, 16]}]


def _cfg(trie=None, match=False, vocab=False):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    if trie is not None:
        cfg.trie_head = trie
    if match:
        cfg.match_model = dict(MATCH)
    if vocab:
        cfg.dc.chunk_vocab = [dict(v) for v in SUFFIX_VOCAB]
    return cfg


def _model(trie=None, seed=0, **kw):
    torch.manual_seed(seed)
    m = build_model(_cfg(trie, **kw))
    m.init_weights()
    return m


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def _randomize(m, std=0.3):
    for t in m.trie_head.tables:
        torch.nn.init.normal_(t.weight, std=std)


def test_default_trie_config_is_the_full_head():
    th = TrieHead(8, 256, rows=101, dim=4)
    assert th.topk == 0
    with pytest.raises(AssertionError):
        TrieHead(8, 256, rows=101, dim=4, topk=257)


@pytest.mark.parametrize("match", [False, True])
def test_zero_tables_leave_the_logits_of_the_model_without_the_head_unchanged(match):
    base = _model(match=match)
    m = _model({**TRIE, "topk": 8}, seed=1, match=match)
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("trie_head.") for k in missing)
    ids = torch.randint(0, 256, (2, 200))
    base.eval(), m.eval()
    with torch.no_grad():
        a, b = base(ids, labels=ids), m(ids, labels=ids)
    assert torch.equal(a.logits.float(), b.logits)
    assert torch.equal(a.loss, b.loss)


def test_topk_equal_to_the_vocabulary_is_the_full_head():
    full = _model(TRIE)
    fast = _model({**TRIE, "topk": 256})
    _randomize(full)
    fast.load_state_dict(full.state_dict())
    ids = torch.randint(0, 256, (2, 160))
    a, b = full(ids, labels=ids), fast(ids, labels=ids)
    assert torch.allclose(a.logits, b.logits, atol=1e-5)
    ga = torch.autograd.grad(a.loss, [full.trie_head.tables[0].weight, full.trie_head.query_proj.weight])
    gb = torch.autograd.grad(b.loss, [fast.trie_head.tables[0].weight, fast.trie_head.query_proj.weight])
    for x, y in zip(ga, gb):
        assert torch.allclose(x, y, atol=1e-6)


def test_rescored_bytes_are_the_topk_of_the_base_logits_with_the_full_heads_scores():
    torch.manual_seed(0)
    T = 90
    ids = torch.randint(0, 256, (T,))
    mask = torch.rand(T) < 0.3
    mask[0] = True
    ctx = _ctx([T])
    full = TrieHead(6, 256, rows=101, dim=4, n_hash=2, max_len=16)
    fast = TrieHead(6, 256, rows=101, dim=4, n_hash=2, max_len=16, topk=5)
    for t in full.tables:
        torch.nn.init.normal_(t.weight)
    fast.load_state_dict(full.state_dict())
    q = torch.randn(T, 6)
    base = torch.randn(T, 256)
    bonus, _ = full(q, ids, ctx, mask)
    cand, vals, length = fast.rescore(q, ids, ctx, mask, base)
    assert torch.equal(cand, base.topk(5, -1).indices)
    assert torch.allclose(vals, bonus.gather(1, cand), atol=1e-5)
    assert torch.equal(length, full.prefix_hashes(ids, ctx, mask)[1])


def test_candidates_include_the_match_bonus():
    """With the match model's logit gate the top-k is taken over LM head + match bonus (the model's own ranking)."""
    m = _model({**TRIE, "topk": 4}, match=True)
    torch.nn.init.normal_(m.match_model.gate.weight, std=1.0)
    torch.nn.init.constant_(m.match_model.gate.bias, 50.0)  # a copy candidate always wins
    seen = {}
    orig = m.trie_head.rescore

    def spy(q, ids, ctx, mask, base):
        out = orig(q, ids, ctx, mask, base)
        seen["base"], seen["cand"] = base, out[0]
        return out

    m.trie_head.rescore = spy
    ids = torch.tensor([list(b"abcabcabcabcabcabcabcabc" * 4)])
    m.eval()
    with torch.no_grad():
        m(ids)
    cand_m = m.match_model.candidates(ids.reshape(-1), _ctx([ids.numel()]))
    has = (cand_m < 256).any(0)
    assert has.any()
    for t in torch.nonzero(has).flatten().tolist():
        for c in cand_m[:, t].tolist():
            if c < 256:
                assert c in seen["cand"][t].tolist()


def test_gradient_paths():
    """Byte CE -> trie tables at step 0 (u != 0), query_proj only once rows are non-zero; through the query (dechunked
    main output) the trie path reaches the main network and the router (EMA dechunker), not the decoder or the LM head
    (the top-k selection is detached)."""
    m = _model({**TRIE, "topk": 8})
    ids = torch.randint(0, 256, (2, 200))
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, [m.trie_head.tables[0].weight, m.trie_head.query_proj.weight])
    assert g[0].abs().sum() > 0 and g[1].abs().sum() == 0
    _randomize(m, 0.1)
    seen = {}
    orig = m.trie_head.rescore

    def spy(q, ids_, ctx, mask, base):
        seen["base"] = base
        return orig(q, ids_, ctx, mask, base)

    m.trie_head.rescore = spy
    out = m(ids, labels=ids)
    r0 = out.bpred[0]
    flat = ids.reshape(-1)
    logits_base = seen["base"].detach()  # LM-head logits before the trie bonus
    cand, vals, _ = orig(r0.main_out, flat, _ctx([200, 200]), r0.boundary_mask, logits_base)
    ce = torch.nn.functional.cross_entropy(logits_base.scatter_add(1, cand, vals), flat)
    # the model's own loss reaches the LM head and the decoder too (through the base logits)
    g_dec = torch.autograd.grad(out.loss, list(m.backbone.decoder.parameters()), retain_graph=True, allow_unused=True)
    assert sum(float(x.abs().sum()) for x in g_dec if x is not None) > 0
    st = m.backbone
    groups = {"query": [m.trie_head.query_proj.weight], "tables": [t.weight for t in m.trie_head.tables],
              "main": list(st.main_network.parameters()), "router": list(st.routing_module.parameters()),
              "decoder": list(st.decoder.parameters()), "lm_head": [m.lm_head.weight]}
    norms = {}
    for k, ps in groups.items():
        gr = torch.autograd.grad(ce, ps, retain_graph=True, allow_unused=True)
        norms[k] = sum(float(x.abs().sum()) for x in gr if x is not None)
    assert norms["query"] > 0 and norms["tables"] > 0 and norms["main"] > 0 and norms["router"] > 0
    assert norms["decoder"] == 0 and norms["lm_head"] == 0


def test_only_candidate_rows_get_gradient():
    torch.manual_seed(0)
    T = 40
    ids = torch.randint(0, 256, (T,))
    mask = torch.zeros(T, dtype=torch.bool)
    mask[0] = True
    th = TrieHead(6, 256, rows=1 << 16, dim=4, n_hash=1, max_len=16, topk=3)
    q = torch.randn(T, 6)
    base = torch.randn(T, 256)
    cand, vals, _ = th.rescore(q, ids, _ctx([T]), mask, base)
    vals.sum().backward()
    rows = torch.nonzero(th.tables[0].weight.grad.abs().sum(-1)).flatten()
    assert 0 < rows.numel() <= T * 3


def _trained_with_random_tables(match):
    m = _model({**TRIE, "topk": 8}, match=match, vocab=True)
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 256, (2, 256))
    for _ in range(3):
        o = m(ids, labels=ids)
        opt.zero_grad()
        (o.loss + o.aux_loss).backward()
        opt.step()
    _randomize(m, 0.3)  # large trie scores, so any leak through the rescored bytes would show
    return m.eval()


def _cands(m, ids, **kw):
    seen = {}
    orig = m.trie_head.rescore

    def spy(*a):
        out = orig(*a)
        seen["cand"] = out[0]
        return out

    m.trie_head.rescore = spy
    with torch.no_grad():
        logits = m(ids, **kw).logits
    m.trie_head.rescore = orig
    return logits.reshape(-1, 256), seen["cand"]


@pytest.mark.parametrize("match", [False, True])
def test_causal_and_packed_equal_separate_with_large_trie_scores(match):
    """The standard invariance suite runs with near-zero tables; here the tables are N(0, 0.3). The top-k selection is
    a discontinuous function of the base logits, so positions whose candidate set differs by float noise (a near-tie
    at rank k) are excluded; everywhere else the logits must agree."""
    torch.manual_seed(1)
    m = _trained_with_random_tables(match)
    ids = torch.randint(0, 256, (1, 300))
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256
    la, ca = _cands(m, ids)
    lb, cb = _cands(m, alt)
    same = (ca[:200] == cb[:200]).all(-1)
    assert same.float().mean() > 0.97
    assert torch.allclose(la[:200][same], lb[:200][same], atol=1e-5)
    a, b = torch.randint(0, 256, (1, 180)), torch.randint(0, 256, (1, 220))
    l1, c1 = _cands(m, a)
    l2, c2 = _cands(m, b)
    lp, cp = _cands(m, torch.cat([a[0], b[0]]), cu_seqlens=torch.tensor([0, 180, 400], dtype=torch.int32),
                    max_seqlen=220)
    ref_l, ref_c = torch.cat([l1, l2]), torch.cat([c1, c2])
    same = (cp == ref_c).all(-1)
    assert same.float().mean() > 0.97
    assert torch.allclose(lp[same], ref_l[same], atol=1e-5)


@pytest.mark.parametrize("match", [False, True])
def test_params_flops_and_training(match):
    k = 8
    base = model_flops(_cfg(match=match))
    cfg = _cfg({**TRIE, "topk": k}, match=match)
    rep = model_flops(cfg)
    d0 = cfg.d_model[0]
    comp = rep["per_stage"][0]["components"]["trie_head"]
    assert comp["params"] == 2 * 4099 * 16 + d0 * 16
    fwd = 2 * d0 * 16 + 2 * k * 16 + k * 16 + k
    assert abs(rep["flops_per_byte"] - base["flops_per_byte"] - 3 * fwd) < 1e-3
    full = model_flops(_cfg(TRIE, match=match))
    assert full["params"] == rep["params"] and full["flops_per_byte"] > rep["flops_per_byte"]
    m = _model(cfg.trie_head, match=match)
    assert sum(p.numel() for p in m.parameters()) == rep["params"]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (4, 128))
    losses = []
    for _ in range(8):
        out = m(ids, labels=ids)
        opt.zero_grad()
        (out.loss + out.aux_loss).backward()
        opt.step()
        losses.append(float(out.loss))
    assert losses[-1] < losses[0]
    assert out.stats["trie/bonus_rms"] > 0 and 0 < out.stats["trie/topk_cov"] <= 1
    assert out.stats["trie/prefix_len"] >= 0
    m.eval()
    with torch.no_grad():
        o = m(ids, labels=ids, return_logits=False)
    assert o.logits is None and torch.isfinite(o.loss)
