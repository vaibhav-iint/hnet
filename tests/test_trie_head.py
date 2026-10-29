"""Prefix-trie output vocabulary (hnet.model.trie_head, HNetConfig.trie_head)."""

import torch

import hnet.model.trie_head as trie_mod
from hnet.config import model_config_from_yaml
from hnet.flops import model_flops
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.trie_head import TrieHead


def _cfg(trie=None):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    if trie is not None:
        cfg.trie_head = trie
    return cfg


TRIE = {"rows": 4099, "dim": 16, "n_hash": 2, "max_len": 8}


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def _ids_mask(texts, ends):
    ids, mask = [], []
    for text, e in zip(texts, ends):
        b = [254] + list(text.encode())
        m = [False] * len(b)
        m[0] = True
        for j in e:
            m[j] = True
        ids += b
        mask += m
    return torch.tensor(ids), torch.tensor(mask), _ctx([len(t.encode()) + 1 for t in texts])


def test_default_config_has_no_trie_head_and_unchanged_state_dict():
    cfg = _cfg()
    assert cfg.trie_head == {}
    m = build_model(cfg)
    assert m.trie_head is None
    assert not any("trie" in k for k in m.state_dict())


def test_zero_tables_leave_the_logits_of_the_default_model_unchanged():
    torch.manual_seed(0)
    base = build_model(_cfg())
    base.init_weights()
    torch.manual_seed(1)
    m = build_model(_cfg(TRIE))
    m.init_weights()
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("trie_head.") for k in missing)
    ids = torch.randint(0, 256, (2, 200))
    base.eval(), m.eval()
    with torch.no_grad():
        a, b = base(ids, labels=ids), m(ids, labels=ids)
    assert torch.equal(a.logits.float(), b.logits)
    assert torch.allclose(a.loss, b.loss, atol=0, rtol=1e-7)


def test_prefix_is_the_chunk_in_progress_and_hashes_by_content():
    # "the cat" | "a cat": chunk ends (end semantics) after "the", " cat" | "a", " cat"
    ids, mask, ctx = _ids_mask(["the cat", "a cat"], [[3, 7], [1, 5]])
    th = TrieHead(8, 256, rows=1 << 20, dim=4, n_hash=2, max_len=16)
    H, length = th.prefix_hashes(ids, ctx, mask)
    # segment 0: BOS t h e ' ' c a t ; ends at 0, 3, 7
    assert length[:8].tolist() == [0, 1, 2, 0, 1, 2, 3, 0]
    assert length[8:].tolist() == [0, 0, 1, 2, 3, 0]
    # prefix " ca" at byte 6 of segment 0 == prefix " ca" at byte 4 of segment 1 (offset 8): same key
    assert torch.equal(H[:, 6], H[:, 8 + 4])
    assert torch.equal(H[:, 5], H[:, 8 + 3])  # " c"
    assert not torch.equal(H[:, 1], H[:, 5])  # "t" vs " c"
    assert torch.equal(H[:, 0], H[:, 3]) and torch.equal(H[:, 3], H[:, 8])  # empty prefix at every chunk end
    assert not torch.equal(H[0], H[1])  # independent hashes


def test_long_prefixes_use_their_last_max_len_bytes():
    ids, mask, ctx = _ids_mask(["x" * 20 + "abcd", "y" * 9 + "abcd"], [[], []])
    th = TrieHead(8, 256, rows=1 << 20, dim=4, n_hash=1, max_len=4)
    H, length = th.prefix_hashes(ids, ctx, mask)
    assert length[24].item() == 24 and length[25 + 13].item() == 13
    assert torch.equal(H[:, 24], H[:, 25 + 13])


def test_scores_equal_brute_force_and_chunking_is_exact(monkeypatch):
    torch.manual_seed(0)
    ids, mask, ctx = _ids_mask(["hello world", "help"], [[5], []])
    th = TrieHead(6, 256, rows=101, dim=4, n_hash=2, max_len=16)
    for t in th.tables:
        torch.nn.init.normal_(t.weight)
    q = torch.randn(ids.numel(), 6)
    bonus, _ = th(q, ids, ctx, mask)
    H, _ = th.prefix_hashes(ids, ctx, mask)
    u = th.query_proj(q)
    from hnet.model.ngram import _MULT, _PRIME

    ref = torch.zeros(ids.numel(), 256)
    with torch.no_grad():
        for t in range(ids.numel()):
            for b in range(256):
                row = sum(th.tables[k].weight[torch.remainder(((H[k, t] * _PRIME) ^ (b + 1)) * _MULT, 101)]
                          for k in range(2))
                ref[t, b] = (row * u[t]).sum()
    assert torch.allclose(bonus, ref, atol=1e-5)
    monkeypatch.setattr(trie_mod, "ELEMENTS_PER_CHUNK", 256 * 4 * 3)  # 3 rows per chunk
    bonus2, _ = th(q, ids, ctx, mask)
    assert torch.allclose(bonus, bonus2, atol=1e-6)


def test_gradient_paths():
    """Byte CE -> trie tables at step 0 (u != 0); -> query_proj only once rows are non-zero; through the query (the
    dechunked main output) the trie path reaches the main network and the router (EMA dechunker), not the decoder."""
    torch.manual_seed(0)
    m = build_model(_cfg(TRIE))
    m.init_weights()
    ids = torch.randint(0, 256, (2, 200))
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, [m.trie_head.tables[0].weight, m.trie_head.query_proj.weight])
    assert g[0].abs().sum() > 0 and g[1].abs().sum() == 0
    for t in m.trie_head.tables:
        torch.nn.init.normal_(t.weight, std=0.1)
    out = m(ids, labels=ids)
    bonus_ce = torch.nn.functional.cross_entropy(m.trie_head(out.bpred[0].main_out, ids.reshape(-1),
                                                             _ctx([200, 200]), out.bpred[0].boundary_mask)[0],
                                                 ids.reshape(-1))
    st = m.backbone
    groups = {"query": [m.trie_head.query_proj.weight], "main": list(st.main_network.parameters()),
              "router": list(st.routing_module.parameters()), "decoder": list(st.decoder.parameters()),
              "lm_head": [m.lm_head.weight]}
    norms = {}
    for k, ps in groups.items():
        gr = torch.autograd.grad(bonus_ce, ps, retain_graph=True, allow_unused=True)
        norms[k] = sum(float(x.abs().sum()) for x in gr if x is not None)
    assert norms["query"] > 0 and norms["main"] > 0 and norms["router"] > 0
    assert norms["decoder"] == 0 and norms["lm_head"] == 0


def test_query_final_uses_the_decoder_state():
    torch.manual_seed(0)
    m = build_model(_cfg({**TRIE, "query": "final"}))
    m.init_weights()
    for t in m.trie_head.tables:
        torch.nn.init.normal_(t.weight, std=0.1)
    ids = torch.randint(0, 256, (2, 120))
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, list(m.backbone.decoder.parameters()), allow_unused=True)
    assert sum(float(x.abs().sum()) for x in g if x is not None) > 0


def test_params_flops_and_training():
    cfg = _cfg()
    base = model_flops(cfg)
    cfg = _cfg(TRIE)
    rep = model_flops(cfg)
    d0, V = cfg.d_model[0], cfg.vocab_size
    comp = rep["per_stage"][0]["components"]["trie_head"]
    assert comp["params"] == 2 * 4099 * 16 + d0 * 16
    fwd = 2 * d0 * 16 + 2 * V * 16 + V * 16 + V
    assert abs(rep["flops_per_byte"] - base["flops_per_byte"] - 3 * fwd) < 1e-3
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    assert sum(p.numel() for p in m.parameters()) == rep["params"]
    assert m.stage_of_param("trie_head.tables.0.weight") == 0
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
    assert out.stats["trie/bonus_rms"] > 0
    m.eval()
    with torch.no_grad():
        o = m(ids, labels=ids, return_logits=False)
    assert o.logits is None and torch.isfinite(o.loss)
