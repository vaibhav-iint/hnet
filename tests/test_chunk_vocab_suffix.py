"""Suffix-keyed chunk memory (hnet.model.chunk_vocab, dc.chunk_vocab.suffix_lens)."""

import torch

from hnet.config import model_config_from_yaml
from hnet.flops import model_flops
from hnet.model import build_model
from hnet.model.chunk_vocab import ByteInfo, ChunkVocab, hash_bank
from hnet.model.chunking.select import select_positions
from hnet.model.ctx import SeqCtx

CV = {"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5, "n_neg": 256}
SUF = [2, 3, 4, 6, 8]


def _ctx(lengths):
    cu = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


def _setup(text_segments, ends_per_segment):
    ids, mask, lengths = [], [], []
    for text, ends in zip(text_segments, ends_per_segment):
        b = [0] + list(text.encode())
        m = [False] * len(b)
        m[0] = True
        for e in ends:
            m[e] = True
        ids += b
        mask += m
        lengths.append(len(b))
    ids, mask = torch.tensor(ids), torch.tensor(mask)
    ctx = _ctx(lengths)
    _, idx, ctx_inner = select_positions(mask, ctx)
    binfo = ByteInfo(hash_bank(ids, ctx, 2, 16), torch.arange(ids.numel()))
    return ids, ctx, ctx_inner, binfo


def _cfg(suffix=None):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    cfg.dc.router = "end_mlp"
    cfg.dc.decision = "quantile"
    cv = dict(CV)
    if suffix is not None:
        cv["suffix_lens"] = suffix
    cfg.dc.chunk_vocab = [cv]
    return cfg


def test_default_has_no_suffix_parameters_and_same_ids():
    cv = ChunkVocab(8, buckets=1 << 20, n_hash=2)
    assert cv.suffix_lens == [] and not hasattr(cv, "suffix_gate")
    assert sorted(n for n, _ in cv.named_parameters()) == ["tables.0.weight", "tables.1.weight"]
    m = build_model(_cfg())
    assert not any("suffix" in k for k in m.state_dict())


def test_default_forward_is_bit_identical_to_a_model_with_closed_suffix_rows():
    """With trained (random) table rows and the suffix gates closed exactly (bias -inf) the logits equal the default
    model's bit for bit: the flag only adds gated rows. (scripts: the default model's logits also match the parent
    commit's code bit for bit, see logbook/evolve/candidates/g1-mem-02/bit_identity.txt.)"""
    torch.manual_seed(0)
    base = build_model(_cfg())
    base.init_weights()
    for t in base.backbone.chunk_vocab.tables:
        torch.nn.init.normal_(t.weight, std=0.5)
    cfg = _cfg(SUF)
    cfg.dc.chunk_vocab[0]["suffix_gate_init"] = float("-inf")
    m = build_model(cfg)
    missing, unexpected = m.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected and all("suffix" in k for k in missing)
    ids = torch.randint(0, 256, (2, 200))
    base.eval(), m.eval()
    with torch.no_grad():
        assert torch.equal(base(ids).logits, m(ids).logits)


def test_suffix_rows_depend_only_on_the_string_ending_at_the_boundary():
    # "the cat" cut as "the"," cat" vs "th","e cat": the realized chunks differ but every suffix ending at "t" agrees
    _, ctx, ctx_inner, binfo = _setup(["the cat", "the cat"], [[3, 7], [2, 7]])
    torch.manual_seed(0)
    cv = ChunkVocab(8, buckets=1 << 16, n_hash=2, suffix_lens=[2, 3, 4], suffix_gate_init=10.0)
    for t in cv.tables:
        torch.nn.init.normal_(t.weight)
    x = torch.zeros(6, 8)
    e, gate = cv.suffix_embed(x, binfo, ctx, ctx_inner)
    # chunk " cat" (len 4): suffix 4 is the realized chunk -> skipped; seg 2's "e cat" (len 5): suffix 4 used
    assert gate[2].tolist()[2] == 0 and gate[5].tolist()[2] > 0.99
    assert torch.allclose(gate[2, :2], gate[5, :2])
    full_cat = cv.embed(cv.chunk_ids(binfo, ctx, ctx_inner), torch.float32)
    # segment 2's last chunk vector = its suffix rows 2,3,4 (gate ~1) = segment 1's suffix rows 2,3 + realized " cat"
    lhs = e[5] + 0  # suffix rows "at", "cat", " cat"
    rhs = e[2] + full_cat[2]  # suffix rows "at", "cat" + realized row " cat"
    assert torch.allclose(lhs, rhs, atol=1e-3)


def test_suffixes_do_not_cross_the_segment_start():
    _, ctx, ctx_inner, binfo = _setup(["ab", "xyzab"], [[2], [5]])
    cv = ChunkVocab(8, buckets=1 << 16, n_hash=1, suffix_lens=[2, 3, 6])
    _, gate = cv.suffix_embed(torch.zeros(4, 8), binfo, ctx, ctx_inner)
    # chunks: BOS | "ab" (avail 3 bytes incl. BOS) | BOS | "xyzab" (avail 6)
    assert gate[1].tolist() == [0.0, 0.5, 0.0]  # 2 = realized length; 6 > avail
    assert gate[3].tolist() == [0.5, 0.5, 0.5]
    assert gate[0].tolist() == [0.0, 0.0, 0.0]  # BOS chunk: avail 1


def test_gradient_paths_and_training():
    torch.manual_seed(0)
    m = build_model(_cfg(SUF))
    m.init_weights()
    cv = m.backbone.chunk_vocab
    assert torch.allclose(torch.sigmoid(cv.suffix_gate.bias), torch.full((len(SUF),), 0.5))
    ids = torch.randint(0, 256, (2, 300))
    out = m(ids, labels=ids)
    g = torch.autograd.grad(out.loss, [cv.tables[0].weight, cv.suffix_gate.weight, cv.suffix_gate.bias,
                                       cv.suffix_len_gate.weight])
    assert g[0].abs().sum() > 0  # tables learn at step 0
    assert all(x.abs().sum() == 0 for x in g[1:])  # gates: zero rows at init -> no gradient yet
    for t in cv.tables:
        torch.nn.init.normal_(t.weight, std=0.1)
    out = m(ids, labels=ids)
    r = out.bpred[0]
    for loss in (out.loss, cv.pred_loss(r.inner_out, r.vocab_tgt)):  # byte CE and next-chunk CE both reach the gates
        g = torch.autograd.grad(loss, [cv.suffix_gate.weight, cv.suffix_gate.bias, cv.suffix_len_gate.weight],
                                retain_graph=True)
        assert all(x.abs().sum() > 0 for x in g)
    assert "suffix_gate/s0" in out.stats and "suffix_gate_n8/s0" in out.stats


def test_params_and_flops():
    base = model_flops(_cfg())
    cfg = _cfg(SUF)
    rep = model_flops(cfg)
    d, S = cfg.d_model[0], len(SUF)
    assert rep["params"] - base["params"] == d * S + S + 17 * S
    assert 0 < rep["flops_per_byte"] - base["flops_per_byte"] < 1e-3 * base["flops_per_byte"]
    m = build_model(cfg)
    assert sum(p.numel() for p in m.parameters()) == rep["params"]
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    ids = torch.randint(0, 64, (4, 128))
    losses = []
    for _ in range(6):
        out = m(ids, labels=ids)
        opt.zero_grad()
        (out.loss + out.aux_loss).backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    assert losses[-1] < losses[0]
