"""ValSet: byte and token models are scored on exactly the same documents and bytes."""
import math

import numpy as np
import pytest
import torch

tiktoken = pytest.importorskip("tiktoken")

from hnet.config import BOS_ID, EOS_ID, IGNORE_INDEX  # noqa: E402
from hnet.data import shards as S  # noqa: E402
from hnet.data.synth import make_synthetic_dataset  # noqa: E402
from hnet.data.valsets import ValSet, chunk_doc, speedrun_target, speedrun_valset  # noqa: E402

torch.set_num_threads(1)
LB, LT = 256, 64  # small windows so long docs get chunked


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_docs=80, shard_tokens=40000, seed=4,
                                  convert=False)


@pytest.fixture(scope="module")
def vs(ds):
    return ValSet("t", 30_000, ds["val_token_path"], byte_seq_len=LB, token_seq_len=LT, windows_per_batch=4)


def test_whole_documents_in_order(ds, vs):
    assert vs.n_docs > 3 and vs.n_bytes >= 30_000
    texts = [d.encode("utf-8") for d in ds["docs"] if d][: vs.n_docs]
    assert vs.n_bytes == sum(len(t) for t in texts)
    assert vs.docs[0][0] == 0  # first doc of the shard


def test_same_bytes_both_sides(vs):
    lens = S.gpt2_vocab_table()[2]
    bb, tb = vs.byte_batches(), vs.token_batches()
    nb_byte = sum(b["n_bytes"] for b in bb)
    nb_tok = sum(b["n_bytes"] for b in tb)
    assert nb_byte == nb_tok == vs.n_bytes
    for b in bb:  # byte side: n_bytes == number of real-byte targets == score_mask count
        lab = b["labels"]
        assert int(b["score_mask"].sum()) == b["n_bytes"] == int(((lab >= 0) & (lab < BOS_ID)).sum())
        assert not ((lab == EOS_ID) | (lab == BOS_ID)).any()
    for b in tb:  # token side: n_bytes == UTF-8 length of the scored token targets
        lab = b["labels"][b["score_mask"]].numpy()
        assert b["n_bytes"] == int(lens[lab].sum()) and not (lab == S.EOT_ID).any()
    assert sum(int(b["score_mask"].sum()) for b in tb) == vs.n_tokens


def test_segments_identical_text(ds, vs):
    enc = tiktoken.get_encoding("gpt2")
    tsegs, bsegs, tw = vs._segments()
    assert len(tsegs) == len(bsegs) > vs.n_docs  # some docs were chunked
    all_t, all_b = [], []
    for ts, bs, w in zip(tsegs, bsegs, tw):
        assert len(ts.x) <= LT and len(bs.x) <= LB
        tt = ts.y[ts.y != IGNORE_INDEX]
        by = bs.y[bs.y != IGNORE_INDEX]
        assert enc.decode_bytes(tt.tolist()) == bytes(by.astype(np.uint8)) and len(by) == w  # same chunk text
        np.testing.assert_array_equal(ts.y[:-1], ts.x[1:])  # next-element targets
        np.testing.assert_array_equal(bs.y[:-1], bs.x[1:])
        all_t += tt.tolist()
        all_b.append(bytes(by.astype(np.uint8)))
    text = b"".join(d.encode("utf-8") for d in ds["docs"] if d)[: vs.n_bytes]
    assert b"".join(all_b) == text == enc.decode_bytes(all_t)


@pytest.mark.parametrize("kind", ["bytes", "tokens"])
def test_packed_batches_wellformed(vs, kind):
    L = LB if kind == "bytes" else LT
    for b in vs.batches(kind):
        T = b["n_windows"] * L
        cu = b["cu_seqlens"].numpy()
        assert b["input_ids"].shape == (T,) and cu[0] == 0 and cu[-1] == T
        assert set(range(0, T, L)) <= set(cu.tolist())  # windows are segment boundaries
        lab = b["labels"].numpy()
        x = b["input_ids"].numpy()
        for s, e in zip(cu[:-1], cu[1:]):  # targets stay within the segment
            assert lab[e - 1] == IGNORE_INDEX
            seg_lab = lab[s:e - 1]
            m = seg_lab != IGNORE_INDEX
            np.testing.assert_array_equal(seg_lab[m], x[s + 1:e][m])


def test_chunk_doc_limits():
    rng = np.random.default_rng(0)
    bl = rng.integers(1, 9, size=1000)
    ch = chunk_doc(bl, 50, 120)
    assert ch[0][0] == 0 and ch[-1][1] == 1000
    for (a, b), (c, _) in zip(ch, ch[1:] + [(1000, None)]):
        assert b == c and b - a <= 49 and bl[a:b].sum() <= 119


def test_bpb_formula():
    assert math.isclose(ValSet.bpb(math.log(2) * 100, 100), 1.0)


def test_speedrun_target(ds):
    n = 5000
    enc = tiktoken.get_encoding("gpt2")
    toks = np.asarray(S.read_token_shard(ds["val_token_path"]))
    tgt = [int(t) for t in toks[1:n + 1] if t != S.EOT_ID]
    nbytes = len(enc.decode_bytes(tgt))
    r = speedrun_target(ds["val_token_path"], n_targets=n, losses=(3.28,))
    assert r["n_bytes"] == nbytes and r["n_eot_targets"] == n - len(tgt)
    assert math.isclose(r["target_bpb"]["3.28"], 3.28 * n / math.log(2) / nbytes)


def test_speedrun_valset_token_limit(ds, monkeypatch):
    import hnet.data.valsets as V

    monkeypatch.setattr(V, "SPEEDRUN_VAL_TOKENS", 6000)
    v = speedrun_valset(ds["val_token_path"], byte_seq_len=LB, token_seq_len=LT)
    assert v.n_docs > 0 and all(b <= 6001 for _, b in v.docs)
