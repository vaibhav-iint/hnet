"""WindowLoader: labels / cu_seqlens semantics, determinism, exact resume, DDP sharding."""
import numpy as np
import pytest
import torch

pytest.importorskip("tiktoken")

from hnet.config import BOS_ID, EOS_ID, IGNORE_INDEX  # noqa: E402
from hnet.data import shards as S  # noqa: E402
from hnet.data.loader import (  # noqa: E402
    ConcatStream, WindowLoader, batch_bytes_per_step, matched_token_seq_len)
from hnet.data.synth import make_synthetic_dataset  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_docs=150, shard_tokens=3000, seed=2)


def paths(ds, kind):
    return ds["train_token_paths"] if kind == "tokens" else ds["train_byte_paths"]


def reference_window(stream: np.ndarray, start: int, L: int, kind: str):
    """Slow per-position reference of the SPEC semantics."""
    x = stream[start : start + L].astype(np.int64)
    nxt = stream[start + 1 : start + L + 1].astype(np.int64)
    starts, labels = [], []
    for t in range(L):
        if kind == "tokens":
            st = t == 0 or x[t] == S.EOT_ID
            lab = IGNORE_INDEX if nxt[t] == S.EOT_ID else nxt[t]
        else:
            st = t == 0 or x[t] == BOS_ID or x[t - 1] == EOS_ID
            lab = IGNORE_INDEX if (x[t] == EOS_ID or nxt[t] == BOS_ID) else nxt[t]
        if st:
            starts.append(t)
        labels.append(lab)
    return x, np.array(labels), starts


@pytest.mark.parametrize("kind,L", [("tokens", 64), ("bytes", 256)])
def test_labels_and_cu_seqlens(ds, kind, L):
    ld = WindowLoader(paths(ds, kind), kind, L, n_windows=3, prefetch=0)
    stream = np.concatenate([np.asarray(a) for a in ld.stream.arrays])
    for step in range(4):
        b = ld.next_batch()
        T = 3 * L
        assert b["input_ids"].dtype == torch.long and b["labels"].dtype == torch.long
        assert b["cu_seqlens"].dtype == torch.int32 and b["input_ids"].shape == (T,)
        assert b["n_windows"] == 3
        cu = b["cu_seqlens"].numpy()
        assert cu[0] == 0 and cu[-1] == T and (np.diff(cu) > 0).all()
        assert b["max_seqlen"] == np.diff(cu).max()
        ref_starts = []
        for j, w in enumerate(ld.window_ids(step * 3)):
            x, lab, st = reference_window(stream, w * L, L, kind)
            np.testing.assert_array_equal(b["input_ids"][j * L:(j + 1) * L].numpy(), x)
            np.testing.assert_array_equal(b["labels"][j * L:(j + 1) * L].numpy(), lab)
            ref_starts += [j * L + s for s in st]
        np.testing.assert_array_equal(cu[:-1], ref_starts)
        # labels never cross a segment: the last position of a segment that ends a doc is ignored
        x = b["input_ids"].numpy()
        lab = b["labels"].numpy()
        for s, e in zip(cu[:-1], cu[1:]):
            if e < T and e % L != 0:  # segment ended by a document boundary (not a window end)
                assert lab[e - 1] == IGNORE_INDEX
            if kind == "tokens":
                assert not (lab == S.EOT_ID).any()
            else:
                assert not (lab == BOS_ID).any()
        if kind == "bytes":
            assert ((lab == EOS_ID) | (lab == IGNORE_INDEX) | (lab < 254)).all()
            assert (lab[x == EOS_ID] == IGNORE_INDEX).all()


def test_strict_window_labels(ds):
    ld = WindowLoader(paths(ds, "bytes"), "bytes", 128, n_windows=2, prefetch=0, cross_window_label=False)
    lab = ld.next_batch()["labels"].view(2, 128)
    assert (lab[:, -1] == IGNORE_INDEX).all()


def test_concat_stream_crosses_shards(ds):
    cs = ConcatStream(ds["train_byte_paths"], "bytes")
    full = np.concatenate([np.asarray(a) for a in cs.arrays])
    for start in [0, int(cs.offsets[1]) - 5, int(cs.offsets[1]) - 1, len(full) - 300]:
        np.testing.assert_array_equal(cs.read(start, 300), full[start:start + 300])


@pytest.mark.parametrize("shuffle", [0, 7])
@pytest.mark.parametrize("prefetch", [0, 2])
def test_determinism_and_resume(ds, shuffle, prefetch):
    mk = lambda: WindowLoader(paths(ds, "bytes"), "bytes", 128, n_windows=2, seed=3,  # noqa: E731
                              shuffle_buffer=shuffle, prefetch=prefetch)
    a, b = mk(), mk()
    seq_a = [a.next_batch() for _ in range(10)]
    seq_b = [b.next_batch() for _ in range(5)]
    sd = b.state_dict()
    c = mk()
    c.load_state_dict(sd)
    seq_c = [c.next_batch() for _ in range(5)]
    for x, y in zip(seq_a, seq_b + seq_c):
        for k in ("input_ids", "labels", "cu_seqlens"):
            assert torch.equal(x[k], y[k])
    # set_step gives the same position as consuming
    d = mk()
    d.set_step(5)
    assert torch.equal(d.next_batch()["input_ids"], seq_a[5]["input_ids"])
    for ld in (a, b, c, d):
        ld.close()


def test_resume_rejects_mismatch(ds):
    a = WindowLoader(paths(ds, "bytes"), "bytes", 128, prefetch=0)
    b = WindowLoader(paths(ds, "bytes"), "bytes", 256, prefetch=0)
    with pytest.raises(AssertionError):
        b.load_state_dict(a.state_dict())


def test_ddp_disjoint_and_complete(ds):
    W, n, steps = 3, 2, 4
    kw = dict(seed=1, shuffle_buffer=5, prefetch=0)
    single = WindowLoader(paths(ds, "tokens"), "tokens", 32, n_windows=n * W, **kw)
    ranks = [WindowLoader(paths(ds, "tokens"), "tokens", 32, n_windows=n, rank=r, world_size=W, **kw)
             for r in range(W)]
    seen = []
    for s in range(steps):
        ids = [ld.window_ids(ld.consumed) for ld in ranks]
        flat = [w for r in ids for w in r]
        assert len(set(flat)) == len(flat)  # disjoint across ranks
        assert flat == single.window_ids(single.consumed)  # same global order as one big rank
        bs = [ld.next_batch() for ld in ranks]
        big = single.next_batch()
        assert torch.equal(torch.cat([b["input_ids"] for b in bs]), big["input_ids"])
        seen += flat
    assert len(set(seen)) == len(seen)  # disjoint across steps


def test_shuffle_is_blockwise_permutation(ds):
    ld = WindowLoader(paths(ds, "tokens"), "tokens", 16, shuffle_buffer=10, seed=0, prefetch=0)
    N = ld.windows_per_epoch
    ids = [ld.physical_window(g) for g in range(N)]
    assert sorted(ids) == list(range(N)) and ids != list(range(N))
    for b0 in range(0, N, 10):
        assert sorted(ids[b0:b0 + 10]) == list(range(b0, min(N, b0 + 10)))
    # next epoch reshuffles, different seed differs
    ids2 = [ld.physical_window(N + g) for g in range(N)]
    assert sorted(ids2) == list(range(N)) and ids2 != ids
    other = WindowLoader(paths(ds, "tokens"), "tokens", 16, shuffle_buffer=10, seed=1, prefetch=0)
    assert [other.physical_window(g) for g in range(N)] != ids


def test_bytes_per_step_helpers():
    assert batch_bytes_per_step("bytes", 8192, 4, world_size=2, grad_accum=3) == 8192 * 24
    assert batch_bytes_per_step("tokens", 1000, 2, bpt=4.5) == 9000
    assert matched_token_seq_len(8192, bpt=4.62) == 1792
    assert matched_token_seq_len(8192) == 1792  # committed FineWeb-Edu stats
