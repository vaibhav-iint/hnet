"""Shard formats, exact GPT-2 byte decoding, token<->byte document alignment."""
import numpy as np
import pytest

tiktoken = pytest.importorskip("tiktoken")

from hnet.config import BOS_ID, EOS_ID  # noqa: E402
from hnet.data import shards as S  # noqa: E402
from hnet.data.synth import make_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_docs=120, shard_tokens=4000, seed=1)


def test_token_shard_roundtrip(tmp_path):
    t = np.random.default_rng(0).integers(0, 50257, size=1000).astype(np.uint16)
    p = tmp_path / "x_val_000000.bin"
    S.write_token_shard(p, t)
    assert S.token_shard_ntok(p) == 1000
    np.testing.assert_array_equal(S.read_token_shard(p), t)
    hdr = np.fromfile(p, dtype=np.int32, count=3)
    assert hdr.tolist() == [20240520, 1, 1000]
    # truncated file is detected
    with open(p, "r+b") as f:
        f.truncate(1024 + 100)
    with pytest.raises(AssertionError):
        S.read_token_shard(p)


def test_byte_shard_roundtrip(tmp_path):
    d = np.random.default_rng(0).integers(0, 256, size=777).astype(np.uint8)
    p = tmp_path / "b.bytes.bin"
    S.write_byte_shard(p, d, ndocs=3, ntok_source=99, last_complete=False)
    h = S.byte_shard_header(p)
    assert h == dict(nbytes=777, ndocs=3, ntok_source=99, last_complete=False)
    np.testing.assert_array_equal(S.read_byte_shard(p), d)


def test_vectorized_decode_matches_tiktoken():
    enc = tiktoken.get_encoding("gpt2")
    rng = np.random.default_rng(0)
    toks = rng.integers(0, 50256, size=5000)  # arbitrary ids, including partial-UTF-8 tokens
    out, off = S.decode_tokens_to_stream(toks)
    assert bytes(out) == enc.decode_bytes(toks.tolist())
    assert off[-1] == len(out)
    lens = S.token_byte_lengths(toks)
    assert lens.sum() == len(out)
    assert S.token_byte_lengths(np.array([S.EOT_ID]))[0] == 0


def test_docs_split_across_shards(ds):
    # the synthetic generator mimics modded-nanogpt: later shards start mid-document
    firsts = [int(S.read_token_shard(p)[0]) for p in ds["token_paths"]]
    assert firsts[0] == S.EOT_ID and any(f != S.EOT_ID for f in firsts[1:])


def test_alignment_byte_doc_i_is_token_doc_i(ds):
    enc = tiktoken.get_encoding("gpt2")
    for tp, bp, ip in zip(ds["token_paths"], ds["byte_paths"], ds["idx_paths"]):
        toks, data, idx = S.read_token_shard(tp), S.read_byte_shard(bp), S.read_doc_index(ip)
        assert S.byte_shard_header(bp)["ndocs"] == len(idx)
        for row in idx:
            tt = S.doc_text_tokens(toks, row).astype(np.int64).tolist()
            assert bytes(S.doc_text_bytes(data, row)) == enc.decode_bytes(tt)
            b0 = int(row[S.BYTE_OFF])
            if row[S.FLAGS] & S.HAS_BOS:
                assert data[b0] == BOS_ID and toks[row[S.TOK_OFF]] == S.EOT_ID
            if row[S.FLAGS] & S.HAS_EOS:
                assert data[b0 + (row[S.FLAGS] & 1) + row[S.N_BYTES]] == EOS_ID
        # docs tile the payloads exactly
        ends = idx[:, S.BYTE_OFF] + idx[:, S.N_BYTES] + (idx[:, S.FLAGS] & 1) + ((idx[:, S.FLAGS] >> 1) & 1)
        assert ends[-1] == len(data) and (ends[:-1] == idx[1:, S.BYTE_OFF]).all()
        tends = idx[:, S.TOK_OFF] + idx[:, S.N_TOKENS] + (idx[:, S.FLAGS] & 1)
        assert tends[-1] == len(toks) and (tends[:-1] == idx[1:, S.TOK_OFF]).all()


def test_concatenated_byte_stream_equals_documents(ds):
    """Byte shards concatenated == [BOS]+utf8+[EOS] per original document (shard splits are seamless)."""
    full = np.concatenate([np.asarray(S.read_byte_shard(p)) for p in ds["byte_paths"]])
    expect = b"".join(bytes([BOS_ID]) + d.encode("utf-8") + bytes([EOS_ID]) for d in ds["docs"])
    assert bytes(full) == expect
    # no marker bytes inside text
    assert np.count_nonzero(full == BOS_ID) == len(ds["docs"]) == np.count_nonzero(full == EOS_ID)


def test_chunked_conversion_invariant(ds):
    toks = np.asarray(S.read_token_shard(ds["token_paths"][1]))
    a = S.convert_token_array(toks, last_complete=False, chunk_tokens=1 << 20)
    b = S.convert_token_array(toks, last_complete=False, chunk_tokens=97)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_last_complete_inference(ds):
    for k, ip in enumerate(ds["idx_paths"]):
        idx = S.read_doc_index(ip)
        nxt_first = (int(S.read_token_shard(ds["token_paths"][k + 1])[0])
                     if k + 1 < len(ds["token_paths"]) else S.EOT_ID)
        assert bool(idx[-1, S.FLAGS] & S.HAS_EOS) == (nxt_first == S.EOT_ID)


def test_verify_sample(ds):
    from hnet.data.download import _verify_sample

    v = _verify_sample(ds["token_paths"][1], ds["byte_paths"][1], ds["idx_paths"][1], 20, 0)
    assert v["decode_exact"] == v["n_checked"] > 0
    assert v["reencode_match"] == v["n_reencode"] > 0
