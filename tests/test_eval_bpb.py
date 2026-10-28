"""bpb on WS-C val sets (synthetic shard) and on texts; byte and token paths agree with direct computation."""

import math

import pytest
import torch

from hnet.config import BOS_ID
from hnet.eval.bpb import bpb_of_texts, valset_bpb, valset_texts
from hnet.eval.fakes import BigramModel
from hnet.eval.scorers import ByteModelScorer, TokenModelScorer

torch.set_num_threads(1)
valsets = pytest.importorskip("hnet.data.valsets")
synth = pytest.importorskip("hnet.data.synth")


@pytest.fixture(scope="module")
def vs(tmp_path_factory):
    ds = synth.make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_docs=40, shard_tokens=3000, convert=False)
    return valsets.ValSet("t", 1500, ds["val_token_path"], byte_seq_len=512, token_seq_len=128, windows_per_batch=2)


def _bigram_bits(m, seq):
    lp = 0.0
    for a, b in zip(seq[:-1], seq[1:]):
        lp += float(torch.log_softmax(m.row(a).double(), -1)[b])
    return -lp / math.log(2)


def test_byte_valset_bpb_matches_texts(vs):
    m = BigramModel(V=256)
    r = valset_bpb(m, vs, "byte", device="cpu")
    assert r["n_bytes"] == vs.n_bytes == r["n_targets"]
    texts = valset_texts(vs)
    assert sum(len(t.encode()) for t in texts) == vs.n_bytes
    # a bigram model sees one unit of context: chunked val set == rolling windows == whole docs
    exp_bits = sum(_bigram_bits(m, [BOS_ID] + list(t.encode())) for t in texts)
    assert r["bpb"] == pytest.approx(exp_bits / vs.n_bytes, rel=1e-5)
    rt = bpb_of_texts(ByteModelScorer(m, device="cpu", max_len=100), texts)
    assert rt["bpb"] == pytest.approx(r["bpb"], rel=1e-5)


def test_token_valset_bpb_matches_texts(vs):
    tiktoken = pytest.importorskip("tiktoken")
    enc = tiktoken.get_encoding("gpt2")
    m = BigramModel(V=50304, R=97)
    r = valset_bpb(m, vs, "token", device="cpu")
    assert r["n_bytes"] == vs.n_bytes and r["n_targets"] == vs.n_tokens
    texts = valset_texts(vs)
    exp_bits = sum(_bigram_bits(m, [enc.eot_token] + enc.encode_ordinary(t)) for t in texts)
    assert r["bpb"] == pytest.approx(exp_bits / vs.n_bytes, rel=1e-5)
    rt = bpb_of_texts(TokenModelScorer(m, tokenizer=enc, device="cpu", max_len=50), texts)
    assert rt["bpb"] == pytest.approx(r["bpb"], rel=1e-5)
