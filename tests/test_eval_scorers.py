"""Scorer correctness on fake models with known log-probs, packing equivalence, prompt-boundary handling."""

import math

import pytest
import torch

from hnet.config import BOS_ID
from hnet.eval.fakes import BigramModel, SegmentSumModel
from hnet.eval.scorers import ByteModelScorer, TokenModelScorer, split_trailing_space

torch.set_num_threads(1)


def _bigram_logprob(m, seq, n):
    return sum(float(torch.log_softmax(m.row(seq[i - 1]).double(), -1)[seq[i]]) for i in range(len(seq) - n, len(seq)))


def test_split_trailing_space():
    assert split_trailing_space("Answer: ", "x") == ("Answer:", " x")
    assert split_trailing_space("Answer:", " x") == ("Answer:", " x")
    assert split_trailing_space("a \n ", "b") == ("a", " \n b")


def test_byte_scorer_known_logprobs():
    m = BigramModel()
    sc = ByteModelScorer(m, device="cpu", batch_tokens=64)
    reqs = [("Question: 2+2?\nAnswer:", " 4"), ("hé ", "llo wörld"), ("", "abc")]
    out = sc.loglikelihood(reqs)
    for (ctx, cont), (lp, greedy, nb) in zip(reqs, out):
        ctx2, cont2 = split_trailing_space(ctx, cont)
        seq = [BOS_ID] + list(ctx2.encode()) + list(cont2.encode())
        n = len(cont2.encode())
        assert nb == n
        assert lp == pytest.approx(_bigram_logprob(m, seq, n), abs=1e-4)
        exp_greedy = all(int(m.row(seq[i - 1]).argmax()) == seq[i] for i in range(len(seq) - n, len(seq)))
        assert greedy == exp_greedy


def test_greedy_flag_true():
    m = BigramModel()
    for x, y in [("a", "b"), ("b", "c"), ("c", "d")]:
        m.table.data[ord(x), ord(y)] = 50.0  # make "a" -> "b" -> "c" -> "d" the argmax chain
    sc = ByteModelScorer(m, device="cpu")
    (lp, greedy, nb), (_, greedy2, _) = sc.loglikelihood([("a", "bcd"), ("a", "bce")])
    assert greedy and nb == 3 and lp == pytest.approx(0.0, abs=1e-6)
    assert not greedy2


@pytest.mark.parametrize("packed", [True, False])
def test_packing_equivalence(packed):
    reqs = [(f"context number {i} " + "x" * (i * 7 % 23), f" continuation {i}" + "y" * (i % 5)) for i in range(17)]
    m = SegmentSumModel()
    one = [ByteModelScorer(m, device="cpu", packed=packed, batch_tokens=1).loglikelihood([r])[0] for r in reqs]
    m.calls = 0
    batched = ByteModelScorer(m, device="cpu", packed=packed, batch_tokens=200).loglikelihood(reqs)
    assert 1 < m.calls < len(reqs)  # really batched
    for a, b in zip(one, batched):
        assert a[0] == pytest.approx(b[0], abs=1e-4) and a[1:] == b[1:]


def test_truncation_keeps_continuation():
    m = BigramModel()
    sc = ByteModelScorer(m, device="cpu", max_len=16)
    lp_long, _, _ = sc.loglikelihood([("z" * 100 + "abc", " de")])[0]
    lp_short, _, _ = sc.loglikelihood([("z" * 3 + "abc", " de")])[0]
    assert lp_long == pytest.approx(lp_short, abs=1e-5)  # bigram only sees the previous byte


def test_token_scorer_boundaries():
    tok = pytest.importorskip("tiktoken").get_encoding("gpt2")
    m = BigramModel(V=50304, R=97)
    sc = TokenModelScorer(m, tokenizer=tok, device="cpu")
    ctx, cont, nb = sc.encode_pair("Question: What is the capital of France?\nAnswer:", " Paris")
    assert tok.decode(cont) == " Paris" and nb == 6 and ctx[0] == tok.eot_token
    ctx, cont, _ = sc.encode_pair("Answer: ", "Paris")  # trailing space moves into the continuation
    assert tok.decode(cont) == " Paris" and tok.decode(ctx[1:]) == "Answer:"
    ctx, cont, _ = sc.encode_pair("", "hello")  # empty context -> EOT prefix
    assert ctx == [tok.eot_token] and tok.decode(cont) == "hello"
    sc2 = TokenModelScorer(m, tokenizer=tok, device="cpu", add_bos=False)
    ctx, cont, _ = sc2.encode_pair("The cat", " sat")
    assert tok.decode(ctx) == "The cat"
    # known logprob of the scored tokens
    (lp, _, nb), = sc2.loglikelihood([("The cat", " sat on the mat")])
    seq = tok.encode_ordinary("The cat sat on the mat")
    n = len(seq) - len(tok.encode_ordinary("The cat"))
    assert lp == pytest.approx(_bigram_logprob(m, seq, n), abs=1e-3)
    assert nb == len(" sat on the mat")


def test_rolling_loglikelihood_matches_single_window():
    m = BigramModel()
    text = "The quick brown fox jumps over the lazy dog. " * 3
    big = ByteModelScorer(m, device="cpu", max_len=4096).rolling_loglikelihood([text])[0]
    small = ByteModelScorer(m, device="cpu", max_len=16).rolling_loglikelihood([text])[0]
    seq = [BOS_ID] + list(text.encode())
    exp = _bigram_logprob(m, seq, len(seq) - 1)
    assert big[1] == len(text.encode())
    assert big[0] == pytest.approx(exp, abs=1e-3) and small[0] == pytest.approx(exp, abs=1e-3)
    assert math.isfinite(big[0])
