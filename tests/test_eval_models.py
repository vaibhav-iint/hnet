"""Scorers on our real (tiny) models: packed == one-by-one == padded; trainer-checkpoint loading in the CLI."""

import json
import os

import pytest
import torch

torch.set_num_threads(1)
hmodel = pytest.importorskip("hnet.model")

from hnet.config import DenseConfig, load_yaml, model_config_from_yaml  # noqa: E402
from hnet.eval.scorers import ByteModelScorer, TokenModelScorer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQS = [(f"Question: what is {i}?\nAnswer:", f" it is {i * 7}" + "!" * (i % 3)) for i in range(5)]
DENSE = dict(variant="llama", vocab_size=50304, d_model=64, n_layers=2, d_intermediate=128, num_heads=4)


def _check(make):
    one = [make(batch_tokens=1).loglikelihood([r])[0] for r in REQS]
    for kw in (dict(batch_tokens=4096), dict(batch_tokens=4096, packed=False)):
        for a, b in zip(one, make(**kw).loglikelihood(REQS)):
            assert a[0] == pytest.approx(b[0], abs=2e-4) and a[2] == b[2]


def test_hnet_packing_equivalence():
    m = hmodel.build_model(model_config_from_yaml(os.path.join(ROOT, "configs/tiny_cpu.yaml")))
    m.init_weights()
    _check(lambda **kw: ByteModelScorer(m, device="cpu", **kw))


def test_dense_packing_equivalence():
    pytest.importorskip("tiktoken")
    m = hmodel.build_model(DenseConfig(**DENSE))
    m.init_weights()
    _check(lambda **kw: TokenModelScorer(m, device="cpu", **kw))


@pytest.mark.parametrize("kind", ["hnet", "dense"])
def test_load_run_checkpoint(tmp_path, kind):
    from hnet.eval.run import load_scorer

    if kind == "hnet":
        section = load_yaml(os.path.join(ROOT, "configs/tiny_cpu.yaml"))["model"]
    else:
        pytest.importorskip("tiktoken")
        section = {"type": "dense", **DENSE}
    from hnet.train.train import build_model_from_section

    model, _ = build_model_from_section(section)
    ck = tmp_path / "run" / "ckpt" / "step_0000010"
    ck.mkdir(parents=True)
    torch.save(model.state_dict(), ck / "model.pt")
    (ck / "meta.json").write_text(json.dumps({"step": 10, "tokens_eq": 1000, "bytes_seen": 4600,
                                              "config": {"model": section}}))
    sc, info = load_scorer(str(tmp_path / "run"), device="cpu")
    assert info["kind"] == ("byte" if kind == "hnet" else "token") and info["step"] == 10
    (lp, greedy, nb), = sc.loglikelihood([("Hello", " world")])
    assert nb == 6 and lp < 0
