"""Trainer on a data mix (hnet.data.mix): `data: {dataset: <mix>, root: ...}` for byte and token models."""

import json
import os

import pytest
import torch

pytest.importorskip("tokenizers")
from hnet.data.synth import make_synthetic_mix  # noqa: E402
from hnet.train.config import load_run_config  # noqa: E402
from hnet.train.data import default_eot  # noqa: E402
from hnet.train.train import Trainer  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def mixroot(tmp_path_factory):
    root = tmp_path_factory.mktemp("mixroot")
    info = make_synthetic_mix(root, dataset="synthmix")
    return root, info


def _run(out, root, kind, vocab, val="standard"):
    cfg = {
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 16, "n_layers": 2,
                  "vocab_size": 256 if kind == "bytes" else vocab},
        "data": {"source": "hnet", "dataset": "synthmix", "root": str(root), "seq_len": 256 if kind == "bytes" else 64,
                 "micro_batch": 2, "batch_windows": 4},
        "train": {"out_dir": str(out), "steps": 4, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "save_final": False, "schedule": "constant"},
        "eval": {"every": 0, "at_end": True, "val": val, "val_bytes": None, "val_byte_seq_len": 256,
                 "val_token_seq_len": 64, "windows_per_batch": 4},
        "log": {"every": 2},
    }
    tr = Trainer(load_run_config(base=cfg))
    tr.train()
    return tr


def _last(out, name):
    with open(os.path.join(out, name)) as f:
        return [json.loads(line) for line in f][-1]


@pytest.mark.parametrize("val", ["standard", "code.a", "heldout"])
def test_byte_and_token_models_on_mix(tmp_path, mixroot, val):
    root, info = mixroot
    man = info["manifest"]
    vocab = man["tokenizer"]["vocab_size"]
    assert default_eot({"dataset": "synthmix", "root": str(root)}) == 0
    tb = _run(tmp_path / "b", root, "bytes", vocab, val)
    tt = _run(tmp_path / "t", root, "tokens", vocab, val)
    assert tt.eot == 0 and tb.bpt == pytest.approx(man["train"]["bytes_per_token"])
    eb, et = _last(tmp_path / "b", "eval.jsonl"), _last(tmp_path / "t", "eval.jsonl")
    want = man["val_sets"]["mix" if val == "standard" else val]["bytes"]
    assert eb["val/bytes"] == et["val/bytes"] == want  # identical scored text for both kinds
    mt = _last(tmp_path / "t", "metrics.jsonl")
    assert mt["bytes_seen"] > 0 and mt["tokens_eq"] == 4 * 4 * 64
