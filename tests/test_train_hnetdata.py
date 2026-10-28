"""Trainer on WS-C's real on-disk formats (hnet.data.synth fixtures): byte and token loaders, ValSet bpb."""

import json
import os

import pytest
import torch

from hnet.train.config import load_run_config
from hnet.train.train import Trainer

torch.set_num_threads(1)
synth = pytest.importorskip("hnet.data.synth")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    try:
        return synth.make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_docs=120, shard_tokens=4000)
    except Exception as e:  # tiktoken vocab unavailable offline
        pytest.skip(f"synthetic dataset unavailable: {e!r}")


def _metrics(out, name):
    with open(os.path.join(out, name)) as f:
        return [json.loads(l) for l in f]


def _run(out, ds, kind):
    vocab = 256 if kind == "bytes" else 50304
    cfg = {
        "model": {"type": "custom", "factory": "hnet.train.toy:build", "d_model": 16, "n_layers": 2,
                  "vocab_size": vocab},
        "data": {"source": "hnet", "kind": kind, "seq_len": 256 if kind == "bytes" else 64, "micro_batch": 2,
                 "batch_windows": 4, "byte_paths": [str(p) for p in ds["train_byte_paths"]],
                 "token_paths": [str(p) for p in ds["train_token_paths"]],
                 "val_token_path": str(ds["val_token_path"]), "dataset": "finewebedu10B"},
        "train": {"out_dir": str(out), "steps": 6, "lr": 3e-3, "precision": "fp32", "ckpt_every": 0,
                  "save_final": False, "schedule": "constant"},
        "eval": {"every": 0, "at_end": True, "val_bytes": None, "val_byte_seq_len": 256, "val_token_seq_len": 64,
                 "windows_per_batch": 4},
        "log": {"every": 2},
    }
    tr = Trainer(load_run_config(base=cfg))
    tr.train()
    return tr


def test_byte_and_token_runs_score_same_val_bytes(tmp_path, dataset):
    tb = _run(tmp_path / "b", dataset, "bytes")
    tt = _run(tmp_path / "t", dataset, "tokens")
    assert tb.bpt == pytest.approx(4.6207, abs=1e-3)  # measured FineWeb-Edu bytes/token (bpt_stats.json)
    mb, mt = _metrics(tmp_path / "b", "metrics.jsonl"), _metrics(tmp_path / "t", "metrics.jsonl")
    assert mb[-1]["bytes_seen"] > 0 and mt[-1]["bytes_seen"] > 0
    assert mt[-1]["tokens_eq"] == 6 * 4 * 64
    eb, et = _metrics(tmp_path / "b", "eval.jsonl")[-1], _metrics(tmp_path / "t", "eval.jsonl")[-1]
    assert eb["val/bytes"] == et["val/bytes"] > 0  # identical scored text for both model kinds
    assert 0 < eb["val/bpb"] < 10 and 0 < et["val/bpb"] < 30
