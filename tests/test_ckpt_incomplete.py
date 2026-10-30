"""A checkpoint left half-written by a preemption (empty meta.json) is skipped, not fatal (evolve g1 bpe_s2)."""

import os

from hnet.train.checkpoint import latest_checkpoint


def _mk(d, step, meta="{}", model=True):
    p = os.path.join(d, "ckpt", f"step_{step:07d}")
    os.makedirs(p)
    open(os.path.join(p, "meta.json"), "w").write(meta)
    if model:
        open(os.path.join(p, "model.pt"), "wb").write(b"x")
    return p


def test_latest_checkpoint_skips_empty_meta_and_missing_model(tmp_path):
    good = _mk(tmp_path, 100)
    _mk(tmp_path, 200, meta="")          # truncated by a preemption
    _mk(tmp_path, 300, model=False)      # model.pt never written
    assert latest_checkpoint(str(tmp_path)) == good
