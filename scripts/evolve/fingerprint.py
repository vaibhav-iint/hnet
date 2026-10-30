"""Bit-level fingerprint of a model's forward and backward pass, to prove a code change leaves existing configs
bit-identical (docs/EVOLVE.md guardrail 3): run it once with the parent commit's ``hnet`` on PYTHONPATH and once with
the candidate's, and diff the outputs.

    PYTHONPATH=<parent checkout> python scripts/evolve/fingerprint.py CONFIG... > parent.txt
    PYTHONPATH=.                 python scripts/evolve/fingerprint.py CONFIG... > cand.txt
    diff parent.txt cand.txt

Per config: sha256 of the logits, the loss (repr), sha256 of every parameter gradient (sorted by name) and of the
stats dict, after seeding, ``init_weights`` and one forward + backward on 2 x 600 bytes of real mix_v1 validation text.
"""

from __future__ import annotations

import hashlib
import sys

import numpy as np
import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model

VAL = "data/mix_v1/bytes/mix_v1_val_000000.bytes.bin"


def sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    torch.set_num_threads(1)
    raw = np.fromfile(VAL, dtype=np.uint8)
    ids = torch.from_numpy(raw[2_000_000:2_000_000 + 1200].astype(np.int64)).view(2, 600)
    lab = torch.roll(ids, -1, dims=1)
    lab[:, -1] = -100
    for path in sys.argv[1:]:
        torch.manual_seed(0)
        m = build_model(model_config_from_yaml(path))
        m.init_weights()
        m.train()
        out = m(ids, labels=lab)
        (out.loss + out.aux_loss).backward()
        g = hashlib.sha256()
        for n, p in sorted(m.named_parameters()):
            g.update(n.encode())
            g.update(b"none" if p.grad is None else sha(p.grad).encode())
        st = hashlib.sha256(repr(sorted((k, float(v)) for k, v in out.stats.items())).encode()).hexdigest()[:16]
        print(f"{path}: logits {sha(out.logits)} loss {float(out.loss)!r} aux {float(out.aux_loss)!r} "
              f"grads {g.hexdigest()[:16]} stats {st} n_params {sum(p.numel() for p in m.parameters())}")


if __name__ == "__main__":
    main()
