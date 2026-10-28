"""Smoke training run: a ~1M-param 2-stage H-Net learns Python source bytes on CPU (Adam, 150 steps)."""

import glob
import os
import sys
import time

import pytest
import torch

from hnet.config import BOS_ID, EOS_ID, HNetConfig
from hnet.model import build_model


def corpus(n_bytes: int = 400_000) -> torch.Tensor:
    files = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    if not files:
        pytest.skip("no stdlib sources found")
    out = bytearray()
    for f in files:
        with open(f, "rb") as fh:
            out += bytes([BOS_ID]) + fh.read() + bytes([EOS_ID])
        if len(out) >= n_bytes:
            break
    return torch.tensor(list(out[:n_bytes]), dtype=torch.long)


def tiny_cfg(router: str) -> HNetConfig:
    return HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T1m1", ["T2"], "m1T1"], "m1"],
        d_model=[64, 96, 128],
        d_intermediate=[0, 192, 256],
        ssm=dict(d_state=16, d_conv=4, expand=2, headdim=16, chunk_size=64),
        attn=dict(num_heads=4, rotary_emb_dim=[8, 12, 16], window_size=[127, 127, -1]),
        dc=dict(target_ratios=[3.0, 2.0], router=router),
    ))


def train(router: str, steps: int = 150, batch: int = 4, seq_len: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    data = corpus()
    model = build_model(tiny_cfg(router))
    model.init_weights()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    hist = []
    for _ in range(steps):
        starts = torch.randint(0, data.numel() - seq_len - 1, (batch,), generator=g)
        win = torch.stack([data[s : s + seq_len + 1] for s in starts.tolist()])
        out = model(win[:, :-1], labels=win[:, 1:], return_logits=False)
        (out.loss + out.aux_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append({"loss": out.loss.item(), **{k: v.item() for k, v in out.stats.items() if "compression" in k}})
    return model, hist


def _mean(hist, key, sl):
    xs = [h[key] for h in hist[sl]]
    return sum(xs) / len(xs)


@pytest.mark.parametrize("router", ["cosine_lookahead", "end_mlp"])
def test_tiny_training_learns_and_compresses(router):
    torch.set_num_threads(1)
    t0 = time.time()
    model, hist = train(router)
    n_params = sum(p.numel() for p in model.parameters())
    first, last = _mean(hist, "loss", slice(0, 10)), _mean(hist, "loss", slice(-10, None))
    init = [hist[0][f"compression/s{s}"] for s in range(2)]
    final = [_mean(hist, f"compression/s{s}", slice(-10, None)) for s in range(2)]
    print(f"\n{router}: params={n_params / 1e6:.2f}M loss {first:.3f} -> {last:.3f}, compression "
          f"{init[0]:.2f}/{init[1]:.2f} -> {final[0]:.2f}/{final[1]:.2f} (targets 3/2), {time.time() - t0:.1f}s",
          file=sys.stderr)
    assert 0.5e6 < n_params < 2e6
    assert last < 0.6 * first, (first, last)
    # The ratio loss pulls compression towards its targets.
    for s, target in enumerate((3.0, 2.0)):
        assert abs(final[s] - target) < 0.8, (s, init, final)
        assert abs(final[s] - target) <= abs(init[s] - target) + 0.3, (s, init, final)
