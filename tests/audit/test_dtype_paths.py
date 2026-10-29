"""Audit: dtype handling around the dynamic-chunking modules.

Tests that demonstrate a bug are marked ``xfail(strict=True)`` so the launch gate (full suite) stays green;
run ``pytest tests/audit --runxfail`` to see them fail. A fix turns them into XPASS (strict -> failure): then drop
the marker.

1. ``hnet.eval.run.load_run_checkpoint(..., dtype=torch.bfloat16)`` (CLI ``--dtype bf16``) casts the whole model,
   including ``residual_proj``, to bf16; ``HNet.forward`` then feeds it ``h.float()`` with autocast disabled and the
   forward raises. (Loud failure, eval-only; ``load_official_checkpoint`` keeps residual_proj in fp32 and is fine.)
2. Under bf16 autocast (the CUDA training default) the learned routers' logits come out of a bf16 matmul, so the
   scores the rank decider sorts are bf16-quantized (a trained 1-stage checkpoint gives 782 distinct scores out of
   32768 positions). The cosine router is unaffected (cosine in fp32 on bf16 q/k, as official).
"""

import pytest
import torch

from hnet.config import HNetConfig
from hnet.model import build_model
from hnet.model.ctx import SeqCtx
from hnet.model.routing.end import EndMLPRouter


def _cfg():
    return HNetConfig.from_dict(dict(
        arch_layout=["m1", ["T1"], "m1"], d_model=[32, 64], d_intermediate=[0, 128],
        ssm=dict(d_state=8, d_conv=4, expand=2, headdim=16, chunk_size=8),
        attn=dict(num_heads=[2, 4], rotary_emb_dim=[8, 8]), dc=dict(target_ratios=[4.0])))


def test_model_cast_to_bf16_still_runs():
    """FAILS: RuntimeError 'mat1 and mat2 must have the same dtype' in residual_proj."""
    torch.manual_seed(0)
    m = build_model(_cfg()).eval()
    m.init_weights()
    m = m.to(torch.bfloat16)
    with torch.no_grad():
        m(torch.randint(0, 254, (40,)))


def test_learned_router_scores_are_not_bf16_quantized_under_autocast():
    """FAILS (precision, low severity): logits are exactly bf16-representable under autocast."""
    torch.manual_seed(0)
    r = EndMLPRouter(64, target_ratio=4.0)
    h = torch.randn(4096, 64)
    ctx = SeqCtx.single(4096)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = r(h, ctx)
    s = out.logit
    assert s.dtype == torch.float32
    assert not torch.equal(s, s.bfloat16().float()), "router scores carry only bf16 precision"
