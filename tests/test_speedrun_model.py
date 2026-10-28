"""modded-nanogpt-style dense model (hnet/model/dense_modded.py): API, packing, causality, backends."""

import warnings

import pytest
import torch

from hnet import registry
from hnet.config import IGNORE_INDEX, DenseConfig
from hnet.model.dense_modded import BLOCK_SIZE, ModdedGPT

torch.set_num_threads(1)
V = 64


def tiny_cfg(**extra) -> DenseConfig:
    x = dict(n_value_embeds=1, skip_attn_layers=[2], window_tokens=[128, 256])
    x.update(extra)
    return DenseConfig(variant="modded", vocab_size=V, d_model=32, n_layers=5, d_intermediate=128, num_heads=2,
                       extra=x)


def randomized(model: ModdedGPT) -> ModdedGPT:
    """Replace the zero-initialized projections so every parameter influences the output."""
    torch.manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim == 2:
                p.normal_(0, 0.3)
            else:
                p.add_(0.1 * torch.randn_like(p))
    return model


def packed_labels(ids: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    labels = torch.roll(ids, -1)
    labels[cu[1:].long() - 1] = IGNORE_INDEX
    return labels


def test_registry_and_api():
    torch.manual_seed(0)
    model = registry.build("dense", "modded", tiny_cfg())
    assert isinstance(model, ModdedGPT)
    assert len(model.blocks()) == 5 and model.stage_of_param("layers.0.mlp.c_fc.weight") == 0
    ids = torch.randint(0, V, (40,))
    cu = torch.tensor([0, 15, 40], dtype=torch.int32)
    labels = packed_labels(ids, cu)
    out = model(ids, cu, labels=labels)
    assert out.logits.shape == (40, V)
    assert int(out.n_targets) == 38 and float(out.aux_loss) == 0.0 and out.bpred == []
    # zero-initialized head: uniform prediction
    assert torch.allclose(out.loss, torch.log(torch.tensor(float(V))))
    assert torch.allclose(out.ce_sum, out.loss * 38)
    assert model(ids, cu, labels=labels, return_logits=False).logits is None


def test_backward_reaches_every_parameter():
    model = randomized(ModdedGPT(tiny_cfg()))
    ids = torch.randint(0, V, (50,))
    cu = torch.tensor([0, 20, 50], dtype=torch.int32)
    model(ids, cu, labels=packed_labels(ids, cu)).loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name


def test_packed_equals_separate_documents():
    model = randomized(ModdedGPT(tiny_cfg())).eval()
    lens = [7, 13, 5]
    docs = [torch.randint(0, V, (n,)) for n in lens]
    cu = torch.tensor([0, 7, 20, 25], dtype=torch.int32)
    with torch.no_grad():
        packed = model(torch.cat(docs), cu).logits
        separate = torch.cat([model(d).logits for d in docs])
        rows = model(torch.stack([docs[0], docs[0]])).logits  # [B, L] input: one segment per row
    torch.testing.assert_close(packed, separate, atol=1e-5, rtol=1e-5)
    assert rows.shape == (2, 7, V)
    torch.testing.assert_close(rows[1], separate[:7], atol=1e-5, rtol=1e-5)


def test_causality_and_segment_isolation():
    model = randomized(ModdedGPT(tiny_cfg())).eval()
    ids = torch.randint(0, V, (30,))
    cu = torch.tensor([0, 18, 30], dtype=torch.int32)
    t = 10
    ids2 = ids.clone()
    ids2[t] = (ids[t] + 1) % V
    with torch.no_grad():
        a, b = model(ids, cu).logits, model(ids2, cu).logits
    torch.testing.assert_close(a[:t], b[:t])  # the past does not see the future
    torch.testing.assert_close(a[18:], b[18:])  # the next document does not see this one
    assert not torch.allclose(a[t:18], b[t:18])


def test_sliding_window_is_blockwise_and_follows_progress():
    """Window of n blocks: query in block i sees keys in blocks i-n+1..i only."""
    model = randomized(ModdedGPT(tiny_cfg())).eval()
    T = 3 * BLOCK_SIZE
    ids = torch.randint(0, V, (T,))
    ids2 = ids.clone()
    ids2[5] = (ids[5] + 1) % V  # a token in block 0
    model.set_progress(0.0)  # 128 tokens -> 1 block
    assert int(model.window_blocks) == 1
    with torch.no_grad():
        a, b = model(ids).logits, model(ids2).logits
    assert not torch.allclose(a[5:BLOCK_SIZE], b[5:BLOCK_SIZE])
    torch.testing.assert_close(a[BLOCK_SIZE:], b[BLOCK_SIZE:])  # blocks 1, 2 cannot see block 0
    model.set_progress(1.0)  # 256 tokens -> 2 blocks: block 1 sees block 0, block 2 does not
    assert int(model.window_blocks) == 2
    with torch.no_grad():
        a, b = model(ids).logits, model(ids2).logits
    assert not torch.allclose(a[BLOCK_SIZE : 2 * BLOCK_SIZE], b[BLOCK_SIZE : 2 * BLOCK_SIZE])


def test_flex_matches_sdpa_on_cpu():
    """The CUDA path (FlexAttention + block mask) agrees with the dense-mask SDPA path, incl. padding to 128."""
    model = randomized(ModdedGPT(tiny_cfg(window_tokens=[128, 128]))).eval()
    T = 300
    ids = torch.randint(0, V, (T,))
    cu = torch.tensor([0, 90, 260, 300], dtype=torch.int32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # uncompiled flex_attention warns that it is slow
        ref = model(ids, cu).logits
        model.extra["attn_backend"] = "flex"
        flex = model(ids, cu).logits
    torch.testing.assert_close(flex, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_flex_matches_sdpa():
    model = randomized(ModdedGPT(tiny_cfg())).cuda().eval()
    ids = torch.randint(0, V, (512,), device="cuda")
    cu = torch.tensor([0, 200, 512], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        flex = model(ids, cu).logits
        model.extra["attn_backend"] = "sdpa"
        ref = model(ids, cu).logits
    torch.testing.assert_close(flex, ref, atol=2e-3, rtol=2e-3)


def test_record_shape_parameter_count():
    """GPT-2 small record shape: 12 layers (no attention in layer 7), 3 value embeddings, untied head."""
    cfg = DenseConfig(variant="modded", vocab_size=50304, d_model=768, n_layers=12, d_intermediate=3072, num_heads=6)
    with torch.device("meta"):
        model = ModdedGPT(cfg)
    n = sum(p.numel() for p in model.parameters())
    D, Vp = 768, 50304
    expected = 5 * Vp * D + 12 * 12 * D * D - 4 * D * D + 12 * 2 + 11 * 2 + 6
    assert n == expected  # ~275M with embeddings, ~85M in blocks
