"""packed_block_mask: flex attention over packed sequences whose length is not a multiple of the 128 block size.

EXP-002 crashed with CUDA illegal memory accesses from the compiled block-mask builder on such lengths; the helper
builds masks eagerly and never lets a mask_mod index past the end of its per-position tensors.
"""

import pytest
import torch

from hnet.model.ctx import SeqCtx
from hnet.model.mixers import attn as A

pytestmark = pytest.mark.skipif(A.flex_attention is None, reason="flex_attention unavailable")


def _ctx(lengths):
    cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
    return SeqCtx.from_cu_seqlens(cu)


@pytest.mark.parametrize("lengths", [[1, 2, 97], [130], [5] * 23 + [12], [129, 1, 3]])
@pytest.mark.parametrize("window", [-1, 7])
def test_flex_matches_sdpa_on_ragged_lengths(lengths, window):
    torch.manual_seed(0)
    ctx = _ctx(lengths)
    T = sum(lengths)
    q, k, v = (torch.randn(T, 2, 16) for _ in range(3))
    ref = A.attention(q, k, v, ctx, window, 0.25, backend="sdpa")
    ctx.cache.clear()
    out = A.attention(q, k, v, ctx, window, 0.25, backend="flex")
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)



@pytest.mark.parametrize("T", [1, 100, 128, 300])
@pytest.mark.parametrize("H", [None, 3])
def test_block_layout_matches_create_block_mask(T, H):
    """Same partial/full block structure as torch's own builder, for a per-head, data-dependent mask."""
    from torch.nn.attention.flex_attention import create_block_mask

    torch.manual_seed(T)
    seq = torch.sort(torch.randint(0, 4, (T,))).values
    gate = torch.rand(3, T) > 0.3

    def mask_mod(b, h, q_idx, kv_idx):
        return (seq[q_idx] == seq[kv_idx]) & (kv_idx <= q_idx) & gate[h % 3, kv_idx]

    ours = A.packed_block_mask(mask_mod, H, T, torch.device("cpu"))
    ref = create_block_mask(mask_mod, None, H, T, T, device="cpu")
    assert ours.seq_lengths == (T, T)
    for name in ("kv_num_blocks", "full_kv_num_blocks"):
        torch.testing.assert_close(getattr(ours, name), getattr(ref, name))
    torch.testing.assert_close(ours.to_dense(), ref.to_dense())


def test_mask_memory_is_chunked():
    """Row chunks bound the evaluated pairs: a tiny budget still gives the exact mask."""
    T = 515
    seq = torch.sort(torch.randint(0, 6, (T,))).values

    def mask_mod(b, h, q_idx, kv_idx):
        return (seq[q_idx] == seq[kv_idx]) & (kv_idx <= q_idx)

    a = A.packed_block_mask(mask_mod, None, T, torch.device("cpu"), max_elems=1)
    b = A.packed_block_mask(mask_mod, None, T, torch.device("cpu"))
    torch.testing.assert_close(a.to_dense(), b.to_dense())
    torch.testing.assert_close(a.full_kv_num_blocks, b.full_kv_num_blocks)
