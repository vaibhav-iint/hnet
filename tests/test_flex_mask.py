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

