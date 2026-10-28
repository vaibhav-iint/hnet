"""Tiny fake LMs with the SPEC forward API, for tests (known log-probs, segment-aware context)."""

from types import SimpleNamespace

import torch


class BigramModel(torch.nn.Module):
    """logits[t] = table[input_ids[t] % R]: log p(next | prev) is known in closed form."""

    def __init__(self, V=256, R=256, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.R = R
        self.table = torch.nn.Parameter(torch.randn(R, V, generator=g), requires_grad=False)

    def row(self, prev):
        return self.table[prev % self.R]

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True):
        return SimpleNamespace(logits=self.table[input_ids % self.R])


class SegmentSumModel(torch.nn.Module):
    """Context-dependent fake LM: logits depend on the running sum of ids *within the segment*
    (so leaking state across packed segments, or padding, would change the scores)."""

    def __init__(self, V=256):
        super().__init__()
        self.V = V
        self.calls = 0

    def _logits(self, ids_1d):
        cs = torch.cumsum(ids_1d.double(), 0)
        v = torch.arange(self.V, dtype=torch.double)
        return 3 * torch.sin(0.37 * v[None] + 0.011 * cs[:, None] + 0.05 * ids_1d.double()[:, None]).float()

    def forward(self, input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True):
        self.calls += 1
        if input_ids.dim() == 2:
            return SimpleNamespace(logits=torch.stack([self._logits(r) for r in input_ids]))
        cu = cu_seqlens.tolist() if cu_seqlens is not None else [0, input_ids.numel()]
        return SimpleNamespace(logits=torch.cat([self._logits(input_ids[a:b]) for a, b in zip(cu[:-1], cu[1:])]))
