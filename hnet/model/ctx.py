"""Sequence context shared by every layer of one stage (packed variable-length batches)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


def seq_idx_from_cu_seqlens(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    """int64 [T]: segment id of every position. Sync-free (``total_len`` is given by the caller).

    Empty segments (repeated entries in ``cu_seqlens``) are handled: ids still count every segment.
    """
    starts = torch.zeros(total_len + 1, dtype=torch.long, device=cu_seqlens.device)
    starts.index_add_(0, cu_seqlens[1:-1].long(), torch.ones_like(cu_seqlens[1:-1], dtype=torch.long))
    return starts[:total_len].cumsum(0)


@dataclass
class SeqCtx:
    """Describes a packed batch ``x[1, T, D]`` made of segments (documents).

    cu_seqlens: int32 [n_seg + 1], cumulative segment lengths, starts at 0, ends at T.
    max_seqlen: longest segment (python int; flash-attn needs it on the host).
    seq_idx:    int32 [1, T], segment id of every position (Mamba-2 kernels reset state where it changes).
    position_ids: int64 [T], position within its segment (RoPE restarts at every segment).
    outer_index: int64 [T] for a stage built by a chunker: index of each position in the outer stage
                (None for the outermost stage).
    cache:      per-stage memo for derived tensors shared by all layers of the stage (rotary cos/sin,
                attention masks). Keys are owned by the module that fills them.
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    seq_idx: torch.Tensor
    position_ids: torch.Tensor
    outer_index: Optional[torch.Tensor] = None
    cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def total_len(self) -> int:
        return int(self.seq_idx.shape[-1])

    @property
    def n_segments(self) -> int:
        return int(self.cu_seqlens.numel() - 1)

    @classmethod
    def from_cu_seqlens(
        cls, cu_seqlens: torch.Tensor, max_seqlen: Optional[int] = None, total_len: Optional[int] = None
    ) -> "SeqCtx":
        """Build the context. Host syncs: none if ``max_seqlen`` and ``total_len`` are given (the model always
        passes ``total_len`` from the input shape); otherwise one ``.item()``-style read for each missing value."""
        cu = cu_seqlens.to(torch.int32)
        T = int(cu[-1]) if total_len is None else int(total_len)
        seq_idx = seq_idx_from_cu_seqlens(cu, T)
        position_ids = torch.arange(T, device=cu.device) - cu.long()[seq_idx]
        if max_seqlen is None:
            max_seqlen = int((cu[1:] - cu[:-1]).max()) if cu.numel() > 1 else 0
        return cls(cu, int(max_seqlen), seq_idx.to(torch.int32).unsqueeze(0), position_ids)

    @classmethod
    def single(cls, T: int, device=None) -> "SeqCtx":
        return cls.from_cu_seqlens(torch.arange(2, dtype=torch.int32, device=device) * T, T, T)
