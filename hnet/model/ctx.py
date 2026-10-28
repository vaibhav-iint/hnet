"""Sequence context shared by every layer of one stage (packed variable-length batches)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SeqCtx:
    """Describes a packed batch ``x[1, T, D]`` made of segments (documents).

    cu_seqlens: int32 [n_seg + 1], cumulative segment lengths, starts at 0, ends at T.
    max_seqlen: longest segment (python int; flash-attn needs it on the host).
    seq_idx:    int32 [1, T], segment id of every position (Mamba-2 kernels reset state where it changes).
    position_ids: int64 [T], position within its segment (RoPE restarts at every segment).
    """

    cu_seqlens: torch.Tensor
    max_seqlen: int
    seq_idx: torch.Tensor
    position_ids: torch.Tensor

    @property
    def total_len(self) -> int:
        return int(self.seq_idx.shape[-1])

    @property
    def n_segments(self) -> int:
        return int(self.cu_seqlens.numel() - 1)

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens: torch.Tensor, max_seqlen: Optional[int] = None) -> "SeqCtx":
        cu = cu_seqlens.to(torch.int32)
        T = int(cu[-1])
        lens = (cu[1:] - cu[:-1]).long()
        seq_idx = torch.repeat_interleave(torch.arange(lens.numel(), device=cu.device), lens)
        starts = torch.repeat_interleave(cu[:-1].long(), lens)
        position_ids = torch.arange(T, device=cu.device) - starts
        if max_seqlen is None:
            max_seqlen = int(lens.max()) if lens.numel() else 0
        return cls(cu, int(max_seqlen), seq_idx.to(torch.int32).unsqueeze(0), position_ids)

    @classmethod
    def single(cls, T: int, device=None) -> "SeqCtx":
        return cls.from_cu_seqlens(torch.tensor([0, T], dtype=torch.int32, device=device), T)
