"""Router output container, boundary semantics and the shared probability -> decision rule.

Boundary semantics (a class attribute ``boundary_semantics`` of every router, copied into ``RouterOutput``):

* ``"start"`` (paper / LH-Net): a selected position is the FIRST byte of a chunk; the chunk vector is h there.
* ``"end"`` (improved H-Net): a selected position is the LAST byte of a chunk; the chunk vector is h there.

In both cases the first position of every segment is selected (p = 1): under "start" it opens the first chunk,
under "end" it is a 1-byte chunk (the BOS byte), so every position has a completed chunk to read. Chunking and
dechunking are the same computation for both: keep the selected positions, and let position t read inner position
``cumsum(mask)[t] - 1`` = the latest selected position at or before t. Under "start" that is the chunk containing
t; under "end" it is the most recent chunk that ended at or before t (the chunk containing t shifted by one,
except at its end byte). Each router decides p_t from h_{<=t}, so both are causal; "end" additionally never sees
the first byte of the next chunk when closing a chunk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..ctx import SeqCtx

SEMANTICS = ("start", "end")


@dataclass
class RouterOutput:
    """Boundary decisions of one stage over its T positions.

    boundary_prob:  fp32 [T, 2], ``[1 - p_t, p_t]``.
    boundary_mask:  bool [T], selected positions (``p_t > 1 - p_t``; ties are not selected, as official argmax).
    selected_probs: fp32 [T, 1], confidence ``c_t = p_t if selected else 1 - p_t`` (paper eq. 6), fed to the STE.
    semantics:      "start" or "end" (see module docstring).
    """

    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor
    semantics: str = "start"

    @property
    def p(self) -> torch.Tensor:
        """fp32 [T] boundary probability."""
        return self.boundary_prob[:, 1]


def router_output(p: torch.Tensor, ctx: SeqCtx, semantics: str) -> RouterOutput:
    """Turn boundary probabilities p [T] into a ``RouterOutput``; p is forced to 1 at the first position of every
    segment (that position is always selected)."""
    assert semantics in SEMANTICS, semantics
    p = torch.where(ctx.position_ids == 0, torch.ones_like(p), p.float())
    prob = torch.stack((1 - p, p), dim=-1)
    mask = prob[:, 1] > prob[:, 0]
    selected = torch.where(mask, prob[:, 1], prob[:, 0]).unsqueeze(-1)
    return RouterOutput(prob, mask, selected, semantics)


def ratio_init_bias(target_ratio) -> float:
    """Logit b with sigmoid(b) = 1/N, so that mean p starts near the target selection rate."""
    return -math.log(float(target_ratio) - 1.0) if target_ratio and target_ratio > 1 else 0.0
