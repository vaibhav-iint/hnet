"""First-order linear recurrence with resets, in pure PyTorch (no Python loop over positions or blocks)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def linear_recurrence(u: torch.Tensor, log_a: torch.Tensor, reset: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Solve ``y_i = a_i * y_{i-1} + u_i`` with ``a_i = exp(log_a_i)``, and ``y_i = u_i`` where ``reset_i``.

    u: [M, D] float; log_a: [M] (<= 0); reset: [M] bool. Returns [M, D] in fp32 (fp64 for fp64 inputs).
    Differentiable w.r.t. u and log_a.

    Blocked algorithm: inside each block of ``block`` positions the solution is a lower-triangular matrix of decay
    products applied to u (one batched matmul); the states carried across blocks satisfy the same recurrence on
    the block level, which is solved recursively (depth log_block(M)). Work O(M * block * D).
    """
    M, D = u.shape
    nb = -(-M // block)
    pad = nb * block - M
    dtype = torch.promote_types(u.dtype, torch.float32)
    u = F.pad(u.to(dtype), (0, 0, 0, pad)).view(nb, block, D)
    cum = F.pad(log_a.to(dtype), (0, pad)).view(nb, block).cumsum(-1)
    n_resets = F.pad(reset, (0, pad)).view(nb, block).long().cumsum(-1)
    # L[b, i, k] = prod_{m=k+1..i} a_m for k <= i with no reset in (k, i]; else 0.
    tril = torch.ones(block, block, dtype=torch.bool, device=u.device).tril()
    keep = tril & (n_resets[:, :, None] == n_resets[:, None, :])
    neg_inf = torch.tensor(float("-inf"), device=u.device, dtype=dtype)
    L = torch.where(keep, cum[:, :, None] - cum[:, None, :], neg_inf).exp()
    y = torch.bmm(L, u)  # zero state entering each block
    if nb > 1:
        # State at the end of every block, then carry it into the next block with decay prod_{m<=i} a_m.
        ends = linear_recurrence(y[:, -1], cum[:, -1], n_resets[:, -1] > 0, block)
        carry = F.pad(ends[:-1], (0, 0, 1, 0))
        decay_in = torch.where(n_resets == 0, cum, neg_inf).exp()
        y = y + decay_in[:, :, None] * carry[:, None, :]
    return y.reshape(nb * block, D)[:M]
