"""Budget math. Pure functions over resource records (dicts from state.py); no network.

Accounting rule (conservative): a resource still active is charged for its whole reservation
(start .. deadline), because the platform-side timeout is what bounds it; a finished resource is
charged for start .. ended_at (capped at the deadline). Where the platform's billing rows for a
resource are known and larger, they win.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def hours_between(t0: float, t1: float) -> float:
    return max(0.0, (t1 - t0) / 3600.0)


def actual_hours(rec: dict, now: float | None = None) -> float:
    """Hours the resource has been up so far (ended_at, or now, capped at its deadline)."""
    now = time.time() if now is None else now
    end = rec.get("ended_at") or now
    if rec.get("deadline"):
        end = min(end, rec["deadline"])
    return hours_between(rec["started_at"], end)


def actual_cost(rec: dict, now: float | None = None) -> float:
    est = rec["price_hr"] * actual_hours(rec, now)
    return max(est, float(rec.get("billed_usd") or 0.0))


def committed_cost(rec: dict, now: float | None = None) -> float:
    """Actual cost for finished resources; the full reservation for active ones."""
    if rec.get("ended_at"):
        return actual_cost(rec, now)
    if rec.get("deadline"):
        full = rec["price_hr"] * hours_between(rec["started_at"], rec["deadline"])
        return max(full, actual_cost(rec, now))
    return actual_cost(rec, now)


def spent_so_far(resources: list[dict], now: float | None = None) -> float:
    return sum(actual_cost(r, now) for r in resources)


def committed_total(resources: list[dict], now: float | None = None) -> float:
    return sum(committed_cost(r, now) for r in resources)


def projected_cost(price_per_gpu_hr: float, count: int, hours: float, extra_hr: float = 0.0) -> float:
    """price x K x H, plus per-hour extras (disk, CPU/RAM for sandboxes)."""
    return (price_per_gpu_hr * max(count, 0) + extra_hr) * hours


@dataclass
class GuardResult:
    ok: bool
    projected: float
    committed: float
    budget_total: float
    launch_cap: float
    reason: str = ""


def check_launch(
    projected: float,
    committed: float,
    budget_total: float,
    launch_cap: float,
    allow_over: bool = False,
    wallet_balance: float | None = None,
) -> GuardResult:
    """Refuse when the launch would push total commitments over budget, exceed the per-launch cap
    (unless allow_over), or exceed what the wallet can pay. allow_over never lifts the total budget."""
    r = GuardResult(True, projected, committed, budget_total, launch_cap)
    if projected < 0:
        r.ok, r.reason = False, "negative projected cost"
    elif committed + projected > budget_total + 1e-9:
        r.ok = False
        r.reason = (f"total budget: committed ${committed:.2f} + projected ${projected:.2f} "
                    f"> budget ${budget_total:.2f}")
    elif projected > launch_cap + 1e-9 and not allow_over:
        r.ok = False
        r.reason = f"per-launch cap: projected ${projected:.2f} > cap ${launch_cap:.2f} (use --allow-over)"
    elif wallet_balance is not None and projected > wallet_balance:
        r.ok = False
        r.reason = f"wallet: projected ${projected:.2f} > balance ${wallet_balance:.2f}"
    return r


def sandbox_price_hr(cpu: float, mem_gb: float, disk_gb: float, gpu_count: int = 0,
                     gpu_price: float = 0.0) -> float:
    from . import config as C
    cpu = max(1, int(cpu))
    return (cpu * C.SANDBOX_CPU_PER_VCPU_HR + mem_gb * C.SANDBOX_MEM_PER_GIB_HR
            + disk_gb * C.SANDBOX_DISK_PER_GIB_HR + gpu_count * gpu_price)
