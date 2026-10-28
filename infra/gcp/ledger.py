"""GCP ledger and budget math (pure functions over records; no network).

A record (one per VM name in <state>/state.json, via infra.prime.state.State):
  name, zone, machine_type, gpu, gpu_count, spot, price_hr (machine $/h estimate), disk_gb, disk_hr,
  max_hours, created_at, ended_at (VM deleted), status, commit, cmd,
  segments: [{"start", "deadline", "end"}]   one per RUNNING period (launch, each resume)

Accounting (conservative):
  * machine: price_hr x running hours; an open segment is committed up to its deadline (the
    --max-run-duration hard cap), so the guard counts the whole reservation;
  * disk: disk_hr from creation until deletion (a stopped VM still pays for its disk); an undeleted
    VM is committed for disk until max(now, open deadline).
"""

from __future__ import annotations

import time
from datetime import datetime

from infra.prime.budget import GuardResult, check_launch  # noqa: F401  (same guard semantics as Prime)

RUNNING = ("PROVISIONING", "STAGING", "RUNNING", "REPAIRING")
STOPPED = ("STOPPING", "STOPPED", "TERMINATED", "SUSPENDING", "SUSPENDED")


def parse_ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _h(t0: float, t1: float) -> float:
    return max(0.0, (t1 - t0) / 3600.0)


def seg_hours(seg: dict, now: float) -> float:
    end = seg.get("end") or now
    if seg.get("deadline"):
        end = min(end, seg["deadline"])
    return _h(seg["start"], end)


def running_hours(rec: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    return sum(seg_hours(s, now) for s in rec.get("segments", []))


def open_segment(rec: dict) -> dict | None:
    segs = rec.get("segments") or []
    return segs[-1] if segs and not segs[-1].get("end") else None


def actual_cost(rec: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    disk_end = rec.get("ended_at") or now
    return rec["price_hr"] * running_hours(rec, now) + rec.get("disk_hr", 0.0) * _h(rec["created_at"], disk_end)


def committed_cost(rec: dict, now: float | None = None) -> float:
    now = time.time() if now is None else now
    machine = 0.0
    for s in rec.get("segments", []):
        if s.get("end") or not s.get("deadline") or rec.get("ended_at"):
            machine += rec["price_hr"] * seg_hours(s, now)
        else:
            machine += rec["price_hr"] * max(_h(s["start"], s["deadline"]), seg_hours(s, now))
    if rec.get("ended_at"):
        disk_end = rec["ended_at"]
    else:
        seg = open_segment(rec)
        disk_end = max(now, seg["deadline"]) if seg and seg.get("deadline") else now
    return machine + rec.get("disk_hr", 0.0) * _h(rec["created_at"], disk_end)


def committed_total(recs: list[dict], now: float | None = None) -> float:
    return sum(committed_cost(r, now) for r in recs)


def actual_total(recs: list[dict], now: float | None = None) -> float:
    return sum(actual_cost(r, now) for r in recs)


def projected_cost(price_hr: float, disk_hr: float, hours: float) -> float:
    return (price_hr + disk_hr) * hours


def apply_observation(rec: dict, inst: dict | None, now: float | None = None) -> str:
    """Update segments/status from a `describe` result (None = instance gone). Returns the status."""
    now = time.time() if now is None else now
    seg = open_segment(rec)
    if inst is None:
        if seg:
            seg["end"] = min(now, seg.get("deadline") or now)
        if not rec.get("ended_at"):
            rec["ended_at"] = now
        rec["status"] = "DELETED"
        return "DELETED"
    st = inst.get("status", "UNKNOWN")
    rec["status"] = st
    last_start = parse_ts(inst.get("lastStartTimestamp"))
    last_stop = parse_ts(inst.get("lastStopTimestamp"))
    if st in RUNNING and seg and last_start and last_start > seg["start"] + 120:
        # stale local state: the VM stopped and was started again while we were not looking
        end = last_stop if (last_stop and seg["start"] - 300 <= last_stop <= last_start) else last_start
        seg["end"] = min(end, seg.get("deadline") or end)
        seg = None
    if st in STOPPED and seg:
        end = last_stop if (last_stop and last_stop >= seg["start"] - 300) else now
        seg["end"] = min(end, seg.get("deadline") or end)
    elif st in RUNNING and not seg:
        # started outside `resume` (console / API): open a segment so the ledger keeps counting
        t0 = last_start or now
        rec.setdefault("segments", []).append({"start": t0, "deadline": t0 + rec["max_hours"] * 3600, "end": None})
    return st
