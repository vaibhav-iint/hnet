"""Local state + ledger: one JSON file (default <scratchpad>/prime_state/state.json).

Layout:
  {"version": 1,
   "wallet_baseline": {"balance": float, "at": epoch} | null,
   "resources": {name: record},   # current record per name (active or ended)
   "history": [record, ...]}      # older records whose name was reused

A record: name, id, kind ("sandbox"|"pod"), gpu_type, gpu_count, price_hr (total $/h),
started_at, deadline (epoch), ended_at (epoch|None), status, remote_root, runs: [{id, cmd,
started_at}], billed_usd (from platform billing rows, if seen), meta (free-form).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
import time
from pathlib import Path

from . import config as C

EMPTY = {"version": 1, "wallet_baseline": None, "resources": {}, "history": []}


class State:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else C.state_dir() / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._read()

    # ---- io ----
    def _read(self) -> dict:
        if not self.path.exists():
            return json.loads(json.dumps(EMPTY))
        with open(self.path) as f:
            d = json.load(f)
        for k, v in EMPTY.items():
            d.setdefault(k, json.loads(json.dumps(v)))
        return d

    @contextlib.contextmanager
    def locked(self):
        """Re-read under an exclusive lock, yield, then save atomically."""
        lock = self.path.with_suffix(".lock")
        with open(lock, "w") as lf:
            fcntl.flock(lf, fcntl.LOCK_EX)
            try:
                self.data = self._read()
                yield self
                self.save()
            finally:
                fcntl.flock(lf, fcntl.LOCK_UN)

    def save(self) -> None:
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".state.", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump(self.data, f, indent=1, sort_keys=True)
        os.replace(tmp, self.path)

    # ---- resources ----
    def all_records(self) -> list[dict]:
        return list(self.data["resources"].values()) + list(self.data["history"])

    def active(self) -> list[dict]:
        return [r for r in self.data["resources"].values() if not r.get("ended_at")]

    def get(self, name: str) -> dict:
        rec = self.data["resources"].get(name)
        if rec is None:
            # allow lookup by id
            for r in self.data["resources"].values():
                if r.get("id") == name:
                    return r
            raise KeyError(f"no resource named {name!r} in {self.path}")
        return rec

    def add(self, rec: dict) -> dict:
        name = rec["name"]
        old = self.data["resources"].get(name)
        if old is not None:
            if not old.get("ended_at"):
                raise ValueError(f"resource {name!r} is still active (id {old.get('id')})")
            self.data["history"].append(old)
        rec.setdefault("runs", [])
        rec.setdefault("ended_at", None)
        rec.setdefault("billed_usd", 0.0)
        rec.setdefault("meta", {})
        self.data["resources"][name] = rec
        return rec

    def mark_ended(self, name: str, when: float | None = None, status: str = "TERMINATED") -> None:
        rec = self.get(name)
        if not rec.get("ended_at"):
            rec["ended_at"] = when or time.time()
        rec["status"] = status

    def add_run(self, name: str, run_id: str, cmd: str) -> None:
        self.get(name)["runs"].append({"id": run_id, "cmd": cmd, "started_at": time.time()})

    def set_wallet_baseline(self, balance: float) -> None:
        if self.data.get("wallet_baseline") is None:
            self.data["wallet_baseline"] = {"balance": balance, "at": time.time()}
