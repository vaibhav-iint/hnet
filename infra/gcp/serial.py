"""Serial-console transport: incremental fetch with persisted offsets, parsing of HNET_* lines and
reconstruction of metrics.jsonl / eval.jsonl.

Per instance, under <state>/serial/<name>/:
  raw.log      every byte ever fetched, in order (gap/reset markers inserted by us)
  cursor.json  {"next": byte offset for the next --start, "lost_bytes", "resets", "fetches", "updated"}

GCE keeps only the last ~1 MB of serial output. A fetch whose returned `start` is past the requested
offset lost the bytes in between: we write a `[hnet-gcp] GAP` marker line (which also terminates any
partial line, so a half line is dropped instead of glued onto the next one). If the instance's output
counter restarts (next < requested start, e.g. after a stop/start), we write a RESET marker and
continue from 0. Lines are only parsed up to the last newline, so a line split across fetches is read
once complete.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("METRIC", "EVAL", "STATUS", "DONE", "STAGE", "INFO", "LOG", "SUMMARY")
LINE_RE = re.compile(r"HNET_(" + "|".join(KINDS) + r")(?: (.*))?$")
GAP_MARK = "[hnet-gcp] GAP"
RESET_MARK = "[hnet-gcp] RESET"


# ----------------------------------------------------------------------------- fetch with offsets
class SerialTracker:
    def __init__(self, root: Path | str):
        self.dir = Path(root)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.raw = self.dir / "raw.log"
        self.cur_path = self.dir / "cursor.json"
        self.cur = {"next": 0, "lost_bytes": 0, "resets": 0, "fetches": 0, "updated": None}
        if self.cur_path.exists():
            self.cur.update(json.loads(self.cur_path.read_text()))

    def _save(self):
        tmp = self.cur_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.cur))
        os.replace(tmp, self.cur_path)

    def _append(self, text: str):
        with open(self.raw, "a", encoding="utf-8", errors="replace") as f:
            f.write(text)

    def fetch(self, fetch_fn, max_rounds: int = 20) -> str:
        """fetch_fn(start) -> {'contents', 'start', 'next'}. Appends new output to raw.log and returns it.
        Loops while a response is non-empty in case the API pages its output."""
        new = []
        for _ in range(max_rounds):
            want = int(self.cur["next"])
            r = fetch_fn(want)
            got, nxt, txt = int(r.get("start", want)), int(r.get("next", want)), r.get("contents") or ""
            self.cur["fetches"] += 1
            if nxt < want and got < want:
                # counter restarted (new boot / buffer reset): start over from what is there now
                mark = f"\n{RESET_MARK} expected>={want} got next={nxt}\n"
                self._append(mark)
                new.append(mark)
                self.cur["resets"] += 1
                self.cur["next"] = 0
                self._save()
                continue
            if got > want:
                lost = got - want
                self.cur["lost_bytes"] += lost
                mark = f"\n{GAP_MARK} lost {lost} bytes ({want}..{got})\n"
                self._append(mark)
                new.append(mark)
            if txt:
                self._append(txt)
                new.append(txt)
            self.cur["next"] = max(nxt, got)
            self.cur["updated"] = time.time()
            self._save()
            if len(txt) < 512 * 1024 or nxt <= want:
                break  # a short answer means we are caught up; a ~1 MB one may have more behind it
        return "".join(new)

    def read_raw(self) -> str:
        return self.raw.read_text(encoding="utf-8", errors="replace") if self.raw.exists() else ""


# ----------------------------------------------------------------------------- parsing
@dataclass
class Event:
    kind: str
    run_dir: str | None = None
    data: object = None
    text: str = ""


@dataclass
class Parsed:
    metrics: dict = field(default_factory=dict)   # run_dir -> [rec]
    evals: dict = field(default_factory=dict)     # run_dir -> [rec]
    status: Event | None = None
    stages: list = field(default_factory=list)
    info: dict | None = None
    summary: dict | None = None
    done: int | None = None
    logs: list = field(default_factory=list)
    corrupt: int = 0
    gaps: int = 0
    resets: int = 0
    events: int = 0


def unesc_dir(d: str) -> str:
    return d.replace("%20", " ").replace("%25", "%")


def complete_lines(text: str) -> list[str]:
    """Lines up to the last newline (a trailing partial line is left for the next fetch)."""
    end = text.rfind("\n")
    if end < 0:
        return []
    return [ln.rstrip("\r") for ln in text[:end].split("\n")]


def parse_line(line: str) -> Event | None:
    m = LINE_RE.search(line)
    if not m:
        return None
    kind, rest = m.group(1), (m.group(2) or "").strip()
    if kind in ("METRIC", "EVAL"):
        run_dir, _, js = rest.partition(" ")
        rec = json.loads(js)  # ValueError -> corrupt
        if not isinstance(rec, dict):
            raise ValueError("not an object")
        return Event(kind, unesc_dir(run_dir), rec, js)
    if kind == "STATUS":
        state, _, js = rest.partition(" ")
        try:
            d = json.loads(js) if js else {}
        except ValueError:
            d = {"raw": js}
        return Event(kind, None, {"state": state, **(d if isinstance(d, dict) else {})}, rest)
    if kind == "DONE":
        return Event(kind, None, int(rest.split()[0]), rest)
    if kind in ("STAGE", "INFO", "SUMMARY"):
        return Event(kind, None, json.loads(rest) if rest else {}, rest)
    return Event(kind, None, rest, rest)


def _add_record(store: dict, run_dir: str, rec: dict):
    """Append, emulating trainer resume: a record whose step is <= the last kept step replaces the tail
    from that step on (exact duplicates collapse)."""
    lst = store.setdefault(run_dir, [])
    step = rec.get("step")
    if step is not None and lst and lst[-1].get("step") is not None and step <= lst[-1]["step"]:
        while lst and lst[-1].get("step") is not None and lst[-1]["step"] >= step:
            lst.pop()
    lst.append(rec)


def parse_text(text: str) -> Parsed:
    p = Parsed()
    for line in complete_lines(text):
        if GAP_MARK in line:
            p.gaps += 1
            continue
        if RESET_MARK in line:
            p.resets += 1
            continue
        if "HNET_" not in line:
            continue
        try:
            ev = parse_line(line)
        except (ValueError, IndexError):
            p.corrupt += 1
            continue
        if ev is None:
            continue
        p.events += 1
        if ev.kind == "METRIC":
            _add_record(p.metrics, ev.run_dir, ev.data)
        elif ev.kind == "EVAL":
            _add_record(p.evals, ev.run_dir, ev.data)
        elif ev.kind == "STATUS":
            p.status = ev
        elif ev.kind == "STAGE":
            p.stages.append(ev.data)
            if isinstance(ev.data, dict) and ev.data.get("stage") == "boot":
                p.done = None  # a new boot: the previous DONE no longer describes the current state
        elif ev.kind == "INFO":
            p.info = ev.data
        elif ev.kind == "SUMMARY":
            p.summary = ev.data
        elif ev.kind == "DONE":
            p.done = ev.data
        elif ev.kind == "LOG":
            p.logs.append(ev.data)
    return p


# ----------------------------------------------------------------------------- reconstruction
def local_run_dir(out_root: Path | str, run_dir: str) -> Path:
    """runs/tiny/x -> <out_root>/tiny/x ; other relative paths kept under out_root."""
    parts = [x for x in Path(run_dir).parts if x not in ("..", ".", "/")]
    if parts and parts[0] == "runs":
        parts = parts[1:]
    return Path(out_root).joinpath(*parts) if parts else Path(out_root)


def _write_jsonl(path: Path, recs: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def reconstruct(p: Parsed, out_root: Path | str) -> list[Path]:
    """Write <out_root>/<run>/metrics.jsonl and eval.jsonl (rewritten each sync; idempotent)."""
    dirs = []
    for run_dir in sorted(set(p.metrics) | set(p.evals)):
        d = local_run_dir(out_root, run_dir)
        _write_jsonl(d / "metrics.jsonl", p.metrics.get(run_dir, []))
        _write_jsonl(d / "eval.jsonl", p.evals.get(run_dir, []))
        dirs.append(d)
    return dirs
