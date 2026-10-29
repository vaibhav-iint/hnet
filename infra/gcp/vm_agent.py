#!/usr/bin/env python3
"""On-VM agent for infra.gcp (stdlib only; shipped inside the startup script, imported locally only by
tests and for a few pure helpers).

Runs on every boot (the startup script starts it as a transient systemd unit). Stages, each idempotent;
`Agent.STAGES` lists them in order and each is the method of the same name, so adding a stage means
adding one method and one name:

  clone    git clone <repo> /opt/hnet at <commit> (skipped if already there); apply the job's overlay
  venv     /opt/hnet-venv with --system-site-packages (the image's CUDA torch) + requirements.txt
  kernels  prebuilt mamba_ssm / causal_conv1d wheels (each pip call capped at 10 min); the model code
           falls back to pure torch when they are absent (that fallback is by design, see SPEC.md)
  data     shards the command needs: bucket cache first (if any), then scripts/data/prepare.py, then
           new files are added to the bucket cache
  run      bucket checkpoint restore if the disk has none; the command under nohup; metrics mirrored
           to the serial console; run dirs rsynced to the bucket every few minutes

Failure policy: nothing is swallowed. Any error in a stage or a background thread (heartbeat, bucket
sync) ends the job: the command is stopped, the traceback goes to the serial console and agent.log,
all logs are uploaded to the bucket, HNET_DONE carries a non-zero code, the VM lingers then shuts down,
and the agent process exits non-zero. The only fallbacks are deliberate and logged: the serial device
(ttyS0 -> /dev/console -> stdout), creating the venv after installing python3-venv, and the optional
fast kernels. Bucket syncs retry SYNC_RETRIES times before failing the job.

Serial lines (port 1):
  HNET_STAGE  {"stage": ..., ...}
  HNET_STATUS <state> {"step": .., "gpu_util": .., "gpu_mem_mb": .., ...}   every 60 s
  HNET_INFO   {capabilities: torch, GPU, kernels}
  HNET_METRIC <run_dir> <json>      one per new line of <run_dir>/metrics.jsonl
  HNET_EVAL   <run_dir> <json>      one per new line of <run_dir>/eval.jsonl
  HNET_LOG    <text>                milestones, tracebacks, the last lines of the command's log
  HNET_SUMMARY {json}
  HNET_DONE   <exit code>
run_dir is relative to the repo root (spaces escaped as %20). Mirror offsets persist in
/opt/hnet-state/mirror.json so a reboot does not re-emit old lines.
"""

from __future__ import annotations

import base64
import glob
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request

STATE = os.environ.get("HNET_STATE_DIR", "/opt/hnet-state")
REPO = os.environ.get("HNET_REPO_DIR", "/opt/hnet")
VENV = os.environ.get("HNET_VENV_DIR", "/opt/hnet-venv")
SERIAL = os.environ.get("HNET_SERIAL_DEV", "/dev/ttyS0")
HB_SEC = float(os.environ.get("HNET_HEARTBEAT_SEC", "60"))
POLL_SEC = float(os.environ.get("HNET_MIRROR_POLL_SEC", "5"))
PIP_TIMEOUT = 600  # never compile/install a kernel for more than 10 min
SYNC_RETRIES = 3   # consecutive failed bucket syncs before the job fails
SYNC_RETRY_WAIT = 20.0
INFRA_FAILURE = 97  # HNET_DONE code when the agent (not the command) failed

DATASETS = {"finewebedu10B": 99, "fineweb10B": 103, "fineweb100B": 1030}
BASE_DEPS = ["numpy", "einops", "pytest", "pyyaml", "tokenizers", "safetensors", "tiktoken", "huggingface_hub",
             "packaging", "ninja", "setuptools", "wheel"]
LOG_FILES = ("agent.log", "agent_lines.log", "setup.log", "cmd.log", "gcs.log")


class StageError(RuntimeError):
    pass


class GcsError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- output
class Emitter:
    """Writes whole lines to the serial console and appends them to a log file.

    The device is opened for every line: serial-getty (agetty) calls vhangup() on ttyS0 when it starts,
    which revokes descriptors opened earlier (writes then fail with EIO)."""

    def __init__(self, dev: str = SERIAL, log_path: str | None = None):
        self.lock = threading.Lock()
        self.devs = [dev, "/dev/console"] if dev == SERIAL else [dev]
        self.log_path = log_path

    def _write_dev(self, data: bytes) -> bool:
        for d in self.devs:  # deliberate fallback chain; stdout (agent.log) is the last resort
            for _ in range(2):  # one reopen after EIO from a hangup
                try:
                    fd = os.open(d, os.O_WRONLY | os.O_NOCTTY | os.O_APPEND | os.O_NONBLOCK)
                except FileNotFoundError:
                    break
                try:
                    view, t0 = memoryview(data), time.time()
                    while view and time.time() - t0 < 10:
                        try:
                            view = view[os.write(fd, view):]
                        except BlockingIOError:
                            time.sleep(0.02)
                    if not view:
                        return True
                except OSError as e:
                    if e.errno != 5:  # EIO after a hangup -> reopen; anything else is a real error
                        raise
                finally:
                    os.close(fd)
        return False

    def emit(self, kind: str, payload: str = "") -> str:
        line = f"{kind} {payload}".replace("\r", " ").replace("\n", " ").rstrip()
        data = ("\n" + line + "\n").encode("utf-8", "replace")  # leading \n: never glue onto a console prompt
        with self.lock:
            if not self._write_dev(data):
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(time.strftime("%H:%M:%S ") + line + "\n")
        return line


def esc_dir(d: str) -> str:
    return d.replace("%", "%25").replace(" ", "%20")


def compact(line: str) -> str:
    """Re-serialize a JSON line without spaces; a non-JSON line is mirrored as is (the local parser
    counts it as corrupt, so it stays visible)."""
    try:
        return json.dumps(json.loads(line), separators=(",", ":"))
    except json.JSONDecodeError:
        return line.strip()


def guest_attr(key: str, value: str) -> None:
    """PUT a guest attribute under namespace hnet (metadata enable-guest-attributes=TRUE is always set)."""
    url = f"http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hnet/{key}"
    req = urllib.request.Request(url, data=value[:1000].encode(), method="PUT", headers={"Metadata-Flavor": "Google"})
    urllib.request.urlopen(req, timeout=10).read()


def gpu_stats() -> dict:
    if not shutil.which("nvidia-smi") or not os.path.exists("/dev/nvidia0"):
        return {}  # CPU machine (the GPU image ships nvidia-smi, which exits 9 without a device)
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30,
                         check=True).stdout
    rows = [[x.strip() for x in r.split(",")] for r in out.strip().splitlines() if r.strip()]
    if not rows:
        return {}
    return {"gpu_util": [int(float(r[0])) for r in rows], "gpu_mem_mb": [int(float(r[1])) for r in rows],
            "gpu_mem_total_mb": int(float(rows[0][2]))}


def tail_lines(path: str, n: int = 30, width: int = 300) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 64 * 1024))
        txt = f.read().decode("utf-8", "replace")
    lines = [ln.split("\r")[-1] for ln in txt.splitlines()]
    return [ln[:width] for ln in lines if ln.strip()][-n:]


# ----------------------------------------------------------------------------- metrics mirror
class Mirror:
    """Emit every new complete line of <run_dir>/{metrics,eval}.jsonl under the watched roots."""

    FILES = (("HNET_METRIC", "metrics.jsonl"), ("HNET_EVAL", "eval.jsonl"))

    def __init__(self, repo: str, emitter: Emitter, state_path: str, roots=("runs",), metrics_every: int = 1):
        self.repo, self.em, self.state_path = repo, emitter, state_path
        self.roots = list(roots)
        self.every = max(1, int(metrics_every))
        self.pos: dict = {}
        if os.path.exists(state_path):
            with open(state_path) as f:
                self.pos = json.load(f)
        self.last_step = None
        self.last_metric: dict = {}
        self.last_eval: dict = {}

    def _files(self):
        for kind, name in self.FILES:
            for root in self.roots:
                base = root if os.path.isabs(root) else os.path.join(self.repo, root)
                for p in sorted(glob.glob(os.path.join(base, "**", name), recursive=True)):
                    yield kind, p

    def scan(self) -> int:
        n_emitted, changed = 0, False
        for kind, path in self._files():
            try:
                st = os.stat(path)
            except FileNotFoundError:  # removed between glob and stat (fresh run replacing a dir)
                continue
            ent = self.pos.get(path) or {"ino": st.st_ino, "off": 0, "n": 0}
            if ent["ino"] != st.st_ino or st.st_size < ent["off"]:
                ent = {"ino": st.st_ino, "off": 0, "n": 0}  # replaced or truncated (fresh run)
            if st.st_size == ent["off"]:
                self.pos[path] = ent
                continue
            with open(path, "rb") as f:
                f.seek(ent["off"])
                buf = f.read()
            end = buf.rfind(b"\n")
            if end < 0:
                continue
            run_dir = esc_dir(os.path.relpath(os.path.dirname(path), self.repo))
            for raw in buf[: end + 1].splitlines():
                line = raw.decode("utf-8", "replace").strip()
                if not line:
                    continue
                ent["n"] += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    rec = None  # mirrored raw below; the local side reports it as corrupt
                if kind == "HNET_METRIC" and isinstance(rec, dict):
                    self.last_step = rec.get("step", self.last_step)
                    self.last_metric = rec
                    if (ent["n"] - 1) % self.every:
                        continue
                elif isinstance(rec, dict):
                    self.last_eval = rec
                self.em.emit(kind, f"{run_dir} {compact(line)}")
                n_emitted += 1
            ent["off"] += end + 1
            self.pos[path] = ent
            changed = True
        if changed:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.pos, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        return n_emitted


# ----------------------------------------------------------------------------- what the command needs
def _train_invocations(cmd: str) -> list[tuple[str, list[str]]]:
    """(config, [--set overrides]) for every `hnet.train.train` invocation in a shell command."""
    toks = shlex.split(cmd)
    seps = ("&&", "||", ";", "|")
    out = []
    i = 0
    while i < len(toks):
        if toks[i] not in ("hnet.train.train", "hnet/train/train.py"):
            i += 1
            continue
        cfg, sets = None, []
        j = i + 1
        while j < len(toks) and toks[j] not in seps:
            t = toks[j]
            if t == "--config" and j + 1 < len(toks):
                cfg = toks[j + 1]
                j += 2
                continue
            if t.startswith("--config="):
                cfg = t.split("=", 1)[1]
            elif t == "--set":
                j += 1
                while j < len(toks) and not toks[j].startswith("--") and toks[j] not in seps:
                    sets.append(toks[j])
                    j += 1
                continue
            j += 1
        if cfg:
            out.append((cfg, sets))
        i = j
    return out


def parse_data_spec(spec: str) -> dict[str, int]:
    """'finewebedu10B:1,fineweb10B:0' -> {dataset: n_train}; 'none' -> {}."""
    out: dict[str, int] = {}
    if not spec or spec in ("none", "auto"):
        return out
    for part in spec.split(","):
        ds, _, n = part.strip().partition(":")
        if ds:
            out[ds] = max(out.get(ds, 0), int(n) if n else DATASETS[ds])
    return out


def merge_needs(needs: list[dict]) -> dict[str, int]:
    """needs: [{"source", "dataset", "n_train_shards", "explicit_paths", "val"}] -> {dataset: n_train}."""
    out: dict[str, int] = {}
    for d in needs:
        if d.get("source", "hnet") == "hnet" and not d.get("explicit_paths"):
            ds = d.get("dataset") or "finewebedu10B"
            n = d.get("n_train_shards")
            n = DATASETS[ds] if n is None else int(n)
            out[ds] = max(out.get(ds, 0), n)
        val = d.get("val")
        if val == "standard":
            out.setdefault("finewebedu10B", 0)
        elif val == "speedrun":
            out.setdefault("fineweb10B", 0)
    return out


PROBE = r"""
import json, sys
from hnet.train.config import load_run_config
c = load_run_config(sys.argv[1], sys.argv[2:])
d, e = c.get("data", {}), c.get("eval", {})
print("HNET_NEEDS " + json.dumps({"source": d.get("source", "hnet"), "dataset": d.get("dataset"),
      "n_train_shards": d.get("n_train_shards"),
      "explicit_paths": bool(d.get("paths") or d.get("byte_paths") or d.get("token_paths")),
      "val": e.get("val") if (e.get("every") or e.get("at_end", True)) else None,
      "out_dir": c.get("train", {}).get("out_dir")}))
"""

KERNEL_PROBE = r"""
import json
import hnet.model.ssd as s, hnet.model.mixers.mamba2 as m, hnet.model.mixers.conv as c
print("HNET_KERNELS " + json.dumps({"ssd_triton": s._mamba_chunk_scan_combined is not None,
      "causal_conv1d": c._causal_conv1d_fn is not None,
      "fused_mamba2": m._split_conv1d_scan is not None and c._causal_conv1d_fn is not None}))
"""


# ----------------------------------------------------------------------------- bucket (optional)
SHARD_RE = re.compile(r"_(val|train)_(\d{6})\.")
CKPT_FILES = ("meta.json", "model.pt")  # a remote step dir is usable only if these are present
GCLOUD_CANDIDATES = ["/snap/bin/gcloud", "/usr/bin/gcloud", "/usr/lib/google-cloud-sdk/bin/gcloud",
                     "/opt/google-cloud-sdk/bin/gcloud", "/usr/local/bin/gcloud"]
NO_MATCH = "matched no objects"


def find_gcloud() -> str | None:
    for c in GCLOUD_CANDIDATES + [shutil.which("gcloud") or ""]:
        if c and os.path.exists(c):
            return c
    return None


def run_rel(out_dir: str) -> str:
    """Repo-relative run dir -> name under the bucket prefix (runs/tiny/x -> tiny/x)."""
    parts = [x for x in os.path.normpath(out_dir).split(os.sep) if x not in ("", ".", "..")]
    if parts and parts[0] == "runs":
        parts = parts[1:]
    return "/".join(parts) or "default"


def wanted_data_object(rel: str, n_train: int) -> bool:
    """Which cached objects a run with n_train train shards needs (non-shard files such as reports: always;
    hidden paths such as .cache/: never)."""
    if any(x.startswith(".") for x in rel.split("/")):
        return False
    m = SHARD_RE.search(os.path.basename(rel))
    if not m:
        return True
    return m.group(1) == "val" or int(m.group(2)) <= n_train


def latest_complete_ckpt(rels: list[str]) -> str | None:
    """From object paths relative to a run dir, the newest ckpt/step_* dir holding all CKPT_FILES."""
    steps: dict[str, set] = {}
    for r in rels:
        parts = r.split("/")
        if len(parts) >= 3 and parts[0] == "ckpt" and re.fullmatch(r"step_\d+", parts[1]):
            steps.setdefault(parts[1], set()).add("/".join(parts[2:]))
    ok = [k for k, v in steps.items() if all(f in v for f in CKPT_FILES)]
    return max(ok) if ok else None


class Gcs:
    """Bucket mirror of run dirs (gs://B/<prefix>/<run>/), logs (<prefix>/_logs/) and the shared
    data-shard cache (gs://B/<data_prefix>/<dataset>/). Every gcloud failure raises GcsError."""

    def __init__(self, cfg: dict | None):
        cfg = cfg or {}
        self.bucket = (cfg.get("bucket") or "").replace("gs://", "").strip("/")
        self.prefix = (cfg.get("prefix") or "runs/unnamed").strip("/")
        self.restore_prefix = (cfg.get("restore_prefix") or self.prefix).strip("/")
        self.data_prefix = (cfg.get("data_prefix") or "data").strip("/")
        self.every = float(cfg.get("sync_every_s", 300))
        self.gcloud = find_gcloud() if self.bucket else None
        self.log_path = os.path.join(STATE, "gcs.log")
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        return bool(self.bucket)

    def url(self, *parts: str) -> str:
        return "gs://" + "/".join([self.bucket] + [p.strip("/") for p in parts if p])

    def _exec(self, args: list[str], timeout: float) -> tuple[int, str, str]:
        """Run gcloud; (code, stdout, stderr). Tests replace this."""
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        env.setdefault("HOME", "/root")
        env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        r = subprocess.run([self.gcloud, *args], capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr

    def gc(self, args: list[str], timeout: float = 3600, allow_no_match: bool = False) -> str:
        if not self.gcloud:
            raise GcsError(f"bucket {self.bucket} configured but no gcloud on this VM ({GCLOUD_CANDIDATES})")
        code, out, err = self._exec(args, timeout)
        with open(self.log_path, "a") as f:
            f.write(f"\n$ gcloud {' '.join(args)}  -> {code}\n{err[-4000:]}\n")
        if code != 0:
            if allow_no_match and NO_MATCH in err.lower():
                return ""
            raise GcsError(f"gcloud {' '.join(args)[:300]} -> exit {code}: {err.strip()[-600:]}")
        return out

    def ls_sizes(self, url: str) -> dict[str, int]:
        """{object url: size} under url (recursive); {} when nothing is there yet."""
        out = self.gc(["storage", "ls", "-l", url.rstrip("/") + "/**"], timeout=600, allow_no_match=True)
        res = {}
        for ln in out.splitlines():
            f = ln.split()
            if len(f) >= 3 and f[-1].startswith("gs://") and f[0].isdigit():
                res[f[-1]] = int(f[0])
        return res

    # ---- data cache ----
    def restore_data(self, ds: str, n_train: int, root: str) -> int:
        base = self.url(self.data_prefix, ds) + "/"
        todo: dict[str, list[str]] = {}
        for u, size in self.ls_sizes(base).items():
            rel = u[len(base):]
            if not rel or rel.endswith("/") or not wanted_data_object(rel, n_train):
                continue
            dst = os.path.join(root, ds, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                continue
            todo.setdefault(os.path.dirname(dst), []).append(u)
        for d, urls in todo.items():
            os.makedirs(d, exist_ok=True)
            self.gc(["storage", "cp", *urls, d + "/"], timeout=3 * 3600)
        return sum(len(v) for v in todo.values())

    def upload_data(self, ds: str, root: str) -> int:
        base = self.url(self.data_prefix, ds) + "/"
        have = {u[len(base):]: s for u, s in self.ls_sizes(base).items()}
        local = os.path.join(root, ds)
        n = 0
        for dp, dirs, files in os.walk(local):
            dirs[:] = [d for d in dirs if not d.startswith(".")]  # e.g. .cache/huggingface download metadata
            for fn in files:
                if fn.startswith(".") or fn.endswith((".tmp", ".part", ".lock")):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, local).replace(os.sep, "/")
                if have.get(rel) == os.path.getsize(full):
                    continue
                self.gc(["storage", "cp", full, base + rel], timeout=3 * 3600)
                n += 1
        return n

    # ---- run dirs ----
    def run_dirs(self, roots: list[str]) -> list[str]:
        out = set()
        for root in roots:
            b = root if os.path.isabs(root) else os.path.join(REPO, root)
            for p in glob.glob(os.path.join(b, "**", "metrics.jsonl"), recursive=True):
                out.add(os.path.dirname(p))
        return sorted(out)

    def sync_runs(self, roots: list[str]) -> int:
        """rsync each run dir (ckpt/ mirrored with deletes so rolling checkpoints do not pile up) + logs."""
        with self.lock:
            dirs = self.run_dirs(roots)
            for d in dirs:
                dst = self.url(self.prefix, run_rel(os.path.relpath(d, REPO)))
                self.gc(["storage", "rsync", "-r", "--exclude=^ckpt/", d, dst], timeout=3600)
                ck = os.path.join(d, "ckpt")
                if glob.glob(os.path.join(ck, "step_*")):
                    self.gc(["storage", "rsync", "-r", "--delete-unmatched-destination-objects",
                             r"--exclude=(^|/)\.tmp_", ck, dst + "/ckpt"], timeout=6 * 3600)
            self.upload_logs()
        return len(dirs)

    def upload_logs(self):
        for f in LOG_FILES:
            fp = os.path.join(STATE, f)
            if os.path.exists(fp):
                self.gc(["storage", "cp", fp, self.url(self.prefix, "_logs", f)], timeout=600)

    def put_text(self, rel: str, text: str):
        fp = os.path.join(STATE, "_put.tmp")
        with open(fp, "w") as f:
            f.write(text)
        self.gc(["storage", "cp", fp, self.url(self.prefix, rel)], timeout=600)

    def restore_run(self, out_dir: str) -> str | None:
        """If the local run has no checkpoint, fetch the newest complete bucket checkpoint (+ jsonl/config)."""
        local = out_dir if os.path.isabs(out_dir) else os.path.join(REPO, out_dir)
        if glob.glob(os.path.join(local, "ckpt", "step_*")):
            return None
        base = self.url(self.restore_prefix, run_rel(out_dir)) + "/"
        rels = [u[len(base):] for u in self.ls_sizes(base)]
        step = latest_complete_ckpt(rels)
        if step is None:
            return None
        os.makedirs(os.path.join(local, "ckpt"), exist_ok=True)
        top = [base + r for r in rels if "/" not in r]
        if top:
            self.gc(["storage", "cp", *top, local + "/"], timeout=1800)
        self.gc(["storage", "cp", "-r", base + "ckpt/" + step, os.path.join(local, "ckpt") + "/"], timeout=6 * 3600)
        return step


# ----------------------------------------------------------------------------- agent
class Agent:
    STAGES = ("clone", "venv", "kernels", "data")  # then "run"; each name is a method below

    def __init__(self, job: dict, em: Emitter):
        self.job, self.em = job, em
        self.stage = "boot"
        self.mirror: Mirror | None = None
        self.t0 = time.time()
        self.info: dict = {}
        self.setup_log = os.path.join(STATE, "setup.log")
        self.cmd_log = os.path.join(STATE, "cmd.log")
        self.py = sys.executable
        self._stop = threading.Event()
        self.fatal: BaseException | None = None  # first error raised in a background thread
        self.gcs = Gcs(job.get("gcs"))
        self.out_dirs: list[str] = []
        self.roots = ["runs"] + list(job.get("watch_dirs") or [])
        self._probed: list[dict] | None = None

    # ---- helpers ----
    def log(self, msg: str):
        self.em.emit("HNET_LOG", msg[:400])

    def sh(self, args, timeout=None, env=None, cwd=None, check=True) -> int:
        """Run a command with output to setup.log. check=True raises StageError on failure/timeout;
        check=False is only for commands whose failure selects a deliberate fallback."""
        shown = args if isinstance(args, str) else " ".join(map(str, args))
        with open(self.setup_log, "a") as lf:
            lf.write(f"\n$ {shown}\n")
            lf.flush()
            try:
                code = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=cwd,
                                      shell=isinstance(args, str)).returncode
            except subprocess.TimeoutExpired as e:
                lf.write(f"[agent] timeout after {timeout}s\n")
                if check:
                    raise StageError(f"{shown[:200]} timed out after {timeout}s") from e
                code = 124
        if check and code != 0:
            raise StageError(f"{shown[:200]} -> exit {code} (see setup.log)")
        return code

    def out(self, args, timeout=600, env=None, cwd=None) -> str:
        """stdout of a command that must succeed (StageError otherwise)."""
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        with open(self.setup_log, "a") as lf:
            lf.write(f"\n$ {' '.join(args)}  -> {r.returncode}\n{r.stdout}{r.stderr}\n")
        if r.returncode != 0:
            raise StageError(f"{' '.join(args)[:200]} -> exit {r.returncode}: {r.stderr.strip()[-300:]}")
        return r.stdout

    def venv_env(self) -> dict:
        env = dict(os.environ)
        env.setdefault("HOME", "/root")
        if os.path.isdir(os.path.join(VENV, "bin")):
            env["PATH"] = f"{VENV}/bin:" + env.get("PATH", "/usr/bin:/bin")
            env["VIRTUAL_ENV"] = VENV
        env.update({"PYTHONUNBUFFERED": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1", "PIP_ROOT_USER_ACTION": "ignore",
                    "HF_HOME": env.get("HF_HOME", "/opt/hnet-cache/hf"), "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                    "TQDM_DISABLE": "1"})
        env.update({k: str(v) for k, v in (self.job.get("env") or {}).items()})
        return env

    def set_stage(self, name: str, **kw):
        self.stage = name
        self.em.emit("HNET_STAGE", json.dumps({"stage": name, "t": round(time.time() - self.t0, 1), **kw}))
        guest_attr("stage", name)

    def check_fatal(self):
        if self.fatal is not None:
            raise RuntimeError(f"background thread failed: {self.fatal!r}") from self.fatal

    def spawn(self, fn, name: str) -> threading.Thread:
        """Daemon thread whose first exception is recorded and fails the job (see check_fatal)."""
        def wrap():
            try:
                fn()
            except BaseException as e:  # handed to the main thread, which re-raises it
                if self.fatal is None:
                    self.fatal = e
                    with open(os.path.join(STATE, "agent.log"), "a") as f:
                        f.write(f"[{name}] " + traceback.format_exc())
        t = threading.Thread(target=wrap, name=name, daemon=True)
        t.start()
        return t

    # ---- heartbeat ----
    def status(self) -> dict:
        s = {"step": self.mirror.last_step if self.mirror else None, "uptime_s": round(time.time() - self.t0),
             **gpu_stats(), "disk_free_gb": round(shutil.disk_usage("/").free / 1e9, 1)}
        if self.stage == "run":
            t = tail_lines(self.cmd_log, 1, 160)
            if t:
                s["last"] = t[-1]
        return s

    def heartbeat_loop(self):
        while not self._stop.wait(HB_SEC):
            s = self.status()
            self.em.emit("HNET_STATUS", f"{self.stage} {json.dumps(s, separators=(',', ':'))}")
            guest_attr("status", json.dumps({"state": self.stage, "step": s.get("step"), "t": int(time.time())}))

    def gcs_loop(self):
        """Periodic bucket sync while the command runs; SYNC_RETRIES consecutive failures fail the job."""
        fails = 0
        while not self._stop.wait(self.gcs.every):
            if self.stage != "run":
                return
            try:
                n = self.gcs.sync_runs(self.roots)
                fails = 0
                guest_attr("gcs_sync", json.dumps({"t": int(time.time()), "runs": n}))
            except GcsError as e:
                fails += 1
                self.log(f"bucket sync failed ({fails}/{SYNC_RETRIES}): {e}")
                if fails >= SYNC_RETRIES:
                    raise

    # ---- stages ----
    def clone(self):
        self._clone()
        self.apply_overlay()

    def apply_overlay(self):
        """Local files shipped in the job (tar.gz, base64), e.g. untracked configs; re-applied every boot."""
        b64 = self.job.get("overlay_b64")
        if not b64:
            return
        n = 0
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz") as tf:
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    raise StageError(f"refusing unsafe overlay member {m.name!r}")
                tf.extract(m, REPO, filter="data")
                n += m.isfile()
        self.log(f"overlay: {n} files -> {REPO}")

    def _clone(self):
        repo, commit = self.job["repo"], self.job["commit"]
        if os.path.isdir(os.path.join(REPO, ".git")):
            head = self.out(["git", "-C", REPO, "rev-parse", "HEAD"]).strip()
            if head.startswith(commit) or commit in ("", "HEAD"):
                self.log(f"repo present at {head[:12]}; skip clone")
                return
            self.log(f"repo at {head[:12]} != {commit[:12]}; fetching")
            self.sh(["git", "-C", REPO, "fetch", "--all", "--tags"], timeout=600)
            self.sh(["git", "-C", REPO, "checkout", "-f", commit], timeout=120)
            return
        tmp = REPO + ".tmp"
        for attempt in range(3):  # retry transient network errors, then fail
            if os.path.exists(tmp):
                shutil.rmtree(tmp)
            if self.sh(["git", "clone", "--quiet", repo, tmp], timeout=900, check=attempt == 2) == 0:
                break
            time.sleep(10 * (attempt + 1))
        if commit and commit != "HEAD":
            self.sh(["git", "-C", tmp, "checkout", "--quiet", commit], timeout=120)
        os.replace(tmp, REPO)
        self.log(f"cloned {repo} at {self.out(['git', '-C', REPO, 'rev-parse', 'HEAD']).strip()[:12]}")

    def find_base_python(self) -> tuple[str, str]:
        """(python, torch version or '') preferring a python that already has torch."""
        r = subprocess.run(["bash", "-lc", "command -v python3 || command -v python"], capture_output=True,
                           text=True, timeout=60)
        login = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else ""
        cands = [self.job.get("python"), login, "/opt/conda/bin/python3", *sorted(glob.glob("/opt/*/bin/python3")),
                 *sorted(glob.glob("/opt/conda/envs/*/bin/python3")), "/usr/local/bin/python3", "/usr/bin/python3",
                 shutil.which("python3")]
        cands = [c for c in dict.fromkeys(cands) if c and os.path.exists(c)]
        if not cands:
            raise StageError("no python3 on this image")
        for c in cands:
            t = subprocess.run([c, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True,
                               timeout=300)
            if t.returncode == 0:
                return c, t.stdout.strip()
        return cands[0], ""

    def venv(self):
        ok = os.path.join(VENV, ".hnet_ok")
        if os.path.exists(ok):
            self.py = os.path.join(VENV, "bin", "python")
            self.log("venv present; skip")
            return
        base, tv = self.find_base_python()
        self.log(f"base python {base} torch {tv or 'none'}")
        if os.path.exists(VENV):
            shutil.rmtree(VENV)
        if self.sh([base, "-m", "venv", "--system-site-packages", VENV], timeout=600, check=False) != 0:
            self.log("venv needs python3-venv: installing it via apt")  # deliberate fallback (Ubuntu images)
            self.sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv",
                    timeout=900)
            if os.path.exists(VENV):
                shutil.rmtree(VENV)
            self.sh([base, "-m", "venv", "--system-site-packages", VENV], timeout=600)
        py = os.path.join(VENV, "bin", "python")
        env = self.venv_env()
        cons = os.path.join(STATE, "constraints.txt")
        with open(cons, "w") as f:
            if tv:
                f.write(f"torch=={tv}\n")  # never let pip replace the image's CUDA torch
        if not tv:  # no torch in the image: install the build matching the driver (slow, ~2.5 GB)
            idx = self.out([py, os.path.join(REPO, "infra", "prime", "gpu_caps.py"), "torch-index"], env=env).strip()
            self.log(f"no torch in the image; installing from {idx}")
            self.sh([py, "-m", "pip", "install", "-q", "torch", "--index-url", idx], timeout=1800, env=env)
        pip = [py, "-m", "pip", "install", "-q", "-c", cons]
        self.sh(pip + BASE_DEPS, timeout=1200, env=env)
        self.sh(pip + ["-r", os.path.join(REPO, "requirements.txt")], timeout=1800, env=env)
        with open(ok, "w") as f:
            f.write(time.strftime("%F %T"))
        self.py = py

    def kernels(self):
        """Optional fast kernels (prebuilt wheels only). Missing kernels are a supported configuration:
        the model code selects its pure-torch path; HNET_INFO reports which path is active."""
        env = self.venv_env()
        caps = os.path.join(REPO, "infra", "prime", "gpu_caps.py")
        ok = os.path.join(STATE, "kernels.done")
        cuda = subprocess.run([self.py, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                              env=env, timeout=300).returncode == 0
        if self.job.get("kernels", True) and cuda and not os.path.exists(ok):
            for pkg in ("causal_conv1d", "mamba_ssm"):
                self.install_kernel(pkg, caps, env)
            with open(ok, "w") as f:
                f.write("1")
        elif not cuda:
            self.log("no CUDA: fast kernels skipped")
        self.sh([self.py, caps, "summary", os.path.join(STATE, "caps.json")], timeout=300, env=env)
        with open(os.path.join(STATE, "caps.json")) as f:
            info = json.load(f)
        kout = self.out([self.py, "-c", KERNEL_PROBE], timeout=300, env=env, cwd=REPO)
        info["hnet_kernels"] = json.loads(kout.split("HNET_KERNELS ", 1)[1].splitlines()[0])
        self.info = info
        self.em.emit("HNET_INFO", json.dumps(info, separators=(",", ":")))

    def install_kernel(self, pkg: str, caps: str, env: dict):
        if subprocess.run([self.py, "-c", f"import {pkg}"], env=env, capture_output=True).returncode == 0:
            self.log(f"{pkg} already importable")
            return
        url = (self.out([self.py, caps, "wheel", pkg], timeout=180, env=env).strip().splitlines() or [""])[-1]
        if url.startswith("http"):
            code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-deps", url], timeout=PIP_TIMEOUT, env=env,
                           check=False)
            self.log(f"{pkg}: prebuilt wheel {os.path.basename(url)} -> {'ok' if code == 0 else f'exit {code}'}")
            if code == 0:
                return
        if pkg == "mamba_ssm":  # Triton kernels need no CUDA build
            code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps", "mamba-ssm"],
                           timeout=PIP_TIMEOUT, env=dict(env, MAMBA_SKIP_CUDA_BUILD="TRUE"), check=False)
            self.log(f"mamba_ssm: Triton-only install -> {'ok' if code == 0 else f'exit {code}; torch SSD path'}")
        elif self.job.get("allow_compile"):
            code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-build-isolation", pkg.replace("_", "-")],
                           timeout=PIP_TIMEOUT, env=dict(env, MAX_JOBS=str(os.cpu_count() or 4)), check=False)
            self.log(f"{pkg}: compile (<=10 min) -> {'ok' if code == 0 else f'exit {code}; torch path'}")
        else:
            self.log(f"{pkg}: no prebuilt wheel; the model uses its pure-torch path")

    def probe_configs(self) -> list[dict]:
        """Data/eval/out_dir facts of every `hnet.train.train` call in the command (composed configs)."""
        if self._probed is None:
            res = []
            for cfg, sets in _train_invocations(self.job["cmd"]):
                o = self.out([self.py, "-c", PROBE, cfg, *sets], timeout=300, env=self.venv_env(), cwd=REPO)
                res += [json.loads(ln.split(" ", 1)[1]) for ln in o.splitlines() if ln.startswith("HNET_NEEDS ")]
            self._probed = res
            self.out_dirs = [g["out_dir"] for g in res if g.get("out_dir")]
        return self._probed

    def data_needs(self) -> dict[str, int]:
        probed = self.probe_configs()
        spec = self.job.get("data", "auto")
        return merge_needs(probed) if spec == "auto" else parse_data_spec(spec)

    def data(self):
        needs = self.data_needs()
        self.log(f"data needs: {json.dumps(needs)}")
        env = self.venv_env()
        root = env.get("HNET_DATA") or os.path.join(REPO, "data")
        for ds, n in sorted(needs.items()):
            marker = os.path.join(STATE, f"data_{ds}_{n}.done")
            if os.path.exists(marker):
                self.log(f"data {ds} n_train={n}: on disk")
                continue
            t = time.time()
            if self.gcs.on:
                k = self.gcs.restore_data(ds, n, root)
                self.log(f"data {ds}: {k} files copied from the bucket cache in {time.time() - t:.0f}s")
            self.sh([self.py, "scripts/data/prepare.py", "--dataset", ds, "--n-train", str(n)], timeout=6 * 3600,
                    env=env, cwd=REPO)
            if self.gcs.on:
                t2 = time.time()
                k = self.gcs.upload_data(ds, root)
                self.log(f"data {ds}: {k} new files added to the bucket cache in {time.time() - t2:.0f}s")
            with open(marker, "w") as f:
                f.write("1")
            self.log(f"data {ds} n_train={n}: ready in {time.time() - t:.0f}s")

    def run(self) -> int:
        env = self.venv_env()
        self.mirror = Mirror(REPO, self.em, os.path.join(STATE, "mirror.json"), self.roots,
                             int(self.job.get("metrics_every", 1)))
        self.mirror.scan()  # lines written before a reboot but not yet emitted
        if self.gcs.on:
            self.probe_configs()
            for od in dict.fromkeys(self.out_dirs):
                step = self.gcs.restore_run(od)
                if step:
                    self.log(f"restored {od} from {self.gcs.url(self.gcs.restore_prefix, run_rel(od))} ({step})")
            self.spawn(self.gcs_loop, "gcs-sync")
        with open(self.cmd_log, "ab") as lf:
            lf.write(f"\n===== run {time.strftime('%F %T')} =====\n".encode())
            lf.flush()
            p = subprocess.Popen(["nohup", "bash", "-c", self.job["cmd"]], cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        with open(os.path.join(STATE, "cmd.pid"), "w") as f:
            f.write(str(p.pid))
        self.log(f"cmd started pid {p.pid}")
        try:
            while p.poll() is None:
                self.mirror.scan()
                self.check_fatal()
                time.sleep(POLL_SEC)
        finally:
            if p.poll() is None:  # the agent failed: stop the command (SIGTERM lets the trainer checkpoint)
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait()
        self.mirror.scan()
        return p.returncode

    # ---- end of job ----
    def summary(self, code: int, failed_stage: str | None) -> dict:
        m = self.mirror.last_metric if self.mirror else {}
        e = self.mirror.last_eval if self.mirror else {}
        keep = ("step", "tokens_eq", "bpb", "loss", "bytes_per_sec", "mfu", "mem_gb", "step_time")
        return {"exit": code, "failed_stage": failed_stage, "elapsed_s": round(time.time() - self.t0),
                "last_metric": {k: m[k] for k in keep if k in m},
                "last_eval": {k: v for k, v in e.items() if k in ("step", "tokens_eq", "val/bpb")},
                "kernels": self.info.get("hnet_kernels")}

    def final_sync(self, js: str):
        """Upload run dirs, all logs and the summary; retried, then raised."""
        for attempt in range(SYNC_RETRIES):
            try:
                t = time.time()
                n = self.gcs.sync_runs(self.roots)
                self.gcs.put_text("_summary.json", js)
                self.log(f"final bucket sync: {n} run dir(s) + logs -> {self.gcs.url(self.gcs.prefix)} "
                         f"in {time.time() - t:.0f}s")
                return
            except GcsError as e:
                self.log(f"final bucket sync failed ({attempt + 1}/{SYNC_RETRIES}): {e}")
                if attempt == SYNC_RETRIES - 1:
                    raise
                time.sleep(SYNC_RETRY_WAIT)

    def finish(self, code: int, failed_stage: str | None):
        self.stage = "done"
        for ln in tail_lines(self.cmd_log if failed_stage is None else self.setup_log, 30):
            self.em.emit("HNET_LOG", ln)
        js = json.dumps(self.summary(code, failed_stage), separators=(",", ":"))
        try:
            if self.gcs.on:
                self.final_sync(js)
        finally:  # the console always gets the outcome, even if the bucket upload failed
            self.em.emit("HNET_SUMMARY", js)
            self.em.emit("HNET_DONE", str(code))
            guest_attr("summary", js)
            guest_attr("exit", str(code))

    def pipeline(self) -> int:
        for name in self.STAGES:
            self.set_stage(name)
            getattr(self, name)()
            self.check_fatal()
        self.set_stage("run")
        return self.run()

    def report_exception(self, exc: BaseException):
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        with open(os.path.join(STATE, "agent.log"), "a") as f:
            f.write(tb)
        self.log(f"stage {self.stage} failed: {exc!r}")
        for ln in tb.strip().splitlines()[-25:]:
            self.em.emit("HNET_LOG", ln[:300])

    def main(self) -> int:
        """Run the job; returns the exit code (0 only if the command and the agent both succeeded).
        Infra exceptions are re-raised after reporting, lingering and shutting down."""
        os.makedirs(STATE, exist_ok=True)
        self.spawn(self.heartbeat_loop, "heartbeat")
        done_path = os.path.join(STATE, "done.json")
        boots_path = os.path.join(STATE, "boots")
        boots = int(open(boots_path).read()) if os.path.exists(boots_path) else 0
        with open(boots_path, "w") as f:
            f.write(str(boots + 1))
        error: BaseException | None = None
        code = 0
        try:
            self.set_stage("boot", boot=boots + 1, commit=self.job.get("commit"),
                           gcs=self.gcs.url(self.gcs.prefix) if self.gcs.on else None)
            prev = json.load(open(done_path)) if os.path.exists(done_path) else None
            if prev is not None and prev.get("nonce") == self.job.get("nonce"):
                self.log(f"already finished (exit {prev.get('exit')}); not re-running")
                for ln in tail_lines(self.cmd_log if prev.get("failed_stage") is None else self.setup_log, 30):
                    self.em.emit("HNET_LOG", ln)
                code = int(prev.get("exit") or 0)
                self.em.emit("HNET_DONE", str(code))
            else:
                if prev is not None:
                    self.log("new job nonce: re-running (venv/data/checkpoints on disk are reused)")
                failed = None
                try:
                    code = self.pipeline()
                except BaseException as e:  # reported below and re-raised at the end of main()
                    error, failed, code = e, self.stage, INFRA_FAILURE
                    self.report_exception(e)
                with open(done_path, "w") as f:
                    json.dump({"exit": code, "failed_stage": failed, "at": time.time(),
                               "nonce": self.job.get("nonce")}, f)
                self.finish(code, failed)
        except BaseException as e:  # failure while finishing/reporting: still linger and shut down
            if error is None:
                error, code = e, INFRA_FAILURE
                self.report_exception(e)
        finally:
            self.stage = "done"
            linger = float(self.job.get("linger_min", 5))
            if linger > 0:
                self.log(f"lingering {linger:g} min so the serial output can be read")
                self._stop.wait(linger * 60)
            self._stop.set()
            if self.job.get("auto_shutdown", True):
                self.em.emit("HNET_STAGE", json.dumps({"stage": "shutdown"}))
                subprocess.run(["shutdown", "-h", "now"], check=True)
        if error is not None:
            raise error
        return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    job_path = argv[0] if argv else os.path.join(STATE, "job.json")
    with open(job_path) as f:
        job = json.load(f)
    os.makedirs(STATE, exist_ok=True)
    em = Emitter(log_path=os.path.join(STATE, "agent_lines.log"))
    return Agent(job, em).main()


if __name__ == "__main__":
    sys.exit(main())
