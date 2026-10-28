#!/usr/bin/env python3
"""On-VM agent for infra.gcp (stdlib only; shipped inside the startup script, never imported locally
except by tests).

Runs on every boot (the startup script starts it as a transient systemd unit). Steps, each idempotent:

  clone    git clone <repo> /opt/hnet and check out <commit>         (skipped if already at the commit)
  venv     /opt/hnet-venv with --system-site-packages (the image's CUDA torch), requirements.txt
  kernels  prebuilt mamba_ssm / causal_conv1d wheels only (each pip call capped at 10 min), else torch fallback
  data     scripts/data/prepare.py for just the datasets/shards the command needs (cached on the disk)
  run      the command under nohup in /opt/hnet; trainer resume=auto picks up checkpoints after a restart
  finish   HNET_DONE, guest attributes (namespace hnet), linger, `shutdown -h now`

Everything the local side needs goes to serial port 1 (/dev/ttyS0) as single lines:

  HNET_STAGE  {"stage": ..., ...}
  HNET_STATUS <state> {"step": .., "gpu_util": .., "gpu_mem_mb": .., ...}   every 60 s
  HNET_INFO   {capabilities: torch, GPU, kernels}
  HNET_METRIC <run_dir> <json>      one per new line of <run_dir>/metrics.jsonl
  HNET_EVAL   <run_dir> <json>      one per new line of <run_dir>/eval.jsonl
  HNET_LOG    <text>                setup milestones, error tails, last lines of the command's log
  HNET_SUMMARY {json}
  HNET_DONE   <exit code>

run_dir is relative to the repo root (spaces escaped as %20). Mirror offsets persist in
/opt/hnet-state/mirror.json so a reboot does not re-emit old lines.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

STATE = os.environ.get("HNET_STATE_DIR", "/opt/hnet-state")
REPO = os.environ.get("HNET_REPO_DIR", "/opt/hnet")
VENV = os.environ.get("HNET_VENV_DIR", "/opt/hnet-venv")
SERIAL = os.environ.get("HNET_SERIAL_DEV", "/dev/ttyS0")
HB_SEC = float(os.environ.get("HNET_HEARTBEAT_SEC", "60"))
POLL_SEC = float(os.environ.get("HNET_MIRROR_POLL_SEC", "5"))
PIP_TIMEOUT = 600  # never compile/install a kernel for more than 10 min

DATASETS = {"finewebedu10B": 99, "fineweb10B": 103, "fineweb100B": 1030}
BASE_DEPS = ["numpy", "einops", "pytest", "pyyaml", "tokenizers", "safetensors", "tiktoken", "huggingface_hub",
             "packaging", "ninja", "setuptools", "wheel"]


# ----------------------------------------------------------------------------- output
class Emitter:
    """Writes whole lines to the serial console and to the agent log.

    The device is opened for every line: serial-getty (agetty) calls vhangup() on ttyS0 when it starts,
    which revokes descriptors opened earlier (writes then fail with EIO)."""

    def __init__(self, dev: str = SERIAL, log_path: str | None = None):
        self.lock = threading.Lock()
        self.devs = [dev, "/dev/console"] if dev == SERIAL else [dev]
        self.log_path = log_path
        self.failures = 0

    def _write_dev(self, data: bytes) -> bool:
        for d in self.devs:
            for _ in range(2):
                try:
                    fd = os.open(d, os.O_WRONLY | os.O_NOCTTY | os.O_APPEND | os.O_NONBLOCK)
                except OSError:
                    break
                try:
                    view = memoryview(data)
                    t0 = time.time()
                    while view and time.time() - t0 < 10:
                        try:
                            n = os.write(fd, view)
                            view = view[n:]
                        except BlockingIOError:
                            time.sleep(0.02)
                    if not view:
                        return True
                except OSError:
                    pass  # EIO after a hangup: reopen once
                finally:
                    os.close(fd)
        return False

    def emit(self, kind: str, payload: str = "") -> str:
        line = f"{kind} {payload}".replace("\r", " ").replace("\n", " ").rstrip()
        data = ("\n" + line + "\n").encode("utf-8", "replace")  # leading \n: never glue onto a console prompt
        with self.lock:
            if not self._write_dev(data):
                self.failures += 1
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
            if self.log_path:
                try:
                    with open(self.log_path, "a") as f:
                        f.write(time.strftime("%H:%M:%S ") + line + "\n")
                except OSError:
                    pass
        return line


def esc_dir(d: str) -> str:
    return d.replace("%", "%25").replace(" ", "%20")


def compact(line: str) -> str:
    try:
        return json.dumps(json.loads(line), separators=(",", ":"))
    except Exception:
        return line.strip()


def guest_attr(key: str, value: str) -> bool:
    """PUT a guest attribute under namespace hnet (needs metadata enable-guest-attributes=TRUE)."""
    url = f"http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/hnet/{key}"
    req = urllib.request.Request(url, data=value[:1000].encode(), method="PUT", headers={"Metadata-Flavor": "Google"})
    try:
        urllib.request.urlopen(req, timeout=5).read()
        return True
    except Exception:
        return False


def gpu_stats() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        rows = [[x.strip() for x in r.split(",")] for r in out.strip().splitlines() if r.strip()]
        if not rows:
            return {}
        return {"gpu_util": [int(float(r[0])) for r in rows], "gpu_mem_mb": [int(float(r[1])) for r in rows],
                "gpu_mem_total_mb": int(float(rows[0][2]))}
    except Exception:
        return {}


def tail_lines(path: str, n: int = 30, width: int = 300) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 64 * 1024))
            txt = f.read().decode("utf-8", "replace")
    except OSError:
        return []
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
        try:
            with open(state_path) as f:
                self.pos = json.load(f)
        except Exception:
            self.pos = {}
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
            except OSError:
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
                except Exception:
                    rec = None
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


# ----------------------------------------------------------------------------- data needs
def _train_invocations(cmd: str) -> list[tuple[str, list[str]]]:
    """(config, [--set overrides]) for every `hnet.train.train` invocation in a shell command."""
    try:
        toks = shlex.split(cmd)
    except ValueError:
        toks = cmd.split()
    out = []
    i = 0
    while i < len(toks):
        if toks[i] in ("hnet.train.train", "hnet/train/train.py"):
            cfg, sets = None, []
            j = i + 1
            while j < len(toks) and toks[j] not in ("&&", "||", ";", "|"):
                t = toks[j]
                if t == "--config" and j + 1 < len(toks):
                    cfg = toks[j + 1]
                    j += 2
                    continue
                if t.startswith("--config="):
                    cfg = t.split("=", 1)[1]
                elif t == "--set":
                    j += 1
                    while j < len(toks) and not toks[j].startswith("--") and toks[j] not in ("&&", "||", ";", "|"):
                        sets.append(toks[j])
                        j += 1
                    continue
                j += 1
            if cfg:
                out.append((cfg, sets))
            i = j
        else:
            i += 1
    return out


def parse_data_spec(spec: str) -> dict[str, int]:
    """'finewebedu10B:1,fineweb10B:0' -> {dataset: n_train}; 'none' -> {}."""
    out: dict[str, int] = {}
    if not spec or spec in ("none", "auto"):
        return out
    for part in spec.split(","):
        ds, _, n = part.strip().partition(":")
        if ds:
            out[ds] = max(out.get(ds, 0), int(n) if n else DATASETS.get(ds, 0))
    return out


def merge_needs(needs: list[dict]) -> dict[str, int]:
    """needs: [{"source", "dataset", "n_train_shards", "explicit_paths", "val"}] -> {dataset: n_train}."""
    out: dict[str, int] = {}
    for d in needs:
        if d.get("source", "hnet") == "hnet" and not d.get("explicit_paths"):
            ds = d.get("dataset") or "finewebedu10B"
            n = d.get("n_train_shards")
            n = DATASETS.get(ds, 0) if n is None else int(n)
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
r = {}
try:
    import hnet.model.ssd as s, hnet.model.mixers.mamba2 as m, hnet.model.mixers.conv as c
    r = {"ssd_triton": s._mamba_chunk_scan_combined is not None,
         "causal_conv1d": c._causal_conv1d_fn is not None,
         "fused_mamba2": m._split_conv1d_scan is not None and c._causal_conv1d_fn is not None}
except Exception as ex:
    r = {"error": repr(ex)[:200]}
print("HNET_KERNELS " + json.dumps(r))
"""


# ----------------------------------------------------------------------------- GCS (optional)
SHARD_RE = re.compile(r"_(val|train)_(\d{6})\.")
CKPT_FILES = ("meta.json", "model.pt")  # a remote step dir is usable only if these are present
GCLOUD_CANDIDATES = ["/snap/bin/gcloud", "/usr/bin/gcloud", "/usr/lib/google-cloud-sdk/bin/gcloud",
                     "/opt/google-cloud-sdk/bin/gcloud", "/usr/local/bin/gcloud"]


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
    """Which cached objects a run with n_train train shards needs (non-shard files, e.g. reports: always)."""
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
    """Bucket mirror of run dirs + shared data-shard cache. Every call is best effort (logged, never fatal)."""

    def __init__(self, cfg: dict | None, agent: "Agent"):
        cfg = cfg or {}
        self.bucket = (cfg.get("bucket") or "").replace("gs://", "").strip("/")
        self.prefix = (cfg.get("prefix") or "runs/unnamed").strip("/")
        self.restore_prefix = (cfg.get("restore_prefix") or self.prefix).strip("/")
        self.data_prefix = (cfg.get("data_prefix") or "data").strip("/")
        self.every = float(cfg.get("sync_every_s", 300))
        self.agent = agent
        self.gcloud = find_gcloud() if self.bucket else None
        self.log_path = os.path.join(STATE, "gcs.log")
        self.lock = threading.Lock()

    @property
    def on(self) -> bool:
        return bool(self.bucket and self.gcloud)

    def url(self, *parts: str) -> str:
        return "gs://" + "/".join([self.bucket] + [p.strip("/") for p in parts if p])

    def gc(self, args: list[str], timeout: float = 3600) -> tuple[int, str]:
        env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
        env.setdefault("HOME", "/root")
        env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
        try:
            r = subprocess.run([self.gcloud, *args], capture_output=True, text=True, timeout=timeout, env=env)
            code, out, err = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            code, out, err = 124, "", "timeout"
        except OSError as e:
            code, out, err = 127, "", repr(e)
        with open(self.log_path, "a") as f:
            f.write(f"\n$ gcloud {' '.join(args)}  -> {code}\n{err[-2000:]}\n")
        return code, out

    def ls_sizes(self, url: str) -> dict[str, int]:
        """{object url: size} under url (recursive); {} if none / error."""
        code, out = self.gc(["storage", "ls", "-l", url.rstrip("/") + "/**"], timeout=600)
        res = {}
        if code != 0:
            return res
        for ln in out.splitlines():
            f = ln.split()
            if len(f) >= 3 and f[-1].startswith("gs://") and f[0].isdigit():
                res[f[-1]] = int(f[0])
        return res

    # ---- data cache ----
    def restore_data(self, ds: str, n_train: int, root: str) -> int:
        base = self.url(self.data_prefix, ds) + "/"
        objs = self.ls_sizes(base)
        todo: dict[str, list[str]] = {}
        for u, size in objs.items():
            rel = u[len(base):]
            if not rel or rel.endswith("/") or not wanted_data_object(rel, n_train):
                continue
            dst = os.path.join(root, ds, rel)
            if os.path.exists(dst) and os.path.getsize(dst) == size:
                continue
            todo.setdefault(os.path.dirname(dst), []).append(u)
        n = 0
        for d, urls in todo.items():
            os.makedirs(d, exist_ok=True)
            code, _ = self.gc(["storage", "cp", *urls, d + "/"], timeout=3 * 3600)
            if code == 0:
                n += len(urls)
        return n

    def upload_data(self, ds: str, root: str) -> int:
        base = self.url(self.data_prefix, ds) + "/"
        have = {u[len(base):]: s for u, s in self.ls_sizes(base).items()}
        local = os.path.join(root, ds)
        n = 0
        for dp, _, files in os.walk(local):
            for fn in files:
                if fn.startswith(".") or fn.endswith((".tmp", ".part", ".lock")):
                    continue
                full = os.path.join(dp, fn)
                rel = os.path.relpath(full, local).replace(os.sep, "/")
                if have.get(rel) == os.path.getsize(full):
                    continue
                if self.gc(["storage", "cp", full, base + rel], timeout=3 * 3600)[0] == 0:
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
        """rsync each run dir (ckpt/ mirrored with deletes so rolling checkpoints do not pile up)."""
        if not self.on:
            return 0
        n = 0
        with self.lock:
            for d in self.run_dirs(roots):
                dst = self.url(self.prefix, run_rel(os.path.relpath(d, REPO)))
                c1, _ = self.gc(["storage", "rsync", "-r", "--exclude=^ckpt/", d, dst], timeout=3600)
                ck = os.path.join(d, "ckpt")
                c2 = 0
                if glob.glob(os.path.join(ck, "step_*")):
                    c2, _ = self.gc(["storage", "rsync", "-r", "--delete-unmatched-destination-objects",
                                     r"--exclude=(^|/)\.tmp_", ck, dst + "/ckpt"], timeout=6 * 3600)
                n += int(c1 == 0 and c2 == 0)
            for f in ("cmd.log", "setup.log", "agent_lines.log"):
                fp = os.path.join(STATE, f)
                if os.path.exists(fp):
                    self.gc(["storage", "cp", fp, self.url(self.prefix, "_logs", f)], timeout=600)
        return n

    def put_text(self, rel: str, text: str):
        if not self.on:
            return
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
        objs = self.ls_sizes(base)
        rels = [u[len(base):] for u in objs]
        step = latest_complete_ckpt(rels)
        if step is None:
            return None
        os.makedirs(os.path.join(local, "ckpt"), exist_ok=True)
        top = [base + r for r in rels if "/" not in r]
        if top:
            self.gc(["storage", "cp", *top, local + "/"], timeout=1800)
        code, _ = self.gc(["storage", "cp", "-r", base + "ckpt/" + step, os.path.join(local, "ckpt") + "/"],
                          timeout=6 * 3600)
        return step if code == 0 else None


# ----------------------------------------------------------------------------- agent
class StageError(RuntimeError):
    pass


class Agent:
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
        self.gcs = Gcs(job.get("gcs"), self)
        self.out_dirs: list[str] = []

    # ---- helpers ----
    def log(self, msg: str):
        self.em.emit("HNET_LOG", msg[:400])

    def sh(self, args, timeout=None, env=None, cwd=None, check=True) -> int:
        with open(self.setup_log, "a") as lf:
            lf.write(f"\n$ {' '.join(map(str, args)) if isinstance(args, list) else args}\n")
            lf.flush()
            try:
                r = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout, env=env, cwd=cwd,
                                   shell=isinstance(args, str))
                code = r.returncode
            except subprocess.TimeoutExpired:
                lf.write(f"[agent] timeout after {timeout}s\n")
                code = 124
        if check and code != 0:
            raise StageError(f"{args if isinstance(args, str) else ' '.join(map(str, args))[:200]} -> {code}")
        return code

    def out(self, args, timeout=600, env=None, cwd=None) -> str:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        with open(self.setup_log, "a") as lf:
            lf.write(f"\n$ {' '.join(args)}\n{r.stdout}{r.stderr}\n")
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

    # ---- heartbeat ----
    def status(self) -> dict:
        s = {"step": self.mirror.last_step if self.mirror else None, "uptime_s": round(time.time() - self.t0),
             **gpu_stats()}
        try:
            du = shutil.disk_usage("/")
            s["disk_free_gb"] = round(du.free / 1e9, 1)
        except OSError:
            pass
        if self.stage == "run":
            t = tail_lines(self.cmd_log, 1, 160)
            if t:
                s["last"] = t[-1]
        return s

    def heartbeat_loop(self):
        while not self._stop.wait(HB_SEC):
            try:
                s = self.status()
                self.em.emit("HNET_STATUS", f"{self.stage} {json.dumps(s, separators=(',', ':'))}")
                guest_attr("status", json.dumps({"state": self.stage, "step": s.get("step"), "t": int(time.time())}))
            except Exception as e:  # never let the heartbeat die
                self.em.emit("HNET_LOG", f"heartbeat error {e!r}"[:200])

    # ---- stages ----
    def clone(self):
        self._clone()
        self.apply_overlay()

    def apply_overlay(self):
        """Local files shipped in the job (tar.gz, base64), e.g. untracked configs; re-applied every boot."""
        b64 = self.job.get("overlay_b64")
        if not b64:
            return
        import base64
        import io
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(b64)), mode="r:gz") as tf:
            names = []
            for m in tf.getmembers():
                if m.name.startswith("/") or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    continue
                try:
                    tf.extract(m, REPO, filter="data")
                except TypeError:  # python without extraction filters
                    tf.extract(m, REPO)
                names.append(m.name)
        self.log(f"overlay: {len([n for n in names if not n.endswith('/')])} entries -> {REPO}")

    def _clone(self):
        repo, commit = self.job["repo"], self.job["commit"]
        if os.path.isdir(os.path.join(REPO, ".git")):
            head = self.out(["git", "-C", REPO, "rev-parse", "HEAD"]).strip()
            if head.startswith(commit) or commit in ("", "HEAD"):
                self.log(f"repo present at {head[:12]}; skip clone")
                return
            self.log(f"repo at {head[:12]} != {commit[:12]}; fetching")
            self.sh(["git", "-C", REPO, "fetch", "--all", "--tags"], timeout=600, check=False)
            self.sh(["git", "-C", REPO, "checkout", "-f", commit], timeout=120)
            return
        tmp = REPO + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        for attempt in range(3):
            if self.sh(["git", "clone", "--quiet", repo, tmp], timeout=900, check=False) == 0:
                break
            shutil.rmtree(tmp, ignore_errors=True)
            time.sleep(10 * (attempt + 1))
        else:
            raise StageError(f"git clone {repo} failed")
        if commit and commit != "HEAD":
            self.sh(["git", "-C", tmp, "checkout", "--quiet", commit], timeout=120)
        os.replace(tmp, REPO)
        head = self.out(["git", "-C", REPO, "rev-parse", "HEAD"]).strip()
        self.log(f"cloned {repo} at {head[:12]}")

    def find_base_python(self) -> str:
        login = ""
        try:  # the login shell's PATH may point at the image's own python environment
            login = subprocess.run(["bash", "-lc", "command -v python3 || command -v python"], capture_output=True,
                                   text=True, timeout=60).stdout.strip().splitlines()[-1:]
            login = login[0] if login else ""
        except Exception:
            login = ""
        cands = [self.job.get("python"), login, "/opt/conda/bin/python3", "/opt/conda/bin/python",
                 *sorted(glob.glob("/opt/*/bin/python3")), *sorted(glob.glob("/opt/conda/envs/*/bin/python3")),
                 "/usr/local/bin/python3", "/usr/bin/python3", shutil.which("python3"), shutil.which("python")]
        for c in cands:
            if not c or not os.path.exists(c):
                continue
            r = subprocess.run([c, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True,
                               timeout=300)
            if r.returncode == 0:
                return c
        for c in cands:
            if c and os.path.exists(c):
                return c
        raise StageError("no python found")

    def venv(self):
        ok = os.path.join(VENV, ".hnet_ok")
        if os.path.exists(ok):
            self.py = os.path.join(VENV, "bin", "python")
            self.log("venv present; skip")
            return
        base = self.find_base_python()
        tv = subprocess.run([base, "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True,
                            timeout=300).stdout.strip()
        self.log(f"base python {base} torch {tv or 'none'}")
        shutil.rmtree(VENV, ignore_errors=True)
        if self.sh([base, "-m", "venv", "--system-site-packages", VENV], timeout=600, check=False) != 0:
            self.log("venv failed; installing python3-venv via apt")
            self.sh("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv",
                    timeout=900, check=False)
            shutil.rmtree(VENV, ignore_errors=True)
            if self.sh([base, "-m", "venv", "--system-site-packages", VENV], timeout=600, check=False) != 0:
                raise StageError("could not create venv")
        py = os.path.join(VENV, "bin", "python")
        env = self.venv_env()
        cons = os.path.join(STATE, "constraints.txt")
        with open(cons, "w") as f:
            if tv:
                f.write(f"torch=={tv}\n")  # never let pip replace the image's CUDA torch
        pip = [py, "-m", "pip", "install", "-q", "-c", cons]
        if not tv:  # no torch in the image: install the build matching the driver (slow, ~2.5 GB)
            caps = os.path.join(REPO, "infra", "prime", "gpu_caps.py")
            idx = self.out([py, caps, "torch-index"], timeout=120, env=env).strip() if os.path.exists(caps) else ""
            self.log(f"no torch in the image; installing from {idx or 'PyPI'}")
            self.sh([py, "-m", "pip", "install", "-q", "torch"] + (["--index-url", idx] if idx else []),
                    timeout=1800, env=env)
        self.sh(pip + BASE_DEPS, timeout=1200, env=env)
        req = os.path.join(REPO, "requirements.txt")
        if os.path.exists(req):
            self.sh(pip + ["-r", req], timeout=1800, env=env)
        open(ok, "w").write(time.strftime("%F %T"))
        self.py = py

    def kernels(self):
        ok = os.path.join(STATE, "kernels.done")
        env = self.venv_env()
        if self.job.get("kernels", True) and not os.path.exists(ok):
            caps = os.path.join(REPO, "infra", "prime", "gpu_caps.py")
            cuda = subprocess.run([self.py, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                                  env=env, timeout=300).returncode == 0
            if not cuda:
                self.log("no CUDA torch: skipping fast kernels")
            for pkg in (("causal_conv1d", "mamba_ssm") if cuda else ()):
                if subprocess.run([self.py, "-c", f"import {pkg}"], env=env, capture_output=True).returncode == 0:
                    self.log(f"{pkg} already importable")
                    continue
                url = ""
                if os.path.exists(caps):
                    try:
                        url = self.out([self.py, caps, "wheel", pkg], timeout=120, env=env).strip().splitlines()[-1:]
                        url = url[0] if url else ""
                    except Exception:
                        url = ""
                if url.startswith("http"):
                    code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-deps", url], timeout=PIP_TIMEOUT,
                                   env=env, check=False)
                    self.log(f"{pkg}: prebuilt wheel {os.path.basename(url)} -> {'ok' if code == 0 else code}")
                    if code == 0:
                        continue
                if pkg == "mamba_ssm":
                    e2 = dict(env, MAMBA_SKIP_CUDA_BUILD="TRUE")
                    code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-build-isolation", "--no-deps",
                                    "mamba-ssm"], timeout=PIP_TIMEOUT, env=e2, check=False)
                    self.log(f"mamba_ssm: no wheel; Triton-only install -> {'ok' if code == 0 else code}")
                elif self.job.get("allow_compile"):
                    e2 = dict(env, MAX_JOBS=str(os.cpu_count() or 4))
                    code = self.sh([self.py, "-m", "pip", "install", "-q", "--no-build-isolation",
                                    pkg.replace("_", "-")], timeout=PIP_TIMEOUT, env=e2, check=False)
                    self.log(f"{pkg}: compile (<=10 min) -> {'ok' if code == 0 else code}")
                else:
                    self.log(f"{pkg}: no prebuilt wheel; using the pure-torch fallback")
            open(ok, "w").write("1")
        info = {}
        caps = os.path.join(REPO, "infra", "prime", "gpu_caps.py")
        if os.path.exists(caps):
            self.sh([self.py, caps, "summary", os.path.join(STATE, "caps.json")], timeout=300, env=env, check=False)
            try:
                info = json.load(open(os.path.join(STATE, "caps.json")))
            except Exception:
                info = {}
        kout = self.out([self.py, "-c", KERNEL_PROBE], timeout=300, env=env, cwd=REPO)
        for ln in kout.splitlines():
            if ln.startswith("HNET_KERNELS "):
                info["hnet_kernels"] = json.loads(ln.split(" ", 1)[1])
        self.info = info
        self.em.emit("HNET_INFO", json.dumps(info, separators=(",", ":")))

    def probe_configs(self) -> list[dict]:
        """Data/eval/out_dir facts of every `hnet.train.train` call in the command (composed configs)."""
        if getattr(self, "_probed", None) is not None:
            return self._probed
        res = []
        env = self.venv_env()
        for cfg, sets in _train_invocations(self.job["cmd"]):
            o = self.out([self.py, "-c", PROBE, cfg, *sets], timeout=300, env=env, cwd=REPO)
            got = [json.loads(ln.split(" ", 1)[1]) for ln in o.splitlines() if ln.startswith("HNET_NEEDS ")]
            if not got:
                raise StageError(f"could not read the config {cfg} (see setup.log)")
            res += got
        self._probed = res
        self.out_dirs = [g["out_dir"] for g in res if g.get("out_dir")]
        return res

    def data_needs(self) -> dict[str, int]:
        probed = self.probe_configs()
        spec = self.job.get("data", "auto")
        if spec != "auto":
            return parse_data_spec(spec)
        return merge_needs(probed)

    def data(self):
        needs = self.data_needs()
        self.log(f"data needs: {json.dumps(needs)}")
        if not needs:
            self.log("data: nothing to prepare")
            return
        env = self.venv_env()
        root = env.get("HNET_DATA") or os.path.join(REPO, "data")
        for ds, n in sorted(needs.items()):
            marker = os.path.join(STATE, f"data_{ds}_{n}.done")
            if os.path.exists(marker):
                self.log(f"data {ds} n_train={n}: cached")
                continue
            t = time.time()
            if self.gcs.on:
                k = self.gcs.restore_data(ds, n, root)
                self.log(f"data {ds}: {k} files copied from the bucket cache in {time.time() - t:.0f}s")
            self.log(f"data {ds} n_train={n}: prepare.py (skips valid shards)")
            self.sh([self.py, "scripts/data/prepare.py", "--dataset", ds, "--n-train", str(n)], timeout=6 * 3600,
                    env=env, cwd=REPO)
            if self.gcs.on:
                t2 = time.time()
                k = self.gcs.upload_data(ds, root)
                self.log(f"data {ds}: {k} new files uploaded to the bucket cache in {time.time() - t2:.0f}s")
            open(marker, "w").write("1")
            self.log(f"data {ds} n_train={n}: done in {time.time() - t:.0f}s")

    def run_cmd(self) -> int:
        env = self.venv_env()
        roots = ["runs"] + list(self.job.get("watch_dirs") or [])
        self.mirror = Mirror(REPO, self.em, os.path.join(STATE, "mirror.json"), roots,
                             int(self.job.get("metrics_every", 1)))
        self.mirror.scan()  # lines written before a reboot but not yet emitted
        self.roots = roots
        if self.gcs.on:
            for od in dict.fromkeys(self.out_dirs):
                step = self.gcs.restore_run(od)
                if step:
                    self.log(f"restored {od} from {self.gcs.url(self.gcs.restore_prefix, run_rel(od))} ({step})")
            threading.Thread(target=self.gcs_loop, daemon=True).start()
        with open(self.cmd_log, "ab") as lf:
            lf.write(f"\n===== boot run {time.strftime('%F %T')} =====\n".encode())
            lf.flush()
            p = subprocess.Popen(["nohup", "bash", "-c", self.job["cmd"]], cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        open(os.path.join(STATE, "cmd.pid"), "w").write(str(p.pid))
        self.log(f"cmd started pid {p.pid}")
        while p.poll() is None:
            try:
                self.mirror.scan()
            except Exception as e:
                self.log(f"mirror error {e!r}")
            time.sleep(POLL_SEC)
        self.mirror.scan()
        return p.returncode

    def gcs_loop(self):
        while not self._stop.wait(self.gcs.every):
            if self.stage != "run":
                return
            try:
                n = self.gcs.sync_runs(self.roots)
                guest_attr("gcs_sync", json.dumps({"t": int(time.time()), "runs": n}))
            except Exception as e:
                self.log(f"gcs sync error {e!r}"[:300])

    def finish(self, code: int, failed_stage: str | None = None):
        self.stage = "done"
        for ln in tail_lines(self.cmd_log if failed_stage is None else self.setup_log, 30):
            self.em.emit("HNET_LOG", ln)
        m = self.mirror.last_metric if self.mirror else {}
        e = self.mirror.last_eval if self.mirror else {}
        keep = ("step", "tokens_eq", "bpb", "loss", "bytes_per_sec", "mfu", "mem_gb", "step_time")
        summ = {"exit": code, "failed_stage": failed_stage, "elapsed_s": round(time.time() - self.t0),
                "last_metric": {k: m.get(k) for k in keep if k in m},
                "last_eval": {k: v for k, v in e.items() if k in ("step", "tokens_eq", "val/bpb")},
                "kernels": (self.info or {}).get("hnet_kernels")}
        js = json.dumps(summ, separators=(",", ":"))
        if self.gcs.on:
            t = time.time()
            n = self.gcs.sync_runs(getattr(self, "roots", ["runs"]))
            self.gcs.put_text("_summary.json", js)
            self.log(f"final bucket sync: {n} run dir(s) -> {self.gcs.url(self.gcs.prefix)} in {time.time() - t:.0f}s")
        self.em.emit("HNET_SUMMARY", js)
        guest_attr("summary", js)
        guest_attr("exit", str(code))
        self.em.emit("HNET_DONE", str(code))

    def main(self):
        os.makedirs(STATE, exist_ok=True)
        hb = threading.Thread(target=self.heartbeat_loop, daemon=True)
        hb.start()
        done_path = os.path.join(STATE, "done.json")
        boots = int(open(os.path.join(STATE, "boots")).read() or 0) if os.path.exists(os.path.join(STATE, "boots")) else 0
        open(os.path.join(STATE, "boots"), "w").write(str(boots + 1))
        self.set_stage("boot", boot=boots + 1, commit=self.job.get("commit"),
                       gcs=self.gcs.url(self.gcs.prefix) if self.gcs.on else None)
        if self.gcs.bucket and not self.gcs.gcloud:
            self.log("bucket configured but no gcloud on the VM; serial channel only")
        d = json.load(open(done_path)) if os.path.exists(done_path) else None
        if d is not None and d.get("nonce") == self.job.get("nonce"):
            self.log(f"already finished (exit {d.get('exit')}); not re-running")
            for ln in tail_lines(self.cmd_log if d.get("failed_stage") is None else self.setup_log, 30):
                self.em.emit("HNET_LOG", ln)
            self.em.emit("HNET_DONE", str(d.get("exit")))
        else:
            if d is not None:
                self.log("new job nonce: re-running (venv/data/checkpoints on disk are reused)")
            code, failed = 0, None
            try:
                for name, fn in (("clone", self.clone), ("venv", self.venv), ("kernels", self.kernels),
                                 ("data", self.data)):
                    self.set_stage(name)
                    fn()
                self.set_stage("run")
                code = self.run_cmd()
            except Exception as e:
                failed = self.stage
                code = 97
                self.log(f"stage {failed} failed: {e!r}"[:400])
            json.dump({"exit": code, "failed_stage": failed, "at": time.time(), "nonce": self.job.get("nonce")},
                      open(done_path, "w"))
            self.finish(code, failed)
        self.stage = "done"
        linger = float(self.job.get("linger_min", 5))
        if linger > 0:
            self.log(f"lingering {linger:g} min so the serial output can be read")
            time.sleep(linger * 60)
        self._stop.set()
        if self.job.get("auto_shutdown", True):
            self.em.emit("HNET_STAGE", json.dumps({"stage": "shutdown"}))
            subprocess.run(["shutdown", "-h", "now"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    job_path = argv[0] if argv else os.path.join(STATE, "job.json")
    with open(job_path) as f:
        job = json.load(f)
    os.makedirs(STATE, exist_ok=True)
    em = Emitter(log_path=os.path.join(STATE, "agent_lines.log"))
    Agent(job, em).main()


if __name__ == "__main__":
    main()
