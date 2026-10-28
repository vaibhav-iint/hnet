"""Shell snippets run on the remote machine. Pure string builders (unit-tested)."""

from __future__ import annotations

import re
import shlex
import time


def root_expr(root: str) -> str:
    """Quote a remote root, keeping a leading ~ expandable."""
    if root.startswith("~/"):
        return '"$HOME"/' + shlex.quote(root[2:])
    return shlex.quote(root)


def new_run_id(cmd: str, now: float | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime(now or time.time()))
    words = re.findall(r"[A-Za-z0-9]+", " ".join(cmd.split()[:4]))
    slug = "-".join(w for w in words if w not in ("python", "m", "bash", "sh"))[:24].strip("-")
    return f"{ts}-{slug}" if slug else ts


def sync_unpack(root: str, tar_remote: str) -> str:
    r = root_expr(root)
    t = shlex.quote(tar_remote)
    return f"mkdir -p {r} && tar -xzf {t} -C {r} && rm -f {t} && echo synced $(cd {r} && pwd)"


def detached(root: str, run_id: str, cmd: str, env: dict[str, str] | None = None) -> str:
    """Start `cmd` in a new session under nohup, cwd=root, logs in runs/<id>/stdout.log, pid in
    runs/<id>/pid, exit status in runs/<id>/exit_code once it finishes. Returns immediately."""
    r = root_expr(root)
    d = f"runs/{run_id}"
    exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in (env or {}).items())
    inner = (f"{exports}export PYTHONUNBUFFERED=1 HNET_RUN_DIR={shlex.quote(d)}; "
             f"( {cmd} ); echo $? > {d}/exit_code")
    return (
        f"cd {r} && mkdir -p {d} && printf '%s\\n' {shlex.quote(cmd)} > {d}/cmd.txt && "
        f"if command -v setsid >/dev/null; then L='setsid nohup'; else L=nohup; fi; "
        f"$L bash -c {shlex.quote(inner)} > {d}/stdout.log 2>&1 < /dev/null & "
        f"echo $! > {d}/pid; sleep 1; echo started {run_id} pid $(cat {d}/pid)"
    )


def ps(root: str, run_ids: list[str]) -> str:
    r = root_expr(root)
    lines = [f"cd {r} 2>/dev/null || {{ echo 'no repo at {root}'; exit 0; }}"]
    for rid in run_ids:
        d = f"runs/{rid}"
        lines.append(
            f"if [ -f {d}/exit_code ]; then echo \"{rid}: exited $(cat {d}/exit_code)\"; "
            f"elif [ -f {d}/pid ] && kill -0 $(cat {d}/pid) 2>/dev/null; then "
            f"echo \"{rid}: running pid $(cat {d}/pid) log=$(wc -c < {d}/stdout.log 2>/dev/null)B\"; "
            f"else echo \"{rid}: not running (no exit_code)\"; fi"
        )
    lines.append("command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,utilization.gpu,"
                 "memory.used,memory.total --format=csv,noheader || true")
    lines.append("uptime; df -h . | tail -1; free -g 2>/dev/null | sed -n 2p || true")
    return "; ".join(lines)


def tail(root: str, run_id: str, lines: int = 40, metrics: bool = False, metrics_lines: int = 5) -> str:
    r = root_expr(root)
    d = f"runs/{run_id}"
    s = f"cd {r} && echo '== {d}/stdout.log' && tail -n {int(lines)} {d}/stdout.log"
    if metrics:
        s += (f"; M=$(ls -t {d}/metrics.jsonl {d}/*/metrics.jsonl 2>/dev/null | head -1); "
              f"[ -z \"$M\" ] && M=$(find runs -name metrics.jsonl -printf '%T@ %p\\n' 2>/dev/null "
              f"| sort -n | tail -1 | cut -d' ' -f2-); "
              f"if [ -n \"$M\" ]; then echo \"== $M\"; tail -n {int(metrics_lines)} \"$M\"; "
              f"else echo '== no metrics.jsonl found'; fi")
    return s


def resolve_remote_path(root: str, path: str) -> str:
    """Absolute paths and ~ pass through; relative paths are relative to the repo root."""
    if path.startswith("/") or path.startswith("~"):
        return path
    return root.rstrip("/") + "/" + path


METRIC_KEYS = ("step", "loss", "bpb", "lr", "grad_norm", "mfu", "bytes_per_sec", "mem_gb", "step_time")


def format_metrics_line(line: str) -> str:
    import json
    try:
        d = json.loads(line)
    except Exception:
        return line
    parts = []
    for k in METRIC_KEYS:
        if k in d:
            v = d[k]
            parts.append(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}")
    extra = [k for k in d if k.startswith("compression/")]
    for k in extra[:3]:
        parts.append(f"{k}={d[k]:.3g}" if isinstance(d[k], float) else f"{k}={d[k]}")
    return " ".join(parts) or line
