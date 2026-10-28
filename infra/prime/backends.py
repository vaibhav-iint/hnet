"""Two resource kinds behind one small interface.

SandboxBackend (preferred): Prime sandboxes via the prime_sandboxes SDK. Everything goes over
HTTPS (exec, upload, download), so it works from behind an HTTPS-only proxy. Platform-side
lifetime timeout = --max-hours.

PodBackend: Prime GPU pods via prime_cli's PodsClient for lifecycle and the local `ssh`/`scp`
binaries for exec/transfer (needs raw TCP to the pod and the account's SSH key at
PRIME_SSH_KEY or the prime config's ssh_key_path). Pods have no platform-side lifetime limit:
max-hours is enforced by a remote `shutdown` timer (best effort) plus the local `reap` command,
which terminates any resource past its deadline.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass

from . import config as C
from .remote import root_expr


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str


GONE = {"TERMINATED", "TIMEOUT", "ERROR", "DELETED", "NOT_FOUND"}


# ----------------------------------------------------------------------------- sandboxes
class SandboxBackend:
    kind = "sandbox"

    def __init__(self, client=None):
        if client is None:
            from prime_sandboxes import APIClient, SandboxClient
            client = SandboxClient(APIClient())
        self.c = client

    def create(self, name: str, *, image: str, cpu: float, mem_gb: float, disk_gb: float,
               gpu_type: str | None, gpu_count: int, timeout_minutes: int,
               idle_timeout_minutes: int | None = None, env: dict | None = None,
               volumes: list[tuple[str, str, bool]] | None = None, region: str | None = None) -> dict:
        from prime_sandboxes import CreateSandboxRequest, VolumeMount
        req = CreateSandboxRequest(
            name=name, docker_image=image, cpu_cores=cpu, memory_gb=mem_gb, disk_size_gb=disk_gb,
            gpu_count=gpu_count, gpu_type=gpu_type if gpu_count else None,
            timeout_minutes=timeout_minutes, idle_timeout_minutes=idle_timeout_minutes,
            environment_vars=env or None, labels=[C.LABEL], region=region,
            volumes=[VolumeMount(name=n, mount_path=p, read_only=ro) for n, p, ro in volumes]
            if volumes else None,
        )
        sb = self.c.create(req)
        return {"id": sb.id, "status": sb.status}

    def wait_ready(self, rid: str, timeout_s: int = 900) -> None:
        self.c.wait_for_creation(rid, max_attempts=max(1, timeout_s // 2))

    def status(self, rid: str) -> dict:
        try:
            sb = self.c.get(rid)
        except Exception as e:  # 404 -> gone
            if "404" in str(e) or "not found" in str(e).lower():
                return {"status": "NOT_FOUND"}
            raise
        return {"status": sb.status, "terminated_at": sb.terminated_at.timestamp()
                if sb.terminated_at else None, "reason": sb.termination_reason,
                "started_at": sb.started_at.timestamp() if sb.started_at else None}

    def exec(self, rid: str, cmd: str, timeout: int = 300) -> ExecResult:
        r = self.c.execute_command(rid, f"bash -lc {shlex.quote(cmd)}", timeout=timeout)
        return ExecResult(r.exit_code, r.stdout, r.stderr)

    def upload(self, rid: str, local: str, remote: str, timeout: int = 900) -> None:
        self.c.upload_file(rid, remote, local, timeout=timeout)

    def download(self, rid: str, remote: str, local: str, timeout: int = 900) -> None:
        self.c.download_file(rid, remote, local, timeout=timeout)

    def delete(self, rid: str) -> None:
        self.c.delete(rid)

    def list_labeled(self) -> list[dict]:
        out, page = [], 1
        while True:
            resp = self.c.list(labels=[C.LABEL], page=page, per_page=100, exclude_terminated=True)
            out += [{"id": s.id, "name": s.name, "status": s.status} for s in resp.sandboxes]
            if not resp.has_next:
                return out
            page += 1


# ----------------------------------------------------------------------------- pods
def parse_ssh(conn) -> tuple[str, str, int]:
    """'root@1.2.3.4 -p 2222' -> (user, host, port)."""
    if isinstance(conn, list):
        conn = next((c for c in conn if c), None)
    if not conn:
        raise RuntimeError("pod has no SSH connection yet")
    toks = str(conn).split()
    port = 22
    if "-p" in toks:
        port = int(toks[toks.index("-p") + 1])
    target = toks[0]
    user, host = target.split("@", 1) if "@" in target else ("root", target)
    if ":" in host and host.count(":") == 1:
        host, p = host.split(":")
        port = int(p)
    return user, host, port


class PodBackend:
    kind = "pod"

    def __init__(self, client=None, ssh_key: str | None = None):
        if client is None:
            from prime_cli.api.pods import PodsClient
            from prime_cli.core import APIClient
            client = PodsClient(APIClient())
        self.c = client
        self._ssh_key = ssh_key

    @property
    def ssh_key(self) -> str:
        if self._ssh_key:
            return self._ssh_key
        k = os.environ.get("PRIME_SSH_KEY")
        if not k:
            from prime_cli.core import Config
            k = Config().ssh_key_path
        return os.path.expanduser(k)

    def create(self, name: str, *, offer, image: str, disk_gb: int | None, gpu_count: int,
               env: dict | None = None) -> dict:
        cfg = {
            "pod": {
                "name": name, "cloudId": offer.cloud_id, "gpuType": offer.gpu_type,
                "socket": offer.socket, "gpuCount": gpu_count, "diskSize": disk_gb,
                "vcpus": offer.vcpu.default_count, "memory": offer.memory.default_count,
                "image": image, "dataCenterId": offer.data_center, "maxPrice": None,
                "country": None, "security": None, "jupyterPassword": None, "autoRestart": False,
                "customTemplateId": None,
                "envVars": [{"key": k, "value": v} for k, v in (env or {}).items()],
            },
            "provider": {"type": offer.provider} if offer.provider else {},
        }
        pod = self.c.create(cfg)
        return {"id": pod.id, "status": pod.status, "price_hr": pod.price_hr}

    def _st(self, rid: str):
        sts = self.c.get_status([rid])
        return sts[0] if sts else None

    def wait_ready(self, rid: str, timeout_s: int = 1800) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            st = self._st(rid)
            if st is not None and st.status == "ACTIVE" and st.ssh_connection:
                if self.exec(rid, "echo ok", timeout=30).exit_code == 0:
                    return
            if st is not None and st.status in ("ERROR", "TERMINATED"):
                raise RuntimeError(f"pod {rid} status {st.status}: {st.installation_failure}")
            time.sleep(15)
        raise TimeoutError(f"pod {rid} not ready after {timeout_s}s")

    def status(self, rid: str) -> dict:
        try:
            st = self._st(rid)
        except Exception as e:
            if "404" in str(e) or "not found" in str(e).lower():
                return {"status": "NOT_FOUND"}
            raise
        if st is None:
            return {"status": "NOT_FOUND"}
        return {"status": st.status, "ssh": st.ssh_connection, "price_hr": st.cost_per_hr}

    def _ssh_base(self, rid: str) -> list[str]:
        user, host, port = parse_ssh(self._st(rid).ssh_connection)
        return ["-i", self.ssh_key, "-p", str(port), "-o", "StrictHostKeyChecking=accept-new",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30", f"{user}@{host}"]

    def exec(self, rid: str, cmd: str, timeout: int = 300) -> ExecResult:
        args = ["ssh", *self._ssh_base(rid), f"bash -lc {shlex.quote(cmd)}"]
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return ExecResult(p.returncode, p.stdout, p.stderr)

    def upload(self, rid: str, local: str, remote: str, timeout: int = 900) -> None:
        base = self._ssh_base(rid)
        with open(local, "rb") as f:
            q = root_expr(remote)
            p = subprocess.run(["ssh", *base, f"mkdir -p \"$(dirname {q})\" && cat > {q}"],
                               stdin=f, capture_output=True, timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"upload failed: {p.stderr.decode()[-500:]}")

    def download(self, rid: str, remote: str, local: str, timeout: int = 900) -> None:
        base = self._ssh_base(rid)
        os.makedirs(os.path.dirname(os.path.abspath(local)), exist_ok=True)
        with open(local, "wb") as f:
            p = subprocess.run(["ssh", *base, f"cat {root_expr(remote)}"], stdout=f, stderr=subprocess.PIPE,
                               timeout=timeout)
        if p.returncode:
            raise RuntimeError(f"download failed: {p.stderr.decode()[-500:]}")

    def delete(self, rid: str) -> None:
        self.c.delete(rid)

    def list_labeled(self) -> list[dict]:
        pods = self.c.list().data
        return [{"id": p.id, "name": p.name, "status": p.status} for p in pods
                if (p.name or "").startswith(C.NAME_PREFIX)]


def backend_for(kind: str):
    return SandboxBackend() if kind == "sandbox" else PodBackend()
