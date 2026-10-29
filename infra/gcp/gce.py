"""Thin gcloud CLI wrapper (subprocess; HTTPS only through the CLI). Every call goes through `run()`,
which tests replace by mocking `subprocess.run`.

Credentials: CLOUDSDK_CONFIG (default: the scratchpad gcloud dir). A preset CLOUDSDK_AUTH_ACCESS_TOKEN
overrides the configured account, so it is dropped from the child environment unless
HNET_GCP_KEEP_TOKEN=1. Nothing here prints or stores tokens.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass, field

from . import config as C


class GcloudError(RuntimeError):
    def __init__(self, args: list[str], code: int, stderr: str):
        self.args_, self.code, self.stderr = args, code, stderr
        super().__init__(f"gcloud {' '.join(args[1:4])} ... exit {code}: {stderr.strip()[-800:]}")


def gcloud_env() -> dict:
    env = dict(os.environ)
    if os.environ.get("HNET_GCP_KEEP_TOKEN") != "1":
        env.pop("CLOUDSDK_AUTH_ACCESS_TOKEN", None)
    cfg = C.gcloud_config_dir()
    if cfg:
        env["CLOUDSDK_CONFIG"] = cfg
    env["CLOUDSDK_CORE_DISABLE_PROMPTS"] = "1"
    return env


def project_flag() -> list[str]:
    p = os.environ.get("HNET_GCP_PROJECT")
    return [f"--project={p}"] if p else []


def run(args: list[str], *, json_out: bool = True, timeout: float = 300, check: bool = True):
    """Run `gcloud <args>`; returns parsed JSON (json_out) or (stdout, stderr)."""
    full = ["gcloud", *args, *project_flag()]
    if json_out and not any(a.startswith("--format") for a in args):
        full.append("--format=json")
    r = subprocess.run(full, capture_output=True, text=True, timeout=timeout, env=gcloud_env())
    if check and r.returncode != 0:
        raise GcloudError(full, r.returncode, r.stderr or "")
    if not json_out:
        return r.stdout, r.stderr
    out = (r.stdout or "").strip()
    return json.loads(out) if out else None


# ----------------------------------------------------------------------------- errors
RETRYABLE = [
    "zone_resource_pool_exhausted", "does not have enough resources available", "resource_pool_exhausted",
    "quota_exceeded", "quota '", "exceeded quota", "quota exceeded",
    "unsupported_operation", "is not available in", "not supported in zone", "stockout",
    "does not exist in zone", "currently unavailable", "resource_not_ready",
]
NOT_FOUND = ["was not found", "not found", "notfound", "404"]


def is_retryable(stderr: str) -> bool:
    s = stderr.lower()
    if "permission" in s and "denied" in s:
        return False
    if "already exists" in s:
        return False
    if "was not found" in s and "/zones/" in s and ("machinetypes" in s or "acceleratortypes" in s):
        return True  # machine/accelerator type not offered in this zone
    return any(p in s for p in RETRYABLE)


def is_not_found(err: Exception) -> bool:
    s = str(getattr(err, "stderr", "") or err).lower()
    return any(p in s for p in NOT_FOUND)


# ----------------------------------------------------------------------------- create
@dataclass
class CreateSpec:
    name: str
    machine_type: str
    accelerator: str | None
    gpu_count: int
    spot: bool
    max_hours: float
    disk_gb: int
    startup_path: str
    image_family: str = C.IMAGE_FAMILY
    image_project: str = C.IMAGE_PROJECT
    labels: dict = field(default_factory=lambda: dict(C.LABELS))
    service_account: str | None = None  # None -> --no-service-account --no-scopes
    extra_metadata: dict = field(default_factory=dict)


def max_run_duration(hours: float) -> str:
    return f"{max(60, int(round(hours * 3600)))}s"


def create_args(spec: CreateSpec, zone: str) -> list[str]:
    md = {"enable-guest-attributes": "TRUE", **spec.extra_metadata}
    args = [
        "compute", "instances", "create", spec.name, f"--zone={zone}",
        f"--machine-type={spec.machine_type}",
        f"--image-family={spec.image_family}", f"--image-project={spec.image_project}",
        f"--boot-disk-size={max(spec.disk_gb, C.MIN_DISK_GB)}GB", f"--boot-disk-type={C.DISK_TYPE}",
        *([f"--service-account={spec.service_account}", f"--scopes={C.VM_SCOPES}"] if spec.service_account
          else ["--no-service-account", "--no-scopes"]),
        f"--metadata-from-file=startup-script={spec.startup_path}",
        "--metadata=" + ",".join(f"{k}={v}" for k, v in md.items()),
        "--labels=" + ",".join(f"{k}={v}" for k, v in spec.labels.items()),
        f"--max-run-duration={max_run_duration(spec.max_hours)}",
    ]
    if spec.accelerator:
        args.append(f"--accelerator=type={spec.accelerator},count={spec.gpu_count}")
    if spec.gpu_count > 0:
        args.append("--maintenance-policy=TERMINATE")
    if spec.spot:
        # STOP keeps the boot disk (checkpoints, data, venv) so a preempted run can be resumed.
        args += ["--provisioning-model=SPOT", "--instance-termination-action=STOP"]
    else:
        args += ["--provisioning-model=STANDARD", "--instance-termination-action=DELETE"]
    return args


@dataclass
class CreateResult:
    ok: bool
    zone: str | None
    tried: list  # [(zone, short error)]
    instance: dict | None = None


class NoCapacity(RuntimeError):
    """Every zone answered with a capacity/quota/availability error."""

    def __init__(self, tried: list):
        self.tried = tried
        super().__init__("no zone could create the VM:\n" + "\n".join(f"  {z}: {e}" for z, e in tried))


def create_with_fallback(spec: CreateSpec, zones: list[str], log=print) -> CreateResult:
    """Try zones in order, moving on only for capacity/quota/availability errors. Any other error is
    raised as is; NoCapacity when every zone refused."""
    tried = []
    for z in zones:
        log(f"[create] {spec.name} in {z} ({spec.machine_type}{', spot' if spec.spot else ''})")
        try:
            out = run(create_args(spec, z), timeout=600)
        except GcloudError as e:
            if "already exists" in e.stderr.lower():
                raise  # an existing VM of that name is never touched
            # a failed insert can leave a half-created instance of our new name behind: remove it
            if describe(spec.name, z) is not None:
                log(f"[create] removing the half-created {spec.name} in {z}")
                delete(spec.name, z)
            if not is_retryable(e.stderr):
                raise
            tried.append((z, _short(e.stderr)))
            log(f"[create] {z}: {tried[-1][1]} -> next zone")
            continue
        inst = out[0] if isinstance(out, list) and out else out
        return CreateResult(True, z, tried, inst)
    raise NoCapacity(tried)


def _short(stderr: str) -> str:
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    for pick in (lambda ln: ln.startswith("- ") or ln.startswith("message:"), lambda ln: "ERROR" in ln):
        for ln in lines:
            if pick(ln):
                return ln[:300]
    return (lines[-1] if lines else "")[:300]


# ----------------------------------------------------------------------------- instance ops
def describe(name: str, zone: str) -> dict | None:
    try:
        return run(["compute", "instances", "describe", name, f"--zone={zone}"], timeout=120)
    except GcloudError as e:
        if is_not_found(e):
            return None
        raise


def delete(name: str, zone: str) -> None:
    try:
        run(["compute", "instances", "delete", name, f"--zone={zone}", "--quiet", "--delete-disks=all"],
            json_out=False, timeout=600)
    except GcloudError as e:
        if not is_not_found(e):
            raise


def start(name: str, zone: str) -> None:
    run(["compute", "instances", "start", name, f"--zone={zone}"], json_out=False, timeout=600)


def stop(name: str, zone: str) -> None:
    run(["compute", "instances", "stop", name, f"--zone={zone}"], json_out=False, timeout=600)


def list_ours() -> list[dict]:
    """Instances whose name starts with the hnet- prefix (other people's VMs are never listed)."""
    out = run(["compute", "instances", "list", f"--filter=name ~ ^{C.NAME_PREFIX}"], timeout=120) or []
    return [i for i in out if i.get("name", "").startswith(C.NAME_PREFIX)]


def zone_of(inst: dict) -> str:
    return (inst.get("zone") or "").rsplit("/", 1)[-1]


def serial_output(name: str, zone: str, start_at: int) -> dict:
    """{'contents': str, 'start': int, 'next': int} from serial port 1, starting at byte `start_at`."""
    out, err = run(["compute", "instances", "get-serial-port-output", name, f"--zone={zone}", "--port=1",
                    f"--start={int(start_at)}", "--format=json"], json_out=False, timeout=180)
    return parse_serial_response(out, err, start_at)


def parse_serial_response(out: str, err: str, start_at: int) -> dict:
    """The JSON form of get-serial-port-output (we always pass --format=json)."""
    d = json.loads(out)
    contents = d.get("contents") or ""
    return {"contents": contents, "start": int(d.get("start", start_at)),
            "next": int(d.get("next", start_at + len(contents.encode())))}


def guest_attributes(name: str, zone: str, path: str = "hnet/") -> dict:
    out = run(["compute", "instances", "get-guest-attributes", name, f"--zone={zone}",
               f"--query-path={path}"], timeout=120) or []
    return {f"{a.get('namespace')}/{a.get('key')}": a.get("value") for a in out}


# ----------------------------------------------------------------------------- Cloud Storage
def storage_ls_sizes(url: str) -> dict[str, int]:
    """{object url: size} for every object under url (recursive); {} when there is none."""
    try:
        out, _ = run(["storage", "ls", "-l", url.rstrip("/") + "/**"], json_out=False, timeout=300)
    except GcloudError as e:
        if "matched no objects" in e.stderr.lower():  # an empty prefix, not an error
            return {}
        raise
    return parse_ls_sizes(out)


def parse_ls_sizes(out: str) -> dict[str, int]:
    res = {}
    for ln in (out or "").splitlines():
        f = ln.split()
        if len(f) >= 3 and f[-1].startswith("gs://") and f[0].isdigit():
            res[f[-1]] = int(f[0])
    return res


def storage_cp(srcs: list[str], dst: str, recursive: bool = False, timeout: float = 3600) -> None:
    run(["storage", "cp", *(["-r"] if recursive else []), *srcs, dst], json_out=False, timeout=timeout)


def storage_rsync(src: str, dst: str, exclude: str | None = None, timeout: float = 6 * 3600) -> None:
    run(["storage", "rsync", "-r", *([f"--exclude={exclude}"] if exclude else []), src, dst], json_out=False,
        timeout=timeout)


STOP_OPS = ("compute.instances.preempted", "compute.instances.guestTerminate", "stop",
            "compute.instances.hostError", "compute.instances.terminateOnHostMaintenance")


def last_stop_reason(name: str, zone: str) -> str | None:
    """Why the VM last stopped (preempted / guest shutdown / stop / max-run-duration ...), best effort."""
    ops = run(["compute", "operations", "list", f"--zones={zone}",
               f"--filter=targetLink ~ /instances/{name}$"], timeout=120) or []
    ops = [o for o in ops if o.get("operationType") in STOP_OPS or "Duration" in (o.get("operationType") or "")
           or "terminat" in (o.get("operationType") or "").lower()]
    if not ops:
        return None
    o = max(ops, key=lambda o: o.get("insertTime", ""))
    return f"{o.get('operationType')} at {o.get('insertTime')}: {o.get('statusMessage', '')}".strip()
