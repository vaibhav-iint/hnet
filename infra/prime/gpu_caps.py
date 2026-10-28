"""Remote helper used by setup_gpu.sh (stdlib only, runs before deps are installed).

  python infra/prime/gpu_caps.py torch-index         -> pip index URL for torch matching the driver
  python infra/prime/gpu_caps.py wheel PKG           -> URL of a prebuilt wheel for this torch/CUDA/py, or ''
  python infra/prime/gpu_caps.py summary [OUT.json]  -> capability summary (GPU, torch, kernels)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request

REPOS = {
    "mamba_ssm": ("state-spaces/mamba", "mamba_ssm"),
    "causal_conv1d": ("Dao-AILab/causal-conv1d", "causal_conv1d"),
    "flash_attn": ("Dao-AILab/flash-attention", "flash_attn"),
}


def driver_cuda() -> tuple[int, int] | None:
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def torch_index() -> str:
    cu = driver_cuda()
    if cu is None:
        return "https://download.pytorch.org/whl/cpu"
    v = cu[0] * 10 + cu[1] if cu[0] < 13 else 130
    for tag in (130, 129, 128, 126, 124, 121, 118):
        if v >= tag:
            return f"https://download.pytorch.org/whl/cu{tag}"
    return "https://download.pytorch.org/whl/cu118"


def wheel_tag() -> dict:
    import torch
    tv = torch.__version__.split("+")[0].split(".")
    return {
        "cuda_major": (torch.version.cuda or "").split(".")[0],
        "torch": f"{tv[0]}.{tv[1]}",
        "abi": str(torch._C._GLIBCXX_USE_CXX11_ABI).upper(),
        "py": f"cp{sys.version_info.major}{sys.version_info.minor}",
    }


def pick_wheel(asset_names: list[str], pkg: str, tag: dict) -> str | None:
    """Pick e.g. mamba_ssm-2.2.4+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl."""
    if not tag["cuda_major"]:
        return None
    pat = re.compile(
        rf"^{pkg}-[\d.]+(\.post\d+)?\+cu{tag['cuda_major']}\d*torch{re.escape(tag['torch'])}"
        rf"cxx11abi{tag['abi']}-{tag['py']}-{tag['py']}-linux_x86_64\.whl$")
    for n in asset_names:
        if pat.match(n):
            return n
    return None


def find_wheel(pkg: str, max_releases: int = 6) -> str:
    repo, wname = REPOS[pkg]
    tag = wheel_tag()
    req = urllib.request.Request(f"https://api.github.com/repos/{repo}/releases?per_page={max_releases}",
                                 headers={"Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            rels = json.load(r)
    except Exception as e:
        print(f"[gpu_caps] github api failed for {repo}: {e}; guessing from PyPI version", file=sys.stderr)
        return guess_wheel(pkg, tag)
    for rel in rels:
        assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
        n = pick_wheel(list(assets), wname, tag)
        if n:
            return assets[n]
    return ""


def wheel_name(wname: str, ver: str, tag: dict) -> str:
    return (f"{wname}-{ver}+cu{tag['cuda_major']}torch{tag['torch']}cxx11abi{tag['abi']}"
            f"-{tag['py']}-{tag['py']}-linux_x86_64.whl")


def guess_wheel(pkg: str, tag: dict) -> str:
    """Same URL scheme the packages' setup.py use: releases/download/v{ver}/{wheel_name}."""
    repo, wname = REPOS[pkg]
    if not tag["cuda_major"]:
        return ""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{pkg.replace('_', '-')}/json", timeout=30) as r:
            ver = json.load(r)["info"]["version"]
        url = f"https://github.com/{repo}/releases/download/v{ver}/{wheel_name(wname, ver, tag)}"
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=30) as r:
            return url if r.status == 200 else ""
    except Exception:
        return ""


def summary(out: str | None = None) -> dict:
    s: dict = {"python": sys.version.split()[0], "driver_cuda": driver_cuda()}
    try:
        import torch
        s["torch"] = torch.__version__
        s["torch_cuda"] = torch.version.cuda
        s["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            s["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            s["capability"] = list(torch.cuda.get_device_capability(0))
            s["bf16"] = torch.cuda.is_bf16_supported()
            s["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception as e:
        s["torch"] = f"missing ({e})"
    for mod, probe in (("triton", "triton"), ("mamba_ssm", "mamba_ssm.ops.triton.ssd_combined"),
                       ("selective_scan_cuda", "selective_scan_cuda"), ("causal_conv1d", "causal_conv1d"),
                       ("flash_attn", "flash_attn")):
        try:
            m = __import__(probe, fromlist=["_"])
            s[mod] = getattr(m, "__version__", "ok")
        except Exception as e:
            s[mod] = f"no ({type(e).__name__})"
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(s, f, indent=1)
    return s


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "torch-index":
        print(torch_index())
    elif cmd == "wheel":
        print(find_wheel(sys.argv[2]))
    else:
        for k, v in summary(sys.argv[2] if len(sys.argv) > 2 else None).items():
            print(f"  {k:20} {v}")
