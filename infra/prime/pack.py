"""Repo tarball for `sync`: the files git would track, minus data/runs/venvs/caches."""

from __future__ import annotations

import io
import os
import subprocess
import tarfile
from pathlib import Path

EXCLUDE_DIRS = {
    ".git", "data", "runs", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".ipynb_checkpoints", "wandb", "ckpt", "checkpoints",
    ".cache", "node_modules", ".tox", "build", "dist", "*.egg-info",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".so", ".bin", ".pt", ".safetensors", ".npy", ".npz",
                    ".tar", ".tar.gz", ".tgz", ".zip")
MAX_FILE_BYTES = 20 * 1024 * 1024


def _excluded(rel: str) -> bool:
    parts = Path(rel).parts
    for p in parts[:-1]:
        if p in EXCLUDE_DIRS or p.endswith(".egg-info"):
            return True
    if parts and parts[-1] in EXCLUDE_DIRS:
        return True
    return rel.endswith(EXCLUDE_SUFFIXES)


def _git_files(root: Path) -> list[str] | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            capture_output=True, check=True, timeout=30,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    files = [f for f in out.decode().split("\0") if f]
    return files or None


def _walk_files(root: Path) -> list[str]:
    out = []
    for dp, dns, fns in os.walk(root):
        rel_dir = os.path.relpath(dp, root)
        dns[:] = [d for d in dns if d not in EXCLUDE_DIRS and not d.endswith(".egg-info")]
        for fn in fns:
            out.append(os.path.normpath(os.path.join(rel_dir, fn)))
    return out


def list_files(root: Path | str) -> list[str]:
    root = Path(root)
    files = _git_files(root)
    if files is None:
        files = _walk_files(root)
    keep = []
    for rel in sorted(set(files)):
        p = root / rel
        if _excluded(rel) or not p.is_file() or p.is_symlink():
            continue
        if p.stat().st_size > MAX_FILE_BYTES:
            continue
        keep.append(rel)
    return keep


def make_tarball(root: Path | str, out_path: Path | str | None = None) -> bytes | Path:
    """gzip tarball of list_files(root) with paths relative to root.
    Returns bytes if out_path is None, else writes the file and returns its path."""
    root = Path(root)
    buf = io.BytesIO() if out_path is None else open(out_path, "wb")
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for rel in list_files(root):
            ti = tf.gettarinfo(str(root / rel), arcname=rel)
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(root / rel, "rb") as f:
                tf.addfile(ti, f)
    if out_path is None:
        return buf.getvalue()
    buf.close()
    return Path(out_path)
