"""Reading HF-hosted parquet datasets row group by row group (for hnet.data.mix fetch).

Two modes:
* ``stream``   (default): HTTP range requests through ``HfFileSystem``, reading only the needed columns of the
               needed row groups. No local copy; best for small builds (a 2 GB file is never downloaded whole).
* ``download``: ``hf_hub_download`` of whole files into ``cache_dir`` (resumable; uses hf_xet / hf_transfer when
               installed), the next file prefetched on a background thread, each file deleted once it has been
               consumed (``keep=False``). Best for full builds on a machine with fast internet.
Both read the same rows in the same order, so the fetched documents do not depend on the mode.
Anonymous access works for every source of mix_v1 (none is gated); an ``HF_TOKEN`` in the environment (optional)
only raises rate limits.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _retry(fn, what: str, retries: int = 8):
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except (FileNotFoundError, PermissionError):
            raise
        except Exception as e:  # network hiccup: back off and retry
            last = e
            time.sleep(min(60.0, 2.0 ** attempt))
    raise RuntimeError(f"{what} failed after {retries} attempts: {last!r}")


class ParquetReader:
    def __init__(self, repo: str, revision: str | None, mode: str = "stream", cache_dir=None, keep: bool = False,
                 block_size: int = 8 << 20):
        assert mode in ("stream", "download"), mode
        self.repo, self.revision, self.mode = repo, revision, mode
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.keep, self.block_size = keep, block_size
        self._meta: dict = {}
        self._pf: dict = {}
        self._dl: dict = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(1) if mode == "download" else None
        self._fs = None

    # ------------------------------------------------------------ listing
    def list_files(self) -> list[str]:
        from huggingface_hub import HfApi

        return _retry(lambda: list(HfApi().list_repo_files(self.repo, repo_type="dataset", revision=self.revision)),
                      f"listing {self.repo}")

    # ------------------------------------------------------------ files
    def _hf_path(self, f: str) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        return f"datasets/{self.repo}{rev}/{f}"

    def _download(self, f: str) -> Path:
        from huggingface_hub import hf_hub_download

        assert self.cache_dir is not None, "download mode needs a cache_dir"
        local = self.cache_dir / self.repo.replace("/", "__") / (self.revision or "main")
        return Path(_retry(lambda: hf_hub_download(self.repo, f, repo_type="dataset", revision=self.revision,
                                                   local_dir=str(local)), f"download {self.repo}/{f}"))

    def prefetch(self, f: str):
        if self._pool is None:
            return
        with self._lock:
            if f not in self._dl:
                self._dl[f] = self._pool.submit(self._download, f)

    def _parquet(self, f: str):
        import pyarrow.parquet as pq

        pf = self._pf.get(f)
        if pf is not None:
            return pf
        if self.mode == "download":
            self.prefetch(f)
            path = self._dl[f].result()
            pf = pq.ParquetFile(str(path))
        else:
            if self._fs is None:
                from huggingface_hub import HfFileSystem

                self._fs = HfFileSystem()
            fh = _retry(lambda: self._fs.open(self._hf_path(f), "rb", block_size=self.block_size),
                        f"open {self.repo}/{f}")
            pf = pq.ParquetFile(fh, pre_buffer=True)
        self._pf = {f: pf}  # keep one open file
        return pf

    def num_row_groups(self, f: str) -> int:
        if f not in self._meta:
            self._meta[f] = _retry(lambda: self._parquet(f).metadata.num_row_groups, f"metadata {self.repo}/{f}")
        return self._meta[f]

    def read(self, f: str, rg_start: int, rg_end: int, columns: list[str]) -> dict:
        """{column: python list}. The first column (the document text) comes back as bytes (UTF-8 is validated by
        the caller, so a bad row is dropped instead of failing the unit); the others as str."""
        import pyarrow as pa

        def go():
            try:
                t = self._parquet(f).read_row_groups(list(range(rg_start, rg_end)), columns=columns)
            except Exception:
                self._pf.pop(f, None)  # reopen on retry (stale connection)
                raise
            out = {}
            for c in columns:
                col = t.column(c)
                if pa.types.is_large_string(col.type):
                    col = col.cast(pa.large_binary())
                elif pa.types.is_string(col.type):
                    col = col.cast(pa.binary())
                out[c] = col.to_pylist()
            for c in columns:  # selection / key columns are compared as str
                if c != columns[0]:
                    out[c] = [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in out[c]]
            return out

        return _retry(go, f"read {self.repo}/{f} row groups {rg_start}:{rg_end}")

    def release(self, f: str):
        self._pf.pop(f, None)
        if self.mode == "download" and not self.keep:
            fut = self._dl.pop(f, None)
            if fut is not None:
                try:
                    Path(fut.result()).unlink(missing_ok=True)
                except Exception:
                    pass

    def close(self):
        self._pf.clear()
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
