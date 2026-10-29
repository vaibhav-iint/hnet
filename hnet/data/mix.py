"""Data mixes: a byte-share mixture of HF sources as aligned byte shards + in-corpus BPE token shards.

Pipeline (``scripts/data/build_mix.py``; spec: ``configs/data/<mix>.spec.yaml``; docs: ``docs/DATA_MIX.md``)

1. fetch     Each source (a HF parquet dataset at a pinned revision) is read sequentially, unit by unit (a unit
             = ``rg_per_unit`` parquet row groups), in a seeded permutation of its files. Each document is routed
             to a leaf (a source, or one language of a multi-language source), split into train/val by a seeded
             hash (of the text, or of the repository for code; held-out leaves are all val), and appended to the
             leaf's *pool*: ``<root>/mix_pools/<mix>-s<seed>/<leaf>/<unit>.{bin,npy}`` (raw UTF-8 + per-doc
             [n_bytes, content_hash, is_val]). A source stops when all its leaves have their train/val quotas.
             Progress is saved after every unit (``_state/<source>.json``), so an interrupted fetch resumes
             exactly; the documents of a leaf are always a prefix of the same deterministic sequence, so a small
             build is a prefix of a bigger one and the pools can be shared between builds.
2. tokenizer (optional) train the in-corpus BPE on the first ``tokenizer_sample_bytes`` of the train stream.
3. assemble  The train stream interleaves the leaves' train documents so that *every prefix* has the mixture's
             byte shares: documents are merged in order of (leaf bytes before the doc's midpoint) / (leaf share),
             i.e. the leaf furthest behind its share goes next. Train documents equal (exact text) to any
             val-assigned document are dropped. The stream is cut into shards of ~``shard_bytes`` text bytes
             (documents are never split), each written twice over exactly the same documents:
               <root>/<dataset>/<dataset>_train_XXXXXX.bin                 token shard (uint16, [EOT] + ids per doc)
               <root>/<dataset>/bytes/<dataset>_train_XXXXXX.bytes.bin     byte shard ([BOS] + bytes + [EOS] per doc)
               <root>/<dataset>/bytes/<dataset>_train_XXXXXX.idx.npy       doc index (hnet.data.shards columns)
               <root>/<dataset>/bytes/<dataset>_train_XXXXXX.src.npy       int16 leaf index per doc
               <root>/<dataset>/bytes/<dataset>_train_XXXXXX.json          shard stats (+ resume key)
             The token encoding of every document is checked to decode to exactly its bytes.
             Val sets (same formats, one shard each, documents disjoint from train):
               <dataset>_val_000000              the combined val set, same mixture as train (``standard_valset``)
               <dataset>_val-<leaf>_000000       one per leaf: en.fineweb_edu, code.python, heldout.haskell, ...
               <dataset>_val-<category>_000000   one per category: en, code, math, ml, heldout
             plus ``tokenizer.json`` (the exact tokenizer of the token shards) and ``manifest.json``.

Formats are those of ``hnet.data.shards`` except that documents are never split across shards (every shard's
last document is complete) and the token ids are the mix tokenizer's (``<|endoftext|>`` = id 0).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import queue
import shutil
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from hnet.config import BOS_ID, EOS_ID
from hnet.data import shards as S

MIX_FORMAT = "hnet-mix-v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = REPO_ROOT / "configs" / "data"
KNOWN_SPECS = {"mix_v1": SPEC_DIR / "mix_v1.spec.yaml"}
POOL_DIRNAME = "mix_pools"
META_NBYTES, META_HASH, META_VAL = range(3)


# ============================================================================ spec
@dataclass
class Leaf:
    name: str
    index: int
    category: str  # en | code | math | ml | heldout
    source: str
    value: str | None  # value of the source's `select` column routed to this leaf
    weight: float
    heldout: bool
    share: float = 0.0  # fraction of the train mixture's bytes (0 for held-out leaves)
    group_share: float = 0.0  # fraction of its category's val set


@dataclass
class Source:
    name: str
    repo: str
    revision: str | None
    files: list
    text: str
    leaves: list  # leaf names
    select: str | None = None
    split_key: str | None = None
    rg_per_unit: int = 1
    p_val: float = 0.02
    max_units: int | None = None

    def columns(self) -> list[str]:
        return [c for c in dict.fromkeys([self.text, self.select, self.split_key]) if c]


@dataclass
class MixSpec:
    name: str
    seed: int
    tokenizer: str | None
    max_doc_bytes: int
    categories: dict
    sources: dict  # name -> Source
    leaves: list  # [Leaf] in spec order (index = position)
    presets: dict = field(default_factory=dict)
    path: str | None = None

    def leaf(self, name: str) -> Leaf:
        return next(lf for lf in self.leaves if lf.name == name)

    @property
    def in_mix(self) -> list:
        return [lf for lf in self.leaves if not lf.heldout]

    def categories_of_leaves(self) -> dict:
        out: dict = {}
        for lf in self.leaves:
            out.setdefault(lf.category, []).append(lf.name)
        return out

    def to_dict(self) -> dict:
        return dict(name=self.name, seed=self.seed, tokenizer=self.tokenizer, max_doc_bytes=self.max_doc_bytes,
                    categories=self.categories, sources={k: asdict(v) for k, v in self.sources.items()},
                    leaves=[asdict(lf) for lf in self.leaves])


def spec_from_dict(d: dict, path=None, seed: int | None = None) -> MixSpec:
    cats = {k: float(v) for k, v in d["categories"].items()}
    assert abs(sum(cats.values()) - 1.0) < 1e-6, f"category shares sum to {sum(cats.values())}"
    sources, leaves = {}, []
    for sname, sd in d["sources"].items():
        names = []
        for lname, ld in sd["leaves"].items():
            ld = ld or {}
            held = bool(ld.get("heldout", False))
            cat = "heldout" if held else lname.split(".")[0]
            assert held or cat in cats, f"leaf {lname}: category {cat!r} not in {sorted(cats)}"
            leaves.append(Leaf(lname, len(leaves), cat, sname, ld.get("value"), float(ld.get("weight", 1.0)), held))
            names.append(lname)
        sel = sd.get("select")
        assert sel or len(names) == 1, f"source {sname}: several leaves need a `select` column"
        sources[sname] = Source(sname, sd["repo"], sd.get("revision"), list(sd["files"]), sd["text"], names,
                                sel, sd.get("split_key"), int(sd.get("rg_per_unit", 1)), float(sd.get("p_val", 0.02)),
                                sd.get("max_units"))
    assert len({lf.name for lf in leaves}) == len(leaves), "duplicate leaf names"
    for cat, share in cats.items():
        group = [lf for lf in leaves if lf.category == cat]
        assert group, f"category {cat} has no leaves"
        wsum = sum(lf.weight for lf in group)
        for lf in group:
            lf.group_share = lf.weight / wsum
            lf.share = share * lf.weight / wsum
    held = [lf for lf in leaves if lf.heldout]
    for lf in held:
        lf.group_share = 1.0 / len(held)
    return MixSpec(d["name"], int(d.get("seed", 0) if seed is None else seed), d.get("tokenizer"),
                   int(d.get("max_doc_bytes", 1_000_000)), cats, sources, leaves, d.get("presets", {}),
                   str(path) if path else None)


def load_spec(spec, seed: int | None = None) -> MixSpec:
    """A spec name (``mix_v1``), a yaml path, or a dict."""
    if isinstance(spec, MixSpec):
        return spec
    if isinstance(spec, dict):
        return spec_from_dict(spec, seed=seed)
    import yaml

    path = KNOWN_SPECS.get(str(spec), Path(spec))
    with open(path) as f:
        return spec_from_dict(yaml.safe_load(f), path, seed)


@dataclass
class BuildConfig:
    dataset: str
    train_bytes: float
    shard_bytes: float = 256e6
    val_leaf_bytes: float = 2e6
    val_group_bytes: float = 5e6
    val_mix_bytes: float = 10e6
    tokenizer_sample_bytes: float = 0.0
    train_margin: float = 0.02  # fetch this much more train text than needed (dedup / exhaustion slack)
    val_margin: float = 0.10


def preset(spec: MixSpec, name: str, **over) -> BuildConfig:
    d = dict(spec.presets[name])
    d.update({k: v for k, v in over.items() if v is not None})
    return BuildConfig(**{k: (v if k == "dataset" else float(v)) for k, v in d.items()})


def leaf_needs(spec: MixSpec, cfg: BuildConfig) -> dict:
    """Leaf -> (train bytes, val bytes) the fetch must collect."""
    T = max(cfg.train_bytes, cfg.tokenizer_sample_bytes or 0.0)
    out = {}
    for lf in spec.leaves:
        val = max(cfg.val_leaf_bytes, lf.group_share * cfg.val_group_bytes, lf.share * cfg.val_mix_bytes)
        out[lf.name] = (0.0 if lf.heldout else lf.share * T * (1 + cfg.train_margin), val * (1 + cfg.val_margin))
    return out


# ============================================================================ paths
def data_root() -> Path:
    from hnet.data.download import data_root as dr

    return dr()


def mix_dir(dataset: str, root=None) -> Path:
    return Path(root or data_root()) / dataset


def pool_dir(spec: MixSpec, root=None) -> Path:
    return Path(root or data_root()) / POOL_DIRNAME / f"{spec.name}-s{spec.seed}"


def known_mix_datasets() -> set:
    out = set()
    for name in KNOWN_SPECS:
        try:
            out |= {p["dataset"] for p in load_spec(name).presets.values()}
        except Exception:  # spec missing / unreadable: only built mixes are recognized
            pass
    return out


def is_mix(dataset: str, root=None) -> bool:
    m = mix_dir(dataset, root) / "manifest.json"
    if m.exists():
        try:
            return json.loads(m.read_text()).get("format") == MIX_FORMAT
        except (OSError, ValueError):
            return False
    from hnet.data.download import DATASETS

    return dataset not in DATASETS and dataset in known_mix_datasets()


def load_manifest(dataset: str, root=None) -> dict:
    p = mix_dir(dataset, root) / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found: build the mix first (scripts/data/build_mix.py)")
    return json.loads(p.read_text())


def _atomic_write_text(path: Path, text: str):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, arr):
    tmp = str(path)[:-4] + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


# ============================================================================ hashing / split
def content_hash(b: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little", signed=True)


def split_u(key: bytes, seed: int) -> float:
    """Uniform [0, 1) from (seed, key); val iff split_u < p_val."""
    h = hashlib.blake2b(key, digest_size=8, salt=int(seed).to_bytes(8, "little"), person=b"hnet-mix-split").digest()
    return int.from_bytes(h, "little") / 2.0 ** 64


def file_order(files: list, seed: int, source: str) -> list:
    files = sorted(files)
    rng = np.random.default_rng([int(seed), zlib.crc32(source.encode())])
    return [files[i] for i in rng.permutation(len(files))]


def unit_name(fi: int, rg: int) -> str:
    return f"{fi:05d}-{rg:06d}"


DONE_ALL = unit_name(99999, 999999)


def route_unit(spec: MixSpec, src: Source, cols: dict, stats: dict | None = None) -> dict:
    """Split one unit's rows into {leaf: (list of doc bytes, meta int64[n, 3])} (see module docstring)."""
    texts = cols[src.text]
    sel = cols.get(src.select) if src.select else None
    keys = cols.get(src.split_key) if src.split_key else None
    by_value = {spec.leaf(n).value: spec.leaf(n) for n in src.leaves} if src.select else None
    single = spec.leaf(src.leaves[0]) if not src.select else None
    out: dict = {}
    st = stats if stats is not None else {}
    for i, b in enumerate(texts):
        lf = by_value.get(sel[i]) if by_value is not None else single
        if lf is None or b is None:
            continue
        if isinstance(b, str):
            b = b.encode("utf-8")
        n = len(b)
        if n == 0 or n > spec.max_doc_bytes:
            st["dropped_size"] = st.get("dropped_size", 0) + 1
            continue
        try:
            b.decode("utf-8")
        except UnicodeDecodeError:
            st["dropped_utf8"] = st.get("dropped_utf8", 0) + 1
            continue
        k = keys[i] if keys is not None else None
        key = (k.encode() if isinstance(k, str) else k) if k else b
        is_val = lf.heldout or split_u(key, spec.seed) < src.p_val
        docs, meta = out.setdefault(lf.name, ([], []))
        docs.append(b)
        meta.append((n, content_hash(b), int(is_val)))
    return {k: (d, np.asarray(m, dtype=np.int64).reshape(-1, 3)) for k, (d, m) in out.items()}


# ============================================================================ fetch
class _Producer:
    """Reads units ahead of the consumer on a background thread (order preserved)."""

    def __init__(self, gen, depth: int = 2):
        self.q: queue.Queue = queue.Queue(maxsize=max(1, depth))
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, args=(gen,), daemon=True)
        self.t.start()

    def _put(self, item):
        while not self.stop.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self, gen):
        try:
            for item in gen:
                if not self._put(("item", item)):
                    return
            self._put(("end", None))
        except BaseException as e:  # surfaced in the consumer
            self._put(("error", e))

    def get(self):
        kind, item = self.q.get()
        if kind == "error":
            raise item
        return None if kind == "end" else item

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass
        self.t.join(timeout=30)


def _init_state(src: Source, spec: MixSpec, reader) -> dict:
    files = [f for f in reader.list_files() if any(fnmatch.fnmatch(f, g) for g in src.files)]
    assert files, f"source {src.name}: no files match {src.files}"
    return dict(source=src.name, repo=src.repo, revision=src.revision, seed=spec.seed,
                files=file_order(files, spec.seed, src.name), n_rg={}, pos=[0, 0], units=0, exhausted=False,
                bytes={n: [0, 0] for n in src.leaves}, docs={n: [0, 0] for n in src.leaves}, given_up=[],
                dropped={}, rows=0)


def source_state(spec: MixSpec, src_name: str, root=None) -> dict | None:
    p = pool_dir(spec, root) / "_state" / f"{src_name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def fetch_source(spec: MixSpec, src: Source, needs: dict, root, reader, log=print, give_up_units: int = 200,
                 prefetch: int = 2, log_every_s: float = 30.0) -> dict:
    pool = pool_dir(spec, root)
    sdir = pool / "_state"
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f"{src.name}.json"
    state = json.loads(spath.read_text()) if spath.exists() else _init_state(src, spec, reader)
    if state.get("revision") != src.revision or state.get("seed") != spec.seed:
        raise RuntimeError(f"{spath}: pool was fetched with revision {state.get('revision')} seed {state.get('seed')},"
                           f" spec says {src.revision} seed {spec.seed}; use another data root or delete the pool")
    for n in src.leaves:  # leaves added to the spec after the pool was started cannot be backfilled
        if n not in state["bytes"]:
            raise RuntimeError(f"{spath}: leaf {n} is not in this pool; delete the source's pool to refetch")

    def leaf_ok(n):
        return state["bytes"][n][0] >= needs[n][0] and state["bytes"][n][1] >= needs[n][1]

    def satisfied():
        return all(leaf_ok(n) for n in src.leaves if n not in state["given_up"])

    # Multi-leaf sources stop saving a leaf once it has its quotas ("capped"), so its pool stays a prefix of its
    # document stream without storing the surplus of common languages while the rare ones fill up. A capped
    # pool cannot grow later (the source has moved on), so a bigger build needs a fresh pool for that source.
    capped = state.setdefault("capped", {})
    short = [n for n in capped if not leaf_ok(n)]
    if short:
        raise RuntimeError(f"source {src.name}: leaves {short} were capped by an earlier, smaller build and cannot "
                           f"grow; use a fresh data root or delete {spath} and {[str(pool / n) for n in src.leaves]}")

    files = state["files"]
    cols = src.columns()

    def gen(fi, rg):
        while fi < len(files):
            f = files[fi]
            n = state["n_rg"].get(str(fi)) or reader.num_row_groups(f)
            if fi + 1 < len(files) and rg >= n // 2:  # past half of this file: start fetching the next one
                reader.prefetch(files[fi + 1])
            if rg >= n:
                reader.release(f)
                fi, rg = fi + 1, 0
                continue
            b = min(n, rg + src.rg_per_unit)
            yield fi, rg, b, n, reader.read(f, rg, b, cols)
            rg = b

    t0 = tlog = time.time()
    b0 = sum(sum(v) for v in state["bytes"].values())
    prod = None
    try:
        while not satisfied():
            if src.max_units and state["units"] >= src.max_units:
                log(f"[fetch] {src.name}: max_units={src.max_units} reached before the quotas")
                break
            if prod is None:
                prod = _Producer(gen(*state["pos"]), prefetch)
            item = prod.get()
            if item is None:
                state["exhausted"] = True
                log(f"[fetch] {src.name}: source exhausted")
                break
            fi, a, b, n, data = item
            state["n_rg"][str(fi)] = n
            routed = route_unit(spec, src, data, state["dropped"])
            uname = unit_name(fi, a)
            for leaf, (docs, meta) in routed.items():
                if leaf in capped:
                    continue
                d = pool / leaf
                d.mkdir(parents=True, exist_ok=True)
                tmp = d / f"{uname}.bin.tmp"
                with open(tmp, "wb") as fh:
                    for x in docs:
                        fh.write(x)
                os.replace(tmp, d / f"{uname}.bin")
                _atomic_save_npy(d / f"{uname}.npy", meta)
                for s in (0, 1):
                    sel = meta[:, META_VAL] == s
                    state["bytes"][leaf][s] += int(meta[sel, META_NBYTES].sum())
                    state["docs"][leaf][s] += int(sel.sum())
            if src.select:
                for leaf in src.leaves:
                    if leaf not in capped and leaf_ok(leaf):
                        capped[leaf] = uname
            state["rows"] += len(data[src.text])
            state["units"] += 1
            state["pos"] = [fi, b] if b < n else [fi + 1, 0]
            if src.select and state["units"] >= give_up_units:
                for leaf in src.leaves:
                    if sum(state["docs"][leaf]) == 0 and leaf not in state["given_up"]:
                        state["given_up"].append(leaf)
                        log(f"[fetch] {src.name}: WARNING no documents for {leaf} after {state['units']} units; "
                            f"giving up on it")
            _atomic_write_text(spath, json.dumps(state))
            if time.time() - tlog > log_every_s:
                tlog = time.time()
                got = sum(sum(v) for v in state["bytes"].values()) - b0
                prog = min(min(state["bytes"][k][0] / needs[k][0] if needs[k][0] else 1.0,
                               state["bytes"][k][1] / needs[k][1] if needs[k][1] else 1.0)
                           for k in src.leaves if k not in state["given_up"])
                log(f"[fetch] {src.name}: unit {state['units']} file {fi + 1}/{len(files)}, "
                    f"{got / 1e6:.1f} MB in {tlog - t0:.0f}s ({got / 1e6 / max(1e-9, tlog - t0):.1f} MB/s), "
                    f"least-filled leaf {100 * prog:.1f}%")
    finally:
        if prod is not None:
            prod.close()
        reader.close()
    _atomic_write_text(spath, json.dumps(state))
    return state


def fetch(spec: MixSpec, cfg: BuildConfig, root=None, reader_factory=None, workers: int = 4, log=print,
          sources: list | None = None, **kw) -> dict:
    """Fill the pools of every source (threads, one per source) up to the build's quotas."""
    from concurrent.futures import ThreadPoolExecutor

    if reader_factory is None:
        from hnet.data.hf_sources import ParquetReader

        def reader_factory(src):
            return ParquetReader(src.repo, src.revision, cache_dir=Path(root or data_root()) / "hf_cache")
    needs = leaf_needs(spec, cfg)
    todo = [s for s in spec.sources.values() if sources is None or s.name in sources]
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {s.name: ex.submit(fetch_source, spec, s, needs, root, reader_factory(s), log, **kw) for s in todo}
        return {k: f.result() for k, f in futs.items()}


# ============================================================================ pools -> plans
class LeafPool:
    """The fetched documents of one leaf (completed units only), in stream order."""

    def __init__(self, pool: Path, leaf: str, done_before: str):
        d = pool / leaf
        units = sorted(p.name[:-4] for p in d.glob("*.npy") if not p.name.endswith(".tmp.npy")
                       and p.name[:-4] < done_before) if d.exists() else []
        metas = [np.load(d / f"{u}.npy") for u in units]
        self.dir, self.units = d, units
        self.meta = np.concatenate(metas) if metas else np.zeros((0, 3), np.int64)
        self.unit_of = np.concatenate([np.full(len(m), i, np.int32) for i, m in enumerate(metas)]) if metas \
            else np.zeros(0, np.int32)
        offs = []
        for m in metas:
            o = np.zeros(len(m), np.int64)
            if len(m) > 1:
                np.cumsum(m[:-1, META_NBYTES], out=o[1:])
            offs.append(o)
        self.off = np.concatenate(offs) if offs else np.zeros(0, np.int64)
        self._mm: dict = {}

    def __len__(self):
        return len(self.meta)

    def read(self, k: int) -> bytes:
        u = int(self.unit_of[k])
        mm = self._mm.get(u)
        if mm is None:
            mm = self._mm[u] = np.memmap(self.dir / f"{self.units[u]}.bin", dtype=np.uint8, mode="r")
        o = int(self.off[k])
        return mm[o: o + int(self.meta[k, META_NBYTES])].tobytes()


def load_pools(spec: MixSpec, root=None) -> dict:
    pool = pool_dir(spec, root)
    out = {}
    for src in spec.sources.values():
        st = source_state(spec, src.name, root)
        done = DONE_ALL if st and st["exhausted"] else (unit_name(*st["pos"]) if st else unit_name(0, 0))
        for n in src.leaves:
            out[n] = LeafPool(pool, n, done)
    return out


def interleave(lens: list, weights: list, total: float | None):
    """Merge several ordered doc lists so every prefix has byte shares ~ weights (see module docstring).

    Returns (list index int64[n], position int64[n]) in stream order, stopping at the first doc where the
    cumulative bytes reach ``total`` (None: all docs)."""
    li, pos, key = [], [], []
    for i, (ln, w) in enumerate(zip(lens, weights)):
        ln = np.asarray(ln, dtype=np.int64)
        if w <= 0 or len(ln) == 0:
            continue
        excl = np.zeros(len(ln), np.float64)
        np.cumsum(ln[:-1], out=excl[1:])
        li.append(np.full(len(ln), i, np.int64))
        pos.append(np.arange(len(ln), dtype=np.int64))
        key.append((excl + 0.5 * ln) / w)  # a doc's midpoint: a long doc waits until half of it is "due"
    if not li:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    li, pos, key = np.concatenate(li), np.concatenate(pos), np.concatenate(key)
    order = np.lexsort((pos, li, key))
    li, pos = li[order], pos[order]
    if total is not None:
        sizes = np.empty(len(li), np.int64)
        for i in np.unique(li):
            m = li == i
            sizes[m] = np.asarray(lens[i], np.int64)[pos[m]]
        cum = np.cumsum(sizes)
        n = int(np.searchsorted(cum, total, side="left")) + 1  # first doc where the total is reached, inclusive
        li, pos = li[:n], pos[:n]
    return li, pos


@dataclass
class Plans:
    leaves: list  # leaf names, spec order
    train_docs: dict  # leaf -> int64 positions into the LeafPool (train docs, dedup'd), stream order
    val_docs: dict  # leaf -> int64 positions of val docs
    train: tuple  # (leaf idx, pool position) arrays of the train stream
    val_sets: dict  # set name -> (leaf idx, pool position)
    n_val_hashes: int = 0
    n_dedup_dropped: dict = field(default_factory=dict)


def make_plans(spec: MixSpec, cfg: BuildConfig, pools: dict, train_bytes: float | None = None) -> Plans:
    names = [lf.name for lf in spec.leaves]
    val_docs, train_docs = {}, {}
    val_hashes = np.unique(np.concatenate([pools[n].meta[pools[n].meta[:, META_VAL] == 1, META_HASH]
                                           for n in names] + [np.zeros(0, np.int64)]))
    dropped = {}
    for n in names:
        m = pools[n].meta
        val_docs[n] = np.flatnonzero(m[:, META_VAL] == 1)
        tr = np.flatnonzero(m[:, META_VAL] == 0)
        keep = ~np.isin(m[tr, META_HASH], val_hashes)
        dropped[n] = int((~keep).sum())
        train_docs[n] = tr[keep]

    def lens(docsd, n):
        return pools[n].meta[docsd[n], META_NBYTES]

    T = cfg.train_bytes if train_bytes is None else train_bytes
    li, p = interleave([lens(train_docs, n) for n in names], [spec.leaf(n).share for n in names], T)
    tdoc = np.empty(len(li), np.int64)
    for i in np.unique(li):
        m = li == i
        tdoc[m] = train_docs[names[i]][p[m]]
    train = (li, tdoc)
    val_sets = {}

    def vset(members, weights, total):
        a, b = interleave([lens(val_docs, n) for n in members], weights, total)
        idx = np.array([names.index(members[i]) for i in a], np.int64) if len(a) else a
        pp = np.empty(len(a), np.int64)
        for i in np.unique(a):
            m = a == i
            pp[m] = val_docs[members[i]][b[m]]
        return idx, pp

    inmix = [lf.name for lf in spec.in_mix]
    val_sets["mix"] = vset(inmix, [spec.leaf(n).share for n in inmix], cfg.val_mix_bytes)
    for cat, members in spec.categories_of_leaves().items():
        val_sets[cat] = vset(members, [spec.leaf(n).group_share for n in members], cfg.val_group_bytes)
    for n in names:
        val_sets[n] = vset([n], [1.0], cfg.val_leaf_bytes)
    return Plans(names, train_docs, val_docs, train, val_sets, int(len(val_hashes)), dropped)


# ============================================================================ aligned shard writer
def write_aligned_shard(out_dir: Path, stem: str, docs: list, leaf_ids, tok, batch_bytes: int = 16 << 20,
                        extra: dict | None = None) -> dict:
    """Write docs (bytes) as token shard + byte shard + doc index + leaf ids, verifying token->byte exactness."""
    from hnet.data.tokenizer import decode_with_table

    out_dir = Path(out_dir)
    bdir = out_dir / "bytes"
    bdir.mkdir(parents=True, exist_ok=True)
    assert tok.vocab_size <= 1 << 16, "token shards are uint16"
    n = len(docs)
    leaf_ids = np.asarray(leaf_ids, dtype=np.int16)
    ids_per_doc: list = []
    i = 0
    while i < n:
        j, nb = i, 0
        while j < n and (nb < batch_bytes or j == i):
            nb += len(docs[j])
            j += 1
        enc = tok.encode_batch([d.decode("utf-8") for d in docs[i:j]])
        flat = np.concatenate(enc) if enc else np.zeros(0, np.int64)
        if flat.size and (flat == tok.eot_id).any():
            raise RuntimeError(f"{stem}: document text produced the document-start token")
        dec = decode_with_table(flat, tok.table)
        if dec.tobytes() != b"".join(docs[i:j]):
            for k, e in enumerate(enc):  # locate the culprit for the error message
                if tok.decode_bytes(e) != docs[i + k]:
                    raise RuntimeError(f"{stem}: doc {i + k} does not round-trip through the tokenizer")
            raise RuntimeError(f"{stem}: token decode mismatch")
        ids_per_doc += enc
        i = j
    ntok = np.array([len(e) for e in ids_per_doc], np.int64)
    nby = np.array([len(d) for d in docs], np.int64)
    tok_stream = np.empty(int(ntok.sum()) + n, np.uint16)
    byte_stream = np.empty(int(nby.sum()) + 2 * n, np.uint8)
    idx = np.zeros((n, 5), np.int64)
    to = bo = 0
    for k in range(n):
        idx[k] = (bo, nby[k], to, ntok[k], S.HAS_BOS | S.HAS_EOS)
        tok_stream[to] = tok.eot_id
        tok_stream[to + 1: to + 1 + ntok[k]] = ids_per_doc[k]
        to += 1 + int(ntok[k])
        byte_stream[bo] = BOS_ID
        byte_stream[bo + 1: bo + 1 + nby[k]] = np.frombuffer(docs[k], np.uint8)
        byte_stream[bo + 1 + nby[k]] = EOS_ID
        bo += 2 + int(nby[k])
    S.write_token_shard(out_dir / f"{stem}.bin", tok_stream)
    bpath, ipath = S.byte_paths(bdir, stem)
    S.write_byte_shard(bpath, byte_stream, n, len(tok_stream), last_complete=True)
    _atomic_save_npy(ipath, idx)
    _atomic_save_npy(bdir / f"{stem}.src.npy", leaf_ids)
    per_leaf: dict = {}
    for lid in np.unique(leaf_ids):
        m = leaf_ids == lid
        per_leaf[int(lid)] = dict(docs=int(m.sum()), bytes=int(nby[m].sum()), tokens=int(ntok[m].sum()))
    st = dict(stem=stem, docs=n, bytes=int(nby.sum()), tokens=int(ntok.sum()), stream_bytes=len(byte_stream),
              stream_tokens=len(tok_stream), per_leaf=per_leaf, tokenizer_sha256=getattr(tok, "sha256", None),
              **(extra or {}))
    _atomic_write_text(bdir / f"{stem}.json", json.dumps(st))
    return st


def _shard_done(out_dir: Path, stem: str, key: str):
    j = out_dir / "bytes" / f"{stem}.json"
    if not j.exists():
        return None
    try:
        st = json.loads(j.read_text())
        if st.get("plan_key") != key:
            return None
        assert S.token_shard_ntok(out_dir / f"{stem}.bin") == st["stream_tokens"]
        assert S.byte_shard_header(out_dir / "bytes" / f"{stem}.bytes.bin")["nbytes"] == st["stream_bytes"]
        assert (out_dir / "bytes" / f"{stem}.idx.npy").exists() and (out_dir / "bytes" / f"{stem}.src.npy").exists()
        return st
    except (AssertionError, OSError, ValueError, KeyError):
        return None


def _plan_key(li, pos, tok_sha) -> str:
    h = hashlib.sha1(np.ascontiguousarray(li, np.int64).tobytes())
    h.update(np.ascontiguousarray(pos, np.int64).tobytes())
    h.update(str(tok_sha).encode())
    return h.hexdigest()


# worker-side caches (one LeafPool set + tokenizer per process)
_W: dict = {}


def _worker_init(threads: int | None):
    if threads:
        os.environ["RAYON_NUM_THREADS"] = str(threads)
        os.environ["RAYON_RS_NUM_CPUS"] = str(threads)


def _shard_job(args):
    spec_d, root, tok_path, out_dir, stem, li, pos, key, extra = args
    from hnet.data.tokenizer import load_tokenizer

    ck = (json.dumps(spec_d, sort_keys=True), str(root))
    if _W.get("key") != ck:
        spec = spec_from_dict(spec_d)
        _W.update(key=ck, pools=load_pools(spec, root), names=[lf.name for lf in spec.leaves])
    pools, names = _W["pools"], _W["names"]
    tok = load_tokenizer(tok_path)
    docs = [pools[names[i]].read(int(p)) for i, p in zip(li, pos)]
    return write_aligned_shard(Path(out_dir), stem, docs, li, tok, extra={"plan_key": key, **(extra or {})})


# ============================================================================ assemble
def _spec_raw(spec: MixSpec) -> dict:
    """Spec as the dict form spec_from_dict reads (for worker processes)."""
    d = dict(name=spec.name, seed=spec.seed, tokenizer=spec.tokenizer, max_doc_bytes=spec.max_doc_bytes,
             categories=spec.categories, sources={})
    for s in spec.sources.values():
        sd = dict(repo=s.repo, revision=s.revision, files=s.files, text=s.text, select=s.select,
                  split_key=s.split_key, rg_per_unit=s.rg_per_unit, p_val=s.p_val, max_units=s.max_units, leaves={})
        for n in s.leaves:
            lf = spec.leaf(n)
            sd["leaves"][n] = dict(value=lf.value, weight=lf.weight, heldout=lf.heldout)
        d["sources"][s.name] = sd
    return d


def assemble(spec: MixSpec, cfg: BuildConfig, tokenizer_path, root=None, workers: int = 1,
             threads: int | None = None, log=print) -> dict:
    """Pools -> train shards + val sets + manifest (resumable: finished shards with the same plan are kept)."""
    from hnet.data.tokenizer import load_tokenizer

    t0 = time.time()
    out = mix_dir(cfg.dataset, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bytes").mkdir(exist_ok=True)
    tok_src = Path(tokenizer_path)
    tok_dst = out / "tokenizer.json"
    if not tok_dst.exists() or tok_dst.read_bytes() != tok_src.read_bytes():
        shutil.copyfile(tok_src, tok_dst)
    tok = load_tokenizer(tok_dst)
    pools = load_pools(spec, root)
    plans = make_plans(spec, cfg, pools)
    names = plans.leaves
    warnings = []

    # ---- train shards
    li, pos = plans.train
    sizes = np.empty(len(li), np.int64)
    for i in np.unique(li):
        m = li == i
        sizes[m] = pools[names[i]].meta[pos[m], META_NBYTES]
    excl = np.cumsum(sizes) - sizes
    shard_of = (excl // int(cfg.shard_bytes)).astype(np.int64)
    n_shards = int(shard_of[-1]) + 1 if len(shard_of) else 0
    bounds = np.searchsorted(shard_of, np.arange(n_shards + 1))
    spec_d = _spec_raw(spec)
    jobs, results = [], {}
    for k in range(n_shards):
        a, b = int(bounds[k]), int(bounds[k + 1])
        stem = f"{cfg.dataset}_train_{k:06d}"
        key = _plan_key(li[a:b], pos[a:b], tok.sha256)
        st = _shard_done(out, stem, key)
        if st is not None:
            results[stem] = st
            continue
        jobs.append((spec_d, str(root) if root else None, str(tok_dst), str(out), stem, li[a:b], pos[a:b], key,
                     {"split": "train", "index": k}))
    for p in list(out.glob(f"{cfg.dataset}_train_*.bin")):  # stale shards of an earlier, longer plan
        k = int(p.name[len(cfg.dataset) + 7: -4])
        if k >= n_shards:
            for q in [p] + [out / "bytes" / f"{p.name[:-4]}{ext}" for ext in
                            (".bytes.bin", ".idx.npy", ".src.npy", ".json")]:
                q.unlink(missing_ok=True)
    val_stems = {}
    for name, (vli, vpos) in plans.val_sets.items():
        stem = f"{cfg.dataset}_val_000000" if name == "mix" else f"{cfg.dataset}_val-{name}_000000"
        val_stems[name] = stem
        if len(vli) == 0:
            warnings.append(f"val set {name} is empty")
            continue
        key = _plan_key(vli, vpos, tok.sha256)
        st = _shard_done(out, stem, key)
        if st is not None:
            results[stem] = st
            continue
        jobs.append((spec_d, str(root) if root else None, str(tok_dst), str(out), stem, vli, vpos, key,
                     {"split": "val", "set": name}))
    log(f"[assemble] {cfg.dataset}: {n_shards} train shards ({len(li):,} docs, {int(sizes.sum()) / 1e6:.1f} MB), "
        f"{len(plans.val_sets)} val sets; {len(jobs)} shards to write, {len(results)} already done")
    done_bytes, tw = 0, time.time()
    if workers > 1 and len(jobs) > 1:
        from multiprocessing import get_context

        threads = threads or max(1, (os.cpu_count() or 1) // workers)  # tokenizer (rayon) threads per worker
        with get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(threads,)) as pool:
            for st in pool.imap_unordered(_shard_job, jobs):
                results[st["stem"]] = st
                done_bytes += st["bytes"]
                log(f"[assemble] {st['stem']}: {st['docs']:,} docs {st['bytes'] / 1e6:.1f} MB "
                    f"{st['tokens']:,} tok ({done_bytes / 1e6 / max(1e-9, time.time() - tw):.1f} MB/s)")
    else:
        for job in jobs:
            st = _shard_job(job)
            results[st["stem"]] = st
            done_bytes += st["bytes"]
            log(f"[assemble] {st['stem']}: {st['docs']:,} docs {st['bytes'] / 1e6:.1f} MB {st['tokens']:,} tok "
                f"({done_bytes / 1e6 / max(1e-9, time.time() - tw):.1f} MB/s)")

    # ---- manifest
    train_stems = [f"{cfg.dataset}_train_{k:06d}" for k in range(n_shards)]
    leaf_train = {n: dict(docs=0, bytes=0, tokens=0) for n in names}
    for s in train_stems:
        for lid, v in results[s]["per_leaf"].items():
            for k2 in ("docs", "bytes", "tokens"):
                leaf_train[names[int(lid)]][k2] += v[k2]
    tot = {k2: sum(v[k2] for v in leaf_train.values()) for k2 in ("docs", "bytes", "tokens")}
    for n, v in leaf_train.items():
        lf = spec.leaf(n)
        v["bytes_per_token"] = v["bytes"] / v["tokens"] if v["tokens"] else None
        v["byte_share"] = v["bytes"] / tot["bytes"] if tot["bytes"] else 0.0
        v["token_share"] = v["tokens"] / tot["tokens"] if tot["tokens"] else 0.0
        v["target_share"] = lf.share
        if not lf.heldout and tot["bytes"] and abs(v["byte_share"] - lf.share) > max(0.002, 0.05 * lf.share):
            warnings.append(f"{n}: byte share {v['byte_share']:.4f} vs target {lf.share:.4f}")
    cats = {}
    for n, v in leaf_train.items():
        c = cats.setdefault(spec.leaf(n).category, dict(docs=0, bytes=0, tokens=0))
        for k2 in ("docs", "bytes", "tokens"):
            c[k2] += v[k2]
    for c in cats.values():
        c["bytes_per_token"] = c["bytes"] / c["tokens"] if c["tokens"] else None
        c["byte_share"] = c["bytes"] / tot["bytes"] if tot["bytes"] else 0.0
        c["token_share"] = c["tokens"] / tot["tokens"] if tot["tokens"] else 0.0
    vsets = {}
    for name, stem in val_stems.items():
        st = results.get(stem)
        if st is None:
            continue
        vsets[name] = dict(stem=stem, docs=st["docs"], bytes=st["bytes"], tokens=st["tokens"],
                           bytes_per_token=st["bytes"] / max(1, st["tokens"]),
                           leaves={names[int(k)]: v for k, v in st["per_leaf"].items()})
        want = cfg.val_mix_bytes if name == "mix" else (cfg.val_group_bytes if name in cats or name == "heldout"
                                                        else cfg.val_leaf_bytes)
        if st["bytes"] < 0.9 * want:
            warnings.append(f"val set {name}: {st['bytes']} bytes < target {int(want)}")
    if tot["bytes"] < 0.99 * cfg.train_bytes:
        warnings.append(f"train: {tot['bytes']} bytes < target {int(cfg.train_bytes)} (pools too small?)")
    srcs = {}
    for s in spec.sources.values():
        stt = source_state(spec, s.name, root) or {}
        srcs[s.name] = dict(repo=s.repo, revision=s.revision, units=stt.get("units"), rows=stt.get("rows"),
                            files_touched=(stt.get("pos", [0, 0])[0] + (stt.get("pos", [0, 0])[1] > 0)),
                            n_files=len(stt.get("files", [])), exhausted=stt.get("exhausted"),
                            given_up=stt.get("given_up"), dropped=stt.get("dropped"),
                            pool_bytes={k: v for k, v in (stt.get("bytes") or {}).items()})
    train_stream_bytes = sum(results[s]["stream_bytes"] for s in train_stems)
    train_stream_tokens = sum(results[s]["stream_tokens"] for s in train_stems)
    man = dict(
        format=MIX_FORMAT, dataset=cfg.dataset, spec=spec.name, spec_path=spec.path, seed=spec.seed,
        build=asdict(cfg), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        tokenizer=dict(file="tokenizer.json", source=str(tokenizer_path), sha256=tok.sha256,
                       vocab_size=tok.vocab_size, eot_id=tok.eot_id, pretokenize=tok.pretokenize),
        train=dict(n_shards=n_shards, shards=train_stems, **tot,
                   bytes_per_token=tot["bytes"] / max(1, tot["tokens"]),
                   stream_bytes=train_stream_bytes, stream_tokens=train_stream_tokens,
                   stream_bytes_per_token=train_stream_bytes / max(1, train_stream_tokens)),
        categories=cats, leaves={n: dict(category=spec.leaf(n).category, heldout=spec.leaf(n).heldout,
                                         value=spec.leaf(n).value, index=spec.leaf(n).index,
                                         train=leaf_train[n],
                                         val_pool=dict(docs=int(len(plans.val_docs[n])),
                                                       bytes=int(pools[n].meta[plans.val_docs[n], 0].sum())),
                                         dedup_dropped=plans.n_dedup_dropped.get(n, 0))
                                 for n in names},
        val_sets=vsets, sources=srcs, spec_full=spec.to_dict(), warnings=warnings,
        assemble_s=time.time() - t0)
    _atomic_write_text(out / "manifest.json", json.dumps(man, indent=1))
    for w in warnings:
        log(f"[assemble] WARNING {w}")
    return man


# ============================================================================ tokenizer training
def tokenizer_sample(spec: MixSpec, cfg: BuildConfig, root=None, sample_bytes: float | None = None):
    """(iterator over the first `sample_bytes` of the train stream as str, n_docs, n_bytes)."""
    pools = load_pools(spec, root)
    plans = make_plans(spec, cfg, pools, train_bytes=sample_bytes or cfg.tokenizer_sample_bytes)
    li, pos = plans.train
    names = plans.leaves
    nbytes = int(sum(pools[names[i]].meta[p, 0] for i, p in zip(li, pos)))

    def it():
        for i, p in zip(li, pos):
            yield pools[names[i]].read(int(p)).decode("utf-8")

    return it(), len(li), nbytes


def train_mix_tokenizer(spec: MixSpec, cfg: BuildConfig, out_path, root=None, sample_bytes: float | None = None,
                        vocab_size: int = 32768, pretokenize: bool = True, min_frequency: int = 2,
                        log=print) -> dict:
    from hnet.data.tokenizer import train_bpe

    it, n, nb = tokenizer_sample(spec, cfg, root, sample_bytes)
    log(f"[tokenizer] training {'BPE' if pretokenize else 'BPE-NP'} vocab={vocab_size} on {n:,} docs "
        f"({nb / 1e6:.1f} MB) of the {spec.name} train stream")
    t0 = time.time()
    js = train_bpe(it, vocab_size=vocab_size, pretokenize=pretokenize, min_frequency=min_frequency, length=n)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(js)
    info = dict(path=str(out_path), vocab_size=vocab_size, pretokenize=pretokenize, sample_docs=n,
                sample_bytes=nb, train_s=time.time() - t0, sha256=hashlib.sha256(js.encode()).hexdigest())
    log(f"[tokenizer] wrote {out_path} in {info['train_s']:.0f}s")
    return info


# ============================================================================ val sets for evaluation
def mix_valset_names(dataset: str, root=None) -> list:
    return list(load_manifest(dataset, root)["val_sets"])


def mix_valsets(dataset: str, root=None, names=None, **kw) -> dict:
    """{val set name: ValSet} for a built mix ("mix" = the combined set; leaves; categories)."""
    from hnet.data.valsets import ValSet

    from hnet.data.tokenizer import dataset_tokenizer

    man = load_manifest(dataset, root)
    tok = dataset_tokenizer(dataset, root)
    d = mix_dir(dataset, root)
    out = {}
    for name in names or list(man["val_sets"]):
        v = man["val_sets"][name]
        out[name] = ValSet(f"{dataset}:{name}", None, d / f"{v['stem']}.bin", tokenizer=tok, last_complete=True, **kw)
    return out


__all__ = ["MixSpec", "Leaf", "Source", "BuildConfig", "load_spec", "preset", "leaf_needs", "fetch", "assemble",
           "make_plans", "interleave", "route_unit", "write_aligned_shard", "train_mix_tokenizer", "is_mix",
           "mix_dir", "pool_dir", "load_manifest", "mix_valsets", "mix_valset_names", "MIX_FORMAT"]
