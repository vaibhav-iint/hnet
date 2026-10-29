"""Sweep specs (v2): load, substitute parameters, expand the grid x seeds into jobs, shard, compose configs.

A spec is a YAML file (``configs/search/*.yaml``)::

    name: exp003_lr_adamw                 # sweep id (GCS prefix, VM names, report titles)
    out_root: runs/search/exp003_lr_adamw # local root; each job trains into <out_root>/<job name>
    seeds: [0, 1, 2]
    set: [train.compile=false]            # --set overrides for every job
    grid:                                 # crossed with every run (cartesian product, in this key order)
      train.lr: [5.0e-4, 1.0e-3, 1.5e-3]
    grid_alias: {train.lr: lr}            # short names used in job names (default: last key component)
    params:                               # named values, used as ${name} / ${group.name} in set/grid/model/config
      best_lr: {bpe: 1.5e-3}              # null = must be supplied (--param best_lr.bpe=... or --params-file)
    params_note: "where the params come from"
    fairness: tuned                       # strict (default) | tuned | {mode: tuned, per_arch: [train.lr]}
    report: {baseline: bpe, groups: {Routers: [paper, end_mlp]}, labels: {bpe: BPE baseline}}
    gcs: gs://bucket/runs/exp003_lr_adamw # optional default for --gcs
    runs:                                 # one entry per architecture
      - {name: bpe, config: configs/runs/tiny_bpe_2m.yaml}
      - {name: end_mlp, config: configs/runs/tiny_hnet2_2m.yaml, set: [model.dc.router=end_mlp]}
      - {name: hnet_mla, config: configs/runs/tiny_hnet2_2m.yaml, model: configs/models/tiny_hnet2_2m_mla.yaml}
        # optional per run: seeds, grid (replaces the global grid; the tuned fairness gate rejects unequal grids)

Job name: ``<run>[__<alias><value>...]_s<seed>``, e.g. ``end_mlp__lr0.0015_s1`` (``<run>_s<seed>`` without a grid,
as in v1 specs). Each job's overrides are, in order: spec ``set``, run ``set``, the grid point, then
``train.seed=<seed>`` and ``train.out_dir=<out_root>/<job>``; ``model`` swaps the run file's model include
(trainer ``--model``).

Job order: seeds outermost, then runs, then grid points, so a partly finished sweep has every architecture and
setting at seed 0 first. ``--shard i/n`` takes jobs ``i, i+n, i+2n, ...`` of that order (0 <= i < n): a
deterministic split that every VM computes identically from the same spec.

Missing config files are explicit: :meth:`Spec.jobs` raises :class:`SpecError` listing every missing file and the
runs that need it, unless ``allow_missing=True``, which drops those runs (listed in ``Spec.missing``).
"""

from __future__ import annotations

import copy
import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

SPEC_KEYS = {"name", "out_root", "seeds", "set", "grid", "grid_alias", "params", "params_note", "fairness",
             "report", "gcs", "description", "runs"}
RUN_KEYS = {"name", "config", "model", "set", "seeds", "grid", "note"}
FAIRNESS_MODES = ("strict", "tuned")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_PARAM_RE = re.compile(r"\$\{([A-Za-z0-9_.-]+)\}")


class SpecError(ValueError):
    pass


@dataclass
class Job:
    name: str  # run directory name under out_root
    arch: str  # the spec run entry (architecture / variant)
    point: Dict[str, object]  # grid key -> value
    seed: int
    config: str
    model: Optional[str]
    sets: List[str]
    out: str
    index: int = 0  # position in the full ordered job list (sharding: index % n)

    @property
    def setting(self) -> str:
        """Grid point label (``train.lr=0.0015``; ``-`` without a grid)."""
        return ", ".join(f"{k}={fmt_value(v)}" for k, v in self.point.items()) or "-"

    def trainer_args(self) -> List[str]:
        args = ["-m", "hnet.train.train", "--config", self.config]
        if self.model:
            args += ["--model", self.model]
        for s in self.sets:
            args += ["--set", s]
        return args

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Spec:
    path: str
    raw: dict
    params: dict
    missing: List[Tuple[str, str]] = field(default_factory=list)  # (run, path) dropped by allow_missing

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str, params: Optional[List[str]] = None, params_file: Optional[str] = None) -> "Spec":
        """``params``: ``group.name=value`` strings; ``params_file``: YAML mapping merged into the spec's
        ``params`` (e.g. ``<exp003 out_root>/report/best_params.yaml``). Precedence: spec < file < ``params``."""
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        unknown = set(raw) - SPEC_KEYS
        if unknown:
            raise SpecError(f"{path}: unknown spec keys {sorted(unknown)} (allowed: {sorted(SPEC_KEYS)})")
        for k in ("name", "out_root", "runs"):
            if k not in raw:
                raise SpecError(f"{path}: missing required key {k!r}")
        p = copy.deepcopy(raw.get("params") or {})
        if params_file:
            with open(params_file) as f:
                p = _merge(p, yaml.safe_load(f) or {})
        for kv in params or []:
            key, sep, val = kv.partition("=")
            if not sep:
                raise SpecError(f"--param must look like name=value, got {kv!r}")
            _set_dotted(p, key.strip(), _coerce(yaml.safe_load(val)))
        spec = cls(path=str(path), raw=raw, params=p)
        spec._validate()
        return spec

    def _validate(self):
        names = set()
        for r in self.raw["runs"]:
            unknown = set(r) - RUN_KEYS
            if unknown:
                raise SpecError(f"run {r.get('name')!r}: unknown keys {sorted(unknown)} (allowed: {sorted(RUN_KEYS)})")
            if "name" not in r or "config" not in r:
                raise SpecError(f"every run needs name and config: {r}")
            if not _NAME_RE.match(str(r["name"])) or "__" in str(r["name"]):
                raise SpecError(f"run name {r['name']!r}: use letters, digits, _ . + - (no '__')")
            if r["name"] in names:
                raise SpecError(f"duplicate run name {r['name']!r}")
            names.add(r["name"])
        self.fairness_mode()  # validates

    # ------------------------------------------------------------------ properties
    @property
    def name(self) -> str:
        return str(self.raw["name"])

    @property
    def out_root(self) -> str:
        return str(self.raw["out_root"])

    @property
    def seeds(self) -> List[int]:
        return [int(s) for s in self.raw.get("seeds", [0])]

    def fairness_mode(self) -> str:
        f = self.raw.get("fairness", "strict")
        mode = f.get("mode", "strict") if isinstance(f, dict) else f
        if mode not in FAIRNESS_MODES:
            raise SpecError(f"fairness mode must be one of {FAIRNESS_MODES}, got {mode!r}")
        if isinstance(f, dict):
            extra = set(f) - {"mode", "per_arch"}
            if extra:
                raise SpecError(f"unknown fairness keys {sorted(extra)}")
        return mode

    def per_arch_keys(self) -> List[str]:
        f = self.raw.get("fairness")
        return list(f.get("per_arch") or []) if isinstance(f, dict) else []

    def report_cfg(self) -> dict:
        return dict(self.raw.get("report") or {})

    def run_names(self) -> List[str]:
        return [r["name"] for r in self.raw["runs"]]

    def grid_of(self, run: dict) -> Dict[str, list]:
        g = run["grid"] if "grid" in run else (self.raw.get("grid") or {})
        if not isinstance(g, dict):
            raise SpecError(f"grid must be a mapping key -> list of values, got {g!r}")
        out = {}
        for k, vals in g.items():
            vals = self.subst(vals)
            if not isinstance(vals, list) or not vals:
                raise SpecError(f"grid {k}: needs a non-empty list of values, got {vals!r}")
            out[k] = [_coerce(v) for v in vals]
        return out

    def aliases(self, keys) -> Dict[str, str]:
        given = self.raw.get("grid_alias") or {}
        al = {k: str(given.get(k, k.split(".")[-1])) for k in keys}
        if len(set(al.values())) != len(al):
            raise SpecError(f"grid aliases collide: {al}; set grid_alias")
        for a in al.values():
            if not _NAME_RE.match(a):
                raise SpecError(f"grid alias {a!r}: letters, digits, _ . + - only")
        return al

    # ------------------------------------------------------------------ params
    def subst(self, obj):
        """Replace ``${name}`` references by parameter values (a string that is exactly one reference keeps the
        value's type). Unresolved or null parameters raise, listing every offender."""
        missing: List[str] = []
        out = self._subst(obj, missing)
        if missing:
            raise SpecError(f"{self.path}: parameters not set: {sorted(set(missing))}. Fill them in the spec's "
                            f"`params:` (see params_note), or pass --param NAME=VALUE / --params-file FILE"
                            + (f". Note: {self.raw.get('params_note')}" if self.raw.get("params_note") else ""))
        return out

    def _subst(self, obj, missing):
        if isinstance(obj, list):
            return [self._subst(o, missing) for o in obj]
        if isinstance(obj, dict):
            return {k: self._subst(v, missing) for k, v in obj.items()}
        if not isinstance(obj, str) or "${" not in obj:
            return obj
        whole = _PARAM_RE.fullmatch(obj)
        if whole:
            v = self._param(whole.group(1), missing)
            return v

        def rep(m):
            v = self._param(m.group(1), missing)
            return "" if v is None else fmt_yaml(v)

        return _PARAM_RE.sub(rep, obj)

    def _param(self, name: str, missing: list):
        node = self.params
        for part in name.split("."):
            if not isinstance(node, dict) or part not in node:
                missing.append(name)
                return None
            node = node[part]
        if node is None:
            missing.append(name)
        return node

    def param_refs(self, run: dict) -> Dict[str, str]:
        """{config key: param name} for run-level ``set`` entries whose value is exactly ``${param}``."""
        out = {}
        for s in run.get("set", []) or []:
            key, _, val = str(s).partition("=")
            m = _PARAM_RE.fullmatch(val.strip())
            if m:
                out[key.strip()] = m.group(1)
        return out

    # ------------------------------------------------------------------ files
    def missing_files(self) -> List[Tuple[str, str]]:
        """(run name, path) for every config/model file that does not exist (after parameter substitution)."""
        out = []
        for r in self.raw["runs"]:
            for k in ("config", "model"):
                if r.get(k):
                    p = self.subst(r[k])
                    if not resolve_path(p).exists():
                        out.append((r["name"], p))
        return out

    # ------------------------------------------------------------------ expansion
    def jobs(self, only=None, shard: Optional[Tuple[int, int]] = None, allow_missing: bool = False) -> List[Job]:
        """Ordered jobs (see module docstring). ``only``: run names to keep (applied after sharding, so shard
        membership never depends on ``only``)."""
        miss = self.missing_files()
        if miss and not allow_missing:
            lines = "\n".join(f"  {p}  (run {r})" for r, p in miss)
            raise SpecError(f"{self.path}: {len(miss)} config file(s) missing:\n{lines}\n"
                            f"(pass --allow-missing to check/run the other runs only)")
        self.missing = miss
        skip = {r for r, _ in miss}
        jobs = []
        for ri, r in enumerate(self.raw["runs"]):
            grid = self.grid_of(r)
            al = self.aliases(grid)
            seeds = [int(s) for s in r.get("seeds", self.seeds)]
            points = [dict(zip(grid, combo)) for combo in itertools.product(*grid.values())] if grid else [{}]
            run_sets = [str(s) for s in self.subst(list(self.raw.get("set", []) or []) + list(r.get("set", []) or []))]
            for si, seed in enumerate(seeds):
                for pi, point in enumerate(points):
                    suffix = "".join(f"__{al[k]}{name_value(v)}" for k, v in point.items())
                    name = f"{r['name']}{suffix}_s{seed}"
                    out = str(Path(self.out_root) / name)
                    sets = run_sets + [f"{k}={fmt_yaml(v)}" for k, v in point.items()] + [
                        f"train.seed={seed}", f"train.out_dir={out}"]
                    model = self.subst(r["model"]) if r.get("model") else None
                    jobs.append(((si, ri, pi), Job(name=name, arch=r["name"], point=point, seed=seed,
                                                   config=str(self.subst(r["config"])), model=model, sets=sets,
                                                   out=out)))
        jobs.sort(key=lambda t: t[0])
        ordered = [j for _, j in jobs]
        names = [j.name for j in ordered]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise SpecError(f"job names collide: {sorted(dup)} (grid values that format identically?)")
        for i, j in enumerate(ordered):
            j.index = i
        if shard is not None:
            i, n = shard
            ordered = [j for j in ordered if j.index % n == i]
        ordered = [j for j in ordered if j.arch not in skip]
        if only:
            unknown = set(only) - set(self.run_names())
            if unknown:
                raise SpecError(f"--only: unknown run names {sorted(unknown)}")
            ordered = [j for j in ordered if j.arch in only]
        return ordered

    def manifest(self, allow_missing: bool = False) -> dict:
        """Everything a reader of the sweep's results needs (written to <out_root>/sweep_manifest.json)."""
        jobs = self.jobs(allow_missing=allow_missing)
        return {"name": self.name, "spec": self.path, "fairness": self.fairness_mode(), "seeds": self.seeds,
                "runs": self.run_names(), "missing": self.missing, "report": self.report_cfg(),
                "jobs": [j.to_json() for j in jobs]}


# ---------------------------------------------------------------------- helpers
def parse_shard(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    m = re.fullmatch(r"(\d+)/(\d+)", s.strip())
    if not m:
        raise SpecError(f"--shard must look like i/n (0-based), got {s!r}")
    i, n = int(m.group(1)), int(m.group(2))
    if n < 1 or not 0 <= i < n:
        raise SpecError(f"--shard {s}: need 0 <= i < n")
    return i, n


def resolve_path(p: str) -> Path:
    q = Path(p)
    if q.is_absolute() or q.exists():
        return q
    return REPO_ROOT / q


def compose_config(job: Job) -> dict:
    """The exact training config of a job (what the trainer will run)."""
    from hnet.train.config import load_run_config

    model = str(resolve_path(job.model)) if job.model else None
    return load_run_config(str(resolve_path(job.config)), job.sets, model=model)


def fmt_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "-".join(fmt_value(x) for x in v)
    return str(v)


def name_value(v) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]", "-", fmt_value(v))


def fmt_yaml(v) -> str:
    """A value as YAML flow text for ``--set key=<text>`` (round-trips through apply_overrides)."""
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return v
    return yaml.safe_dump(v, default_flow_style=True, width=10_000).strip().removesuffix("...").strip()


def _coerce(v):
    """YAML 1.1 reads 5e-4 as a string; treat numeric-looking strings as floats (same rule as --set)."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _set_dotted(d: dict, key: str, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def status_filename(shard: Optional[Tuple[int, int]] = None, only=None) -> str:
    """Per-selection status file, so several VMs mirroring into one GCS prefix never overwrite each other."""
    if shard is not None:
        return f"sweep_status.shard-{shard[0]}-of-{shard[1]}.json"
    if only:
        import hashlib

        return f"sweep_status.only-{hashlib.sha1(','.join(sorted(only)).encode()).hexdigest()[:8]}.json"
    return "sweep_status.json"


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
