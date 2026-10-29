"""Run-config loading: yaml ``include:`` composition, ``--set`` overrides and defaults.

Composition rules
-----------------
* Top level: ``include: path`` or ``include: [p1, p2]``. Included files are deep-merged in order, then the
  including file's own keys are merged on top (local wins).
* Section level: ``model: {include: configs/models/hnet_2stage_125m.yaml, dc: {router: ...}}``. If the
  included file has a top-level key equal to the section name (``model:``), that sub-dict is used, else the
  whole file. Local keys override (deep merge; lists are replaced, not merged).
* A dict whose ``name`` differs from the one it overrides (e.g. ``schedule: {name: constant}`` over a ``wsd``
  schedule) replaces it instead of merging, so stale keys do not leak between components.
* Paths are resolved relative to the including file's directory first, then to the current directory,
  then to the repository root. Includes nest.

Model swap: ``load_run_config(path, model="configs/models/x.yaml")`` (trainer flag ``--model``) replaces the path
of the ``model: {include: ...}`` section include(s) with ``x.yaml``; the run file's other ``model:`` keys (e.g.
``lr_multiplier_n_gpt: data``) still apply on top. A run file without a ``model`` include is an error.

Precedence (later wins), see :func:`load_run_config`: run file (+ includes, + model swap) < ``flags`` (typed
``(key, value)`` pairs, e.g. from the trainer's first-class flags) < ``overrides`` (``a.b=c`` strings, applied
left to right with :func:`hnet.config.apply_overrides`) < nothing: defaults only fill keys that are still missing.
"""

from __future__ import annotations

import copy
import os
from typing import Any, Optional

import yaml

from hnet.config import DenseConfig, HNetConfig, apply_overrides

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if (isinstance(v, dict) and isinstance(out.get(k), dict)
                and v.get("name", out[k].get("name")) == out[k].get("name")):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: str, rel_dir: Optional[str]) -> str:
    cands = [path] if os.path.isabs(path) else [
        os.path.join(rel_dir or ".", path), os.path.join(os.getcwd(), path), os.path.join(REPO_ROOT, path)]
    for c in cands:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"include {path!r} not found (tried {cands})")


def _load_file(path: str, _stack=(), swaps: Optional[dict] = None) -> dict:
    path = os.path.abspath(path)
    assert path not in _stack, f"include cycle: {_stack + (path,)}"
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    return _expand(d, os.path.dirname(path), _stack + (path,), section=None, swaps=swaps)


def _expand(d: Any, rel_dir: str, stack, section: Optional[str], swaps: Optional[dict] = None) -> Any:
    """``swaps``: {section: [abs path, n_used]}: a section-level include of that section is replaced by the path."""
    if not isinstance(d, dict):
        return d
    d = dict(d)
    inc = d.pop("include", None)
    base: dict = {}
    if inc is not None:
        if section is not None and swaps and section in swaps:
            paths = [swaps[section][0]]
            swaps[section][1] += 1
        else:
            paths = [_resolve(p, rel_dir) for p in ([inc] if isinstance(inc, str) else list(inc))]
        for p in paths:
            sub = _load_file(p, stack, swaps)
            if section is not None and isinstance(sub.get(section), dict):
                sub = sub[section]
            base = deep_merge(base, sub)
    local = {k: (_expand(v, rel_dir, stack, section=k, swaps=swaps) if isinstance(v, dict) else v)
             for k, v in d.items()}
    return deep_merge(base, local)


def set_key(d: dict, key: str, value) -> dict:
    """``d[a][b][c] = value`` for ``key = "a.b.c"`` (creates intermediate dicts; value used as is)."""
    node = d
    parts = key.split(".")
    for p in parts[:-1]:
        nxt = node.setdefault(p, {})
        if not isinstance(nxt, dict):
            raise ValueError(f"cannot set {key!r}: {p!r} is {nxt!r}, not a mapping")
        node = nxt
    node[parts[-1]] = copy.deepcopy(value)
    return d


def get_key(d: dict, key: str, default=None):
    node = d
    for p in key.split("."):
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node


def override_key(ov: str) -> str:
    key, sep, _ = ov.partition("=")
    if not sep:
        raise ValueError(f"override must look like key=value, got {ov!r}")
    return key.strip()


def check_override_conflicts(flags, overrides):
    """A key set by a first-class flag and by ``--set`` (or one being a prefix of the other, e.g. ``--warmup``
    vs ``--set train.schedule={...}``) must not disagree silently: raise unless the values are identical."""
    flag_vals = dict(flags or [])
    problems = []
    for ov in overrides or []:
        k = override_key(ov)
        for fk, fv in flag_vals.items():
            if k == fk:
                val = apply_overrides({}, [ov])
                if get_key(val, k) != fv:
                    problems.append(f"{k}: flag value {fv!r} vs --set {ov!r}")
            elif k.startswith(fk + ".") or fk.startswith(k + "."):
                problems.append(f"--set {ov!r} overlaps the flag-set key {fk!r}")
    if problems:
        raise ValueError("conflicting flag and --set values (use one or the other):\n  " + "\n  ".join(problems))


def load_run_config(path: Optional[str] = None, overrides=None, base: Optional[dict] = None,
                    model: Optional[str] = None, flags=None) -> dict:
    """Compose a run config. Order (later wins): yaml ``path`` with its includes (``model`` swaps the path of
    the ``model:`` section include) or dict ``base`` (merged on top of ``path`` if both); ``flags``: typed
    ``(dotted key, value)`` pairs; ``overrides``: ``a.b=c`` strings (YAML values) left to right; finally
    defaults fill missing keys. Flags and overrides that disagree on a key raise (:func:`check_override_conflicts`).
    """
    swaps = None
    if model is not None:
        if not path:
            raise ValueError("model= (--model) needs a run yaml whose model section uses include:")
        swaps = {"model": [_resolve(model, None), 0]}
    d = _load_file(path, swaps=swaps) if path else _expand(copy.deepcopy(base or {}), os.getcwd(), (), None)
    if swaps is not None and swaps["model"][1] == 0:
        raise ValueError(f"--model {model}: {path} has no `model: {{include: ...}}` section to swap")
    if base and path:
        d = deep_merge(d, _expand(copy.deepcopy(base), os.getcwd(), (), None))
    check_override_conflicts(flags, overrides)
    for k, v in flags or []:
        set_key(d, k, v)
    apply_overrides(d, overrides)
    return fill_defaults(d)


DEFAULTS = {
    "model": {},
    "data": {
        "source": "hnet",  # hnet (WS-C shards) | synthetic | text
        "dataset": "finewebedu10B",  # hnet.data dataset name (finewebedu10B | fineweb10B | fineweb100B)
        "root": None,  # data root (default $HNET_DATA or <repo>/data)
        "paths": None,  # explicit shard list (overrides dataset/root)
        "byte_paths": None,  # explicit byte shards (byte models; wins over paths)
        "token_paths": None,  # explicit token shards (token models; wins over paths)
        "val_token_path": None,  # explicit val token shard for the ValSet (default: the dataset's val shard)
        "cross_window_label": True,  # last window position predicts the next stream element (same doc only)
        "n_train_shards": None,  # use only the first n train shards
        "shuffle_buffer": 0,  # WindowLoader block-shuffle size in windows (0 = stream order)
        "kind": None,  # bytes | tokens (default: bytes for hnet / vocab<=256, tokens otherwise)
        "seq_len": 8192,
        "micro_batch": 1,  # windows per micro-step per rank
        "batch_bytes": None,  # global bytes per optimizer step (byte models)
        "batch_tokens": None,  # global tokens per optimizer step (token models)
        "batch_windows": None,  # alternative: global windows per step
        "bytes_per_token": None,  # tokens_eq for byte models = bytes / this (default: measured, bpt_stats.json)
        "num_workers": 1,  # prefetch threads (0 = synchronous)
        "prefetch": 4,
    },
    "train": {
        "out_dir": "runs/default",
        "seed": 0,
        "optimizer": "adamw",
        "lr": 6e-4,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "weight_decay": 0.1,
        "wd_embeddings": True,
        "no_wd_1d": True,
        "schedule": {"name": "wsd", "warmup": 0.1, "decay": 0.2, "decay_shape": "inverse_sqrt", "min_lr_ratio": 0.0},
        "steps": None,
        "train_tokens": None,  # alternative to steps: tokens_eq budget
        "train_bytes": None,  # alternative to steps: byte budget
        "stop_at_step": None,  # stop (with checkpoint) before `steps`, keeping the schedule
        "grad_accum": "auto",
        "grad_clip": 1.0,
        "precision": "bf16",  # bf16 (autocast on CUDA, fp32 master weights) | fp32
        "compile": False,  # false | blocks | model
        "compile_dynamic": None,  # None = auto (True for H-Net)
        "compile_backend": "inductor",  # torch.compile backend (aot_eager / eager: debugging graph breaks)
        "compile_mode": None,  # torch.compile mode (None = default; e.g. max-autotune-no-cudagraphs)
        "activation_checkpointing": False,
        "max_hours": None,
        "ckpt_every": 1000,
        "ckpt_every_tokens": None,  # permanent checkpoints every N tokens_eq (data-multiplier runs)
        "ckpt_steps": [],  # permanent checkpoints at these steps
        "ckpt_tokens": [],  # permanent checkpoints when tokens_eq first reaches each of these values
        "ckpt_keep": 2,  # rolling checkpoints kept
        "save_final": True,
        "resume": "auto",  # auto | <ckpt dir> | <run dir> (its latest ckpt) | null
        "resume_strict": True,  # refuse to resume if model/data/optimizer/schedule config differs from the ckpt's
        "stop_on_nonfinite": 3,  # stop (diverged.json) after N consecutive non-finite logged losses; 0 = never
        "ddp_bucket_mb": 25,
        "check_signals_every": 10,  # DDP: steps between cross-rank stop-flag checks
        "device": None,  # default cuda if available
    },
    "eval": {
        "every": 1000,
        "at_end": True,
        "val": "standard",  # standard (hnet.data.standard_valset) | speedrun | none
        "val_bytes": 10_000_000,  # standard valset size (whole docs, text bytes)
        "val_byte_seq_len": 8192,  # ValSet chunking; keep identical across compared runs
        "val_token_seq_len": 1792,
        "windows_per_batch": 8,  # val windows per forward
        "max_docs": 200,  # builtin (synthetic/text) val only
        "batch_bytes": 65536,  # builtin val only: positions per batch
        "downstream": None,  # e.g. {every: 5000, tasks: [hellaswag], limit: 500}
    },
    "log": {
        "every": 10,
        "wandb": None,  # project name; requires WANDB_API_KEY
        "run_name": None,
    },
}


def fill_defaults(d: dict) -> dict:
    out = deep_merge(DEFAULTS, d)
    return out


def model_config_from_section(section: dict):
    d = dict(section)
    kind = d.pop("type", "hnet")
    d.pop("factory", None)
    if kind == "hnet":
        return HNetConfig.from_dict(d)
    if kind == "dense":
        return DenseConfig.from_dict(d)
    raise ValueError(f"model.type must be hnet|dense, got {kind!r}")


def dump_yaml(d: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)
