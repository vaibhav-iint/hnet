"""Name -> class registries for swappable components.

Every pluggable piece (mixers, routers, chunkers, dechunkers, aux losses, optimizers,
schedules, dense model variants) registers itself here, and configs refer to it by
name. Adding an idea = new file with ``@register("router", "my_router")`` + a config.

    from hnet.registry import register, build, names

    @register("router", "cosine_lookahead")
    class CosineLookaheadRouter(nn.Module): ...

    router = build("router", cfg.router, d_model=d, **cfg.router_kwargs)
"""

from __future__ import annotations

from typing import Callable, Dict

KINDS = (
    "mixer",       # sequence mixers: mamba2, attn, ... (layout codes map to these)
    "router",      # boundary predictors
    "chunker",     # downsampling given boundaries
    "dechunker",   # upsampling back to the outer resolution
    "aux_loss",    # ratio loss and alternatives
    "dense",       # isotropic token-level baselines
    "optimizer",
    "schedule",
)

_REGISTRY: Dict[str, Dict[str, Callable]] = {k: {} for k in KINDS}


def register(kind: str, name: str):
    if kind not in _REGISTRY:
        raise KeyError(f"unknown registry kind {kind!r}; known: {sorted(_REGISTRY)}")

    def deco(obj):
        table = _REGISTRY[kind]
        if name in table and table[name] is not obj:
            raise KeyError(f"{kind} {name!r} already registered by {table[name]!r}")
        table[name] = obj
        return obj

    return deco


def get(kind: str, name: str) -> Callable:
    _ensure_builtins_imported()
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"no {kind} named {name!r}; registered: {sorted(_REGISTRY[kind])}") from None


def build(kind: str, name: str, *args, **kwargs):
    return get(kind, name)(*args, **kwargs)


def names(kind: str) -> list:
    _ensure_builtins_imported()
    return sorted(_REGISTRY[kind])


_IMPORTED = False


def _ensure_builtins_imported():
    """Import the packages whose modules register built-in components."""
    global _IMPORTED
    if _IMPORTED:
        return
    _IMPORTED = True
    import importlib

    # Packages register their built-ins on import.
    for mod in ("hnet.model", "hnet.model.dense_modded", "hnet.model.mixers.deepseek", "hnet.model.mixers.kimi",
                "hnet.optim", "hnet.optim.muon"):
        importlib.import_module(mod)
