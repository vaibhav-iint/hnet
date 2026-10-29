"""Muon (MomentUm Orthogonalized by Newton-Schulz) for hidden matrices + Adam(W) for everything else.

Registered as ``("optimizer", "muon")`` -> :func:`build_muon_optimizer`. Algorithm and defaults follow
modded-nanogpt speedrun record #18 (2025-01-04), our S1 baseline (docs/SPEEDRUN.md):

* Muon: SGD momentum (0.95, warmed up linearly from 0.85 over the first 300 steps), Nesterov, then the update
  is orthogonalized with 5 quintic Newton-Schulz iterations (coefficients 3.4445, -4.7750, 2.0315; bf16 on CUDA,
  fp32 on CPU), scaled by ``sqrt(max(1, rows / cols))`` and applied with lr 0.05. Optional decoupled weight decay.
* Adam on the rest: betas (0.8, 0.95), eps 1e-8, lr 0.6 for embeddings, 0.008 for the LM head, 0.04 for
  everything else (norm weights, scalars, biases, conv filters, ...). No weight decay by default.

API
---
``build_muon_optimizer(model, train_cfg, param_groups_fn=None) -> MuonAdamW``

* ``train_cfg``: the run yaml's ``train`` section (dict or object). Only its ``muon`` sub-section is read; keys
  and defaults are ``MUON_DEFAULTS`` (unknown keys raise). ``train.lr`` / ``train.weight_decay`` are *not*
  used: every group carries its kind's ``base_lr`` and its ``lr_mult``, and the trainer's schedule sets
  ``lr = base_lr * lr_mult * factor(step)`` (``hnet.optim.build.set_lr``).
* Parameters are split by :func:`classify_params` into four kinds, each covered exactly once:

  ======== =========================================================== ==========
  kind     parameters                                                  optimizer
  ======== =========================================================== ==========
  muon     2-D weights of hidden layers: attention q/k/v/out, MLP,     Muon
           Mamba-2 ``in_proj`` / ``out_proj``, H-Net ``residual_proj``
  embed    weights of ``nn.Embedding`` modules (incl. value embeds)    Adam
  head     parameters of modules named ``lm_head``                     Adam
  other    norms, scalars/lambdas, biases, Mamba-2 ``conv1d``,         Adam
           ``A_log``, ``D``, ``dt_bias``; router projections unless
           ``muon.router == "muon"``
  ======== =========================================================== ==========

  Router parameters are those under a module whose name contains ``routing_module`` or ``router``.
* ``param_groups_fn(model, filter_fn=..., weight_decay=...) -> list[dict]`` (optional; WS-D's
  ``hnet.optim.make_param_groups_fn``) sub-divides one kind's parameters (``filter_fn(name, p)`` selects them)
  into groups by H-Net stage and decay/no-decay. Its groups' ``lr_mult`` (per-stage LR multiplier) multiplies
  the kind's lr, its ``weight_decay`` (the kind's value, or 0 for biases/norms/1-D params) is kept, and other
  keys (``names``, ``stage``) pass through. Without it: one group per kind, ``lr_mult`` 1.
* Every group has ``use_muon`` (bool) and ``kind``; Muon groups also keep their own ``step`` counter (used for
  the momentum warmup, saved in ``state_dict``).

Distributed: under DDP the gradients are already all-reduced, so every rank runs the same Muon update on every
matrix (redundant but exact). The record instead gives each rank 1/world_size of the matrices and all-gathers
the orthogonalized updates, which removes the redundant Newton-Schulz work; later records go further
(reduce-scatter of gradients, sharded momentum, batched NS on stacked same-shape matrices). Worth doing only when
Muon's share of step time matters (for 124M on 1 GPU it is a few percent).
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch
from torch import nn

from hnet.registry import register

NS_COEFFS = (3.4445, -4.7750, 2.0315)

MUON_DEFAULTS = dict(
    lr=0.05,  # Muon lr for hidden matrices
    momentum=0.95,
    momentum_warmup_start=0.85,  # momentum ramps linearly from this value ...
    momentum_warmup_steps=300,  # ... to `momentum` over this many steps (0 = off)
    nesterov=True,
    ns_steps=5,
    weight_decay=0.0,  # decoupled, Muon params
    router="adam",  # adam | muon: optimizer for 2-D router projections (H-Net)
    lr_embed=0.6,
    lr_head=0.008,
    lr_other=0.04,
    adam_betas=(0.8, 0.95),
    adam_eps=1e-8,
    adam_weight_decay=0.0,  # decoupled, Adam params
    # Decoupled weight decay for the H-Net residual bypass matrices (``*.residual_proj.weight``), which then form
    # their own Muon group(s) (``role: residual``). None = no split (the default; unchanged behaviour). Muon's
    # update has spectral norm ~lr whatever the gradient's size, so a zero-init bypass grows by sum(lr) with nothing
    # anchoring its scale; decoupled decay wd gives it a fixed point at spectral norm ~1/wd (independent of the LR).
    residual_wd=None,
)

KINDS = ("muon", "embed", "head", "other")


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7, dtype=None) -> torch.Tensor:
    """Approximately orthogonalize G (replace USV^T by ~US'V^T with S' in roughly [0.7, 1.2]) with a quintic
    Newton-Schulz iteration whose coefficients maximize the slope at zero (Keller Jordan's NS5). Works on the
    last two dims; runs in bf16 on CUDA and fp32 elsewhere unless ``dtype`` is given. Returns that dtype."""
    assert G.ndim >= 2
    a, b, c = NS_COEFFS
    if dtype is None:
        dtype = torch.bfloat16 if G.is_cuda else torch.float32
    X = G.to(dtype)
    tall = G.size(-2) > G.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)  # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if tall else X


class MuonAdamW(torch.optim.Optimizer):
    """One optimizer holding Muon groups (``use_muon=True``) and Adam(W) groups (``use_muon=False``).

    Muon group keys: lr, momentum, momentum_warmup_start, momentum_warmup_steps, nesterov, ns_steps,
    weight_decay. Adam group keys: lr, betas, eps, weight_decay. Missing keys take ``MUON_DEFAULTS``. Every group
    also gets ``lr_mult`` (default 1) and ``base_lr`` (default ``lr / lr_mult``) for the trainer's schedule.
    """

    def __init__(self, param_groups: Iterable[dict]):
        groups = []
        d = MUON_DEFAULTS
        for g in param_groups:
            g = dict(g)
            if g["use_muon"]:
                for k in ("lr", "momentum", "momentum_warmup_start", "momentum_warmup_steps", "nesterov",
                          "ns_steps", "weight_decay"):
                    g.setdefault(k, d[k])
                g.setdefault("step", 0)
                for p in g["params"]:
                    assert p.ndim == 2, "Muon only handles 2-D parameters"
            else:
                g.setdefault("lr", d["lr_other"])
                g.setdefault("betas", tuple(d["adam_betas"]))
                g.setdefault("eps", d["adam_eps"])
                g.setdefault("weight_decay", d["adam_weight_decay"])
            g.setdefault("lr_mult", 1.0)
            g.setdefault("base_lr", g["lr"] / g["lr_mult"])
            groups.append(g)
        super().__init__(groups, {})

    def _muon_momentum(self, group) -> float:
        n = group["momentum_warmup_steps"]
        frac = min(group["step"] / n, 1.0) if n > 0 else 1.0
        return (1 - frac) * group["momentum_warmup_start"] + frac * group["momentum"]

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adam_step(group)
        return loss

    def _muon_step(self, group):
        """Same update as one Newton-Schulz per matrix, but matrices of equal shape are orthogonalized together as
        one batch ([n, rows, cols] bmm) and the elementwise parts use foreach ops: a few dozen kernel launches per
        shape instead of ~20 per matrix (the optimizer step is launch-bound for small models)."""
        lr, wd, momentum = group["lr"], group["weight_decay"], self._muon_momentum(group)
        by_shape: dict = {}
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p.grad)
            by_shape.setdefault((tuple(p.shape), p.dtype, p.device), []).append(p)
        for (shape, _, _), ps in by_shape.items():
            grads = [p.grad for p in ps]
            bufs = [self.state[p]["momentum_buffer"] for p in ps]
            torch._foreach_lerp_(bufs, grads, 1 - momentum)
            gs = torch._foreach_lerp(grads, bufs, momentum) if group["nesterov"] else bufs
            updates = zeropower_via_newtonschulz5(torch.stack(gs), steps=group["ns_steps"]).unbind(0)
            scale = max(1.0, shape[0] / shape[1]) ** 0.5
            if wd:
                torch._foreach_mul_(ps, 1 - lr * wd)
            torch._foreach_add_(ps, [u.to(ps[0].dtype) for u in updates], alpha=-lr * scale)
        group["step"] += 1

    def _adam_step(self, group):
        lr, wd, eps = group["lr"], group["weight_decay"], group["eps"]
        beta1, beta2 = group["betas"]
        by_step: dict = {}  # params with the same step count share bias corrections (one foreach batch)
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            state["step"] += 1
            by_step.setdefault((state["step"], p.device, p.dtype), []).append(p)
        for (t, _, _), ps in by_step.items():
            grads = [p.grad for p in ps]
            m = [self.state[p]["exp_avg"] for p in ps]
            v = [self.state[p]["exp_avg_sq"] for p in ps]
            torch._foreach_lerp_(m, grads, 1 - beta1)
            torch._foreach_mul_(v, beta2)
            torch._foreach_addcmul_(v, grads, grads, value=1 - beta2)
            denom = torch._foreach_div(v, 1 - beta2**t)
            torch._foreach_sqrt_(denom)
            torch._foreach_add_(denom, eps)
            if wd:
                torch._foreach_mul_(ps, 1 - lr * wd)
            torch._foreach_addcdiv_(ps, m, denom, value=-lr / (1 - beta1**t))


def _is_router(name: str) -> bool:
    return "routing_module" in name or "router" in name


def classify_params(model: nn.Module, router: str = "adam") -> dict[str, list[tuple[str, nn.Parameter]]]:
    """Split ``model.named_parameters()`` into the kinds of ``KINDS`` (see module docstring)."""
    assert router in ("adam", "muon")
    owner = {}  # id(param) -> (module name, module)
    for mname, mod in model.named_modules():
        for p in mod.parameters(recurse=False):
            owner.setdefault(id(p), (mname, mod))
    out = {k: [] for k in KINDS}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mname, mod = owner[id(p)]
        if isinstance(mod, nn.Embedding):
            kind = "embed"
        elif mname.split(".")[-1] == "lm_head":
            kind = "head"
        elif p.ndim == 2 and (router == "muon" or not _is_router(mname)):
            kind = "muon"
        else:
            kind = "other"
        out[kind].append((name, p))
    return out


def _muon_cfg(train_cfg) -> dict:
    sec = train_cfg.get("muon") if isinstance(train_cfg, dict) else getattr(train_cfg, "muon", None)
    sec = dict(sec or {})
    unknown = set(sec) - set(MUON_DEFAULTS)
    assert not unknown, f"unknown train.muon keys: {sorted(unknown)}"
    return {**MUON_DEFAULTS, **sec}


@register("optimizer", "muon")
def build_muon_optimizer(model: nn.Module, train_cfg, param_groups_fn: Optional[Callable] = None) -> MuonAdamW:
    """Muon on hidden matrices, Adam on the rest; see the module docstring for the contract."""
    c = _muon_cfg(train_cfg)
    kind_hparams = {
        "muon": dict(use_muon=True, lr=c["lr"], momentum=c["momentum"], nesterov=c["nesterov"],
                     momentum_warmup_start=c["momentum_warmup_start"],
                     momentum_warmup_steps=c["momentum_warmup_steps"], ns_steps=c["ns_steps"],
                     weight_decay=c["weight_decay"]),
        **{
            kind: dict(use_muon=False, lr=c[f"lr_{kind}"], betas=tuple(c["adam_betas"]), eps=c["adam_eps"],
                       weight_decay=c["adam_weight_decay"])
            for kind in ("embed", "head", "other")
        },
    }
    groups = []
    for kind, named in classify_params(model, router=c["router"]).items():
        if not named:
            continue
        hp = kind_hparams[kind]
        if param_groups_fn is None:
            sub = [{"params": [p for _, p in named], "names": [n for n, _ in named]}]
        else:
            ids = {id(p) for _, p in named}
            sub = param_groups_fn(model, filter_fn=lambda name, p: id(p) in ids, weight_decay=hp["weight_decay"])
        for g in sub:
            group = {**hp, **g, "kind": kind}
            group.setdefault("lr_mult", 1.0)
            group["base_lr"] = hp["lr"]
            group["lr"] = hp["lr"] * group["lr_mult"]
            if kind == "muon" and c["residual_wd"] is not None:
                groups.extend(_split_residual(group, float(c["residual_wd"])))
            else:
                groups.append(group)
    return MuonAdamW(groups)


def _is_residual_proj(name: str) -> bool:
    return name.endswith("residual_proj.weight")


def _split_residual(group: dict, wd: float) -> list:
    """Move a Muon group's ``residual_proj`` weights into a sibling group (same LR) with weight decay ``wd``."""
    names = group.get("names")
    assert names is not None and len(names) == len(group["params"]), "residual_wd needs named param groups"
    keep = [i for i, n in enumerate(names) if not _is_residual_proj(n)]
    res = [i for i, n in enumerate(names) if _is_residual_proj(n)]
    out = []
    if keep:
        out.append({**group, "params": [group["params"][i] for i in keep], "names": [names[i] for i in keep]})
    if res:
        out.append({**group, "params": [group["params"][i] for i in res], "names": [names[i] for i in res],
                    "weight_decay": wd, "role": "residual"})
    return out
