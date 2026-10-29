"""Command line of the trainer (``python -m hnet.train.train``): first-class hyperparameter flags on top of
``--config`` and ``--set``.

Every flag maps onto config keys (table :data:`FLAGS`; ``--help`` lists it). Composition order (later wins):

1. ``--config run.yaml`` with its ``include:`` files; ``--model m.yaml`` swaps the run file's model include
   (the run file's other ``model:`` keys still apply);
2. first-class flags, as typed values, in the order of :data:`FLAGS`;
3. ``--set a.b=c`` overrides (YAML values), left to right, so a later ``--set`` of the same key wins;
4. defaults (``hnet.train.config.DEFAULTS``) fill keys that are still unset.

A key set both by a flag and by ``--set`` (or one being a prefix of the other) is an error unless the values are
identical, so a command line never depends on which of the two silently wins. ``--print-config`` prints the
fully composed config as YAML and exits.

Adding a flag: append a :class:`Flag` to :data:`FLAGS` (the parser, help and tests pick it up).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import yaml

from hnet.train.config import load_run_config


def _float(x: str) -> float:
    return float(x)  # accepts 3e-4, 1.5e-3, 5e7


def _int_like(x: str) -> int:
    v = float(x)  # accepts 5e7 for counts
    if v != int(v):
        raise argparse.ArgumentTypeError(f"expected an integer, got {x!r}")
    return int(v)


def _int_list(x: str) -> List[int]:
    return [_int_like(t) for t in x.split(",") if t.strip()]


def _float_list(x: str) -> List[float]:
    return [float(t) for t in x.split(",") if t.strip()]


_NULL = "\0null"  # parsed value of a flag that sets its key to null (argparse cannot tell None from "not given")


def _resume(x: str):
    return _NULL if x.lower() in ("none", "null") else x


@dataclass(frozen=True)
class Flag:
    """``--name VALUE`` sets ``key`` to ``type(VALUE)``; ``also`` are extra (key, fixed value) pairs set with it
    (e.g. ``--train-tokens`` clears ``train.steps`` so the token budget is not silently ignored)."""
    name: str
    key: str
    type: Callable[[str], Any]
    help: str
    metavar: Optional[str] = None
    choices: Optional[Tuple[str, ...]] = None
    also: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)


FLAGS: List[Flag] = [
    Flag("--out", "train.out_dir", str, "run directory (metrics, eval, config, ckpt/)", "DIR"),
    Flag("--seed", "train.seed", _int_like, "seed (model init, data order, RNG streams)"),
    Flag("--optimizer", "train.optimizer", str, "optimizer registry name", choices=("adamw", "muon")),
    Flag("--lr", "train.lr", _float, "peak LR (AdamW; H-Net stages get lr x lambda^s)"),
    Flag("--muon-lr", "train.muon.lr", _float, "Muon LR for hidden matrices (optimizer muon)"),
    Flag("--weight-decay", "train.weight_decay", _float, "AdamW weight decay"),
    Flag("--warmup", "train.schedule.warmup", _float, "schedule warmup (fraction < 1 or steps)"),
    Flag("--decay", "train.schedule.decay", _float, "WSD decay (fraction < 1 or steps)"),
    Flag("--steps", "train.steps", _int_like, "optimizer steps (clears train_tokens/train_bytes)",
         also=(("train.train_tokens", None), ("train.train_bytes", None))),
    Flag("--train-tokens", "train.train_tokens", _float, "token-equivalent budget (clears steps/train_bytes)",
         "N", also=(("train.steps", None), ("train.train_bytes", None))),
    Flag("--train-bytes", "train.train_bytes", _float, "byte budget (clears steps/train_tokens)", "N",
         also=(("train.steps", None), ("train.train_tokens", None))),
    Flag("--stop-at-step", "train.stop_at_step", _int_like, "stop (with checkpoint) early, keeping the schedule"),
    Flag("--micro-batch", "data.micro_batch", _int_like, "windows per micro-step per rank (grad_accum follows)"),
    Flag("--seq-len", "data.seq_len", _int_like, "window length (bytes or tokens)"),
    Flag("--precision", "train.precision", str, "autocast precision on CUDA", choices=("bf16", "fp32")),
    Flag("--device", "train.device", str, "cpu | cuda (default: cuda if available)"),
    Flag("--eval-every", "eval.every", _int_like, "val bpb every N steps (0 = only at the end)"),
    Flag("--log-every", "log.every", _int_like, "train metrics every N steps"),
    Flag("--ckpt-every", "train.ckpt_every", _int_like, "rolling checkpoint every N steps (0 = off)"),
    Flag("--ckpt-keep", "train.ckpt_keep", _int_like, "rolling checkpoints kept (>= 1)"),
    Flag("--ckpt-steps", "train.ckpt_steps", _int_list, "permanent checkpoints at these steps", "S1,S2"),
    Flag("--ckpt-tokens", "train.ckpt_tokens", _float_list, "permanent checkpoints at these tokens_eq", "T1,T2"),
    Flag("--resume", "train.resume", _resume, "auto (newest ckpt in --out) | none | checkpoint or run dir",
         "{auto,none,PATH}"),
    Flag("--max-hours", "train.max_hours", _float, "stop with a checkpoint before this wall time"),
]


def _dest(f: Flag) -> str:
    return "flag_" + f.name.lstrip("-").replace("-", "_")


def build_parser(description: str = "") -> argparse.ArgumentParser:
    table = "\n".join(f"  {f.name:<16} -> {f.key}" + (f" (+ {', '.join(k + '=null' for k, _ in f.also)})"
                                                         if f.also else "") for f in FLAGS)
    epilog = (
        "flag -> config key:\n" + table + "\n  --compile [MODE] -> train.compile (blocks|model; bare flag = blocks)"
        "\n  --no-compile     -> train.compile=false\n\n"
        "precedence (later wins): --config (+ includes, --model swap) < flags < --set (left to right); "
        "defaults only fill keys left unset.\nA key given by both a flag and --set (or overlapping, e.g. --warmup vs "
        "--set train.schedule={...}) must agree, else error.\n"
        "examples:\n"
        "  python -m hnet.train.train --config configs/runs/tiny_hnet2_2m.yaml --lr 3e-3 --seed 1 --out runs/x\n"
        "  python -m hnet.train.train --config configs/runs/tiny_bpe_2m.yaml --model configs/models/m.yaml "
        "--print-config\n"
        "  python -m hnet.train.train --config configs/runs/tiny_cpu_smoke.yaml --resume runs/tiny/cpu_smoke")
    ap = argparse.ArgumentParser(description=description, epilog=epilog,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run yaml (configs/runs/*.yaml)")
    ap.add_argument("--model", default=None, metavar="YAML",
                    help="model yaml that replaces the run file's `model: {include: ...}`")
    ap.add_argument("--set", nargs="*", default=[], action="extend", metavar="KEY=VALUE",
                    help="config overrides a.b=c (YAML values), applied after the flags, left to right")
    g = ap.add_argument_group("hyperparameter flags (each maps onto a config key; see the table below)")
    for f in FLAGS:
        g.add_argument(f.name, dest=_dest(f), type=f.type, default=None, help=f.help, choices=f.choices,
                       metavar=f.metavar or (None if f.choices else f.name.lstrip("-").split("-")[-1].upper()))
    comp = g.add_mutually_exclusive_group()
    comp.add_argument("--compile", dest="flag_compile", nargs="?", const="blocks", default=None,
                      choices=("blocks", "model"), help="torch.compile per block (default) or the whole model")
    comp.add_argument("--no-compile", dest="flag_compile", action="store_const", const=False,
                      help="disable torch.compile")
    ap.add_argument("--print-config", action="store_true", help="print the composed config as YAML and exit")
    ap.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps, print stats")
    return ap


def flag_values(args: argparse.Namespace) -> List[Tuple[str, Any]]:
    """Typed (key, value) pairs of the flags that were given, in :data:`FLAGS` order."""
    out: List[Tuple[str, Any]] = []
    for f in FLAGS:
        v = getattr(args, _dest(f))
        if v is None:
            continue
        out.append((f.key, None if v == _NULL else v))
        out.extend(f.also)
    if args.flag_compile is not None:
        out.append(("train.compile", args.flag_compile))
    return out


def parse_args(argv=None, description: str = ""):
    return build_parser(description).parse_args(argv)


def compose(args) -> dict:
    """The run config for parsed ``args`` (see the module docstring for the order)."""
    return load_run_config(args.config, args.set, model=args.model, flags=flag_values(args))


def dump_config(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, default_flow_style=None, width=120)
