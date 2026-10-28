"""Trainer: ``python -m hnet.train.train --config run.yaml [--set a.b=c ...] [--dry-run]``.

Single GPU by default; under ``torchrun --nproc_per_node=K`` it uses DDP (NCCL on CUDA, gloo on CPU).
See ``configs/README.md`` for every config key. Outline of one optimizer step:

1. ``lr = base_lr * lambda^s * schedule(step)`` per param group (hnet.optim);
2. ``grad_accum`` micro-steps (``no_sync`` on all but the last under DDP), each with bf16 autocast on CUDA
   (fp32 master weights), ``loss = ce_sum / n_targets(step, rank) + aux_loss / grad_accum``, so the
   per-step loss is the exact mean CE over the rank's targets regardless of how windows split into
   micro-batches; batches are produced by a background thread, pinned, and copied ``non_blocking``;
3. clip the global grad norm, optimizer step;
4. at log steps only: per-stage grad norms, cross-rank stat reduction, one host sync, MFU;
5. eval / checkpoint / stop checks (SIGTERM/SIGINT, ``max_hours``).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import os
import signal
import sys
import time
from typing import Optional

import torch
import torch.distributed as dist

from hnet.config import BOS_ID, IGNORE_INDEX
from hnet.optim import build_optimizer, build_schedule, params_by_stage, set_lr
from hnet.train import checkpoint as ckpt
from hnet.train.config import dump_yaml, load_run_config, model_config_from_section
from hnet.train.data import (GPT2_EOT, Prefetcher, build_train_loader, build_val_batches, data_kind,
                             default_bytes_per_token)
from hnet.train.metrics import FlopsEstimator, MetricsLogger, StatAccumulator, grad_norms_by_stage, peak_flops

LN2 = math.log(2.0)


def _import_obj(path: str):
    import importlib

    mod, _, name = path.partition(":")
    return getattr(importlib.import_module(mod), name)


def build_model_from_section(section: dict):
    """Returns (model, model_cfg). ``type: custom`` + ``factory: pkg.mod:fn`` -> ``fn(section)`` (tests)."""
    if section.get("type") == "custom":
        return _import_obj(section["factory"])(section)
    mcfg = model_config_from_section(section)
    from hnet.model import build_model  # WS-A2

    return build_model(mcfg), mcfg


def _get(out, key, default=None):
    if isinstance(out, dict):
        return out.get(key, default)
    return getattr(out, key, default)


def _wrap_checkpointing(block):
    from torch.utils.checkpoint import checkpoint

    orig = block.forward

    @functools.wraps(orig)
    def fwd(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(orig, *args, use_reentrant=False, **kwargs)
        return orig(*args, **kwargs)

    block.forward = fwd


class Trainer:
    def __init__(self, cfg: dict, dry_run: bool = False):
        self.cfg, self.dry_run = cfg, dry_run
        tc, dc = cfg["train"], cfg["data"]
        self.tc, self.dc, self.ec, self.lc = tc, dc, cfg["eval"], cfg["log"]

        # ---- distributed / device
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dev = tc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cuda":
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device(dev)
        if self.world > 1 and not dist.is_initialized():
            import datetime

            dist.init_process_group("nccl" if self.device.type == "cuda" else "gloo",
                                    timeout=datetime.timedelta(minutes=float(tc.get("dist_timeout_min", 60))))
        self.is_main = self.rank == 0
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.out_dir = tc["out_dir"]

        # ---- model
        seed = int(tc["seed"])
        torch.manual_seed(seed)
        msec = cfg["model"]
        if msec.get("type", "hnet") == "hnet" and msec.get("lr_multiplier_n_gpt", "data") in (None, "data"):
            # N_GPT of App. C = measured GPT-2 bytes/token of the training data (hnet/data/bpt_stats.json)
            msec = cfg["model"] = {**msec, "lr_multiplier_n_gpt": default_bytes_per_token(dc, None)}
        self.model, self.mcfg = build_model_from_section(msec)
        if hasattr(self.model, "init_weights"):
            self.model.init_weights()
        self.model.to(self.device)
        self.is_hnet = hasattr(self.mcfg, "arch_layout")
        self.n_stages = self.mcfg.n_stages if self.is_hnet else 0
        self.kind = data_kind(dc, self.mcfg)
        self.seq_len = int(dc["seq_len"])
        self.bpt = default_bytes_per_token(dc, self.mcfg)
        self.eot = int(dc.get("eot", GPT2_EOT))

        # ---- batch arithmetic
        mb = int(dc["micro_batch"])
        if dc.get("batch_windows"):
            W = int(dc["batch_windows"])
        elif self.kind == "bytes" and dc.get("batch_bytes"):
            W = int(dc["batch_bytes"]) // self.seq_len
        elif self.kind == "tokens" and dc.get("batch_tokens"):
            W = int(dc["batch_tokens"]) // self.seq_len
        elif self.kind == "tokens" and dc.get("batch_bytes"):  # approximate: uses bytes_per_token
            W = max(1, round(int(dc["batch_bytes"]) / (self.seq_len * self.bpt)))
        else:
            W = mb * self.world * (int(tc["grad_accum"]) if str(tc["grad_accum"]) != "auto" else 1)
        assert W % (mb * self.world) == 0, (
            f"global windows/step {W} not divisible by micro_batch {mb} x world {self.world}")
        self.accum = W // (mb * self.world)
        if str(tc["grad_accum"]) != "auto":
            assert int(tc["grad_accum"]) == self.accum, f"grad_accum {tc['grad_accum']} != derived {self.accum}"
        self.windows_per_step = W
        pos_per_step = W * self.seq_len
        est_bytes_per_step = pos_per_step if self.kind == "bytes" else pos_per_step * self.bpt
        est_tokens_per_step = pos_per_step / self.bpt if self.kind == "bytes" else pos_per_step
        if tc.get("steps"):
            self.total_steps = int(tc["steps"])
        elif tc.get("train_tokens"):
            self.total_steps = math.ceil(float(tc["train_tokens"]) / est_tokens_per_step)
        elif tc.get("train_bytes"):
            self.total_steps = math.ceil(float(tc["train_bytes"]) / est_bytes_per_step)
        else:
            raise ValueError("set train.steps, train.train_tokens or train.train_bytes")
        self.est_tokens_per_step = est_tokens_per_step

        # ---- optimizer / schedule
        self.opt = build_optimizer(self.model, self.mcfg if self.is_hnet else None, tc)
        self.lr_cfg = [(g["base_lr"], g["lr_mult"]) for g in self.opt.param_groups]
        self.schedule = build_schedule(tc["schedule"], self.total_steps)
        self.stage_params = params_by_stage(self.model)
        self.all_params = [p for p in self.model.parameters() if p.requires_grad]

        # ---- state + resume
        self.step = 0
        self.bytes_local = 0  # this rank's text bytes
        self.tokens_local = 0.0  # this rank's tokens_eq
        self.loader_state = {}
        self.resumed_from = None
        res = tc.get("resume")
        if res and not dry_run:
            path = ckpt.latest_checkpoint(self.out_dir) if res == "auto" else res
            if path:
                st = ckpt.load_checkpoint(path, self.model, self.opt, self.rank)
                for g, (b, m) in zip(self.opt.param_groups, self.lr_cfg):
                    g["base_lr"], g["lr_mult"] = b, m
                meta, rs = st["meta"], st["rank_state"]
                self.step = int(meta["step"])
                if rs and int(meta.get("world_size", 1)) == self.world:
                    self.bytes_local, self.tokens_local = rs.get("bytes_local", 0), rs.get("tokens_local", 0.0)
                    self.loader_state = rs.get("loader", {})
                else:  # world size changed: split global counters evenly
                    self.bytes_local = int(meta.get("bytes_seen", 0)) // self.world
                    self.tokens_local = float(meta.get("tokens_eq", 0.0)) / self.world
                self.resumed_from = path
                if self.is_main:
                    print(f"[train] resumed from {path} at step {self.step}", flush=True)
        if self.resumed_from is None:
            torch.manual_seed(seed + 1000 * self.rank)  # per-rank stream for any stochastic op

        # ---- wrappers: activation checkpointing, compile, DDP
        self.raw = self.model
        blocks = self.raw.blocks() if hasattr(self.raw, "blocks") else []
        if tc.get("activation_checkpointing"):
            for b in blocks:
                _wrap_checkpointing(b)
        comp = tc.get("compile")
        if comp:
            dyn = tc.get("compile_dynamic")
            if dyn is None:
                dyn = True if self.is_hnet else None
            if comp == "blocks" and blocks:
                for b in blocks:
                    b.compile(dynamic=dyn)
            else:
                self.raw.compile(dynamic=dyn)
        if self.world > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP

            self.model = DDP(self.raw, device_ids=[self.local_rank] if self.cuda else None,
                             find_unused_parameters=False, gradient_as_bucket_view=True,
                             bucket_cap_mb=tc.get("ddp_bucket_mb", 25))
        self.autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                         if self.cuda and tc.get("precision", "bf16") == "bf16" else contextlib.nullcontext())

        # ---- data
        self.loader = build_train_loader(dc, self.kind, self.accum, self.rank, self.world, seed)
        if self.loader_state:
            self.loader.load_state_dict(self.loader_state)
        self.val_batches = None
        if self.ec.get("val") and self.ec["val"] != "none":
            try:
                self.val_batches = build_val_batches(dc, self.ec, self.kind, seed)
            except Exception as e:
                if self.is_main:
                    print(f"[train] val set unavailable: {e!r}", file=sys.stderr)

        # ---- metrics
        self.flops = FlopsEstimator(self.mcfg, self.raw, self.seq_len, self.kind)
        self.peak = peak_flops() if self.cuda else None
        self.logger = MetricsLogger(self.out_dir, self.is_main and not dry_run, self.lc.get("wandb"),
                                    self.lc.get("run_name"), cfg, resume=self.resumed_from is not None)
        self._stop_reason: Optional[str] = None
        self._zero = torch.zeros((), device=self.device)
        if self.is_main and not dry_run:
            os.makedirs(self.out_dir, exist_ok=True)
            dump_yaml(cfg, os.path.join(self.out_dir, "config.yaml"))

    # ------------------------------------------------------------------ helpers
    def _signal(self, signum, frame):
        self._stop_reason = f"signal {signum}"

    def _to_device(self, b: dict) -> dict:
        nb = self.cuda
        return {
            "input_ids": b["input_ids"].to(self.device, non_blocking=nb),
            "labels": b["labels"].to(self.device, non_blocking=nb),
            "cu_seqlens": b["cu_seqlens"].to(self.device, non_blocking=nb),
            "max_seqlen": int(b["max_seqlen"]),
        }

    def _forward(self, model, b: dict):
        return model(b["input_ids"], cu_seqlens=b["cu_seqlens"], max_seqlen=b["max_seqlen"], labels=b["labels"],
                     return_logits=False)

    def _global_counters(self):
        if self.world == 1:
            return self.bytes_local, self.tokens_local
        t = torch.tensor([float(self.bytes_local), float(self.tokens_local)], dtype=torch.float64,
                         device=self.device)
        dist.all_reduce(t)
        return int(t[0].item()), float(t[1].item())

    def _should_stop(self, t_start: float, step_time: float) -> bool:
        flag = 0.0
        if self._stop_reason:
            flag = 1.0
        mh = self.tc.get("max_hours")
        if mh and (time.time() - t_start + 2 * step_time) / 3600.0 > float(mh):
            self._stop_reason = self._stop_reason or "max_hours"
            flag = 1.0
        if self.world > 1:
            t = torch.tensor([flag], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            flag = float(t.item())
            if flag and not self._stop_reason:
                self._stop_reason = "peer stop"
        return flag > 0

    def save(self, keep: bool = False):
        if self.dry_run:
            return
        bytes_seen, tokens_eq = self._global_counters()
        meta = {"step": self.step, "bytes_seen": bytes_seen, "tokens_eq": tokens_eq, "world_size": self.world,
                "config": self.cfg}
        rank_state = {"bytes_local": self.bytes_local, "tokens_local": self.tokens_local,
                      "loader": self.loader.state_dict()}
        path = ckpt.save_checkpoint(self.out_dir, self.step, self.raw, self.opt, rank_state, meta, self.rank,
                                    self.world, keep=keep)
        if self.is_main:
            ckpt.prune_checkpoints(self.out_dir, int(self.tc.get("ckpt_keep", 2)))
            print(f"[train] saved {path}", flush=True)

    # ------------------------------------------------------------------ eval
    @torch.no_grad()
    def evaluate(self) -> dict:
        if not self.val_batches:
            return {}
        model = self.raw
        was_training = model.training
        model.eval()
        ce = torch.zeros((), dtype=torch.float64, device=self.device)
        nbytes, ndocs = 0, 0
        for i, b in enumerate(self.val_batches):
            if i % self.world != self.rank:
                continue
            labels = b["labels"].clone()
            if self.kind == "bytes":
                labels[labels >= BOS_ID] = IGNORE_INDEX  # only real-byte targets count (EOS excluded)
                nb = int(b["n_bytes"]) if "n_bytes" in b else int((labels != IGNORE_INDEX).sum())
            else:
                labels[labels == self.eot] = IGNORE_INDEX  # <|endoftext|> targets are not text
                nb = int(b["n_bytes"])
            d = self._to_device({**b, "labels": labels})
            with self.autocast:
                out = self._forward(model, d)
            ce_sum = _get(out, "ce_sum")
            if ce_sum is None:
                ce_sum = _get(out, "loss") * (labels != IGNORE_INDEX).sum()
            ce += ce_sum.double()
            nbytes += nb
            ndocs += int(b["cu_seqlens"].numel() - 1)
        t = torch.stack([ce, torch.tensor(float(nbytes), dtype=torch.float64, device=self.device),
                         torch.tensor(float(ndocs), dtype=torch.float64, device=self.device)])
        if self.world > 1:
            dist.all_reduce(t)
        ce_v, nb_v, nd_v = t.tolist()
        model.train(was_training)
        return {"val/bpb": ce_v / LN2 / max(nb_v, 1.0), "val/bytes": int(nb_v), "val/docs": int(nd_v)}

    def downstream_eval(self) -> dict:
        """Optional zero-shot eval via hnet.eval (WS-E), rank 0 only. Config ``eval.downstream``:
        ``{every, suite (paper7), tasks (null = suite default), limit, n_boot}``. Returns flat numeric leaves."""
        ds = self.ec.get("downstream")
        if not ds:
            return {}
        try:  # lazy, optional
            from hnet.eval.harness import evaluate_suite
            from hnet.eval.scorers import make_scorer
        except ImportError as e:
            print(f"[train] downstream eval unavailable: {e!r}", file=sys.stderr)
            return {}
        was_training = self.raw.training
        try:
            scorer = make_scorer(self.raw, "byte" if self.kind == "bytes" else "token", device=self.device)
            summary, _ = evaluate_suite(scorer, ds.get("suite", "paper7"), tasks=ds.get("tasks"),
                                        limit=ds.get("limit"), n_boot=int(ds.get("n_boot", 200)))
        except Exception as e:  # never kill a training run over eval
            print(f"[train] downstream eval failed: {e!r}", file=sys.stderr)
            return {}
        finally:
            self.raw.train(was_training)
        flat = {}

        def walk(prefix, v):
            if isinstance(v, dict):
                for k, x in v.items():
                    walk(f"{prefix}/{k}", x)
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                flat[prefix] = v

        walk("downstream", {k: v for k, v in summary.items() if k != "suite"})
        return flat

    def run_eval(self, final: bool = False, val: bool = True, downstream: bool = False):
        """Collective: every rank must call it at the same step."""
        rec = self.evaluate() if val else {}
        ds = self.ec.get("downstream") or {}
        if ds and self.is_main and (final or downstream):
            rec.update(self.downstream_eval())
        if self.world > 1 and ds and (final or downstream):
            dist.barrier()
        bytes_seen, tokens_eq = self._global_counters()
        if rec:
            rec = {"step": self.step, "tokens_eq": tokens_eq, "bytes_seen": bytes_seen, **rec}
            self.logger.log(rec, kind="eval")
        return rec

    # ------------------------------------------------------------------ train
    def describe(self) -> dict:
        n_params = sum(p.numel() for p in self.raw.parameters())
        info = {"params": n_params, "params_by_stage": {s: sum(p.numel() for p in ps)
                                                        for s, ps in self.stage_params.items()},
                "lr_multipliers": {int(g.get("stage", 0)): round(g["lr_mult"], 4) for g in self.opt.param_groups},
                "kind": self.kind, "seq_len": self.seq_len, "windows_per_step": self.windows_per_step,
                "micro_batch": int(self.dc["micro_batch"]), "grad_accum": self.accum, "world_size": self.world,
                "total_steps": self.total_steps, "est_tokens_eq_per_step": self.est_tokens_per_step}
        if self.is_hnet:
            bpp, c = [1.0], 1.0
            for r in self.mcfg.dc.target_ratios:
                c *= r
                bpp.append(c)
            stats = {f"compression/s{s}": r for s, r in enumerate(self.mcfg.dc.target_ratios)}
            info["flops_per_byte@target"] = self.flops.flops_per_byte(stats, None)
        else:
            info["flops_per_byte@bpt"] = self.flops.flops_per_byte({}, self.bpt)
        return info

    def train(self):
        tc, lc = self.tc, self.lc
        log_every = int(lc.get("every", 10))
        eval_every = int(self.ec.get("every") or 0)
        ds_every = int((self.ec.get("downstream") or {}).get("every") or 0)
        ckpt_every = int(tc.get("ckpt_every") or 0)
        ckpt_steps = set(int(s) for s in (tc.get("ckpt_steps") or []))
        ckpt_every_tok = float(tc.get("ckpt_every_tokens") or 0)
        stop_at = tc.get("stop_at_step")
        end_step = self.total_steps if not stop_at else min(self.total_steps, int(stop_at))
        if self.dry_run:
            end_step = min(self.total_steps, self.step + 2)
            log_every = 1
        check_every = 1 if self.world == 1 else int(tc.get("check_signals_every", 10))
        grad_clip = float(tc.get("grad_clip") or 0.0)

        if self.is_main:
            print("[train] " + json.dumps(self.describe(), default=str), flush=True)

        old_handlers = {}
        if not self.dry_run:
            import threading

            if threading.current_thread() is threading.main_thread():
                for sig in (signal.SIGTERM, signal.SIGINT):
                    old_handlers[sig] = signal.signal(sig, self._signal)

        pf = Prefetcher(self.loader, self.step, self.accum, pin=self.cuda, depth=int(self.dc.get("prefetch", 4)),
                        threads=int(self.dc.get("num_workers", 1)))
        acc = StatAccumulator()  # ce_sum / aux on device, every step (adds only, no sync)
        stats_acc = StatAccumulator()  # model stats + grad norms, log steps only
        host_acc = {"n_targets": 0.0, "n_bytes": 0.0, "tokens": 0.0}  # host-side counts since last log
        t_start = time.time()
        t_log = time.perf_counter()
        steps_since_log, step_time = 0, 0.0
        self.raw.train()
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        try:
            while self.step < end_step:
                step = self.step
                is_log = ((step + 1) % log_every == 0) or (step + 1 == end_step)
                lr_factor = self.schedule(step)
                lr = set_lr(self.opt, lr_factor)
                host = [pf.next() for _ in range(self.accum)]
                n_tgt = [int((hb["labels"] != IGNORE_INDEX).sum()) for hb in host]
                n_tgt_step = max(sum(n_tgt), 1)
                tokens_before = self.tokens_local
                for m, hb in enumerate(host):
                    b = self._to_device(hb)
                    sync_ctx = (self.model.no_sync() if self.world > 1 and m < self.accum - 1
                                else contextlib.nullcontext())
                    with sync_ctx:
                        with self.autocast:
                            out = self._forward(self.model, b)
                        ce_sum = _get(out, "ce_sum")
                        if ce_sum is None:
                            ce_sum = _get(out, "loss").float() * n_tgt[m]
                        aux = _get(out, "aux_loss")
                        if not isinstance(aux, torch.Tensor):
                            aux = self._zero if not aux else torch.tensor(float(aux), device=self.device)
                        loss = ce_sum.float() / n_tgt_step + aux.float() / self.accum
                        loss.backward()
                    acc.add("ce_sum", ce_sum)
                    acc.add("aux", aux)
                    if is_log:
                        stats_acc.add_dict(_get(out, "stats") or {})
                    nbytes = int(hb.get("n_bytes", hb["input_ids"].numel()))
                    if "n_tokens" in hb:
                        ntok = float(hb["n_tokens"])
                    else:
                        ntok = float(hb["input_ids"].numel()) if self.kind == "tokens" else nbytes / self.bpt
                    self.bytes_local += nbytes
                    self.tokens_local += ntok
                    host_acc["n_targets"] += n_tgt[m]
                    host_acc["n_bytes"] += nbytes
                    host_acc["tokens"] += ntok
                if is_log:  # pre-clip per-stage norms, log steps only
                    for k, v in grad_norms_by_stage(self.stage_params).items():
                        stats_acc.add(k, v)
                if grad_clip > 0:
                    gn = torch.nn.utils.clip_grad_norm_(self.all_params, grad_clip, foreach=True)
                elif is_log:
                    gn = torch.nn.utils.get_total_norm([p.grad for p in self.all_params if p.grad is not None])
                if is_log:
                    stats_acc.add("grad_norm", gn)
                self.opt.step()
                self.opt.zero_grad(set_to_none=True)
                self.step += 1
                steps_since_log += 1

                if is_log:
                    rec = self._log_record(acc, stats_acc, host_acc, lr, t_log, steps_since_log)
                    rec["lr_factor"] = lr_factor
                    step_time = rec["step_time"]
                    t_log = time.perf_counter()
                    steps_since_log = 0
                    acc.clear()
                    stats_acc.clear()
                    host_acc = dict.fromkeys(host_acc, 0.0)
                    self.logger.log(rec)
                    if self.dry_run and self.is_main:
                        self._dry_run_report(rec)
                if self.dry_run:
                    continue

                do_val = bool(eval_every) and self.step % eval_every == 0
                do_ds = bool(ds_every) and self.step % ds_every == 0
                if (do_val or do_ds) and self.step < self.total_steps:
                    self.run_eval(val=do_val, downstream=do_ds)
                stop = False
                if self.step % check_every == 0 or self.step == end_step:
                    stop = self._should_stop(t_start, step_time)
                keep = self.step in ckpt_steps
                if ckpt_every_tok:  # rank-local tokens x world ~ global tokens (no sync needed)
                    keep |= (math.floor(self.tokens_local * self.world / ckpt_every_tok)
                             > math.floor(tokens_before * self.world / ckpt_every_tok))
                regular = bool(ckpt_every) and self.step % ckpt_every == 0
                at_end = self.step == end_step
                if (keep or regular or stop) and not at_end:
                    self.save(keep=keep)
                if stop and not at_end:
                    if self.is_main:
                        print(f"[train] stopping at step {self.step}: {self._stop_reason}", flush=True)
                    break
            if not self.dry_run and self.step == end_step:
                finished = self.step >= self.total_steps
                if finished and self.ec.get("at_end", True):
                    self.run_eval(final=True)
                if tc.get("save_final", True) or not finished:
                    self.save(keep=finished or self.step in ckpt_steps)
        finally:
            pf.close()
            for sig, h in old_handlers.items():
                signal.signal(sig, h)
            self.logger.close()
        return self

    def _log_record(self, acc, stats_acc, host_acc, lr, t_log, n_steps) -> dict:
        dev = self.device
        vals = torch.stack([acc.sums["ce_sum"].to(dev, torch.float64), acc.sums["aux"].to(dev, torch.float64)])
        host = torch.tensor([float(acc.counts["aux"]), host_acc["n_targets"], host_acc["n_bytes"],
                             host_acc["tokens"]], dtype=torch.float64, device=dev)
        buf = torch.cat([vals, host])
        if self.world > 1:
            dist.all_reduce(buf)
        stats = stats_acc.reduce_means(dev, self.world)
        ce_sum, aux_sum, n_micro, n_tgt, n_bytes, n_tok = buf.tolist()
        ce = ce_sum / max(n_tgt, 1.0)
        aux = aux_sum / max(n_micro, 1.0)
        bpb = ce / LN2 if self.kind == "bytes" else ce_sum / LN2 / max(n_bytes, 1.0)
        dt = time.perf_counter() - t_log
        bytes_seen, tokens_eq = self._global_counters()
        bps = n_bytes / max(dt, 1e-9)
        fpb = self.flops.flops_per_byte(stats, n_bytes / max(n_tok, 1e-9) if self.kind == "tokens" else None)
        rec = {"step": self.step, "tokens_eq": tokens_eq, "bytes_seen": bytes_seen, "lr": lr,
               "loss": ce + aux, "ce": ce, "bpb": bpb, "aux_loss": aux}
        rec.update(stats)
        rec.update({
            "flops_per_byte": fpb,
            "mfu": (fpb * bps / (self.peak * self.world)) if (fpb and self.peak) else None,
            "bytes_per_sec": bps,
            "tokens_per_sec": n_tok / max(dt, 1e-9),
            "mem_gb": torch.cuda.max_memory_allocated(dev) / 1e9 if self.cuda else None,
            "step_time": dt / max(n_steps, 1),
        })
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(dev)
        return rec

    def _dry_run_report(self, rec):
        keys = ["step", "loss", "grad_norm", "step_time", "mem_gb", "flops_per_byte", "mfu"]
        print("[dry-run] " + ", ".join(f"{k}={rec.get(k)}" for k in keys), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], action="extend", help="overrides a.b=c (yaml values)")
    ap.add_argument("--dry-run", action="store_true", help="build everything, run 2 steps, print stats")
    args = ap.parse_args(argv)
    cfg = load_run_config(args.config, args.set)
    tr = Trainer(cfg, dry_run=args.dry_run)
    tr.train()
    if args.dry_run and tr.is_main:
        if tr.cuda:
            print(f"[dry-run] peak memory {torch.cuda.max_memory_allocated() / 1e9:.2f} GB", flush=True)
        print("[dry-run] ok", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()
    return tr


if __name__ == "__main__":
    main()
