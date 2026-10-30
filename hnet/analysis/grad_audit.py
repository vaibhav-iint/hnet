"""Step-0 gradient audit (CLAUDE.md rules 1-2): which objective reaches which parameter group, and what is degenerate
at initialization. Runs on CPU in seconds on a tiny model; nothing goes to a GPU before this has been read.

    python -m hnet.analysis.grad_audit configs/runs/tiny_hnet1_2m.yaml --set model.dc.chunk_vocab=... [--text FILE]

For every objective term (byte CE, ratio loss, next-chunk CE, main-aux CE, router prior, consistency) the audit takes
``torch.autograd.grad`` against every parameter group (embeddings, each stage's encoder / router / decider / chunk
vocabulary / residual projection / decoder, the innermost main network, the LM head) and prints the gradient norm,
marking the exact zeros: a loss is not a training signal for a module until autograd says so. It also prints the
mechanism metrics of the forward pass (boundary fraction, compression, router margin, fraction of undecided p,
dechunk vs residual rms) so that degenerate initializations (p stuck at 0.5, a dead bypass, a dead stage) are visible
before the first step. ``audit()`` returns the table for tests; ``--steps N`` repeats after N plain SGD steps.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Tuple

import torch

from hnet.config import HNetConfig
from hnet.model import build_model

ZERO = 0.0


def param_groups(model) -> Dict[str, List[torch.nn.Parameter]]:
    """Name -> parameters, by stage and role (encoder / routing_module / decider / chunk_vocab / residual_proj /
    decoder / main_network for the innermost stage), plus embeddings, n-gram tables and the LM head."""
    groups: Dict[str, List[torch.nn.Parameter]] = {}
    roles = {"encoder", "routing_module", "decider", "chunk_vocab", "residual_proj", "decoder", "pad_dimension"}
    for st in model.backbone.stages():  # any other module / parameter owned directly by a hierarchy stage is a role too
        if not st.is_innermost:
            roles |= {n for n, _ in st.named_children()} | {n for n, _ in st.named_parameters(recurse=False)}
    roles.discard("main_network")
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        parts = name.split(".")
        if parts[0] != "backbone":
            key = parts[0] if parts[0] != "lm_head" or not model.config.tie_embeddings else "embeddings"
        else:
            stage = 0
            i = 1
            while i < len(parts) and parts[i] == "main_network":  # each HNet.main_network is the next stage ...
                stage += 1
                i += 1
            if i < len(parts) and parts[i] == "memory":  # HNetConfig.main_memory: values vs addressing (query/keys)
                key = f"s{stage}.memory." + ("values" if i + 1 < len(parts) and parts[i + 1] == "values"
                                             else "addressing")
            elif i < len(parts) and parts[i] in roles:
                key = f"s{stage}.{parts[i]}"
                if parts[i] == "chunk_vocab" and i + 1 < len(parts) and parts[i + 1].startswith("suffix"):
                    key += ".suffix_gates"  # dc.chunk_vocab.suffix_lens gates, reported separately from the tables
                if parts[i] == "decider" and i + 1 < len(parts):  # decision_kwargs score features
                    key += ".router_probe" if parts[i + 1] == "router_probe" else ".feat_gain"
            else:  # ... except the innermost one, which is that stage's Isotropic main network
                key = f"s{stage - 1}.main_network"
        groups.setdefault(key, []).append(p)
    return groups


def objective_terms(model, out, ids) -> Dict[str, torch.Tensor]:
    """The separate objective terms of a forward pass (unweighted), rebuilt from the model's outputs."""
    terms = {"byte_ce": out.loss}
    aux = model.backbone.aux_losses(out.bpred)
    for s, (loss, _) in enumerate(aux):
        terms[f"ratio/s{s}"] = loss
    for st, r in zip(model.backbone.stages(), out.bpred):
        cv = getattr(st, "chunk_vocab", None)
        if cv is not None and r.vocab_tgt is not None:
            terms[f"next_chunk_ce/s{st.stage_idx}"] = cv.pred_loss(r.inner_out, r.vocab_tgt)
    for s, r in enumerate(out.bpred):
        if getattr(r, "probe_loss", None) is not None:  # decision_kwargs.probe_rank: the decider's next-byte probe
            terms[f"router_probe_ce/s{s}"] = r.probe_loss
        if getattr(r, "teach_loss", None) is not None:  # decision_kwargs.teach: the probe teaches the router
            terms[f"router_teach/s{s}"] = r.teach_loss
    dc = model.config.dc
    if dc.router_prior_weight > 0 and out.bpred and out.bpred[0].logit is not None:
        tgt = ((ids == 0x20) | (ids == 0x0A)).float()
        terms["router_prior/s0"] = torch.nn.functional.binary_cross_entropy_with_logits(out.bpred[0].logit.float(), tgt)
    if dc.router_surprise_weight > 0 and out.bpred and getattr(model, "_audit_h", None) is not None:
        from hnet.model.lm import pack_inputs

        B = out.logits.shape[0] if out.logits is not None and out.logits.dim() == 3 else 1
        _, ctx, _ = pack_inputs(ids.view(B, -1), None, None)  # the audit's [B, L] segments
        terms["router_surprise/s0"] = model._router_surprise_loss(model._audit_h, out.bpred[0], ids, ctx)[0]
    if dc.router_consistency_weight > 0 and out.bpred and out.bpred[0].logit is not None:
        from hnet.model.lm import pack_inputs

        _, ctx, _ = pack_inputs(ids.unsqueeze(0), None, None)
        terms["router_consistency/s0"] = model._router_consistency(out.bpred[0].logit, ids, ctx)
    if dc.router_gain_weight > 0 and out.bpred and getattr(out.bpred[0], "residual_in", None) is not None \
            and getattr(model, "_audit_h", None) is not None:
        from hnet.model.lm import pack_inputs

        _, ctx, _ = pack_inputs(ids.unsqueeze(0), None, None)
        terms["router_gain/s0"] = model._router_gain_loss(model._audit_h, out.bpred[0], ctx, ids)[0]
    th = getattr(model, "trie_head", None)
    if th is not None and out.bpred and getattr(model, "_audit_h", None) is not None:
        # the byte CE through the trie bonus only (the LM-head logits detached): where the trie path reaches
        from hnet.model.lm import pack_inputs

        shape = getattr(model, "_audit_shape", None)
        _, ctx, _ = pack_inputs(ids.view(shape) if shape is not None else ids.unsqueeze(0), None, None)
        r0 = out.bpred[0]
        q = r0.main_out if th.query == "main" else model._audit_h
        base = model.lm_head(model._audit_h).float().detach()
        if th.topk > 0:  # fast head: the trie scores of the top-k bytes of the (detached) base logits
            mm = getattr(model, "match_model", None)
            if mm is not None and mm.gate is not None:
                base = base + mm.logit_bonus(model._audit_h, mm.candidates(ids, ctx), base.shape[-1]).detach()
            cand, vals, _ = th.rescore(q, ids, ctx, r0.boundary_mask, base)
            logits = base.scatter_add(1, cand, vals)
        else:
            bonus, _ = th(q, ids, ctx, r0.boundary_mask)
            logits = base + bonus
        terms["byte_ce_via_trie"] = torch.nn.functional.cross_entropy(logits, ids)
    return terms


def grad_table(model, terms: Dict[str, torch.Tensor], groups: Dict[str, List[torch.nn.Parameter]]):
    """{term: {group: grad norm}}; exact zeros (no path, or a zero-initialised path) are reported as 0.0."""
    table = {}
    for tname, loss in terms.items():
        if not loss.requires_grad:
            table[tname] = {g: ZERO for g in groups}
            continue
        row = {}
        for gname, ps in groups.items():
            grads = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
            row[gname] = float(sum(g.float().pow(2).sum() for g in grads if g is not None) ** 0.5) if any(
                g is not None for g in grads) else ZERO
        table[tname] = row
    return table


def mechanism_metrics(out) -> Dict[str, float]:
    keep = ("compression/", "p_mid/", "p_margin/", "dechunk_rms/", "residual_rms/", "score_std/", "tau/", "F/", "G/",
            "residual_gate/", "trie/", "suffix_gate", "explore_", "chunk_copy", "copy_route/", "probe_", "feat_gain",
            "router_agree", "router_own", "comp_seg", "attn_ctx", "bnd_on_ws", "memory/")
    m = {k: float(v) for k, v in out.stats.items() if k.startswith(keep)}
    for s, r in enumerate(out.bpred):
        m[f"boundary_frac/s{s}"] = float(r.boundary_mask.float().mean())
    return m


def audit(cfg: HNetConfig, ids: torch.Tensor, steps: int = 0, lr: float = 1e-2, seed: int = 0,
          progress: Optional[float] = None):
    """Build + init the model, run forward/backward on ``ids`` [B, T] and return (grad table, mechanism metrics).
    With ``steps`` > 0 first takes that many plain SGD steps on the total loss (the step-10/50/200 views)."""
    torch.manual_seed(seed)
    model = build_model(cfg)
    model.init_weights()
    model.train()
    if progress is not None:  # training progress seen by curricula / annealed options (default: the model's 1.0)
        model.set_progress(progress)
    if steps > 0:
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for _ in range(steps):
            o = model(ids, labels=ids)
            opt.zero_grad()
            (o.loss + o.aux_loss).backward()
            opt.step()
    hook = model.backbone.register_forward_hook(lambda mod, args, res: setattr(model, "_audit_h", res[0]))
    out = model(ids, labels=ids)
    hook.remove()
    model._audit_shape = tuple(ids.shape)
    flat = ids.reshape(-1)
    terms = objective_terms(model, out, flat)
    groups = param_groups(model)
    return grad_table(model, terms, groups), mechanism_metrics(out), groups


def format_table(table, groups) -> str:
    names = list(groups)
    w = max(len(n) for n in names) + 2
    head = " " * 20 + "".join(f"{n:>{w}}" for n in names)
    lines = [head]
    for t, row in table.items():
        cells = "".join(f"{'0 (none)' if row[n] == ZERO else f'{row[n]:.2e}':>{w}}" for n in names)
        lines.append(f"{t:<20}{cells}")
    return "\n".join(lines)


def degeneracies(table, metrics, groups) -> List[str]:
    """Plain-language warnings: groups no objective reaches, routers that are undecided, dead stages, a bypass that
    already dominates the hierarchy."""
    warn = []
    for g in groups:
        if all(row[g] == ZERO for row in table.values()):
            warn.append(f"{g}: no objective reaches it (zero gradient at init)")
    for t, row in table.items():
        if t.startswith("next_chunk_ce") and all(row[g] == ZERO for g in groups if "routing_module" in g):
            warn.append(f"{t}: sends the router no gradient (dc.chunk_ste=false; the STE sits after dechunking)")
    for k, v in metrics.items():
        if k.startswith("p_mid/") and v > 0.5:
            warn.append(f"{k} = {v:.2f}: most boundary decisions undecided (p in 0.4-0.6)")
        if k.startswith("compression/") and (v > 64 or v < 1.5):
            warn.append(f"{k} = {v:.1f}: stage selects (almost) nothing or everything")
    for s in range(8):
        d, r = metrics.get(f"dechunk_rms/s{s}"), metrics.get(f"residual_rms/s{s}")
        if d is not None and r is not None and r > 2 * d > 0:
            warn.append(f"stage {s}: the residual bypass (rms {r:.2f}) already dominates the hierarchy (rms {d:.2f})")
    return warn


def _load_cfg(path: str, overrides) -> HNetConfig:
    from hnet.train.config import load_run_config  # resolves `include:` composition like the trainer

    d = load_run_config(path, overrides=overrides)
    m = dict(d["model"])
    assert m.pop("type", "hnet") == "hnet", "the audit is for H-Net models"
    return HNetConfig.from_dict(m)


def _ids(text: Optional[str], batch: int, length: int, seed: int) -> torch.Tensor:
    if text:
        raw = open(text, "rb").read()
        n = batch * length
        assert len(raw) >= n, f"{text}: need {n} bytes"
        return torch.frombuffer(bytearray(raw[:n]), dtype=torch.uint8).long().view(batch, length)
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, length), generator=g)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="run yaml with a `model:` section")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. model.dc.chunk_ste=true")
    ap.add_argument("--text", default=None, help="bytes to feed (default: random bytes)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--steps", type=int, nargs="*", default=[0], help="audit after these many SGD steps (default 0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--progress", type=float, default=None, help="training progress in [0, 1] for curricula")
    ap.add_argument("--json", default=None, help="write all tables here")
    args = ap.parse_args(argv)
    cfg = _load_cfg(args.config, args.set)
    ids = _ids(args.text, args.batch, args.length, args.seed)
    report = {}
    for n in args.steps:
        table, metrics, groups = audit(cfg, ids, steps=n, seed=args.seed, progress=args.progress)
        print(f"\n== gradient norms by (objective, parameter group) after {n} steps")
        print(format_table(table, groups))
        print("\n== mechanism metrics")
        print("  " + "  ".join(f"{k}={v:.3g}" for k, v in sorted(metrics.items())))
        warn = degeneracies(table, metrics, groups)
        print("\n== warnings" if warn else "\n== no degeneracies flagged")
        for w in warn:
            print("  - " + w)
        report[str(n)] = {"grads": table, "metrics": metrics, "warnings": warn}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
