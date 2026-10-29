"""Manual audit of eval prompts and scoring (CPU).

For a few items per task it writes the exact strings scored (context | continuation), gold answer and the
length-normalization text, then recomputes every choice's log-probability independently:

* byte model (tiny H-Net, random init): one unpacked forward over BOS + context + continuation bytes, summing
  log-softmax over the continuation bytes only (after moving trailing context spaces into the continuation, as the
  scorer does), vs ByteModelScorer (which packs all requests into shared batches);
* HF GPT-2 (cached): forward over tokenizer(context + continuation) with the lm-eval split, vs HFScorer.

It then re-derives acc / acc_norm_char / acc_norm_byte / soft bpb from the recomputed numbers and compares them with the
harness records. Output: logbook/audits/eval_examples.md

    PYTHONPATH=. python scripts/audit/audit_eval.py
"""

import sys as _sys
from pathlib import Path as _Path

REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _sys.path:  # allow `python scripts/...` from anywhere without PYTHONPATH
    _sys.path.insert(0, str(REPO_ROOT))

import math
from pathlib import Path

import numpy as np
import torch

from hnet.config import BOS_ID, model_config_from_yaml
from hnet.eval.harness import score_items
from hnet.eval.scorers import ByteModelScorer, HFScorer, split_trailing_space
from hnet.eval.tasks import get_task
from hnet.eval.tasks.base import MCItem
from hnet.model import build_model

torch.manual_seed(0)
LN2 = math.log(2)
OUT = Path("logbook/audits/eval_examples.md")
TASKS = {"paper7": ["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "openbookqa", "lambada_openai"],
         "smollm3_cf": ["hellaswag", "arc_easy", "piqa", "winogrande", "commonsenseqa", "mmlu", "boolq"]}
lines, fails = [], []


def check(cond, msg):
    if not cond:
        fails.append(msg)


@torch.no_grad()
def manual_byte_lp(model, ctx, cont):
    ctx, cont = split_trailing_space(ctx, cont)
    c, k = ctx.encode(), cont.encode()
    ids = torch.tensor([BOS_ID] + list(c) + list(k))
    T = ids.numel()
    out = model(ids, cu_seqlens=torch.tensor([0, T], dtype=torch.int32), max_seqlen=T)
    logp = torch.log_softmax(out.logits.float(), -1)
    pos = torch.arange(1 + len(c) - 1, T - 1)  # positions predicting the continuation bytes
    return float(logp[pos, ids[pos + 1]].sum()), len(k)


@torch.no_grad()
def manual_hf_lp(hf, ctx, cont):
    tok, model = hf.tok, hf.model
    ctx, cont = split_trailing_space(ctx, cont)
    whole = tok(ctx + cont)["input_ids"]
    ctx_ids = tok(ctx)["input_ids"]
    n_ctx = len(ctx_ids)
    ids = torch.tensor([tok.bos_token_id] + whole) if hf.add_bos else torch.tensor(whole)
    off = 1 if hf.add_bos else 0
    logp = torch.log_softmax(model(ids[None]).logits[0].float(), -1)
    tgt = range(off + n_ctx, ids.numel())
    return float(sum(logp[t - 1, ids[t]] for t in tgt))


byte_model = build_model(model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml"))
byte_model.init_weights()
byte_model.eval()
byte_scorer = ByteModelScorer(byte_model, device="cpu")
hf = HFScorer("gpt2", device="cpu")

for suite, names in TASKS.items():
    for name in names:
        task = get_task(f"{suite}/{name}")
        items = task.items(limit=2, subset_seed=0)
        lines.append(f"\n## {suite} / {name}  ({task.full_name})\n")
        for scorer_name, scorer in (("byte", byte_scorer), ("gpt2", hf)):
            recs = score_items(scorer, items)
            for it, rec in zip(items, recs):
                reqs = it.requests
                manual = []
                for ctx, cont in reqs:
                    if scorer_name == "byte":
                        lp, _ = manual_byte_lp(byte_model, ctx, cont)
                    else:
                        lp = manual_hf_lp(hf, ctx, cont)
                    manual.append(lp)
                lp_h = rec["lp"] if isinstance(rec["lp"], list) else [rec["lp"]]
                diff = max(abs(a - b) for a, b in zip(manual, lp_h))
                check(diff < 2e-3, f"{suite}/{name}/{scorer_name} doc {it.doc_id}: logprob mismatch {diff:.2e}")
                if isinstance(it, MCItem):
                    nc = [len(t) for t in it.norm_texts]
                    nb = [len(split_trailing_space(c, k)[1].encode()) for c, k in reqs]
                    exp = {"acc": int(np.argmax(manual)),
                           "acc_norm_char": int(np.argmax(np.array(manual) / np.maximum(nc, 1))),
                           "acc_norm_byte": int(np.argmax(np.array(manual) / np.maximum(nb, 1)))}
                    for m, p in exp.items():
                        score = {"acc": np.array(manual), "acc_norm_char": np.array(manual) / np.maximum(nc, 1),
                                 "acc_norm_byte": np.array(manual) / np.maximum(nb, 1)}[m]
                        tie = abs(score[rec["pred"][m]] - score[p]) < 1e-4  # argmax among numerically tied choices
                        check(rec["pred"][m] == p or tie, f"{suite}/{name}/{scorer_name} doc {it.doc_id}: {m} pred mismatch "
                              f"(harness lp {np.round(lp_h, 4).tolist()} vs manual {np.round(manual, 4).tolist()})")
                    g = it.gold[0]
                    check(abs(rec["gold_nll_bits"] - (-manual[g] / LN2)) < 3e-3 and rec["gold_bytes"] == nb[g],
                          f"{suite}/{name}/{scorer_name} doc {it.doc_id}: soft bpb inputs mismatch")
                if scorer_name == "gpt2":  # print strings once
                    lines.append(f"**doc {it.doc_id}**, gold = {getattr(it, 'gold', '(single continuation)')}  \n")
                    for i, (ctx, cont) in enumerate(reqs):
                        mark = "✓" if i in getattr(it, "gold", [0]) else " "
                        nt = it.norm_texts[i] if isinstance(it, MCItem) else ""
                        lines.append(f"- {mark} ctx=`{ctx[-90:]!r}` cont=`{cont!r}` norm_len={len(nt)} "
                                     f"gpt2_lp={manual[i]:.2f}")
                    lines.append("")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("# Eval audit: exact scored strings and independent re-scoring\n\n"
               "Generated by `scripts/audit/audit_eval.py`. Context strings are truncated to their last 90 characters; "
               "`repr` shows spaces and newlines exactly. gpt2_lp = log-probability of the continuation under GPT-2 "
               "small, recomputed by an unpacked forward pass.\n" + "\n".join(lines) + "\n")
print(f"items audited; {len(fails)} mismatches")
for f in fails[:30]:
    print("FAIL", f)
print("wrote", OUT)
