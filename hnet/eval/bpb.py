"""Bits per byte (SPEC definition) on the shared validation sets and on arbitrary texts.

* byte models:  sum NLL over targets that are real bytes (0..253) / ln 2 / count(those targets);
* token models: sum NLL over text tokens (no <|endoftext|> targets) / ln 2 / UTF-8 bytes of the same text.

``valset_bpb(model, valset, kind)`` runs the packed batches of ``hnet.data.valsets.ValSet`` (WS-C), whose labels
already carry IGNORE_INDEX on everything that is not a scored text element, so
``bpb = sum(out.ce_sum) / ln 2 / sum(batch["n_bytes"])`` and byte/token models are scored on exactly the same text.
As a guard we additionally mask byte targets outside 0..253 and token targets == <|endoftext|>.

``bpb_of_texts(scorer, texts)`` uses any scorer's ``rolling_loglikelihood`` (BOS/EOT prefix not scored, no
end-of-document target), so it also covers HF checkpoints with other tokenizers (SmolLM2/3).
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from hnet.config import BOS_ID, IGNORE_INDEX

LN2 = math.log(2.0)
EOT_ID = 50256


def _mask_labels(labels: torch.Tensor, kind: str) -> torch.Tensor:
    labels = labels.clone()
    if kind == "byte":
        labels[(labels >= BOS_ID)] = IGNORE_INDEX  # 254 BOS / 255 EOS are not bytes of text
    elif kind == "token":
        labels[labels == EOT_ID] = IGNORE_INDEX
    return labels


def _ce_sum(model, batch, labels, device, hf: bool = False):
    ids = batch["input_ids"].to(device)
    labels = labels.to(device)
    if not hf:
        out = model(ids, cu_seqlens=batch["cu_seqlens"].to(device), max_seqlen=int(batch["max_seqlen"]),
                    labels=labels, return_logits=False)
        if getattr(out, "ce_sum", None) is not None:
            return float(out.ce_sum)
        lg = out.logits.reshape(-1, out.logits.shape[-1]).float()
        return float(F.cross_entropy(lg, labels.reshape(-1), ignore_index=IGNORE_INDEX, reduction="sum"))
    # HF model: no packed attention -> run each segment as its own row
    cu = batch["cu_seqlens"].tolist()
    tot = 0.0
    for a, b in zip(cu[:-1], cu[1:]):
        lab = labels[a:b]
        if bool((lab == IGNORE_INDEX).all()):
            continue
        lg = model(input_ids=ids[a:b][None]).logits[0].float()
        tot += float(F.cross_entropy(lg, lab, ignore_index=IGNORE_INDEX, reduction="sum"))
    return tot


@torch.inference_mode()
def valset_bpb(model, valset, kind: str, device=None, amp_dtype: Optional[torch.dtype] = None,
               max_batches: Optional[int] = None, hf: bool = False) -> dict:
    """kind: "byte" (``valset.byte_batches()``) or "token" (``valset.token_batches()``, GPT-2 tokenizer)."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if amp_dtype is None and device.type == "cuda":
        amp_dtype = torch.bfloat16
    batches = valset.byte_batches() if kind == "byte" else valset.token_batches()
    if max_batches is not None:
        batches = batches[:max_batches]
    ce, nb, nt = 0.0, 0, 0
    was_training = getattr(model, "training", False)
    model.eval()
    amp = torch.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
    for b in batches:
        labels = _mask_labels(b["labels"], kind)
        with amp:
            ce += _ce_sum(model, b, labels, device, hf=hf)
        nb += int(b["n_bytes"])
        nt += int((labels != IGNORE_INDEX).sum())
    if was_training:
        model.train()
    return {"bpb": ce / LN2 / max(nb, 1), "ce_sum": ce, "n_bytes": nb, "n_targets": nt,
            "valset": getattr(valset, "name", None), "kind": kind}


def bpb_of_texts(scorer, texts: Sequence[str]) -> dict:
    """bpb of a list of texts with any scorer (rolling windows of ``scorer.max_len``)."""
    res = scorer.rolling_loglikelihood(list(texts))
    nll = -sum(lp for lp, _ in res)
    nb = sum(n for _, n in res)
    return {"bpb": nll / LN2 / max(nb, 1), "n_bytes": nb, "n_texts": len(texts),
            "per_text_bits": [-lp / LN2 for lp, _ in res], "per_text_bytes": [n for _, n in res]}


def valset_texts(valset) -> list:
    """Document texts of a ValSet (for scorers with other tokenizers, via ``bpb_of_texts``)."""
    from hnet.data.shards import decode_tokens_to_stream

    return [decode_tokens_to_stream(t)[0].tobytes().decode("utf-8", errors="replace")
            for t in valset._doc_tokens]
