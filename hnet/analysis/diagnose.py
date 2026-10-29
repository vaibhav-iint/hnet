"""Diagnostics for trained H-Nets (and their BPE baselines) on held-out documents.

    python -m hnet.analysis.diagnose RUN_DIR[/ckpt/step_N] [--bpe BPE_RUN_DIR] [--docs 128] [--max-bytes 2048]

What it measures, on the same packed validation documents for every model:

1. Per-byte loss (bits) broken down by byte class (space, word start, word-internal, punctuation, digit, newline,
   UTF-8 continuation) and, per stage, by whether the predicted byte opens a chunk and by its position in the chunk.
2. Against a BPE model on the identical text: per GPT-2 token, the H-Net's summed bits over the token's bytes vs the
   BPE model's bits for the token; the gap is split into the token's first byte and its remaining bytes.
3. Router: boundary-probability histogram, share of confident decisions, realized bytes per chunk, share of chunk
   starts that are word starts (previous byte is whitespace) and that fall inside a UTF-8 character.
4. Use of the hierarchy: loss when the output of a stage's main network is zeroed (the decoder then only sees the
   residual path), and the RMS of the dechunked signal vs the residual entering each decoder.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hnet.config import BOS_ID
from hnet.data.shards import read_byte_shard
from hnet.eval.run import load_run_checkpoint, model_kind
from hnet.model.probes import zero_main

LN2 = math.log(2.0)
WS = b" \t\n\r"
VAL_BYTES = "data/finewebedu10B/bytes/finewebedu_val_000000.bytes.bin"
VAL_IDX = "data/finewebedu10B/bytes/finewebedu_val_000000.idx.npy"


# ----------------------------------------------------------------------------------------------------------- data
def load_docs(n_docs: int, max_bytes: int, min_bytes: int = 256, seed: int = 0, root: str = ".") -> List[bytes]:
    """``n_docs`` validation documents (deterministic sample), each cut to at most ``max_bytes`` bytes."""
    data = read_byte_shard(Path(root) / VAL_BYTES)
    idx = np.load(Path(root) / VAL_IDX)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx))
    docs = []
    for i in order:
        off, n = int(idx[i, 0]), int(idx[i, 1])
        if n >= min_bytes:
            d = bytes(data[off: off + min(n, max_bytes + 1)])
            assert d[0] == BOS_ID, "byte shards store each document with its leading BOS byte"
            docs.append(d[1:])  # _pack adds the BOS back
        if len(docs) == n_docs:
            break
    return docs


def byte_class(prev: int, cur: int) -> str:
    if 0x80 <= cur < 0xC0:
        return "utf8_cont"
    if cur >= 0xC0:
        return "utf8_lead"
    c = chr(cur)
    if c == " ":
        return "space"
    if c in "\n\r":
        return "newline"
    if c.isalpha():
        return "word_start" if (prev < 0 or chr(prev) in " \t\n\r" or not chr(prev).isalnum()) else "word_internal"
    if c.isdigit():
        return "digit"
    return "punct"


# ------------------------------------------------------------------------------------------------------- byte model
def _pack(docs: List[bytes], device):
    seqs = [torch.tensor([BOS_ID] + list(d), dtype=torch.long) for d in docs]
    lens = [len(s) for s in seqs]
    ids = torch.cat(seqs).to(device)
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32, device=device)
    return ids, cu, max(lens)


def _labels(ids: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    labels = torch.roll(ids, -1)
    labels[(cu[1:] - 1).long()] = -100  # last position of each segment has no target
    return labels


@contextlib.contextmanager
def capture_decoder_inputs(model):
    """Record RMS of the dechunked signal and of the residual entering each stage's decoder."""
    rec = {}
    handles = []
    for st in model.backbone.stages():
        if st.is_innermost:
            continue

        def make(s):
            orig_dechunk = s.dechunk_layer  # a plain callable or an nn.Module
            orig_res = s.residual_proj.forward

            def dech(*a, **k):
                out = orig_dechunk(*a, **k)
                rec.setdefault(s.stage_idx, {})["dechunk_rms"] = out.float().pow(2).mean().sqrt().item()
                return out

            def res(x):
                out = orig_res(x)
                rec.setdefault(s.stage_idx, {})["residual_rms"] = out.float().pow(2).mean().sqrt().item()
                rec[s.stage_idx]["encoder_rms"] = x.float().pow(2).mean().sqrt().item()
                return out

            object.__setattr__(s, "_diag_dechunk", orig_dechunk)
            if isinstance(orig_dechunk, torch.nn.Module):
                orig_dechunk.forward = dech
                dech_orig_fwd = type(orig_dechunk).forward.__get__(orig_dechunk)
                orig_dechunk = dech_orig_fwd
            else:
                s.dechunk_layer = dech
            s.residual_proj.forward = res
            handles.append((s, s._diag_dechunk, orig_res))

        make(st)
    try:
        yield rec
    finally:
        for s, d, r in handles:
            if isinstance(d, torch.nn.Module):
                del d.forward
            else:
                s.dechunk_layer = d
            del s.residual_proj.forward


@torch.no_grad()
def byte_losses(model, ids, cu, max_len) -> torch.Tensor:
    out = model(ids, cu_seqlens=cu, max_seqlen=max_len, return_logits=True)
    labels = _labels(ids, cu)
    nll = F.cross_entropy(out.logits.float(), labels.clamp(min=0), reduction="none")
    return torch.where(labels >= 0, nll, torch.zeros_like(nll)) / LN2, labels >= 0, out


def chunk_starts_per_stage(out, T: int) -> List[torch.Tensor]:
    """Per stage, bool [T] over packed byte positions: True where a chunk of that stage begins."""
    positions = torch.arange(T)
    starts = []
    for r in out.bpred:
        mask = r.boundary_mask.reshape(-1).bool().cpu()
        b = positions[mask]
        s = torch.zeros(T, dtype=torch.bool)
        if getattr(r, "semantics", "start") == "end":
            s[(b + 1).clamp(max=T - 1)] = True
        else:
            s[b] = True
        starts.append(s)
        positions = b
    return starts


def router_stats(ps: List[torch.Tensor], masks: List[torch.Tensor]) -> List[dict]:
    """ps/masks: per stage, fp32 [N_s] boundary probabilities and bool [N_s] decisions (all documents)."""
    res = []
    for s, (p, mask) in enumerate(zip(ps, masks)):
        hist, _ = np.histogram(p.numpy(), bins=[0, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 1.0001])
        res.append({
            "stage": s, "positions": int(p.numel()), "selected": int(mask.sum()),
            "bytes_per_chunk_vs_prev_stage": round(p.numel() / max(1, int(mask.sum())), 3),
            "p_mean": round(float(p.mean()), 4),
            "p_hist": [round(float(x), 4) for x in hist / hist.sum()],
            "confident(<.05|>.95)": round(float(((p < 0.05) | (p > 0.95)).float().mean()), 4),
            "near_half(.4-.6)": round(float(((p > 0.4) & (p < 0.6)).float().mean()), 4),
        })
    return res


def boundary_alignment(starts: List[torch.Tensor], ids: torch.Tensor) -> List[dict]:
    raw = ids.cpu().tolist()
    res = []
    for s, st in enumerate(starts):
        pos = torch.nonzero(st).flatten().tolist()
        pos = [p for p in pos if p > 0 and raw[p] != BOS_ID and raw[p - 1] != BOS_ID]
        n = max(1, len(pos))
        after_ws = sum(raw[p - 1] in WS for p in pos) / n
        mid_utf8 = sum(0x80 <= raw[p] < 0xC0 for p in pos) / n
        word_internal = sum(chr(raw[p - 1]).isalnum() and raw[p] < 0x80 and chr(raw[p]).isalnum()
                            for p in pos if raw[p - 1] < 0x80) / n
        res.append({"stage": s, "chunk_starts": len(pos), "after_whitespace": round(after_ws, 4),
                    "word_internal": round(word_internal, 4), "mid_utf8": round(mid_utf8, 4)})
    return res


def breakdown(bits: torch.Tensor, valid: torch.Tensor, ids: torch.Tensor, starts: List[torch.Tensor]) -> dict:
    """Bits per predicted byte by byte class, and per stage by chunk position of the predicted byte."""
    raw = ids.cpu().tolist()
    tgt = torch.roll(ids, -1).cpu().tolist()
    b = bits.cpu().tolist()
    v = valid.cpu().tolist()
    by_class = defaultdict(lambda: [0.0, 0])
    per_stage = [defaultdict(lambda: [0.0, 0]) for _ in starts]
    run = [0] * len(starts)
    for t in range(len(raw)):
        for s, st in enumerate(starts):  # position within chunk of byte t+1 (the predicted byte)
            run[s] = 0 if (t + 1 < len(raw) and st[t + 1]) else run[s] + 1
        if not v[t]:
            continue
        cls = byte_class(raw[t] if raw[t] != BOS_ID else -1, tgt[t])
        by_class[cls][0] += b[t]
        by_class[cls][1] += 1
        for s in range(len(starts)):
            k = "chunk_start" if run[s] == 0 else ("pos1" if run[s] == 1 else ("pos2" if run[s] == 2 else "pos3+"))
            per_stage[s][k][0] += b[t]
            per_stage[s][k][1] += 1
    tot = sum(n for _, n in by_class.values())

    def fmt(d):
        return {k: {"bits": round(x / n, 4), "share_bytes": round(n / tot, 4), "share_bits": None, "_sum": x}
                for k, (x, n) in sorted(d.items())}

    out = {"by_class": fmt(by_class), "by_chunk_pos": [fmt(d) for d in per_stage]}
    total_bits = sum(x for x, _ in by_class.values())
    for grp in [out["by_class"], *out["by_chunk_pos"]]:
        for k in grp:
            grp[k]["share_bits"] = round(grp[k].pop("_sum") / total_bits, 4)
    return out


def help_breakdown(delta: torch.Tensor, valid: torch.Tensor, ids: torch.Tensor, starts: List[torch.Tensor]) -> dict:
    """Mean bits saved per predicted byte (zeroed-main loss - normal loss) by byte class and by stage-0 chunk
    position, plus the share of all saved bits."""
    raw = ids.tolist()
    tgt = torch.roll(ids, -1).tolist()
    dl = delta.tolist()
    v = valid.tolist()
    st0 = starts[0].tolist()
    by = defaultdict(lambda: [0.0, 0])
    run = 0
    for t in range(len(raw)):
        run = 0 if (t + 1 < len(raw) and st0[t + 1]) else run + 1
        if not v[t]:
            continue
        cls = byte_class(raw[t] if raw[t] != BOS_ID else -1, tgt[t])
        for k in (f"class:{cls}", "chunk0:start" if run == 0 else "chunk0:inside", "ALL"):
            by[k][0] += dl[t]
            by[k][1] += 1
    total = by["ALL"][0]
    return {k: {"bits_saved": round(x / n, 4), "share_of_saving": round(x / total, 4) if total else None,
                "share_bytes": round(n / by["ALL"][1], 4)} for k, (x, n) in sorted(by.items())}


# ------------------------------------------------------------------------------------------------------ BPE model
def bpe_token_bits(model, docs: List[bytes], device) -> List[List[tuple]]:
    """Per doc: [(token_byte_start, token_byte_len, bits)] for every GPT-2 token (first token predicted from EOT)."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    res = []
    for d in docs:
        toks = enc.encode(d.decode("utf-8", errors="replace"), allowed_special=set())
        # map tokens back to byte spans of the *decoded-then-reencoded* text (identical for valid UTF-8 docs)
        spans, pos = [], 0
        for t in toks:
            n = len(enc.decode_single_token_bytes(t))
            spans.append((pos, n))
            pos += n
        x = torch.tensor([eot] + toks, dtype=torch.long, device=device)
        cu = torch.tensor([0, x.numel()], dtype=torch.int32, device=device)
        with torch.no_grad():
            out = model(x, cu_seqlens=cu, max_seqlen=x.numel(), return_logits=True)
        nll = F.cross_entropy(out.logits[:-1].float(), x[1:], reduction="none") / LN2
        res.append([(s, n, float(b)) for (s, n), b in zip(spans, nll.tolist())])
    return res


def compare_with_bpe(hnet_bits_per_doc: List[List[float]], bpe_tokens: List[List[tuple]], docs: List[bytes]) -> dict:
    """Split the bits gap (H-Net - BPE) over GPT-2 tokens into first-byte and rest-of-token parts, by token type."""
    agg = defaultdict(lambda: {"n": 0, "bytes": 0, "bpe": 0.0, "hnet": 0.0, "hnet_first": 0.0})
    for hb, toks, d in zip(hnet_bits_per_doc, bpe_tokens, docs):
        if len(hb) != len(d) or sum(n for _, n, _ in toks) != len(d):
            continue  # doc not valid UTF-8 (e.g. cut mid-character): token spans don't align with its bytes
        for s, n, bits in toks:
            tok = d[s: s + n]
            if tok.startswith(b" ") and tok[1:].isalpha():
                kind = "space+word"
            elif tok.isalpha():
                kind = "word_piece"
            elif tok.strip(b" ").isdigit():
                kind = "number"
            elif tok.strip(b" \t") == b"" or b"\n" in tok:
                kind = "whitespace/newline"
            else:
                kind = "punct/other"
            for k in (kind, "ALL"):
                a = agg[k]
                a["n"] += 1
                a["bytes"] += n
                a["bpe"] += bits
                a["hnet"] += sum(hb[s: s + n])
                a["hnet_first"] += hb[s]
    out = {}
    tot_bytes = agg["ALL"]["bytes"]
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["bytes"]):
        gap = a["hnet"] - a["bpe"]
        out[k] = {"tokens": a["n"], "share_bytes": round(a["bytes"] / tot_bytes, 4),
                  "bpe_bits_per_byte": round(a["bpe"] / a["bytes"], 4),
                  "hnet_bits_per_byte": round(a["hnet"] / a["bytes"], 4),
                  "gap_bits_per_byte": round(gap / a["bytes"], 4),
                  "share_of_total_gap": None, "_gap": gap,
                  "hnet_first_byte_bits": round(a["hnet_first"] / a["n"], 3),
                  "hnet_rest_bits_per_byte": round((a["hnet"] - a["hnet_first"]) / max(1, a["bytes"] - a["n"]), 4),
                  "bpe_bits_per_token": round(a["bpe"] / a["n"], 3)}
    total_gap = out["ALL"]["_gap"]
    for k in out:
        out[k]["share_of_total_gap"] = round(out[k].pop("_gap") / total_gap, 4) if total_gap else None
    return out


# ------------------------------------------------------------------------------------------------------------ main
def _run_docs(model, docs, device):
    """Run documents one at a time (CPU attention masks are dense). Returns concatenated per-byte arrays."""
    acc = defaultdict(list)
    dec_acc = defaultdict(lambda: defaultdict(list))
    n_stages = None
    for d in docs:
        ids, cu, max_len = _pack([d], device)
        with capture_decoder_inputs(model) as dec:
            bits, valid, out = byte_losses(model, ids, cu, max_len)
        for k, v in dec.items():
            for kk, vv in v.items():
                dec_acc[k][kk].append(vv)
        starts = chunk_starts_per_stage(out, ids.numel())
        n_stages = len(starts)
        acc["ids"].append(ids.cpu())
        acc["bits"].append(bits.cpu())
        acc["valid"].append(valid.cpu())
        for s, st in enumerate(starts):
            acc[f"start{s}"].append(st)
        for s, r in enumerate(out.bpred):
            acc[f"p{s}"].append(r.p.reshape(-1).float().cpu())
            acc[f"m{s}"].append(r.boundary_mask.reshape(-1).bool().cpu())
    cat = {k: torch.cat(v) for k, v in acc.items()}
    dec = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in dec_acc.items()}
    return cat, dec, n_stages


def diagnose(path: str, docs: List[bytes], bpe_path: Optional[str] = None, device="cpu") -> dict:
    model, mcfg, meta = load_run_checkpoint(path, device=device)
    rep: Dict = {"checkpoint": meta["checkpoint"], "kind": model_kind(mcfg), "docs": len(docs)}
    if rep["kind"] != "byte":
        raise SystemExit(f"{path}: diagnose expects an H-Net; pass BPE runs with --bpe")
    cat, dec, n_stages = _run_docs(model, docs, device)
    bits, valid, ids = cat["bits"], cat["valid"], cat["ids"]
    rep["bpb"] = round(float(bits.sum()) / int(valid.sum()), 4)
    rep["decoder_inputs"] = {f"s{k}": {kk: round(vv, 4) for kk, vv in v.items()} for k, v in sorted(dec.items())}
    starts = [cat[f"start{s}"] for s in range(n_stages)]
    rep["router"] = router_stats([cat[f"p{s}"] for s in range(n_stages)], [cat[f"m{s}"] for s in range(n_stages)])
    rep["alignment"] = boundary_alignment(starts, ids)
    rep["breakdown"] = breakdown(bits, valid, ids, starts)
    rep["zero_main_bpb"] = {}
    rep["main_help"] = {}  # where the hierarchy below each stage helps: bits saved per predicted byte, by class
    for st in model.backbone.stages():
        if st.is_innermost:
            continue
        with zero_main(st):
            zc, _, _ = _run_docs(model, docs, device)
        rep["zero_main_bpb"][f"s{st.stage_idx}"] = round(float(zc["bits"].sum()) / int(zc["valid"].sum()), 4)
        rep["main_help"][f"s{st.stage_idx}"] = help_breakdown(zc["bits"] - bits, valid, ids, starts)
    if bpe_path:
        bmodel, bcfg, bmeta = load_run_checkpoint(bpe_path, device=device)
        toks = bpe_token_bits(bmodel, docs, device)
        per_doc, off = [], 0
        bl = bits.tolist()
        for d in docs:  # bits for predicting doc byte j sit at packed position off + j (BOS shift)
            per_doc.append(bl[off: off + len(d)])
            off += len(d) + 1
        rep["bpe_checkpoint"] = bmeta["checkpoint"]
        cmp = compare_with_bpe(per_doc, toks, docs)
        rep["vs_bpe"] = cmp
        rep["bpe_bpb_same_text"] = cmp["ALL"]["bpe_bits_per_byte"]
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--bpe", default=None)
    ap.add_argument("--docs", type=int, default=128)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--out", default=None, help="write the JSON report here")
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    docs = load_docs(a.docs, a.max_bytes)
    rep = diagnose(a.run, docs, a.bpe)
    txt = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
