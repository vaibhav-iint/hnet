"""Capture and intervene on the outer stage of a trained H-Net (analysis wrapper; the model code is not modified).

``capture(model, doc)`` replays ``HNetForCausalLM.forward`` + the outer ``HNet.forward`` step by step for one
document and returns every intermediate of stage 0 as numpy arrays (ids, router raw score / p' / mask, encoder
output, residual bypass, chunk vectors before and after the chunk vocabulary, main-network output, dechunked
hierarchy signal, decoder output, per-byte bits). ``decode(model, cap, hier=..., resid=...)`` re-runs only the
decoder + LM head on a modified decoder input, for path ablations. ``attention_profile`` records the attention
of every attention layer of the innermost main network over chunk offsets.

Byte classes (of the byte a quantity is about): ``space``, ``newline``, ``punct``, ``digit``, ``word_start``
(letter after a non-alphanumeric byte), ``word_end`` (letter followed by a non-letter: the last letter of a word),
``word_mid`` (other letters), ``utf8`` (non-ASCII), ``other``.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from hnet.config import BOS_ID
from hnet.model.ctx import SeqCtx
from hnet.model.hnet import _no_autocast

LN2 = math.log(2.0)
CLASSES = ["space", "newline", "punct", "digit", "word_start", "word_mid", "word_end", "utf8", "other"]


def _isalpha(b):
    return (65 <= b <= 90) or (97 <= b <= 122)


def _isalnum(b):
    return _isalpha(b) or (48 <= b <= 57)


def byte_classes(ids: List[int]) -> np.ndarray:
    """Class index (into CLASSES) of every byte, using its neighbours (word_end looks one byte ahead)."""
    n = len(ids)
    out = np.full(n, CLASSES.index("other"), dtype=np.int8)
    for t, b in enumerate(ids):
        prev = ids[t - 1] if t > 0 else -1
        nxt = ids[t + 1] if t + 1 < n else -1
        if b == BOS_ID and t == 0:
            c = "other"
        elif b >= 0x80:
            c = "utf8"
        elif b == 0x20 or b == 0x09:
            c = "space"
        elif b in (0x0A, 0x0D):
            c = "newline"
        elif 48 <= b <= 57:
            c = "digit"
        elif _isalpha(b):
            if not (prev >= 0 and _isalnum(prev)):
                c = "word_start"
            elif not (nxt >= 0 and _isalpha(nxt)):
                c = "word_end"
            else:
                c = "word_mid"
        else:
            c = "punct"
        out[t] = CLASSES.index(c)
    return out


def pos_in_word(ids: List[int]) -> np.ndarray:
    """For letters: index within the run of letters (0 = first); -1 for spaces right after a word; -2 otherwise."""
    out = np.full(len(ids), -2, dtype=np.int16)
    run = 0
    for t, b in enumerate(ids):
        if _isalpha(b) and b < 0x80:
            out[t] = run
            run += 1
        else:
            if b == 0x20 and run > 0:
                out[t] = -1
            run = 0
    return out


@torch.no_grad()
def capture(model, doc: bytes, keep_hidden: bool = True) -> Dict[str, np.ndarray]:
    """One document (BOS prepended) through the model, stage 0 opened up."""
    st = model.backbone
    assert not st.is_innermost
    dev = next(model.parameters()).device
    ids = torch.tensor([BOS_ID] + list(doc), dtype=torch.long, device=dev)
    T = ids.numel()
    ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, T], dtype=torch.int32, device=dev), T, T)
    x = model.embeddings(ids)
    if model.ngram_embeddings is not None:
        x = x + model.ngram_embeddings(ids, ctx)
    binfo = None
    vocab_stages = [s for s in st.stages() if getattr(s, "chunk_vocab", None) is not None]
    if vocab_stages:
        from hnet.model.chunk_vocab import ByteInfo, hash_bank

        bank = hash_bank(ids, ctx, max(s.chunk_vocab.n_hash for s in vocab_stages),
                         max(s.chunk_vocab.max_len for s in vocab_stages))
        binfo = ByteInfo(bank, torch.arange(T, device=dev))
    h = x
    if st.pad_dimension is not None:
        h = torch.cat((h, st.pad_dimension.to(h.dtype).expand(h.shape[0], -1)), dim=-1)
    h = st.encoder(h, ctx)
    with _no_autocast(h.device):
        residual = st.residual_proj(h.float())
    r = st.routing_module(h, ctx, input_ids=ids, boundaries=None)
    raw_logit = r.logit.clone() if r.logit is not None else torch.logit(r.p.clamp(1e-6, 1 - 1e-6))
    if st.decider is not None:
        r = st.decider(r, ctx)
    xs, ctx_inner = st.chunk_layer(h, r.boundary_mask, ctx)
    x_enc = xs.clone()
    vocab_emb = None
    if st.chunk_vocab is not None:
        bucket = st.chunk_vocab.chunk_ids(binfo, ctx, ctx_inner)
        vocab_emb = st.chunk_vocab.embed(bucket, xs.dtype)
        xs = xs + vocab_emb
    inner_ids = ids.index_select(0, ctx_inner.outer_index)
    inner_binfo = binfo.select(ctx_inner.outer_index) if binfo is not None else None
    z, inner_preds = st.main_network(xs, ctx_inner, input_ids=inner_ids, boundaries=None, binfo=inner_binfo)
    out = st.dechunk_layer(z, r, ctx, ctx_inner)
    if st.residual_gate == "margin":
        margin = (2 * (r.p.detach().float() - 0.5).abs().mean()).clamp(0, 1)
        residual = residual * margin
    dec_in = (out + residual).to(h.dtype)
    hd = st.decoder(dec_in, ctx)[:, : x.shape[-1]]
    logits = model.lm_head(hd).float()
    labels = torch.roll(ids, -1)
    nll = F.cross_entropy(logits, labels, reduction="none") / LN2
    nll[-1] = float("nan")
    ilist = ids.tolist()
    probs = logits.softmax(-1)
    alpha = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    alpha[65:91] = True
    alpha[97:123] = True
    p_next_nonalpha = probs[:, ~alpha].sum(-1)
    capd = {
        "ids": ids.cpu().numpy().astype(np.int16),
        "cls": byte_classes(ilist),
        "piw": pos_in_word(ilist),
        "bits": nll.cpu().numpy(),
        "logit": raw_logit.float().cpu().numpy(),
        "p": r.p.float().cpu().numpy(),
        "mask": r.boundary_mask.cpu().numpy().astype(bool),
        "sel_idx": ctx_inner.outer_index.cpu().numpy(),
        "out_rms": out.float().pow(2).mean(-1).sqrt().cpu().numpy(),
        "res_rms": residual.float().pow(2).mean(-1).sqrt().cpu().numpy(),
        "cos_out_res": F.cosine_similarity(out.float(), residual.float(), dim=-1).cpu().numpy(),
        "p_next_nonalpha": p_next_nonalpha.float().cpu().numpy(),
    }
    if inner_preds:
        capd["inner_mask"] = inner_preds[0].boundary_mask.cpu().numpy().astype(bool)
        capd["inner_p"] = inner_preds[0].p.float().cpu().numpy()
    if keep_hidden:
        capd.update({
            "h_enc": h.float().cpu().numpy(), "residual": residual.float().cpu().numpy(),
            "out": out.float().cpu().numpy(), "x_enc": x_enc.float().cpu().numpy(), "x": xs.float().cpu().numpy(),
            "z": z.float().cpu().numpy(), "h_dec": hd.float().cpu().numpy(),
        })
        if vocab_emb is not None:
            capd["vocab_emb"] = vocab_emb.float().cpu().numpy()
    capd["_ctx"] = ctx
    return capd


@torch.no_grad()
def decode(model, cap: dict, hier: Optional[np.ndarray] = None, resid: Optional[np.ndarray] = None) -> np.ndarray:
    """Bits per byte with the decoder input replaced by hier + resid (arrays [T, D]; defaults: the captured ones)."""
    st = model.backbone
    dev = next(model.parameters()).device
    out = torch.from_numpy(cap["out"] if hier is None else hier).to(dev)
    res = torch.from_numpy(cap["residual"] if resid is None else resid).to(dev)
    ctx = cap["_ctx"]
    hd = st.decoder((out + res).float(), ctx)[:, : st.decoder.rmsnorm.weight.shape[0]]
    logits = model.lm_head(hd[:, : model.lm_head.weight.shape[1]]).float()
    ids = torch.from_numpy(cap["ids"].astype(np.int64)).to(dev)
    nll = F.cross_entropy(logits, torch.roll(ids, -1), reduction="none") / LN2
    nll[-1] = float("nan")
    return nll.cpu().numpy()


@torch.no_grad()
def lens_bits(model, vec: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Logit lens: the decoder's final RMSNorm + LM head applied directly to a decoder-input component."""
    st = model.backbone
    dev = next(model.parameters()).device
    v = torch.from_numpy(vec).to(dev).float()
    hn = st.decoder.rmsnorm(v, out_dtype=torch.float32) if "out_dtype" in st.decoder.rmsnorm.forward.__code__.co_varnames else st.decoder.rmsnorm(v)
    logits = model.lm_head(hn[:, : model.lm_head.weight.shape[1]].to(model.lm_head.weight.dtype)).float()
    t = torch.from_numpy(ids.astype(np.int64)).to(dev)
    nll = F.cross_entropy(logits, torch.roll(t, -1), reduction="none") / LN2
    nll[-1] = float("nan")
    return nll.cpu().numpy()


def attention_profile(model, doc: bytes, bins=(0, 1, 2, 4, 8, 16, 32, 1 << 30)) -> dict:
    """Mean attention mass of each innermost attention layer over relative offsets (in chunks), plus to position 0
    (the BOS chunk) and the mean entropy, for one document."""
    from hnet.model.mixers.attn import CausalAttention, apply_rotary, rotary_cos_sin

    inner = model.backbone
    while not inner.is_innermost:
        inner = inner.main_network
    layers = [m for m in inner.main_network.modules() if isinstance(m, CausalAttention)]
    rec = []

    def make(mod):
        orig = mod.forward

        def fwd(x, ctx):
            T = x.shape[1]
            qkv = mod.Wqkv(x[0])
            hq, hk, hd = mod.num_heads, mod.num_kv_heads, mod.head_dim
            q, k, v = qkv.split((hq * hd, hk * hd, hk * hd), dim=-1)
            q, k = q.view(T, hq, hd), k.view(T, hk, hd)
            if mod.rotary_emb_dim > 0:
                cos, sin = rotary_cos_sin(ctx, mod.rotary_emb_dim, mod.rope_theta)
                q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            if hk != hq:
                k = k.repeat_interleave(hq // hk, dim=1)
            att = torch.einsum("thd,shd->hts", q.float(), k.float()) * mod.softmax_scale
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            if mod.window_size > 0:
                causal &= ~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-mod.window_size - 1)
            att = att.masked_fill(~causal, float("-inf")).softmax(-1)  # [H, T, S]
            off = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).clamp(min=0)
            res = {}
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = ((off >= lo) & (off < hi)).float()
                res[f"off{lo}-{hi - 1 if hi < 1 << 29 else 'inf'}"] = float((att * m).sum(-1).mean())
            res["to_pos0"] = float(att[:, :, 0].mean())
            ent = -(att.clamp_min(1e-12).log() * att).sum(-1)
            res["entropy_nats"] = float(ent.mean())
            res["mean_offset"] = float((att * off.float()).sum(-1).mean())
            rec.append(res)
            return orig(x, ctx)

        mod.forward = fwd

    for m in layers:
        make(m)
    try:
        capture(model, doc, keep_hidden=False)
    finally:
        for m in layers:
            del m.forward
    return {f"layer{i}": r for i, r in enumerate(rec)}


def null() -> nullcontext:
    return nullcontext()
