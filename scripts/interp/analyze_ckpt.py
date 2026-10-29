"""Activation-level analysis of one H-Net checkpoint (outer stage), on validation documents.

    python scripts/interp/analyze_ckpt.py CKPT_OR_RUN_DIR [--docs 48] [--max-bytes 2048] [--out NAME.json]

Sections of the JSON (all bits are bits per predicted byte; "class" = class of the predicted byte unless noted):
  router      raw-score and p' statistics by class of the boundary-candidate byte, where boundaries fall,
              eval-mode compression, histograms of the standardized score by class, n-gram R^2 of the raw score
              (how much of the decision is a function of the last n bytes: the context dependence)
  tokenizer   chunk strings: types, singletons, whitespace alignment / hit rate, word consistency, top chunks
  paths       decoder input: rms of the hierarchy (dechunked) and bypass (residual) signals, their cosine; bits
              under ablations (hierarchy zeroed / mean-replaced, bypass zeroed / mean-replaced, both) by class and
              by position in word; logit lens of each path through the decoder's final norm + LM head;
              input x gradient attribution of each path
  probes      held-out linear probes from the chunk vector before the vocabulary (x_enc), after it (x) and from
              the main-network output (z): chunk identity (top-255 strings), length, first byte, last byte,
              next chunk's first byte, previous chunk's last byte; unigram baselines; effective ranks
  attention   attention mass of every main-network attention layer over chunk offsets (8 documents)
"""

from __future__ import annotations

import argparse
import collections
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acts import CLASSES, attention_profile, capture, decode, lens_bits  # noqa: E402
from common import DATA, dump  # noqa: E402

from hnet.analysis.diagnose import load_docs  # noqa: E402
from hnet.eval.run import load_run_checkpoint  # noqa: E402

LN2 = math.log(2.0)


def eff_rank(X: np.ndarray) -> dict:
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    s = s / s.sum()
    pr = 1.0 / (s ** 2).sum()
    ent = float(np.exp(-(s * np.log(s + 1e-12)).sum()))
    return {"participation_ratio": float(pr), "entropy_rank": ent, "dim": int(X.shape[1]),
            "top1_var": float(s[0]), "top10_var": float(s[:10].sum())}


def probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_cls: int, steps: int = 250) -> dict:
    """Multinomial logistic probe, split by document (60/20/20 train/val/test), standardized inputs, weight decay
    chosen on the validation docs. Held-out bits and accuracy, against the train-set unigram distribution."""
    rng = np.random.default_rng(0)
    ug = rng.permutation(np.unique(groups))
    n = len(ug)
    te_g, va_g = set(ug[: max(1, n // 5)].tolist()), set(ug[max(1, n // 5): max(2, 2 * n // 5)].tolist())
    te = np.array([g in te_g for g in groups])
    va = np.array([g in va_g for g in groups])
    tr = ~(te | va)
    if tr.sum() < 50 or te.sum() < 20 or va.sum() < 20:
        return {}
    Xt = torch.tensor(X, dtype=torch.float32)
    mu, sd = Xt[tr].mean(0), Xt[tr].std(0) + 1e-5
    Xt = (Xt - mu) / sd
    yt = torch.tensor(y, dtype=torch.long)
    cnt = torch.bincount(yt[tr], minlength=n_cls).float() + 0.5
    prior = torch.log(cnt / cnt.sum())

    def fit(wd):
        W = torch.zeros(X.shape[1], n_cls, requires_grad=True)
        b = prior.clone().requires_grad_(True)
        opt = torch.optim.Adam([W, b], lr=0.03)
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                loss = F.cross_entropy(Xt[tr] @ W + b, yt[tr]) + wd * (W ** 2).sum()
                loss.backward()
                opt.step()
        return W.detach(), b.detach()

    best = None
    for wd in (1e-4, 1e-3, 1e-2):
        W, b = fit(wd)
        v = float(F.cross_entropy(Xt[va] @ W + b, yt[va]))
        if best is None or v < best[0]:
            best = (v, W, b, wd)
    _, W, b, wd = best
    with torch.no_grad():
        lg = Xt[te] @ W + b
        bits = float(F.cross_entropy(lg, yt[te]) / LN2)
        acc = float((lg.argmax(-1) == yt[te]).float().mean())
        uni = float(-prior[yt[te]].mean() / LN2)
    return {"bits": round(bits, 4), "acc": round(acc, 4), "unigram_bits": round(uni, 4),
            "info_bits": round(uni - bits, 4), "wd": wd, "n_train": int(tr.sum()), "n_test": int(te.sum())}


def ngram_r2(logits: list, ids: list, ns=(1, 2, 3, 4, 6, 8)) -> dict:
    """Fraction of the raw-score variance explained by the exact n-byte context ending at the byte (group means;
    groups seen at least twice)."""
    out = {}
    allv = np.concatenate([l[1:] for l in logits])
    tot = allv.var()
    for n in ns:
        groups = collections.defaultdict(list)
        for lg, idd in zip(logits, ids):
            b = bytes(idd.astype(np.uint8).tolist())
            for t in range(1, len(idd)):
                groups[b[max(0, t - n + 1): t + 1]].append(lg[t])
        within, cnt = 0.0, 0
        for v in groups.values():
            if len(v) >= 2:
                v = np.asarray(v)
                within += ((v - v.mean()) ** 2).sum()
                cnt += len(v)
        sub = np.concatenate([np.asarray(v) for v in groups.values() if len(v) >= 2])
        out[f"n{n}"] = {"r2": round(float(1 - within / (cnt * sub.var() + 1e-12)), 4), "coverage": round(cnt / len(allv), 4)}
    out["total_var"] = float(tot)
    return out


def chunk_strings(cap) -> list:
    """(start, end) byte ranges (inclusive) of the chunks, by the router's semantics (BOS chunk excluded)."""
    sel = np.nonzero(cap["mask"])[0].tolist()
    T = len(cap["ids"])
    if cap["semantics"] == "end":
        ends = [s for s in sel if s >= 1]
        starts = [1] + [e + 1 for e in ends[:-1]]
        return [(a, b) for a, b in zip(starts, ends) if a <= b]
    starts = sorted(set([1] + [s for s in sel if s >= 1]))
    return [(a, b - 1) for a, b in zip(starts, starts[1:] + [T])]


def analyse(path: str, docs: list, attn_docs: int = 8, grad_attr: bool = True) -> dict:
    t0 = time.time()
    model, mcfg, meta = load_run_checkpoint(path)
    st = model.backbone
    sem = st.routing_module.boundary_semantics
    rep = {"checkpoint": meta["checkpoint"], "semantics": sem, "router": type(st.routing_module).__name__,
           "decider": type(st.decider).__name__ if st.decider is not None else "threshold",
           "chunk_vocab": st.chunk_vocab is not None, "ngram": model.ngram_embeddings is not None,
           "n_stages": len(st.stages()) - 1, "docs": len(docs)}
    caps = []
    abl = collections.defaultdict(list)
    for di, d in enumerate(docs):
        c = capture(model, d)
        c["semantics"] = sem
        c["doc"] = di
        T = len(c["ids"])
        mean_out = np.repeat(c["out"].mean(0, keepdims=True), T, 0)
        mean_res = np.repeat(c["residual"].mean(0, keepdims=True), T, 0)
        zero = np.zeros_like(c["out"])
        abl["full"].append(c["bits"])
        abl["hier_zero"].append(decode(model, c, hier=zero))
        abl["hier_mean"].append(decode(model, c, hier=mean_out))
        abl["resid_zero"].append(decode(model, c, resid=zero))
        abl["resid_mean"].append(decode(model, c, resid=mean_res))
        abl["both_mean"].append(decode(model, c, hier=mean_out, resid=mean_res))
        abl["lens_hier"].append(lens_bits(model, c["out"], c["ids"]))
        abl["lens_resid"].append(lens_bits(model, c["residual"], c["ids"]))
        abl["lens_sum"].append(lens_bits(model, c["out"] + c["residual"], c["ids"]))
        if grad_attr:
            ig = grad_attribution(model, c)
            abl["ig_hier"].append(ig[0])
            abl["ig_resid"].append(ig[1])
        caps.append(c)
    # target class = class of the predicted byte (t+1)
    tcls = np.concatenate([np.roll(c["cls"], -1) for c in caps])
    tpiw = np.concatenate([np.roll(c["piw"], -1) for c in caps])
    A = {k: np.concatenate(v) for k, v in abl.items()}
    valid = np.isfinite(A["full"])
    rep["bpb"] = round(float(A["full"][valid].mean()), 4)
    # ------------------------------------------------------------------ paths
    paths = {"rms_hier": float(np.mean(np.concatenate([c["out_rms"] for c in caps]))),
             "rms_resid": float(np.mean(np.concatenate([c["res_rms"] for c in caps]))),
             "cos_hier_resid": float(np.mean(np.concatenate([c["cos_out_res"] for c in caps]))),
             "bpb": {k: round(float(v[valid].mean()), 4) for k, v in A.items() if not k.startswith("ig_")}}
    by_cls = {}
    for ci, cn in enumerate(CLASSES):
        m = valid & (tcls == ci)
        if m.sum() < 20:
            continue
        by_cls[cn] = {"share": round(float(m.mean()), 4), **{k: round(float(v[m].mean()), 4) for k, v in A.items()}}
    paths["by_class"] = by_cls
    by_piw = {}
    for p in list(range(0, 8)) + [-1]:
        m = valid & (tpiw == p) if p < 7 else valid & (tpiw >= 7)
        if m.sum() < 20:
            continue
        key = "space_after_word" if p == -1 else (f"letter{p}" if p < 7 else "letter7+")
        by_piw[key] = {"share": round(float(m.mean()), 4),
                       **{k: round(float(A[k][m].mean()), 4) for k in ("full", "hier_zero", "hier_mean", "resid_zero",
                                                                        "resid_mean", "lens_hier", "lens_resid")}}
    paths["by_pos_in_word"] = by_piw
    rep["paths"] = paths
    # ------------------------------------------------------------------ router
    logits = [c["logit"] for c in caps]
    allc = np.concatenate([c["cls"] for c in caps])
    alll = np.concatenate(logits)
    allm = np.concatenate([c["mask"] for c in caps])
    allp = np.concatenate([c["p"] for c in caps])
    first = np.concatenate([np.arange(len(c["ids"])) == 0 for c in caps])
    keep = ~first
    if st.decider is not None:
        mu, sig, ztau = float(st.decider.mu), float(st.decider.sigma), float(st.decider.z_tau)
        z = (alll - mu) / sig - ztau
    else:
        mu, sig, ztau = 0.0, 1.0, 0.0
        z = alll
    rt = {"eval_bytes_per_chunk": round(float(keep.sum() / max(1, allm[keep].sum())), 3),
          "decider_mu": mu, "decider_sigma": sig, "decider_z_tau": ztau,
          "raw_logit_std": float(alll[keep].std()), "raw_logit_mean": float(alll[keep].mean()),
          "p_margin": float(np.abs(allp[keep] - 0.5).mean()),
          "p_mid_frac": float(((allp[keep] > 0.4) & (allp[keep] < 0.6)).mean())}
    cls_stats, hist = {}, {}
    edges = np.linspace(-8, 8, 65)
    for ci, cn in enumerate(CLASSES):
        m = keep & (allc == ci)
        if m.sum() < 10:
            continue
        cls_stats[cn] = {"share_bytes": round(float(m.mean()), 4),
                         "share_boundaries": round(float(allm[m].sum() / max(1, allm[keep].sum())), 4),
                         "boundary_rate": round(float(allm[m].mean()), 4),
                         "z_mean": round(float(z[m].mean()), 3), "z_std": round(float(z[m].std()), 3),
                         "raw_mean": round(float(alll[m].mean()), 3), "raw_std": round(float(alll[m].std()), 3)}
        hist[cn] = np.histogram(np.clip(z[m], -8, 8), bins=edges)[0].tolist()
    rt["by_byte_class"] = cls_stats
    rt["z_hist"] = {"edges": edges.tolist(), "counts": hist}
    rt["ngram_r2"] = ngram_r2(logits, [c["ids"] for c in caps])
    # is the router a word-end *predictor*? At letters, the next byte is unseen: compare the boundary score with
    # whether the next byte is a non-letter, and with the full model's own P(next byte is a non-letter)
    nxt = np.concatenate([np.roll(c["ids"], -1) for c in caps]).astype(np.int64)
    cur = np.concatenate([c["ids"] for c in caps]).astype(np.int64)
    is_alpha = lambda a: ((a >= 65) & (a <= 90)) | ((a >= 97) & (a <= 122))  # noqa: E731
    letter = keep & is_alpha(cur)
    y = ~is_alpha(nxt)
    pna = np.concatenate([c["p_next_nonalpha"] for c in caps])
    zz = z[letter]
    yy = y[letter]
    pos, neg = zz[yy], zz[~yy]
    sub = np.random.default_rng(0)
    pos_s = pos if len(pos) < 4000 else sub.choice(pos, 4000, replace=False)
    neg_s = neg if len(neg) < 4000 else sub.choice(neg, 4000, replace=False)
    auc = float((pos_s[:, None] > neg_s[None, :]).mean()) if len(pos_s) and len(neg_s) else None
    lp = np.log(np.clip(pna[letter], 1e-6, 1 - 1e-6) / np.clip(1 - pna[letter], 1e-6, 1))
    rho = float(np.corrcoef(np.argsort(np.argsort(zz)), np.argsort(np.argsort(lp)))[0, 1]) if len(zz) > 10 else None
    bm = allm & keep
    is_ws = (cur == 0x20) | (cur == 0x0A)
    rt["word_end_prediction"] = {
        "auc_z_for_next_nonletter_at_letters": auc,
        "spearman_z_vs_model_logodds_next_nonletter": rho,
        "precision_letter_boundaries": float(y[bm & letter].mean()) if (bm & letter).any() else None,
        "recall_word_ends_at_letters": float(allm[letter & y].mean()) if (letter & y).any() else None,
        "boundary_types": {
            "at_whitespace": float((bm & is_ws).sum() / max(1, bm.sum())),
            "letter_then_nonletter": float((bm & letter & y).sum() / max(1, bm.sum())),
            "letter_then_letter(intra_word)": float((bm & letter & ~y).sum() / max(1, bm.sum())),
            "other": float((bm & ~is_ws & ~letter).sum() / max(1, bm.sum())),
        },
    }
    # within the class "letters": does the same letter decide differently by context? std of z within (byte, class)
    rep["router"] = rt
    # ------------------------------------------------------------------ tokenizer
    chunks, words_seg = [], collections.defaultdict(set)
    n_bytes = 0
    on_ws = hit = n_ws = 0
    for c in caps:
        ids = c["ids"].astype(np.int64).tolist()
        b = bytes([x & 0xFF for x in ids])
        rng = chunk_strings(c)
        ends = set(e for _, e in rng)
        for a, e in rng:
            chunks.append(b[a: e + 1])
            n_bytes += e - a + 1
        # whitespace alignment under end semantics: a chunk end at a space/newline byte
        for e in ends:
            on_ws += ids[e] in (0x20, 0x0A)
        wsb = [t for t in range(1, len(ids) - 1) if ids[t] in (0x20, 0x0A)]
        n_ws += len(wsb)
        hit += sum(1 for t in wsb if t in ends)
        # word consistency: for each whitespace-delimited word (with its trailing space), its internal cut pattern
        t = 1
        while t < len(ids):
            u = t
            while u < len(ids) and ids[u] not in (0x20, 0x0A):
                u += 1
            w = b[t: u + 1]
            cut = tuple(k - t for k in range(t, min(u + 1, len(ids))) if k in ends)
            if len(w) > 1:
                words_seg[w].add(cut)
            t = u + 1
    cnt = collections.Counter(chunks)
    rec = [w for w, s in words_seg.items() if sum(1 for _ in [0]) and len(w) > 1]
    occ = collections.Counter()
    for c in caps:
        ids = c["ids"].astype(np.int64).tolist()
        b = bytes([x & 0xFF for x in ids])
        for w in b.split(b" "):
            occ[w + b" "] += 1
    frequent = [w for w in words_seg if occ.get(w, 0) >= 3]
    rep["tokenizer"] = {
        "chunks": len(chunks), "bytes_per_chunk": round(n_bytes / max(1, len(chunks)), 3),
        "types": len(cnt), "singleton_frac": round(sum(1 for v in cnt.values() if v == 1) / max(1, len(chunks)), 4),
        "ends_on_whitespace": round(on_ws / max(1, len(chunks)), 4),
        "whitespace_hit": round(hit / max(1, n_ws), 4),
        "word_consistency": round(sum(1 for w in frequent if len(words_seg[w]) == 1) / max(1, len(frequent)), 4),
        "top_chunks": [[k.decode("latin-1"), v] for k, v in cnt.most_common(25)],
    }
    del rec
    # ------------------------------------------------------------------ probes on chunk vectors
    rows = {"x_enc": [], "x": [], "z": []}
    y_id, y_len, y_first, y_last, y_next, y_prev, grp = [], [], [], [], [], [], []
    top = {k: i for i, (k, _) in enumerate(cnt.most_common(255))}
    for c in caps:
        ids = c["ids"].astype(np.int64)
        sel = c["sel_idx"]
        b = bytes([int(x) & 0xFF for x in ids])
        # map each selected (inner) position to its chunk byte range
        if sem == "end":
            prev_end = np.concatenate([[-1], sel[:-1]])
            ranges = [(pe + 1, e) for pe, e in zip(prev_end, sel)]
        else:
            nxt = np.concatenate([sel[1:], [len(ids)]])
            ranges = [(s, n - 1) for s, n in zip(sel, nxt)]
        for j, (a, e) in enumerate(ranges):
            if j == 0 or e + 1 >= len(ids) or a < 1:
                continue
            s_ = b[a: e + 1]
            y_id.append(top.get(s_, 255))
            y_len.append(min(e - a + 1, 16) - 1)
            y_first.append(ids[a])
            y_last.append(ids[e])
            y_next.append(ids[e + 1] if sem == "end" else ids[min(e + 1, len(ids) - 1)])
            y_prev.append(ids[a - 1])
            grp.append(c["doc"])
            for k in rows:
                rows[k].append(c[k][j])
    pr = {}
    grp = np.array(grp)
    for k in rows:
        X = np.stack(rows[k])
        pr[k] = {"eff_rank": eff_rank(X)}
        for tname, y, n in (("identity_top255", y_id, 256), ("length", y_len, 16), ("first_byte", y_first, 256),
                            ("last_byte", y_last, 256), ("next_first_byte", y_next, 256),
                            ("prev_last_byte", y_prev, 256)):
            pr[k][tname] = probe(X, np.array(y), grp, n)
    pr["n_chunks"] = int(len(grp))
    pr["identity_in_top255_frac"] = round(float(np.mean(np.array(y_id) < 255)), 4)
    # effective ranks of byte-level signals
    pr["byte_eff_rank"] = {k: eff_rank(np.concatenate([c[k] for c in caps])[::3])
                           for k in ("h_enc", "residual", "out", "h_dec")}
    if any("vocab_emb" in c for c in caps):
        pr["vocab_emb_rms"] = float(np.sqrt(np.mean(np.concatenate([c["vocab_emb"] for c in caps]) ** 2)))
        pr["x_enc_rms"] = float(np.sqrt(np.mean(np.concatenate([c["x_enc"] for c in caps]) ** 2)))
    rep["probes"] = pr
    # ------------------------------------------------------------------ attention
    if attn_docs > 0:
        profs = [attention_profile(model, d) for d in docs[:attn_docs]]
        rep["attention"] = {L: {k: round(float(np.mean([p[L][k] for p in profs])), 4) for k in profs[0][L]}
                            for L in profs[0]}
    rep["seconds"] = round(time.time() - t0, 1)
    return rep


def grad_attribution(model, c) -> tuple:
    """Per byte: |sum_d grad_d * x_d| of the loss w.r.t. the hierarchy and bypass decoder inputs (first-order share
    of each path in the byte's loss)."""
    st = model.backbone
    out = torch.from_numpy(c["out"]).requires_grad_(True)
    res = torch.from_numpy(c["residual"]).requires_grad_(True)
    ctx = c["_ctx"]
    with torch.enable_grad():
        hd = st.decoder((out + res).float(), ctx)
        logits = model.lm_head(hd[:, : model.lm_head.weight.shape[1]]).float()
        ids = torch.from_numpy(c["ids"].astype(np.int64))
        nll = F.cross_entropy(logits[:-1], ids[1:], reduction="none")
        # per-byte attribution needs one backward per byte; use the summed loss and attribute per input position,
        # which, for a causal decoder, is the contribution of position t's input to all later losses
        g_out, g_res = torch.autograd.grad(nll.sum(), (out, res))
    a = (g_out * out).sum(-1).abs().detach().numpy()
    b = (g_res * res).sum(-1).abs().detach().numpy()
    return a, b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--docs", type=int, default=48)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--attn-docs", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    docs = load_docs(a.docs, a.max_bytes)
    rep = analyse(a.ckpt, docs, attn_docs=a.attn_docs)
    name = a.out or (a.ckpt.replace("runs/gcp/", "").replace("/ckpt/", "_").replace("/", "__") + ".json")
    p = dump(rep, f"ckpt/{name}") if (DATA / "ckpt").exists() else None
    if p is None:
        (DATA / "ckpt").mkdir(parents=True, exist_ok=True)
        p = dump(rep, f"ckpt/{name}")
    print(p, rep["bpb"], rep["seconds"], "s")


if __name__ == "__main__":
    main()
