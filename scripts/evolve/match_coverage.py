#!/usr/bin/env python
"""Offline coverage of the in-context match index, byte-level (MatchModel) vs token-level (TokenMatchModel), on the
same mix_v1 validation documents with the training windowing (bytes: 8512-byte windows, tokens: 2048-token windows;
a segment starts at every window start and every document start, so matches never cross either).

    python scripts/evolve/match_coverage.py [--root data] [--sets mix,en,code,heldout,math,ml] [--md out.md]

Per set: for the byte index (orders 4/8/16/32) the share of target bytes with an order>=8 match and its accuracy
(the numbers quoted by g1-wild-01); for the token index (orders 1/2/4/8) per order the share of target tokens with a
match and the accuracy of the proposed token; and, mapped onto the same text bytes, how much of the text the two
indices cover: the share of bytes inside target tokens with a (correct) order>=2 token match, and the overlap
P(token index correct | byte index correct) and P(byte index correct | token index correct). The "longest
match" candidate of a family of orders is the candidate of its highest order with a match (what a PPM model uses).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

from hnet.config import BOS_ID, EOS_ID
from hnet.data import shards as S
from hnet.data.tokenizer import dataset_tokenizer
from hnet.model.ctx import SeqCtx
from hnet.model.match import match_candidates

BYTE_ORDERS, TOK_ORDERS = [4, 8, 16, 32], [1, 2, 4, 8]
BYTE_WIN, TOK_WIN = 8512, 2048
STEMS = {"mix": "mix_v1_val", "en": "mix_v1_val-en", "code": "mix_v1_val-code", "heldout": "mix_v1_val-heldout",
         "math": "mix_v1_val-math", "ml": "mix_v1_val-ml"}


def _cands(x: np.ndarray, starts: np.ndarray, orders, none_id):
    T = len(x)
    cu = np.unique(np.concatenate([[0], starts[(starts > 0) & (starts < T)], [T]])).astype(np.int32)
    ctx = SeqCtx.from_cu_seqlens(torch.from_numpy(cu), None, T)
    return match_candidates(torch.from_numpy(x.astype(np.int64)), ctx, orders, none_id=none_id).numpy()


def _longest(C: np.ndarray, none_id: int, rows) -> np.ndarray:
    out = np.full(C.shape[1], none_id, dtype=np.int64)
    for r in rows:  # ascending order: a higher order overrides
        out = np.where(C[r] != none_id, C[r], out)
    return out


def measure(root: str, stem: str, tok) -> dict:
    by = np.asarray(S.read_byte_shard(f"{root}/mix_v1/bytes/{stem}_000000.bytes.bin"))
    tk = np.asarray(S.read_token_shard(f"{root}/mix_v1/{stem}_000000.bin")).astype(np.int64)
    idx = np.asarray(S.read_doc_index(f"{root}/mix_v1/bytes/{stem}_000000.idx.npy"))
    # ---- byte index: segments at window starts, BOS and after EOS; targets = text bytes
    b_starts = np.concatenate([np.arange(0, len(by), BYTE_WIN), np.flatnonzero(by == BOS_ID),
                               np.flatnonzero(by == EOS_ID) + 1])
    CB = _cands(by, b_starts, BYTE_ORDERS, 256)
    pred_b = _longest(CB, 256, [1, 2, 3])  # orders >= 8
    nxt = np.concatenate([by[1:], [EOS_ID]])
    in_win = (np.arange(len(by)) + 1) % BYTE_WIN != 0  # the last position of a window: target in the next window
    tgt_b = (nxt != BOS_ID) & (nxt != EOS_ID) & (by != EOS_ID)  # cross_window_label: the last position keeps it
    hit_b, ok_b = (pred_b != 256) & tgt_b, (pred_b == nxt) & tgt_b
    # per target byte (position p predicted from p - 1), in byte-stream coordinates
    B_hit = np.zeros(len(by), bool)
    B_ok = np.zeros(len(by), bool)
    B_hit[1:], B_ok[1:] = hit_b[:-1], ok_b[:-1]
    # ---- token index: segments at window starts and at every document's start token; targets = text tokens
    doc_tok = idx[:, S.TOK_OFF][(idx[:, S.FLAGS] & S.HAS_BOS) > 0]
    CT = _cands(tk, np.concatenate([np.arange(0, len(tk), TOK_WIN), doc_tok]), TOK_ORDERS, tok.vocab_size)
    nxt_t = np.concatenate([tk[1:], [tok.eot_id]])
    tgt_t = nxt_t != tok.eot_id
    res = {"bytes": int(tgt_b.sum()), "tokens": int(tgt_t.sum()), "byte_ord8_cov": hit_b.sum() / tgt_b.sum(),
           "byte_ord8_acc": ok_b.sum() / max(hit_b.sum(), 1)}
    for r, n in enumerate(BYTE_ORDERS):
        h = (CB[r] != 256) & tgt_b
        res[f"byte_o{n}_cov"], res[f"byte_o{n}_acc"] = h.sum() / tgt_b.sum(), ((CB[r] == nxt) & h).sum() / max(h.sum(), 1)
    for r, n in enumerate(TOK_ORDERS):
        h = (CT[r] != tok.vocab_size) & tgt_t
        res[f"tok_o{n}_cov"], res[f"tok_o{n}_acc"] = h.sum() / tgt_t.sum(), ((CT[r] == nxt_t) & h).sum() / max(h.sum(), 1)
    # ---- the same text: map every target token onto its text bytes in the byte stream
    T_hit = {k: np.zeros(len(by), bool) for k in (1, 2)}
    T_ok = {k: np.zeros(len(by), bool) for k in (1, 2)}
    pred_t = {1: _longest(CT, tok.vocab_size, [0, 1, 2, 3]), 2: _longest(CT, tok.vocab_size, [1, 2, 3])}
    lens = tok.byte_lengths(tk)
    tok_units = {"n": 0, "byte_all": 0, 1: 0, 2: 0, "both1": 0, "both2": 0}  # target tokens (with bytes), per unit
    for row in idx:
        nb, nt = int(row[S.N_BYTES]), int(row[S.N_TOKENS])
        if nt == 0:
            continue
        hb = int(row[S.FLAGS] & S.HAS_BOS)
        t0, b0 = int(row[S.TOK_OFF]) + hb, int(row[S.BYTE_OFF]) + hb
        L = lens[t0:t0 + nt]
        assert int(L.sum()) == nb, (stem, row)
        start = b0 + np.concatenate([[0], np.cumsum(L)[:-1]])
        for k in (1, 2):  # token t0 + j is the target of position t0 + j - 1 (the first one has no predecessor)
            pj = pred_t[k][t0 - 1 + np.arange(1, nt)] if nt > 1 else np.zeros(0, np.int64)
            hit = pj != tok.vocab_size
            ok = pj == tk[t0 + 1:t0 + nt]
            T_hit[k][b0:b0 + nb] = np.repeat(np.concatenate([[False], hit]), L)
            T_ok[k][b0:b0 + nb] = np.repeat(np.concatenate([[False], ok]), L)
        if nt > 1:  # whole-token units: is every byte of target token j correctly proposed by the byte index?
            starts_j, L_j = start[1:] - b0, L[1:]
            keep = L_j > 0
            all_ok = np.minimum.reduceat(B_ok[b0:b0 + nb].astype(np.int8), starts_j[keep]).astype(bool)
            tok_units["n"] += int(keep.sum())
            tok_units["byte_all"] += int(all_ok.sum())
            for k in (1, 2):
                okk = (pred_t[k][t0:t0 + nt - 1] == tk[t0 + 1:t0 + nt])[keep]
                tok_units[k] += int(okk.sum())
                tok_units[f"both{k}"] += int((okk & all_ok).sum())
    text = np.zeros(len(by), bool)
    text[1:] = tgt_b[:-1]
    nb = text.sum()
    for k in (1, 2):
        res[f"tokbytes_o{k}_cov"] = (T_hit[k] & text).sum() / nb
        res[f"tokbytes_o{k}_ok"] = (T_ok[k] & text).sum() / nb
        res[f"P(tok{k}_ok|byte_ok)"] = (T_ok[k] & B_ok & text).sum() / max((B_ok & text).sum(), 1)
        res[f"P(byte_ok|tok{k}_ok)"] = (T_ok[k] & B_ok & text).sum() / max((T_ok[k] & text).sum(), 1)
        res[f"byte_ok_tok{k}_nohit"] = (B_ok & ~T_hit[k] & text).sum() / nb
    res["byte_ok"] = (B_ok & text).sum() / nb
    n = max(tok_units["n"], 1)
    res["unit_byte_all"] = tok_units["byte_all"] / n
    for k in (1, 2):
        res[f"unit_tok{k}"] = tok_units[k] / n
        res[f"unit_P(tok{k}|byte_all)"] = tok_units[f"both{k}"] / max(tok_units["byte_all"], 1)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data")
    ap.add_argument("--sets", default="mix,en,code,heldout,math,ml")
    ap.add_argument("--md", default=None)
    a = ap.parse_args(argv)
    tok = dataset_tokenizer("mix_v1", a.root)
    rows = {s: measure(a.root, STEMS[s], tok) for s in a.sets.split(",")}
    pct = lambda v: f"{100 * v:.1f}"  # noqa: E731
    lines = ["| set | bytes | tokens | byte o>=8 cov / acc | " + " | ".join(f"tok o{n} cov / acc" for n in TOK_ORDERS)
             + " | text bytes in tokens with o>=2 match (correct) | byte o>=8 correct (of text) | "
               "P(tok o>=2 ok \\| byte ok) | P(byte ok \\| tok o>=2 ok) | byte ok, no tok o>=2 match |",
             "|" + "---|" * (9 + len(TOK_ORDERS))]
    for s, r in rows.items():
        lines.append(f"| {s} | {r['bytes'] / 1e6:.2f}M | {r['tokens'] / 1e6:.2f}M | {pct(r['byte_ord8_cov'])} / "
                     f"{pct(r['byte_ord8_acc'])} | " + " | ".join(f"{pct(r[f'tok_o{n}_cov'])} / {pct(r[f'tok_o{n}_acc'])}"
                                                                for n in TOK_ORDERS)
                     + f" | {pct(r['tokbytes_o2_cov'])} ({pct(r['tokbytes_o2_ok'])}) | {pct(r['byte_ok'])} | "
                       f"{pct(r['P(tok2_ok|byte_ok)'])} | {pct(r['P(byte_ok|tok2_ok)'])} | {pct(r['byte_ok_tok2_nohit'])} |")
    lines.append("")
    lines.append("Byte index per order (cov / acc): " + "; ".join(
        f"{s}: " + ", ".join(f"o{n} {pct(r[f'byte_o{n}_cov'])}/{pct(r[f'byte_o{n}_acc'])}" for n in BYTE_ORDERS)
        for s, r in rows.items()))
    lines.append("")
    lines.append("Token order>=1 mapped to text bytes (cov (correct)): " + "; ".join(
        f"{s}: {pct(r['tokbytes_o1_cov'])} ({pct(r['tokbytes_o1_ok'])}), P(tok1 ok | byte ok) "
        f"{pct(r['P(tok1_ok|byte_ok)'])}" for s, r in rows.items()))
    lines.append("")
    lines.append("Whole-token units (share of target tokens): every byte correctly proposed by the byte index (o>=8) "
                 "vs the token correctly proposed by the token index (longest match o>=2 / o>=1), and "
                 "P(token index correct | byte index copies the whole token): " + "; ".join(
                     f"{s}: {pct(r['unit_byte_all'])} vs {pct(r['unit_tok2'])} / {pct(r['unit_tok1'])}, "
                     f"P {pct(r['unit_P(tok2|byte_all)'])} / {pct(r['unit_P(tok1|byte_all)'])}" for s, r in rows.items()))
    out = "\n".join(lines)
    print(out)
    if a.md:
        with open(a.md, "w") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    sys.exit(main())
