"""Offline probe on a trained L03 H-Net checkpoint for the generation-3 output-memory candidates (g3-mem-01/02):

1. Reach of a top-k output head: how often the target byte is among the k bytes the trained model's own logits rank
   highest, and what share of the model's bits sit on positions where it is not (a top-k rescoring head cannot help
   there; a full-vocabulary head can, in principle). Split by chunk-initial vs non-initial targets.
2. Which FLOP-free input table is cheaper to lose: simulate a smaller table on the trained weights without retraining
   (halving a table = tying its rows in random pairs to their mean, which is what a table with half the buckets
   would hold given the same content; dropping an n-gram order = replacing its rows by their usage-weighted mean) and
   measure the val bpb increase per million parameters freed.
3. Key statistics on the evaluated bytes: distinct trie keys (chunk prefix + next byte, the model's own boundaries)
   and distinct byte n-grams per order, with the share of positions a table of R rows could hold collision-free
   (the R most frequent keys).

    OMP_NUM_THREADS=2 PYTHONPATH=. python scripts/evolve/trie_budget_probe.py --windows 48
"""

from __future__ import annotations

import argparse
import collections
import json
import math

import numpy as np
import torch

from hnet.config import HNetConfig, load_yaml
from hnet.model import build_model

RUN = "/home/user/hnet-smollm3/runs/gcp/hnet-ladder-1e16/search/ladder/1e16/L03/hnet_ws_s0"
VAL = "data/mix_v1/bytes/mix_v1_val_000000.bytes.bin"
LN2 = math.log(2)


def load(run):
    d = dict(load_yaml(f"{run}/config.yaml")["model"])
    d.pop("type", None)
    m = build_model(HNetConfig.from_dict(d))
    sd = torch.load(f"{run}/ckpt/step_0000908/model.pt", map_location="cpu", weights_only=False)
    m.load_state_dict(sd)
    return m.eval()


def windows(n, T, stride):
    raw = np.fromfile(VAL, dtype=np.uint8)
    for i in range(n):
        s = 500_000 + i * stride
        yield torch.from_numpy(raw[s:s + T].astype(np.int64))


@torch.no_grad()
def evaluate(m, wins, ks=(), stats=False):
    """bpb over the windows; with ks: top-k reach; with stats: trie keys and n-gram counts."""
    nll = n = 0.0
    reach = {k: collections.Counter() for k in ks}
    keys = collections.Counter()
    grams = {g: collections.Counter() for g in range(3, 9)}
    for ids in wins:
        lab = torch.roll(ids, -1)
        lab[-1] = -100
        out = m(ids[None], labels=lab[None])
        logits = out.logits[0].float()
        lp = torch.log_softmax(logits, -1)
        ok = lab != -100
        tl = -lp[torch.arange(len(ids)), lab.clamp(min=0)]
        nll += float(tl[ok].sum())
        n += float(ok.sum())
        mask = out.bpred[0].boundary_mask
        if ks:
            init = mask  # t ends a chunk: the target is the first byte of the next chunk
            rank = (logits > logits[torch.arange(len(ids)), lab.clamp(min=0)][:, None]).sum(-1)
            for k in ks:
                hit = rank < k
                for cls, sel in (("init", init & ok), ("inner", ~init & ok)):
                    reach[k][cls + "_n"] += int(sel.sum())
                    reach[k][cls + "_miss"] += int((sel & ~hit).sum())
                    reach[k][cls + "_bits"] += float(tl[sel].sum()) / LN2
                    reach[k][cls + "_bits_miss"] += float(tl[sel & ~hit].sum()) / LN2
        if stats:
            b = bytes(ids.tolist())
            last = -1
            for t in range(len(ids) - 1):
                if mask[t]:
                    last = t
                    pre = b""
                else:
                    pre = b[max(last + 1, t - 15):t + 1]
                keys[pre + b[t + 1:t + 2]] += 1
                for g in grams:
                    if t + 1 >= g:
                        grams[g][b[t + 1 - g:t + 1]] += 1
    res = {"bpb": nll / n / LN2}
    if ks:
        res["reach"] = {}
        for k in ks:
            r = reach[k]
            tot_bits = r["init_bits"] + r["inner_bits"]
            res["reach"][k] = {
                "coverage": 1 - (r["init_miss"] + r["inner_miss"]) / (r["init_n"] + r["inner_n"]),
                "coverage_init": 1 - r["init_miss"] / r["init_n"], "coverage_inner": 1 - r["inner_miss"] / r["inner_n"],
                "bits_share_outside_topk": (r["init_bits_miss"] + r["inner_bits_miss"]) / tot_bits,
                "bits_share_outside_topk_init": r["init_bits_miss"] / tot_bits,
                "bits_share_outside_topk_inner": r["inner_bits_miss"] / tot_bits,
                "bpb_outside_topk": (r["init_bits_miss"] + r["inner_bits_miss"]) / n}
    if stats:
        def cov(c, R):
            tot = sum(c.values())
            return sum(v for _, v in c.most_common(R)) / tot

        res["trie_keys"] = {"distinct": len(keys), "positions": sum(keys.values()),
                            **{f"top{R}_share": cov(keys, R) for R in (20000, 40000, 77383)}}
        res["ngrams"] = {g: {"distinct": len(c), "top4817_share": cov(c, 4817), "top2408_share": cov(c, 2408)}
                         for g, c in grams.items()}
    return res


def pair_tie(w: torch.Tensor):
    """Rows b and b + B//2 replaced by their mean (a table with half the buckets, same content)."""
    B = w.shape[0]
    h = B // 2
    m = 0.5 * (w[:h] + w[h:2 * h])
    w[:h] = m
    w[h:2 * h] = m


def usage_mean(m, wins, order_idx):
    """Usage-weighted mean row of n-gram table ``order_idx`` over the windows."""
    ng = m.ngram_embeddings
    from hnet.model.ctx import SeqCtx

    acc, cnt = 0.0, 0
    for ids in wins:
        ctx = SeqCtx.from_cu_seqlens(torch.tensor([0, len(ids)], dtype=torch.int32))
        H, V = ng.multi_hashes(ids, ctx.position_ids, ng.ns, [n * 7919 + 1 for n in ng.ns])
        idx = torch.remainder(H[order_idx], ng.buckets[order_idx])[V[order_idx]]
        acc = acc + ng.tables[order_idx].weight[idx].sum(0)
        cnt += idx.numel()
    return acc / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=48)
    ap.add_argument("--bytes", type=int, default=8512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    wins = list(windows(args.windows, args.bytes, 104_729))
    m = load(RUN)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    report = {"checkpoint": RUN, "bytes": args.windows * args.bytes}
    base = evaluate(m, wins, ks=(8, 16, 32, 64), stats=True)
    report["base"] = base
    print(json.dumps(base, indent=1))
    ng = m.ngram_embeddings
    cv = m.backbone.chunk_vocab
    d0 = ng.tables[0].embedding_dim
    arms = {}

    def arm(name, fn, params_freed):
        m.load_state_dict(sd)
        with torch.no_grad():
            fn()
        r = evaluate(m, wins)["bpb"]
        arms[name] = {"bpb": r, "delta": r - base["bpb"], "params_freed_M": params_freed / 1e6,
                      "delta_per_M": (r - base["bpb"]) / (params_freed / 1e6)}
        print(name, arms[name], flush=True)

    arm("chunk_vocab_halved", lambda: [pair_tie(t.weight.data) for t in cv.tables],
        sum(t.weight.shape[0] // 2 * t.weight.shape[1] for t in cv.tables))
    arm("ngram_all_orders_halved", lambda: [pair_tie(t.weight.data) for t in ng.tables],
        sum(t.weight.shape[0] // 2 * d0 for t in ng.tables))
    means = {i: usage_mean(m, wins[:8], i) for i in range(len(ng.ns))}

    def drop(orders):
        for i, n in enumerate(ng.ns):
            if n in orders:
                ng.tables[i].weight.data[:] = means[i]

    for orders in ([7, 8], [6, 7, 8], [5, 6, 7, 8], [3]):
        arm(f"ngram_drop_{''.join(map(str, orders))}", lambda o=orders: drop(o),
            sum(ng.buckets[i] * d0 for i, n in enumerate(ng.ns) if n in orders))
    report["ablations"] = arms
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
