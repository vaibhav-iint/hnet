"""usage: PYTHONPATH=. python logbook/evolve/candidates/g5-compute-01/probe.py CKPT_DIR
g5-compute probe, part 4: where does the main network's (deep, chunk-rate) compute pay? Mean-ablate the dechunked
main output entering the stage-0 decoder (replace it by its mean over the document; keeps the offset, removes the
information) and measure the per-byte CE increase by offset in the chunk and by the model's own entropy."""
import math
import os
import sys

import numpy as np
import torch

from hnet.config import BOS_ID
from hnet.data.shards import read_byte_shard
from hnet.eval.run import load_run_checkpoint

torch.set_num_threads(4)
model, mcfg, meta = load_run_checkpoint(sys.argv[1])
NB = {"en": 140_000, "code": 40_000, "math": 10_000, "ml": 10_000}
st0 = model.backbone
orig = st0.dechunk_layer
mode = {"abl": False}


class Abl:
    def __call__(self, *a, **k):
        out = orig(*a, **k)
        if mode["abl"]:
            out = out.mean(0, keepdim=True).expand_as(out)
        return out

    def __getattr__(self, n):
        return getattr(orig, n)


st0.dechunk_layer = Abl()
rows = {k: [] for k in ["ce", "ce_abl", "ent", "off", "src"]}
for si, (src, nb) in enumerate(NB.items()):
    p = f"data/mix_v1/bytes/mix_v1_val-{src}_000000.bytes.bin"
    data = np.asarray(read_byte_shard(p))
    idx = np.load(p.replace(".bytes.bin", ".idx.npy"))
    order = np.random.default_rng(0).permutation(len(idx))
    got = 0
    for i in order:
        off, n = int(idx[i, 0]), int(idx[i, 1])
        if n < 512:
            continue
        ids = torch.from_numpy(data[off: off + min(n, 4096)].astype(np.int64))
        res = {}
        for a in (False, True):
            mode["abl"] = a
            with torch.no_grad():
                o = model(ids, return_logits=True)
            lp = torch.log_softmax(o.logits.float(), -1)[:-1]
            res[a] = (-lp.gather(1, ids[1:, None]).squeeze(1) / math.log(2), lp, o.bpred[0].boundary_mask.reshape(-1))
        ce, lp, b = res[False]
        ar = torch.arange(len(b))
        offs = ar - torch.cummax(torch.where(b, ar, torch.zeros_like(ar)), 0).values
        rows["ce"].append(ce)
        rows["ce_abl"].append(res[True][0])
        rows["ent"].append(-(lp.exp() * lp).sum(-1) / math.log(2))
        rows["off"].append(offs[:-1])
        rows["src"].append(torch.full_like(ce, si))
        got += len(ids)
        if got >= nb:
            break
R = {k: torch.cat(v) for k, v in rows.items()}
ce, ca, off, ent = R["ce"], R["ce_abl"], R["off"], R["ent"]
d = ca - ce
print(f"bpb {ce.mean():.4f}  mean-ablated main {ca.mean():.4f}  hierarchy information {d.mean():.4f} bits/byte")
for k in range(9):
    m = off == k if k < 8 else off >= 8
    print(f"  off {k}{'+' if k == 8 else ' '}: pos {m.float().mean():.3f}  bits {ce[m].mean():.3f}  "
          f"main-info {d[m].mean():.3f} bits/byte  share of main-info {d[m].sum() / d.sum():.3f}  "
          f"share of bits {ce[m].sum() / ce.sum():.3f}")
o = torch.argsort(ent, descending=True)
for q in (0.1, 0.25, 0.5):
    k = int(q * len(o))
    print(f"  top {int(q * 100)}% by entropy: share of bits {ce[o[:k]].sum() / ce.sum():.3f}  "
          f"share of main-info {d[o[:k]].sum() / d.sum():.3f}")
# chunk regions: bytes t with plug-back chunk c; off 0 = the chunk-end byte (predicts the next chunk's first byte)
reg = torch.cumsum((off == 0).long(), 0) - 1
nR = int(reg.max()) + 1
z = torch.zeros_like(ce)
first = torch.zeros(nR).index_add_(0, reg, torch.where(off == 0, ce, z))  # first-byte surprise
rest = torch.zeros(nR).index_add_(0, reg, torch.where(off >= 1, ce, z))
rest_main = torch.zeros(nR).index_add_(0, reg, torch.where(off >= 1, d, z))
nrest = torch.zeros(nR).index_add_(0, reg, (off >= 1).float())
print(f"\nchunk regions {nR}: bits after the first byte {rest.sum() / ce.sum():.3f} of all; "
      f"corr(first-byte surprise, rest bits) {torch.corrcoef(torch.stack([first, rest]))[0, 1]:.3f}")
o = torch.argsort(first, descending=True)
for q in (0.25, 0.5):
    k = int(q * nR)
    print(f"  top {int(q * 100)}% chunks by first-byte surprise: share of off>=1 bits "
          f"{rest[o[:k]].sum() / rest.sum():.3f}  of off>=1 main-info {rest_main[o[:k]].sum() / rest_main.sum():.3f}"
          f"  of off>=1 bytes {nrest[o[:k]].sum() / nrest.sum():.3f}")
