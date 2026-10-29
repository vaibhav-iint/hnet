"""Linear probes on the innermost main network of a trained H-Net: does it compute next-chunk information that the
decoder fails to use (a decoder bottleneck), or does it not compute it at all (a training-signal problem)?

    python -m hnet.analysis.probe_main RUN_DIR [--docs 192] [--max-bytes 2048]

At every innermost position (one per stage-(S-1) chunk) we take the main network's input x (the chunk vector built
by the enclosing encoder) and its output z, and fit multinomial logistic-regression probes for the byte that follows
the chunk's anchor position in the text (for "end" routers: the first byte of the next chunk; for "start" routers: the
second byte of the current chunk). Probes are fit on 70% of positions and scored (bits) on the rest. The model's
own loss on those same bytes is reported for reference.
"""

from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

from hnet.analysis.diagnose import _pack, load_docs
from hnet.eval.run import load_run_checkpoint

LN2 = math.log(2.0)


@torch.no_grad()
def collect(model, docs):
    inner = model.backbone
    while not inner.is_innermost:
        inner = inner.main_network
    cap = {}

    def hook(mod, inputs, output):
        cap["x"] = inputs[0].detach().float().reshape(-1, inputs[0].shape[-1])
        cap["z"] = output.detach().float().reshape(-1, output.shape[-1])

    hd = inner.main_network.register_forward_hook(hook)
    xs, zs, ys, model_bits = [], [], [], []
    try:
        for d in docs:
            ids, cu, max_len = _pack([d], "cpu")
            out = model(ids, cu_seqlens=cu, max_seqlen=max_len, return_logits=True)
            T = ids.numel()
            pos = torch.arange(T)
            for r in out.bpred:
                pos = pos[r.boundary_mask.reshape(-1).bool()]
            # target: the byte right after each innermost position's anchor byte
            ok = pos + 1 < T
            tgt = ids[(pos + 1).clamp(max=T - 1)]
            nll = F.cross_entropy(out.logits[pos].float(), tgt, reduction="none") / LN2
            xs.append(cap["x"][ok])
            zs.append(cap["z"][ok])
            ys.append(tgt[ok])
            model_bits.append(nll[ok])
    finally:
        hd.remove()
    return torch.cat(xs), torch.cat(zs), torch.cat(ys), torch.cat(model_bits)


def fit_probe(X: torch.Tensor, y: torch.Tensor, train: torch.Tensor, steps: int = 400, wd: float = 1e-3) -> float:
    """Multinomial logistic regression (full-batch L-BFGS) on standardized features; held-out bits per target."""
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-5
    Xs = (X - mu) / sd
    W = torch.zeros(X.shape[1], 256, requires_grad=True)
    b = torch.zeros(256, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=steps, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xs[train] @ W + b, y[train]) + wd * W.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return float(F.cross_entropy(Xs[~train] @ W + b, y[~train]) / LN2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--docs", type=int, default=192)
    ap.add_argument("--max-bytes", type=int, default=2048)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    torch.manual_seed(0)
    model, mcfg, meta = load_run_checkpoint(a.run)
    X, Z, y, mb = collect(model, load_docs(a.docs, a.max_bytes))
    g = torch.Generator().manual_seed(0)
    train = torch.rand(len(y), generator=g) < 0.7
    counts = torch.bincount(y[train], minlength=256).float() + 0.5
    unigram = float(-(counts / counts.sum()).log()[y[~train]].mean() / LN2)
    rep = {
        "checkpoint": meta["checkpoint"], "positions": int(len(y)),
        "unigram_bits": round(unigram, 3),
        "probe_main_input_bits": round(fit_probe(X, y, train), 3),
        "probe_main_output_bits": round(fit_probe(Z, y, train), 3),
        "probe_input_and_output_bits": round(fit_probe(torch.cat([X, Z], 1), y, train), 3),
        "model_bits_same_bytes": round(float(mb[~train].mean()), 3),
    }
    txt = json.dumps(rep, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
