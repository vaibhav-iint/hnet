"""Is the innermost main network untrained (near its random init) or trained toward something useless?

    python -m hnet.analysis.main_signal RUN_DIR [--docs 48]

For a trained checkpoint and a freshly initialized copy (same seed/config), on identical held-out documents:
1. weight drift: relative change ||W_trained - W_init|| / ||W_init|| per parameter group (main network, stage-1
   encoder/decoder, stage-0 encoder/decoder, router).
2. output similarity: cosine between the trained main network's output z and (a) its input x, (b) the output of
   the randomly initialized main network given the same input: "is z ~ x + small noise?".
3. gradient decomposition at the main network: ||dL_ce/dtheta_main|| vs ||dL_aux/dtheta_main|| on a batch, so we
   see whether the main network's small gradient is dominated by the LM loss or by the ratio loss through STE/EMA.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from hnet.analysis.diagnose import _pack, load_docs
from hnet.eval.run import load_run_checkpoint
from hnet.model import build_model


def groups(model):
    g = {"main": [], "stage1_encdec": [], "stage0_encdec": [], "router": [], "residual_proj": []}
    for n, p in model.named_parameters():
        if "routing_module" in n:
            g["router"].append(p)
        elif "residual_proj" in n:
            g["residual_proj"].append(p)
        elif n.startswith("backbone.main_network.main_network"):
            g["main"].append(p)
        elif n.startswith("backbone.main_network"):
            g["stage1_encdec"].append(p)
        elif n.startswith("backbone."):
            g["stage0_encdec"].append(p)
    return g


def drift(trained, init):
    out = {}
    gt, gi = groups(trained), groups(init)
    for k in gt:
        num = sum((a.detach() - b.detach()).pow(2).sum() for a, b in zip(gt[k], gi[k]))
        den = sum(b.detach().pow(2).sum() for b in gi[k]) + 1e-12
        out[k] = round(float((num / den).sqrt()), 4)
    return out


def innermost(model):
    st = model.backbone
    while not st.is_innermost:
        st = st.main_network
    return st.main_network


@torch.no_grad()
def output_similarity(trained, init, docs):
    cap = {}
    inner_t = innermost(trained)
    inner_i = innermost(init)
    h = inner_t.register_forward_hook(lambda m, i, o: cap.update(x=i[0].detach().float(), ctx=i[1],
                                                                 z=o.detach().float()))
    cos_xz, cos_init, rel = [], [], []
    try:
        for d in docs:
            ids, cu, ml = _pack([d], "cpu")
            trained(ids, cu_seqlens=cu, max_seqlen=ml, return_logits=False)
            x, z, ctx = cap["x"].reshape(-1, cap["x"].shape[-1]), cap["z"].reshape(-1, cap["z"].shape[-1]), cap["ctx"]
            z0 = inner_i(cap["x"], ctx).float().reshape(-1, x.shape[-1])
            cos_xz.append(F.cosine_similarity(x, z, dim=-1).mean())
            cos_init.append(F.cosine_similarity(z0, z, dim=-1).mean())
            rel.append(((z - x).norm(dim=-1) / (x.norm(dim=-1) + 1e-6)).mean())
    finally:
        h.remove()
    return {"cos(x, z)": round(float(torch.stack(cos_xz).mean()), 4),
            "cos(z_init, z)": round(float(torch.stack(cos_init).mean()), 4),
            "||z - x|| / ||x||": round(float(torch.stack(rel).mean()), 4)}


def grad_decomposition(model, docs):
    model.train()
    ids, cu, ml = _pack(docs[:8], "cpu")
    labels = torch.roll(ids, -1)
    labels[(cu[1:] - 1).long()] = -100
    out = model(ids, cu_seqlens=cu, max_seqlen=ml, labels=labels)
    g = groups(model)
    res = {}
    for name, loss in (("ce", out.loss), ("aux", out.aux_loss)):
        model.zero_grad()
        loss.backward(retain_graph=True)
        res[name] = {k: round(float(sum((p.grad.pow(2).sum() for p in ps if p.grad is not None),
                                         torch.zeros(())).sqrt()), 5) for k, ps in g.items()}
    model.zero_grad()
    model.eval()
    return res


def grad_snr(model, docs, k: int = 8, per: int = 6):
    """Gradient signal-to-noise per parameter group over k independent batches:
    r = ||mean_i g_i|| / mean_i ||g_i||. Pure noise gives ~1/sqrt(k); a consistent direction gives ~1."""
    model.train()
    g = groups(model)
    grads = {name: [] for name in g}
    for i in range(k):
        batch = docs[i * per:(i + 1) * per]
        ids, cu, ml = _pack(batch, "cpu")
        labels = torch.roll(ids, -1)
        labels[(cu[1:] - 1).long()] = -100
        model.zero_grad()
        out = model(ids, cu_seqlens=cu, max_seqlen=ml, labels=labels)
        out.loss.backward()
        for name, ps in g.items():
            grads[name].append(torch.cat([p.grad.reshape(-1).clone() for p in ps if p.grad is not None]))
    model.zero_grad()
    model.eval()
    # Unbiased split: E||g_bar||^2 = ||mu||^2 + sigma^2/k and E mean_i||g_i - g_bar||^2 = sigma^2 (k-1)/k.
    # SNR at the training batch (bytes_train) follows from sigma^2 ~ 1/batch.
    bytes_here = sum(len(d) + 1 for d in docs[:per])
    res = {"_bytes_per_batch": bytes_here, "_k": k}
    for name, gs in grads.items():
        G = torch.stack(gs)
        gbar = G.mean(0)
        sigma2 = float((G - gbar).pow(2).sum(1).mean() * k / (k - 1))
        mu2 = max(float(gbar.pow(2).sum()) - sigma2 / k, 0.0)
        res[name] = {"signal_norm": round(mu2 ** 0.5, 5), "noise_norm_per_batch": round(sigma2 ** 0.5, 5),
                     "snr_at_131k_bytes": round((mu2 / sigma2 * 131072 / bytes_here) ** 0.5, 3) if sigma2 else None}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--docs", type=int, default=48)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    trained, mcfg, meta = load_run_checkpoint(a.run)
    seed = int(meta["config"].get("train", {}).get("seed", 0))
    torch.manual_seed(seed)
    init = build_model(mcfg)
    init.init_weights()
    init.eval()
    docs = load_docs(a.docs, 1024)
    rep = {"checkpoint": meta["checkpoint"], "weight_drift": drift(trained, init),
           "main_output_similarity": output_similarity(trained, init, docs),
           "grad_norm_by_loss": grad_decomposition(trained, docs),
           "grad_snr_ce": grad_snr(trained, docs)}
    txt = json.dumps(rep, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
