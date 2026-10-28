"""Manual audit of training batches, losses and normalization on real FineWeb-Edu data (CPU).

Checks, each recomputed independently of the code under test:
 1. byte windows: BOS/EOS placement, segment starts, labels = next byte within a segment, ignored across docs
 2. token windows: same text as the byte stream, labels, n_bytes = UTF-8 bytes of the predicted tokens
 3. H-Net forward: ce_sum / n_targets / loss vs a manual F.cross_entropy over logits
 4. ratio loss: aux/s{s} vs N/(N-1)((N-1)FG + (1-F)(1-G)) from the router outputs; aux_loss = alpha * sum
 5. untrained models: val bpb vs theory (uniform byte model: 8 bits/byte; uniform token model: log2(V)/bytes per token)

    python scripts/audit/audit_loss.py
"""

import math

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from hnet.config import BOS_ID, EOS_ID, IGNORE_INDEX, model_config_from_yaml
from hnet.data import WindowLoader
from hnet.data.download import data_root
from hnet.model import build_model

torch.manual_seed(0)
root = data_root() / "finewebedu10B"
byte_shard = root / "bytes" / "finewebedu_train_000001.bytes.bin"
tok_shard = root / "finewebedu_train_000001.bin"
enc = tiktoken.get_encoding("gpt2")
EOT = enc.eot_token
ok = lambda c, msg: print(("PASS " if c else "FAIL ") + msg)  # noqa: E731

# ---------------------------------------------------------------- 1. byte windows
bl = WindowLoader([byte_shard], "bytes", seq_len=2048, n_windows=4, prefetch=0, pin_memory=False)
b0 = bl.next_batch()  # windows 0-3: kept for the byte/token alignment check below
bl.consumed = 4  # windows 4-11 cover documents 1-4 of the shard (see its doc index), i.e. several boundaries
bl.n_windows = 8
b = bl.next_batch()
x, y, cu = b["input_ids"], b["labels"], b["cu_seqlens"].long()
T = x.numel()
ok(T == 8 * 2048 and int(cu[0]) == 0 and int(cu[-1]) == T, f"byte batch shape T={T}, {cu.numel() - 1} segments")
starts = set(cu[:-1].tolist())
bad_start = [s for s in starts if s % 2048 != 0 and int(x[s - 1]) != EOS_ID]
ok(not bad_start, "every segment starts at a window start or right after EOS")
n_bad_lab = 0
for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
    seg_x, seg_y = x[s:e], y[s:e]
    inner = seg_y[:-1]
    n_bad_lab += int((inner != seg_x[1:]).sum())
    last_ok = int(seg_y[-1]) == IGNORE_INDEX or (e % 2048 == 0 and int(seg_x[-1]) != EOS_ID)
    n_bad_lab += 0 if last_ok else 1
ok(n_bad_lab == 0, "labels[t] = x[t+1] inside segments; last label ignored at doc end (or cross-window continuation)")
eos_pos = (x == EOS_ID).nonzero().flatten().tolist()
ok(all(int(y[p]) == IGNORE_INDEX for p in eos_pos), f"EOS positions ({len(eos_pos)}) predict nothing")
ok(all(int(y[p - 1]) == EOS_ID for p in eos_pos if p % 2048 != 0 and (p - 1) not in []),
   "the byte before each EOS is trained to predict EOS")
bos_pos = (x == BOS_ID).nonzero().flatten().tolist()
ok(all(p in starts for p in bos_pos), f"every BOS ({len(bos_pos)}) starts a segment")
text0 = bytes(x[:300].tolist()).decode("utf-8", "replace").replace(chr(BOS_ID), "<BOS>").replace(chr(EOS_ID), "<EOS>")
print("     window 0 starts:", repr(text0[:160]))

# ---------------------------------------------------------------- 2. token windows (alignment + n_bytes)
tl = WindowLoader([tok_shard], "tokens", seq_len=448, n_windows=4, prefetch=0, pin_memory=False)
tb = tl.next_batch()
tx, ty = tb["input_ids"], tb["labels"]
tok_text = enc.decode_bytes([t for t in tx[:448].tolist() if t != EOT])
byte_stream = bytes(v for v in b0["input_ids"][:2048].tolist() if v not in (BOS_ID, EOS_ID))
common = min(len(tok_text), len(byte_stream))
ok(tok_text[:common] == byte_stream[:common], f"token window 0 decodes to the same text as byte window 0 ({common} B)")
valid = ty != IGNORE_INDEX
# hnet.train.data adds n_bytes = UTF-8 bytes of the INPUT tokens of each micro-batch.

nb_target = sum(len(enc.decode_bytes([t])) for t in ty[valid].tolist() if t != EOT)
nb_input = sum(len(enc.decode_bytes([t])) for t in tx.tolist() if t != EOT)
ok(abs(nb_target - nb_input) / nb_target < 0.01,
   f"trainer's token n_bytes (input-token bytes {nb_input}) vs bytes of predicted tokens {nb_target}: "
   f"{100 * (nb_input - nb_target) / nb_target:+.3f}% (train-curve bpb only; val uses exact counts)")
ok(int(((ty == EOT) & valid).sum()) == 0, "no <|endoftext|> targets (doc starts are ignored)")

# ---------------------------------------------------------------- 3 + 4. H-Net forward: CE and ratio loss
cfg = model_config_from_yaml("configs/models/tiny_hnet2_2m.yaml")
for router in ("cosine_lookahead", "end_mlp"):
    cfg.dc.router = router
    m = build_model(cfg)
    m.init_weights()
    m.eval()
    with torch.no_grad():
        out = m(x, cu_seqlens=b["cu_seqlens"], max_seqlen=b["max_seqlen"], labels=y, return_logits=True)
    ce_manual = F.cross_entropy(out.logits.float(), y, ignore_index=IGNORE_INDEX, reduction="sum")
    nt = int((y != IGNORE_INDEX).sum())
    ok(torch.allclose(out.ce_sum, ce_manual, rtol=1e-5) and int(out.n_targets) == nt
       and abs(float(out.loss) - float(ce_manual) / nt) < 1e-5,
       f"[{router}] ce_sum {float(out.ce_sum):.2f} = manual; loss = ce_sum / {nt} targets = {float(out.loss):.4f} "
       f"(uniform would be ln 256 = {math.log(256):.4f})")
    aux_manual = 0.0
    for s, r in enumerate(out.bpred):
        N = cfg.dc.target_ratios[s]
        Fv = r.boundary_mask.float().mean().item()
        Gv = r.boundary_prob[..., -1].float().mean().item()
        L = N / (N - 1) * ((N - 1) * Fv * Gv + (1 - Fv) * (1 - Gv))
        aux_manual += L
        ok(abs(L - float(out.stats[f"aux/s{s}"])) < 1e-5,
           f"[{router}] ratio loss s{s}: F={Fv:.3f} G={Gv:.3f} N={N} -> {L:.4f} (stats {float(out.stats[f'aux/s{s}']):.4f})")
    ok(abs(cfg.dc.ratio_loss_alpha * aux_manual - float(out.aux_loss)) < 1e-6,
       f"[{router}] aux_loss = alpha * sum = {float(out.aux_loss):.5f}")

# ---------------------------------------------------------------- 5. untrained val bpb vs theory
from hnet.data import standard_valset  # noqa: E402

vs = standard_valset("finewebedu10B", n_bytes_target=200_000, byte_seq_len=2048, token_seq_len=448)
for kind, path in (("bytes", "configs/models/tiny_hnet2_2m.yaml"), ("tokens", "configs/models/tiny_bpe_2m.yaml")):
    mc = model_config_from_yaml(path)
    m = build_model(mc)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()  # all-zero weights -> uniform softmax
    ce = nb = 0.0
    for vb in vs.batches(kind)[:8]:
        with torch.no_grad():
            o = m(vb["input_ids"], cu_seqlens=vb["cu_seqlens"], max_seqlen=vb["max_seqlen"], labels=vb["labels"],
                  return_logits=False)
        ce += float(o.ce_sum)
        nb += float(vb["n_bytes"])
    bpb = ce / math.log(2) / nb
    V = mc.vocab_size
    if kind == "bytes":
        theory = math.log2(V)
    else:
        n_tgt = sum(int((vb["labels"] != IGNORE_INDEX).sum()) for vb in vs.batches(kind)[:8])
        theory = math.log2(V) * n_tgt / nb
    ok(abs(bpb - theory) < 1e-3 * theory, f"uniform {kind} model val bpb {bpb:.4f} = theory {theory:.4f}")
print("val set:", vs.describe() if hasattr(vs, "describe") else "")
