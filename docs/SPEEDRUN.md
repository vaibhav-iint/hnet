# Speedrun track (S0 / S1 / S2)

The modded-nanogpt speedrun is our public, falsifiable end-to-end test: data, model, trainer and eval against a
benchmark with hundreds of published runs. This page covers the target, the record we port, how to run the
reference and our runs, and what is still open.

Owned files: `hnet/model/dense_modded.py`, `hnet/optim/muon.py`, `scripts/speedrun/`, `configs/speedrun/`,
`tests/test_speedrun_*.py`.

## The benchmark

* **Track 1 (GPT-2 small):** train to **3.28** mean cross-entropy (nats/token) on the first 10,485,760 tokens of
  the FineWeb val shard, on 8xH100. Target from Karpathy's llm.c GPT-2 124M reproduction (45 min, 10B tokens).
* **Current record** (README as of the pinned commit `4ea6b93`, 2026-09-28): **#92, 0.665 min (~40 s)**,
  2026-08-30, "ANVIL2". It combines sampled softmax over the early stages, full FP8 MLP, an 84.6M-row hashed
  n-gram embedding table sharded across GPUs, the ANVIL optimizer stack (twin-rail momentum, cautious decay),
  depth reduction with mixed-width attention and CUDA-graph capture of the whole step. It reaches 3.28 in
  under 330M tokens.
* **Track 2 (GPT-2 medium):** target **2.92** (llm.c's 350M baseline, 5.8 h). Current record: **#18, 17.35 min**
  (2025-12-31, "bulk transfer of short-track features"), `train_gpt_medium.py`.

### Target in bits per byte

Byte-level models cannot report nats/token, so we convert (`scripts/speedrun/target.py`, built on WS-C's
`hnet.data.speedrun_target`):

    target_bpb = loss * n_targets / ln 2 / UTF-8 bytes of those targets

| loss (nats/token) | track | target bpb | if EOT targets cost nothing |
|---|---|---|---|
| 3.28 | small | **1.0667** | 1.0652 |
| 2.92 | medium | **0.9496** | 0.9483 |

The 10,485,760 targets hold 46,515,256 text bytes (4.436 bytes/target) and 14,982 `<|endoftext|>` targets. The
record's mean includes those doc-end predictions, while our bpb leaves them out. The main column spreads their
loss over the text; the second column assumes they cost nothing. The true value lies in between (a 0.15% band).

Our val metric is computed differently from the record's. `eval.val: speedrun` uses
`hnet.data.speedrun_valset`: whole documents from the same 10.49M tokens, cut into ≤1792-token chunks, scored
identically for byte and token models. The record instead scores contiguous 64K windows with document masking
and a 1792-token attention window. The two protocols see almost the same context, so S1's val bpb should match
the record's loss/bpb conversion closely, but not exactly. To compare S1 with S0 in nats, use the training
`ce`/`bpb` curves together with the val curve.

## The ported record: #18 (2025-01-04, "Lower logit softcap from 30 to 15", 3.4 min)

**Why this record.** This is the most complete recipe that still fits in one ~550-line pure-PyTorch file with
no custom kernels and no FP8. The records after it trade simplicity for speed:

* #19: FP8 LM head;
* #20: merged QKV, long-short attention, batched Muon;
* #25 onward: BOS-aligned batches and a max document length, which change the data pipeline our shared loader
  fixes;
* #29 onward: FlashAttention-3, Triton kernels, NorMuon / Polar Express, batch-size schedules, multi-token
  prediction, hashed n-gram tables, sampled softmax, CUDA graphs.

Most of that progress is either systems work (irrelevant when we compare tokens/FLOPs to target in our own
trainer) or tied to tokens (n-gram tables, MTP, sampled softmax over the vocabulary). Those token-tied tricks do
not transfer to a byte-level H-Net, so including them would muddle the byte-vs-token comparison. #18 is also
well characterized: 80 runs give mean 3.2791 ± 0.0019. Its #17 parent has 1261 runs at 3.2794 ± 0.0018. Its
logged curve is 4.369 / 3.943 / 3.653 / 3.406 / 3.279 at steps 125 / 250 / 500 / 1000 / 1375.

**The cost of this choice.** #18 needs 1390 × 524,288 = **729M tokens** to reach 3.28, about 2.2x the current
record's tokens. That is fine for our purpose: S1 is a trainer check against S0 = #18 re-run on the same
hardware, and the H-Net comparison (S2) is made at equal bytes and FLOPs under the same recipe.

### What is kept

| Trick | Record #18 | Our port |
|---|---|---|
| Shape | 12 layers, d 768, 6 heads × 128, vocab 50304, MLP 4x | same (`DenseConfig` + `extra`) |
| Muon | NS5 (3.4445, -4.7750, 2.0315) in bf16, Nesterov, momentum 0.85→0.95 over 300 steps, lr 0.05 × sqrt(max(1, rows/cols)) | same (`hnet/optim/muon.py`) |
| Adam (non-matrix) | betas (0.8, 0.95), lr 0.6 embeddings, 0.008 head, 0.04 scalars, no wd | same |
| Schedule | constant, then linear to 0 over the last 40% of 1390 steps | `wsd` (warmup 0, decay 0.4, linear); off by one step at the start of the decay |
| QK-norm, ReLU², weightless RMSNorm, x0 = norm(embed) | yes | yes |
| Zero-init attn/MLP output projections and LM head | yes | yes (`init_weights`) |
| Untied head, tanh softcap 15 | yes | yes |
| Half-truncated RoPE (base 1024, half of each head's dims position-free) | yes | yes; positions restart per document, which is equivalent because RoPE is relative |
| Value embeddings: 3 tables mixed into v of layers 0-2 and 9-11 (`v = l0 v + l1 ve`) | yes | yes |
| Embedding shortcut in every block (`x = l0 x + l1 x0`) | yes | yes |
| U-net skips with learned weights | yes | yes |
| No attention in layer 7 | yes | yes (`skip_attn_layers`) |
| FlexAttention: document-causal mask + block sliding window, 128 → 1792 tokens over training | yes | yes (record's block-mask construction); window also enforced in `mask_mod`, see below |
| 64K-token document-masked windows per GPU, 8 GPUs | yes | 1 window of 65,536 tokens per micro-step, 8 micro-steps per step |

### What is dropped or different, and why

* **bf16 parameters and activations** (embeddings stored in bf16, bf16 residual stream). Our trainer keeps fp32
  master weights and autocasts the matmuls. This is slower and slightly more precise, and it is the
  precision every other run uses.
* **Doc-end targets.** Our packed batches mask the target that crosses a document boundary (SPEC labels), so we
  train on ~0.14% fewer targets than the record.
* **Distributed Muon.** The record splits Newton-Schulz work across ranks and all-gathers the updates. We run on
  one GPU, and under DDP every rank updates every matrix (exact but redundant). See the `muon.py` docstring for
  the upgrade path.
* **Eval** uses the shared whole-document protocol (above), not contiguous windows.
* **norm eps.** The record uses `F.rms_norm`'s default (dtype eps); we use `norm_eps: 1e-6`.
* **Window enforced in `mask_mod`.** The record enforces the sliding window only through block sparsity.
  Uncompiled `flex_attention` ignores block sparsity (we checked: it evaluates only `mask_mod`), so we repeat the
  window condition in `mask_mod`. Compiled results are unchanged, and the CPU test can check exact agreement.

### Model API (`hnet/model/dense_modded.py`, registry `("dense", "modded")`)

* SPEC forward:
  * input is `input_ids[T]` + `cu_seqlens`, or `[B, L]`;
  * `out.loss / ce_sum / n_targets / logits`, with `aux_loss = 0`, `stats = {}`, `bpred = []`;
  * `stage_of_param(name) = 0`; `blocks()` returns the 12 layers (attribute `layers`).
* Backends: FlexAttention on CUDA, compiled once on first use (T is padded to a multiple of 128 with a dummy
  segment); dense-mask SDPA on CPU. `extra.attn_backend` forces either one.
* **`model.set_progress(step / total_steps)`** drives the window schedule. Until it is called, the model uses the
  final window (1792 tokens), which is also what the record evaluates with.
* Knobs: `extra = {n_value_embeds, skip_attn_layers, unet_skips, logit_softcap, rope_base, rope_frac,
  window_tokens, attn_backend}`.
* Size: 276M parameters including the 5 embedding/head tables. The blocks hold 82.6M matmul parameters (vs 85.0M
  for WS-B's Llama-style `bpe_speedrun_124m`).

### Muon API (`hnet/optim/muon.py`, registry `("optimizer", "muon")`)

`build_muon_optimizer(model, train_cfg, param_groups_fn=None) -> MuonAdamW`. The trainer calls it when
`train.optimizer: muon`. It returns one optimizer with four kinds of groups:

* `muon`: 2-D hidden matrices. These include attention, MLP, Mamba-2 `in_proj`/`out_proj` and H-Net
  `residual_proj`.
* `embed`: `nn.Embedding` weights.
* `head`: `lm_head`.
* `other`: norms, scalars, biases, Mamba-2 `conv1d`/`A_log`/`D`/`dt_bias`, and router q/k projections (moved to
  Muon with `muon.router: muon`).

Hyperparameters live in `train.muon` (defaults = record #18; unknown keys raise). With WS-D's
`param_groups_fn`, each kind is further split by H-Net stage and decay/no-decay. The per-stage `lr_mult`
multiplies the kind's `base_lr`, and the trainer's schedule sets `lr = base_lr * lr_mult * factor`.

## Runs

| ID | What | Config | Notes |
|---|---|---|---|
| S0 | modded-nanogpt's own scripts, unmodified | `scripts/speedrun/run_reference.sh` | `record18` = the S1 reference; `current` = record #92 (8 GPUs only); `medium` = track 2 |
| S1 | record #18 in our trainer | `configs/speedrun/s1_modded.yaml` | pass: tokens to 1.0667 bpb within 10% of S0 |
| S2 | H-Net 1-/2-stage, same bytes/step and steps, Muon or AdamW | `configs/speedrun/s2_hnet{1,2}_{muon,adamw}.yaml` | models: WS-B's `configs/models/hnet{1,2}_speedrun.yaml` |
| tiny | CPU smoke tests of S1 and each S2 variant | `configs/speedrun/tiny_*.yaml` | synthetic data, 12 steps |

### Single GPU

```bash
# S0: the record S1 ports, on 1 GPU (built-in accumulation; ~8x the 8-GPU wall clock, ~31 GB peak at 64K)
scripts/speedrun/run_reference.sh setup             # clones the pinned commit, downloads 25 train shards
scripts/speedrun/run_reference.sh record18 1        # MICRO_TOKENS=32768 on a 24 GB card
# S1 / S2 (FineWeb shards via hnet.data: dataset fineweb10B, tokens for S1, decoded bytes for S2)
python -m hnet.train.train --config configs/speedrun/s1_modded.yaml
python -m hnet.train.train --config configs/speedrun/s2_hnet2_muon.yaml
# report: time / FLOPs / tokens / bytes to the target, plus curves
python scripts/speedrun/target.py
python scripts/speedrun/speedrun_report.py runs/speedrun/s1_modded runs/speedrun/s2_hnet{1,2}_{muon,adamw} \
    --plot runs/speedrun/curves.png
```

### Reference runs on fewer than 8 GPUs, and comparability

* **Current record (#92):** cannot run on fewer than 8 GPUs as published. `track_1_short/distributed.py`
  asserts `WORLD_SIZE == 8`; the sampled softmax, the sharded n-gram table and the sparse row exchanges assume
  one micro-batch per rank on 8 ranks. Its FA3 kernel also needs Hopper and a CUDA 13 runtime. Timing it needs
  one 8xH100 node for ~15 min (about 7 min of compile, untimed). We only need it for the headline comparison,
  not for S1.
* **Record #18 (S0 for S1):** on N GPUs each rank takes 524,288/N tokens per step and splits them into
  `max_device_batch_size` (64K) micro-batches. Gradient accumulation is therefore built in.
  * At 64K micro-batches, the data order, the windows and the val batches are identical to the 8-GPU run.
  * The gradient is 8/N times larger, because micro-batch means are summed. Muon normalizes this away and Adam
    is scale-invariant up to eps.
  * Result: the same loss curve, about 8/N times the wall clock.
  * With 32K micro-batches (24 GB cards), windows cut documents at 32K instead of 64K. This is a negligible,
    but not bit-exact, change.
* **Medium track:** `grad_accum_steps = 8 // world_size` is built in, so 1, 2 or 4 GPUs reproduce the 8-GPU math.
* **Comparing to published times:** on other hardware we compare **tokens, FLOPs and GPU-seconds to the
  target**, not wall clock. Time S0 on the same card as S1/S2. The published 3.4 min (8xH100) for #18
  corresponds to ~27 H100-minutes.

### S1 details

* Same batch (524,288 tokens), schedule and 1390 steps as the record.
* 8 micro-steps of one 65,536-token window: 64K is the record's per-GPU window, so document truncation matches.
* `grad_clip: 0` (the record does not clip).
* The CE is computed in checkpointed chunks (the record materializes fp32 [65536 × 50304] logits, ~13 GB plus
  their gradient), so the 64K window should fit well under the record's 31 GB peak (not yet measured). If it does
  not, use `--set data.seq_len=32768`.
* Expected cost: ~30-45 min on one H100/H200.

### S2 details and matching

* **Bytes per step:** 512 windows × 4608 bytes = 2,359,296, vs S1's 524,288 tokens × 4.436 = 2,325,700 bytes
  (+1.4%). Same 1390 steps.
* **FLOPs:** WS-B's `hnet{1,2}_speedrun` are matched (±2%) to `bpe_speedrun_124m` (Llama-style, 1024-token
  context), not to S1. By the SPEC count, S1 is ~0.77 training GFLOPs/token vs ~0.80 for `bpe_speedrun_124m`
  (fewer matmul parameters; windowed, document-masked attention). So the H-Nets carry ~4% more FLOPs/byte than
  S1. Re-match once `hnet/flops.py` counts the `modded` variant (WS-B TODO), or report FLOPs-to-target, which
  the report script does from each run's measured `flops_per_byte`.
* **Muon variant:** the record's split (Muon for matrices, Adam for the rest), with the paper's per-stage
  multipliers (`lr_multiplier: auto`) applied to every group. The LRs are a **proposal**: Muon 0.02, embed 0.02,
  head 0.008, other 0.004. The record's 0.6 / 0.04 assume N(0,1) embeddings and dense-only scalars; on Mamba-2
  `A_log`/`conv1d` they would be far too large. Router q/k stay on Adam (they start at identity; a fixed-norm
  Muon step would move them a lot).
* **AdamW variant:** lr 2e-3, betas (0.9, 0.95), wd 0.1, 100 warmup steps, the same 40% linear decay, grad clip
  1.0. Also a proposal.
* Before the matched runs, sweep `muon.lr` ∈ {0.01, 0.02, 0.04}, `muon.lr_other` ∈ {0.002, 0.004, 0.01} and the
  AdamW lr ∈ {1e-3, 2e-3, 4e-3} on ~300-step truncations (`train.stop_at_step`).

## Open issues

1. **Trainer hook:** `hnet/train/train.py` must call `model.set_progress(step / total_steps)` before each step
   (when present). Without it, S1 trains with the final 1792-token window throughout. Its loss should still be
   close, but it uses more attention FLOPs early and is not the record's schedule.
2. **FLOPs for the `modded` variant** in `hnet/flops.py` (currently counted as Llama: 3 MLP matrices, all 12
   attention layers, full context) and an S1-matched rematch of the speedrun H-Nets.
3. **S2 hyperparameters** are unswept proposals (see above).
4. Muon treats fused matrices (Mamba-2 `in_proj` = [z, x, B, C, dt], SwiGLU `fc1`, GQA `Wqkv`) as one matrix.
   Later records orthogonalize Q/K/V separately. Splitting by row blocks is a likely improvement for H-Net and
   needs the modules to expose their split sizes.
5. Distributed Muon (shard the NS work across ranks) only matters for multi-GPU runs.
