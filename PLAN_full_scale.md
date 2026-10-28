# H-Net vs SmolLM3: full experiment plan

*Draft 2, 2026-10-28. Checked against the rendered blog posts (charts and the JS data behind them), the arXiv HTML
(figures and tables), goombalab/hnet, huggingface/smollm, and modded-nanogpt. Items marked **[inferred]** are our
derivations. Reference data extracted from the blog is in `configs/reference/`.*

---

## 0. Summary

* **Headline claim.** Blog I, Fig. 7: a 3.67B 2-stage byte H-Net, trained on a rebuild of SmolLM3's stage-1 mix at
  matched FLOPs/byte, reaches SmolLM3's final six-task average (63.8%) after 0.99T tokens instead of 11.04T (">10x").
* **Most of that 10x is a cooldown comparison.** The H-Net was cooled down over its last 15% (from 0.855T); SmolLM3's
  stage-1 checkpoints are mid-stable-phase. Comparing stable phase to stable phase on Cartesia's own curves:
  * Δ = **+1.8** points at 94B tokens, **+1.25** at 189B.
  * Data multiplier **~1.7x** at 94B and **~3.0x** at 189B (quadratic fit in log-tokens to SmolLM3's stage-1 points).
* **The architecture behind those numbers is unreleased.** With the public LH-Net router we expect less. How much less
  is the main open question.
* **Budget.** The Prime team wallet holds **$1,843**. Prime currently lists only single H200 pods, at $5.40/GPU-h.
  That buys about 340 GPU-hours, enough for every validation and pilot tier (A, about $1.3k with margin). The 3.7B main run (tier B)
  needs about $8k per 40k steps at that price, plus a 64-GPU allocation that Prime is not listing right now.
* **Additions you asked for:**
  * a byte-level speedrun against modded-nanogpt and other small Transformers (section 3), as the cheapest end-to-end
    test of our implementation;
  * a modular codebase (section 9) so new routers, mixers and losses are config changes.

---

## 1. Claims under test (verified numbers)

| ID | Claim | Source (verified) | Test |
|---|---|---|---|
| C1 | 2-stage H-Net beats a BPE Transformer at matched FLOPs/byte and data. XL: 0.715 vs 0.730 FineWeb-Edu bpb, 58.2 vs 55.5 seven-task avg. L: 0.743 vs 0.756, 55.5 vs 53.3. XL crossover at 30B bytes; at L the H-Net leads from the start. | paper Table 2, Fig. 3 | E0 (eval released ckpts), P1 (retrain at L) |
| C2 | Smoothing is the critical DC component (large bpb loss without it). Removing cosine routing or STE costs little. Mamba-2 enc/dec beat Transformer enc/dec. | paper Fig. 7, Fig. 8 | P2 ablations, speedrun scale |
| C3 | 3.67B H-Net vs SmolLM3 on six tasks: 55.7 vs 53.9 at 94B, 58.1 vs 56.9 at 189B, 63.8 (0.99T, after cooldown) vs 63.8 (11.04T) | blog I Fig. 7 data | **M1** (tier B), P3 (tier A proxy) |
| C4 | The lookahead router (LH-Net) gives poor stage-0 splits ("r hino", "fro m") and breaks CJK characters in multi-stage models | blog I "Aside" | Boundary diagnostics on every H-Net; P2(e) router variant |
| C5 | 1-stage H-Net ladder vs BPE ladder (Muon, iso-FLOP): 1.28x validation, 1.83x Paloma, 2.06x OLMES compute multipliers; code 1.49–2.40x, OOD code up to 5.37x. Blog II: vs Byte 9.19x, BPE-NP 1.89x, BLT-like 1.40x (compute→val). | blog I Fig. 4–5, blog II table | **Tier C, optional** (ladder-lite) |

---

## 2. Models

### 2.1 Evaluated only (no training cost beyond eval)

| ID | Model | Role |
|---|---|---|
| B1 | `HuggingFaceTB/SmolLM3-3B-checkpoints`, `stage1-step-{40k…800k}` (86 stage-1 revisions exist, every 40k steps) | Primary baseline curve. **Harness target: reproduce Cartesia's SmolLM3 series in `configs/reference/blog1_fig7_smollm3.csv`** (e.g. step 40k: HS 62.44, ARC 50.88, PIQA 74.59, Wino 56.75, CSQA 43.08, MMLU 35.49) |
| B2 | `HuggingFaceTB/SmolLM3-3B-Base` (11.04T) | Ceiling. Cartesia's harness gives 63.83 (six tasks). Note its Winogrande is 63.69 vs the model card's 58.88, so Cartesia's harness differs from the card on Winogrande and we validate against Fig. 7, not the card |
| B3 | `cartesia-ai/hnet_{1,2}stage_{L,XL}` | Reproduce paper Table 2 (bpb + 7 tasks); logit parity of our code vs the official code |
| B4 | Small public LMs: GPT-2 124M/355M, Pythia-160M/410M, SmolLM2-135M/360M | Reference points for the speedrun (FineWeb val bpb) |
| B5 | Cartesia's H-Net curve (`blog1_fig7_hnet.csv`) | Target curve for M1/P3: how much of Cartesia's gain the public LH-Net recovers |

### 2.2 Trained by us

| Tier | ID | Model | Data | Budget | Purpose |
|---|---|---|---|---|---|
| A | **S-\*** | Speedrun family (section 3) | FineWeb 10B (modded-nanogpt shards, decoded to bytes) | ~35 GPU-h | End-to-end implementation test against public records |
| A | P1-H | H-Net 2-stage **L** (paper Table 1: M4 + T1M4 + T26 + M4T1 + M4, D = 1024/1024/1536, (3,3)-DC, 870M) | FineWeb-Edu `sample-100BT`, 8192-byte seqs, batch 256 | 30B bytes | C1 retrain; at L the paper's gap is visible early (Fig. 3) |
| A | P1-T | BPE Transformer L (T24, D = 1536, GPT-2 tok., 1792-token seqs, 760M) | same | 30B bytes | Paper's baseline (never released) |
| A | P2-a…e | ~125M-FLOPs-equivalent 2-stage H-Nets, one change each: (a) no smoothing, (b) no LR modulation, (c) Transformer enc/dec, (d) no STE, (e) **causal end-of-chunk router** (our guess at the blog-II fix) | FineWeb-Edu | 5B bytes each | C2 + C4; picks the router for later tiers |
| A | P3-H | H-Net ~0.5B (main net: SmolLM3 block shape, depth solved for FLOPs) | **our SmolLM3 stage-1 byte mix** | 10B tok-eq | Pilot of the main recipe (SmolLM3 optimizer, byte mix) |
| A | P3-D | Dense SmolLM3-architecture ~0.5B, FLOPs/byte-matched | **same shards, same order** | 10B tok-eq | Fully controlled baseline (identical data and code) |
| B | **M1** | H-Net 3.7B (`configs/hnet_3p7b_smollm3.yaml`) | our SmolLM3 stage-1 byte mix | 40k steps → gate → 80k | Headline vs B1 at 94B / 189B tokens |
| B | M1-cd | 15% cooldown branch from M1's 80k checkpoint | same | +12k steps | Measures how much the cooldown adds; Cartesia's H-Net gained +2.3 from it |
| C | L-\* | Ladder-lite: 1-stage H-Net vs BPE (in-corpus 128k BPE), budgets 10^18.5–10^19.5, Muon + WSD | blog-I ladder mix (70/20/5/5 by bytes) | ~600 GPU-h | C5 at the small end only |

Not planned: MambaByte, LlamaByte, SpaceByte (paper baselines that the SmolLM3 claim doesn't need), BPE-NP, Byte and
BLT-like ladders (blog II), DNA and vision.

---

## 3. Speedrun track (new)

Purpose: a fast, public, falsifiable test of the whole stack (data → model → trainer → eval) before spending on pilots.

* **Task.** modded-nanogpt's target is **3.28 nats/token on the FineWeb val set** (first 10,485,760 GPT-2 tokens,
  8×H100). We decode the same val tokens to text and convert the target to **bits per byte**:
  `bpb* = 3.28 / ln 2 / (val UTF-8 bytes / val GPT-2 tokens)`. That gives ≈1.0 bpb **[inferred until measured]**.
  Byte models train on the same FineWeb train shards, decoded to UTF-8.
* **Reported per run:** wall clock, GPU-seconds, training FLOPs and bytes seen until val bpb ≤ bpb*, plus the full
  bpb-vs-FLOPs curve. Hardware is noted. Prime lists only single H200s right now, so we time on 1×H200 and convert to
  GPU-seconds. Official modded-nanogpt records use 8×H100; we also re-time the record script on our hardware for a
  like-for-like reference.
* **Runs:**
  | ID | Model | Optimizer | Notes |
  |---|---|---|---|
  | S0 | modded-nanogpt current record script, unmodified | as published | Hardware calibration |
  | S1 | llm.c-style GPT-2 124M (AdamW) in **our trainer** | AdamW | Validates the trainer against the published llm.c curve (3.28 at ~10B tokens) |
  | S2 | modded-nanogpt-style dense Transformer in our trainer (Muon, ReLU², QK-norm, …) | Muon + AdamW | Shows our trainer reaches near-record efficiency |
  | S3 | H-Net 1-stage, FLOPs/byte-matched to S1 | AdamW | Byte vs token at speedrun scale |
  | S4 | H-Net 2-stage, matched to S1 | AdamW | Paper's best layout at small scale |
  | S5 | S4 + modded-nanogpt tricks (Muon on hidden matrices, etc.) | Muon | Playground for new ideas; starting point for "byte speedrun" records |
  | B4 | Small public LMs, eval only | — | bpb on the same val text |
* **Gate to tier-A pilots:** S1 within 10% of llm.c's tokens-to-3.28. S3/S4 train stably, compression converges to
  target, and no causality leak (val bpb identical whether or not the future context is padded).

---

## 4. Matching rules

| Quantity | Rule |
|---|---|
| Compute | Training FLOPs/byte within ±2%, from realized per-stage compression on held-out data. Paper App. B counting: forward FLOPs, backward = 2× forward, causal-only attention (blog I), Mamba-2 terms, LM head counted, embedding lookup not counted (this reproduces the blog's 0.35 GFLOPs/byte for SmolLM3's head) |
| Data per step | Equal bytes per step. Main: 576 sequences × (4096 × measured Llama-3 bytes/token). The blog's Fig. 9 implies ~9.34M bytes/step, below our assumed 10.6M. Measure bytes/token on the mix and record which batch we use |
| Data content | Same sources and **byte** weights (M1, P3); identical shards for P1, P3, S |
| Optimizer | Token model's recipe unchanged. SmolLM3: AdamW (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, peak 2e-4, 2000-step warmup, constant. Paper P1: AdamW, WSD 10% warmup / 20% inverse-sqrt decay, peak 6.25e-4 (L) |
| LR modulation | Paper App. C: λ^s = sqrt(N_GPT · D^S/D^s / ∏_{i<s} N^i). For M1 (N_GPT ≈ 4.5, N = (4, 2), all D = 2048): λ ≈ (2.12, 1.06, 0.75). The blog doesn't say whether it was used, so P3 and P2(b) decide |
| Comparison phase | M1 vs B1: both in stable phase. P1/P3/S: full curves, and final points after an identical cooldown |
| Doc boundaries | Attention and SSM state reset at document boundaries |

---

## 5. Benchmarks

One harness that we run for all models, zero-shot, cloze formulation, prompts replicated from SmolLM3's lighteval
`tasks.py`. Validated against Cartesia's own SmolLM3 numbers (B1) and paper Table 2 (B3) before any comparison.

### 5.1 Primary: Cartesia's six-task average (verified from the Fig. 7 source)

**HellaSwag, ARC (Easy + Challenge mean), PIQA, WinoGrande, CommonsenseQA, MMLU**, all CF. Normalization: we determine
which lighteval normalization reproduces Cartesia's SmolLM3 series (token- vs char-norm) and use that for the SmolLM3
rows. For cross-model deltas we use char-norm for every model. If the two normalizations disagree on direction, we
report both.

### 5.2 Secondary

| Group | Tasks | Use |
|---|---|---|
| SmolLM3 extra | BoolQ, OBQA, MMLU-Pro CF | Reported |
| Paper suite | LAMBADA (acc), HellaSwag (acc_n), PIQA (acc), ARC-e (acc), ARC-c (acc_n), Winogrande (acc), OBQA (acc_n) | P1, B3 |
| Robustness | HellaSwag with AntSpeak, Drop, RandomCase, Repeat, UpperCase (BLT protocol); robustness score = 100·(perturbed − 0.25)/max(clean − 0.25, 0) | P1, P3, M1 |
| Soft accuracy | bpb of the gold continuation on each MC task (blog's OLMES-style "soft accuracy") | Smoother than accuracy early in training; all tiers |
| Multilingual | `mlmm_hellaswag_*`, `belebele_*` CF (fra, spa, deu, ita, por, zho, rus) | P3, M1 |
| Generative | GSM8K 5-shot, MATH 4-shot | Skipped below ~0.5T tokens |

### 5.3 Language-modeling metrics

Held-out bpb per source group (web / multilingual / code / math). FineWeb-Edu val bpb (P1, B3). FineWeb val bpb (S).
Paloma bpb if accessible.

---

## 6. Metrics

**Every step:** loss, CE, bpb · ratio loss, F (boundary fraction), G (mean boundary prob) per stage · realized
compression per stage, bytes per main position · realized FLOPs/byte, MFU · grad norm (global, per stage) · LR per group ·
bytes/s/GPU · step time · peak memory · padding waste · chunk-cap events.

**Every 1k steps:**

* held-out bpb and compression **per source**;
* boundary diagnostics: % of boundaries inside a UTF-8 multi-byte character, % mid-word, word-stem alignment;
* chunk counts on the H-Net paper abstract, Cartesia's Fig. 9 probe (their reference at 671B bytes: 396 stage-0 and 208
  stage-1 chunks, 8.4 bytes/chunk);
* an explicit monitor for the early compression collapse-and-recovery that Fig. 9 shows (stage-0 chunks drop from ~950
  to ~50 within ~500 steps, then recover).

**Headline (eval points):**

* Δ six-task average with a paired bootstrap 95% CI;
* gap over the curve: mean Δ over all eval points from 20k–80k steps, because single checkpoints are noisy (SmolLM3's
  curve moves ±1 point between neighbours);
* data multiplier: tokens at which a smooth fit to SmolLM3's stage-1 curve reaches our score, ÷ our tokens;
* **fraction of Cartesia's gain recovered** = our Δ / Cartesia's Δ at the same tokens;
* per-task deltas, bpb deltas, robustness score, and cost in GPU-h and $.

---

## 7. Decision rules (pre-registered, calibrated to Cartesia's own curves)

| Experiment | Pass | If it fails |
|---|---|---|
| E0 harness | B1 within ±1.0 of Cartesia's SmolLM3 series on each task at ≥ 3 checkpoints; B3 within ±0.5 of Table 2 | Fix the harness |
| E1 parity | Max logit diff vs official code < 1e-2 (bf16) on `hnet_2stage_XL`; ≥ 99.9% identical boundaries | Fix the model |
| S | Section 3 gate | Fix the trainer |
| P1 | H-Net-L bpb < BPE-L bpb at 30B bytes, by ≥ 0.008 (paper gap ≈ 0.013 at end) | Debug LR modulation, ratio loss, init |
| P3 | P3-H bpb < P3-D bpb at 10B tok-eq; stable compression; MFU ≥ 25% | Tune or stop before M1 |
| M1 directional | Mean Δ over 20k–80k > 0 with CI excluding 0 | — |
| M1 "Cartesia-level" | Δ ≥ +1.2 at 80k, multiplier ≥ 2.5x (Cartesia: +1.25, ~3.0x) | — |
| M1 fail | Δ ≤ 0 | Gain likely needs the unreleased router → scale up P2(e) |

The draft-1 rule "the gap must not shrink from 40k to 80k" is dropped: Cartesia's own gap shrinks (+1.8 → +1.25).

---

## 8. Budget (Prime, current list price $5.40/H200-h; wallet $1,843)

| Tier | Items | GPU-h | Cost |
|---|---|---|---|
| A0 | E0 harness + B1 sweep (~12 SmolLM3 ckpts × 6–10 tasks), B3, B4; E1 parity | ~15 | ~$80 |
| A1 | Speedrun S0–S5 incl. iterations (S1/S3/S4 each ≈ llm.c's 6 GPU-h) | ~40 | ~$215 |
| A2 | P1 pair at L, 30B bytes (1.26 train GFLOPs/byte × 30B ≈ 3.8e19 FLOPs per model) | ~70 | ~$380 |
| A3 | P2 ablations (5 × ~3 h) | ~15 | ~$80 |
| A4 | P3 pair at 0.5B, 10B tok-eq | ~55 | ~$300 |
| A | **Subtotal + ~20% debugging margin** | **~235** | **~$1,270** |
| B | M1 to 40k / to 80k (~1,500 / ~3,000 H200-h); needs ~64 GPUs | 3,000 | ~$16k at $5.40 (~$8k at $2.50/H100-h) |
| C | Ladder-lite | ~600 | ~$3.2k |

Tier A fits the current wallet. Tier B needs more funding and a multi-node allocation, so it is decided after tier A. Any
single launch above ~$100 waits for your OK.

---

## 9. Implementation design (modular)

New ideas should be a new registered component plus a config, not edits to core code.

```
hnet/
  config.py               dataclasses + YAML + --set overrides; every component chosen by name
  registry.py             register("router", "cosine_lookahead")(cls) ...; build(kind, name, **cfg)
  model/
    mixers/               "m" Mamba-2 (pure-torch SSD + mamba_ssm kernel), "t" attention (SDPA / flex / flash-attn,
                          GQA, RoPE / NoPE, sliding window), "T"/"M" add a SwiGLU MLP; new codes register here
    routing/              cosine_lookahead (paper LH-Net), causal_end_of_chunk (ours), fixed_pool, space_like,
                          entropy_proxy (BLT-like), external (frozen boundaries)
    chunking/             downsample(select) | mean/max pool ; dechunk: ema_smoothing | repeat ; STE on/off
    losses/               ratio_loss (paper), plus a slot for alternatives
    hnet.py               recursive stage container: encoder → router → chunk → inner (HNet | Isotropic) →
                          dechunk → residual → decoder. Depth comes from config, not code
    dense.py              SmolLM3/Llama-style and modded-nanogpt-style baselines
  optim/                  AdamW, Muon; per-stage LR multipliers; WSD / constant / cosine schedules
  data/                   source registry, bytes/token measurement, byte and token shard builders, packed loader
  train/                  one trainer (FSDP2, act. checkpointing, metrics hooks, checkpoint/resume)
  eval/                   scorers (byte / token, same interface), tasks, perturbations, bpb, bootstrap, report
  flops.py                analytic FLOPs/params for any config (matching + MFU)
infra/prime/              launch, sync code, run, tail metrics, pull checkpoints, teardown
configs/                  experiments are YAML files only (speedrun/, paper/, smollm3/, ablations/)
```

Contracts that keep components swappable:

* every router returns `(boundary_prob[T,2], boundary_mask[T], selected_probs[T,1])`;
* every mixer is `forward(x, seq_ctx)`, where `seq_ctx` carries `cu_seqlens`, `seq_idx` and `max_seqlen`;
* every stage reports its stats into one metrics dict;
* parameter names stay compatible with goombalab/hnet so Cartesia's checkpoints load.

Tests: causality (no future-byte leakage through routing), packed-vs-separate equivalence, SSD vs sequential reference,
kernel vs pure-torch parity, official-checkpoint parity, and a tiny-model convergence test, all runnable on CPU.

---

## 10. Corrections to the draft-1 write-up

1. **The six tasks** are HellaSwag, ARC, PIQA, WinoGrande, CSQA and **MMLU**. OBQA is not among them.
2. **The 10x** compares a cooled-down H-Net to stable-phase SmolLM3 checkpoints. Stable vs stable is ~1.7x (94B) and
   ~3x (189B).
3. **Code data**: the SmolLM3 config comment says Stack v2 (StarCoder2Data); the stack-edu S3 names are a typo. The blog
   table also lists `stack-v2-*` sources. We use Stack v2 where its contents are fetchable, otherwise stack-edu, and
   label which one was used.
4. pes2o, wiki and stackexchange come from `allenai/dolmino-mix-1124`. Most languages use FineWeb2-HQ; kor/hin/tha use
   fineweb-2.
5. LR modulation (App. C) is now exact: λ^s = sqrt(N_GPT · D^S/D^s / ∏_{i<s} N^i), with a suggested default of
   (2.0, 1.5, 1.0).
6. The paper's 7 tasks include LAMBADA. Metrics: acc for LAMBADA, PIQA, ARC-e and Winogrande; acc_norm for HellaSwag,
   ARC-c and OBQA.
7. The robustness suite is 5 HellaSwag perturbations plus a robustness score (App. D.1).
8. The ratio loss in the official code is computed per device and then averaged.
9. SmolLM3 applies weight decay to norms (it excludes only the embedding). The official H-Net code excludes norms and
   biases. Each model keeps its own convention.
10. **[inferred]** Stage-1 width/FFN for the 3.7B model: MHA with d_intermediate 5504 at width 2048 reproduces the blog's
    0.32B stage-1 params.
11. **Open:** the blog's bytes/step (Fig. 9 implies ~9.34M) vs SmolLM3's 2.36M tokens × 4.5 B/tok; whether LR modulation
    and α = 0.03 were used at 3.7B; Cartesia's eval normalization.

---

## 11. Execution order

1. **CPU, no spend (parallel workstreams):** modular model, FLOPs, data (FineWeb/FineWeb-Edu/SmolLM3 mix), trainer,
   eval harness, Prime tooling. All CPU tests green.
2. **Tier A0 + A1 (~$295):** harness validation, checkpoint parity, speedrun.
3. **Tier A2–A4 (~$760)** after the speedrun gate.
4. Decide on tier B with the tier-A results in hand.
