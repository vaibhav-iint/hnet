# H-Net vs tokenized Transformers at 100M–500M scale: $500 plan

*Draft 3, 2026-10-28. Replaces the large-scale plan, which is kept in `PLAN_full_scale.md` (claims, verified reference
numbers, corrections). Costs assume a single H200 at $5.40/h (Prime's current list price), 30% MFU for dense models and
25% for H-Nets. Cheaper GPUs only lower the bill (section 6).*

## 1. What this budget can and cannot answer

* **Can:** the paper's core claim at small scale. A 2-stage byte-level H-Net beats a BPE Transformer at matched
  FLOPs/byte and matched data. We measure the gap, the **data multiplier** (computed the blog's way: stable phase vs
  stable phase), which components matter, and whether a causal "end-of-chunk" router fixes the lookahead problem the
  blog describes.
* **Cannot:** the 3B SmolLM3 ">10x" claim directly. That needs a ~3.7B model (~$16k). We apply the same methodology at
  125M and 350M and report how the gap changes with scale.

## 2. Runs

Data: **FineWeb-Edu** (the paper's dataset; `sample-10BT` is enough). Byte models see 8192-byte sequences. BPE models use
the GPT-2 tokenizer at ~1792 tokens/seq (≈8192 bytes), so both see the same bytes per step. Speedrun runs use **FineWeb**
(modded-nanogpt's data).

| # | Run | Size | Data | Est. cost | Purpose |
|---|---|---|---|---|---|
| 0 | Eval-only checks: GPT-2 124M/355M, SmolLM2-135M/360M, `cartesia-ai/hnet_2stage_L` (paper Table 2), 2 SmolLM3 checkpoints | — | — | ~$25 | Validates the eval harness and our model code against published numbers |
| **Speedrun** | | | | | |
| S0 | modded-nanogpt record script, unmodified | 124M | FineWeb | ~$5 | Same-hardware reference time |
| S1 | Our dense GPT-2-style model, modded-nanogpt recipe ported into our trainer | 124M | FineWeb | ~$10 | Trainer check: should land near S0's tokens-to-3.28 |
| S2 | H-Net 1-stage and 2-stage, FLOPs/byte-matched to S1, to the equivalent bpb target | ~125M-eq | FineWeb (bytes) | ~$30 | Byte vs token on a public benchmark; iterations included |
| **125M core** | | | | | |
| A1 | BPE Transformer (GPT-3 Small shape), constant LR, checkpoint every 0.5B tokens, to **10B tokens** | 125M | FineWeb-Edu | ~$40 | Baseline curve, long enough to read off a 2–4x data multiplier |
| A2 | H-Net 2-stage, (3,3)-DC, FLOPs/byte-matched to A1, to 5B tok-eq | ~125M-eq | same | ~$24 | Main comparison |
| A3 | H-Net 1-stage, 6-DC, matched, to 5B tok-eq | ~125M-eq | same | ~$24 | 1- vs 2-stage |
| **125M ablations** (each: A2 with one change, 2.5B tok-eq) | | | | | |
| B1 | no smoothing | | | ~$12 | Paper Fig. 7 says this matters most |
| B2 | no per-stage LR multipliers | | | ~$12 | |
| B3 | **causal end-of-chunk router** (our design) | | | ~$12 | Our guess at the blog's "lookahead" fix |
| B4 | Transformer encoder/decoder instead of Mamba-2 | | | ~$12 | Paper Fig. 8 |
| **350M scale-up** | | | | | |
| C1 | BPE Transformer (GPT-3 Medium shape) to 5B tokens | 350M | FineWeb-Edu | ~$61 | Does the gap grow with scale? |
| C2 | Best H-Net from A/B (2-stage, router chosen by B3), matched, to 5B tok-eq | ~350M-eq | same | ~$73 | |
| | **Subtotal** | | | **~$355** | |
| | Debugging and reruns margin (~30%) | | | ~$100 | |
| | **Total** | | | **~$455** | |

Optional if money is left: run C2 at 500M instead of 350M (+$50), or add cooldown branches to A1/A2 (~$5 each) to
compare cooled-down models the way the blog's headline does.

## 2b. Architecture search: guessing the blog's improved H-Net (1–10M params, ~$20)

The blog's H-Net is an unreleased improvement on the paper. Its chunk-viewer data (EXP-001) shows the improved
router marks the **last** byte of each chunk, decided causally. Its stage-1 chunks end after spaces ("word␣") and
almost never split a word or a UTF-8 character. The router function itself is not published, so we test guesses on
tiny models before scaling. Axes, ordered by how strongly the evidence supports them:

| Axis | Variants | Evidence |
|---|---|---|
| Boundary semantics | start (paper `cosine_lookahead`) vs **end** | EXP-001: 99.8% of p ≥ 0.5 sit on chunk-final bytes |
| End-router function | `end_linear`, `end_mlp`, `end_cosine_shift` | none, pure guess |
| Target ratios | (3,3) paper vs (4,2) blog | blog text |
| Smoothing | EMA vs none | paper Fig. 7 (critical for LH-Net); unknown for end routers |
| Ratio-loss weight | α = 0.03 vs 0.1 ("strong auxiliary loss" in the blog ladder); local vs global scope | blog text, official code |
| Optimizer | AdamW vs Muon; LR multipliers on/off | blog ladders use Muon |

**Protocol:**
* Screen 2-stage H-Nets of ~2M params on FineWeb-Edu bytes at equal FLOPs/byte and bytes, 3 seeds each.
* Promote the top 2–3 to ~10M params (2 seeds). Compare against the matched tiny BPE model.
* Score each variant on val bpb at a fixed byte budget and on chunk diagnostics measured against the blog's reference
  (stage-1 ≈ 7 bytes/chunk in English, ≈ 0% word-internal cuts, ≈ 90% of chunks ending after a space).
* The winner replaces the paper router in the 125M runs; the paper router stays as a control (A2 vs A2-paper).
* Cost: the 2M screen runs on CPU or a few H200-hours total; budget ~$20 from the margin.

## 3. Matching rules (unchanged from the full plan)

* Equal **bytes per step** and **training FLOPs/byte within ±2%**, computed from realized compression. Counting follows
  paper App. B: causal attention only, Mamba-2 terms included, LM head counted.
* The BPE model's optimizer recipe is copied to the H-Net (AdamW, WSD with constant LR during comparisons). The H-Net
  adds the paper's per-stage LR multipliers λ^s = sqrt(N_GPT · D^S/D^s / ∏_{i<s} N^i) and ratio loss α = 0.03.
* Compare in the stable phase. The data multiplier is the tokens at which A1's smoothed curve reaches A2's score,
  ÷ A2's tokens.

## 4. Metrics

* **Primary at this scale: FineWeb-Edu validation bpb.** Most accuracy benchmarks sit near chance at 125M.
* **Secondary:**
  * "soft accuracy": bpb of the gold answer on HellaSwag, ARC-E/C, PIQA, Winogrande, CSQA, OBQA, MMLU (smooth even when
    accuracy is at chance);
  * zero-shot accuracy on the same tasks (char-normalized), plus LAMBADA;
  * HellaSwag robustness under the paper's 5 perturbations.
* **H-Net diagnostics:**
  * compression per stage;
  * ratio-loss terms;
  * % of boundaries splitting a UTF-8 character or a word;
  * chunk visualizations;
  * early compression collapse/recovery (blog Fig. 9).
* **Systems:** FLOPs/byte, MFU, bytes/s, GPU-hours, $ per run.

## 5. Pass/fail rules (pre-registered)

| Check | Pass |
|---|---|
| Harness (#0) | Reproduces Cartesia's `hnet_2stage_L` Table 2 row within ±0.5, and GPT-2's published bpb/accuracy within ±0.5 |
| Trainer (S1) | Within 10% of S0's tokens-to-target |
| Core (A2 vs A1) | H-Net bpb lower than BPE at equal tokens from ~1B tok-eq on (the paper's L-scale curves separate early), with a data multiplier > 1 at 5B |
| Ablations | Report effect sizes; no threshold |
| Scale (C) | Report whether the 350M gap and multiplier are larger than at 125M |

## 6. Hardware

Smaller GPUs work. Nothing in this plan needs an H100.

* **Required:** bf16 support (NVIDIA Ampere or newer: A100, A10/A6000, RTX 3090/4090, L4/L40S, H100/H200). Every fast
  kernel (Mamba-2 Triton, FlashAttention) has a pure-PyTorch fallback, so other GPUs run too, just slower.
* **Memory:** 24 GB is enough for every run here (micro-batch 1–4 sequences plus activation checkpointing). No
  multi-GPU or multi-node is needed; one GPU per run, with runs in parallel if several GPUs are rented.
* **Cost:** what matters is $ per useful FLOP. Rough cost per 1e18 training FLOPs (realistic MFU):
  * H200 at $5.40/h: ~$5
  * A100 at ~$1.50/h: ~$3.30
  * RTX 4090 at ~$0.40/h: ~$1.70

  So consumer or Ampere cards could cut the plan to ~$150–300, or buy the 500M run.
* **Caveats:**
  * Prime currently lists only H200 pods; availability changes, so I'd poll for cheaper cards before launching.
  * Newer Blackwell cards (RTX 5090 / RTX PRO 6000) may lack prebuilt Mamba kernels; the fallback still works.
  * The speedrun is timed on 8×H100 officially. On other hardware we compare tokens and FLOPs to target and re-time
    modded-nanogpt's script on the same card.

## 7. Implementation

Modular as described in `PLAN_full_scale.md` §9: routers, mixers, chunking, losses and optimizers are registered
components, and experiments are YAML configs. All development and testing happens on CPU before any GPU spend. I'll
ask before any single launch above ~$50.
