# What our reproduction of "From Tokens to Bytes" I & II is missing: gap analysis

*2026-10-29. Inputs: the two blog posts (local copies, every footnote, caption and outbound link read), the per-run
data behind the blog's scaling plots (page bundles `scaling-plot.DGhX5y6p.js` and `scaling-law-chart.Dv37u5gU.js`,
read as data, no page code executed), the H-Net paper (arXiv 2507.07955, HTML), dnaHNet (2602.10603) and ATDC
(2605.30080) method sections, and abstracts of every other cited or related paper. All arXiv ids below were checked
against the arXiv API. Our side: `docs/BLOG_REPRO_PLAN.md`, `docs/LADDER.md`, `docs/DATA_MIX.md`, `docs/IDEAS.md`,
`logbook/EXPERIMENTS.md`, `logbook/WORKLOG.md`, `logbook/db/experiments.csv`, `configs/ladder/*`, `hnet/model/hnet.py`,
`hnet/optim/{muon,groups}.py`, `hnet/config.py`. Fetch budget used: 17 of ~30.*

## TL;DR

1. **Every run we have made is shorter than the blog's shortest run, and in the blog's own data the H-Net loses in
   its shortest runs.** The blog's ladders start at 1e18 FLOPs. Its plot bundle holds all 30 BPE and 30 H-Net
   iso-FLOP runs (table below). At 1e18, its two most data-starved H-Net runs (≈0.6–0.8B BPE-token-equivalents,
   3–8 tokens per backbone parameter) **lose** to BPE by 0.015 and 0.038 bpb. The H-Net's lead grows with tokens per
   parameter: −0.014 at ≈21 tokens/param, up to **−0.063 on the L04-sized rung** (D = 256, the same rung as our L04)
   at ≈4.4B tokens. Our 1e16 runs see 32–119M tokens in 243–908 optimizer steps. Our 1e17 runs see 0.15–0.52B
   tokens. Our router "adopts" the hierarchy between steps 200 and 600. So our ladder sits in a regime the blog never
   sampled, and where its trend points against the H-Net. (Token counts are my estimates: 6·N FLOPs per token, D
   recovered from the ladder rule.)
2. **The blog's fits cannot be checked below 1e18.** Its fit is L = E + K·(C/1e20)^−g, with E and g *shared* between
   BPE and H-Net; only K differs. That makes the multiplier a constant (1.28x) by construction. Extrapolated to our
   budgets, the fit predicts an H-Net lead of only ≈0.03 bpb at 1e16 and ≈0.02 bpb at 1e17. Our measured *deficit*
   of 0.18–0.34 bpb is therefore not something the blog's regime effects can explain. Its biggest small-data penalty
   is 0.04 bpb. The other ≈0.15–0.3 points to our protocol and implementation.
3. **The most direct protocol gap is the vocabulary, which sets the outer-network budget.** The blog's BPE uses a 128k
   vocabulary, and the H-Net's encoder and decoder are FLOP-matched to that embedding + LM head. At L04 the head is
   ≈78% of training FLOPs, ≈4x the outer budget our 32k-vocab ladder gives (48%). At L08 it is ≈53% vs our 22%. The
   blog's small-rung wins are partly against a BPE model that spends most of its compute on a 128k head.
4. **Our LR transfer adds a knob the blog does not describe.** We stack the paper's Adam-derived per-stage
   multipliers (`lr_multiplier: auto`, outer ×2.35 at L04) and Muon's sqrt(rows/cols) scale on top of BPE's
   Muon LR. That puts the outer Mamba projections at roughly 2.3–5.5x BPE's step size. The H-Net's own optimum is
   half of BPE's LR (0.01 vs 0.02). Transferring BPE's LR, as the blog does, costs us ≈0.04 at L04.
5. **Code is the better small-compute testbed, and per-source evaluation is free.** At 1e18 the blog's per-language
   gaps are 2–4x the overall gap: C −0.044, Go −0.077, against −0.018 on validation. The bpb of correct OLMES answers
   favours the H-Net by 0.12–0.17 on the smallest rungs. We have per-source validation sets (`mix_v1`, 41 sets) and
   have not scored them. DNA is a clean prior-free testbed for the router, but the blog's 6x is against an
   *uncompressed* nucleotide Transformer. On text the comparable figure (H-Net vs Byte Transformer) is 9.2x, so the
   DNA numbers do not show that learned chunking beats fixed tokenization.
6. **The cheapest decisive step is a bisection between our tiny parity result and the ladder.** EXP-024 matches BPE
   (1.635 vs 1.634). The 1e16 L04 ladder rung trails by 0.24–0.29. Between the two, the following all change at once:
   steps (1763 → 728), batch (131k → 545k bytes), optimizer (AdamW + multipliers → Muon at BPE's LR + auto
   multipliers), data (FineWeb-Edu → mix), ratio (6 → 4.145) and baseline (GPT-2 50k → in-corpus 32k BPE). Changing
   one at a time costs 6 tiny runs, about $5.

## What the blog states (beyond docs/BLOG_REPRO_PLAN.md)

- **Ladder recipe (Part I):** "modern scaling recipes such as Kimi and MAI" (Kimi K3, arXiv 2607.24653; MAI-Thinking-1
  report). Rungs defined by L, D ≈ 75 L in multiples of 128, heads = L at 128 dims, SwiGLU 3.6 D. Muon and WSD with
  fixed-fraction warmup and decay. Batch-size and LR *scaling laws* fitted on selected rungs and extrapolated.
  Iso-FLOP fits. Only causal attention FLOPs counted, LM head included. In-corpus 128k BPE, 4.57 bytes/token (Llama-3:
  4.45), worth ≈1.1x.
- **H-Net conversion:** the inner network is the BPE Transformer unchanged. The outer network is "FLOP- and
  parameter-matched" to the embedding + LM head. Compression is matched to 4.57 "by using a strong auxiliary loss".
  BPE's batch size and LR are transferred. "Setting [hyperparameters] equal to a BPE model is suboptimal." The ladder
  "can be viewed as being very favorable to the BPE Transformer baseline".
- **The improved H-Net vs LH-Net:** LH-Net ("lazy" or "lookahead") "draw[s] chunk boundaries only after it sees the
  start of the next chunk", which "negatively impact[s]" next-byte prediction and multi-stage hierarchies. Footnote 2
  says BPE also looks ahead implicitly. The improved model's stage-0 cuts follow morphemes (`track ers`,
  `wonder ed`), and it segments CJK correctly where a multi-stage LH-Net does not. Footnote 1 calls gated attention,
  MoE and hybrid SSM / linear attention "orthogonal" and says they "synergize". The architecture is to be released
  "over time". **There is no code release:** `cartesia-ai` on GitHub has no H-Net repository, and goombalab/hnet still
  has no training code (issue #3, open since Jul 2025; the maintainer said "a few (or several) weeks").
- **Part I per-component numbers:** code multipliers of 1.49x (Markdown) to 2.40x (Go), median 2.04x. Out-of-mix code:
  2.19x (Haskell) to 5.37x (Prolog), median 3.10x. The H-Net compresses almost every code language *more* than BPE,
  so it spends less compute on code (Fig. 6).
- **SmolLM3 run:** AdamW (0.9, 0.95), weight decay 0.1, clipping 1.0, peak LR 2e-4, WSD with 5B token-equivalents of
  warmup and a 15% cooldown, all identical to the baseline. Stage 0 has 4+4 Mamba-2 layers. Stage 1 has one SWA block
  plus 4+4 Mamba-2 layers. The main network has 40 layers. Targets are 4x and 2x (realized ≈7.6x). The outer stages
  take 43% of FLOPs (SmolLM3's embedding + head take 8%). FLOPs/byte are matched within 2%.
- **DNA:** a 1:6 H-Net vs a nucleotide Transformer on OpenGenome GTDB (≈274B nt, 8192-nt segments). Scaling covers
  10^18.5 to 10^20.5 FLOPs, with multipliers of 5.95x, 5.83x and 7.06x. The closest public description of a Cartesia
  or Goomba DNA H-Net is **dnaHNet (arXiv 2602.10603, co-author S. Hwang)**: GTDB prokaryotes, 10M–1B parameters.
  Its recipe: α ≈ 0.01, because α ≥ 0.03 was "overly aggressive"; ≈30% of parameters in the encoder/decoder (text
  H-Nets use ≈15%); outer, middle and inner LR multipliers of 2.0, 1.5 and 1.0; targets (3,2); compute-optimal H-Nets
  train on far more tokens than Chinchilla suggests (140B vs 68B for StripedHyena2 at 8e19).
- **Part II:** BLT-like is worse than BPE at 1e18 (0.9591 vs 0.9503). Table 2 (entropy proxy vs frozen H-Net
  boundaries vs end-to-end) starts at 1e19, and its effects are 0.002–0.016 bpb. Loss-to-loss curves are implied by
  the fitted frontiers, and off-frontier runs fall on them within ≈1% RMSE. Footnote 5 says BLT's own architecture
  differs and "could matter especially when learning chunks end-to-end".
- **Links:** BitterLesson essay; H-Net paper and goombalab blog (hnet-past); Kimi Linear (2510.26692); Attention
  Residuals (2603.15031); Kimi K3 (2607.24653); MAI pdf; Paloma (2312.10523); OLMES; SmolLM3 blog and card; JiT
  (2511.13720); Marin / Delphi; LLMs on the Line (2502.12120); prompt-boundary fix (2506.14123); Tokenization Is More
  Than Compression (2402.18376); tiktoken. **The blog does not link** BLT, SpaceByte, MegaByte, MrT5, AU-Net or Muon.
  I verified those ids separately (sources list).

### The blog's per-run ladder data (from the page's chart bundle; bold = iso-FLOP optimum)

Validation bpb on held-out pretraining data. The OLMES column is the bpb of correct answers. Token counts are my
estimates: FLOPs / (6 × (backbone + LM-head params)), with D recovered from the ladder rule and attention ignored.

| budget (FLOPs) | BPE rung (total params, 128k vocab) | ≈ BPE tokens | ≈ tokens / backbone param | BPE val bpb | H-Net params | H-Net val bpb | H-Net − BPE (val) | H-Net − BPE (OLMES) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1e18 | 71M | 4.37B | 819 | 1.0292 | 42M | 0.9659 | -0.0633 | -0.1695 |
| 1e18 | 115M | 2.53B | 154 | 0.9710 | 100M | **0.9321** | -0.0389 | -0.1245 |
| 1e18 | 221M | 1.20B | 21 | **0.9503** | 204M | 0.9367 | -0.0136 | -0.1159 |
| 1e18 | 301M | 0.82B | 8 | 0.9574 | 290M | 0.9728 | +0.0154 | -0.0512 |
| 1e18 | 399M | 0.59B | 3 | 0.9739 | 399M | 1.0114 | +0.0375 | -0.0101 |
| 1e18.5 | 115M | 8.02B | 486 | 0.9289 | 100M | 0.8764 | -0.0525 | -0.1406 |
| 1e18.5 | 221M | 3.79B | 67 | 0.8845 | 204M | **0.8600** | -0.0245 | -0.0829 |
| 1e18.5 | 301M | 2.60B | 25 | **0.8769** | 290M | 0.8642 | -0.0127 | -0.0670 |
| 1e18.5 | 399M | 1.86B | 11 | 0.8799 | 399M | 0.8798 | -0.0001 | -0.0441 |
| 1e18.5 | 525M | 1.34B | 5 | 0.8885 | 492M | 0.8890 | +0.0004 | -0.0466 |
| 1e19 | 221M | 12.00B | 211 | 0.8431 | 204M | 0.8122 | -0.0309 | -0.0901 |
| 1e19 | 301M | 8.23B | 79 | 0.8230 | 290M | 0.8036 | -0.0194 | -0.0846 |
| 1e19 | 399M | 5.87B | 35 | 0.8164 | 399M | **0.8000** | -0.0164 | -0.0750 |
| 1e19 | 525M | 4.23B | 16 | **0.8152** | 492M | 0.8035 | -0.0118 | -0.0699 |
| 1e19 | 673M | 3.17B | 8 | 0.8182 | 655M | 0.8151 | -0.0031 | -0.0422 |
| 1e19.5 | 399M | 18.55B | 110 | 0.7729 | 399M | 0.7527 | -0.0202 | -0.0699 |
| 1e19.5 | 525M | 13.39B | 51 | 0.7644 | 492M | 0.7499 | -0.0145 | -0.0479 |
| 1e19.5 | 673M | 10.03B | 27 | **0.7604** | 655M | **0.7496** | -0.0108 | -0.0504 |
| 1e19.5 | 859M | 7.77B | 16 | 0.7609 | 861M | 0.7508 | -0.0101 | -0.0504 |
| 1e19.5 | 1216M | 5.17B | 6 | 0.7652 | 1209M | 0.7603 | -0.0049 | -0.0454 |
| 1e20 | 525M | 42.34B | 161 | 0.7305 | 492M | 0.7142 | -0.0163 | -0.0450 |
| 1e20 | 673M | 31.73B | 84 | 0.7212 | 655M | 0.7067 | -0.0145 | -0.0426 |
| 1e20 | 859M | 24.57B | 49 | 0.7157 | 861M | 0.7042 | -0.0115 | -0.0475 |
| 1e20 | 1216M | 16.36B | 20 | **0.7123** | 1209M | **0.7021** | -0.0102 | -0.0469 |
| 1e20 | 1500M | 12.95B | 12 | 0.7137 | 1522M | 0.7086 | -0.0052 | -0.0452 |
| 1e20.5 | 1216M | 51.72B | 63 | 0.6760 | 1209M | 0.6671 | -0.0089 | -0.0427 |
| 1e20.5 | 1500M | 40.96B | 38 | 0.6726 | 1522M | 0.6655 | -0.0071 | -0.0479 |
| 1e20.5 | 1814M | 33.27B | 25 | 0.6714 | 1803M | 0.6656 | -0.0058 | -0.0304 |
| 1e20.5 | 2192M | 27.09B | 16 | **0.6708** | 2215M | **0.6644** | -0.0064 | -0.0381 |
| 1e20.5 | 2625M | 22.31B | 11 | 0.6721 | 2615M | 0.6736 | +0.0015 | -0.0276 |

What this table says:

- **The H-Net's validation lead shrinks, or reverses, as tokens per parameter fall.** It loses at ≤≈10
  tokens/param at 1e18 and ties at 1e18.5. On OLMES it wins in every run.
- **The H-Net's optimum sits one rung smaller than BPE's** at 1e18, 1e18.5 and 1e19 (100M vs 221M; 204M vs 301M;
  399M vs 525M). It wants more data per parameter, as dnaHNet also reports. Our 1e16 result (H-Net best at L03, BPE
  at L04) is the same pattern, not "the opposite of the blog" as `logbook/WORKLOG.md` puts it.
- **The H-Net is not parameter-matched at small rungs.** It has 42/100/204/290/399M against BPE's
  71/115/221/301/399M. Even so, the H-Net carries far more parameters than its FLOP budget would buy as dense
  byte-rate layers. At L08: 148M outer parameters vs ≈18M dense parameters affordable from the head's 36 MFLOPs/byte.
  So the improved H-Net very likely has a large FLOP-free parameter store (tables, MoE or similar) in its outer
  network. This is my inference; the blog does not say. Our hashed chunk-vocabulary and n-gram tables are one
  plausible reading.
- **Per-language code fits share this form.** At 1e18: C 0.5665 → 0.5224, C++ 0.6084 → 0.5659, Go 0.8285 → 0.7514,
  Java 0.4820 → 0.4410 (BPE → H-Net).
- **DNA has the same shared-exponent form.** At 10^18.5: Transformer 1.6923 vs H-Net 1.5745 (−0.118 bpb).

## Gap table

| recipe element | blog / cited paper | ours | match? | expected impact of the mismatch at 1e16–1e17 |
|---|---|---|---|---|
| Architecture | 1-stage; inner = the BPE rung's Transformer unchanged | same (`hnet_LNN`: main = identical Transformer) | yes | — |
| Boundary semantics | improved H-Net is causal, not lookahead; viewer data show end-of-chunk cuts (EXP-001) | end-of-chunk routers | yes, as far as is known | — |
| Router function | unpublished. The paper's ablation: cosine-similarity routing beats "direct probability prediction from individual inputs" for stability and loss | `end_mlp` (direct prediction from the encoder state) | unknown / partly against the paper's ablation | small to medium. The seed lottery may be partly router form; EXP-002 could not rank routers (spread 0.05) |
| Compression control | "strong auxiliary loss" to 4.57 (soft). Paper α = 0.03; dnaHNet α ≈ 0.01 on DNA; ATDC ramps the target from low to high compression | rank decider (top 1/N over the batch in training, running threshold at eval); ratio loss inert | **no** | small to medium: train/eval decision mismatch, batch coupling, and the router gets no compression gradient. Unmeasured at ladder scale |
| Compression target | 4.57 = BPE's bytes/token | 4.145 = our BPE's (realized 4.126) | yes in spirit | — |
| Outer-network FLOP budget | matched to embedding + head of a **128k** vocab: ≈78% of L04 training FLOPs, ≈53% at L08 | matched to a **32k** head: 48% at L04, 22% at L08 | **no** | **large at small rungs.** Our encoder/decoder get 4x (L04) / 2.4x (L08) less compute relative to the main network than the blog's, and our BPE wastes less on its head. Plausibly 0.05–0.15 bpb |
| Parameter matching | "FLOP- and parameter-matched"; the bundle shows H-Net 42–399M vs BPE 71–399M; implies FLOP-free outer parameters | exact, via hashed chunk-vocab + n-gram tables | approximately | small: our tables gave 0.07–0.14 at tiny scale; the blog's store is unknown |
| Normalization / init | paper: RMSNorm closing each network; residual projection "initialized close to 0", "may be less important" with LR modulation | as in the paper; zero-init weight, but the `nn.Linear` bias is live | yes | small |
| Optimizer | Muon + WSD with fixed-fraction warmup/decay for all ladder runs | Muon (hidden matrices) + Adam (embeddings, tables, router, norms, conv, A/D/dt) at a fixed, **untuned** 0.003; WSD 10% / 20% 1−sqrt | mostly | small to medium (the Adam-part LR on 64% of parameters is untuned) |
| LR transfer | BPE's batch and LR transferred directly; no per-stage multipliers mentioned for the ladder; the SmolLM3 run used identical AdamW settings | BPE Muon LR 0.02 **× paper's Adam-derived `auto` multipliers** (outer 2.35x at L04) × Muon sqrt(rows/cols); the H-Net's own optimum is 0.01 | **no** (extra knob) | medium: 0.04 measured from the transfer alone; the multiplier stacking is untested |
| Batch size | batch-size scaling law fitted on selected rungs (values unpublished) | 64 windows (0.13M tokens) at 1e16, ∝ C^(1/3); 4x the tiny runs' batch | unknown | medium to large via step count (next row) |
| Steps / tokens at the smallest rung | shortest blog run ≈0.6B tokens; the L04-sized rung ≈4.4B tokens at 1e18; H-Net loses at ≤≈10 tokens/param | 1e16: 32–119M tokens, 243–908 steps; 1e17: 0.15–0.52B tokens, 585–1974 steps; adoption at steps 200–600 | **no** (below the blog's range) | **large.** The blog's data show up to +0.04 against the H-Net at 0.6B tokens; with adoption taking a third to most of a 728-step run, a 0.1+ penalty is plausible |
| Sequence length | ladder unpublished (paper: 8192 bytes) | 8512 bytes / 2048 tokens | ≈ | small |
| Data mix | 70/20/5/5 by bytes; Stack v2 (15 languages), FineWeb2-**HQ**, FineMath / InfiWebMath | same shares; github-code-clean (no Swift), FineWeb-2 (not HQ) | ≈ | small for English bpb; affects the code and multilingual comparisons |
| Tokenizer | in-corpus 128k BPE, Llama-3 regex, 4.57 bytes/token | in-corpus 32k, same regex, 4.13 bytes/token | **no** | see outer-network budget; also a stronger small-rung BPE baseline |
| Priors | none (the H-Net learns word-like chunks from scratch) | `hnet` prior-free (seed lottery) and `hnet_ws` (10% whitespace warm start) | partly | — |
| Iso-FLOP grid | 5 rungs per half-decade from 1e18; the H-Net optimum sits one rung below BPE's | 4–5 rungs centred on BPE's Chinchilla prior; the H-Net optimum sits at the grid's edge (L03 at 1e16) | **no** | small to medium: the H-Net frontier point is not bracketed, so its measured gap is an upper bound |
| Evaluation | validation bpb on the mix; per-source and per-language fits; Paloma, OLMES (bpb of correct answers) | 10 MB mix val + paper7; per-source val sets exist but are not scored | **no** | none on the gap itself, but it hides where the H-Net should win first (code) |
| FLOPs convention | causal attention only, LM head counted | causal attention, LM head counted, chunk-vocab head counted, lookups free | yes (LADDER.md's "not the blog's half-cost causal" is not a real difference) | — |
| Fit form | L = E + K (C/1e20)^−g; E and g shared between architectures, so the multiplier is constant by construction | not fitted yet | — | the blog's 1.28x is not a statement about 1e16 |
| Noise floor | one run per point; frontier differences of 0.01–0.02 | prior-free seed spread 0.03–0.07 bpb | — | a 1.28x multiplier (≈0.01–0.02 bpb) cannot be resolved at our noise level without seeds |

## Ranked missing ingredients, each with a cheap experiment

The ranking is by likelihood × size of the share of our 0.2–0.3 bpb deficit each could explain. Step 0 runs before
all of them.

**Step 0 (diagnostic, about $5): bisect tiny parity against the ladder gap.** Start from EXP-024 `prior` (1.635, at
BPE parity). Move it toward `ladder_1e16_L04_hnet_ws` (0.24 behind) one factor at a time, at ≈3e15 FLOPs with BPE
rerun on the same change: (i) data FineWeb-Edu → mix_v1; (ii) AdamW 1.5e-3 + paper multipliers → Muon 0.02 + `auto`
multipliers + Adam part 0.003; (iii) batch 131k → 545k bytes, i.e. 2.4x fewer steps; (iv) ratio 6 → 4.145 with
re-sized tables; (v) GPT-2 50k baseline → in-corpus 32k. The factor that opens most of the gap points to which item
below to work on first.

1. **Training horizon / optimizer steps.** The blog has no run under ≈0.6B tokens. Its H-Net loses at low tokens per
   parameter, and our router needs 200–600 steps to commit.
   - *Experiment:* at a fixed 1e16 FLOPs on L03 and L04, sweep batch 16 / 32 / 64 windows (4x / 2x / 1x steps) for
     bpe, hnet and hnet_ws: 18 runs, ≈ the cost of the 1e16 ladder (<$10).
   - *Prediction if this is the cause:* the H-Net's deficit shrinks by more than BPE gains from the extra steps.
   - *The decisive version:* one 1e18 pair on the blog's own smallest rung (L04, ≈4–10B tokens). There the blog's
     H-Net won by 0.063 bpb. Cost ≈ 14 A100-h (H-Net at 6.5% MFU) + 8 A100-h (BPE), about $30 spot.
2. **Outer-network budget set by a 128k vocabulary (protocol).** Regenerate L04 and L06 with a 128k in-corpus BPE (or
   the Llama-3 tokenizer, to save the tokenizer build), with the H-Net's outer network FLOP-matched to that head
   (d0 / layers about 4x larger at L04). Run bpe and hnet_ws at 1e16 and 1e17: 8 runs, a few dollars.
   - *Prediction:* BPE gets worse at small rungs (the blog's 71M rung at 1e18 is its worst BPE point), and the H-Net
     gets better.
   - This is what "the H-Net beats the matched BPE rung" means in the blog's protocol. Keep our 32k ladder as the
     stricter, fairer comparison.
3. **LR transfer under Muon.** At L04, 1e16, hnet_ws, Muon 0.02, try: `lr_multiplier: null`; multipliers applied to
   the Adam part only; Adam-part LR {0.001, 0.01}; and the Mamba `in_proj` on Adam instead of Muon (Muon
   orthogonalizes the fused z / x / B / C / dt projection as one matrix). That is 5 runs.
   - *Prediction:* "multipliers off under Muon" recovers most of the 0.04 we lose by transferring BPE's LR, and lets
     the H-Net use BPE's LR as the blog does.
4. **Compression control and router form.** Replace the rank decider with threshold decisions plus a strong ratio
   loss (α = 1; EXP-010 held compression within 0.8% at tiny scale), and try a cosine-style end-of-chunk router (the
   paper's ablation favours similarity routing). Optionally add ATDC's compression ramp (start below the target and
   raise it).
   - *Free check first:* on existing ladder checkpoints, measure realized bytes per chunk *at eval* (running
     threshold) per source against training's batch quantile. A mismatch means our eval bpb is measured on a
     different tokenizer than the one trained.
   - Then 3 runs at L04 1e16, prior-free, with 3 seeds where affordable: this is also the seed-lottery lever.
5. **Evaluation and frontier protocol.**
   - (a) Score every existing ladder checkpoint on the per-source and per-language validation sets (code, held-out
     code, multilingual, math) at no training cost. The blog's per-language gaps are 2–4x the overall gap.
   - (b) Add an L02 rung (or longer L03 runs) at 1e16 so the H-Net's iso-FLOP minimum is bracketed: it sits one rung
     below BPE's in the blog's data and at our grid edge.
   - (c) Use ≥2 seeds for H-Net frontier points. The multiplier we are trying to see (≈0.02 bpb) is below our seed
     spread.

Expected sizes, as guesses calibrated on our own numbers: horizon 0.05–0.15, vocabulary budget 0.05–0.15, LR 0.04–0.1,
decider/router ≤0.03 on the mean but most of the seed variance, frontier bracketing 0.02–0.05. Items 1 and 2 together
can plausibly account for a 0.2–0.3 gap; neither alone is likely to.

## Direct answers

**(a) Is the blog's smallest-compute claim within reach, and where does the H-Net start winning?** Yes, at 1e18 and
not below.

- The blog's smallest budget is 1e18 FLOPs (DNA: 10^18.5).
- The blog claims a win "at all model sizes" from there, with a constant 1.28x multiplier. The data do not quite
  support that: at 1e18 the two largest rungs lose on validation bpb (+0.015, +0.038), and at 1e18.5 the 399M and
  492M rungs tie.
- The wins are at the iso-FLOP optimum (−0.018 at 1e18) and on small, data-rich rungs (−0.063 on the L04-sized rung
  at ≈4.4B tokens).
- The blog says nothing about 1e16–1e17. Its shared-exponent fit would extrapolate to a ≈0.02–0.03 lead there, but
  that is an assumption of the fit, not a measurement.
- Within our budget:
  - A single 1e18 pair at L04 or L05 costs ≈$30–35 on A100 spot and reproduces the blog's smallest data point
    directly.
  - The planned full 1e18 iso-FLOP sweep (≈79 A100-h, ≈$114) reproduces the blog's first frontier point.
  - 1e16–1e17 runs cannot confirm or refute the blog; they test a regime it never sampled.

**(b) The five most likely missing ingredients, ranked:**
1. training horizon / steps;
2. the outer-network budget from the 128k-vocabulary protocol;
3. LR transfer under Muon (stacked Adam-derived multipliers);
4. compression control (rank decider vs a strong auxiliary loss) and router form;
5. evaluation and frontier protocol (per-source scoring, bracketing the H-Net optimum, seeds).

The experiments are above, preceded by the step-0 bisection.

**(c) Is code-heavy or non-text data a better testbed for learned chunking at small scale?**
- *Code: yes for evaluation, probably yes for training.*
  - The blog's H-Net gains on code are 1.5–2.4x in distribution and 2.2–5.4x out of distribution, against 1.28x
    overall. In absolute terms that is 0.04–0.08 bpb at 1e18 vs 0.018 overall.
  - It also compresses code more than BPE does, which suggests learned units fit code better than BPE's
    pretokenized ones.
  - "Patch where the text is new" (2610.22100) reports that H-Net finds structure in math and code on its own, and
    that chunking-rule effects appear from ≈12M parameters on math.
  - *Caveat:* the blog never trained code-only. Start with per-source evaluation of the mix, then a code-heavy
    (e.g. 50% code) ladder rung, where the whitespace warm start is also the wrong prior. Tests IDEAS.md A5.
- *DNA: a good mechanism testbed, weak evidence for "learned beats fixed".*
  - It has no whitespace prior to lean on, a known ground truth (codon periodicity: dnaHNet's stage 1 learns triplet
    periodicity), small vocabularies and cheap models.
  - The blog shows a 0.118 bpb gap already at 10^18.5, and dnaHNet shows H-Net ahead of StripedHyena2 and
    Transformers from 8e18.
  - But both compare against *uncompressed* nucleotide models; the text analogue is 9.2x vs the Byte Transformer.
  - A DNA testbed must include fixed-stride (3-mer / 6-mer pooling) and DNA-BPE baselines at equal FLOPs to say
    anything about learned chunking. LDARNet (2606.04552) reports learned beating fixed-grid boundaries by up to
    14 points at identical compute, but for masked-LM fine-tuning, not pretraining bpb.
  - Use dnaHNet's recipe: α ≈ 0.01, ≈30% outer parameters, LR multipliers 2.0 / 1.5 / 1.0, and the expectation of
    more data per parameter.

## Corrections and omissions in docs/BLOG_REPRO_PLAN.md

- **I.1 is wrong as stated.** The data show the H-Net losing or tying on validation bpb in the low-tokens-per-parameter
  runs at 1e18, 1e18.5 and 1e20.5. It wins at the frontier and on every OLMES point.
- **I.2: the range is 1e18–10^20.5 (fit points), not 1e18–1e21** (1e21 is only the axis). The constant 1.28x is
  forced by a fit with shared E and exponent.
- **Reachability row "I.1, I.2: yes, at 1e16–1e18" is too optimistic.** 1e16–1e17 lies below every blog run (≥0.6B
  tokens). Only 1e18 overlaps.
- **II.4 "yes, cheap" is cheap in compute but not in signal.** Table 2's effects (0.002–0.016 bpb, at ≥1e19) are below
  our seed spread (0.03–0.07).
- **"Outer FLOP- and parameter-matched": the bundle shows H-Net totals 29–0M below BPE at the small rungs**, and
  implies a FLOP-free parameter store. Our exact matching with hash tables is a choice, not the blog's documented
  recipe.
- **Missing deviation: 32k vs 128k vocabulary** changes the outer-network FLOP budget by 2.4–4x at our rungs (not only
  the baseline's head share).
- **Missing deviation: rank decider vs "strong auxiliary loss"**, and the `auto` LR multipliers under Muon, which the
  blog's "hyperparameters directly transferred" does not include.
- **Missing: the H-Net's iso-FLOP optimum is one rung smaller than BPE's** in the blog's own data (and in dnaHNet). The
  WORKLOG's "opposite of the blog's claim" should be corrected, and the 1e16 grid extended downward.
- **I.10 DNA:** the multiplier is vs an uncompressed nucleotide Transformer, and the closest public recipe is dnaHNet
  (2602.10603). The cheap DNA plan needs fixed-stride and BPE baselines to test learned chunking.
- **Missing: the per-run data of both posts are in the page bundles** (used above). Our 1e18 runs can be compared
  run-for-run, not only frontier-for-frontier.
- **The compute estimate assumes 25–35% MFU.** Measured H-Net MFU is 6.5–24% (WORKLOG), so the 1e18 sweep is ≈79
  A100-h, not ≈8 H100-h.

## Sources

Blog and page data
- https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1
- https://www.cartesia.ai/blog/from-tokens-to-bytes-part-2
- Per-run data: https://www.cartesia.ai/_astro/scaling-plot.DGhX5y6p.js and
  https://www.cartesia.ai/_astro/scaling-law-chart.Dv37u5gU.js (read as data)

Papers (ids verified with the arXiv API)
- H-Net, Hwang, Wang, Gu: https://arxiv.org/abs/2507.07955 (method read from https://arxiv.org/html/2507.07955v2)
- dnaHNet: https://arxiv.org/abs/2602.10603 (recipe read from the v3 HTML)
- ATDC, compression curriculum: https://arxiv.org/abs/2605.30080
- H-Net++: https://arxiv.org/abs/2508.05628
- What Tokens are Learned when Tokenization is Optimized Jointly with Language Modeling?:
  https://arxiv.org/abs/2608.17325
- Patch where the text is new: https://arxiv.org/abs/2610.22100
- GeneZip: https://arxiv.org/abs/2602.17739
- LDARNet: https://arxiv.org/abs/2606.04552
- DC-DiT: https://arxiv.org/abs/2603.06351
- ConceptMoE: https://arxiv.org/abs/2601.21420
- Bolmo: https://arxiv.org/abs/2512.15586
- BLT: https://arxiv.org/abs/2412.09871
- SpaceByte: https://arxiv.org/abs/2404.14408
- MegaByte: https://arxiv.org/abs/2305.07185
- MrT5: https://arxiv.org/abs/2410.20771
- AU-Net: https://arxiv.org/abs/2506.14761
- LLMs on the Line: https://arxiv.org/abs/2502.12120
- Sampling from your LM one byte at a time (prompt-boundary fix): https://arxiv.org/abs/2506.14123
- Tokenization Is More Than Compression: https://arxiv.org/abs/2402.18376
- Kimi Linear / KDA: https://arxiv.org/abs/2510.26692
- Attention Residuals: https://arxiv.org/abs/2603.15031
- Kimi K3: https://arxiv.org/abs/2607.24653
- Paloma: https://arxiv.org/abs/2312.10523
- JiT: https://arxiv.org/abs/2511.13720

Other links cited by the blog (not fetched)
- https://goombalab.github.io/blog/2025/hnet-past/
- https://github.com/allenai/olmes
- https://huggingface.co/blog/smollm3
- https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- https://microsoft.ai/pdf/mai-thinking-1.pdf
- https://openathena.ai/blog/delphi/
- http://www.incompleteideas.net/IncIdeas/BitterLesson.html

Code and issues
- https://github.com/goombalab/hnet/issues (20 issues; none on training instability or routing collapse)
- https://github.com/goombalab/hnet/issues/3 (no training code; third-party https://github.com/main-horse/hnet-impl
  "tiny runs seem to train successfully")
- https://github.com/orgs/cartesia-ai/repositories?q=hnet (no H-Net repository)

Searches that found no independent reproduction reporting a small-scale training recipe or failure modes: web search
for "H-Net dynamic chunking reproduction small scale router collapse"; the arXiv API (abs:"dynamic chunking" OR
abs:"H-Net").
