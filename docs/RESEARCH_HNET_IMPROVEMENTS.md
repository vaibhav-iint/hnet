# What changed between H-Net (July 2025) and Cartesia's "improved H-Net" (Oct 2026)

*Research note, 2026-10-29.*

**Labels used throughout:**
* **[C]**: stated by Cartesia or the paper's authors (Hwang, Wang, Gu).
* **[T]**: stated by a third party.
* **[I]**: our inference.

**Sources not reached from this session:**
* openreview.net (the ICLR 2026 camera-ready) and x.com were blocked by the network allowlist.
* The `gh` API for `goombalab/hnet` was blocked; I read its issues and PRs through the web pages instead.

## 0. Bottom line

1. **Nothing about the improved H-Net is public apart from the blog.**
   * **[C]** `goombalab/hnet` has 4 commits since the paper (training utilities on 2025-07-31, an SWA KV-cache fix, boundary-visualization code on 2025-11-20). None changes the router, the losses or the architecture.
   * **[C]** No training code has been released; issue #3 is still asking for it in Aug 2026.
   * **[C]** dnaHNet (Feb 2026, co-authored by Wang and Hwang) still uses the paper's lookahead cosine router.
2. **The only disclosed change is the router's semantics: causal end-of-chunk.**
   * **[C]** The paper's router is now called "LH-Net" (lazy/lookahead).
   * **[T, our EXP-001]** In the blog's chunk-viewer data, the improved router marks the *last* byte of each chunk and decides using only bytes ≤ t.
   * **[C]** Sukjun Hwang (Jul 2025) said that naively selecting the byte before each boundary with the lookahead router "performs much worse". So the new router must be a genuinely causal predictor, not the old router shifted by one byte.
3. **Our parameter-matching inference holds for the 1-stage ladder but not for the 3.67B model (§2).**
   * **[I]** At L08, about 146M of the 164M parameters of the matched outer networks must be FLOP-free. The likely candidates are hashed byte n-gram tables or MoE.
   * **[I]** The 3.67B model's stated per-stage parameters and FLOPs/byte fit dense layers exactly, so it has no large tables. The blog itself says "the byte embedding is tiny".
4. **Every independent small-scale attempt we found loses to BPE.**
   * **[T]** main-horse ran HEBO compute-optimal sweeps from 1e16 to 1e18 FLOPs; the byte H-Net is never more compute-efficient than BPE (§4).
   * **[T]** A third-party study of 22.5M-param models (6000 steps) finds the main network is worth only about 0.07 bpb.
   * **[T, our EXP-002]** Our own ~2M-param runs trail BPE by about 0.19 bpb.
   * **[C]** Cartesia reports wins from L08 upward, on a ladder spanning roughly 1e18–1e21 FLOPs.
   * **[I]** The gap between these results most plausibly comes from four things:
     * the new router;
     * compression pinned by a strong auxiliary loss;
     * Muon;
     * parameter-heavy outer networks.

   A 128k-vocab BPE head that eats about 60% of L08's FLOPs probably adds to the gap.

## 1. Evidence on what changed (primary sources)

**Repository, `github.com/goombalab/hnet`** (cloned; branches `main`, `train-utils`, `brwa/visualize`):
* **Commit cd51d82 (2025-07-31, Brandon Wang): `hnet/utils/train.py`.**
  * It contains `load_balancing_loss`, computed per device. Its docstring says: *"This is the loss we used for all experiments. It computes the loss on each device/minibatch… It is possible that computing the loss on each example is better, or that computing the statistics over the entire (global) batch would have been better."*
  * Initialization: std 0.02, scaled by 1/√n_residuals on `out_proj`/`fc2` across the whole hierarchy; embeddings std 1.0.
  * Per-stage LR multipliers are applied to the embedding and head at stage 0.
* **Issues and PRs:**
  * [#12](https://github.com/goombalab/hnet/issues/12), Sukjun Hwang (2025-07-22): they tested selecting the previous token at each boundary (one-step look-ahead), and *"it performs much worse"*.
  * [#17](https://github.com/goombalab/hnet/issues/17), Brandon Wang (2025-08-15): removing the STE from the upsampling step *"increases the router confidence"*. Probabilities *"often dip to about 0.53 during training and then rise again"*.
  * [#9](https://github.com/goombalab/hnet/issues/9): 1-stage checkpoints put every byte of a repeated-character run into its own chunk; 2-stage checkpoints handle repeats.
  * PRs #24–#26 are third-party (loss inside forward, early-exit probes, entropy patching). All were closed without merging.
* **Goomba blog "H-Net past" (2025-07-11), `goombalab.github.io` source:**
  * Post-norms on every sub-network and the residual projection *"seem very useful—maybe essential"*.
  * A pure "sparsity loss" in place of the targeted ratio loss *"sort of worked but it didn't seem as consistent"*.
  * BLT's hash n-gram embeddings: *"We think both of these are not necessary… although I believe that they could also be incorporated into H-Net"*.
  * A hybrid main network (3:1 Mamba:attention) gives better perplexity.
  * No H-Net post has appeared since; the site's `scaling-thoughts` branch (2026-10-27) contains only an empty draft.
* **[dnaHNet, arXiv 2602.10603](https://arxiv.org/abs/2602.10603) (Wang and Hwang co-authors):**
  * Router: the paper's p_t = ½(1 − cos(W_q h_t, W_k h_{t−1})).
  * Each encoder and decoder is 4 Mamba-2 + 1 Transformer layer, and the outer networks hold about 30% of the parameters.
  * LR multipliers 2.0 / 1.5 / 1.0; α = 0.02–0.03 (lower than for text).
  * Training from scratch *"can lead to degenerate solutions where the boundary predictor either selects all positions or none"*.
  * Compute-optimal runs *"prefer training on substantially more data than… Chinchilla"*.
  * The blog's 6x DNA ladder matches this (3,2) setup.
* **Blog figures [C]:** The architecture figure's routed bytes in `The_end_of_tokens` are at indices 0, 3, 7, 10 and 15, i.e. mostly the trailing `_`. This matches end-of-chunk semantics (EXP-001).

## 2. Testing the parameter-matching inference

* **Arithmetic.** Training FLOPs/byte ≈ 6 × (active params) ÷ (bytes per position) + attention. All figures below are training FLOPs/byte.
* **L08 (d = 640, about 128k vocab):**
  * Embedding + head = 221 − 56.4 = 164.6M params.
  * The head costs 6 × 82.3M ÷ 4.57 ≈ 108 MFLOPs/byte. That is about 60% of the BPE rung's compute, against about 74 MFLOPs/byte for the backbone.
  * A dense byte-level outer network within that budget can hold only about 18M params, which leaves about 146M params to be (near-)FLOP-free.
* **L28:** 520M params to match, with a head-FLOP budget that allows about 57M dense params, so about 463M must be FLOP-free.
* **3.67B model (blog Fig. 8):**

  | Stage | Params | GFLOPs/byte | 6 × params ÷ bytes per position |
  |---|---|---|---|
  | Stage 0 | 0.22B | 1.37 | 6 × 0.22 = 1.32 |
  | Stage 1 | 0.32B | 0.52 | 6 × 0.32 ÷ ~3.8 ≈ 0.50 |
  | Main | 3.13B | 2.55 | 6 × 3.13 ÷ 7.6 ≈ 2.47 |

  Every stage is dense; there are no FLOP-free tables.
* **Verdict [I]:**
  * The ladder (and Blog II's matched suite) almost certainly adds FLOP-free parameters to the outer networks.
  * Hashed n-gram embeddings are the most natural choice: BLT-like baselines have them natively, and Blog II says the BLT-like model *"keeps the exact same hierarchical structure… as our H-Net"*. MoE is the alternative.
  * The flagship SmolLM3 comparison did not need them. It matched FLOPs/byte only, and has 3.67B params against 3.08B.

## 3. Candidate improvements

The "Strength" column rates how solid the evidence is that this matters for beating BPE.

| # | Change | Source | Strength | How to do it in our code | Expected effect |
|---|---|---|---|---|---|
| 1 | **Causal end-of-chunk router.** The chunk vector is the last byte, the decision uses bytes ≤ t, and the main network predicts the first byte of the next chunk. | [Blog I](https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1) "Aside"; EXP-001; [#12](https://github.com/goombalab/hnet/issues/12) | **Strong** that the semantics changed; **none** for the exact function | Already registered as `end_linear`, `end_mlp`, `end_cosine_shift`. Bolmo reports that end prediction *"is hard, especially since this task has to be performed by the shallow local encoder"* ([2512.15586](https://arxiv.org/abs/2512.15586) §6.1). So also try a deeper router input (MLP on the concatenated outputs of the last 2 encoder layers) and logit-bias initialization at the target ratio. | Better stem splits, no splits inside UTF-8 characters (CJK), more effective multi-stage models. Cartesia says LH-Net's lookahead *"negatively impact[s] its ability to perform next-token (byte) prediction as well as model multiple hierarchies"*. |
| 2 | **Pin compression with a "strong auxiliary loss".** | [C] blog ladder; [T] [main-horse "Intuitions"](https://main-horse.github.io/hnet/intuitions/); [ATDC 2605.30080](https://arxiv.org/abs/2605.30080) | Medium: it is a stated part of the winning recipe, but its isolated effect is unknown. main-horse found that runs whose compression spiked often had *better* loss. | Sweep α ∈ {0.03, 0.1, 0.3} with `scope: global`. Log realized bytes/position; FLOPs matching needs the realized ratio, which is the reason for pinning. | Makes FLOPs-matched comparisons valid (the ratio stays at the BPE value) and stabilizes the main network's input rate. |
| 3 | **Threshold warmup / compression control at initialization.** | [T] [main-horse "Parameterization S≥1"](https://main-horse.github.io/hnet/param-s1/); our worklog (transient collapse up to about 300 bytes/position) | Medium (one author, several runs) | For the first about 1k steps, set the selection threshold to the (1 − 1/N) quantile of p (no-grad), cosine-blended back to 0.5. Feed the ratio loss and STE the "unbiased" b = p ≥ 0.5. | Removes the early collapse ("hockey stick"). main-horse: *"when you don't see a quick initial hockey-stick… very likely your run will suck"*. Also fixes inner-length crashes and peak VRAM. |
| 4 | **No STE on the upsampling path.** | [C] Brandon Wang, [#17](https://github.com/goombalab/hnet/issues/17) | Weak to medium (anecdotal, from a maintainer) | `dc.ste: false` ablation | More decisive, bimodal p (the blog's viewer shows a bimodal p, EXP-001). |
| 5 | **Parameter-heavy, FLOP-light outer networks (hashed byte n-grams).** | [I] §2; [T] [Byte LMs: Scaling… 2610.05978](https://arxiv.org/abs/2610.05978): *"With token-superposition training and hash embeddings, byte Transformers consistently outperform subword Transformers"* (n = 3–6, 4 × 32k slots); BLT | Medium (inference plus strong third-party evidence for byte models). [C] Gu (2025) called them "not necessary". | `hnet/model/ngram.py` exists. Size the tables so total params equal BPE embedding + head (bpe_small: 77.3M). Hashing costs no matmul FLOPs. | Likely the largest small-scale lever under the blog's "params AND FLOPs" matching. It gives the byte encoder the memorized lexicon that a 50–128k embedding gives BPE. |
| 6 | **Muon + WSD, BPE hyperparameters transferred.** | [C] blog ladder (*"setting them equal to a BPE model is suboptimal"*); [T] main-horse: with LR modulation, byte H-Nets want *"a significantly lower base learning rate than a similarly sized BPE transformer"*, and μP transfer fails because routing changes with width | Medium | Our `muon` optimizer with the router in its own Adam group (`muon.router: adam`) and its own LR (main-horse's "segregated routing LR"). Sweep the H-Net LR separately (EXP-003). | A fair baseline. A lower H-Net LR may matter more than any architecture change. |
| 7 | **Batch matched in tokens at L0, not bytes, or tune BPE batch separately.** | [T] [main-horse "Scaling Laws (Byte)"](https://main-horse.github.io/hnet/scaling-laws-byte/): *"the optimal token batch size tends to be similar for both"*, so equal bytes/batch under-batches BPE | Medium (one study) | Tune BPE batch × LR at 125M. Report H-Net vs best-tuned BPE. | Prevents a baseline bias that critics will raise. |
| 8 | **Ratios: 1-stage at the BPE ratio; 2-stage (4,2), not (3,3).** | [C] blog (4.57; 4x then 2x, realized 7.6); [T] [Compute-Optimal Tokenization 2605.01188](https://arxiv.org/abs/2605.01188): optimal BLT bytes/patch is about 3.7 at 1e20 and 3.3 at 2e21 FLOPs (no hash embeddings), and *"decreases with compute"* | Medium | Our `_ratios42` configs; add a 1-stage config at our measured bytes/token (about 4.6–4.75). | Over-compression (9 bytes/position) at 125M is probably beyond the optimum. |
| 9 | **Gated attention / MoE / hybrid layers.** | [C] blog footnote 1 (*"H-Nets synergize with them"*); Gu 2025 (3:1 hybrid main is better) | Weak (claim only) | Gated attention is cheap to add to the main network and SWA layers; an MoE main network is an alternative to row 5 for parameter matching. | Orthogonal gains. Keep these out of the core comparison. |
| 10 | **Stage-0 encoder gets one attention/SWA layer; post-norms; zero-init `residual_proj`; identity-init W_q/W_k.** | [C] dnaHNet (4M + 1T); blog stage 1 = SWA + 4 Mamba-2; Gu 2025; [T] main-horse (zero-init W_res drives the early main-network push) | Established | Mostly in place (E1 = T1M4). | Stability. |
| 11 | **Compression curriculum (low → high N).** | [T] ATDC | Weak (small downstream gains, similar bpb) | `target_ratio` schedule | Smoother start; overlaps with row 3. |
| 12 | **Score-function (RL) boundary learning instead of STE.** | [T] [Dauncey & Wattenhofer 2602.13940](https://arxiv.org/abs/2602.13940): beats STE "at the 100 million parameter scale" | Weak | Research option only. | Unknown |
| ✗ | **Frozen or pretrained boundaries (BLT-like, proxy H-Net).** | [C] Blog II Table 2: end-to-end 0.8001 vs H-Net proxy 0.8139 vs entropy proxy 0.8160 bpb at 1e19 FLOPs | Strong against | Do not do this. | — |

## 4. Reported failure modes at small scale

* **[T] main-horse, *H-Net – Scaling Laws (Byte)* (Sep 2025): byte H-Net never more compute-efficient than BPE.**
  * Setup: thousands of HEBO-tuned compute-optimal runs on FineWeb-10BT, N = [1,5] or [1,3,9], AdamW, 1 GPU.
  * Best bpb, byte H-Net vs Llama-2 BPE:

    | FLOPs | Byte H-Net | BPE |
    |---|---|---|
    | 1e17 | 1.296 | 1.302 |
    | 3e17 | 1.213 | 1.206 |
    | 1e18 | 1.127 | 1.113 |

  * In his words, *"none of the H-Net (S>0) runs ever achieve more compute efficiency than L2"*, with *"the dreaded scaling curve bounce"* as FLOPs grew.
  * He also argues the paper's BPE learning rate was 2–10x too low.
* **[T] main-horse, training dynamics and hyperparameter sweeps:**
  * Compression first collapses toward 0, then overshoots and settles within about 500 steps.
  * A larger init σ in the encoder gives correlated adjacent states, so compression at initialization is "negligible" and *"generally never goes back up"*.
  * Hyperparameter sweeps under about 5000 steps are *"misleading"*.
  * He concludes that end-to-end H-Nets are a poor fit for μP.
* **[T] [AdaptiveChunking pilots](https://huggingface.co/AdaptiveChunking/hnet-chunking-pilots) (31 runs, 22.5M params, 6000 steps):**
  * *"The chunk-level main network is worth only ~0.07 BPB at this scale"*.
  * *"segmentation crystallises after capability"*.
* **[C] dnaHNet:** an all-or-none router collapse.
* **[T, ours]:**
  * EXP-002 (~2M params, about 2.7e15 FLOPs): H-Net at 1.82–1.83 bpb vs BPE at 1.635.
  * Every run shows the transient early collapse.
* **[I] Why Cartesia can win at L08 when main-horse lost at 1e18:**
  * The L08 BPE head (128k × 640) costs about 60% of the BPE rung's FLOPs. The H-Net gets that whole budget for byte-level processing, plus (we infer) about 146M FLOP-free params.
  * main-horse used a 50k GPT-2 vocabulary, unconstrained compression drift, AdamW and the LH-Net router.

## 5. What to change in our setup to reach the "H-Net beats BPE" regime

1. **Stop drawing go/no-go conclusions below about 1e18 training FLOPs and about 5k steps.**
   * Tiny runs (2–10M params, at most 1e16 FLOPs) are only useful for debugging and comparing router variants against each other.
   * Every external datapoint and our EXP-002 lose there.
   * Our A-series (125M, 10B tokens, about 8e18 FLOPs) and the B-ablations (about 2e18) are in range.
2. **Add a blog-protocol 1-stage ladder as the primary comparison (2–3 rungs, e.g. 56M/85M/170M backbones).**
   * The main network is identical to the BPE backbone.
   * Outer networks are FLOP-matched to the head: for bpe_small, 6 × 38.6M ÷ 4.6 ≈ 50 MFLOPs/byte, i.e. about 8M dense params, for example E = M4 and D = M4 at d ≈ 384.
   * Outer networks are parameter-matched with hash n-gram tables (about 69M).
   * Compression is pinned at our tokenizer's bytes/token with a strong α plus threshold warmup.
   * Batch and LR are copied from the tuned BPE baseline (blog style), *and* tuned separately (fair style).
3. **Run the BPE baseline with a 128k vocabulary (Llama-3 or in-corpus) in addition to GPT-2's 50k.**
   * This reproduces the blog's setting, in which the BPE head is very expensive at small rungs.
   * Report both, and flag the vocabulary as a protocol choice that favors H-Net at small scale.
4. **Router:** run end-of-chunk variants with and without the STE (`dc.ste: false`), with threshold warmup and a separate router LR group. Keep `cosine_lookahead` as the LH-Net control.
5. **Ratios:** use the BPE ratio for 1-stage; use (4,2) for 2-stage, not (3,3). Re-match FLOPs from the realized ratio averaged over the whole run, not only at the end (main-horse's critique).
6. **Optimizer:** use Muon + WSD for both models. Sweep the H-Net base LR downward (EXP-003) and the BPE batch size.
7. **Metrics:**
   * Evaluate on downstream bpb (OLMES/Paloma-style answer bpb) and code subsets, not only validation bpb. Cartesia's multiplier grows from 1.28x (validation) to about 2x (downstream) and 2–5x (code).
   * Track realized bytes/position per source.
8. **Data horizon for the SmolLM3-style claim:** compare stable-phase curves out to many tokens per parameter. The 3.67B claim comes from about 1T tokens (≈ 300 tokens/param), a heavily overtrained regime. dnaHNet also found H-Nets' compute-optimal data budgets above Chinchilla's.

## 6. Open questions

* **The end router's exact function.** Is it a linear or MLP read-out of h_t? A cosine to a *predicted* next state? Does it use decoder-side boundary prediction ("fused" byte+boundary vocabulary, as in Bolmo)? Is the STE still used?
* **Parameter matching.** What exactly fills the ladder's outer networks: hash n-grams, MoE, or something else? Does the 2-stage SmolLM3 model use any of these?
* **The "strong auxiliary loss".** What α? Is it global or per-sequence? Does a hard controller replace the ratio loss? main-horse's data says forcing the ratio can hurt loss; Cartesia pins it anyway for matching.
* **Optimization.** Are the LR multipliers still used with Muon? Is the router optimized with Muon or Adam?
* **Gains per change.** How much of the reported 1.28x comes from the router change versus the ladder protocol (128k head, matched parameters)? Only an LH-Net run under the same protocol would tell. The blog shows LH-Net only qualitatively.
* **ICLR 2026 camera-ready changes.** Does the camera-ready (openreview PDF 908dc1ab…, blocked here) change anything relative to arXiv v2? It is worth checking once openreview.net is reachable.
