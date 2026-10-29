# Ideas: creative H-Net variants and tokenizer-free designs from first principles

Two independent lists were written on 2026-10-29 (the coordinator's, and a subagent's drafted before it read any
project code; both verbatim under `logbook/ideas/`), then merged here against what the measurements say. Status
column: **tested** (with the experiment), **running**, **falsifier done**, or **open**.

## What the measurements constrain (read first)

1. The router's only training signal in the paper design is the straight-through estimator on the dechunked output;
   it carries no preference between a word tokenizer and a suffix tokenizer (next-chunk CE identical, 4.9 at step 500).
2. The hierarchy's measured gain does carry that preference: 0.165 vs 0.068 bits/byte at step 500, 6× at word starts
   (`logbook/diagnostics/diag_exp023_vp_s{0,1}_step500.json`).
3. Refereeing the adoption race (chunk STE, margin-gated bypass; EXP-027) does not remove the seed lottery and
   costs 0.07 bpb; forcing early adoption of a bad tokenizer is worse than letting the bypass win.
4. Consistency (same string, same cut) is 92% at init and destroyed by training; the whitespace warm start restores
   it and gives BPE parity (1.635). A context-free router (EXP-019, last-k-bytes MLP) was worse (1.813), so
   consistency by construction alone is not enough: the window reaches into the previous word.
5. Offline teachers: 5-gram entropy spikes put 66% of cuts on word ends (consistency 60%); unigram MDL without a
   length prior yields long rare units (84% singletons). Neither beats whitespace on English.
6. At 1e16 FLOPs on the mixed corpus, both H-Nets prefer the smallest rung and trail BPE by 0.18-0.29 bpb.

## A. Router objectives (the disease, per 1-3)

| idea | mechanism | expected failure | falsifier | status |
|---|---|---|---|---|
| **A1 gain-based REINFORCE** | rerun the stage-0 decoder on the bypass alone; credit each byte's gain to its chunk; standardized advantage × log-prob of the realized decisions | variance; zero gain if the decoder ignores the hierarchy (pair with gate) | step-500 gain separates basins 2.4× (done) | **running (EXP-028)** |
| A2 segmental marginal likelihood | forward algorithm over boundaries with K-bounded units; exact posterior marginals, no STE; the EMA dechunker already is this recursion with hard inputs | all-1 / all-K units without a length prior | offline MDL probe: not a tokenizer without a prior (done) | open: add a length prior + conditional unit model, retest offline |
| A3 online MDL / unigram entropy of chunk types | hashed EMA counts of chunk buckets; minimize the unigram bits/byte of the segmentation (BPE's criterion made differentiable via the chunk STE) | long rare units again; needs the ratio pinned (we have rank decisions) | chunk_stats unigram bits/byte: word 1.78 vs suffix ? on the same text | open, cheap |
| A4 consistency as objective with hard negatives | same string → same decision across contexts, scale-free (we have the penalty: EXP-020 neutral) | satisfied by constant logits unless scale-free | EXP-020 | tested, neutral |
| A5 data-derived prior replacing whitespace | 5-gram entropy-spike curriculum via `router_prior_weight` machinery | weaker than whitespace on English; the point is code / multilingual rungs | probe done: 66% on word ends | open: run on the mix_v1 ladder rung as the `hnet_ws` alternative |
| A6 variational router with a length prior | KL(q(s)‖p(s)) replaces the ratio loss; dense gradient from step 0 | REINFORCE variance | tiny run | open |

## B. Remove the race instead of refereeing it

| idea | mechanism | expected failure | falsifier | status |
|---|---|---|---|---|
| B1 no bypass + byte n-gram memory | delete `residual_proj`; decoder sees the dechunked hierarchy + n-gram tables (+ optional 1-layer local attention) | slow first 200 steps; suffix basin with nowhere to return | 3 seeds, 1000 steps, `dechunk_rms` + whitespace alignment at 200/500/1000 | open, cheapest structural test |
| B2 narrow bypass | low-rank (8-16) residual projection: identity can pass, prediction cannot | the main network still bypassed through the encoder | tiny run | open |
| B3 mask-mixing curriculum | use whitespace masks *as the mask* with probability π(progress) → 0, router trained toward whichever mask was used; adoption from step 0 | router learns whitespace and nothing more | compare with `external` whitespace router at equal FLOPs | open |
| B4 two-timescale SSM, no boundaries | inner block runs on a learned-phase fixed clock; Mamba's Δt already is a soft boundary detector | gate saturates open | correlation of learned Δt with whitespace | open |

## C. Change what a chunk is

| idea | mechanism | expected failure | falsifier | status |
|---|---|---|---|---|
| C1 pooled summary + hash row | attention-pooled, order-aware chunk vector concatenated with the chunk-string hash row, never the boundary state alone | loses the context the boundary state carried | `probe_main --identity` | partially tested: pool chunker (EXP-012) stopped early; chunk hash tables kept (EXP-017/018, +0.03-0.06) |
| C2 cross-attention dechunker | each byte attends to the last 4 chunk vectors instead of the EMA | collapses onto the current chunk | swap in a 1-stage run | open |
| C3 VQ chunk dictionary | EMA codebook (32k) on chunk vectors; predict next *code* with a real softmax; codebook usage = tokenizer diagnostic | codebook collapse | tiny run | open |
| C4 chunk-bigram table on the inner output | FLOP-free table keyed by chunk bucket, added to z before dechunking, trained by the next-chunk loss | no recurring buckets under a singleton-heavy segmentation | offline chunk-bigram cost on learned vs whitespace segmentation | open, cheap |
| C5 string-keyed router | score = Σ_n S_n[hash(last n bytes)], zero FLOPs, consistent by construction | cannot see word length; EXP-019's windowed MLP lost | 3 seeds prior-free | partially tested (EXP-019 negative) |

## D. Tokenizer-free from first principles (what a tokenizer buys, and the minimal substitute)

| function of a tokenizer | minimal byte-level substitute | status |
|---|---|---|
| compute per bit | mixture-of-depths over bytes (top-k by a learned score; same rank trick; no segmentation) | open: MoD at r = 1/6 vs the 1-stage H-Net at equal FLOPs is the clean ablation of whether hard segmentation buys anything |
| string-keyed memory | hashed n-gram tables (have, +0.06) and chunk-string hash tables (have); product-key memories | kept |
| sequence-length reduction | fixed-stride pooling (U-Net), MoD | open; fixed6 oracle (EXP-009) was worse than learned chunks |
| whitespace / morphology bias | the warm start (have) or an entropy prior (A5) | kept / open |
| multi-byte output interface | sampled softmax over hashed next chunk (have); multi-byte parallel heads (J bytes per chunk state) | open |
| fixed decode cost | speculative whole-chunk decoding from the vocabulary head + byte verification | open, inference only |
| determinism | eval-mode running statistics in the rank decider (have); string-keyed router (C5) | kept |

## Ranked bets (expected impact × ease)

1. **A1 gain objective** (running). The only objective measured to prefer the useful basin.
2. **B1 no bypass + byte memory**: one flag, three seeds, answers whether the bypass is load-bearing or merely a
   symptom. Pair with A1 if A1 alone adopts but does not win.
3. **A3 online unigram-entropy objective**: BPE's own criterion, differentiable through the chunk STE; cheap.
4. **MoD ablation** (D1): if a boundary-free byte MoD matches the 1-stage H-Net at equal FLOPs, hard segmentation is
   not where the value is and the whole program should move to compute allocation + memory.
5. **A5 entropy prior on code / multilingual rungs**: the case where whitespace is the wrong prior and H-Net's
   learned units can beat BPE's pretokenizer; the blog's code results point there too.
6. C4 chunk-bigram table and C1 pooled summary: cheap memory-side additions with offline falsifiers.

## Verdict on "can learned tokenization beat BPE"

At 27M parameters and 50M token-equivalents on English, parity is the ceiling: BPE's advantage is 12.9M parameters
of FLOP-free, 100%-consistent string-keyed memory, and an H-Net can only rebuild that memory once its key (the
segmentation) is stable. A measurable win needs (i) a router with its own objective or consistency by construction,
(ii) scale where the outer networks' FLOP share stops dominating (60% at tiny, 12% at L12) and BPE's rare-word tail
starts to cost, and (iii) domains where whitespace pretokenization is wrong (code, numbers, non-Latin scripts). The
ladder tests (ii); A1/A3/B1 test (i); A5 on the code and multilingual sources tests (iii).
