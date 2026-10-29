# Tokenizer-free byte-level LMs: first principles, then H-Net variants

Part A was written before reading any project code or notes. Part B was written after reading `hnet.py`, `routing/decide.py`, `chunk_vocab.py`, `DCConfig`, and the two WORKLOG sections named in the brief.

## Part A. What a tokenizer buys, and the minimal end-to-end substitute for each function

### A.1 The functions, decomposed

A BPE tokenizer with a whitespace pretokenizer provides at least seven separable things:

1. **Compute allocation per bit.** Frequent (low-surprisal) strings become one unit, so compute per *bit* is roughly uniform. BPE is a crude entropy equaliser.
2. **A dictionary keyed by string.** The embedding table is content-addressable memory: a frequent string gets a vector for zero compute, and the same string hits the same row in every context, so statistics pool.
3. **Sequence-length reduction** (~4x), so attention cost drops ~16x and credit assignment spans fewer positions.
4. **Whitespace/morphology bias.** The pretokenizer forbids merges across spaces; units never straddle words.
5. **A multi-byte output interface.** One softmax emits ~4 bytes, so each loss term carries several bits.
6. **Fixed decode cost and ~4x fewer decode steps.**
7. **Determinism.** Train and test segment identically.

A byte-level design needs 1, 3, 5, 6 to compete on compute and 2, 4, 7 to compete on *loss*. Most tokenizer-free work attacks 1 and 3 and silently loses 2, 4 and 7; that is the story of why learned segmentation loses by a few hundredths of a bit.

### A.2 Learned boundaries, and why the downstream-only objective is weak

Mechanism: a router scores each byte position, hard decisions select chunk ends, a main network runs on one vector per chunk, a dechunker broadcasts back to bytes. The router receives gradient only via a straight-through estimator (STE) from the byte loss, plus a ratio penalty pinning the boundary rate.

Why this objective is weak, mechanistically:

- The router's gradient is "did nudging this boundary lower the byte loss *given the current main network*". Early on the main network is random, so that is noise; the ratio penalty is the only coherent force and any segmentation with the right rate satisfies it.
- Segmentations related by a one-unit shift (cut before vs after the space; word vs stem+suffix) give the main network nearly the same information per chunk. The downstream loss is flat across them, so initial conditions decide. A seed lottery is the predicted outcome of a flat objective.
- Any residual path around the hierarchy is an easier descent direction than fixing the router; gradient descent will grow the bypass.
- A contextual router cuts the same string differently in different contexts, so functions 2 and 7 are lost even when the segmentation looks word-like.

What a tokenizer's objective actually is, stated so that it has a gradient:

**(i) MDL / unigram.** A segmentation s of x into units u_1..u_n costs L(s) = sum_i -log q(u_i) + |D|·c. With maximum unit length K the forward algorithm gives log Z = log sum_s prod_i q(u_i) in O(L·K) with exact gradients, and posterior boundary marginals b_t = P(boundary at t | x) need no STE (segmental LMs: Sun & Deng, Kawakami et al.). q may be conditional, q(u_i | h_{i-1}), making the recursion a proper LM, if the segment scorer is cheap: a span score f(h_end) - f(h_start) from a byte encoder, as in segmental CRFs. Degenerate solution: all-K or all-1 units; countered by the dictionary cost or a length prior.

**(ii) Branching entropy (Harris 1955, Tanaka-Ishii).** A boundary belongs where H(x_{t+1} | x_{<=t}) spikes relative to the previous position: inside a unit the next byte is nearly determined, at a unit end it is not. A tiny byte LM (2 layers, or a 5-gram table) computes it with no learned router; it is consistent up to that small model and implements function 1 directly, since boundaries fall where bits arrive. A relative threshold (spike size) is more stable than an absolute one.

**(iii) Variational segmentation.** Router q(s | x), prior p(s) (geometric lengths, or mass on whitespace-adjacent cuts), ELBO = E_q[log p(x | s)] - beta·KL(q || p). The KL replaces the ratio penalty with a shape-aware regulariser that gives the router a dense gradient while the main network is still random. The expectation needs Gumbel-ST or K-sample REINFORCE with a leave-one-out baseline (low variance at K=4).

**(iv) Consistency as architecture.** Make the router a function of a bounded causal window (last 8-16 bytes). Same string, same decision, for free; the router becomes a finite-state tokenizer. It cannot use long context to segment, which a tokenizer never could either.

**(v) Usefulness for the byte loss**, the current STE objective, should remain, but as a fine-tuning term on top of (i)-(iv), not as the only signal.

### A.3 Alternatives to boundaries

**Fixed strided pooling (MegaByte/Canine).** Patches of k bytes, local encoder, global model, local decoder; full gradient. Failure: patches cut words at random offsets and compute is spent uniformly, not per bit; fixed patches lose to BPE unless space-aligned (SpaceByte). Falsifier: k=4 patches vs BPE at 27M; a gap above ~0.05 bpb kills it.

**Byte-level mixture-of-depths.** Every byte is a position; a scalar score selects the top-rL bytes per sequence for the heavy layers, the rest skip via residual, and selected outputs are scaled by their score, so the router learns without STE. This is a boundary model whose chunk vector is the selected byte's state and whose dechunker is the residual; the batch coupling of top-k is a feature. Failure: positional routers (every 4th byte). Falsifier: MoD at r=0.25 vs an H-Net at the same rate; if it matches, hard segmentation and pooling buy nothing.

**Perceiver-style latent slots.** Per window of W bytes, M learned queries cross-attend to give M latents; main network on latents; byte decoder cross-attends back. Fully differentiable, but M/W is a fixed compression (function 1 lost), windows cut words, and slots collapse to positional bins. Falsifier: W=16, M=4 vs mean-pool-by-4; if latents do not win by >0.02 bpb the slots are positional.

**Hashed n-gram memory, no segmentation.** At each byte hash the suffixes of length 2..n into tables and sum the rows into the byte input: function 2 with no segmentation, direct sparse gradient, no STE, but nothing for sequence length. Falsifier: byte transformer with and without tables at equal parameters; the gain should be large (>0.05 bpb) and should *shrink* when a good segmentation with a chunk dictionary is added, since they provide the same function.

**Two-timescale SSM.** A slow SSM whose update is gated by sigma(g_t); Mamba's input-dependent dt is already a soft boundary detector. Failure: the gate saturates open unless the update rate is penalised, and attention length is not reduced. Falsifier: correlation of learned dt with whitespace; above 0.5, the SSM already finds words and a hard router is redundant for those layers.

### A.4 The output side

Byte-by-byte decoding costs one step per byte and gives ~1-2 bits of gradient per position. Three repairs:

- **Chunk-level sampled softmax over hashed strings.** True class hash(next chunk), sampled negatives: a dense multi-byte signal and an output-side dictionary. Not decodable (collisions, no inverse), so it is an auxiliary loss, and a good one *only if attached where it reaches the router*; it rewards segmentations whose chunks are predictable as wholes, which is MDL in disguise and, as such, blind to usefulness.
- **Multi-byte parallel heads.** Predict the next J bytes from the chunk state with independent heads: decodable, J loss terms per chunk, free speculative decoding.
- **Dictionary-proposal speculative decoding.** Propose a whole chunk from a stored string table via the hashed head, verify byte by byte: function 6 at inference with no change to training.

### A.5 What the main network should operate on

Candidates for the chunk vector: (a) the encoder state at the chunk's last byte, contextual but inconsistent across contexts; (b) a pooled summary of the chunk's bytes, stabler but order-blurring; (c) a context-free embedding keyed by the chunk string, which is exactly what BPE provides. The answer is (a)+(c). (a) alone is what learned-segmentation models do, and it is why they lose function 2.

### A.6 Ranking (by expected loss gain per unit of risk, at 2M-50M params)

1. Hashed n-gram input memory plus a chunk-string hash on the main-network input (restores function 2 under any segmentation; no STE; cheapest).
2. Branching-entropy boundaries from a frozen tiny byte LM, used as teacher for the router (gives function 1 directly; removes the seed lottery; no gradient needed).
3. Segmental marginal likelihood with a K-bounded forward algorithm as the router's own objective (principled, exact gradient, but O(K) extra scorer compute).
4. Variational router with a length prior replacing the ratio penalty (cheap, fixes the regulariser, still STE/REINFORCE).
5. Byte-level MoD instead of pooling (a clean ablation of whether hard segmentation matters at all).
6. Multi-byte heads for the decoder (decode speed and denser gradient; neutral on loss).
7. Perceiver slots and fixed pooling (likely losers; useful baselines).

## Part B. H-Net variants that change the structure

Facts from the code and the two WORKLOG sections: the stage computes `h = dechunk(z) * STE(p) + residual_proj(enc)` with a zero-init bypass; the router's only gradient is the STE on the dechunked path (plus `chunk_ste`); the rank decider takes the top 1/N standardised scores of the whole batch; `ChunkVocab` hashes each chunk's last 16 bytes through a per-batch bank `bank[k, n-1, t]` (hash of the n bytes ending at t) and the next-chunk head is a sampled softmax; the chunk vector is the boundary byte's state plus the hash row. Measured: the bypass opens at steps 60-130 while the router is undecided; the lottery is word vs suffix tokenizer; the next-chunk CE is equal for both. What the system lacks is a router objective that is (a) available at step 0, (b) independent of the decoder's adoption, (c) not blind to usefulness. Each variant supplies one of those or removes the race.

**B1. Hierarchy-only stage, no residual, byte n-gram memory in the decoder.** Delete `residual_proj`. The decoder sees `dechunk(z) * STE(p)` plus a byte-level `HashNgramEmbedding` (orders 1-4), so it keeps a FLOP-free local memory for letters-of-words while the hierarchy is the only contextual path; nothing can be adopted instead of it. With the `select` chunker non-boundary encoder states would get gradient only through the router score, so use `pool`. Failure mode: slow first 200 steps (the hierarchy is noise at init), and the router may still fall into the suffix basin, now with no decoder to "return". Falsifier: 1-stage ratio-6 `vp` with the residual removed, 3 seeds, 1000 steps, logging `dechunk_rms` and whitespace alignment at 200/500/1000. If seeds still bifurcate and the loser is above 1.80, removing the bypass only removed the escape hatch.

**B2. Segmental-marginal (unigram MDL) teacher computed from the hash bank.** The bank already hashes every span of length 1..16 ending at every byte. Add a learned table `logq[bucket]` and run the forward algorithm `alpha_t = logsumexp_n (alpha_{t-n} + logq[bank[0, n-1, t]])` plus the backward pass; the posterior boundary marginal `b_t = exp(alpha_t + beta_t - log Z)` is exact and differentiable, and `-log Z / T` is a per-byte MDL loss that trains `logq` (EM in gradient form). Train the router's logit by BCE toward `b_t` (detached) and keep the rank decider. Cost: O(16 T) scalar ops. Degenerate solution: unigram MDL loves long rare units; add a length prior `-lambda * n` per unit, or cap the table so rare strings collide and inherit low `logq`. Falsifier: forward-backward offline on 1 MB of val with `logq` fitted by 10 EM iterations, `chunk_stats` on the Viterbi path. Whitespace alignment below ~60% or consistency below 90% means the teacher is not a tokenizer; drop it before touching the model.

**B3. A usefulness objective for the router: REINFORCE on the hierarchy's gain.** Run the stage-0 decoder (1-2 layers, cheap) twice, on `dechunk(z) + residual` and on `residual` alone. The per-byte gain `g_t = CE_res(t) - CE_full(t)` is exactly what the hierarchy adds. Credit each byte's gain to the chunk covering it, subtract a running baseline, and train the router logit with `-(advantage) * log p(decision)`. This is the first objective that separates word from suffix tokenizers: suffix chunks summarise the word that just ended and add nothing for the next word's bytes; word chunks do. Failure mode: variance, and a chicken-and-egg with adoption (gain is zero if the decoder ignores the hierarchy), so pair it with B1 or `residual_gate=margin`. Falsifier (no training): on EXP-023 seed 0 (word) and seed 1 (suffix), compute `g_t` with the residual zeroed and average over the first 3 bytes after each boundary. If the word tokenizer's gain is not clearly larger, this objective is blind too.

**B4. Context-free string-keyed router: the tokenizer as a lookup table.** Replace the MLP router with `score_t = sum_{n=1..4} S_n[bank[k, n-1, t]]`, zero-init scalar tables indexed by the hash of the last n bytes, and keep the rank decider. Same string, same score, in every context, by construction, so the finding that training destroys consistency becomes moot. Zero FLOPs, STE gradient flows into the tables, and the batch-wide rank now means "select the top 1/N most boundary-like strings from a global dictionary", which is literally what a tokenizer is; the batch coupling becomes a feature. Failure mode: a 4-byte window cannot see word length, so long words are never split and short function words are over-split. Falsifier: 1-stage ratio 6, 1000 steps, 3 seeds, no warm start; consistency reads ~100% by construction and the only question is bpb against the warm-started contextual router (1.635). Within 0.01 and the seed lottery is over.

**B5. Data-derived boundary prior (entropy spikes) replacing the whitespace prior.** The warm-start target "boundary at whitespace" is a hand-written pretokenizer. Replace it by `prior_t = 1[ H_t - H_{t-1} > tau ]`, with `H_t` the next-byte entropy of a frozen 5-gram byte model counted over the hash bank (or a 2-layer byte LM trained 200 steps): Harris's branching entropy, BLT's patching rule. It finds English word boundaries but also morphemes, code tokens and digit groups, which whitespace does not. Use it exactly as `router_prior_weight` is used now. Failure mode: the spike sits on the first byte *after* the space (start semantics), so the target must be shifted by one under end semantics; spikes inside rare words produce fragments. Falsifier: `chunk_stats` on the entropy-threshold segmentation at ratio 6; whitespace alignment above 70% and consistency above 90% qualify it as a prior, then a 3-seed run with the prior swapped.

**B6. Two-level dictionary: a FLOP-free chunk-bigram memory on the main network's output.** Add a table `B[bucket(chunk_j)]` whose row is *added to the inner output* z_j before dechunking and trained by the next-chunk sampled softmax: a chunk-level n-gram LM (BPE's output embedding, keyed by the learned segmentation) that the main network only has to correct. Failure mode: under a singleton-heavy segmentation no bucket recurs and the table learns nothing, so this doubles as a segmentation diagnostic. Falsifier: fit a chunk-bigram model offline on the whitespace and on the learned segmentation over 10 MB; if the learned one's bigram cost is more than 0.1 bpb worse, no table will save it.

**B7. Chunk vector as an order-aware pooled summary plus the hash row, never the boundary state.** `pool` exists as a mean; make it an attention pool with a learned query and a within-chunk positional bias (so "stop" and "pots" differ), concatenated with the chunk-string hash row. The boundary state depends on the whole prefix, so two occurrences of one word get different vectors and the main network cannot build word-level statistics. Failure mode: the pooled vector loses the context the boundary state carried. Falsifier: the existing `probe_main --identity`: the pooled vector should reach high bucket-identity accuracy where the boundary state does not, and the bpb gap at 1000 steps says whether identity or context matters more.

**B8. Decoder reads chunks by causal cross-attention, not EMA dechunking.** Each byte attends to the last 4 chunk vectors ending at or before it, instead of an EMA that blurs chunks into a p-weighted running average; the decoder then sees the *previous* words distinctly, which is what predicting the next word's first bytes needs. Keep `chunk_ste` so the router still gets gradient. Failure mode: attention collapses onto the current chunk (reproducing `select`), and the STE signal gets sparser without the EMA's smooth p-dependence. Falsifier: swap the dechunker in a 1-stage ratio-6 run; below 0.01 bpb difference, the dechunker is not where the loss is.

**B9. Mask-mixing curriculum from external to learned boundaries.** Instead of a BCE prior on the logit, use the whitespace boundaries *as the mask* with probability pi(progress) per segment (1 to 0 over the first 30%) and the router's mask otherwise, training the router toward whichever mask was used. The decoder adopts the hierarchy in the first 60 steps because the chunks are words from step 0, and the router is trained in a world where the hierarchy is already load-bearing. Failure mode: the router learns whitespace and nothing more (acceptable), or is destabilised when pi hits 0 and the masks disagree. Falsifier: compare with the `external` whitespace router at equal FLOPs; if the learned router never wins by more than seed variance, learning boundaries at this scale is worth nothing beyond whitespace.

**B10. Speculative whole-chunk decoding from the vocabulary head.** At inference, `pred_proj(z_j)` scores buckets; a string table of the most frequent string per bucket (counted during training) proposes a whole chunk, and the byte decoder verifies it in one batched pass. This recovers function 6 without changing training. Failure mode: low acceptance under an inconsistent segmentation, since the proposal must match the router's next boundary exactly. Falsifier: acceptance rate on val with the EXP-024 checkpoint; below 50% there is no speedup at ratio 6.

### Top three bets

1. **B4, the string-keyed router.** It attacks the one property the WORKLOG identifies as destroyed by training (consistency), costs no FLOPs, turns the rank decider's batch coupling into a dictionary selection, and makes the seed lottery structurally impossible. If it reaches ~1.64 without a warm start it also explains the warm start's success: the prior was buying consistency, not whitespace per se.
2. **B3, the gain-based REINFORCE objective** (with B1 or the margin gate to keep the hierarchy load-bearing). It is the only proposal here whose objective is provably not blind to the word-vs-suffix distinction, because it measures what the hierarchy adds for the *following* bytes, and its no-training falsifier on existing checkpoints costs a few minutes.
3. **B2, the hash-bank unigram-MDL teacher** (with B5 as the cheaper fallback). It gives the router an objective available at step 0 that does not pass through the decoder, is exact (no STE), and reuses the bank already computed per batch; its offline falsifier decides in under ten minutes whether it is a tokenizer at all.

### Can learned tokenization beat BPE at all?

At 27M parameters and 50M token-equivalents, probably not by much, for accounting reasons rather than architectural ones. BPE's tables are 12.9M of 15M parameters of FLOP-free, string-keyed memory with a 100%-consistent key; an H-Net must rebuild that memory in hashed tables keyed by a segmentation it is still learning, or compute word identity with FLOPs budgeted for something else. Everything above is a way of giving the model a *stable key* early enough that the tables fill; once the key is stable the two models are the same model with different keys, and parity is the expected result, which is what the warm-started 1.635 shows. The upside beyond parity has three sources: adaptive compute (boundaries where bits arrive), cross-word units ("of the", which BPE's pretokenizer forbids), and domains where whitespace pretokenization is wrong (code, numbers, non-Latin scripts, agglutinative languages). On English FineWeb-Edu at this scale each is worth ~0.01 bpb, the size of the seed variance, so a win is not measurable here. A real win needs (i) a router that is consistent by construction or by an objective of its own, so the dictionary has types; (ii) enough data (1B+ tokens) that the outer networks' FLOP share (60% here, 12% at L12) stops dominating and BPE's rare-word tail, split into uninformative pieces, starts to cost; (iii) a mixed domain where whitespace is the wrong prior. Without (i) no scale helps: a model whose tokens are mostly singletons has no vocabulary, and its main network is a byte model wearing a hierarchy.
