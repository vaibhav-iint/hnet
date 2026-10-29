# Creative H-Net variants and tokenizer-free designs (coordinator's own list, written before reading the agents')

Premise from the measurements: the router has no objective of its own that prefers *useful* units; the decoder has a
parallel path (the bypass) that wins the early race; the next-chunk objective rates word and suffix tokenizers the
same; the hierarchy's only verified gain is identity (hash tables) and a 20% share of the compute at tiny scale.

## A. Give the router an objective that a tokenizer would have

1. **Segmental marginal likelihood (unigram-LM style, differentiable).** Instead of one hard segmentation, keep the
   byte-level encoder states and define per-position chunk scores s_t; the boundary probabilities p_t define a
   distribution over segmentations. Train the model on the *expected* loss under that distribution with a forward
   algorithm over boundaries (linear chain: a position is either a boundary or not; the dechunked state at t is the
   p-weighted mixture of "chunk continues" and "new chunk starts here"). This is exactly the EMA dechunker's recursion
   with p as the mixing weight, so the model already computes it; what is missing is that *the chunk vectors entering
   the main network* are hard. Variant: soft chunking = run the main network on the EMA-smoothed byte stream subsampled
   at fixed stride (a U-Net) while the router only controls the smoothing. Cheap: no top-k, fully differentiable.
   Falsifier: tiny run; does the learned p align with whitespace without a prior?

2. **MDL router.** A tokenizer minimizes description length: bits to encode the segmentation + bits to encode each
   chunk's string under a learned dictionary + bits of the byte loss. We already have the chunk hash table and a
   sampled-softmax next-chunk loss; add the *chunk-identity* term: predict the chunk's own hash bucket from its boundary
   state (a "what am I" loss). Word-like units are highly predictable from their own content (few types, repeated);
   suffix units are not better here... wait, suffixes are also few types. The discriminating term is the *next* chunk
   given the current one plus the cost of the segmentation itself (number of boundaries is fixed by the ratio, so only
   the type-distribution entropy differs): word tokenizers have a Zipfian, lower-entropy unigram distribution than
   suffix tokenizers (chunk_stats: unigram bits/byte differs). So: minimize the unigram entropy of chunk types under a
   hashed count table (an EMA count of buckets, differentiable through p via the STE on chunk identity). That is BPE's
   compression criterion made differentiable and applied online.

3. **Consistency as the objective, not a penalty.** Train the router on a self-supervised "same string → same
   decision" task with hard negatives: the decision at the end of a string should be invariant to context (tokenizer
   property) but the *type* should be rare enough. Combine with 2.

4. **Router distilled from byte surprisal (data-derived prior instead of whitespace).** Replace the whitespace warm
   start with a prior computed from a byte n-gram model's conditional entropy spikes (BLT's entropy patching as a
   *curriculum target*, not as the tokenizer). It is language-agnostic and works for code/math where whitespace is a
   poor tokenizer. Trivial to implement: an n-gram surprisal table on the fly (we have hashed n-gram tables) and the
   same BCE curriculum machinery as `router_prior_weight`.

## B. Remove the race instead of refereeing it

5. **No bypass; byte memory instead.** Delete the residual path entirely; the decoder sees only the dechunked
   hierarchy plus the byte-level n-gram embedding (identity) and a short local attention window over raw bytes (a
   SpaceByte-style local model). Then the hierarchy is the only long-range path and the router gets the full gradient.
   The risk is early training: the hierarchy is noise at init, so the decoder must survive on local context (it can:
   a byte n-gram model is ~2.3 bpb). Never tested; the cheapest structural test there is.

6. **Bypass as a scheduled resource: late, gated, and *narrow*.** Keep the bypass but make it a low-rank projection
   (rank 8-16) so it can carry identity but not the whole prediction; the hierarchy must carry the rest. Pairs with
   EXP-027's margin gate.

7. **Two-timescale state space, no boundaries.** Replace the chunk hierarchy by an SSM/attention stack where the
   inner block runs every k-th byte with a learned k-phase (fixed-stride downsampling, learned alignment via a
   small shift head). The "tokenizer" becomes a learned alignment of a fixed-rate clock to the text. Loses variable
   length but keeps everything else and has no race (no decision to collapse).

## C. Change what a chunk *is*

8. **Chunk = pooled summary + boundary state.** Probes showed the chunk vector is a boundary state, not a summary.
   Add an attention-pooled summary (one query per chunk over its bytes) concatenated to the boundary state. Cost
   small (chunks are short). Makes chunk identity available without hashing and gives the main network a real
   "token embedding".

9. **Decoder reads chunks by cross-attention, not EMA.** Each byte position attends to the k most recent chunk
   vectors (causal), replacing the EMA dechunker; the decoder can pick which chunk to read and the router gets
   gradient through attention weights (softer than STE).

10. **Learned discrete dictionary (VQ) at the chunk level.** Quantize chunk vectors with a 32k-entry codebook
    (EMA VQ) and feed the code embedding to the main network; the output side predicts the next *code* (a real
    softmax over 32k, like BPE) alongside bytes. The model then has an explicit learned vocabulary with the sampled
    softmax replaced by a proper one; codebook usage statistics are a direct tokenizer diagnostic.

## D. Tokenizer-free from first principles (what a tokenizer buys, and the minimal replacement)

- Compute per unit of information → mixture-of-depths over bytes (a per-byte router to a deep path with a fixed
  capacity, no segmentation; top-k by a learned score, same rank trick we already have). Falsifier: does MoD at the
  same FLOPs beat BPE at tiny scale? It removes chunking entirely while keeping "attention over fewer positions".
- Discrete memory for frequent strings → hashed n-gram / chunk tables (have) and product-key memories keyed by
  byte-window hashes (bigger, FLOP-free).
- Sequence-length reduction → fixed-stride pooling (U-Net) or MoD; variable-length chunks are the least necessary
  function and the one that causes every failure we have.
- Output interface → multi-byte prediction heads (predict next 4 bytes jointly with a factorized softmax) so one
  forward step emits several bytes, decoupling decode cost from byte count.

Conclusion before reading the agents: the variable-length *decision* is the part worth removing first (5, 7, MoD),
and the router objective worth adding first is 2 (online MDL / unigram entropy), which is the only one that
distinguishes word from suffix tokenizers in our measurements.
