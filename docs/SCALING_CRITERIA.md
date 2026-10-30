# Which H-Net ideas should survive to 100B and to images / video

Written 2026-10-30, after the evolve loop (docs/EVOLVE.md, logbook/evolve/notes_g4.md, notes_g5.md). Every verdict below
is a prediction: no idea has been tested above 38M parameters. The strongest evidence we have is the L03 1e16 → L05
1e17 transfer check: anything whose gain shrank between those two rungs is assumed worthless at 100B.

**Rule for keeping an idea:** it is (a) a fix to how the model learns or decides, not a memorised shortcut, and (b) not
tied to text (no exact byte strings, no whitespace).

## Drop: will not scale, or text-only

| idea (evolve id) | gain at L03 1e16 | why it goes |
|---|---|---|
| Copying / in-context match model (g1-wild-01) | −0.324 | Its gain fell ~4x for the H-Net and ~10x for BPE from L03 1e16 to L05 1e17 (BPE: −0.19 → −0.015). Large models learn in-context copying with attention (induction heads). Images and video rarely repeat byte-exactly. |
| Word-completion memory / prefix trie (g1-mem-01, g3-mem-01/02, g5-output-02) | −0.025 on top of copying | Memorised exact byte prefixes: knowledge a large model holds in its weights. Text-only. |
| Suffix-keyed chunk memory (g1-mem-02) | −0.019 on top of copying | Exact byte strings; a small-model crutch. Text-only. |
| Prefix dictionary (g1-wild-02), chunk pointer copy (g3-copy-01), copy-aware compression (g3-copy-02) | small or neutral | All depend on exact repeats; several were already neutral at L03. |
| Whitespace warm start / space hint (`router_prior_weight`) | −0.03 | A hand-written rule for space-delimited text; no meaning for pixels or audio. |

**Consequence:** these account for almost all of the loop's gain. Without them the best small model is about 1.79 bpb
against BPE's 1.646 at L03 1e16, still about 9% behind. The loop found shortcuts that help small models, not a better
hierarchy.

## Keep: should scale and is modality-agnostic

| idea | evidence | why it should hold |
|---|---|---|
| Self-taught chunking: causal next-byte probe whose entropy leads the rank decision early, plus a probe-to-router teacher, both decaying to 0 by 30% of training (g3-route-02) | −0.052 vs the same model without it (g2-match-pf); word basin on every seed; replaces the space hint at +0.010 | Chunking by prediction difficulty applies to any stream (text, audio, image patches, video frames). Meta's Byte Latent Transformer used entropy-based patching up to 8B parameters. For images the probe needs a continuous (or quantised-patch) target instead of a 256-way byte. |
| Muon-native LR transfer: no Adam-derived per-stage multiplier on Muon matrices (g1-opt-01) | −0.031 (2 seeds) | A correctness fix to LR transfer across width; matters more as models get wider. Modality-agnostic. |
| Rank (quantile) boundary decision with exact compression; fp32 router logits; standardized scores | made training stable (EXP-014..023, audit B2) | Exact, predictable compute budget per step; engineering that matters more at scale. Modality-agnostic. |
| Position-aware read-back, `dechunk_slots` (g1-arch-02) | −0.032 (2 seeds) | How the bytes / patches inside a chunk read the chunk's state. Plausible at scale; unverified. |

## Unproven: test at a second rung before deciding

| idea | result at L03 | why it might matter at scale |
|---|---|---|
| Compression that varies with predictability (g3-route-01 entropy regions; g3-copy-02 copy-aware) | neutral | Where video should benefit most: a static scene can become one chunk, compute goes where things change. A 20M text model has too little redundancy to show it. |
| Extra deep-layer pass on surprising chunks, chunk refresh (g5-compute-01/02) | neutral; −1.8% FLOPs/byte for -02 | Adaptive compute tends to pay more with larger models and data. |
| Product-key memory inside the main network (g5-output-01) | neutral | Meta's "Memory Layers at Scale" reported gains at billions of parameters; modality-agnostic. |
| Hashed byte n-gram input tables (`ngram_embed`, part of every H-Net baseline here) | load-bearing (ablation: 0.01-0.17 bpb per M params) | BLT found hash n-gram embeddings useful at scale; text-only, so keep only for a text model. |

(BLT and "Memory Layers at Scale" are cited from memory; re-check before relying on their numbers.)

## Changes to the search (proposed for the next evolve round)

1. **Accept a gain only if it holds or grows across two rungs.** Each candidate runs at L03 and L05 (same 1e16
   budget), and its gain must not shrink. About +30% cost per generation; it filters crutches before they are stacked.
   (Already in docs/EVOLVE.md as the "second rung before stacking" rule.)
2. **No exact-match lookups in the search space.** No copy indexes, tries, suffix memories or new n-gram tables, so
   proposers must improve the hierarchy itself.
3. **A non-text benchmark early.** Raw audio or image bytes at small scale, with a patch Transformer (not BPE) as the
   baseline. Self-taught chunking and predictability-dependent compression are the H-Net's real argument there, and
   text cannot test them.
