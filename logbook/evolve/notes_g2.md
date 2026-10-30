**The bar has moved.** A BPE model given the same in-context copy mechanism (g2-bpe-match, token-level match model)
scores 1.460 on seed 1, against 1.656 for plain BPE. Any H-Net candidate that uses copying must beat 1.460, and any new
mechanism that BPE could equally use must be tested on BPE too before it counts (the fairness contract now requires a
BPE analogue for every non-H-Net-specific mechanism).
**What is H-Net-specific so far.** Copying helps the byte model more than the token model (-0.325 vs -0.196): the byte
index also copies text that tokenization splits differently (~5.6% of mix bytes). The confirmed H-Net components
(two seeds, vs hnet_ws 1.829 ± 0.002): g1-mem-02 suffix-keyed chunk memory -0.074, g1-arch-02 offset-resolved read-back
-0.032, g1-opt-01 Muon-native LR transfer -0.031. On top of copying they add only 0.01-0.02 each (g2-match-mem 1.483,
-arch 1.491, -opt 1.493 vs g1-wild-01 1.502): partly overlapping effects. Best H-Net: 1.483, 0.023 behind BPE + copy.
**What failed and why.** Longer chunks with a wider main network (g1-alloc-01/02): word-aligned (90-98% of cuts on
whitespace) but no gain at 6 bytes and worse at 8 (the byte decoder must spell more bytes per chunk state). Balancing
the decoder inputs (g1-arch-01) and anchoring the bypass by weight decay (g1-opt-02): no gain; the bypass's scale was
never the problem (interp audit). Annealed exploration of boundaries (g1-router-02): worse, and seed 2 fell into the
suffix basin anyway. The prior-free H-Net with copying is 0.056 worse than the warm-start one (suffix basin).
**Pending:** prefix-trie output vocabulary (g1-mem-01, 1.737 seed 1, 4.5× slower wall clock), prefix dictionary
(g1-wild-02, 1.770 seed 1), self-surprise teacher (g2-router-01, rerun), mem-02 × arch-02 (g2-combo-02).
**Open directions with an H-Net-specific edge:** (1) copying units that only a learned segmentation can offer (copy whole
chunks, match on chunk ids, or let the router cut where a copy starts/ends); (2) output-side memory keyed by the learned
chunk (the trie) made fast; (3) making the router prior-free without losing to the suffix basin (the self-surprise
teacher, init screening); (4) compression that adapts per domain (BPE cannot); (5) byte-level robustness: held-out code
and multilingual text, where the per-source gaps were smallest.
