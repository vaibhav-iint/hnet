# Evolve generation 3: proposer context

Benchmark: L03 at 1e16 FLOPs on mix_v1 (docs/EVOLVE.md). BPE reference val bpb: **1.6691**. Fitness = H-Net val bpb − BPE (lower is better; negative beats BPE). Target: beat BPE by a wide margin.

## Population (ranked)

| id | gen | track | parents | fitness | mean ± sd | seeds | mechanism at end (seed 0) | hypothesis |
|---|---|---|---|---|---|---|---|---|
| g2-bpe-match | 2 | – | – | -0.1958 | 1.4602 | {'1': 1.4602} |  |  |
| g2-match-mem | 2 | – | – | -0.1731 | 1.4829 | {'1': 1.4829} | bypass_ratio 4.12, p_margin 0.29, p_mid 0.17, score_std 22.41 |  |
| g2-match-arch | 2 | – | – | -0.1649 | 1.4911 | {'1': 1.4911} | bypass_ratio 1.41, p_margin 0.21, p_mid 0.46, score_std 8.94 |  |
| g2-match-opt | 2 | – | – | -0.1626 | 1.4934 | {'1': 1.4934} | bypass_ratio 2.29, p_margin 0.16, p_mid 0.58, score_std 7.86 |  |
| g1-wild-01 | 1 | – | – | -0.1537 | 1.5023 | {'1': 1.5023} | bypass_ratio 4.09, p_margin 0.26, p_mid 0.23, score_std 9.82 |  |
| g2-match-pf | 2 | – | – | -0.0984 | 1.5576 | {'1': 1.5576} | bypass_ratio 9.05, p_margin 0.17, p_mid 0.51, score_std 3.81 |  |
| g1-mem-01 | 1 | – | – | +0.0815 | 1.7375 | {'1': 1.7375} | bypass_ratio 9.16, p_margin 0.40, p_mid 0.06, score_std 8.13 |  |
| g1-mem-02 | 1 | – | – | +0.0994 | 1.7554 ± 0.034 | {'1': 1.7791, '2': 1.7316} | bypass_ratio 5.69, p_margin 0.38, p_mid 0.06, score_std 7.38 |  |
| g1-wild-02 | 1 | – | – | +0.1137 | 1.7697 | {'1': 1.7697} | bypass_ratio 6.94, p_margin 0.40, p_mid 0.04, score_std 15.05 |  |
| hnet_ws | 0 | control | – | +0.1272 | 1.8227 | {'0': 1.8227} | bypass_ratio 5.09, p_margin 0.27, p_mid 0.11, score_std 14.74 | same as hnet + whitespace warm start (router BCE toward whitespace ends for the first 10% of training, decays to 0) |
| g1-arch-02 | 1 | – | – | +0.1412 | 1.7972 ± 0.001 | {'1': 1.798, '2': 1.7965} | bypass_ratio 1.34, p_margin 0.21, p_mid 0.48, score_std 12.95 |  |
| g1-opt-01 | 1 | – | – | +0.1423 | 1.7983 ± 0.002 | {'1': 1.7967, '2': 1.7999} | bypass_ratio 2.43, p_margin 0.16, p_mid 0.55, score_std 13.17 |  |
| g1-alloc-01 | 1 | allocation | hnet_ws | +0.1651 | 1.8211 ± 0.007 | {'1': 1.8258, '2': 1.8165} | bypass_ratio 5.24, p_margin 0.21, p_mid 0.50, score_std 17.31 | A BPE model's compute per position is pinned by its vocabulary (4.1 bytes/token); an H-Net's is not. Compressing to 6 bytes per chunk instead of 4.145 cuts the  |
| g1-arch-01 | 1 | – | – | +0.1685 | 1.8245 ± 0.002 | {'1': 1.8256, '2': 1.8234} | bypass_ratio 1.12, p_margin 0.14, p_mid 0.61, score_std 13.17 |  |
| hnet_ws | 1 | – | – | +0.1731 | 1.8291 ± 0.002 | {'1': 1.8274, '2': 1.8308} | bypass_ratio 4.83, p_margin 0.23, p_mid 0.25, score_std 11.34 |  |
| g1-opt-02 | 1 | – | – | +0.1776 | 1.8336 ± 0.003 | {'1': 1.8357, '2': 1.8316} | bypass_ratio 0.97, p_margin 0.22, p_mid 0.46, score_std 12.03 |  |
| hnet | 0 | control | – | +0.1834 | 1.8789 | {'0': 1.8789} | bypass_ratio 8.19, p_margin 0.33, p_mid 0.04, score_std 5.24 | 1-stage H-Net conversion of L03 (Mamba-2 outer, end_mlp router, rank decisions, chunk vocabulary + next-chunk loss, n-gram tables), prior-free |
| g1-router-02 | 1 | – | – | +0.1959 | 1.8519 | {'1': 1.8519} | bypass_ratio 5.83, p_margin 0.24, p_mid 0.24, score_std 6.27 |  |
| hnet | 1 | – | – | +0.2041 | 1.8601 ± 0.049 | {'1': 1.8255, '2': 1.8946} | bypass_ratio 4.81, p_margin 0.21, p_mid 0.35, score_std 9.77 |  |
| g1-alloc-02 | 1 | allocation | hnet_ws | +0.2091 | 1.8651 ± 0.003 | {'1': 1.8671, '2': 1.863} | bypass_ratio 6.52, p_margin 0.30, p_mid 0.33, score_std 8.04 | A BPE model's compute per position is pinned by its vocabulary (4.1 bytes/token); an H-Net's is not. Compressing to 8 bytes per chunk instead of 4.145 cuts the  |
| bpe | 0 | control | – | ref | 1.6952 | {'0': 1.6952} |  | BPE Transformer L03, in-corpus 32k BPE, Muon LR 0.02 (the reference) |
| bpe | 1 | – | – | ref | 1.6560 | {'1': 1.656} |  |  |
| bpe | 2 | – | – | ref | 1.6560 | {'1': 1.656} |  |  |

## Orchestrator notes (why things won or lost)

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


## Read before proposing

- `docs/EVOLVE.md`: the loop, the benchmark, the fairness contract, the candidate format
- `CLAUDE.md`: working rules; the bug-finding meta (gradient paths, step-0 audit, lazy optimizer)
- `docs/BLOG_GAP_ANALYSIS.md`: what the blog's recipe has that ours lacks; the blog's own per-run data
- `docs/IDEAS.md`: ideas already considered, with status and falsifiers
- `docs/IMPL_AUDIT.md`: implementation audit vs the official code and the paper (bugs found)
- `logbook/interp/REPORT.md`: mechanistic interpretability of every experiment so far
- `logbook/EXPERIMENTS.md`: results of EXP-002..028 (tiny scale)
- `hnet/config.py`: DCConfig: every existing router / decision / hierarchy option, documented
