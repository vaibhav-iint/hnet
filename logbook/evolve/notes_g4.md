# Evolve generation 4: notes (orchestrator)

Benchmark L03 1e16 mix_v1, 908 steps. VM 0.81 h, $4.73 (loop total $30.10 of $50). Branch evolve/g4 @ 3c7ad2f.
Bar, now on three seeds: **g2-bpe-match (BPE + token copy) 1.461 ± 0.001** (1.460 1.462 1.461).

| id | val bpb | per seed | vs bar | mechanism read |
|---|---|---|---|---|
| **g4-combo-02** match + suffix memory + fast trie + read-back + Muon-native LR | **1.433 ± 0.002** | 1.432 1.432 1.435 | **−0.028** | prediction 1.440 ± 0.01 held; trie topk_cov 0.998-1.0; bnd_on_ws 0.58 @300, 0.49 end |
| **g4-combo-03** same, no whitespace warm start, g3-route-02 router | **1.442 ± 0.004** | 1.440 1.445 | **−0.019** | word basin on both seeds (bnd_on_ws 0.56-0.61 @300, 0.49 end; router_agree 0.97); cost of dropping the prior +0.010 (prediction +0.017 ± 0.01) |
| g3-combo-01 seed 3 | 1.461 | – | 0.000 | three seeds 1.464 1.460 1.461 |
| g3-mem-02 seed 3 | 1.462 | – | +0.001 | three seeds 1.462 1.460 1.462 |

Findings
1. **First H-Net that beats BPE + copy at equal FLOPs, params and data**, by 0.028 bpb (about 15 standard deviations of
   either arm's seed spread), and a tokenizer-free one (no whitespace warm start, no permanent prior) by 0.019.
2. The two disjoint routes added almost exactly: −0.025 (trie) and −0.024 (read-back + LR) on g2-match-mem, −0.053
   together. Crossovers of parts with disjoint parameters and paths add; parts that compete for the same role do not
   (g2-combo-02: suffix memory and read-back without copying).
3. Against plain BPE (1.646, three seeds) the margin is −0.21, but most of it is copying, which BPE also gets (−0.16).
   The H-Net-specific margin is the −0.028 against the copy-matched bar.
4. Open: does the margin survive scale? EVOLVE.md asks for a transfer check at L05 1e17 before promotion.
