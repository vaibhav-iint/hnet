# Evolve generation 3: notes (orchestrator)

Benchmark L03 1e16 mix_v1, 908 steps, seeds 1-2. Bar: **g2-bpe-match (BPE + token copy) 1.461 ± 0.001**.
VM 1.27 h, $7.38 (loop total $25.37 of $50). Branch evolve/g3 @ 7751276.

| id | val bpb | per seed | vs parent | vs bar | mechanism read |
|---|---|---|---|---|---|
| g3-mem-02 fast trie on match+mem (chunk table halved) | **1.461 ± 0.001** | 1.462 1.460 | −0.025 | 0.000 | topk_cov 0.999 from step 300; bnd_on_ws 0.51-0.53 |
| g3-combo-01 match + mem + read-back + Muon-native LR | **1.462 ± 0.003** | 1.464 1.460 | −0.024 | +0.001 | parts on top of copying add (without copying they did not: g2-combo-02) |
| g3-copy-02 copy-aware compression | 1.485 ± 0.005 | 1.482 1.489 | −0.001 | +0.024 | boundaries inside copies 0.07-0.17 (predicted <= 0.03): the bias was partly overridden; neutral |
| g3-route-01 entropy-region compression | 1.485 ± 0.004 | 1.489 1.482 | −0.001 | +0.024 | neutral, as its proposer feared (LM loss pulls the gain down) |
| g3-copy-01 chunk pointer | 1.488 ± 0.002 | 1.490 1.487 | +0.002 | +0.027 | whole-chunk copy is redundant with byte copy at the decoder |
| g3-route-02 prior-free router with probe curriculum | 1.503 ± 0.000 | 1.503 1.503 | **−0.052** vs g2-match-pf | +0.042 | word basin on both seeds (bnd_on_ws 0.60 @300, 0.55 end; router_agree 0.95) |
| g3-mem-01 fast trie on hnet_ws | 1.736 ± 0.025 | 1.753 1.718 | +0.002 vs full trie | – | same gain as the full trie at ~1.2x step instead of 1.84x; seeds split 0.035 |

Findings
1. Two disjoint routes reach the bar: an output memory (trie) and a decoder/optimizer route (read-back + Muon LR).
   They touch disjoint parameters, so g4 crosses them (g4-combo-02, predicted ~1.44).
2. The prior-free router problem has a working fix: a causal next-byte probe whose entropy leads the rank decision
   and teaches the router for the first 30% of training. Both seeds in the word basin, 0.052 better than without it.
   g4-combo-03 = g4-combo-02 with this router and no warm start: the tokenizer-free candidate.
3. Copy-structure tricks on top of byte copying (chunk pointer, copy-aware compression) are neutral: the decoder's
   byte-level copy already captures what they offer at this scale. Do not repeat without a new reason.
4. Top-k output heads: k = 64 keeps the full trie's gain (the k = 16 prefix head lost ~40% of it; 14% of the bits sit
   outside the top 16).
