# Evolve generation 5 and the L05 1e17 transfer check: notes (orchestrator)

## Generation 5 (L03 1e16, seeds 1-2; VM 1.14 h, $6.66; branch evolve/g5 @ ed87aa8)
Parent g4-combo-02 1.433 ± 0.002 (3 seeds). Bar g2-bpe-match 1.461 ± 0.001 (3 seeds).

| id | val bpb | per seed | vs parent | prediction | verdict |
|---|---|---|---|---|---|
| g5-compute-02 refresh routed to the 50% most surprising chunk starts, stage-1 MLP 896 -> 512 | 1.432 ± 0.003 | 1.434 1.430 | −0.001 | ~1.432 | neutral, but at −1.8% FLOPs/byte vs the parent (a free saving, not a gain) |
| g5-output-02 freed chunk-table rows to the trie (77k -> 116k) | 1.437 ± 0.000 | 1.437 1.437 | +0.004 | 1.430 ± 0.008 | neutral / slightly worse |
| g5-compute-01 refresh on every chunk, main T3 -> T2 | 1.438 ± 0.001 | 1.439 1.437 | +0.005 | 1.418 | falsified |
| g5-output-01 product-key memory in the main network | 1.441 ± 0.001 | 1.440 1.442 | +0.008 | 1.421 ± 0.010 | falsified |
| g5-output-bpe-mem (control: BPE + copy + the same memory, +2.6M params) | 1.460 ± 0.001 | 1.460 1.461 | −0.001 vs g2-bpe-match | – | the memory does nothing for BPE either |

Generation 5 found nothing: at L03 the g4-combo-02 stack is at a local optimum for these directions.

## Transfer check (L05 1e17: 1974 steps x 1,089,536 bytes, params <= BPE L05 38.20M; VM $10.37; branch evolve/g4 @ d8b7c91)

| arm | val bpb | per seed | vs BPE + copy |
|---|---|---|---|
| g2-bpe-match-L05 (BPE + token copy) | **1.209 ± 0.002** | 1.208 1.211 | – |
| ladder bpe L05 (seed 0) | 1.224 | – | +0.015 |
| g4-combo-02-L05 | 1.239 ± 0.013 | 1.248 1.230 | **+0.030** |
| g4-combo-03-L05 (tokenizer-free) | 1.254 | 1.254 | +0.045 |
| ladder hnet_ws L05 (seed 0) | 1.353 | – | +0.144 |

**The L03 win does not transfer.** At L03 the crossover led BPE + copy at every eval (−0.08 at step 90 to −0.028 at the
end); at L05 it trails at every eval (+0.013 at step 394, +0.03-0.04 mid-run, +0.030 at the end).

Reading (hypotheses, not yet tested):
1. **Copying is worth much less to BPE at L05** (−0.015 vs plain BPE, against −0.19 at L03): a 5-layer, 5-head BPE
   learns in-context copying with attention; a 3-layer one at 1e16 does not. The lookup components (match model, trie,
   n-gram and chunk tables) are crutches whose value falls as the network can learn the same function.
2. **They still help the H-Net a lot at L05** (−0.114 vs hnet_ws), but they take it only to about plain BPE (+0.015),
   not past BPE + copy. The H-Net's own deficit is flat on this ladder (hnet_ws − BPE = +0.128 at L03 1e16, +0.129 at
   L05 1e17); what shrinks with scale is how much the lookup components buy (−0.40 at L03 → −0.11 at L05 for the
   H-Net, −0.19 → −0.015 for BPE).
3. The loop optimized one small benchmark for five generations; the transfer check is the guardrail that catches
   that, and it did. Promotion rule (EVOLVE.md): no candidate is promoted to the ladder default.

What would test this cheaply next (not run; budget cap reached): the same g4-combo-02 vs BPE + copy pair at L04 1e16
and L03 1e17 (separate model size from data/compute), and a mean-ablation of the match model / trie at L05 to measure
how much each component still carries.

Note: score.py's `fitness` column for the `-L05` rows in population.jsonl is computed against the L03 BPE reference
and is meaningless; compare the L05 rows with each other (table above).
