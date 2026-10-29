# Evolve generation 1: proposer context

Benchmark: L03 at 1e16 FLOPs on mix_v1 (docs/EVOLVE.md). BPE reference val bpb: **1.6952**. Fitness = H-Net val bpb − BPE (lower is better; negative beats BPE). Target: beat BPE by a wide margin.

## Population (ranked)

| id | gen | track | parents | fitness | mean ± sd | seeds | mechanism at end (seed 0) | hypothesis |
|---|---|---|---|---|---|---|---|---|
| hnet_ws | 0 | control | – | +0.1272 | 1.8227 | {'0': 1.8227} | bypass_ratio 5.09, p_margin 0.27, p_mid 0.11, score_std 14.74 | same as hnet + whitespace warm start (router BCE toward whitespace ends for the first 10% of training, decays to 0) |
| hnet | 0 | control | – | +0.1834 | 1.8789 | {'0': 1.8789} | bypass_ratio 8.19, p_margin 0.33, p_mid 0.04, score_std 5.24 | 1-stage H-Net conversion of L03 (Mamba-2 outer, end_mlp router, rank decisions, chunk vocabulary + next-chunk loss, n-gram tables), prior-free |
| bpe | 0 | control | – | ref | 1.6952 | {'0': 1.6952} |  | BPE Transformer L03, in-corpus 32k BPE, Muon LR 0.02 (the reference) |

## Orchestrator notes (why things won or lost)

Generation 0 is the three controls of the 1e16 L03 ladder rung (one seed each; seeds 1-2 are added with generation 1).
What is known about why the H-Net loses here (details in the documents listed below):
- In every prior-free run the residual bypass into the byte decoder ends 5-40x the hierarchy's rms; the router falls into
  a word-like or suffix-like tokenizer depending on the seed; the next-chunk objective cannot tell the two apart, while
  the hierarchy's measured per-byte gain can (2.4x at step 500).
- Refereeing the race (chunk STE, margin-gated bypass; EXP-027) and a REINFORCE gain objective at weight 1 (EXP-028,
  interim: worse, router undecided) did not help at tiny scale.
- Protocol: our outer networks are FLOP-matched to a 32k-vocab head (48-60% of FLOPs at L03-L04) vs the blog's 128k
  (79-86%); the paper's Adam-derived per-stage LR multipliers are stacked on Muon; the H-Net's best Muon LR is 0.01 vs
  BPE's 0.02 (0.04 bpb lost by the transfer). These are benchmark-protocol questions the orchestrator owns; candidates
  may change the H-Net's own optimizer settings (multipliers, per-group LRs) but not BPE's.
- At 1e17 L05 the gap at matched steps shrank from 0.34 (step 985) to 0.20 (step 1379): more steps help the H-Net.


## Read before proposing

- `docs/EVOLVE.md`: the loop, the benchmark, the fairness contract, the candidate format
- `CLAUDE.md`: working rules; the bug-finding meta (gradient paths, step-0 audit, lazy optimizer)
- `docs/BLOG_GAP_ANALYSIS.md`: what the blog's recipe has that ours lacks; the blog's own per-run data
- `docs/IDEAS.md`: ideas already considered, with status and falsifiers
- `docs/IMPL_AUDIT.md`: implementation audit vs the official code and the paper (bugs found) (not written yet)
- `logbook/interp/REPORT.md`: mechanistic interpretability of every experiment so far (not written yet)
- `logbook/EXPERIMENTS.md`: results of EXP-002..028 (tiny scale)
- `hnet/config.py`: DCConfig: every existing router / decision / hierarchy option, documented
