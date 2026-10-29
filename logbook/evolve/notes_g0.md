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
