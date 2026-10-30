# Evolve loop: proposer agents + orchestrator, AlphaEvolve style

Goal: an H-Net that beats BPE at equal training FLOPs and data by a wide margin, found by repeated rounds of
*insight → candidate → fair GPU evaluation → selection*. The orchestrator (main session) owns the benchmark, the
budget, the population database and every GPU launch. Proposer agents own ideas and code.

## The benchmark (fixed; candidates may not change it)

* **Rung L03 at 1e16 training FLOPs on mix_v1** (70/20/5/5 English / code / math / multilingual), the 1e16 ladder spec
  `configs/search/ladder_1e16_L03.yaml`: 908 steps × 544,768 bytes (= 131,072 BPE tokens), Muon + WSD, BPE's Muon LR
  0.02 transferred, 10 MB validation set. One run ≈ 21 min on one A100 (H-Net), ≈ 12 min (BPE).
* **Fitness** = mean over seeds of (H-Net val bpb − BPE val bpb), BPE = mean of 3 seeds of the `bpe` arm. Lower is
  better; negative = beats BPE. Secondary (reported, not selected on until per-source eval is cheap): per-source gap on
  code, held-out code, math, multilingual (`scripts/eval/per_source.py --mix mix_v1`).
* **Controls** every generation re-runs nothing; the reference numbers live in `logbook/evolve/population.jsonl`
  (`bpe`, `hnet`, `hnet_ws` at 2-3 seeds).
* **Seeds**: 2 per candidate; a candidate whose mean beats its parent by more than the larger seed spread gets a 3rd
  seed and a transfer check at L05 1e17 before it is promoted to the ladder default.

## Guardrails (the fairness contract; a candidate that breaks one is rejected without a GPU run)

1. Same data, same steps, same batch in bytes, same validation set, same evaluation code. No change to the BPE arm,
   the tokenizer, the data pipeline, `hnet/eval/*` or the FLOPs accounting.
2. Training FLOPs per byte at the *realized* compression within 2% of the `hnet` arm (check with
   `python scripts/search/fairness.py <spec>`; it fails on mismatch). Parameter count ≤ the BPE arm's (20.0M at L03).
   FLOP-free parameters (hash tables) count toward the parameter budget like any other.
3. Every code change is behind a new config flag whose default leaves the existing model bit-identical
   (prove it: a test that the default forward pass is unchanged). Proposers run their new tests plus
   `tests/test_model_hnet.py tests/test_invariances.py tests/test_flops.py`; the orchestrator runs the full suite once
   on the merged generation before launching (the VMs run it again as a gate). CLAUDE.md rules apply: write the
   gradient paths of every new objective and verify them with `hnet/analysis/grad_audit.py`.
4. No candidate may use a hand-written segmentation prior (whitespace, punctuation lists) as a *permanent* input;
   curricula that decay to zero are allowed and must be labelled as such (the `hnet_ws` control already is one).
5. Inference must stay causal and autoregressive (`tests/test_invariances.py` must pass with the new flag on).

## Candidate format

One directory per candidate: `logbook/evolve/candidates/<id>/` with `candidate.yaml`:

```yaml
id: g1-router-03                # g<generation>-<track>-<nn>
parents: [hnet_ws]              # population ids it modifies
track: router                   # router | architecture | optimization | memory | wildcard
hypothesis: one paragraph: the mechanism, why it should beat the parent, which measured fact it builds on
prediction: {fitness: -0.02, mechanism: "bypass/hierarchy rms < 2 by step 300"}   # falsifiable numbers
branch: evolve/g1-router-03     # git branch holding the code (local commit in the proposer's worktree)
model: configs/evolve/models/g1-router-03.yaml   # optional: a model config (copy of hnet_L03 / hnet_ws_L03 + edits)
set: [model.dc.router_gain_weight=0.3]           # trainer overrides on top of the ladder_hnet run config
checks: {tests: pass, fairness: pass, grad_audit: logbook/evolve/candidates/<id>/grad_audit.txt, smoke: pass}
```

and the evidence files named in `checks`. The smoke check is a short CPU run through `hnet.train.train`
with the candidate's settings on a small batch (e.g. the L03 model with `data.batch_bytes=8192`, 20 steps, or the
tiny config `configs/runs/tiny_hnet1_2m.yaml` if the mix_v1 train shards are not local): loss decreases, no NaN,
compression on target.

## One generation

1. Orchestrator writes `logbook/evolve/context_g<N>.md` (`scripts/evolve/context.py`): the benchmark, the population
   table sorted by fitness with mechanism summaries, every failed idea with its cause, the open hypotheses, and the
   latest audits (implementation, interpretability, blog gap).
2. Four proposer agents run in parallel, one per track, each in its own git worktree. Each reads the context, picks
   or invents **two** candidates (at least one must be a genuinely new mechanism, not a hyperparameter of an old one),
   implements them, passes the checks, commits locally on its branch and returns the candidate ids. Proposers do not
   launch VMs, push, or edit other tracks' files.
3. Orchestrator reviews each candidate against the guardrails (reject with a reason, or accept), pushes the branches,
   builds `configs/evolve/gen<N>.yaml` (`scripts/evolve/make_gen.py`) and launches it on 4×A100 spot.
4. Scoring (`scripts/evolve/score.py`) appends fitness, per-seed values, mechanism metrics and cost to
   `logbook/evolve/population.jsonl`; the training report shows the generation live.
5. Selection: the next context ranks by fitness, keeps the best two per track as parents (diversity), and records
   *why* each loser lost (mechanism metrics, interpretability notes) so the next proposers do not repeat it.

Cost: ~8 candidates × 2 seeds ≈ 16 A100-runs ≈ 1.5 h of a 4×A100 spot VM ≈ $9 per generation.

## Budget and ledger
User-approved cap for the loop: **+$50** (2026-10-29), about five generations. The orchestrator stops launching when the
ledger below reaches the cap and reports. Generation 1 runs seeds 1 and 2 for candidates *and* for the `bpe` and
`hnet_ws` controls (which already have seed 0), so the references get three seeds.

| generation | VM-hours | cost | candidates | best fitness |
|---|---|---|---|---|
| 1 (12 candidates + bpe/hnet/hnet_ws controls, seeds 1-2; 30 runs; branch evolve/g1a @ efe1625) | running | est. $17 (cap 4.5 h = $26) | | |
| 2 (7 candidates + bpe, seeds 1-2; 16 runs; branch evolve/g2 @ 19043df) | running | est. $12 (cap 3 h = $17.5) | | |
