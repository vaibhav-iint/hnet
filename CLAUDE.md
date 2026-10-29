# CLAUDE.md — working rules for this repository

H-Net (dynamic chunking, byte-level) vs BPE/SmolLM3 at small scale. Start with `README.md`, `docs/BLOG_REPRO_PLAN.md`,
`docs/LADDER.md`; the chronological record is `logbook/WORKLOG.md`, the experiment table `logbook/EXPERIMENTS.md`.
Python: `/home/user/.venv310/bin/python` (training, tests); `/home/user/.venv/bin/python` for `logbook/build_page.py`.
Tests: `python -m pytest tests -q` (CPU, ~10 min). Never print, copy or commit API keys or service-account files.

## Commit policy
WIP snapshots go to `main` (`git commit` + `git push -u origin main`); milestones to `milestone-*` branches. No pull
requests unless asked. Commit messages end with the attribution lines given in the session; no model identifiers in
commits, code comments or docs. Subagents never commit; the coordinator does.

## Visibility into training (every status update, every few minutes while anything runs)
A status update is never a bare number. It is, in this order: a **TL;DR** (best arm per sweep vs the reference,
what finished, what is in flight, every alert), a **table** per sweep (seeds done, step/total, val bpb mean ± sd with
per-seed values, zero-main probe, train bpb, throughput, alerts), and **plots**: val bpb vs step with the reference
levels, the stage-0 mechanism panels (router margin, undecided fraction, raw-score std, hierarchy vs bypass rms,
next-chunk CE, grad norm) and, for LR sweeps, val bpb vs LR per architecture. All three come from one tool:

    bash $S2/report.sh          # rsync the bucket mirrors, then
    python logbook/training_report.py --live $S2/live --out logbook/figures/training_report.html --md ...

(`$S2` = this session's scratchpad.) The HTML embeds the PNGs, is published as an artifact and sent as a file; the
markdown TL;DR + tables go in the chat message. Mechanism panels, not only loss curves: the loss curve says *whether*
a run is behind, the panels say *why*. Read the tokenizer basin from `bnd_on_ws/s0` (share of boundaries on whitespace:
~0.9 word basin vs ~0 suffix basin, decided by step ~300) and the hierarchy's value from a *mean*-ablation probe,
not from rms ratios: the bypass/hierarchy rms ratio tracks the optimizer's LR, not adoption (logbook/interp/REPORT.md),
and zero-ablation overstates the hierarchy 1.4-26x because it also removes a constant offset.
Alerts are computed, not eyeballed (`alerts()` in `logbook/training_report.py`: compression off target, score std → 0,
tau saturated, margin undecided, grad-norm spike, seed spread > 0.03 = bifurcation).

## Finding training-dynamics bugs: the meta (read before touching the model or launching a sweep)

This section exists because most of the project's real findings were found late, after paid sweeps, although every one
of them was derivable from the code or from a five-minute CPU run. The list of misses and the detection lag:

| What was wrong | How we found it | When we could have found it |
|---|---|---|
| Stage-1 cosine router never leaves p ≈ 0.5 (cos ≈ 0 on random states; ratio loss has zero gradient once F is on target) | post-hoc deep dive after the EXP-002 sweep | by hand, from the loss formula, day 0 |
| (Corrected by the interp audit) the hierarchy carries ≤0.005 bits for the first ~100 steps while the bypass is informative from step ~35: there was never a 'switch'; the large early zero-ablation effects were a constant offset | zero-main probe, read as information | a mean-ablation probe on a step-100 checkpoint |
| Seed lottery attributed to noise / data order; data order is identical across seeds (`shuffle_buffer: 0`), so the basin is chosen by the initialization | final bpb spread over seeds | ranking seeds across arms that share a parameter layout (same seed always loses) |
| Detached quantile threshold left the logit *scale* unanchored → sigmoid saturates → stage dead (EXP-014) | tau hit 1.000 in the metrics of a paid sweep | asking "what anchors the scale?" when writing the decider |
| Consistency penalty satisfied by shrinking all logits to a constant (EXP-020) | `score_std` fell to 0.15 in the metrics of a paid sweep | asking "what is the cheapest way to zero this penalty?" |
| Median/MAD scale estimator collapses when 5/6 of scores coincide (EXP-021) | divergence, grad norm 1e6 | unit test on a bimodal / near-constant score vector |
| Seeds bifurcate into word vs suffix tokenizers; the "±" hid a two-basin system for ~10 experiments | tokenizer statistics at step 500 (EXP-023), weeks later | the first time a seed spread exceeded the BPE baseline's |
| The next-chunk loss sends the router no gradient (computed before dechunking; the STE sits after) | reasoning during the adoption deep dive; verified by `autograd.grad` only in `tests/test_adoption_options.py` | an autograd check the day the loss was added |
| A zero-initialised chunk table gives zero gradient to everything upstream at init | the same test failing | a step-0 gradient audit |
| Launcher started training before the data rsync finished | runs failed with missing files | an assertion on the data directory before `train` |

The common pattern: we evaluated **outcomes** (val bpb at the end, averaged over seeds) and inferred mechanisms
afterwards. Mechanisms must be instrumented and tested first; outcomes come last. The rules:

1. **Draw the gradient graph before training.** For every objective (CE, ratio, STE confidence, next-chunk CE, aux
   heads, penalties) write down which parameter groups it reaches and through which path. Then *verify* with
   `torch.autograd.grad(loss_term, params, allow_unused=True)` on a tiny model. A loss is not a training signal for a
   module until autograd says so. Add the check as a test (pattern: `tests/test_adoption_options.py`).
2. **Step-0 audit.** One forward/backward on CPU at init: per-group gradient norms, which are exactly zero and why
   (zero-init tables, zero residual projection, ones from an STE), which activations are degenerate (router p within
   0.4–0.6, cos ≈ 0, constant scores), boundary fraction F, compression. Repeat at steps 10, 50, 200. Nothing goes to
   a GPU before this printout has been read.
3. **Every auxiliary loss is a controller.** Find its equilibrium and the gradient *at* the equilibrium (ratio loss:
   ∂L/∂G = 0 once F is on target → nothing moves p off 0.5). Ask what the controlled quantity does when the controller
   is satisfied, and what anchors any scale or offset it does not control (the quantile decider controlled the rank,
   nothing controlled the scale).
4. **Ask what the lazy optimizer does.** For every penalty or decision rule enumerate the cheap ways to satisfy it:
   shrink everything to a constant, saturate, select nothing, route around through a parallel path. If one exists
   the optimizer will find it within a few hundred steps. Under scale-free decisions every router-side term must be
   scale-free too.
5. **Statistics must survive the ugly distributions.** Before choosing a mean/std, median/MAD, quantile or softmax
   normaliser, test it on: heavy tails, a bimodal vector with 5/6 of the mass in one mode, all-identical values, a
   handful of huge outliers. Score distributions of a working tokenizer *are* bimodal.
6. **Instrument the mechanism from step 1.** Log per step from the start, and look at the first 500 steps before
   looking at bpb: boundary fraction and compression per stage, router margin, `tau`/`score_std`, `dechunk_rms` vs
   `residual_rms`, per-stage gradient norms, and the tokenizer statistics (`hnet/analysis/chunk_stats.py`: types,
   singleton fraction, whitespace alignment, consistency) at every eval, not only at the end.
7. **Seeds are experiments, not noise.** If the seed spread exceeds the baseline's, do not average: find the step at
   which the runs diverge, diff the mechanism metrics at that step, and look at what each run's tokenizer is. A
   bimodal outcome means a race or a bifurcation, and the fix is upstream of the split.
8. **Cheap before expensive.** A five-minute CPU/L4 probe of the mechanism precedes any multi-seed, multi-hour sweep.
   Sweeps answer "does it help"; only probes answer "why", and "why" is what we needed every time.
9. **Read the paper as a list of fragile components, not a spec.** STE, ratio loss, zero-init residual, cosine
   router, EMA dechunker, detached thresholds: for each, write the failure mode it invites *before* running. On day 0
   search for implementers' notes (blogs, GitHub issues, friends' write-ups); most of what we found the hard way was
   already written down by someone.
10. **Keep a standing invariance suite** and run it on every model change: causality (change a future byte → earlier
    logits unchanged), packed sequences ≡ single sequences, cached decoding ≡ teacher forcing, a backward-only flag
    leaves the forward pass bit-identical, realized FLOPs within a few percent of the budgeted FLOPs (the paper
    router ran 10–16% over budget for weeks).
10b. **The full test suite runs before every launch**, not the quick core subset: the sweep VMs run it as a gate and
    stop if it fails, which cost EXP-028 its first launch (a new model buffer changed the state dict and broke the
    dense-parity test). A failed gate on a VM is paid for in GPU minutes and a relaunch; locally it is ten minutes.
11. **Fail loudly.** A stage that selects no positions raises; a launcher asserts the data and tokenizer exist
    before training starts; a metric outside its sane range (F far from 1/ratio, tau at 1.000, score_std → 0, grad norm
    × 100) stops the run instead of filling a log we read a day later.
12. **Post-mortem every failure against this list.** Which rule would have caught it, and which test or assertion now
    does. Add the row to the table above and the check to the suite before relaunching.

### Checklist when adding a component (router, decider, loss, gate)
- Gradient paths written and autograd-verified (rule 1); step-0 audit done (rule 2).
- Equilibrium and the gradient there; what anchors scale and offset (rule 3); lazy-optimizer escape routes (rule 4).
- Statistics tested on heavy-tailed / bimodal / constant inputs (rule 5).
- New mechanism metric logged per step and plotted for the first 500 steps of a tiny run (rule 6).
- Tiny probe run read *before* the sweep is launched (rule 8); forward-pass-unchanged test for training-only flags.
- Generated files (`docs/LADDER.md`, `configs/ladder/**`, `configs/search/ladder_*`) are checked by tests against
  their generator: never hand-edit them; notes go to `docs/ANALYSIS_LADDER.md` or the work log.
