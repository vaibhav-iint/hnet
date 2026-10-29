# Mechanistic interpretability of the H-Net experiments (EXP-002 … EXP-027, ladder 1e16)

*2026-10-29 · CPU-only analysis of 281 runs' metrics and 20 checkpoints (32 FineWeb-Edu validation documents of
≤ 2048 bytes per checkpoint unless stated). Code: `scripts/interp/` (entry point `scripts/interp/run_all.py`,
`--quick` for a smoke test). Data: `logbook/interp/data/`. Figures: `logbook/interp/figures/`.*

Labels: **[M]** measured here (file and number given), **[I]** inferred from measurements, **[W]** taken from the
work log without re-measurement. Definitions used throughout:

* **zero probe** = bpb with the hierarchy's decoder input (the dechunked main-network output) set to zero, minus bpb:
  the trainer's `val/zero_main_s0_bpb` contribution.
* **information** (or mean-ablation) = the same with the hierarchy signal replaced by its per-document mean vector:
  removes what it says about each position but keeps its scale and offset. Same for the bypass.
* **R** = `dechunk_rms/s0 / residual_rms/s0`, the hierarchy-to-bypass rms ratio at the stage-0 decoder input.
* **word basin / suffix basin**: the two tokenizers the 1-stage router converges to (defined in fact 2 below).

## TL;DR

* **The seed lottery is not noise: it is the initialization.** Data order does not depend on the seed; arms with the
  same parameter layout share each seed's init exactly; the full seed order (s2 < s1 < s0) is identical in all 5
  prior-free n-gram-layout arms (chance 1/1296), and s1 loses in all 3 no-n-gram arms. No intervention except the
  whitespace prior changed which seed loses (decider, temperature, residual warmup, chunk STE, margin gate).
* **The two basins are "cut at the space" and "predict the space".** Suffix seeds never cut at whitespace; they cut
  at the last letter of words (55–64%) or inside words, and their score is a word-end *predictor* (AUC 0.75–0.80 for
  "next byte is a non-letter"). End-of-chunk semantics hide the next byte, so these cuts are guesses that land after
  common endings — the "morphological" tokens are a side effect.
* **In the suffix basin the hierarchy carries no information at step 500 (0.002 bits vs 0.082 for the word seed)
  and the chunk table is unused (ablation +0.009 bpb vs +0.46).** The hierarchy's value, where it exists, is on
  punctuation (1.6–2.1 bits/byte), newlines, digits and the first letters of the next word.
* **The zero-main probe overstates the hierarchy's information 1.4–26×**: it removes a constant offset as well.
  `hnet1` (EXP-006) "0.63 bpb" is 0.025 bits of information; EXP-002/008 "0.69" is 0.18–0.27.
* **"Bypass rms 5–40× the hierarchy's = dead hierarchy" is wrong.** Under Muon the bypass rms is proportional to the
  LR (log-log r = 0.96, n = 28) because `residual_proj` has no weight decay; EXP-025's best tiny H-Net (1.630) runs at
  R = 0.13 with a working hierarchy, and live and dead ladder hierarchies have the same R (0.12–0.22).
* **The ladder's prior-free H-Nets are in the losing basin** (≤ 8% of cuts on whitespace, 0.03–0.04 bits of hierarchy
  information); the warm-start ladder H-Nets have a working hierarchy (0.24–0.27 bits) but the 4.15 bytes/chunk target
  forces ~30% of their boundaries off whitespace. One seed per rung, no ablation probe: the ladder gap is not yet a
  measurement of the architecture.
* **The outcome is readable at step ~210 from the hierarchy rms** (within-arm |r| ≥ 0.5 from step 210, AUC 0.88 at
  step 250 without residual modifications); the router margin only from ~440; raw-score std and next-chunk CE never.
* **The tiny "BPE parity" is against BPE at an untuned LR**: BPE at its own best LR is 1.564 (vs 1.634), H-Nets are
  flat in LR. At parity the H-Net wins on short tokens and loses on long whole words (+0.08–0.10 bits/byte on 6–8+-byte
  GPT-2 tokens).
* **Work-log timing of the race is off**: the margin *rises* during steps 60–130 and collapses at 150–300 (trough
  300–400); the bypass opens on a fixed, seed-independent clock (CV 3%) set by the LR schedule.
* **Recurring lazy-optimizer escapes**: EXP-020's normalized penalty was met with heavy tails (margin 0.01–0.07),
  EXP-027's gate by growing the pre-gate bypass (4.6–6.8 rms, effective bypass unchanged).

## Synthesis

### The mechanistic facts that matter most

1. **The seed lottery is a deterministic function of the initialization.** Data order is seed-independent; arms
   with the same parameter layout share each seed's initialization exactly; the losing seed is the same in 3/3
   no-n-gram arms and the complete seed order (s2 < s1 < s0) is identical in 5/5 prior-free n-gram arms (chance
   1/1296). Decider, temperature, residual warmup, chunk STE and margin gate did not change it; only the whitespace
   prior did. [M: `seed_rankings.json`, `init_basin.json`]
2. **The two basins are "cut at the space" vs "predict the space".** Word seeds put a boundary on every whitespace
   byte (an observable, context-free rule: R² 0.88–0.98 from the current byte). Suffix seeds cut on the last letter
   of words (55–64%) and inside words (27–42%), never on spaces, and their score is a word-end *predictor* (AUC
   0.75–0.80 for "the next byte is a non-letter"); end semantics hide that byte, so the cuts are guesses that land after
   common endings (`s`, `d`, `ing`, `ed`) — hence the "morphological" look. At step 500 both are unigram byte rules.
   The suffix router actively drives spaces to the lowest score (z ≈ −3). [M: `router_hist_by_class.png`]
3. **Only the word basin gives the hierarchy information, and early.** Mean-ablation at step 500: word seed 0.082
   bits, suffix seed 0.002 bits; final 0.375 vs 0.056. The hierarchy's value is concentrated on punctuation
   (1.6–2.1 bits per byte), newlines (1.1–1.2), digits, and the first letters of the next word (0.5 → 0.1 from
   letter 0 to letter 4). The chunk table is load-bearing only in the word basin without n-gram tables (ablation
   +0.46 bpb vs +0.009 in the suffix basin; +0.014–0.034 in n-gram models). Hierarchy signals of word-basin models
   converge across seeds and arms (CKA 0.68–0.83) and differ from suffix models (0.15–0.32), whose encoders and
   bypasses are as similar to the word models as to each other (CKA 0.63–0.79). [M]
4. **The zero-main probe overstates the hierarchy's information 1.4–26×.** It removes the hierarchy's constant
   offset along with its information. Mean-ablation: word finals 0.25–0.38 (zero: 0.41–0.75), 2-stage EXP-002/008
   0.18–0.27 (0.69), 1-stage threshold `hnet1` 0.025 (0.646). Several work-log statements built on the zero probe
   ("main-network contribution drops 0.63 → 0.38 with n-gram tables", "the hierarchy is worth 0.04–0.06" per class
   tables) measure offsets. [M]
5. **The bypass/hierarchy rms ratio measures the optimizer, not adoption.** The hierarchy signal is RMS-normalized
   (~1–1.7); `residual_proj` is not, and under Muon (no weight decay) its norm grows ~linearly in steps × LR
   (log-log r = 0.96, n = 28). Live hierarchies exist at R = 0.13 (EXP-025, 1.630 bpb) and R = 0.20 (ladder warm start,
   0.24–0.27 bits); dead ones at R = 0.12–0.14 (ladder prior-free) and R = 0.4–0.6 (tiny suffix seeds). [M]
6. **The ladder's prior-free H-Nets are in the losing basin; the warm-start H-Nets have a working hierarchy.** L03/L04
   prior-free: ≤ 8% of cuts on whitespace, 0.03–0.04 bits of hierarchy information, table unused; warm start: 71% on
   whitespace (the rest forced elsewhere by the 4.15-byte target ratio), 0.24–0.27 bits. One seed per rung. [M]
7. **The adoption race starts on a fixed clock and the outcome is readable at step ~210.** The bypass opens at
   steps 50–110 identically in every seed (CV 3%); the hierarchy rms separates the winners from step ~200 (AUC 0.88
   at 250 without residual modifications); the margin does so only from ~440; score std and next-chunk CE never. [M]
8. **The headline parity is against untuned BPE.** BPE improves from 1.634 (1.5e-3) to 1.564 (6e-3) while the H-Nets
   are flat in LR; the best tiny H-Net (Muon prior, 1.630) is 0.066 behind tuned BPE. At parity the H-Net wins on
   short tokens and loses on long whole words (+0.08–0.10 bits/byte on 6–8+-byte tokens). [M]
9. **Muon is the best tiny H-Net optimizer and removes the dip**, but at LR 0.05 its unbounded bypass growth is
   followed by stage-0 gradient explosions (225–380) at steps 340–410 that kill one seed's hierarchy for good. [M]
10. **There is no early "switch" away from the hierarchy: it has nothing to say yet.** In the dense paper-router run
    the hierarchy's information is ≤ 0.005 bits for the first 100 steps (negative for the first 15) while the bypass
    becomes informative from step 35 (2.2 bits at 100); the large early zero-ablation effects are the decoder's
    dependence on the hierarchy's constant offset. Gating or delaying the bypass therefore cannot create adoption;
    only making the hierarchy informative early can (the whitespace warm start does this). Lazy-optimizer escapes
    recur around such fixes: EXP-027's gate was undone by a growing pre-gate bypass, EXP-020's normalized penalty by
    heavy tails. [M: `early_dynamics.json`, metrics]

### Earliest step at which each failure was detectable

| failure | first visible at | metric (logged or one line away) |
|---|---|---|
| p-space quantile saturation (EXP-014) | step 20–30 | `tau/s*` ≥ 0.999 |
| stage-1 coin flips (EXP-002 paper router) | step ~40 | `p_mid/s1` > 0.5 (dense probe run: 0.63 at step 40, 0.84 at 100) |
| bypass opening (all AdamW runs) | step 50 | `residual_rms/s0` ≥ ½ `dechunk_rms/s0` — but it is the same in winners and losers, so not a failure signal |
| unbounded bypass growth under Muon | step 100 | `residual_rms/s0` slope ∝ LR (2.3 at step 100 for Muon 0.05) |
| losing basin (word vs suffix) | step ~210 | within-arm `dechunk_rms/s0` (AUC 0.88 at 250); directly: share of boundaries on whitespace (0 vs 0.91 at step 500 in the checkpoints; not logged) |
| consistency penalty gamed (EXP-020) | step ~350 | `p_margin/s0` < 0.1 with `score_std/s0` rising |
| Muon 0.05 instability | step 340 | stage-0 grad norm 10–225 (stage-1 unaffected) |
| ladder prior-free hierarchy dead | first eval | not logged: needs the ablation probe or the whitespace-boundary share |
| BPE parity only vs untuned BPE | before EXP-024 | EXP-003 table (BPE at 6e-3 = 1.564) |

### Ranked next experiments and diagnostics (each with a falsifiable prediction)

1. **Log the basin and the information, not the rms.** Add to every eval: share of stage-0 boundaries on
   whitespace (and on letters followed by a non-letter), and a mean-ablation probe next to the zero probe; re-score the
   CLAUDE.md alert on the ablation. *Prediction:* across the existing tiny checkpoints the whitespace share separates
   adopted from non-adopted runs perfectly (≥ 0.9 vs ≤ 0.1); on new runs it does so by step 300. Cost: zero GPU.
2. **Ladder: 2 more seeds of the prior-free L03 arm, and the ablation probe on all rungs.** *Prediction:* at least one
   of three seeds lands in the word basin (whitespace share > 0.8), carries ≥ 0.2 bits of hierarchy information, and
   closes ≥ 0.05 of the 0.18 gap; if all three are in the suffix basin, the basin probability shifts with scale and the
   prior is required.
3. **Ladder warm start with a ratio that whitespace can fill** (target ≈ 6 bytes/chunk on the English-heavy mix, or
   per-language). *Prediction:* whitespace share > 90% after release, hierarchy information ≥ 0.35 bits, gap to BPE at
   L03 shrinks from 0.13 by ≥ 0.04 at equal FLOPs (the main network gets 1.45× fewer positions, so re-balance its width).
4. **Bound the bypass under Muon** (weight decay or Adam for `residual_proj`, or an RMSNorm on its output).
   *Prediction:* `residual_rms/s0` stays ≤ 2.5 at every LR; Muon 0.05 shows no stage-0 grad explosion and all three
   seeds keep ≥ 0.2 bits; Muon 0.025 bpb unchanged within 0.005.
5. **LR-tune both sides of the parity claim.** AdamW warm-start H-Net at 3e-3 / 6e-3 (3 seeds), BPE-Muon with its
   embedding/head on a tuned Adam LR. *Prediction:* the H-Net gains ≤ 0.02 (it is flat in LR in every sweep so far),
   so the tuned tiny gap stays ≥ 0.05; if it gains ≥ 0.05, the parity claim survives tuning.
6. **Replace the whitespace prior by a next-byte-entropy prior** (cut at t when the encoder's predicted entropy of byte
   t+1 is high: spaces precede word starts, while suffix-basin positions precede the predictable space). *Prediction:*
   ≥ 5 of 6 seeds in the word basin on FineWeb-Edu with whitespace share ≥ 0.85, without naming whitespace; on code /
   multilingual slices of mix_v1 it beats the whitespace prior by ≥ 0.02 bpb.
7. **Init screening as a cheap fix and a probe of the mechanism.** 300-step runs of 8 seeds of `vp_ng` (CPU-feasible
   at reduced batch); read whitespace share and hierarchy rms at 250. *Prediction:* bimodal whitespace share; the
   step-250 classification matches the full-run outcome in ≥ 7/8 seeds. Then perturb only the router's init (keep the
   rest) to test whether the router or the encoder init decides: *prediction:* router-only re-init flips the basin in
   ≥ 1/3 of seeds; if never, the encoder's initial features decide.
8. **EXP-028 (gain objective) read-out with this lens.** The gain signal is there (0.082 vs 0.002 bits at step 500) but
   both basins carry ~0 information during steps 150–300 when the basin is chosen. *Prediction:* unless its weight is
   large before step 300, EXP-028 keeps the s2 < s1 < s0 order of the n-gram layout.
9. **Re-measure the old "contribution" tables with mean-ablation** (EXP-006/008/009/011) before reusing any of their
   conclusions. *Prediction:* the fixed-chunk and oracle arms' information contributions are ≤ ½ of the zero-probe
   values, and the ranking of training-signal interventions in EXP-011 changes.

### Work-log and EXPERIMENTS.md claims, tested

| claim (source) | verdict | evidence |
|---|---|---|
| "The decoder switches to the zero-init bypass by step 20–130; the main network's gradient collapses" (CLAUDE.md table, row 2) | **mechanism wrong** | in the dense probe run the hierarchy carries ≤ 0.005 bits of information up to step 100 (mean-ablation; −0.04 to −0.09 at steps 1–10): the decoder never used it. Zero-ablation costs 1.9–2.9 bits at steps 40–60 because the decoder relies on the hierarchy's constant offset (`early_dynamics.json`) |
| "Steps 60–130: the bypass opens, hierarchy rms crashes 1.6 → 0.7 and the margin collapses 0.36 → 0.15 in all seeds" (WORKLOG, adoption deep dive, item 2) | **timing wrong** | bypass crosses the hierarchy at 70–120 [M]; the hierarchy rms minimum is at 120–190; the margin *rises* to 0.34–0.41 by step 60–100 and collapses only at 150–300, trough at 300–400 (`phase_timelines.png`, `phases_summary.json`) |
| "The bifurcation is in the router's initialization or the first few hundred steps of data" (WORKLOG, EXP-023 stats) | **sharpened: it is the initialization** | data order is seed-independent (`shuffle_buffer: 0`); arms with the same parameter layout have bit-identical step-0 routers per seed (`init_basin.json`); the worst seed is the same in 3/3 arms of the no-n-gram layout and the full seed order s2 < s1 < s0 is the same in 5/5 prior-free arms of the n-gram layout (chance 1/9 and 1/1296; `seed_rankings.json`). Decider, temperature, residual warmup, chunk STE and margin gate never changed which seed loses; only the whitespace prior did |
| "The suffix seed learned a morphological tokenizer" (WORKLOG, EXP-018 stats) | **re-described** | the suffix basin is a *predicted-word-end* tokenizer: 57–64% of its boundaries are on the last letter of a word, 27–35% inside words, 0–4% on whitespace; its score separates "next byte is a non-letter" from "next byte is a letter" with AUC 0.75–0.80 [M, Part B]. The router cannot see the next byte (end semantics), so it fires where a word is likely to end (after `s`, `d`, `ing`, `ed`, `ly`); the suffix vocabulary is a side effect of guessing the space |
| "Bypass rms 5–40× the hierarchy's = dead hierarchy" (CLAUDE.md, `alerts()`) | **refuted for Muon, weak for AdamW** | EXP-025 prior_muon 0.025: R = 0.13, contribution 0.38, bpb 1.630 (the best tiny H-Net). Under Muon the bypass rms is ∝ LR (r = 0.96, n = 28); AdamW: Spearman(R, contribution) = 0.45 (n = 111) |
| "The bypass dominates in every prior-free ladder run, so the race is the normal state" (WORKLOG, ladder 1e16; "the race may be a small-scale artefact") | **right outcome, wrong evidence and mechanism** | L03/L04 prior-free hierarchies carry 0.03–0.04 bits because their routers are in the suffix/sub-word basin (≤ 8% of cuts on whitespace); the warm-start ones carry 0.24–0.27 bits at the same R (0.20–0.22 vs 0.12–0.14). The race is not a small-scale artefact: it is decided the same way at L03/L04 |
| "Every H-Net peaks at 1.5e-3 and degrades above it because of the stage-1 limit cycle" (EXPERIMENTS EXP-003, WORKLOG) | **not supported** | H-Nets are flat within seed noise from 1e-3 to 6e-3 (end_mlp 1.820 / 1.815 / 1.860 / 1.817; paper best at 1e-3); limit cycles exist but cost little because stage 1 is barely used |
| "H-Net matches BPE at equal FLOPs" (EXP-024 headline) | **true only against untuned BPE** | BPE at its own best LR (6e-3, still improving at the grid edge) is 1.564 ± 0.001 vs 1.634 at 1.5e-3; the warm-start H-Net was never LR-swept under AdamW. Tuned gap ≥ 0.066 bpb (EXP-025 Muon prior 1.630) |
| "EXP-002: all 18 runs done" (EXPERIMENTS) | **wrong** | `paper_s1`, `end_linear_s0/s1` stopped at step 500–630 (runs.csv); the "paper router 1.78–1.80" range is one finished seed |
| "EXP-010 alpha1 1.818 at +0.8%; resid_id 1.783" (EXPERIMENTS) | **one or two finished seeds** | five of nine EXP-010 runs stopped at step 750 |
| "EXP-020: consistency can be imposed without cost" (EXPERIMENTS) | **hides a failure** | `cons10` margin 0.014–0.071 vs 0.33, raw score std 6.8–11.5 vs 2.8–3.3: the scale-free penalty was satisfied by heavy tails |
| "EXP-027 is neutral" (correction entry) | **confirmed, plus** | finals cste 1.778 / 1.661 / 1.640, cste_rgate 1.742 / 1.679 / 1.648, rgate 1.746 / 1.699 / 1.653: same seed order as vp_ng in all three arms. The gate was neutralized by the optimizer: in the losing seeds the margin fell to 0.06–0.11 and the pre-gate bypass rms grew to 4.6–6.8, keeping the effective bypass (gate × rms) at 0.74–0.96 |
| "The next-chunk CE is the same for both tokenizers (4.9)" (WORKLOG) | **confirmed and generalized** | the next-chunk CE has no predictive value for the outcome at any step (within-arm r and AUC ≈ chance, n = 24) |
| "`residual_proj` is zero-init only in its weight; the bias is live (rms 0.064)" (WORKLOG) | **confirmed** | bypass rms 0.064–0.071 at step 10 in every run |

## Part A. Training progress from the metrics of every run

Tools: `scripts/interp/phases.py` (events, snapshots, cross-run statistics → `data/phases_runs.json`,
`data/phases_summary.json`, `figures/phase_timelines.png`, `figures/phase_events.png`,
`figures/bypass_scale_vs_lr.png`, `figures/early_predictors.png`) and `scripts/interp/predictability.py`
(step-resolved predictability → `data/predictability.json`, `figures/predictability_vs_step.png`). 281 runs with metrics
(runs.csv plus the live mirrors; 45 unfinished runs are excluded from every outcome statistic), 202 finished
H-Nets, 117 with a zero-main probe.

Event definitions (stage 0): *bypass cross* = first step with residual_rms > dechunk_rms; *hierarchy dip* = minimum
of dechunk_rms in steps ≤ 600; *margin trough* = minimum p_margin in steps 100–800, *recovery* = p_margin ≥ 0.3
again; compression collapse = compression > 3× or < 1/3 of target; tau saturation = tau ≥ 0.999; grad spike =
grad norm > 5× its rolling median; loss spike = train bpb > running minimum + 0.25 after step 200.

### A1. The common timeline (AdamW, tiny, lr 1.5e-3, warmup 176 steps)

| phase | steps (median over arms) | what the metrics show | seed-dependent? |
|---|---|---|---|
| 0. hierarchy-only | 0–30 | bypass rms 0.07 (the live bias of `residual_proj`, [M] 0.064–0.071 at step 10 in every run), R ≈ 20; main-network grad share 0.10–0.20 | no |
| 1. warmup gradient crash | 20–40 | main-network grad / stage-0 grad falls to ≈ 0.03 at step 30 in every AdamW run, then recovers to 0.15–0.3 by step 100 (`phase_timelines.png`, last column) | no |
| 2. bypass opens | 50–110 (bypass reaches ½ of the hierarchy rms at 50, overtakes it at 70–120) | residual rms 0.07 → ~1.0, identical across seeds (coefficient of variation across seeds of an arm 3% at steps 50–500, [M]) | **no** |
| 3. hierarchy dip | 120–190 | dechunk rms falls from 1.6 to 0.45–0.75 (vocab arms 0.6–0.7, prior 0.94) | weakly (seed CV of hierarchy rms 6–7% from step 100, [M]) |
| 4. router undecided | 150–500 | p_margin trough 0.11–0.14 at steps 300–400 (vocab arms), raw score std peaks 10–17 at steps 300–450 | yes |
| 5. lock-in / divergence | 400–800 | hierarchy rms climbs back to 1.3–1.6 in adopted seeds, stays 0.9–1.2 in the others; margin recovers to ≥ 0.3 at 500–860 | yes |

Cross-run medians per experiment are in `data/phases_summary.json` (`phase_table_by_exp`). The timeline is the same in
every AdamW tiny experiment from EXP-008 to EXP-027; what differs is phases 4–5.

### A2. What predicts the outcome, and how early ([M], `data/predictability.json`)

Population: tiny 1-stage chunk-vocabulary runs, prior-free AdamW (n = 24: EXP-018 `h1_vocab_pred`, EXP-023 `vp`,
`vp_ng`, `vp_rwarm`, EXP-024 `temp4`, EXP-027 `cste`, `rgate`, `cste_rgate`; in brackets the n = 15 subset without
residual warmup or gate, whose rms is comparable). "Within-arm r" correlates the deviation of a metric from its arm
mean with the deviation of the final val bpb from its arm mean: the seed lottery with architecture removed. AUC:
adoption = final contribution ≥ 0.3 bpb.

| metric | within-arm \|r\| ≥ 0.5 from step (and for every later step) | AUC ≥ 0.8 (or ≤ 0.2) from step | within-arm r / AUC at 250 | at 500 |
|---|---|---|---|---|
| hierarchy rms (dechunk_rms) | **210** (210) | 440 (**210**) | −0.64 / 0.79 (−0.71 / 0.88) | −0.80 / 0.88 (−0.87 / 0.98) |
| R = hierarchy / bypass | 220 (210) | 550 (220) | −0.73 / 0.64 (−0.84 / 0.93) | −0.92 / 0.78 (−0.97 / 1.00) |
| train bpb | 260 (150) | never | 0.43 / 0.44 | 0.91 / 0.37 |
| router margin (p_margin) | 440 (510) | 470 (380) | −0.03 / 0.59 | −0.62 / 0.83 |
| raw score std | never | never | 0.09 / 0.44 | 0.45 / 0.38 |
| next-chunk CE | never | never | 0.06 / 0.32 | 0.04 / 0.32 |
| main-network grad share | 820 | 820 | | |

* The seed that will lose is identifiable at **step ~210** (12% of training) from the hierarchy rms; in the
  unmodified-residual arms the AUC is already 0.88 at step 250 and 0.98 at 500. The router margin, on which the work
  log builds its account, carries no information until step ~440, and the raw-score std and the next-chunk CE never
  do (the work log's "next-chunk CE is the same, 4.9" is right and generalizes: it is uninformative everywhere).
* Final bpb and adoption are different outcomes: train bpb at step 500 predicts the final bpb within an arm
  (r = 0.91) but not adoption (AUC 0.37).
* Across all 1-stage tiny runs with the probe (n = 48, `adoption_logistic_1stage_tiny`), `R@500` is the best single
  adoption predictor (leave-one-out logistic accuracy 0.77, best threshold 0.81, base rate 0.58).

### A3. The bypass scale is set by the optimizer and the LR, not by adoption ([M], `bypass_scale_by_optimizer_lr`)

| optimizer, scale | peak LR | n | bypass rms (end) | hierarchy rms (end) | R (end) | contribution (end) |
|---|---|---|---|---|---|---|
| AdamW tiny | 1.5e-3 | 108 | 1.66 | 0.92 | 0.56 | 0.53 |
| AdamW tiny | 6e-3 | 6 | 3.04 | 1.62 | 0.56 | 0.67 |
| Muon ladder L04 | 0.0025 / 0.005 / 0.01 / 0.02 / 0.04 / 0.08 | 2 each (12 at 0.02, all rungs) | 1.38 / 2.21 / 3.66 / 6.72 / 12.1 / 21.7 | 1.18 / 1.28 / 1.25 / 1.20 / 1.04 / 0.92 | 0.86 / 0.59 / 0.35 / 0.19 / 0.09 / 0.04 | not logged |
| Muon tiny (EXP-025 prior) | 0.025 / 0.05 | 3 / 3 | 9.30 / 26.1 | 1.22 / 0.98 | 0.13 / 0.04 | **0.38** / 0.25 |

* Under Muon, log(bypass rms) vs log(LR) has r = 0.96 (n = 28): the bypass rms is the LR. `residual_proj` is a Muon
  matrix with no weight decay (`MUON_DEFAULTS["weight_decay"] = 0`); every step adds an orthogonalized update of
  fixed spectral size, so its norm grows ~linearly in steps × LR (EXP-025 0.05: 6.5 at step 200, 17 at 500, 31 at
  1763). The hierarchy signal is bounded at ~1–1.3 by the inner network's final RMSNorm.
* **The rule "bypass rms 5–40× the hierarchy's = dead hierarchy" (CLAUDE.md, `alerts()` in `training_report.py`) is
  wrong as stated.** EXP-025 `prior_muon` at 0.025 has R = 0.13 (bypass 7.6×) and is the best tiny model of the
  project (1.630 / 1.635 / 1.630 bpb) with a hierarchy worth 0.36–0.40 bpb. Across AdamW tiny runs R_final and the
  contribution are only moderately related (Spearman 0.45, n = 111 finished runs). An rms ratio compares a normalized signal with
  an unnormalized one; it says which optimizer ran, not which path the decoder reads. Use the ablation.
* The ladder never logged the zero-main probe, so "the bypass dominates (rms 5×) in every prior-free run" (WORKLOG,
  ladder 1e16) was a statement about Muon at LR 0.02. The checkpoints (Part B, ladder) show the prior-free ladder
  hierarchies are indeed dead, but because their routers are in the losing basin; the warm-start ones are alive at
  the same R.

### A4. Other metric-level findings

* **LR (EXP-003, 3 seeds) [M].** BPE 1.830 / 1.690 / 1.635 / 1.581 / **1.564** at 5e-4 … 6e-3 (still improving at the
  edge); end_mlp 1.859 / 1.820 / 1.815 / 1.860 / 1.817; paper 1.885 / **1.807** / 1.901 (one 2.10 outlier) / 1.823 /
  1.834. The stage-1 limit cycles at ≥ 3e-3 are real (6–15 compression-collapse records per run) but cost little bpb,
  because stage 1 is barely used. The H-Nets do not degrade at high LR; they fail to improve while BPE does.
* **A train-bpb bump at step 360 in nearly every run** of every experiment is a data-order event (same window order
  everywhere), not a model event; mask it in alerts.
* **Grad-norm spikes rarely precede collapses**: 10 of 187 loss/compression events had a > 5× spike in the preceding
  100 steps. Exception: EXP-025 Muon 0.05 (stage-0 grad norm 225–380 at steps 340–410, stage 1 < 0.7).
* **Muon removes the adoption dip** (`phase_timelines.png` row 4): hierarchy rms ≥ 1.47 through step 200, margin ≥ 0.26
  in steps 100–500, and no step-30 crash of the main-network gradient share (0.3 vs 0.03 under AdamW). With AdamW the
  prior gives a dip to 0.9 and a margin trough of 0.28–0.31; prior-free 0.6–0.7 and 0.11–0.14.
* **The rank decider pins F, not G**: G − F ends at 0.069 (median of 96 rank-decider runs) vs 0.019 for threshold
  routers (143); the ratio loss is exactly 1.000 under rank decisions, so G is free, as designed.

### A5. The first 400 steps under the microscope: the decoder never "switched away" from an informative hierarchy

Dense checkpoints of the paper 2-stage probe run (`runs/gcp/probe/tiny_hnet2_dyn`, 27 checkpoints, steps 1–400),
8 validation docs per checkpoint (`scripts/interp/early_dynamics.py` → `data/early_dynamics.json`,
`figures/early_dynamics.png`):

| step | bpb | hierarchy zeroed | hierarchy mean-replaced (information) | bypass mean-replaced | cos(hier, bypass) | stage-0 bytes/chunk | stage-1 p in (0.4, 0.6) |
|---|---|---|---|---|---|---|---|
| 1 | 7.98 | +0.10 | **−0.04** | +0.000 | −0.05 | 2.3 | 0.91 |
| 10 | 7.33 | +0.10 | **−0.09** | +0.009 | 0.18 | 7.6 | 0.61 |
| 20 | 6.45 | +0.01 | −0.001 | +0.013 | **0.69** | 40.1 | 0.24 |
| 30 | 5.43 | +0.17 | −0.000 | +0.028 | 0.54 | 3.3 | 0.10 |
| 40 | 4.72 | **+1.85** | +0.001 | +0.14 | 0.11 | 1.8 | 0.65 |
| 50 | 4.29 | **+2.85** | +0.001 | +0.52 | −0.01 | 2.7 | 0.50 |
| 100 | 3.18 | +0.81 | +0.005 | +2.18 | 0.11 | 2.6 | 0.80 |
| 400 | 2.31 | +0.46 | +0.049 | +3.16 | 0.06 | 2.7 | 0.86 |

* **The hierarchy carried no information during the whole "race".** Mean-replacing it *improves* the loss for the
  first 15 steps (it is noise) and costs ≤ 0.005 bits up to step 100 and 0.05 at step 400. The bypass carries
  information from step 35 on (0.14 bits at 40, 2.18 at 100). The large zero-ablation numbers at steps 40–60 (1.9–2.9
  bits) are the decoder depending on the hierarchy's *constant offset*, which it then learns to do without. [M]
* So the work log's picture — the decoder used the hierarchy, then switched to the bypass when it opened, and the
  router lost its signal — is not what happened in this run: there was nothing to switch away from. The bypass is a
  linear map of the encoder state and becomes useful within ~40 steps; the hierarchy needs a trained router, encoder
  and 7-layer main network before it can say anything, and in this run it had not by step 400. [M for the run, I for
  the generalization]
* Consequence for interventions: closing or gating the bypass early (EXP-011 warmup / path-drop, EXP-027 gate) can
  only force the decoder to depend on a signal that carries no information yet; consistent with all of them being
  neutral or harmful. What could help is making the hierarchy informative earlier (a teacher for the router, a chunk
  table that is useful from step 0) — which is what the whitespace warm start does. [I]
* The bypass first aligns with the hierarchy direction (cosine 0.69 at step 20) and then becomes orthogonal to it
  (≤ 0.11 from step 40): the residual projection first learns to reproduce the hierarchy's offset. [M]

## Part B. Per experiment: what was tested, how training went, what the activations show, what went wrong, what to do better

EXP-000/001 (reference re-analyses, no training) and EXP-028 (no runs mirrored locally) are out of scope. Checkpoint numbers: `data/ckpt/<name>.json`, summary `data/ckpt_summary.json`, figures `router_hist_by_class.png`, `probe_accuracy.png`, `decoder_path_ablation.png`, `cka.png`, `bpe_gap.png`.

### EXP-002 / EXP-003 / EXP-006 / EXP-008 / EXP-009 · The 2-stage paper-era models and the oracle

* **Tested.** Paper 2-stage (3,3) H-Net and end-of-chunk routers vs BPE (EXP-002), AdamW LR sweep (EXP-003), n-gram
  tables (EXP-006), training signals (EXP-008), oracle chunk boundaries (EXP-009).
* **Activations [M]** (stage-0 view; "hierarchy" = everything below stage 0):

| checkpoint | bpb | hierarchy info / zero | stage-0 bytes/chunk | ends on whitespace | whitespace hit | word consistency | margin |
|---|---|---|---|---|---|---|---|
| EXP-002 paper s0 @500 | 2.107 | 0.063 / 0.528 | 2.72 | 14% | 31% | 27% | 0.24 |
| EXP-002 paper s0 final | 1.776 | 0.227 / 0.693 | 2.60 | 16% | 38% | 29% | 0.31 |
| EXP-002 end_mlp s1 @500 | 2.108 | 0.057 / 0.436 | 2.71 | 43% | 98% | 74% | 0.40 |
| EXP-002 end_mlp s1 final | 1.765 | 0.272 / 0.688 | 2.60 | 42% | 100% | 83% | 0.41 |
| EXP-008 base s0 final | 1.785 | 0.180 / 0.693 | 2.63 | 17% | 41% | 25% | 0.30 |
| EXP-006 hnet1 s0 final (1-stage, threshold) | 1.834 | **0.025 / 0.646** | 5.70 | 12% | 14% | 52% | 0.30 |

1. **The zero-main probe was a large overestimate in this era**: 0.69 bpb "below stage 0" in EXP-002/008 contains
   0.18–0.27 bits of information; the rest is the constant offset of the hierarchy signal that the decoder expects.
   For the 1-stage `hnet1` (EXP-006) the hierarchy is worth 0.025 bits, not the 0.63 the work log reports ("the main
   network's contribution drops 0.63 → 0.38" with n-gram tables compares two DC offsets).
2. **The stage-0 end_mlp router cuts at every space plus inside words** (whitespace hit 98–100%, 42% of its cuts on
   whitespace, 2.6 bytes per chunk): a word-boundary-respecting character-pair tokenizer, more consistent (83%) than
   the paper router's (29%). The paper router's chunk vector encodes its *first* byte (start semantics, probe 4.5–4.7
   bits); end_mlp's encodes the last byte and 1.8 bits of the first, as the work log found.
3. **EXP-009 oracle chunks [M, metrics]**: `space_end` 1.793 / 1.788 / 1.835 with contributions 0.66 / 0.80 / 0.22:
   even with fixed oracle word chunks one seed of three does not use the hierarchy (and is 0.045 worse). The race
   exists without a learned router; a word tokenizer is necessary but not sufficient — what the oracle lacked was the
   chunk table + next-chunk objective (EXP-018: h1_quant with an 82%-whitespace tokenizer has 0.044 bits of
   hierarchy information; with the table + objective 0.375).
* **What went wrong.** The diagnosis "the main network is barely used (0.04 bpb)" was right in direction but the
  instrument (zeroing) mixes a constant offset with information; the per-class "help" tables in the work log
  inherited the bias (e.g. word-internal letters "helped" by 0.69 bits in zero-ablation vs 0.10 in mean-ablation for
  EXP-008 base).
* **Better.** Mean-ablation (or resample-ablation: the signal of another position) for every "how much does X
  contribute" number.

### EXP-010 · Strong ratio loss, identity residual, Mamba-free outer networks

* **Progress [M].** With the identity residual the bypass exceeds the hierarchy from step 10 (vs 60). Finished seeds
  only: `alpha1` 1.802 / 1.818 (zero-probe contribution 0.57 / 0.94), `resid_id_s0` 1.783 (0.32),
  `alpha1_resid_id_s2` 1.789 (0.24); five of nine runs stopped at step 750, SWA at 250.
* **Wrong / better.** Arm-level statements rest on one or two finished seeds; "identity residual starves the hierarchy"
  is supported in direction (0.24–0.32 vs 0.57–0.94, zero probe), bpb differences are within seed spread. Print the
  number of finished seeds next to every arm value.

### EXP-011 · Router-free fixed (3,3) hourglass + training signals

* **Progress [M].** A clean test of the bypass race without a router (margin 0.5 by construction). Identity residual:
  bypass ahead from step 20, contribution 0.05 / 0.20 / 0.05; zero-init: from step 60, 0.92 / 0.77 / 0.81 (bpb 1.823 /
  1.834 / 1.839); path-drop: 0.06 / 0.03; residual warmup 0.31 / **1.56** / 0.59 at identical bpb (1.839 / 1.847 /
  1.848).
* **Mechanism.** Contribution and quality are decoupled: a forced dependency relocates work without adding
  capability (A5: early on there is nothing for the hierarchy to say). The same mistake recurs in EXP-026 and EXP-027,
  where the arm with the largest contribution is not the best. [I]

### EXP-012 · Pooling chunker

* **Progress [M].** Finished: `select` 1.917 / 1.820, `pool_mean_s0` 1.789; the other four runs stopped at steps
  250–750; `select_s0` has a burst of grad spikes (240–1000) and a stage-1 compression collapse at 270.
* **Wrong / better.** One finished seed per pooling arm and 0.1 bpb between the two `select` seeds: no conclusion is
  supported. The motivation ("the chunk vector does not know its chunk") holds for character-pair chunks only; with
  word chunks the boundary state carries 2.7–2.9 bits of chunk identity (EXP-018/023 section).

### EXP-013 · BPE's LR (6e-3) for the H-Net

* **Progress [M].** Finished: `lr6_s2` 1.867, `lr6_alpha1` 1.927 / 1.940, `lr6_nomult_s0` **1.785**, `lr6_rent`
  1.907 / 1.781. The bypass crosses at step 30 (vs 60–90) and the hierarchy dip comes at 40–50 instead of 150–190;
  `lr6_alpha1` has 4–5 grad spikes per run.
* **Reading [I].** High LR compresses the race into the first 50 steps and widens the seed spread rather than
  breaking the model (`lr6_nomult_s0` equals the best 2-stage runs). The per-stage multipliers put stage 0 at
  ~2e-2 effective, where the spikes sit.

### EXP-014 · First rank decider (p-space quantile)

* **Progress [M].** `tau` = 1.000 at stage 0 or 1 in all 9 runs between steps 20 and 270 (stage 0 in 8, median 175);
  compression collapses follow. A `tau ≥ 0.999` assertion would have stopped the sweep at step ~100. Mechanism as in
  the work log.

### EXP-017 · Stage-0 chunk vocabulary (2-stage)

* **Progress [M].** `vocab` 1.802 / 1.774 / 1.777 vs `quant_fix` 1.794 / 1.804 / 1.798. Adding next-chunk prediction
  raises the raw-score std peak to 9.8–13 (vs 4.1–4.3) and drops the margin trough to 0.12–0.13 (vs 0.30): the
  signature of the non-adopting 1-stage seeds. Consistent with the work log (2.6-byte chunks are byte bigrams).

### EXP-018 / EXP-023 · Word-scale chunk vocabulary: the word seeds vs the suffix seed

* **Tested.** 1-stage, ratio 6, end_mlp router with rank decisions, a 2 × 65 536 × 96 hashed chunk table added to
  the chunk vectors, a sampled-softmax next-chunk loss (w 0.5); EXP-023 changes the decider (trimmed statistics) and
  adds `vp_ng` (byte n-gram tables) and `vp_rwarm` (residual warmup).
* **Outcome [M].** `h1_vocab_pred` 1.720 / 1.863 / 1.737, `vp` 1.732 / 1.855 / 1.733, `vp_rwarm` 1.810 / 1.875 /
  1.745, `vp_ng` 1.720 / 1.669 / 1.656. The losing seed is the same seed in every arm of a parameter layout (s1 without
  n-gram tables, s0 with them; `seed_rankings.json`), because those arms share the initialization exactly and the
  data order is seed-independent.
* **Progress [M].** Steps 0–130 are indistinguishable across seeds (A1). From step ~150 the router margin falls in
  every seed (trough 0.12–0.14 at 300–400); the hierarchy rms separates from step ~200 (within-arm r −0.64 at step
  250) and the losing seed is identifiable from it at step ~210 (A2).

What the checkpoints show (`data/ckpt/*.json`, 32 val docs; "info" = bits lost when the signal is replaced by its
mean, which removes its information but keeps its scale; "zero" = the trainer's zero-main probe):

| checkpoint | bpb | hierarchy info / zero | bypass info | boundaries on whitespace | on last letter of a word | inside words | word consistency | singletons | router score R² from the current byte / last 4 bytes | margin |
|---|---|---|---|---|---|---|---|---|---|---|
| vp s0 @500 (word) | 2.127 | 0.082 / 0.166 | 3.17 | 91% | 0.2% | 4.7% | 92.5% | 31% | 0.47 / 0.97 | 0.23 |
| vp s0 @1000 (word) | 1.915 | 0.248 / 0.459 | 3.35 | 96% | 0% | 0% | 99.8% | 30% | 0.97 / 1.00 | 0.44 |
| h1_vocab_pred s0 final (word) | 1.703 | **0.375** / 0.752 | 3.34 | 97% | 0% | 0% | 99.8% | 30% | 0.98 / 1.00 | 0.41 |
| vp s1 @500 (suffix) | 2.172 | **0.002** / 0.068 | 3.21 | 0% | 64% | 27% | 75% | 40% | 0.97 / 1.00 | 0.11 |
| vp s1 @1000 (suffix) | 2.017 | 0.017 / 0.092 | 3.05 | 0% | 56% | 40% | 82% | 40% | 0.51 / 0.99 | 0.26 |
| vp s1 final (suffix) | 1.840 | 0.080 / 0.190 | 3.28 | 4% | 57% | 35% | 89% | 40% | 0.26 / 0.97 | 0.36 |
| h1_vocab_pred s1 final (suffix) | 1.848 | 0.056 / 0.127 | 3.24 | 0% | 55% | 42% | 88% | 40% | 0.31 / 0.97 | 0.37 |
| vp_ng s2 final (word) | 1.637 | 0.248 / 0.413 | 3.39 | 94% | 0% | 0% | 96% | 30% | 0.88 / 0.97 | 0.43 |
| h1_quant s0 final (no vocabulary) | 1.846 | 0.044 / 0.174 | 3.47 | 82% | 1% | 10% | 83% | 35% | 0.46 / 0.95 | 0.26 |

1. **The two basins are "cut after the space" and "guess where the space will be".** The word seeds cut at the
   whitespace byte itself (rate 1.00 at spaces and newlines by step 1000); their router is a function of the current
   byte (R² 0.88–0.98 from the byte alone). The suffix seeds never cut at a space; 55–64% of their cuts are on the
   last letter of a word and the rest inside words, and their score separates "the next byte is a non-letter" from
   "the next byte is a letter" with AUC 0.75–0.80 (word seeds: 0.09–0.55, i.e. no such signal). Under end
   semantics the router cannot see byte t+1, so a tokenizer that ends chunks *before* the space has to predict word
   ends; it fires after likely word endings (`s`, `d`, `ed`, `ing`, `ly`), which is why its frequent chunks look
   morphological. At step 500 both routers are unigram rules (R² 0.97 from the current byte): the word seed's rule
   is "the space byte", the suffix seed's is "a few word-final letters" (`s`, `,`, `d`, `n`, `r`). [M]
2. **At step 500 the suffix seed's hierarchy carries no information at all**: mean-replacing it costs 0.002 bpb; the
   trainer's zero-main probe reports 0.068 because zeroing also removes the hierarchy's constant offset. The word
   seed at step 500 carries 0.082 bits of information (0.166 by the zero probe). **The zero-main probe overstates the
   information carried by the hierarchy by about 2× in every checkpoint** (word finals 0.375 vs 0.752, 0.248 vs 0.413;
   prior 0.356 vs 0.568). [M]
3. **The chunk table is used only in the word basin, and only without n-gram tables** (`vocab_ablation.json`, 24
   docs, table output zeroed at inference): h1_vocab_pred s0 (word) +0.445 bpb — more than mean-replacing the whole
   hierarchy (+0.391): the main network depends on it; vp s0 @1000 (word) +0.161; suffix seeds +0.008 / +0.009; with
   n-gram tables the word-basin models barely use it (vp_ng s2 +0.014, prior s0 +0.034, while their hierarchies carry
   0.25–0.36 bits). With 88% consistency and 40% singletons suffix chunks are not types a table can store; with n-gram
   tables the hierarchy's value comes from elsewhere. The work log's "the chunk table gives the hierarchy word
   identity" holds for the no-n-gram layout only. [M]
4. **Where the hierarchy helps (word seed, final, bits per predicted byte, info):** punctuation 1.65, newline 1.19,
   first letter of a word 0.52, last letter 0.45, space 0.26, word-internal 0.24; by position in the word it decays
   from 0.35 (letter 0) to 0.09 (letter 4+) at step 1000. Suffix seed: ≤ 0.12 everywhere, ~0 at punctuation, newline
   and space. The hierarchy's value is sentence structure (what comes after a word) and the identity of the next
   word's first letters. [M]
5. **The chunk vector does know its chunk once chunks are words.** Linear probes on the boundary state x_enc (before
   the table): word seed final, chunk identity among the 63 most frequent chunks 2.76 bits of information (accuracy
   0.82 vs unigram), chunk length 1.77 bits, first byte 2.09 bits. The EXP-002 finding "the chunk vector encodes its
   last byte and barely its first" was a property of character-pair chunks and the 2-stage paper router, not of
   the architecture. The table adds 0.1–0.45 bits of identity on top. [M]
6. **The main network adds little linearly decodable next-chunk information**: next chunk's first byte from z vs x:
   +0.33 bits (word final), +0.22 (suffix final); it loses 0.3–0.45 bits of identity on the way. Its attention is
   diffuse: mean offset 35–51 chunks, entropy 3.1–3.7 nats, 6–15% of the mass on the current and previous chunk. In
   the word seed attention becomes more local with training (6.9% → 13.5% on offsets 0–1 from step 500 to 1000). [M]
7. **The bypass is indispensable in every model** (mean-replacing it costs 3.0–3.5 bpb) and by itself, read through
   the decoder's final norm and the LM head, predicts the next byte at 2.7–3.7 bpb; the hierarchy signal read the
   same way gives 4.9–5.7 bpb. The two signals are orthogonal (cosine 0.01–0.03). The decoder is a byte model on the
   bypass that reads the hierarchy as side information. [M]
* **What went wrong.** The objective (CE through the STE + next-chunk CE) is satisfied by either tokenizer early on;
  which unigram rule the router locks into at steps 150–500 is set by the initialization; once in the suffix basin
  the table is useless, the hierarchy carries no information at step 500, and the decoder learns everything on the
  bypass. Recovery afterwards (vp s1: 0.002 → 0.080 bits from step 500 to the end) is slow.
* **Better.** (i) Recognize the basin at step ~250 (hierarchy rms, or directly: fraction of boundaries on
  whitespace, available from the batch for free) and restart or re-seed instead of finishing the run; (ii) since the
  basin is an init property, screen inits with 300-step probes; (iii) give the router the information it lacks
  instead of a prior (Part C: a one-byte-delayed decision or start-of-word semantics would make the word boundary
  observable for both basins).

### EXP-019 · Context-free (`local_mlp`) router

* **Progress [M].** `local` 1.814 / 1.812 / 1.812 — the most seed-stable 2-stage arm in the project (sd 0.001 vs 0.005
  for the contextual router at equal mean); its score std stays ≤ 5.4. Part B shows that even the contextual end_mlp
  router is 90–100% a function of the last 4 bytes (4-gram R² of its score, 20 checkpoints), so context-freedom costs little; what it
  buys is that the score distribution cannot grow heavy tails.

### EXP-020 · Router self-consistency penalty

* **Progress [M].** `cons10` ends with margin 0.014–0.071 (quant_fix 0.33) and raw-score std 6.8–11.5 (2.8–3.3); bpb
  1.775 / 1.783 / 1.840. The normalized penalty (within-group / total variance) was satisfied by inflating the total
  variance with a few extreme groups; trimmed standardization then squeezes the bulk to p' ≈ 0.5 [I: no checkpoint].
  The "neutral, consistency is free" reading hid an undecided router. **Better:** log numerator and denominator of
  every ratio penalty.

### EXP-021 · median/MAD decider (stopped)

No local metrics. The work-log account (MAD → 0 under a bimodal score) matches the measured score histograms: in
word-basin checkpoints 84–86% of positions sit in one narrow mode (`router_hist_by_class.png`).

### EXP-024 · Whitespace warm start (`prior`) and `temp4`

* **Outcome [M].** `prior` 1.6337 / 1.6346 / 1.6364, contribution 0.56 / 0.57 / 0.60; `temp4` 1.753 / 1.713 / 1.704,
  0.02 / 0.14 / 0.21 (same seed order as every other n-gram-layout arm: s0 worst).
* **Progress [M].** The prior keeps the hierarchy rms at ≥ 0.94 through the dip (prior-free: 0.6–0.7), the margin
  trough at 0.28–0.31 (prior-free 0.11–0.14), and the score-std peak earlier (step 200 vs 330–450). R at step 250 is
  0.83–0.86 vs 0.36–0.63 without the prior.
* **Activations (prior s0 final) [M].** A whitespace tokenizer that survived release: 96% of boundaries on
  whitespace, 99.4% of whitespace hit, 98.5% word consistency, router R² 0.94 from the current byte. Hierarchy
  information 0.356 bpb (zero probe 0.568); by class: punctuation 2.13 bits, newline 1.13, digit 0.97, first letter
  0.55, space 0.23; by position in word 0.55 → 0.12 from letter 0 to letter 4. Table ablation costs only 0.034 bpb
  (`vocab_ablation.json`): in the n-gram layout the chunk table is not where the hierarchy's value is.
* **Against BPE on identical text** (`bpe_compare.json`, 48 docs, BPE = EXP-006 bpe_s0 at lr 1.5e-3, 1.618 on
  these docs vs 1.620): the H-Net is better on short and rare tokens (gap −0.52 / −0.20 / −0.06 bits per byte on
  2 / 3 / 4-byte GPT-2 tokens, −0.13 on word pieces, −0.21 on numbers) and worse on long tokens (+0.08 / +0.10 /
  +0.09 on 6 / 7 / 8+-byte tokens), punctuation (+0.12) and whitespace/newline tokens (+0.31). Parity is a balance
  of two opposite errors, not a uniform match: BPE's advantage is memorized long words, the H-Net's is sub-word
  generalization. Every other H-Net loses on long tokens by +0.17 to +0.32 and on 1-byte tokens by +0.23 to +0.50.
* **What went wrong.** The comparison point. BPE's own LR optimum (6e-3) is 0.07 bpb better and the prior arm was
  never LR-swept. And the "prior" is BPE's pretokenizer: at ratio 6 on English the space period (1/6.4 of bytes) fills
  the boundary budget almost exactly, which is why this prior can be "released" without drift. At the ladder's ratio
  4.1–4.5 it cannot (ladder section).
* **Better.** Sweep the warm-start arm's LR (it is the only arm that can afford it: seeds agree to 0.003), and report
  the gap to LR-tuned BPE as the headline.

### EXP-025 · Muon vs AdamW

* **Result [M].** `prior_muon` 0.025: **1.630 / 1.635 / 1.630** (contribution 0.38 / 0.36 / 0.40); 0.05: 1.816 /
  1.778 / 1.774 (0.01 / 0.26 / 0.25). `bpe_muon` 0.05: 1.713 / 1.710 / 1.707 (0.025 stopped at step 750).
* **Mechanism [M].** Bypass rms grows linearly under Muon (no weight decay on `residual_proj`): 9.3 (0.025), 26–31
  (0.05). At 0.05 the stage-0 grad norm explodes (225–380) at steps 340–410 in two seeds; one never re-adopts. Muon has
  no adoption dip (A4). The Muon BPE is worse than AdamW BPE because its embedding + head (13 of 15M parameters) stay
  on Adam at lr 1.5e-3 [I from the config: `lr_embed = lr_head = 0.0015`] — the comparison is not about Muon.

### EXP-026 · Mamba-free outer networks

* **Result [M].** `swa` 1.742 / 1.744 / 1.741, contributions 0.58 / 0.65 / 0.96 (higher than the Mamba prior's
  0.56–0.60, at +0.11 bpb); `kda` failed at step 270. A weaker outer network makes the hierarchy carry more and the
  model worse: contribution is not quality.

### EXP-027 · Chunk STE and margin-gated bypass

* **Tested.** `cste` (next-chunk loss reaches the router through the chunk vectors), `rgate` (bypass × router
  margin), both; base = `vp_ng` (n-gram layout).
* **Outcome [M].** cste 1.778 / 1.661 / 1.640; cste_rgate 1.742 / 1.679 / 1.648; rgate 1.746 / 1.699 / 1.653;
  contributions 0.02 / 0.41 / 0.55, 0.28 / 0.35 / 0.53, 0.77 / 0.25 / 0.51. **Seed order s2 < s1 < s0 in all three
  arms, as in vp_ng and temp4** (`seed_rankings.json`): the interventions did not touch the lottery because they do
  not touch the initialization that decides it.
* **Progress [M].** `cste_s0` keeps a *decided* router (margin 0.36–0.39 throughout) and still ends with a dead
  hierarchy (0.02): the "undecided router → non-adoption" link of the work log is not necessary. In `rgate` the gate
  value fell with the margin in the losing seeds (s0: 0.80 → 0.11; cste_rgate s0: 0.76 → 0.21) and the pre-gate bypass
  grew to compensate (rms 6.8 and 4.6 vs 2.4–2.5 in the winning seeds), so the effective bypass stayed at 0.74–0.96:
  the optimizer undid the gate. rgate s0 has the largest contribution of the arm (0.77) and the worst bpb.
* **What went wrong.** The arms intervene on the race (timing of adoption), while the outcome is set by which basin
  the router enters, which is fixed by the initialization; contribution was again read as quality.
* **Better.** Test interventions on the losing *initialization* (rerun the exact s0 init with the intervention and
  check whether its basin changes) instead of on three fresh seeds; for gates, log the effective (post-gate) bypass.

### LADDER (1e16 FLOPs, L03–L06, Muon 0.02, mix_v1) · checkpoints L03 / L04 analysed here

* **Outcome [M].** Gap to BPE 0.18–0.29 bpb prior-free, 0.13–0.24 with the warm start (WORKLOG). Bypass rms ends 5.6–
  8.2 vs hierarchy 1.0–1.2 in all four H-Nets (R 0.12–0.22).
* **Activations [M]** (32 FineWeb-Edu val docs, i.e. the English part of the mix only):

| checkpoint | bpb (these docs) | hierarchy info / zero | bytes/chunk | boundaries on whitespace / last letter / inside words | word consistency | top chunks | margin |
|---|---|---|---|---|---|---|---|
| L03 hnet (prior-free) | 1.739 | **0.042** / 0.069 | 4.28 | 5% / 57% / 25% | 60% | `s`, `e`, ` th`, `,`, `.` | 0.33 |
| L04 hnet (prior-free) | 1.770 | **0.030** / 0.060 | 4.74 | 8% / 42% / 36% | 56% | `e`, `h`, ` `, `o`, ` t` | 0.15 |
| L03 hnet_ws (warm start) | 1.670 | **0.265** / 0.389 | 4.46 | 71% / 14% / 0.4% | 85% | ` `, `the `, `and`, `of ` | 0.28 |
| L04 hnet_ws (warm start) | 1.703 | **0.241** / 0.335 | 4.45 | 71% / 15% / 0.4% | 81% | `the `, `and`, `of `, `\n` | 0.28 |

* **The prior-free ladder H-Nets lost the race in the same way as the tiny suffix seeds** (predicted-word-end and
  sub-word cuts, ≤ 8% on whitespace, a hierarchy worth 0.03–0.04 bits, router AUC for "next byte is a non-letter"
  0.59–0.91), with one seed each, so the ladder's prior-free numbers measure the losing basin, not the architecture.
  The table is unused there (ablation +0.003 bpb). **The warm-start ladder H-Nets have a working hierarchy**
  (0.24–0.27 bits) at the same bypass/hierarchy rms ratio as the dead ones (R 0.20–0.22 vs 0.12–0.14): at ladder
  scale too, the rms ratio does not tell a dead hierarchy from a live one.
* **Why the warm start fits the ladder worse than tiny [I, from the measured numbers].** The ladder's target ratio is
  4.15 bytes per chunk, but whitespace occurs every ~6 bytes in English; the rank decider must place ~30% of its
  boundaries elsewhere (measured: 71% on whitespace, the rest on predicted word ends and punctuation), so the
  released whitespace tokenizer cannot be kept as is and the extra boundaries are of the suffix kind. At tiny scale
  ratio 6 ≈ the space period. The hierarchy's information at L03/L04 (0.24–0.27) is below the tiny prior's (0.36).
* **What went wrong.** Conclusions ("the bypass dominates in every prior-free run", "the race may be a small-scale
  artefact") were drawn from rms ratios, without a zero-main or mean-ablation probe on the ladder and with one seed
  per rung, while the tiny experiments had already shown that the seed decides the basin.
* **Better.** Add the (mean-)ablation probe to the ladder eval; log the fraction of boundaries on whitespace per
  eval (one line of code, it identifies the basin); set the ladder H-Net's ratio so that whitespace can fill the
  budget (≈ 6 on English-heavy mixes) or let the extra boundaries come from a principled rule (next-byte entropy).

## Files

| path | what |
|---|---|
| `scripts/interp/run_all.py` | entry point (`--quick`: smoke test into `logbook/interp/quick/`; `--skip-ckpt`: metrics only) |
| `scripts/interp/common.py` | run enumeration (runs.csv + live mirrors, settings from each run's config.yaml), loaders |
| `scripts/interp/phases.py` | events, phase timings, snapshots, cross-run statistics → `data/phases_runs.json`, `data/phases_summary.json`, `figures/phase_timelines.png`, `phase_events.png`, `bypass_scale_vs_lr.png`, `early_predictors.png` |
| `scripts/interp/predictability.py` | step-resolved within-arm correlation / AUC → `data/predictability.json`, `figures/predictability_vs_step.png` |
| `scripts/interp/seed_rankings.py` | seed order per arm and layout family → `data/seed_rankings.json` |
| `scripts/interp/init_basin.py` | step-0 router statistics per run's exact init → `data/init_basin.json` |
| `scripts/interp/acts.py` | capture / path-ablation / logit-lens / attention-profile wrapper around the outer H-Net stage |
| `scripts/interp/analyze_ckpt.py`, `run_ckpts.py` | per-checkpoint analysis (router, tokenizer, paths, probes, attention) → `data/ckpt/*.json` |
| `scripts/interp/figures_ckpt.py` | `data/ckpt_summary.json`, `figures/router_hist_by_class.png`, `probe_accuracy.png`, `decoder_path_ablation.png` |
| `scripts/interp/early_dynamics.py` | dense steps 1–400 → `data/early_dynamics.json`, `figures/early_dynamics.png` |
| `scripts/interp/cka.py` | cross-checkpoint linear CKA, boundary Jaccard → `data/cka.json`, `figures/cka.png` |
| `scripts/interp/bpe_compare.py` | token-aligned H-Net vs BPE → `data/bpe_compare.json`, `figures/bpe_gap.png` |
| `scripts/interp/vocab_ablation.py` | chunk-table ablation → `data/vocab_ablation.json` |

Not analysed at checkpoint level (no local weights or skipped for time): EXP-017/019/020/025/026/027 (metrics only),
EXP-009 (checkpoint files without weights), EXP-002 `paper_s0`/`end_mlp_s1` at step 1000, `end_cosine_shift`,
`paper_nosmooth`, EXP-013 `lr6_nomult`, EXP-006 `hnet1_ng` (in `run_ckpts.py`; rerun fills them). Ladder checkpoints
were evaluated on FineWeb-Edu documents only, not on the code / multilingual parts of mix_v1.
