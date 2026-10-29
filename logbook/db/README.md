# Experiment database (`logbook/db/`)

A normalized (3NF-style) relational record of every experiment of the H-Net vs BPE project, as UTF-8 CSV tables with a
header row. One fact per cell, no repeating groups; every non-key column depends on the whole key of its table.
`NULL` is written literally for "unknown / not applicable".

Sources of truth, in priority order: `logbook/EXPERIMENTS.md`, `logbook/WORKLOG.md`, the sweep specs
`configs/search/*.yaml` (arms, `set:` overrides, `report.labels`), the run directories
`runs/gcp/<exp>/<arm>_s<seed>/{eval,metrics}.jsonl` + `done.json`, and the live mirrors of EXP-025/026/027 and the
ladder sweeps. Where the logbook and `eval.jsonl` disagree, `eval.jsonl` wins and the row's `notes` say so.
`SNAPSHOT.txt` records when `runs.csv` was harvested; runs that were in flight then carry `status = running`.

`python logbook/db/build_views.py` prints three denormalized markdown views (`big`, `runs`, `findings`); `counts`
prints row counts; `verify` re-reads every `run_dir` in `runs.csv` and reports values that have moved since the
snapshot (expected for running experiments).

## Tables

### `experiments.csv` — one row per experiment or sweep
Key: `exp_id` (`EXP-000` … `EXP-028`, plus `LADDER-LR-1e16`, `LADDER-LREXT-1e16`, `LADDER-<budget>-<rung>`).

| column | meaning |
|---|---|
| `name` | title as in EXPERIMENTS.md / the spec header |
| `spec_file` | sweep spec (or analysis script for EXP-000/001); `NULL` for EXP-007 (number never used) |
| `date` | YYYY-MM-DD from the work log / git history |
| `question` | one sentence: what it tested |
| `status` | `done` · `running` · `stopped` (launched, not completed, or never launched after being built) · `voided` (results invalid) · `superseded` (spec replaced before launch) · `planned` (spec written, not launched) |
| `scale` | `tiny_2m` (≈2M non-embedding params, FLOPs-matched to `tiny_bpe_2m`), `ladder_L0x`, `reference data` |
| `data` | `finewebedu10B` (train shard 1, GPT-2 shards decoded to bytes) or `mix_v1` |
| `budget_tokens_eq` | training budget per run (`50M` token-equivalents for tiny runs; BPE tokens and FLOPs for the ladder) |
| `seeds` | planned seeds per arm |
| `gpu` | `L4` (spot, 3 runs per VM) or `A100` (spot); `none` for analyses |
| `cost_usd` | only where the work log records a figure (see `notes` for whether it is a worst case) |
| `outcome` | one-sentence result |
| `worklog_section` | heading text of the WORKLOG.md section that carries the result |
| `category` | grouping used by the big view: `baselines & calibration`, `router / decision rule`, `hierarchy use & bypass`, `chunk vocabulary & memory`, `curricula & priors`, `optimizer & mixers`, `ladder / scaling` |
| `notes` | provenance remarks, inconsistencies between sources |

### `arms.csv` — one row per arm (configuration) of an experiment
Key: (`exp_id`, `arm`). FK `exp_id → experiments`. `parent_arm` is `"<exp_id>/<arm>"` → `arms`.
For grid sweeps (EXP-003, EXP-014, EXP-025, ladder LR sweeps) one arm per grid point, named as the run directories
(`end_mlp__lr0.0015`). `base_config` is the run yaml the arm starts from; `label` is the human description
(`report.labels` in the spec where present).

### `arm_settings.csv` — the settings that define an arm (entity–attribute–value)
Key: (`exp_id`, `arm`, `setting`). FK (`exp_id`, `arm`) → `arms`. `value` is the string as it appears in the spec
(or base config); `origin` says where it comes from: `base_config` (the run/model yaml the arm includes),
`spec_set` (the spec's top-level `set:`), `run_set` (the arm's own `set:`), `derived` (e.g. `mixer` from
`mixer_codes`). Setting names drop the `model.` / `model.dc.` / `train.` prefixes (`router`, `decision`,
`chunk_vocab`, `ngram_embed`, `router_prior_weight`, `residual_init`, `residual_warmup`, `residual_gate`,
`chunk_ste`, `router_consistency_weight`, `optimizer`, `lr`, `muon_lr`, `adam_part_lr`, `schedule`,
`target_ratios`, `tokenizer`, `mixer`, `n_stages`, `arch_layout`, `d_model`, …). `decision_version` records which
rank-decider implementation an arm used (p-space quantile · mean/std · median/MAD · trimmed 98%). Infra-only
overrides (`train.compile`, `eval.*`) are not recorded.

Base configs: `configs/runs/tiny_bpe_2m.yaml` (BPE, GPT-2 50K vocab, 10 layers d128, AdamW 1.5e-3, WSD 10%/20%,
50M tokens); `configs/runs/tiny_hnet2_2m.yaml` (2-stage (3,3) H-Net m4/[T1m4/[T7]/m4T1]/m4, d 96/128/256,
cosine_lookahead router, threshold decision, α 0.03); `configs/runs/tiny_hnet1_2m.yaml` (1-stage ratio 6,
m4/[T8]/m4, d 96/256); `configs/ladder/runs/ladder_{bpe,hnet}.yaml` + `configs/ladder/models/*_L0x.yaml`
(mix_v1, Muon, docs/LADDER.md).

### `runs.csv` — one row per (arm, seed) run
Key: (`exp_id`, `arm`, `seed`). FK (`exp_id`, `arm`) → `arms`.

| column | meaning |
|---|---|
| `run_dir` | directory the numbers were read from (repo-relative, or absolute for the live mirrors); `NULL` for work-log-only runs |
| `status` | `done` (done.json rc 0, or final step reached) · `running` · `failed` (crash/divergence, or sweep_status `failed`) · `stopped` (unfinished, not running) · `voided` (EXP-014 collapsed runs, EXP-008 combo arms) |
| `steps_done` / `steps_total` | last logged step (metrics or eval) / the run's planned steps (1763 tiny H-Net, 1744 tiny BPE, 908/728/395/243 for ladder L03–L06 at 1e16) |
| `final_step` | step of the last `eval.jsonl` record with `val/bpb` |
| `val_bpb_final` | that record's `val/bpb`, rounded to 4 decimals |
| `zero_main_bpb_final` | that record's `val/zero_main_s0_bpb` (zero-main probe; `NULL` if the probe was off: EXP-002/003/006 and BPE runs) |
| `main_contribution` | `zero_main_bpb_final − val_bpb_final` (bits the innermost hierarchy is worth) |
| `train_bpb_final` | `bpb` of the last `metrics.jsonl` record |
| `adopted` | the work log's judgment whether the decoder adopted the hierarchy (`yes`/`no`), else `NULL` |
| `tokenizer_kind` | from `chunk_stats` sections: `word`, `suffix` (morpheme tokenizer), `mixed` (2-stage: stage-0 character pairs + stage-1 non-tokenizer; or h1_quant's partial word alignment); `NULL` where no chunk_stats was run |
| `source` | `eval_jsonl` · `worklog` · `experiments_md` |
| `notes` | provenance (alternative mirrors, final-step-without-done.json), crash causes, logbook discrepancies |

### `arm_results.csv` — per-arm aggregate over finished seeds
Key: (`exp_id`, `arm`). FK → `arms`; `reference` → `references.reference_id`.
`n_seeds` counts runs with `status = done`; `val_bpb_sd` is the sample standard deviation (`NULL` for n < 2).
`delta_vs_bpe` = mean − 1.634 for tiny-scale experiments (finewebedu10B, 50M token-eq; `reference = REF-BPE-TINY`),
`NULL` for the ladder, whose `reference` is the rung's own BPE run; `delta_vs_reference` is the difference to the
named reference for every arm. `verdict` is a short phrase (hand-written for the arms the logbook discusses,
rule-based otherwise); `notes` carries interim values of running arms.

### `references.csv` — the baselines everything is compared to
Key: `reference_id`. FK (`exp_id`, `arm`) → `arms`. `val_bpb` and `n_seeds` are computed from `runs.csv`
(e.g. `REF-BPE-TINY` = EXP-006 bpe 1.634 ± 0.002; `REF-PRIOR` = EXP-024 prior 1.635; `REF-LADDER-BPE-L04-1e16` = 1.622).

### `findings.csv` — one row per finding
Key: `finding_id`. FK `exp_id → experiments` (the experiment the finding came out of, one per row).
`category` ∈ {training-dynamics, router-objective, architecture, optimizer, data, infra, scaling}; `statement`
(one sentence), `evidence` (the numbers), `consequence` (what changed in code or plan), `worklog_section`.

### `components.csv` — code-level mechanisms
Key: `component`. FK `introduced_in_exp → experiments`. `module_path` (file / class), `config_key`, `description`,
`status` ∈ {kept, neutral, harmful, untested} summarizing the experimental verdict.

## Foreign-key summary

```
experiments(exp_id)
  ← arms(exp_id)                     parent_arm → arms(exp_id/arm)
      ← arm_settings(exp_id, arm)
      ← runs(exp_id, arm)
      ← arm_results(exp_id, arm)     reference → references(reference_id)
      ← references(exp_id, arm)
  ← findings(exp_id)
  ← components(introduced_in_exp)
```

## Known inconsistencies between sources (recorded in `notes` columns)

* EXP-003: EXPERIMENTS.md says "~40/60 done"; 55/60 runs are finished in the run dirs.
* EXP-006 `hnet1_ng`: the work-log harvest "1.722 ± 0.023 (2 seeds)" used `hnet1_ng_s0`, which stopped at step 1680 (1.7376 at the step-1500 eval); only `hnet1_ng_s1` (1.7054) finished.
* EXP-008 `mainaux`: stage-0 compression collapsed to 1.25 bytes/position, far over the FLOPs budget; the "aux head is marginal (1.821 vs 1.823)" readout is not an honest-compute comparison.
* EXP-010 SWA: the EXP-026 spec says "EXP-010's SWA arm ended at 2.005 bpb"; the only record is `swa_s0` at step 250 (2.8607).
* EXP-012, EXP-013, EXP-019 are "running" in EXPERIMENTS.md; the run dirs show them stopped (012/013, incomplete) or finished (019, all 9 runs done: local 1.813, local_vocab 1.838, local_vocab_pred 1.836, never written up).
* EXP-023 `vp_ng`: the EXP-027 entry and spec say "seed 1 not adopted"; by the zero-main probe the non-adopted seed is seed 0 (1.7201, contribution 0.09); seeds 1 and 2 (0.41 / 0.42) adopted.
* EXP-025 and EXP-026 have no EXPERIMENTS.md entry; their finals exist only in the live mirrors (prior_muon mlr 0.025 = 1.631; swa = 1.742; kda failed).
* LADDER-1e16-L04 `bpe_s0` has byte-identical eval records to LADDER-LR-1e16 `bpe__mlr0.02_s0` (the LR-sweep run reused as the rung's BPE run).
