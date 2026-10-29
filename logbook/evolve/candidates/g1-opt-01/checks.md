# g1-opt-01 checks

## Tests
`tests/test_muon_lr_mult_kinds.py` (new: default groups and 3 training steps bit-identical to the flag set to every
kind; [embed, head, other] gives Muon groups lr_mult 1 and leaves every other group, name and base LR unchanged;
exact parameter cover; unknown kind rejected; forward pass independent of the flag; model-config route ≡ train.muon
route, and refused under AdamW) plus `test_optim_groups`, `test_speedrun_muon`, `test_model_hnet`,
`test_invariances`, `test_flops`: **170 passed, 4 skipped**. Also `test_train_config`, `test_train_resume`,
`test_search_fairness`, `test_model_official`, `test_chunk_pred`, `test_train_cli`, `test_train_e2e`: pass.
`test_ladder.py::test_committed_files_match_the_generator` fails on the starting commit too (docs/LADDER.md was
hand-edited after generation); unrelated.

## Fairness
`make_gen.py` spec with controls bpe, hnet_ws + this candidate → `scripts/search/fairness.py`: **Problems: none**.
Params 20.01M (20,010,025, identical to hnet_ws), 0.02037 GFLOPs/byte for all three arms. The knob lives in the
model config (`model.lr_multiplier_kinds`), next to `lr_multiplier`, so the strict gate's held-equal Muon section
is identical to BPE's.

## Peak LRs actually applied (trainer startup line `peak_lr_by_group`, benchmark spec values)
| group | parent hnet_ws | g1-opt-01 |
|---|---|---|
| muon / stage 0 (Mamba in/out_proj x4, chunk_vocab.pred_proj, residual_proj) | 0.04702 | **0.02** |
| muon / stage 1 (main Transformer) | 0.02 | 0.02 |
| embed / stage 0 (byte emb, n-gram + chunk-vocab tables) | 0.007053 | 0.007053 |
| head / stage 0 | 0.007053 | 0.007053 |
| other / stage 0 (norms, Mamba scalars, conv, router), wd 0 and 0.1 | 0.007053 | 0.007053 |
| other / stage 1 | 0.003 | 0.003 |
Muon's own sqrt(rows/cols) factor still applies on top (in_proj 1094x192: x2.39).

## Smoke / mechanism probe (CPU)
L03 hnet_ws model, the benchmark's full 908-step schedule (10% warmup, momentum warmup 90) and LRs, but one
2048-byte window per step on the local FineWeb-Edu shard (mix_v1 train shards are not local), first 120 steps.
Loss decreases (bpb 7.68 → 2.60), no NaN, compression 4.13-4.14 (target 4.145). Parent = same run without the flag.

| step | resid lr (parent / g1-opt-01) | W_r spectral norm | W_r eff. rank | residual_rms | dechunk_rms | bypass ratio | train bpb (1 window) | compression |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0005 / 0.0002 | 0.00 / 0.00 | 103 / 103 |  |  |  |  |  |
| 10 | 0.0052 / 0.0022 | 0.03 / 0.01 | 87 / 86 | 0.05 / 0.04 | 1.10 / 1.10 | 0.04 / 0.04 | 7.618 / 7.682 | 4.14 / 4.14 |
| 20 | 0.0103 / 0.0044 | 0.11 / 0.05 | 92 / 88 | 0.09 / 0.06 | 1.10 / 1.09 | 0.09 / 0.05 | 5.535 / 5.758 | 4.14 / 4.14 |
| 40 | 0.0207 / 0.0088 | 0.39 / 0.18 | 101 / 89 | 0.30 / 0.14 | 1.08 / 1.10 | 0.28 / 0.13 | 3.519 / 4.167 | 4.14 / 4.14 |
| 60 | 0.0310 / 0.0132 | 0.80 / 0.37 | 133 / 106 | 0.59 / 0.29 | 1.08 / 1.08 | 0.55 / 0.27 | 3.162 / 3.533 | 4.13 / 4.13 |
| 80 | 0.0413 / 0.0176 | 1.38 / 0.61 | 145 / 130 | 1.00 / 0.48 | 1.09 / 1.08 | 0.92 / 0.45 | 3.077 / 3.187 | 4.14 / 4.14 |
| 90 | 0.0465 / 0.0198 | 1.74 / 0.76 | 148 / 138 | 1.26 / 0.60 | 1.12 / 1.09 | 1.12 / 0.55 | 3.146 / 3.155 | 4.13 / 4.13 |
| 100 | 0.0470 / 0.0200 | 2.15 / 0.93 | 151 / 142 | 1.56 / 0.72 | 1.13 / 1.06 | 1.39 / 0.68 | 2.890 / 2.892 | 4.14 / 4.14 |
| 110 | 0.0470 / 0.0200 | 2.54 / 1.09 | 152 / 144 | 1.90 / 0.87 | 1.13 / 1.01 | 1.68 / 0.85 | 2.883 / 2.846 | 4.14 / 4.14 |
| 120 | 0.0470 / 0.0200 | 2.93 / 1.25 | 152 / 145 | 2.20 / 1.00 | 1.12 / 1.03 | 1.96 / 0.97 | 2.657 / 2.601 | 4.14 / 4.14 |

Reading: the bypass spectral norm is 0.8 x sum(lr) of its group in both runs (2.93 / 1.25 = 2.35 = the multiplier
ratio): its growth is set by the optimizer's clock, not by a learned demand. The candidate is slower in the warmup
(the outer network learns its byte statistics at less than half the step) and has caught up by step 90-100;
train bpb on one window is noisy (±0.05) and says nothing about step 908.
