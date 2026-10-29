# g1-opt-02 checks

## Tests
`tests/test_muon_residual_wd.py` (new: default groups and 3 training steps bit-identical to `residual_wd: null`; with
a value, `residual_proj.weight` alone moves to a sibling Muon group with the same LR, multiplier, momentum and stage
and the given decay, every other parameter keeps its group, exact cover; fixed point of decoupled decay under
Muon on a zero-init square matrix with a constant gradient: spectral norm ~1/wd at two LRs, linear growth without
decay; forward pass independent of the flag; `model.residual_proj_wd` ≡ `train.muon.residual_wd`, conflicting values
and AdamW refused) plus `test_optim_groups`, `test_speedrun_muon`, `test_model_hnet`, `test_invariances`,
`test_flops`, `test_train_config`, `test_train_resume`, `test_search_fairness`, `test_model_official`:
**188 passed, 5 skipped**.

## Fairness
`make_gen.py` spec with controls bpe, hnet_ws + this candidate (`model.residual_proj_wd=0.5`) →
`scripts/search/fairness.py`: **Problems: none**. Params 20.01M (20,010,025, identical to hnet_ws), 0.02037
GFLOPs/byte. The knob lives in the model config next to `lr_multiplier`, so the strict gate's held-equal Muon
section is identical to BPE's (`train.muon.residual_wd` also exists but would fail the strict gate).

## Peak LRs actually applied (trainer startup line, benchmark spec values)
`"peak_lr_by_group": ["muon/s0/wd0: 0.04702", "muon/s0/wd0.5/residual: 0.04702", "muon/s1/wd0: 0.02",
"embed/s0/wd0.1: 0.007053", "head/s0/wd0.1: 0.007053", "other/s0/wd0: 0.007053", "other/s0/wd0.1: 0.007053",
"other/s1/wd0: 0.003"]`. Parent: identical without the `residual` group (residual_proj.weight is in muon/s0/wd0).
Every LR is unchanged; only the decay of one 192x192 matrix changes.

## Smoke / mechanism probe (CPU)
L03 hnet_ws model, the benchmark's full 908-step schedule (10% warmup, momentum warmup 90) and LRs, one 2048-byte
window per step on the local FineWeb-Edu shard (mix_v1 train shards are not local), first 150 steps. Loss decreases
(bpb 7.62 → 2.69), no NaN, compression 4.13-4.14 (target 4.145). Parent = same run without the flag (same data order).

| step | resid lr (parent / wd 0.5) | W_r spectral norm | W_r eff. rank | residual_rms | dechunk_rms | bypass ratio | train bpb (1 window) | compression |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0005 / 0.0005 | 0.00 / 0.00 | 103 / 103 |  |  |  |  |  |
| 10 | 0.0052 / 0.0052 | 0.03 / 0.03 | 87 / 87 | 0.05 / 0.05 | 1.10 / 1.10 | 0.04 / 0.04 | 7.618 / 7.618 | 4.14 / 4.14 |
| 20 | 0.0103 / 0.0103 | 0.11 / 0.11 | 92 / 92 | 0.09 / 0.09 | 1.10 / 1.10 | 0.09 / 0.08 | 5.535 / 5.535 | 4.14 / 4.14 |
| 40 | 0.0207 / 0.0207 | 0.39 / 0.36 | 101 / 102 | 0.30 / 0.27 | 1.08 / 1.07 | 0.28 / 0.26 | 3.519 / 3.529 | 4.14 / 4.14 |
| 60 | 0.0310 / 0.0310 | 0.80 / 0.62 | 133 / 136 | 0.59 / 0.49 | 1.08 / 1.09 | 0.55 / 0.45 | 3.162 / 3.223 | 4.13 / 4.13 |
| 80 | 0.0413 / 0.0413 | 1.38 / 0.93 | 145 / 149 | 1.00 / 0.72 | 1.09 / 1.09 | 0.92 / 0.66 | 3.077 / 3.124 | 4.14 / 4.14 |
| 100 | 0.0470 / 0.0470 | 2.15 / 1.25 | 151 / 158 | 1.56 / 0.99 | 1.13 / 1.11 | 1.39 / 0.89 | 2.890 / 2.918 | 4.14 / 4.14 |
| 120 | 0.0470 / 0.0470 | 2.93 / 1.45 | 152 / 158 | 2.20 / 1.19 | 1.12 / 1.12 | 1.96 / 1.07 | 2.657 / 2.688 | 4.14 / 4.14 |
| 130 | 0.0470 / 0.0470 | 3.31 / 1.50 | 152 / 157 | 2.53 / 1.26 | 1.10 / 1.12 | 2.30 / 1.13 | 2.656 / 2.642 | 4.14 / 4.14 |
| 140 | 0.0470 / 0.0470 | 3.61 / 1.50 | 152 / 157 | 2.79 / 1.30 | 1.08 / 1.10 | 2.58 / 1.17 | 2.695 / 2.692 | 4.13 / 4.13 |
| 150 | 0.0470 / 0.0470 | 3.90 / 1.49 | 151 / 155 | 3.10 / 1.35 | 1.10 / 1.09 | 2.83 / 1.23 | 2.834 / 2.860 | 4.14 / 4.14 |

Three-way, including the wd 1.0 probe (stopped by a session time limit at step 149; logged metrics to step 140):

| step | bypass ratio parent / wd 0.5 / wd 1.0 | train bpb parent / wd 0.5 / wd 1.0 | main-network grad norm parent / wd 0.5 / wd 1.0 | router margin parent / wd 0.5 / wd 1.0 |
|---|---|---|---|---|
| 20 | 0.09 / 0.08 / 0.08 | 5.535 / 5.535 / 5.536 | 0.124 / 0.128 / 0.124 | 0.181 / 0.177 / 0.179 |
| 40 | 0.28 / 0.26 / 0.23 | 3.519 / 3.529 / 3.562 | 0.329 / 0.531 / 0.418 | 0.178 / 0.177 / 0.192 |
| 60 | 0.55 / 0.45 / 0.37 | 3.162 / 3.223 / 3.233 | 0.210 / 0.211 / 0.273 | 0.213 / 0.212 / 0.234 |
| 80 | 0.92 / 0.66 / 0.51 | 3.077 / 3.124 / 3.146 | 0.152 / 0.146 / 0.208 | 0.220 / 0.216 / 0.228 |
| 100 | 1.39 / 0.89 / 0.63 | 2.890 / 2.918 / 2.957 | 0.149 / 0.223 / 0.232 | 0.261 / 0.260 / 0.230 |
| 120 | 1.96 / 1.07 / 0.68 | 2.657 / 2.688 / 2.708 | 0.209 / 0.332 / 0.324 | 0.306 / 0.269 / 0.240 |
| 140 | 2.58 / 1.17 / 0.71 | 2.695 / 2.692 / 2.745 | 0.284 / 0.245 / 0.362 | 0.317 / 0.273 / 0.220 |

Reading:
- Parent: the bypass spectral norm is 0.8 x sum(lr) at every step, effective rank ~150/192, linear to step 150 and
  not slowing; residual rms crosses the hierarchy's at step ~85 and is 2.8x it at step 150 (the GPU run reached 3.8 at
  step 300 and 5.1 at 908). The encoder output gain (which also scales the bypass) moved only 1.00 → 1.15 and the
  hierarchy's final gain 1.00 → 1.04: the ratio is almost entirely residual_proj.
- wd 0.5: the opening is unchanged for ~40 steps, then the spectral norm plateaus at 1.50 from step ~125 (the
  fixed point ~1/wd = 2, reached at ~0.75 of it because NS singular values are < 1 and the updates are not fully
  aligned); bypass ratio 1.2 and nearly flat. Train bpb is ~0.03 behind during steps 60-120 and level from 130.
- wd 1.0: ratio capped at 0.7, main-network gradient norm consistently 1.3-2x the parent's (the 1/rms attenuation
  removed), but ~0.05 bpb behind through step 140, so 0.5 was chosen.
- Caveat, read honestly: the router margin is *lower* with the anchor (0.27 vs 0.31 at wd 0.5, 0.22 at wd 1.0 at
  step 140, during the whitespace warm start). One window per step on CPU is noisy, but if the GPU run shows a less
  decided router (p_margin at step 300 below the parent's 0.26), the attenuated-STE half of the hypothesis is wrong
  even if the bpb improves.
