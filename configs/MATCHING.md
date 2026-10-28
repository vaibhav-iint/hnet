# Model matching (WS-B)

All configs in `configs/models/*.yaml` (`model:` section loads with `hnet.config.model_config_from_yaml`; the extra
`matching:` section is metadata for `scripts/flops_report.py`). Numbers come from `hnet/flops.py`; regenerate with

```
python scripts/flops_report.py [--bytes-per-token 4.75] [--bytes-per-position hnet2_small=1,3.2,8.4] [--rematch]
```

once the trainer has measured bytes/token and realized compression (the table below uses target ratios and 4.6
bytes/token).

## Convention used for matching (SPEC)

Training FLOPs = 3 x forward (paper App. B terms): all matmuls 2*in*out (GQA-aware), attention quadratic terms
counted **causally** (average context (L+1)/2, exact sliding-window average for SWA layers), Mamba-2 SSD
`6*d_inner*d_state` + conv + gating, SwiGLU (`d_intermediate` rounded up to 128 as in goombalab/hnet), router q/k
and `residual_proj` of every H-Net stage, **LM head counted, embedding lookup not counted**. H-Net stage s runs at
`1/bytes_per_position[s]` positions per byte with sequence length `seq_len_bytes/bytes_per_position[s]`.

Sequence lengths: core 8192 bytes <-> 1792 tokens; speedrun 4608 bytes <-> 1024 tokens; tiny 2048 bytes <-> 448
tokens. BPE FLOPs/byte = FLOPs/token / 4.6.

## Matching table (training FLOPs/byte, target compression)

"bytes/main pos" = bytes per main-network position (dense: bytes/token; H-Net: product of target ratios = BPIC).
"vs ref" = FLOPs/byte relative to the `matching.reference` BPE model (target: within +-2%).

| config | layout | params (M) | non-emb (M) | train GFLOPs/byte | bytes/main pos | per-stage GFLOPs/byte | vs ref |
|---|---|---:|---:|---:|---:|---|---:|
| bpe_medium | T24 d1024 | 411.4 | 308.3 | 0.5275 | 4.60 | 0.5275 |  |
| bpe_small | T12 d768 | 162.2 | 85.0 | 0.1830 | 4.60 | 0.1830 |  |
| bpe_speedrun_124m | T12 d768 | 162.2 | 85.0 | 0.1737 | 4.60 | 0.1737 |  |
| hnet1_small | m4/T9/m4 | 78.5 | 78.3 | 0.1801 | 6.00 | 0.1069, 0.0733 | -1.6% |
| hnet1_speedrun | m4/T9/m4 | 78.5 | 78.3 | 0.1760 | 6.00 | 0.1069, 0.0691 | +1.3% |
| hnet2_medium | m4/T1m4/T27/m4T1/m4 | 413.2 | 412.8 | 0.5281 | 9.00 | 0.1885, 0.0914, 0.2482 | +0.1% |
| hnet2_small | m4/T1m4/T5/m4T1/m4 | 71.2 | 70.9 | 0.1845 | 9.00 | 0.1069, 0.0517, 0.0260 | +0.8% |
| hnet2_small_end_cosine_shift | m4/T1m4/T5/m4T1/m4 | 71.2 | 70.9 | 0.1845 | 9.00 | 0.1069, 0.0517, 0.0260 | +0.8% |
| hnet2_small_end_linear | m4/T1m4/T5/m4T1/m4 | 70.1 | 69.9 | 0.1803 | 9.00 | 0.1037, 0.0506, 0.0260 | -1.5% |
| hnet2_small_end_mlp | m4/T1m4/T5/m4T1/m4 | 70.3 | 70.0 | 0.1808 | 9.00 | 0.1041, 0.0507, 0.0260 | -1.2% |
| hnet2_small_nolrmult | m4/T1m4/T5/m4T1/m4 | 71.2 | 70.9 | 0.1845 | 9.00 | 0.1069, 0.0517, 0.0260 | +0.8% |
| hnet2_small_nosmooth | m4/T1m4/T5/m4T1/m4 | 71.2 | 70.9 | 0.1845 | 9.00 | 0.1069, 0.0517, 0.0260 | +0.8% |
| hnet2_small_ratios42 | m4/T1m4/T5/m4T1/m4 | 71.2 | 70.9 | 0.1750 | 8.00 | 0.1069, 0.0385, 0.0295 | -4.4% |
| hnet2_small_tfenc | T2/T3/T5/T3/T2 | 69.4 | 69.1 | 0.1829 | 9.00 | 0.1065, 0.0505, 0.0260 | -0.1% |
| hnet2_small_xlpattern | m4/T1m4/T11/m4T1/m4 | 112.7 | 112.5 | 0.1855 | 9.00 | 0.0650, 0.0634, 0.0571 | +1.4% |
| hnet2_speedrun | m4/T1m4/T3/m4T1/m4 | 57.0 | 56.8 | 0.1729 | 9.00 | 0.1069, 0.0510, 0.0150 | -0.5% |
| tiny_bpe_10m | T12 d256 | 36.0 | 10.2 | 0.0320 | 4.60 | 0.0320 |  |
| tiny_bpe_2m | T10 d128 | 15.0 | 2.1 | 0.0119 | 4.60 | 0.0119 |  |
| tiny_hnet1_2m | m4/T8/m4 | 7.4 | 7.4 | 0.0118 | 6.00 | 0.0042, 0.0075 | -1.5% |
| tiny_hnet2_10m | m4/T1m4/T10/m4T1/m4 | 23.2 | 23.1 | 0.0319 | 9.00 | 0.0102, 0.0093, 0.0124 | -0.3% |
| tiny_hnet2_10m_end_cosine_shift | m4/T1m4/T10/m4T1/m4 | 23.2 | 23.1 | 0.0319 | 9.00 | 0.0102, 0.0093, 0.0124 | -0.3% |
| tiny_hnet2_10m_end_linear | m4/T1m4/T10/m4T1/m4 | 23.0 | 23.0 | 0.0314 | 9.00 | 0.0099, 0.0091, 0.0124 | -1.9% |
| tiny_hnet2_10m_end_mlp | m4/T1m4/T10/m4T1/m4 | 23.1 | 23.0 | 0.0315 | 9.00 | 0.0099, 0.0091, 0.0124 | -1.7% |
| tiny_hnet2_10m_ratios42 | m4/T1m4/T10/m4T1/m4 | 23.2 | 23.1 | 0.0311 | 8.00 | 0.0102, 0.0068, 0.0140 | -2.8% |
| tiny_hnet2_2m | m4/T1m4/T7/m4T1/m4 | 7.9 | 7.9 | 0.0119 | 9.00 | 0.0042, 0.0034, 0.0043 | -0.0% |
| tiny_hnet2_2m_end_cosine_shift | m4/T1m4/T7/m4T1/m4 | 7.9 | 7.9 | 0.0119 | 9.00 | 0.0042, 0.0034, 0.0043 | -0.0% |
| tiny_hnet2_2m_end_linear | m4/T1m4/T7/m4T1/m4 | 7.9 | 7.8 | 0.0118 | 9.00 | 0.0041, 0.0034, 0.0043 | -1.5% |
| tiny_hnet2_2m_end_mlp | m4/T1m4/T7/m4T1/m4 | 7.9 | 7.8 | 0.0118 | 9.00 | 0.0041, 0.0034, 0.0043 | -1.3% |
| tiny_hnet2_2m_ratios42 | m4/T1m4/T7/m4T1/m4 | 7.9 | 7.9 | 0.0116 | 8.00 | 0.0042, 0.0025, 0.0048 | -3.0% |

Design (paper Table 1 pattern): BPE = Llama-style, GPT-2 vocab padded to 50304, untied, MHA, RoPE 10000,
`d_intermediate = round_up(8D/3, 128)`. H-Nets: E0/D0 = M4 (no MLP), E1/D1 = T1M4 / M4T1 with SWA window 1023,
main = Transformer at the baseline width with depth chosen to match FLOPs; head_dim 64 (tiny: 32), rotary dim =
head_dim/2, Mamba-2 d_state 128 / headdim 64 (tiny: 32 / 32), `lr_multiplier: auto`, `lr_multiplier_n_gpt: 4.6`.

| config | widths | notes |
|---|---|---|
| hnet2_small (+ ablations) | 512, 512, 768 | paper L pattern (outer = 2/3 main) |
| hnet2_small_xlpattern | 384, 576, 768 | paper XL pattern (1/2, 3/4, 1); alternative, see caveat |
| hnet1_small | 512, 768 | 6-DC |
| hnet2_medium | 704, 704, 1024 | 704 = closest multiple of 64 to 2/3 x 1024 (11 heads) |
| hnet2/1_speedrun | as hnet2/1_small | matched at 4608 bytes |
| tiny_hnet2_2m / tiny_hnet1_2m | 96, 128, 256 / 96, 256 | |
| tiny_hnet2_10m | 160, 224, 384 | |

**Caveat (important):** at 125M scale the paper's L pattern spends most of the budget in the byte-level stage
(stage 0 = 58% of FLOPs), leaving hnet2_small a T5 main network (hnet2_speedrun: T3). The paper at L scale and the
blog at 3B spend ~30% on stage 0. `hnet2_small_xlpattern` (T11 main, 35/34/31% split) is the alternative; pick one
before launching A2.

Ablations of hnet2_small (and router/ratio variants of tiny_hnet2_2m, tiny_hnet2_10m) differ only in DC / mixer fields:
* `_nosmooth`: `dc.dechunker: repeat`; `_nolrmult`: `lr_multiplier: null` (same FLOPs).
* `_tfenc`: E0/D0 = T2, E1/D1 = T3 (each M pair replaced by one T with MLP, paper Fig. 8 rule; needs
  `d_intermediate[0] = 1408`), same main: -0.1% vs bpe_small.
* `_end_linear`, `_end_mlp`, `_end_cosine_shift`: only `dc.router`. Not re-matched. The paper's cosine router
  costs 4d^2 FLOPs per outer position, so the cheaper routers lower FLOPs/byte by up to ~2% (end_linear: -1.5% to
  -1.9% vs ref, end_mlp -1.2% to -1.7%). Router params are taken from the registered implementations
  (meta-device build); end_mlp = d -> d/4 -> 1 MLP, end_linear = Linear(d, 1), end_cosine_shift = two d x d projections.
* `_ratios42`: blog's `target_ratios: [4, 2]` (BPIC 8 instead of 9) with the same layers: **not matched**
  (hnet2_small -4.4%, tiny_hnet2_2m -3.0%, tiny_hnet2_10m -2.8%). Re-matching would need main T6 (-1.2%),
  T8 (+2.7%), T11 (+1.6%) respectively (`--rematch`).

## Calibration: which convention reproduces which published number

Tested in `tests/test_flops.py`.

| source | numbers | reproduced by |
|---|---|---|
| Paper Table 2, eval GFLOPs/byte | Transformer L 0.42 / XL 0.69; H-Net 1-stage 0.43 / 0.72; 2-stage 0.43 / 0.69 | `PAPER_APPB`: App. B literally: forward only, **full (non-causal) attention**, **embedding counted** as a 2Vd matmul, LM head counted, SWA layers at w+1 keys; 4.6 bytes/token, 1792 tokens / 8192 bytes, BPIC from Table 1. 2-stage intermediate stage at **2.93 bytes/position** (stage-1 ratio ~2.4) reproduces both L and XL; a sqrt split reproduces L (0.43) but gives XL 0.709. |
| Paper Table 1 params | 680M / 1.3B / 870M / 1.6B | exact module counts from the official json configs (within 3%) |
| Blog Fig. 8 SmolLM3 | 3.08B; emb+head 0.26B / 0.35; body 2.81B / 4.00; total 4.36 GFLOPs/byte | `BLOG_FIG8`: causal attention at **half** the standard causal cost (average context L/4; e.g. document masking), embedding not counted, LM head counted, 4096 tokens, **4.44 bytes/token**. Standard causal counting cannot give body 4.00 together with head 0.35 for any bytes/token. |
| Blog Fig. 8 H-Net 3.67B | stage0 0.22B / 1.37, stage1 0.32B / 0.52, main 3.13B / 2.55, total 4.44 | same `BLOG_FIG8` plus **residual_proj not counted** (its 0.025 is exactly the stage-0 excess); byte sequence 18432 (= 4096 tokens x 4.5), main at 7.6 bytes/position, stage 1 at **3.8-3.85** bytes/position. |

Our SPEC convention is standard causal attention with everything counted except the embedding lookup. Under SPEC
the paper's own L-scale pairing is not matched: Transformer L 0.358 vs H-Net 2-stage L 0.411 eval GFLOPs/byte
(+15%): the paper's convention charges the BPE Transformer more (embedding as a matmul, full attention over 1792 tokens).
So our H-Nets have fewer main layers than a direct scale-down of the paper's configs would give.

**3.7B stage 1 (`configs/hnet_3p7b_smollm3.yaml`):** the inferred MHA + `d_intermediate` 5504 gives exactly the
blog's 0.32B (0.321B). It gives 0.52 GFLOPs/byte when stage 1 runs at the realized ~3.8-3.85 bytes/position (main
7.6 / ~2.0). At exactly 4.0 it gives 0.50. FLOPs ~ 2 x params here, so GQA-4 with `d_intermediate` ~6400
(same params) fits equally well; params and FLOPs cannot tell the two apart. Global (non-SWA) stage-1 attention
would give 0.51 at 4.0. Note that this yaml's `dc.smoothing: true` key is not a `DCConfig` field, so
`model_config_from_yaml` currently rejects it (tests strip it).
