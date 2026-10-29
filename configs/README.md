# Run configs

A run is one yaml with five sections: `model`, `data`, `train`, `eval`, `log`.

```
python -m hnet.train.train --config configs/runs/A2_hnet2_small.yaml [--set train.lr=3e-4 data.micro_batch=2 ...] [--dry-run]
torchrun --nproc_per_node=4 -m hnet.train.train --config configs/runs/A2_hnet2_small.yaml      # DDP
python -m hnet.train.plot runs/core/A1_bpe_small_const runs/core/A2_hnet2_small_const           # curves.png
```

`--dry-run` builds everything, runs 2 steps without writing checkpoints, and prints parameters per stage, LR
multipliers, FLOPs/byte, step time and peak memory. `--set a.b=c` values are parsed as YAML (`[1,2]`, `null`, `3e-4`).

## Layout

* `configs/models/*.yaml` (WS-B): `model:` sections, FLOPs-matched (see `configs/MATCHING.md`).
* `configs/runs/*.yaml` (WS-D): run files. Files starting with `_` are shared fragments, not runs.
  * `_core.yaml`: the paper recipe on FineWeb-Edu bytes. `_core_bpe.yaml`: the same recipe on GPT-2 tokens.
  * `_const.yaml`: constant-LR variant (1000-step warmup, no decay, permanent checkpoints every 0.5B tok-eq).
* `configs/speedrun/*.yaml` (WS-G): modded-nanogpt recipe runs (Muon).

## Composition (`include:`)

* Top level: `include: file.yaml` or `include: [a.yaml, b.yaml]`. Included files are deep-merged in order and the
  including file's own keys win.
* Per section: `model: {include: ../models/hnet2_small.yaml, dc: {router: end_mlp}}`. If the included file has a
  top-level key equal to the section name (`model:`), that sub-dict is used, otherwise the whole file. Local keys
  override (deep merge).
* Lists are replaced, not merged. A dict whose `name` differs from the one it overrides (a `schedule`) replaces it.
* Paths resolve relative to the including file, then the working directory, then the repository root.
* Precedence: defaults (`hnet/train/config.py: DEFAULTS`) < includes < file < `--set`. The composed config is
  written to `<out_dir>/config.yaml` and into every checkpoint's `meta.json`.

## `model`

`type: hnet` (fields of `hnet.config.HNetConfig`) or `type: dense` (`hnet.config.DenseConfig`, `variant: llama |
modded`). See `hnet/config.py` and `configs/MATCHING.md`. Trainer-relevant fields:

| key | meaning |
|---|---|
| `lr_multiplier` | per-stage LR multipliers λ^s (outer first): a list, `null` (all 1), or `auto` = paper App. C Eq. 11, `λ^s = sqrt(N_GPT · D^S/D^s / ∏_{i<s} N^i)` (`hnet/optim/groups.py`) |
| `lr_multiplier_n_gpt` | N_GPT in Eq. 11. `data` = measured GPT-2 bytes/token of the training data (FineWeb-Edu 4.621, FineWeb 4.454; `hnet/data/bpt_stats.json`). The run files set `data`. |
| `dc.ratio_loss_alpha` | ratio (aux) loss weight α (paper 0.03) |
| `type: custom`, `factory: pkg.mod:fn` | tests only: `fn(section) -> (model, cfg)` (e.g. `hnet.train.toy:build`) |

## `data`

| key | default | meaning |
|---|---|---|
| `source` | `hnet` | `hnet` = WS-C shards (`hnet.data.WindowLoader`); `synthetic` = built-in Markov-word corpus (CPU tests); `text` = local text files (`paths`, `val_paths`, docs split by `doc_sep`) |
| `kind` | auto | `bytes` or `tokens`; default `bytes` if `vocab_size <= 256` |
| `dataset` | `finewebedu10B` | `finewebedu10B`, `fineweb10B`, `fineweb100B` |
| `root` | `$HNET_DATA` or `<repo>/data` | data root |
| `n_train_shards` | all | use only the first n local train shards |
| `paths` / `byte_paths` / `token_paths` | null | explicit train shard lists (override dataset/root) |
| `val_token_path` | null | explicit val token shard for the ValSet |
| `seq_len` | 8192 | window length in positions (bytes or tokens; BPE 1792 ≈ 8192 bytes) |
| `micro_batch` | 1 | windows per micro-step per GPU (memory knob) |
| `batch_bytes` | null | global positions per optimizer step for byte data (`/ seq_len` = windows) |
| `batch_tokens` | null | global tokens per step for token data |
| `batch_windows` | null | alternative: global windows per step |
| | | `grad_accum = windows_per_step / (micro_batch × world_size)`, must be an integer |
| `cross_window_label` | true | the last position of a window predicts the next stream element if it is in the same document (both kinds) |
| `shuffle_buffer` | 0 | WindowLoader block shuffle (0 = stream order) |
| `bytes_per_token` | measured | text bytes per GPT-2 token, for `tokens_eq` of byte models |
| `num_workers`, `prefetch` | 1, 4 | background batch threads and queue depth (batches are pinned on CUDA, copied non-blocking) |

Byte and token runs see the same text per step up to the ~1% gap between 256×1792 tokens and 256×8192 bytes.
`bytes_seen` is exact either way: real bytes for byte data, GPT-2 token byte lengths for token data.

## `train`

| key | default | meaning |
|---|---|---|
| `out_dir` | `runs/default` | metrics.jsonl, eval.jsonl, config.yaml, ckpt/ |
| `seed` | 0 | model init seed. Data order depends only on (seed, step, rank, world size). |
| `optimizer` | `adamw` | registry `optimizer`: `adamw` (fused on CUDA) or `muon` (WS-G; reads `train.muon`, see `hnet/optim/muon.py`) |
| `lr` | 6e-4 | peak base LR. Group LR = `base_lr × λ^stage × schedule(step)` |
| `betas`, `eps`, `weight_decay` | (0.9, 0.95), 1e-8, 0.1 | AdamW |
| `wd_embeddings` | true | weight decay on embedding tables. Biases, norms, 1-D params and params flagged `_no_weight_decay` never get it. |
| `no_wd_1d` | true | no weight decay on any `ndim < 2` parameter |
| `lr_multiplier_override` | null | list overriding the model's λ (debugging) |
| `schedule` | WSD 10% / 20% 1-sqrt | `{name: wsd, warmup, decay, decay_shape: linear\|inverse_sqrt\|cosine\|none, min_lr_ratio}`, `{name: constant, warmup}`, `{name: cosine, warmup, min_lr_ratio}`. Warmup and decay below 1 are fractions of the run, otherwise step counts. |
| `steps` / `train_tokens` / `train_bytes` | — | run length: steps, or a tok-eq / byte budget converted with the nominal per-step amount |
| `stop_at_step` | null | stop early (with checkpoint) and keep the full-length schedule |
| `grad_accum` | `auto` | derived from the batch; an int asserts the value |
| `grad_clip` | 1.0 | global-norm clipping (0 = off) |
| `precision` | `bf16` | bf16 autocast on CUDA with fp32 master weights; `fp32` |
| `compile` | false | `blocks` (each `model.blocks()` unit; dynamic shapes for H-Nets) or `model` |
| `compile_dynamic` | auto | dynamic-shape flag for torch.compile (auto = true for H-Nets) |
| `compile_backend` | inductor | torch.compile backend (`aot_eager` / `eager` to debug graph breaks without codegen) |
| `compile_mode` | null | torch.compile mode (e.g. `max-autotune-no-cudagraphs`) |
| `activation_checkpointing` | false | non-reentrant checkpointing of each `model.blocks()` unit |
| `max_hours` | null | wall-clock guard: checkpoint and exit before exceeding it |
| `ckpt_every` | 1000 | rolling checkpoints every N steps, keep `ckpt_keep` |
| `ckpt_every_tokens` | null | permanent checkpoints whenever tok-eq crosses a multiple of N |
| `ckpt_steps` | [] | permanent checkpoints at these steps |
| `ckpt_tokens` | [] | permanent checkpoints at the first step reaching each of these tok-eq values |
| `ckpt_keep` | 2 | rolling checkpoints kept (>= 1) |
| `save_final` | true | permanent checkpoint at the end |
| `resume` | `auto` | `auto` = latest `<out_dir>/ckpt/step_*`; a checkpoint dir or a run dir (its latest); or null. An explicit path without a checkpoint is an error |
| `resume_strict` | true | refuse to resume when model/data/optimizer/LR/schedule/budget/seed/precision differ from the checkpoint's config |
| `stop_on_nonfinite` | 3 | stop after N consecutive non-finite logged losses, write `diverged.json`, no checkpoint (0 = never) |
| `device` | auto | `cuda` if available |
| `check_signals_every` | 10 | DDP: steps between cross-rank stop checks (SIGTERM/SIGINT/max_hours) |
| `dist_timeout_min` | 60 | process-group timeout (covers rank-0 downstream eval) |
| `muon` | — | WS-G Muon settings (only with `optimizer: muon`) |

Checkpoint `ckpt/step_XXXXXXX/` holds `model.pt`, `optim.pt`, `rank{r}.pt` (RNG, loader state, counters),
`meta.json` and `KEEP` for permanent ones. They are written to `.tmp_*` and renamed into place. Resume is exact:
the same per-step losses as an uninterrupted run, also after SIGTERM/SIGKILL at an arbitrary step (tested on CPU
with AdamW and Muon, `tests/test_train_resume_kill.py`). Command-line flags for these keys (`--lr`, `--seed`,
`--resume`, `--print-config`, ...): `python -m hnet.train.train --help`; workflow: docs/EXPERIMENTS_HOWTO.md.

## `eval`

| key | default | meaning |
|---|---|---|
| `every` | 1000 | val bpb every N steps (0 = off), plus at the end if `at_end` |
| `val` | `standard` | `standard` = `hnet.data.standard_valset` (whole FineWeb-Edu val docs, same text for byte and token models), `speedrun` = whole docs in the speedrun's first 10.5M FineWeb val tokens, `none` |
| `val_bytes` | 1e7 | ValSet size in text bytes |
| `val_byte_seq_len`, `val_token_seq_len` | 8192, 1792 | ValSet chunking. Keep them identical across compared runs. |
| `windows_per_batch` | 8 | val windows per forward |
| `downstream` | null | `{every, suite: paper7, tasks, limit, n_boot}` → `hnet.eval` zero-shot suite, rank 0 (`every: 0` = end only) |
| `max_docs`, `batch_bytes` | 200, 65536 | built-in val (synthetic/text sources) only |

`val/bpb` = Σ NLL over text targets / ln 2 / text bytes (EOS and `<|endoftext|>` targets excluded) → `eval.jsonl`.

## `log`

`every` (10): steps between metric lines. `wandb`: project name, used only if `WANDB_API_KEY` is set and `wandb`
is installed. `run_name`.

`metrics.jsonl` keys: `step, tokens_eq, bytes_seen, lr` (first param group's base LR × schedule), `lr_factor`,
`loss` (ce + aux), `ce`, `bpb` (train: ce/ln2 for bytes incl. EOS targets, Σce/ln2/text bytes for tokens), `aux_loss`,
`grad_norm` (pre-clip), `grad_norm/stage{s}`, all model `out.stats` (`compression/s{s}`, `F/s{s}`, `G/s{s}`,
`aux/s{s}`, `positions/s{s}`), `flops_per_byte` (`hnet.flops.model_flops` at the realized compression),
`mfu` (vs bf16 dense peak by GPU name: H100/H200 989, A100 312, L40S 362, RTX 4090 165, RTX 3090 71, A10 125,
L4 121 TFLOP/s; `HNET_PEAK_FLOPS` overrides; null on CPU), `bytes_per_sec`, `tokens_per_sec`, `mem_gb`, `step_time`.
Loss terms are averaged over every micro-step since the previous log line. Stats and grad norms come from the
log step only, averaged over micro-steps and ranks. Host syncs happen only at log, eval and checkpoint steps.

## The PLAN.md runs

| file | PLAN | model | budget | schedule |
|---|---|---|---|---|
| `A1_bpe_small.yaml` / `_const` | A1 | bpe_small (125M) | 10B tokens | WSD / constant + 0.5B ckpts |
| `A2_hnet2_small.yaml` / `_const` | A2 | hnet2_small (3,3)-DC | 5B tok-eq | WSD / constant + 0.5B ckpts |
| `A3_hnet1_small.yaml` / `_const` | A3 | hnet1_small 6-DC | 5B tok-eq | WSD / constant + 0.5B ckpts |
| `B1_nosmooth.yaml` | B1 | no EMA smoothing | 2.5B | constant |
| `B2_nolrmult.yaml` | B2 | λ = 1 | 2.5B | constant |
| `B3_end_mlp.yaml` (+ `B3b_end_cosine_shift`, `B3c_end_linear`) | B3 | causal end-of-chunk router | 2.5B | constant |
| `B4_tfenc.yaml` | B4 | Transformer enc/dec | 2.5B | constant |
| `C1_bpe_medium.yaml` / `_const` | C1 | bpe_medium (350M) | 5B tokens | WSD / constant |
| `C2_hnet2_medium.yaml` / `_const` | C2 | hnet2_medium | 5B tok-eq | WSD / constant |
| `tiny_bpe_2m.yaml`, `tiny_hnet2_2m.yaml` | GPU sanity | 2M | 50M | WSD |
| `tiny_cpu_smoke.yaml` | CPU smoke | tiny_cpu | 20 steps | WSD |

PLAN §3 measures the data multiplier in the stable phase, so the `_const` runs are the A-runs to launch first.
The WSD runs give the post-decay comparison if budget remains. Ablations use the constant schedule so they can be
compared with `A2_hnet2_small_const` at its 2.5B checkpoint (same warmup, same LR at every step). For C2, swap the
model include for the router B3 picks.

**Batch.** 256 × 8192 bytes per step (bytes: `batch_bytes: 2097152`; BPE: `batch_tokens: 458752` = 256 × 1792 ≈
0.46M tokens ≈ 2.12 MB of text), as in the paper. Set `micro_batch` to fit memory; `grad_accum` follows.

**Learning rate.** The paper (Sec. 3) uses WSD with peak LR 2.5× the GPT-3 value for the same size, i.e. half of
Mamba's 5× (Hägele et al. 2024: WSD wants about half the cosine peak). That gives 6.25e-4 at L (GPT-3 2.5e-4) and
5e-4 at XL (2e-4). The run files use the coordinator's **6e-4 (Small) / 3e-4 (Medium)**, the plain GPT-3 Small/Medium
peaks. These are deliberately more conservative than the paper's 2.5× rule, which would give 1.5e-3 / 7.5e-4,
because H-Net outer stages train at λ^0 ≈ 3× the base LR. Override with `--set train.lr=1.5e-3` to follow the paper
rule literally. Whatever value is used, it must be identical for the BPE and H-Net runs being compared
(PLAN §3), and H-Nets add only λ^s and the ratio loss.
