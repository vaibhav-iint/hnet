# hnet-smollm3: shared interface spec (v2)

Read `PLAN.md` (what we run) and this file (how the pieces fit) before writing code. This is the contract between
parallel workstreams. If you must change a contract, keep the old call working and say so in your final report.

**Scope:** 100M–500M models on single GPUs. FineWeb-Edu 10B is the main data; FineWeb 10B serves the modded-nanogpt
speedrun. Byte-level H-Nets are compared against BPE (GPT-2 tokenizer) Transformers at matched FLOPs/byte and matched
bytes per step.

## Environment (all workstreams)

* Python: `/home/user/.venv/bin/python` (torch 2.15 CPU+CUDA build, numpy, einops, pytest, pyyaml, tokenizers, safetensors).
  Add pure-python deps with `VIRTUAL_ENV=/home/user/.venv uv pip install <pkg>` and list them in your report and in
  `requirements.txt` (create/append).
* **Shared dev box:** no GPU, 4 CPU cores, 15 GB RAM, ~25 GB free disk, shared by ~7 agents. Keep tests tiny (seconds).
  Use `OMP_NUM_THREADS=1` and `torch.set_num_threads(1)` in tests. Never allocate > 1 GB. Do not download more than
  ~200 MB in total (HF streaming or a single small shard is fine for a smoke test).
* **Network:** open (HF, GitHub, PyPI). The GPU machines also have HF access.
* **Reference code (read-only, MIT):** `/tmp/claude-0/-home-user/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad/ref/`
  * `hnet/`: goombalab/hnet, official model code;
  * `mamba/`: state-spaces/mamba;
  * `modded-nanogpt/`: speedrun code, data scripts, `evals/hellaswag.py`;
  * `smollm/`: SmolLM3 configs and lighteval `tasks.py`.

  Reading them for semantics and parameter names is expected; write our own code.
* **Paper and blog text:** `/tmp/claude-0/-home-user/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad/web/{paper,blog1,blog2}.txt`.
  Figure screenshots are in `.../scratchpad/shots/`.
* **Target GPU runtime:** CUDA 12.x, torch ≥ 2.5, optional `mamba_ssm`, `causal_conv1d`, `flash_attn`. Every
  fast-kernel path has a pure-PyTorch fallback chosen automatically, and a test that the two agree (skipped without
  CUDA).
* No `git commit`, and don't edit files another workstream owns. Code against the interfaces below; mark anything you
  stubbed with `TODO(interface)`.

## Layout and ownership

```
hnet/config.py, hnet/registry.py       coordinator   configs + name->class registries (read these first)
hnet/model/ssd.py, mixers/mamba2.py    WS-A1         Mamba-2 SSD scan (pure torch + kernel dispatch), Mamba2 mixer
hnet/model/** (everything else)        WS-A2         norm, rotary, attention mixer, mlp, block, isotropic,
                                                     routing/, chunking/, losses/, hnet.py, lm.py, dense.py (llama),
                                                     load_official_checkpoint
hnet/model/dense_modded.py             WS-G          modded-nanogpt-style dense model ("dense" registry: "modded")
hnet/optim/muon.py                     WS-G          Muon (registered "muon")
hnet/optim/** (rest)                   WS-D          AdamW groups, per-stage LR multipliers, schedules
hnet/flops.py, configs/{paper,core,speedrun,ablations}/*.yaml (model sections)   WS-B
hnet/data/**                           WS-C          shard download/convert, byte+token loaders, val sets
hnet/train/**                          WS-D          trainer, metrics, checkpointing
hnet/eval/**                           WS-E          scorers, tasks, perturbations, bpb, bootstrap, reports/plots
infra/prime/**                         WS-F          Prime Intellect launch/sync/monitor/teardown
scripts/speedrun/**, configs/speedrun/*.yaml (train sections)   WS-G
tests/test_<area>_*.py                 owner of the area
```

## Registries (hnet/registry.py)

Kinds: `mixer, router, chunker, dechunker, aux_loss, dense, optimizer, schedule`. Register with
`@register(kind, name)`. Modules must be imported by `hnet/model/__init__.py` / `hnet/optim/__init__.py` so that
registration happens on `import hnet.model`.

Built-in names expected:
* `mixer`: `mamba2`, `attn`;
* `router`: `cosine_lookahead` (paper), `causal_end_of_chunk` (ours, see below), `fixed_pool` (every k), `space_like`
  (SpaceByte delimiter heuristic), `external` (boundaries passed in);
* `chunker`: `select`;
* `dechunker`: `ema`, `repeat`;
* `aux_loss`: `ratio`;
* `dense`: `llama`, `modded`;
* `optimizer`: `adamw`, `muon`;
* `schedule`: `wsd`, `constant`, `cosine`.

## Data (WS-C produces; WS-D/E/G consume)

**Source of truth: modded-nanogpt's pre-tokenized GPT-2 shards** on HF:
* `kjj0/finewebedu10B-gpt2` (main);
* `kjj0/fineweb10B-gpt2` (speedrun);
* `kjj0/fineweb100B-gpt2` (optional).

File format: 256 × int32 header (`magic=20240520`, `version=1`, `ntok`), then `ntok` × uint16 tokens. Each document is
preceded by `<|endoftext|>` (50256). Val split: `*_val_000000.bin`. The speedrun uses its first 10,485,760 tokens.

* **Token stream** (BPE models): read shards directly.
* **Byte stream** (byte models): decode each document's GPT-2 tokens to UTF-8 bytes (lossless) and write byte shards
  with the same document order. Each document is `[BOS_ID=254] + bytes + [EOS_ID=255]`, uint8. A byte shard and the
  token shard it came from hold the same documents. Also write a per-shard document index (byte offsets plus matching
  token offsets) so bytes/token can be measured per document.
* **Windows:** fixed `seq_len` (bytes: 8192 default; tokens: 1792 for the paper setup, 1024/2048 for speedrun
  configs) cut from the concatenated stream. Windows may start or end mid-document.
* **Packed micro-batch** (one rank):
  `{"input_ids": LongTensor[T], "labels": LongTensor[T], "cu_seqlens": IntTensor[n_seg+1] (int32, 0..T),
    "max_seqlen": int, "n_windows": int}`, with T = n_windows × seq_len. Segments are documents: they start at the
  window start or right after a document's EOS (bytes) or before `<|endoftext|>` (tokens). Attention and SSM state
  never cross a segment.
* **Labels:** `labels[t] = input_ids[t+1]` within a segment, else `IGNORE_INDEX` (-100).
* **bpb**, the only cross-model LM metric:
  * byte models: `sum NLL(nats) over targets that are real bytes (0..253) / ln 2 / count(those targets)`;
  * token models: `sum NLL over the document's tokens (excluding <|endoftext|> targets) / ln 2 / UTF-8 bytes of the same
    text`.

  The val sets for both are built from the same val documents.
* The loader is deterministic and resumable from `(step, rank, world_size, seed)`.

## Model API (WS-A2; consumed by B, D, E, G)

```python
from hnet.model import build_model, load_official_checkpoint
model = build_model(cfg)         # HNetConfig -> HNetForCausalLM ; DenseConfig -> registry "dense"[cfg.variant]
model.init_weights()
out = model(input_ids, cu_seqlens=None, max_seqlen=None, labels=None, return_logits=True)
# input_ids LongTensor[T] (packed, needs cu_seqlens) or [B, L] (each row one segment)
# out.logits [T,V] | [B,L,V] | None ; out.loss (mean CE over labels != -100) ; out.ce_sum (fp32 sum) ; out.n_targets
# out.aux_loss (alpha * sum_s aux_s; 0 for dense) ; out.stats: dict[str, tensor] flat keys, e.g.
#   "aux/s0", "F/s0" (boundary frac), "G/s0" (mean prob), "compression/s0" (L^s / L^{s+1}), "positions/s1"
# out.bpred: list per stage of RouterOutput(boundary_prob [T,2], boundary_mask [T] bool, selected_probs [T,1])
model.stage_of_param(name) -> int      # for LR multipliers; embeddings/lm_head -> 0
model.blocks() -> list[nn.Module]      # activation-checkpointing / compile units
```

* Parameter names match goombalab/hnet, so `cartesia-ai/hnet_*` checkpoints `load_state_dict` cleanly.
* GQA `Wqkv` rows are `[q | k | v]`.
* Dense models expose the same `forward` signature and `out` fields.

**Component contracts.** These are what make the code modular:
* **Mixer:** `forward(x, ctx)`. `ctx` is a `SeqCtx` dataclass with `cu_seqlens`, `max_seqlen` and `seq_idx`, shared by
  every layer in a stage. `x` is `[1, T, D]` packed.
* **Router:** `forward(h[T,D], ctx, **extra) -> RouterOutput`, where the extras are e.g. `input_ids` for `space_like`
  and `boundaries` for `external`. The first position of every segment is always a boundary (p = 1).
* **Chunker:** `(h, boundary_mask, ctx) -> (h_inner, ctx_inner)`.
* **Dechunker:** `(h_inner, router_out, ctx, ctx_inner) -> h_outer`.
* **Aux loss:** `(router_out, target_ratio, scope) -> (loss, stats)`.
* **Causal end-of-chunk router (ours):** a boundary *after* position t, decided only from h_≤t. Implement it as
  `p_end_t = sigmoid(MLP(h_t))` (or cosine between h_t and a learned projection of h_t predicting h_{t+1}). The chunk
  vector is taken at the chunk's **last** position. Downsampling is then strictly causal with no lookahead. Dechunking
  must shift by one chunk so position t only sees chunks that ended at or before t. Document the exact rule in the
  module docstring and test causality.

## SSD / Mamba-2 (WS-A1 → WS-A2)

```python
from hnet.model.ssd import ssd_chunk_scan   # semantics of mamba_ssm mamba_chunk_scan_combined
y = ssd_chunk_scan(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False,
                   seq_idx=None, dt_limit=(0.0, float("inf")))
# x [b,l,h,p], dt [b,l,h], A [h], B/C [b,l,g,n], seq_idx [b,l] int32 -> y [b,l,h,p]
from hnet.model.mixers.mamba2 import Mamba2   # registered as mixer "mamba2"
m = Mamba2(d_model, d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, chunk_size=256, layer_idx=None)
y = m(x[1,T,D], ctx)                          # param names as mamba_ssm.modules.mamba2.Mamba2
```

## FLOPs (WS-B → WS-D)

```python
from hnet.flops import model_flops, count_params
r = model_flops(cfg, bytes_per_position=[1.0, 3.0, 9.0] | None, seq_len=8192, bytes_per_token=None, training=True)
# -> {"flops_per_byte", "flops_per_token" (dense), "per_stage": [...], "params", "params_nonembed"}
```

Counting follows paper App. B: forward FLOPs, backward = 2× forward, causal-only attention, Mamba-2 terms, LM head
counted, embedding lookup not counted. The same function handles HNetConfig and DenseConfig.

## Trainer (WS-D)

* Entry point: `python -m hnet.train.train --config <run.yaml> [--set a.b=c ...]`. It is single-GPU by default and also
  supports `torchrun` (DDP).
* Run yaml sections: `model`, `data`, `train` (optimizer, lr, schedule, batch, steps, precision, compile,
  grad_accum), `eval`, `log`. Documented in `configs/README.md`.
* Outputs:
  * `<out>/metrics.jsonl`, flat keys every `log_every` steps;
  * `<out>/eval.jsonl`;
  * `<out>/ckpt/step_XXXXXXX/` (model, optimizer, loader state).
* Required metric keys:
  `step, tokens_eq, bytes_seen, lr, loss, ce, bpb, aux_loss, grad_norm, grad_norm/stage{s}, compression/s{s}, F/s{s},
  G/s{s}, flops_per_byte, mfu, bytes_per_sec, mem_gb, step_time`.

## Eval (WS-E)

```python
scorer.loglikelihood(requests: list[tuple[str, str]]) -> list[tuple[float, bool, int]]  # (sum logprob, greedy, n_bytes)
```

* Scorers:
  * `ByteModelScorer(model)`: scores `BOS + context + continuation` bytes, sums the continuation only;
  * `TokenModelScorer(model, tokenizer)`: our dense models;
  * `HFScorer(name, revision)`: GPT-2, SmolLM2, SmolLM3.
* All scorers share task code and normalization.
