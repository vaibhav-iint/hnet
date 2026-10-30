# Work log

Chronological record of what was done, why, and what came out of it. Newest entries at the bottom.
Times are UTC. Experiments (anything with a result) are written up in [EXPERIMENTS.md](EXPERIMENTS.md).

---

### 2026-10-28 21:26 · Session start, environment check
- Dev box: 4 CPU cores, 15 GB RAM, no GPU. Python 3.13.
- The network policy initially blocked Prime Intellect, Hugging Face, arXiv and cartesia.ai. The user widened it at
  ~21:37.

### 21:28 · Reference code pulled
- Cloned goombalab/hnet (official model code, no trainer), huggingface/smollm (SmolLM3 configs and eval suite) and
  state-spaces/mamba into the scratchpad as read-only references.
- Confirmed from the official code: BOS = 0xFE and EOS = 0xFF; the router is identity-initialized; the residual
  projection is zero-initialized; the ratio loss is computed per device.

### 21:33 · Prime CLI checked
- Prime sandboxes support GPUs and run commands over HTTPS, so no SSH is needed.
- The first API key the user gave was rejected (`Invalid token`).

### 21:35 · Draft-1 plan, then pause
- Wrote a shared config module (`hnet/config.py`) and a first interface spec.
- The user asked for the full plan before any implementation, so implementation paused.

### 21:39 · References read, including rendered charts
- Downloaded both Cartesia posts and the arXiv HTML. Rendered every figure with headless Chromium and viewed them.
- Extracted the data behind blog I Fig. 7 from the page's JS bundle into `configs/reference/blog1_fig7_*.csv`.
- **Finding:** the blog's six tasks are HellaSwag, ARC, PIQA, WinoGrande, CSQA and MMLU.
- **Finding:** the ">10x" compares a cooled-down H-Net to stable-phase SmolLM3. Stable vs stable gives ~1.7x at 94B
  tokens and ~3x at 189B (EXP-000).
- **Finding:** paper App. C LR modulation is λ^s = sqrt(N_GPT · D^S/D^s / ∏_{i<s} N^i).
- Wrote `PLAN_full_scale.md` (draft 2).

### 21:50 · Second Prime key works; budget reality
- The team wallet holds $1,842.94. The only listed GPUs are single H200 pods at $5.40/h.
- The full-scale plan (~$16k+ for the 3.7B run) does not fit.

### 22:00 · Re-scoped to $500 on the user's request
- `PLAN.md` (draft 3): 125M/350M scale on FineWeb-Edu plus a modded-nanogpt speedrun, ~$455 including margin.
- Answered the hardware question: any Ampere-or-newer GPU works; 24 GB is enough.

### 22:04 · Implementation started (user: "go ahead with FineWeb-Edu and the modded-nanogpt datasets")
- Data source: modded-nanogpt's GPT-2-tokenized shards (`kjj0/finewebedu10B-gpt2`, `kjj0/fineweb10B-gpt2`), decoded to
  bytes per document. Byte and token models therefore see exactly the same documents.
- Wrote shared infrastructure: `hnet/registry.py` (name → component registries), a modular `DCConfig` (router,
  chunker, dechunker and aux loss chosen by name per stage), `SeqCtx`, and SPEC v2.
- Launched 7 parallel workstreams (Opus 5.5 subagents):
  - A1: Mamba-2 SSD scan and mixer
  - A2: modular H-Net model
  - B: FLOPs calculator and matched configs
  - C: data pipeline
  - D: trainer and optimizers
  - E: eval harness
  - F: Prime infrastructure
  - G: speedrun port and Muon
- Per the user, every brief stresses optimization and testing: parity tests, causality tests and benchmarks.
  Quick 1–10M-parameter loss runs come next, once model and trainer integrate.

### 22:09 · Logbooks started (user request)
- This file, plus `EXPERIMENTS.md` with figures in `logbook/figures/`. Both are mirrored to a private page.

### 22:14 · Git repo and logbook page
- The repo now lives at https://github.com/vaibhav-iint/hnet (branch `main`).
- First commit `7397824` contains the plan, spec, shared config/registry and logbooks. Agent work is committed after
  integration and tests, not mid-flight.
- Logbook page (private): https://claude.ai/artifact/Q9DtCHGtkktYVtPNbx7YUB. Built by `logbook/build_page.py` from
  the two Markdown files and republished at milestones.

### 22:16 · Commit policy
- The container is ephemeral, so work-in-progress is snapshotted to `main` whenever I stop. These are commits titled
  "WIP snapshot" and may contain partial, untested agent code.
- Tested integration points get a normal commit message and a `milestone-*` branch on the remote (tag pushes are blocked by this session's git proxy; tags exist locally only).

### 22:30 · Reverse-engineering the blog's router (user: "the blog architecture differs; we'll have to guess")
- Downloaded the blog's chunk-viewer data bundles and parsed them as data. Running the page JS was refused by the
  sandbox policy, so `scripts/reference/parse_chunk_viewer.py` reads the JS literals in Python.
- First pass got the encoding wrong (boundary positions are delta-coded; H-Net positions mark chunk-final bytes).
  Fixed after reading the viewer's decoding function as text.
- EXP-001: the improved H-Net uses causal end-of-chunk boundaries, and its stage-1 chunks are clean words with
  trailing spaces.
- Told WS-A2 to support start/end boundary semantics plus three end-router guesses, and WS-B to add matching
  ablation and tiny configs.
- PLAN.md §2b: 1–10M architecture search over boundary semantics, router function, ratios, smoothing, α and
  optimizer, ~$20.

### 22:45 · WS-B (FLOPs + matched configs) finished
- `hnet/flops.py` with three counting conventions. Paper Table 2 is reproduced only by App. B taken literally
  (forward only, non-causal attention, embedding counted). The blog's Fig. 8 needs causal-halved attention and no
  embedding.
- Our matching rule (causal attention, LM head counted, embedding not) is applied to all 28 configs. 80 tests pass,
  including exact param counts against the real model.
- The 3.7B stage-1 guess (FFN 5504, MHA) reproduces the blog's 0.32B params and 0.52 GFLOPs/byte; GQA + FFN ≈ 6400
  is indistinguishable.
- **Decision:** at 125M, the paper's L width pattern leaves a 5-layer main net (58% of FLOPs at byte level). Switched
  the small and speedrun H-Nets to the XL width pattern (outer = ½ and ¾ of main → 11-layer main); the L pattern stays
  as an ablation. (4,2) variants get re-matched. WS-B was resumed to regenerate the configs.
- Note for the tiny runs: tiny BPE models are dominated by the 50K-vocab LM head, so FLOPs-matched tiny H-Nets have
  3–4x the parameters.
- Fixed `configs/hnet_3p7b_smollm3.yaml` (it had an invalid `dc.smoothing` key).

### 22:50 · WS-F (Prime infrastructure) finished; GPU access blocker found
- `python -m infra.prime` provides offers / launch (budget guard: $500 total, $50 per launch) / sync / setup / run
  (detached) / tail / pull / cost / teardown. 14 unit tests pass.
- Live end-to-end test on one CPU sandbox: 2.6 minutes, billed $0.0013, torn down and verified gone.
- **Blocker:** H200 pods ($5.55/h effective) are controlled only via SSH, and this container can only make HTTPS
  connections. GPU *sandboxes* (HTTPS, which our tooling supports) need an explicit grant from Prime. No cheaper GPU
  types are listed right now.
- The team wallet is shared with three other running team sandboxes, so spend is tracked in our own ledger plus our
  resource ids.
- Options taken to the user:
  - (a) ask Prime to enable GPU sandboxes;
  - (b) attach this session to a folder on the user's computer (Remote Control) so pod SSH runs there;
  - (c) last resort: pods with a self-running custom template.
- CPU sandboxes (≤16 vCPU, ~$1.1/h) work now and can host the 1–10M architecture search.

### 22:55 · WS-C (data pipeline) finished
- Byte and token shards are aligned per document (verified byte-identical decoding on all sampled documents); 31
  tests pass.
- modded-nanogpt shards split documents at shard edges; the converter carries the fragments across.
- Measured GPT-2 bytes/token: **FineWeb-Edu 4.621**, FineWeb 4.454. The paper's 1792 tokens ↔ 8192 bytes holds.
- Speedrun targets in bpb: **1.0667** (3.28 nats/token) and **0.9496** (2.92). These are near-equivalents only:
  modded-nanogpt also counts `<|endoftext|>` predictions.
- Val-set protocol: whole documents, split at identical text positions for byte and token models; every byte scored
  once.
- Decided that both model types use the loader's cross-window label rule; passed to WS-D.
- Relayed a bf16-autocast backward bug in `ssd.py`, found by WS-A2, to WS-A1.

### 23:05 · WS-A1 (Mamba-2) finished; WS-B follow-up finished; GPU decision
- **WS-A1:**
  - The pure-torch SSD scan matches a sequential reference to 1e-9 in float64 over 40 random configs, plus gradcheck.
  - Packed segments give the same results as separate segments, and the mixer is causal.
  - Parameter names match mamba_ssm, so official checkpoints load.
  - The custom backward avoids storing per-chunk [Q,Q] matrices: 1.5x faster and 30% less memory than the naive
    einsum version.
  - Fixed the bf16-autocast backward bug WS-A2 found.
  - CPU: chunk_size 64 is 1.4x faster than 256. The 20 CUDA kernel-parity tests are written but await a GPU.
- **WS-B follow-up:** `hnet2_small` now uses the XL width pattern (384/576/768, 11-layer main) at +1.4% FLOPs/byte
  vs `bpe_small`. All ablations are rebuilt on it, and every config including `*_ratios42` is within ±2%.
- Coordinator check: tests for SSD, Mamba-2, FLOPs, data and infra give **183 passed, 20 skipped (CUDA)**.
- **GPU access:** the user chose to ask Prime to enable GPU sandboxes. Until then, all work is CPU-only (local plus
  CPU sandboxes).

### 23:12 · GPU sandbox probe (user: "I've spawned sandboxes via the API before")
- Tried to create one 10-minute GPU sandbox. Nothing was created and nothing was charged:
  - team context, RTX_PRO_6000 → HTTP 403 "GPU VM sandboxes require an explicit account grant and are not included in
    the CPU VM public beta";
  - team context, H200_141GB → the same 403;
  - personal context → refused by our budget guard (personal wallet $0.00); the personal account has no sandboxes.
- The team's three running sandboxes are CPU-only (4 CPU / 16 GB, `terminal-bench-science`). CPU sandboxes work via
  the API; GPU sandboxes need Prime to enable them for this team.

### 23:20 · WS-G (speedrun + Muon) finished
- Ported modded-nanogpt **record #18** (2025-01-04, 3.4 min on 8×H100, 3.28 at 729M tokens). It is the last plain
  PyTorch + FlexAttention record; later records are mostly systems work or token-only tricks. The current record is
  #92, 0.665 min, which requires exactly 8 Hopper GPUs.
- `dense_modded` keeps every trick of record #18 (QK-norm, ReLU², value embeddings, U-net skips, softcap 15, window
  warmup). The loss is computed in chunks so full logits are never built.
- Muon (`build_muon_optimizer`): hidden 2-D matrices (including Mamba in/out projections) use Muon; everything else
  uses Adam; router q/k use Adam by default.
- Target bpb: 1.0667 (3.28) and 0.9496 (2.92), consistent with WS-C.
- Configs: S1 (`s1_modded.yaml`) and S2 H-Nets with AdamW and Muon; one-GPU reference script for S0.
- Coordinator check: speedrun tests pass (see commit). Open issues:
  - the trainer must call `set_progress` (relayed to WS-D);
  - FLOPs for the modded model are approximated by the Llama count (about 4% off);
  - S2 learning rates are unswept;
  - Muon on fused matrices could be split.

### 23:35 · WS-A2 (modular H-Net model) and WS-D (trainer) finished
- **WS-A2:** 51 model tests pass.
  - They cover causality for every router under both semantics, packed vs separate documents, dense/H-Net parity,
    attention backends vs a naive loop, the scan vs a loop, and STE gradient flow.
  - **Official-checkpoint check:** Cartesia's `hnet_2stage_L` loads into our code (328/328 keys) and scores 0.37–0.66
    bpb on English paragraphs.
  - Greedy "The capital of France is" → " Paris. It is the capital of the French region of".
  - Realized compression 3.09 / 2.39. The checkpoint was deleted afterwards.
  - End-of-chunk routers share the select/plug-back code with start semantics; each router declares its semantics.
- **WS-D:** 25 trainer tests pass, covering real H-Net and dense e2e training, exact resume and 2-process DDP.
  - All workstreams are integrated: data, FLOPs/MFU, eval, Muon with `set_progress`.
  - Run files are in `configs/runs/`.
- **Decision:** peak LR follows the paper's rule (GPT-3 LR × 2.5): 1.5e-3 for the 125M runs and 7.5e-4 for 350M, the
  same for BPE and H-Net. To be sanity-checked in the tiny sweep.
- CPU smoke run through the real trainer: loss 4.8 → 1.7 in 20 steps; per-stage LR multipliers 3.03 / 1.43 / 0.88;
  compression logged.
- Downloaded and converted FineWeb-Edu train shard 1 (100M tokens → 462 MB of bytes, exact decoding).

### 23:40 · GCP (user: "try using gcp instead")
- The user uploaded a service-account key for project `neat-dispatch-501718-a5`. It is kept in an isolated gcloud
  config in the scratchpad; the environment's preset `CLOUDSDK_AUTH_ACCESS_TOKEN` must be unset to use it.
- Enabled: Compute Engine, Vertex AI, Storage, Logging.
- Regional quotas (on-demand / spot): A100-40GB 16/64, L4 16/16, T4 8/8, V100 8/16; no H100.
- The service account **cannot** create or list GCS buckets (403). Resource Manager API is disabled, so no IAM
  self-check.
- Probe: created one spot L4 VM (`hnet-gpu-probe`, us-central1-a, Deep Learning VM PyTorch 2.9/CUDA 12.9) with a
  15-minute max run duration and auto-delete, no service account attached. Creation succeeded.

### 23:55 · All 7 workstreams done; milestone-1
- **WS-E (eval):** 60 tests pass.
  - Per-choice logprobs match an independent HF lm-eval re-implementation within 6e-5.
  - GPT-2 HellaSwag acc_norm 31.2 [26.9, 35.8] on 400 items vs lm-eval's 31.1 on full val.
  - Two prompt suites: `paper7` (lm-eval yamls) and `smollm3_cf` (lighteval at a pinned commit, quirks included).
  - Robustness perturbations follow BLT/ByT5 definitions.
  - The multiplier estimator recovers 1.75x / 2.95x on Cartesia's SmolLM3 curve.
- **Full suite: 342 passed, 26 skipped (CUDA-only), 2 slow deselected.** Tagged `milestone-1`.
- CPU is too slow for the architecture search: a tiny H-Net takes ~35 s per 131 KB step on the shared 4 cores. GPU
  runs are needed.
- **GCP probe result:** the spot L4 VM booted with driver 580, torch 2.9.1+cu129, CUDA OK and HF reachable. Output was
  read via the serial console, and the VM was deleted after ~4 minutes; no VMs of ours remain.
- The service account can create GPU VMs but cannot write Cloud Storage. Results would come back via serial console
  and guest attributes unless the user grants bucket access.

### 00:05 · GCP runner workstream (WS-H) launched
- The repo is public, so VMs `git clone` it at a pinned commit. Results come back by streaming metrics lines to the
  serial console, which we read via the API and rebuild locally as `metrics.jsonl`.
- Spot VMs stop (not delete) on preemption so they can resume; every VM has a hard `--max-run-duration`.
- Separate GCP ledger: $300 total, $25 per launch. The live test is capped at 2 spot VMs, ≤ 30 min each, < $3.
- Rough cost per 1e18 training FLOPs at ~35% MFU:
  - A100 spot ≈ $3.7 (~$1.45/h, estimate);
  - L4 spot ≈ $2–2.5 (~$0.3–0.35/h, estimate);
  - Prime H200 ≈ $5.

  The $455 plan would cost roughly $260 on A100 spot.

### 00:30 · Container restart, GCS access granted
- The container restarted and stopped WS-H mid live test. Its spot VM `hnet-l4live` was left stopped (TERMINATED; disk
  only). WS-H was resumed with the same limits, plus instructions to handle "local state RUNNING, VM stopped".
- The user created `gs://neat-dispatch-501718-a5-hnet` and granted the service account Storage Object Admin on it,
  plus Service Account User on itself.
- Verified from here: write/list/read/delete of a test object, and `iam.serviceAccounts.actAs` via
  testIamPermissions.
- WS-H is extending infra/gcp:
  - VMs run as the service account;
  - run dirs and checkpoints rsync to the bucket;
  - converted data shards are cached in the bucket;
  - metrics are pulled from GCS, with the serial console as fallback.

### 00:40 · EXP-002 setup; manual audits (user: "go through examples, double-check loss and normalization")
- Built `hnet/analysis/chunks.py`: per-stage chunk statistics for trained H-Nets, comparable with the EXP-001 blog
  numbers.
- Built `scripts/search/sweep.py`: a resumable multi-run sweep with per-run diagnostics and GCS mirroring.
- Wrote the EXP-002 spec: 9 variants × 2 seeds at ~2M params.
- Turned on in-training evals: val CE/bpb every 250 steps; paper7 task metrics (acc, acc_norm, soft bpb) every 500
  steps.
- **Loss/normalization audit** (`scripts/audit/audit_loss.py`, real FineWeb-Edu windows): all checks pass.
  - BOS/EOS placement, segment starts and next-byte labels (ignored across documents) are correct.
  - Token windows decode to exactly the same text as byte windows.
  - `ce_sum` equals a manual F.cross_entropy, and the loss is ce_sum / n_targets.
  - The ratio loss matches the paper formula per stage, and aux = α·Σ.
  - Uniform-model val bpb equals theory exactly: 8.000 for bytes, 3.289 for tokens.
  - The trainer's token byte count is within 0.024% of exact. It counts input-token bytes; val uses exact counts.
- **Eval audit** (`scripts/audit/audit_eval.py`; examples in `logbook/audits/eval_examples.md`): 0 mismatches.
  - Every choice's log-prob was recomputed by an unpacked forward pass, for the byte model and HF GPT-2 on
    paper7 + smollm3_cf.
  - acc, acc_norm_char, acc_norm_byte and soft bpb were re-derived independently.
  - The one initial "mismatch" was an exact tie between two choices of an untrained model.
- Prompts look right (lm-eval / lighteval formats). smollm3_cf Winogrande keeps lighteval's lower-casing and
  double space; this is identical for every model.

### 00:55 · Apples-to-apples gate (user: "we need fair comparisons")
- `scripts/search/fairness.py` composes every run's exact config and checks that these are held equal:
  - data, shards and window order;
  - bytes/step, total bytes and FLOPs/byte (±2%);
  - optimizer, LR, schedule, precision;
  - val set, eval cadence, downstream suite.
- It writes `fairness.md`. `sweep.py` refuses to start if the check fails.
- EXP-002 passes: 231M bytes per run; bytes/step within 1.1%; FLOPs/byte within 1.9% of BPE (0.01189 GFLOPs/byte).

### 01:00 · Launch failures found and fixed (two bugs)
1. `.gitignore` had unanchored `data/` and `runs/`, which hid `hnet/data`, `scripts/data` and `configs/runs` from
   GitHub (found by WS-H). Anchored them to `/data/` and `/runs/` and committed the code.
2. `python scripts/search/sweep.py` cannot import `hnet` without PYTHONPATH. I missed this locally because I used
   PYTHONPATH=.
   - Reproduced exactly with a Python 3.10 + torch 2.9.1 env that mirrors the VM image, using a fresh clone.
   - Scripts now add the repo root themselves.
   - A 3-step end-to-end mini-sweep then passed in that env: train, val, task evals, diagnostics.
- The first two EXP-002 VMs died within minutes and cost cents. Their console logs were lost on stop, so the new
  startup script uploads its full log to GCS.

### 01:00 · Fail-loudly policy (user: "do not swallow exceptions")
- Removed silent fallbacks:
  - optional kernels now fall back only if the package is absent; a broken install raises;
  - the trainer's val set, downstream eval, FLOPs accounting and wandb now raise on failure;
  - registry imports, data helpers, flops router counting, shard-edge `first_token` and the report reader no longer
    swallow errors;
  - the sweep raises on diagnostic and GCS sync errors, and stops starting new runs after the first failure.
- Fused mamba_ssm/causal_conv1d kernels are wrapped in `torch.compiler.disable`. This fixes the dynamo crash WS-H
  hit on the GPU with `compile: blocks`.
- `infra/sweeps/sweep_vm_startup.sh` runs with `set -euo pipefail`, uploads its full log to GCS on exit, and shuts
  the VM down.
- Full suite: 376 passed, 26 skipped (CUDA).
- WS-H was asked to apply the same rule to `infra/gcp`.

### 01:02 · EXP-002 relaunched (commit f988ba8)
- Spot L4 `hx-exp002-l4` in us-east1-c, 5-hour cap, 3 runs in parallel on the GPU.
- Progress figures: `logbook/exp002_progress.py`, which pulls metrics from GCS.

### EXP-002 split across three spot L4s (commits be2fa47, 950576d)
- The first VM had a sweep stopped by fail-fast (the paper_s0 OOM). It was only finishing the two BPE seeds, so the
  sweep is now split three ways via the new `hnet-only` VM metadata:
  - `hx-exp002a-l4`: bpe (2 seeds), resumes from the step-500 checkpoints;
  - `hx-exp002b-l4`: the boundary routers paper, end_linear, end_mlp and end_cosine_shift;
  - `hx-exp002c-l4`: the ablations paper_r42, paper_nosmooth, paper_nolrmult and paper_alpha01.
- `hnet-only` also scopes the GCS restore and the live mirror to that VM's runs (Python regex exclude). Without it,
  VMs sharing one sweep prefix would overwrite each other's `metrics.jsonl` with stale copies. The failed OOM
  `paper_s0` directory was removed from GCS for the same reason.
- Bug found while planning the BPE resume: on resume, `MetricsLogger` appended to `metrics.jsonl`/`eval.jsonl`,
  so steps between the last checkpoint and the preemption were logged twice. Resume now truncates both files to
  records with step ≤ checkpoint step. New test `test_resume_after_crash_drops_records_past_checkpoint`.

### WS-H done: GCP runner `python -m infra.gcp`
- Commands: `launch`, `status`, `tail`, `sync-metrics`, `watch`, `pull`, `resume`, `stop`, `teardown`, `ls` and
  `cost`.
- Safety: a budget guard ($300 total, $25 per launch) and a ledger that reconciles with the real VM state. Teardown
  is limited to VMs with the `hnet-` prefix that are in the ledger.
- Tested live on spot L4s (us-central1-a after zone fallback):
  - Fused kernels from prebuilt wheels: mamba_ssm 2.3.2.post1, causal_conv1d 1.7.0.
  - Tiny H-Net: 231k bytes/s, 0.57 s/step, 7.7 GB with `compile=false`.
  - Bucket shard cache: 8 s, versus about 65 s to download and convert.
  - Restore and continue across VMs worked from step 177 to 250, through two preemptions.
- Error handling: nothing is swallowed. Errors in any stage or background thread put a traceback on serial, upload
  logs to `runs/<vm>/_logs/`, report `HNET_DONE 97`, and exit non-zero. 38 tests (gcloud mocked).
- **Open issue.** With the fused kernels, `compile: blocks` makes the tiny model about 4.6x *slower* (2.6 s/step).
  The likely cause is recompiles around the `torch.compiler.disable`d kernels. EXP-002 runs with `compile=false`.
  Investigate before the 125M runs.
- About $0.30 spent; all of WS-H's VMs are deleted.

### WS-I2 done: Kimi KDA, MoBA, gated attention (+ Attention Residuals option)
- New mixers in `hnet/model/mixers/kimi.py`:
  - `kda`: chunkwise gated delta rule in pure torch. Matches the recurrent reference to about 1e-12 in fp64.
  - `moba`: top-k block attention, with flex or SDPA backends. Equals `attn` when every block is selected.
  - `gated_attn`: the G1 sigmoid output gate. Equals `attn` when the gate is forced open.
- Each new mixer defines `flops_per_position`.
- Eight tiny configs (BPE and H-Net, including the KDA:attn 5:2 hybrid), all within ±2% FLOPs/byte of
  `tiny_bpe_2m`.
- Tests: 50 Kimi mixer tests; the full suite gave 463 passed.
- Attention Residuals (arXiv 2603.15031; the blog's "1.25×") is implemented as `Isotropic(attn_res=S)` but not yet
  wired into configs or flops.py; I'll do that wiring.
- Open: CUDA parity for fla and flex is not run yet; no dense hybrid layout; KDA has no decoding cache.

### WS-I1 done: DeepSeek MLA, NSA, DSA
- New mixers in `hnet/model/mixers/deepseek.py`:
  - `mla`: low-rank KV latent with a decoupled RoPE key.
  - `nsa`: compressed, selected and sliding-window branches with sigmoid gates.
  - `dsa`: lightning indexer with top-k selection, built on MLA.
- Six configs (tiny BPE and H-Net); FLOPs per byte within ±0.05% of `tiny_bpe_2m`, and the fairness gate reports
  "Problems: none".
- Tests: 59 DeepSeek mixer tests (plus 9 CUDA-only); full suite 535 passed, 37 skipped.
- The DSA indexer KL loss is injected into backward through an identity autograd function, since the trainer only
  adds `out.aux_loss`. Its gradient is not divided by `grad_accum`.
- Caveat: at these context lengths NSA still sees 64–97% of the keys dense attention sees, so it is not yet a test of
  sparsity.
- Open: CUDA parity for the flex and flash fast paths still has to run on a GPU VM before EXP-005.

### WS-J done: trainer flags, resume hardening, sweep spec v2, launcher, report
- Trainer flags (`hnet/train/cli.py`):
  - Covers optimizer and Muon LR, schedule, budget, batch, checkpointing, resume, `--model YAML` and
    `--print-config`.
  - Precedence: config < flags < `--set`; a flag and a `--set` that conflict are an error.
- Resume:
  - Bitwise-identical resume on CPU, including after a SIGTERM or SIGKILL at an arbitrary step (AdamW and Muon),
    tested in a subprocess.
  - `resume_strict` refuses to resume if the config changed; `stop_on_nonfinite` writes `diverged.json`.
  - My metrics truncation-on-resume is kept, made atomic, and tolerates a torn last line.
- Sweep spec v2 (`scripts/search/spec.py`): grids, params, `--shard i/n`. Fairness "tuned" mode for per-arch LR
  sweeps. `infra/sweeps/launch.py create|status|teardown`. `scripts/search/report.py` writes report.md,
  best_params.yaml and figures.
- Specs ready: EXP-003 (AdamW LR sweep), EXP-004 (Muon), EXP-005 (attention variants). EXP-004/005 take their
  LRs from EXP-003's `best_params.yaml`.
- Full suite: 576 passed, 37 skipped.

### EXP-002 VM b: CUDA illegal memory access in mamba_ssm SSD backward
- All three runs on `hx-exp002b-l4` (paper_s0, paper_s1, end_linear_s0) crashed within about 30 steps:
  `ssd_combined.py:971 dxBC_given.copy_` → `cudaErrorIllegalAddress`.
- paper_s0 logged compression 5.3 → 44 bytes/position at step 20 just before the crash, which looks like corrupted
  numerics rather than router behaviour.
- VM c's runs (paper_r42, paper_nosmooth, same micro_batch 16) are healthy at step 170+. WS-H's 177-step run with
  the same kernels used micro_batch 64.
- Repro: `scripts/debug/repro_ssd_crash.sh` runs five cases one at a time with `CUDA_LAUNCH_BLOCKING=1`.

### Root cause of the EXP-002 crashes: flex attention mask builder, not mamba_ssm (commit after 88237a3)
- Repro on debug VM `hnet-ssddbg2`, five cases run one at a time with `CUDA_LAUNCH_BLOCKING=1`:
  `paper_mb16_s1` aborted (rc 134); the other four passed, including `paper_mb16_s0`, which had crashed on VM b.
  The crash is intermittent.
- With synchronous launches the faulting kernel is the inductor-compiled `create_block_mask` (`_compile=True`,
  deprecated) at `attn.py:98`, in the main network's attention. The mamba_ssm backward frame in the VM b logs was
  only where the asynchronous error surfaced.
- Inner-stage lengths change every step and are almost never a multiple of 128. During the transient early router
  collapse (stage 1 reaching up to ~300 bytes/position) they are very short, about one position per document.
- VM c then hit the same crash in paper_nolrmult_s1 and paper_nosmooth_s1, so it isn't specific to one VM or variant.
- Fix: `attn.packed_block_mask`. It builds the mask eagerly and clamps the `mask_mod` indices to T−1, using a tensor
  bound so flex does not recompile when T changes. It is used by attn, NSA, DSA and MoBA.
- Regression test: flex vs SDPA on ragged packed lengths (`tests/test_flex_mask.py`). Full suite: 584 passed.
- Verification on GPU: `hnet-ssddbg3` runs paper s0–s3 and end_linear s0–s1.
- Also found: every healthy H-Net run shows the transient early collapse (it recovers to target by ~10M bytes). To
  look at after EXP-002: ratio-loss warmup or a lower stage-0 LR multiplier during warmup.

### Second failure mode: post-run diagnostics raced in Triton's autotuner (commit 9e9f8c4)
- VM c: paper_nolrmult_s0 and paper_nosmooth_s0 finished together. The sweep runs each run's chunk-diagnostic
  forward pass in its worker thread, so two passes ran at once, and Triton's autotuner (fused Mamba kernels) is not
  thread-safe: `TypeError: 'NoneType' object is not a mapping`.
- paper_nosmooth_s0 finished training; only its diagnostics and `done.json` are missing.
- Fix: `diag_lock` in `scripts/search/sweep.py` serializes the diagnostics.
- VMs b, d and e (still in setup) were moved to the fixed commit by updating their metadata and resetting them, which
  keeps their spot slots. VM d was preempted in us-central1-a and restarted.
- New VM f (us-central1-b) runs paper_nosmooth (s0 resumes from its final checkpoint, then diagnostics) and
  paper_nolrmult (s1).
- Finished: bpe ×2, paper_r42 ×2, paper_nolrmult_s0.

### More VM hardening; the eager mask ran out of memory → chunked BlockMask (commit 1bdcc27)
- VM setup now:
  - verifies the kernel entry points and force-reinstalls a wheel left broken by a preemption mid-install;
  - clears the Triton and inductor caches on every boot (a reset left truncated autotune JSON → JSONDecodeError in 4–6
    GPU tests on VMs b and e).
- Preemption babysitter (scratchpad `launch/babysit_exp002.sh`): restarts a VM whose latest stop was a preemption, or
  recreates it in another zone if the start fails for lack of capacity.
- The eager `create_block_mask` (from the illegal-memory-access fix) materializes int64 [H, T, T] intermediates and
  ran out of memory with 4 runs per L4 (end_mlp_s1 at T ≈ 12k, a 1.17 GiB allocation).
  - `packed_block_mask` now evaluates `mask_mod` in row chunks by broadcasting, reduces to partial/full block maps,
    and calls `BlockMask.from_kv_blocks`. Index clamping is kept.
  - Tests: identical block layout and dense mask to torch's `create_block_mask`, and exact under a tiny chunk budget.
- VMs b and e (in setup) were reset onto 1bdcc27. VMs d and f keep running on 9e9f8c4. The other three runs on d
  survived end_mlp_s1's failure; end_mlp_s1 needs a rerun.

### EXP-002 at 10/18; EXP-003 launched (commit df6ad43)
- EXP-002: 10 runs are done. Every finished H-Net ablation ends at val ≈ 1.82–1.83 bpb vs BPE 1.635: H-Net trails
  by ~0.19 bpb at the ~2M scale, and the ablations barely differ.
- The bytes-per-position panels are now log scale.
- Sweep uploads retry transient GCS errors (an HTTP 503 had failed VM f's final sync). Restarting f uploaded the
  missing final checkpoints.
- VM b (paper, end_linear) was preempted five times in us-central1-a and is now **on-demand** in us-central1-c
  (about $1.3 extra). Its runs restarted from scratch: no checkpoint had been reached.
- EXP-003 (AdamW LR sweep: bpe, paper, end_mlp and end_cosine_shift × LR {5e-4, 1e-3, 1.5e-3, 3e-3, 6e-3} × 3
  seeds = 60 runs) was launched with `infra/sweeps/launch.py` on 5 spot L4s (12 jobs each, 3 in parallel). Worst case
  $9.26; fairness gate (tuned) reports no problems.
- Babysitter (`launch/babysit.sh`) covers EXP-002 VM d and the EXP-003 VMs. It restarts a preempted VM in place, or
  recreates it in another zone from the exact `launch.py` create command.

### Attention Residuals wired into configs, models and FLOPs
- `HNetConfig.attn_res` (scalar or per-stage list: block size S, 0 = off) is passed to every Isotropic stack of the
  stage. `DenseConfig.attn_res` does the same for the dense backbone.
- `flops.attn_res_counts`:
  - params: 2d(height+1);
  - forward FLOPs: operator j attends over 1 + ⌈j/S⌉ sources at 4d per source, plus RMSNorm and softmax elementwise
    work.
- Tests (`tests/test_attn_res_wiring.py`): the parameter count equals the real model's for H-Net and dense with
  S ∈ {0, 1, 3}; the FLOPs overhead is under 5%; the counts match a hand calculation; H-Net forward works with AttnRes
  on in the main network only. Full suite: 603 passed.
- It can now be added as an EXP-005 variant, e.g. `--set model.attn_res=[0,0,2]` for the main network only.

### Deep dive: what the tiny H-Net learns, and why it trails BPE
Tools: `hnet/analysis/diagnose.py` (per-byte loss by byte class and chunk position, router confidence, boundary
alignment, zero-main ablations, a token-aligned comparison with BPE) and `hnet/analysis/probe_texts.py` (HTML view of
chunks, loss and main-network help on contrasting inputs: `logbook/figures/probe_texts.html`). Per-checkpoint JSON is
in `logbook/diagnostics/`. Findings on EXP-002 (paper 2-stage (3,3), final checkpoint, 64–128 val docs):
1. **The deepest network is barely used.**
   - Zeroing the innermost main network (6.0M of 7.9M params, 38% of FLOPs) costs 0.04 bpb (paper_s0) or 0.06
     (end_mlp_s1).
   - Zeroing everything below stage 0 costs 0.69, mostly from the stage-1 encoder and decoder.
   - Over training the main network's help grows only 0.008 → 0.018 → 0.040 (steps 500, 1000, final).
2. **The main network's gradient collapses early in every variant.** Its gradient norm drops from 7–23% of stage 0's
   at step 10 to 1–3% by step 20 and stays there. This includes end-of-chunk routers, which never show the
   early compression spike (up to 104 bytes/chunk at step 20 for the paper router). The decoders learn to rely on the
   (zero-init) residual path almost immediately.
3. **The stage-1 paper router never leaves its initialization.**
   - 46–70% of p sit in 0.4–0.6, and only 2% are confident.
   - With identity q/k on near-random states, cos ≈ 0 gives p ≈ 0.5 at init.
   - The ratio loss gives no push either: ∂L/∂G = N/(N−1)·(N·F − 1) = 0 once F is on target.
   - The only remaining signals (EMA and STE paths) scale with the main network's usefulness, so routing and the
     main network are stuck in a self-reinforcing equilibrium.
   - Stage-1 chunk starts after whitespace: 13% (Cartesia's improved H-Net: 92%).
4. **The end-of-chunk MLP router does learn word-like chunks**: 72% of stage-1 starts follow whitespace, and 67%
   of stage-0 decisions are confident. Yet the main network's help only rises from 0.04 to 0.06 bpb, and bpb improves
   by just 0.01. Better chunks alone don't unlock the hierarchy here.
5. **Where the gap to BPE is (token-aligned, identical text).** 84% of it is in "space+word" GPT-2 tokens, spent on
   the letters after the space (1.78 bits/byte) rather than the space itself (0.45 bits): word identity. The main
   network's help goes to newlines (1.3 bits saved), punctuation (0.4) and digits, but only 0.03 bits at word
   starts.
6. **Scale check: neither model copies in context.** Repeated phrases cost ~2.1–2.4 bits/byte on the 4th–6th
   repetition, for BPE as well (no induction heads at ~50M tokens). The tiny regime is dominated by local
   statistics, where BPE's 13M lookup and head parameters are worth more than any hierarchy. Tiny runs are for
   debugging mechanisms, not for judging H-Net vs BPE.
- Hypotheses for 4–5: a decoder bottleneck (word-level knowledge must be spelled out through a d=96, 4-layer byte
  decoder and a 256-way head, while BPE has a 6.4M-param word-level head), plus missing capacity at tiny scale.
- Running:
  - EXP-008 (residual path-drop / main-path aux head / router entropy / combined, 3 seeds);
  - EXP-009 (oracle word chunks vs fixed vs learned at matched FLOPs);
  - EXP-006 (n-gram embeddings);
  - a dense-checkpoint probe run for steps 1–400.

### Linear probes on the main network: it is not a decoder bottleneck (`hnet/analysis/probe_main.py`)
Question: does the innermost main network compute next-chunk information that the byte decoder cannot spell out
(a decoder bottleneck), or does it compute nothing useful (a training-signal problem)? Multinomial logistic probes
on the main network's input x and output z at every innermost position, target = the byte after the chunk's anchor,
70/30 split, held-out bits (logbook/diagnostics/probe_main_*.json):

| model | positions | unigram | probe on main INPUT x | probe on main OUTPUT z | probe on [x, z] | model's own loss on those bytes |
|---|---|---|---|---|---|---|
| paper_s0 | 35.7k | 3.98 | **2.08** | 2.62 | 2.03 | 1.45 |
| end_mlp_s1 | 37.3k | 4.92 | **3.71** | 3.89 | 3.66 | 3.22 |

- The main network's output is *less* linearly predictive of the next byte than its own input (2.62 vs 2.08 bits
  for paper_s0; 3.89 vs 3.71 for end_mlp_s1). A 7-layer, 6M-parameter Transformer destroys information its input
  already carried. Concatenating x and z recovers only ~0.05 bits over x alone.
- So the hierarchy is not computing next-chunk predictions that the decoder then fails to use. It is simply not
  trained: this is consistent with the gradient-norm collapse (main-network grad ≈ 1–3% of stage 0's from step 20)
  and the 0.04-bpb zero-main ablation. The decoder bottleneck hypothesis (item 5 above) is **not** supported for
  the main network; the gap to BPE at word starts is mostly a capacity/signal problem at the chunk level.
- Implication: fixes must deliver gradient to the main network early (bottleneck-first residual warmup, main-path
  auxiliary head; EXP-008/011) rather than widen the decoder.
- Caveat: the end_mlp probe targets differ (end-of-chunk anchor → the first byte of the next chunk, a harder,
  higher-entropy target: unigram 4.92 vs 3.98), so the two rows are not comparable to each other, only within row.

### Finding: the chunk vector is a boundary *state*, not a chunk *summary* (`probe_main.py --identity`)
Probes from the innermost main network's input x (the chunk vector) to the bytes of the chunk it represents, on
27–33k held-out positions (logbook/diagnostics/probe_identity_*.json):

| model (semantics) | target | unigram | linear from x | MLP from x | linear from z (main output) |
|---|---|---|---|---|---|
| paper_s0 (start) | first byte of its chunk | 4.83 | **0.46** | – | 1.61 |
| paper_s0 (start) | last byte of its chunk | 4.41 | 4.11 | – | 4.13 |
| end_mlp_s1 (end) | last byte of its chunk | 2.01 | **0.18** | 0.08 | 0.58 |
| end_mlp_s1 (end) | first byte of its chunk | 4.76 | 3.60 | **3.26** | 3.81 |
| end_mlp_s1 (end) | second byte | 4.21 | 3.32 | 2.99 | 3.48 |
| end_mlp_s1 (end) | chunk length (1–16) | 3.64 | 2.95 | 2.51 | 3.07 |

- The vector handed to the main network is the encoder's residual-stream state at the boundary byte. It encodes
  that byte almost losslessly and the rest of the chunk barely (only ~1.5 of 4.8 bits about the first byte of a
  ~9-byte chunk, even with a nonlinear probe). **The main network does not know which word it is looking at**, so
  it cannot do word-level language modelling, which is where the whole gap to BPE sits (84% of the gap is in the
  letters of "space+word" tokens). The BPE model's input, by contrast, is the token's identity.
- With the paper's "start" semantics the vector encodes the chunk's *first* byte and nothing of its content, which
  can only arrive one chunk later: a built-in one-chunk lag for within-chunk prediction.
- Also: the main network's output z encodes *less* of everything than its input (the 7-layer stack destroys the
  identity information it was given), and its gradient SNR at the training batch size is normal (2.4 at step 500 vs
  1.9 for stage 0; `main_signal.py`), so this is not a noisy-gradient problem: the main network is trained on a signal
  that contains little word identity.
- Fix to test next: a **pooling chunker**. The chunk vector becomes a learned summary over all bytes of the chunk
  (mean or attention pooling of encoder states within the chunk, plus the boundary state), so the main network's
  input carries the chunk's identity. This is BLT's byte→patch cross-attention in H-Net clothing, and it is natural
  only with "end" semantics (the chunk must be complete when its vector is formed).

### First results of the fix sweeps (val bpb, mean ± sd over seeds; BPE = 1.634 ± 0.002, 3 seeds)
| experiment | arm | n | val bpb | main-network contribution (zero-main cost) |
|---|---|---|---|---|
| EXP-006 | 2-stage paper router | 1 | 1.795 | |
| EXP-006 | 2-stage paper router + n-gram embeddings | 3 | **1.714 ± 0.016** | |
| EXP-006 | 1-stage learned router | 3 | 1.845 ± 0.002 | |
| EXP-006 | 1-stage + n-gram embeddings | 1 | **1.705** | |
| EXP-006 | 2-stage end-of-chunk MLP + n-gram | 1 | 1.696 | |
| EXP-009 | 1-stage, fixed 6-byte chunks | 3 | 1.859 ± 0.007 | +0.25 |
| EXP-009 | 1-stage, oracle word-START chunks | 3 | 1.827 ± 0.006 | +0.45 |
| EXP-009 | 1-stage, oracle word-END chunks | 3 | 1.806 ± 0.026 | **+0.56** |
| EXP-008 | 2-stage base (zero-main probe) | 1 | 1.795 | +0.036 (innermost), +0.687 (all below stage 0) |

1. **Parameter matching is real**: FLOP-free n-gram tables (matching BPE's 15M total params) remove ~40% of the gap
   (0.21 → 0.07 for 1-stage). Cartesia's ladder matches the outer networks to BPE's embedding + head in params AND
   FLOPs; this is what that buys at tiny scale.
2. **Chunk semantics change how much the hierarchy is used**: oracle word-end chunks make the main network worth
   0.56 bpb vs 0.25 with fixed chunks; "end" beats "start" by 0.11 of main-network contribution (the lag).
3. **Routing is not the main gap**: oracle word chunks are only 0.04 better than the learned router and still 0.17
   behind BPE. What remains is what the main network receives (chunk identity: EXP-012) and how early it gets
   trained (EXP-008/011).

### Where the n-gram gain lands; LR sweep; mid-training router collapses
- **n-gram embeddings help the byte stage, not the hierarchy** (`diagnose.py` on EXP-006 hnet1 vs hnet1_ng, same
  96 docs): bpb 1.790 → 1.676, but the main network's contribution falls from 0.63 to 0.38 bpb; the gains are in
  punctuation (gap 0.56 → 0.16) and word-internal bytes. The tables are a BPE-like local-statistics crutch.
- **EXP-003 (partial, 34/60 done): BPE's best LR is 6e-3 at the grid edge (1.564 ± 0.012); every H-Net peaks at
  1.5e-3 (1.78–1.80) and degrades above.** At tuned settings the gap is 0.235 bpb, not 0.17. Note the paper's LR
  multipliers put stage 0 at 3.5× base (5.3e-3 at base 1.5e-3) and the main network at 0.72× base (1.1e-3).
- **Mid-training router collapses at LR ≥ 3e-3**: stage-1 compression spikes to 18–631 bytes/position for tens of
  steps (end_mlp lr 6e-3 s0 at step 550; end_cosine_shift lr 3e-3 s1 at step 340), then the ratio loss pulls it back
  at a cost of ~0.05 bpb. The stage-1 router sits at p ≈ 0.5 (no signal keeps it decided), so a large update can
  flip a whole batch to "no boundaries". This is the mechanism behind H-Net's LR ceiling.
- Next: EXP-013 tests whether stabilizing the router (strong ratio loss, entropy penalty) lets the H-Net use BPE's
  LR, and whether the main network is under-trained (inverted multipliers [1, 1, 4]).

### The stage-1 router is a limit cycle at high LR (`logbook/figures/exp003_collapses.png`)
- At lr 6e-3 the stage-1 router oscillates for the whole run between compression 1 (every position selected) and
  ~500 (none), every few steps; at 3e-3 in bursts; at 1.5e-3 it holds near 3. Stage 0 is stable at every LR.
- Training bpb barely reacts, because the decoder relies on the residual path: nothing damps the router.
- Mechanism: the ratio loss constrains only the *mean* of p; the stage-1 p's are clustered near 0.5 (the
  "undecided router" of the diagnostics), so a uniform shift of the router's bias flips every position at once. The
  ratio loss then pushes back and Adam's momentum overshoots.
- The EXP-008 `combo` OOMs (both seeds) are the same failure: the router-entropy penalty drove the stage-1 router to
  select everything, the main network saw ~32k positions instead of ~3.6k, and attention ran out of memory. The
  entropy penalty amplifies the all-flip mode; verdict: harmful.
- Fix to test: **rank-based (quantile) boundary decisions**. In training, select the top 1/N positions by p (batch
  quantile) and keep a running threshold for eval (BatchNorm-style); p is recentred so the threshold maps to 0.5
  for the STE and the EMA dechunker. Compression is pinned exactly, the decision is immune to uniform shifts, and
  the ratio loss becomes an exact constant (inert), so α disappears.

### Realized compute: the paper router runs 10–16% over its FLOPs budget (`logbook/sweep_summary.py`)
The fairness gate checks the *configured* ratios. Recomputing training FLOPs/byte from each run's realized
compression (last 10 logged steps):

| arm | realized vs budget | val bpb |
|---|---|---|
| 2-stage paper router, α 0.03 (EXP-006 `paper`, EXP-008 `base`, EXP-010 `resid_id`) | +10.5% to +15.6% | 1.78–1.80 |
| same with strong ratio loss α 1.0 (EXP-010 `alpha1`) | +0.8% | 1.818 |
| fixed (3,3) pooling, oracle word chunks, n-gram arms | +0.2% to +1.2% | |
| EXP-008 `combo_end` (router-entropy penalty) | **+379%** (compression 1.15 / 1.54) | 1.652: invalid |

- The CE gradient always favours more positions (more compute); at α = 0.03 the ratio loss cannot hold them (its
  maximum penalty is N·α ≈ 0.09 nats). The paper router's apparent 0.03-bpb edge over fixed chunks was extra compute.
  EXP-002's paper-router runs were likewise ~7% over budget (2.8/8.5 bytes per position instead of 3/9).
- `combo_end` was not a result: the entropy penalty drove both routers to "select almost everything". It does show
  what the hierarchy can do when it gets positions and training signal, which is the point of the next sweep.
- Policy from here: H-Net arms use rank (quantile) decisions, which pin compression exactly, or α ≥ 1; every
  summary reports realized FLOPs vs budget and flags > 2%.

### Interim readouts of EXP-008/010/011 (honest-compute view; `logbook/sweep_summary.py`)
| arm | val bpb | realized compute | main-network contribution |
|---|---|---|---|
| paper router, strong ratio loss (α 1) | 1.810 ± 0.012 (2 seeds) | +0.8% | 0.065 |
| paper router, residual path-drop 0.3 | 1.846 | **+31%** | 0.053 |
| paper router, entropy penalty | 1.921 | +6.6% | 0.000 |
| fixed pooling, identity-init residual | 1.862 | +0.2% | 0.000 |
| fixed pooling (zero-init residual) | 1.823 | +0.2% | 0.059 |

- Path-drop makes the decoder lean on the hierarchy, which makes the LM loss's pull on the router *stronger*: the
  model answers by selecting 31% more positions. Any fix that makes the hierarchy useful inflates compression unless
  compression is pinned (quantile decisions, EXP-014/015/016).
- Identity-init residual: the bypass is strong from step 0 and the decoder never uses the hierarchy (contribution
  0.000, worse than zero-init). Zero-init starves the hierarchy early; identity-init starves it for good. The
  hierarchy needs a window where it is necessary but not noise: residual warmup (EXP-011 `fixed_rwarm`, pending).
- Entropy penalty: kills the stage-1 hierarchy outright (contribution 0.000) even where it does not collapse
  compression. Dropped from further use.
- New intervention (EXP-016): chunk-level multi-byte prediction, a head on the innermost main network's output
  predicting the next chunk's first k bytes, bypassing dechunker, residual and decoder.

### EXP-014 post-mortem: the p-space quantile rule saturates and collapses
`metrics.jsonl` of the stopped runs: `tau/s1` climbs to 1.000 within 70–90 steps, then `compression/s1` jumps to
440–600 (only forced boundaries) and stays there (`quant__lr0.0015_s1`, `quant__lr0.006_s0`, ...); in
`quant__lr0.0015_s2` the same happens at stage 0 at step ~290 (`compression/s0` 1300–1800, then stage 1 selects
everything because it sees no free positions). The one healthy-looking run had `tau/s0 = 1.000` too.

Mechanism: the quantile threshold was detached, so nothing anchored the *scale* of the router's logits; the STE
confidence gradient pushes selected logits up and unselected ones down forever (the same force that makes the paper
router over-select, now with no ratio loss to oppose it). Once logits exceed ~17, `sigmoid` is exactly 1.0 in fp32
for more than 1/N of the positions, the quantile sits on the tie, recentring gives p' = 0.5 for all of them, the
strict `>` selects none, and the stage is dead (no selected positions → no gradient → no recovery).

Fix (`routing/decide.py`, tests `test_quantile_decision.py`): rank the router's raw fp32 logit (`RouterOutput.logit`)
with `topk` (ties broken by rank, never "none"); standardize the scores over the batch *in the graph* (BatchNorm
style) before the threshold and the confidence, which makes the gradient's common-shift and common-scale components
exactly zero (unit test), so the logits cannot inflate; midpoint threshold between the k-th and (k+1)-th score;
running mean/std/threshold for eval; `temp` controls the confidence margins; `score_std/s*` is logged as the drift
diagnostic. A general lesson for H-Net routers: any decision rule needs a scale anchor; the paper's is the ratio loss
(weak), ours is normalization.

### Zooming out: the hierarchy needs a vocabulary (EXP-017)
The user asked for coarser-grained ideas and reminded that the point of H-Net is *learned tokenization*. Reading the
findings as one story: (1) 84% of the gap to BPE is word identity; (2) BPE gets it from a FLOP-free 12.9M-parameter
table, the H-Net has to compute it; (3) the main network receives the boundary byte's state, not the chunk's
identity, and gets no word-level training signal; (4) oracle word boundaries barely move bpb, so routing quality is
not the bottleneck at this scale: what the main network is fed and trained on is. EXP-017 treats the router's chunks
as tokens: hashed chunk-content embeddings (parameter-matched to BPE, zero FLOPs) enter the main network, and a
sampled-softmax next-chunk objective trains it. This also makes the segmentation matter to the loss (a chunk's
identity is its token), so the learned tokenizer finally has something to be good at. Router-dynamics work stops
here; the remaining budget (~$120 of the $200 cap) goes to this line.

### Harvest of the finished sweeps (`logbook/sweep_summary.py --all`, honest-compute view)
| arm (3 seeds unless noted) | val bpb | main-network contribution (below stage 0) | realized compute |
|---|---|---|---|
| BPE (EXP-006) | 1.634 ± 0.002 | | |
| paper + n-gram tables (EXP-006 `paper_ng`) | 1.714 ± 0.016 | | +1.2% |
| 1-stage + n-gram tables (`hnet1_ng`, 2 seeds) | 1.722 ± 0.023 | | +0.6% |
| oracle word-end chunks (EXP-009 `space_end`) | 1.806 ± 0.026 | 0.56 | +0.7% |
| fixed (3,3) pooling + byte aux head (EXP-011 `fixed_aux`) | **1.821 ± 0.002** | 0.37 | +0.2% |
| fixed (3,3) pooling (EXP-011 `fixed`) | 1.832 ± 0.008 | 0.84 | +0.2% |
| fixed + residual warmup (`fixed_rwarm`) | 1.844 ± 0.005 | 0.82 | +0.2% |
| fixed + identity residual (`fixed_rid`) | 1.854 ± 0.035 | 0.10 | +0.2% |
| fixed + path-drop 0.3 (`fixed_rdrop`, 2 seeds) | 1.889 ± 0.009 | 0.04 | +0.2% |
| paper router α 0.03 (EXP-008 `base`) | 1.796 ± 0.001 | 0.69 | **+11.1%** |

- At honest compute the best learned-router or fixed-chunk H-Net is ~1.82, i.e. 0.19 bpb behind BPE; every
  "training signal" intervention (warmup, path-drop, identity residual) is neutral or harmful, the aux head gives
  0.01. The n-gram tables (parameters, not FLOPs) give 0.11. This is the quantitative version of the zoom-out: the
  gap is memory for identity, not signal routing. Hence EXP-017.
- EXP-014 (p-space quantile) rows are the collapsed runs (compression 440–1800, −20% to −58% realized compute),
  recorded for the post-mortem only.

### What tokenizer did the router learn? (`hnet/analysis/chunk_stats.py`, 128 val docs, 127 kB)
Three fully trained 2-stage runs (paper cosine router EXP-008 `base`; end-of-chunk MLP router EXP-003 `end_mlp` at
lr 1.5e-3; paper router at lr 6e-3 EXP-013 `lr6_nomult`) against a whitespace tokenizer on the same text
(`logbook/diagnostics/cs_*.json`):

| segmentation | bytes/chunk | types / chunks | singleton chunks | boundaries on word boundaries | word types segmented the same way every time |
|---|---|---|---|---|---|
| whitespace | 6.1 | 0.35 | 25% | 100% | 100% |
| stage 0, paper router | 2.7 | 0.18 | 12% | 17% | 67% |
| stage 0, end_mlp | 2.6 | 0.15 | 9% | **3%** | 73% |
| stage 0, paper lr 6e-3 | 2.7 | 0.17 | 12% | 19% | 82% |
| stage 1, paper router | 8.0 | 0.53 | **48%** | 14% | 53% |
| stage 1, end_mlp | 7.7 | 0.62 | **55%** | 7% | 30% |
| stage 1, paper lr 6e-3 | 7.8 | 0.58 | **52%** | 17% | 23% |

- Stage 0 learns a *character-pair* tokenizer: the most frequent chunks are single letters and punctuation
  (`' '`, `e`, `,`, `s`, `.`), and almost no boundary sits on a word boundary (3–19%, vs 44% of word boundaries hit
  at best). This is the level that the chunk vocabulary of EXP-017 sits on: it will behave like a character
  bigram/trigram table (useful, cf. the n-gram tables, but it cannot carry word identity).
- Stage 1 is **not a tokenizer**: half of its chunks are strings that occur once in 127 kB, and a word seen 3+ times
  is segmented the same way only 23–53% of the time. The innermost main network therefore sees mostly unique
  "tokens". No embedding table, no auxiliary loss and no capacity can give it word identity under this segmentation:
  this is the mechanism behind "main output less predictive than its input" and the 0.04-bpb contribution.
- The paper's end-to-end router objective (CE through the STE confidence + a ratio loss) contains nothing that
  rewards *consistency*: the router is free to place boundaries by context, and it does. A vocabulary needs types.
- Consequences: (i) EXP-018's 1-stage ratio-6 shape is the right place for a vocabulary (word-scale chunks, but
  only if they recur); (ii) next: routers that are consistent by construction, a **context-free router** whose
  score is a function of the last k bytes only (the same string is always cut the same way, like BPE), and a
  **surprise router** (boundary where the model's own next-byte entropy is high, the BLT patching rule) — both as
  scores for the rank decider, so compression stays pinned; measured with `chunk_stats` before looking at bpb.

### Consistency at initialization vs after training: the router objective is anti-tokenizer
`chunk_stats` on freshly initialized models (96 val docs): the contextual `end_mlp` router starts out **92–95%
consistent** (a recurring word is cut the same way almost every time: a random Mamba encoder has short memory, so
its state at a byte is a near-function of the last few bytes), while the trained `end_mlp` is 73% (stage 0) / 30%
(stage 1). Training *removes* consistency: the CE-through-STE signal rewards boundaries that help prediction in
context, never agreement across contexts. The context-free `local_mlp` is, counter-intuitively, *less* consistent at
init (8% at k = 8, 32% at k = 4): an 8-byte window reaches into the previous word, so the same word gets different
scores after different predecessors; a tokenizer is a function of the *current chunk's* string, not of a fixed
window. (EXP-019 stays as the cheap test of whether training fixes this; its windowed design is not the answer.)

Direct answer (new, `dc.router_consistency_weight`): a **self-consistency regularizer** on the byte-level router,
the variance of its logit across all positions in the batch that end in the same `n` = 4 bytes (hashed group means
via `index_add`, differentiable, negligible cost). The router keeps its context, but is pulled to agree with itself:
the same string → the same decision, which is what gives the hierarchy types to work with.

### EXP-020 first launch invalid: an unnormalized consistency penalty is satisfied by shrinking the logits
`cons1_s0` metrics: `router_consistency/s0` falls to 0.000 while `score_std/s0` falls 1.2 → 0.15 (the baseline's
rises to 3.9). Under rank decisions on normalized scores the forward pass is invariant to the logit scale, so the
cheapest way to zero a within-group variance is to make every logit (nearly) the same constant: the penalty bound
nothing. Lesson (same family as the EXP-014 drift, mirrored): once decisions are scale-free, every auxiliary term on
the router must be scale-free too. Fix: penalty = within-group variance / total variance (1 − R² of the 4-gram as a
predictor of the logit); relaunched as EXP-020 (second launch, same arms).

### EXP-017/018/019 interim (step 1000–1250, 3 seeds): the vocabulary works only at word scale
| arm | step 1000 bpb | main-network contribution | note |
|---|---|---|---|
| EXP-017 `quant_fix` (2-stage, fixed rank decider) | 1.950–1.964 | 0.01–0.02 | on par with / slightly ahead of threshold routers |
| EXP-017 `vocab` (stage-0 chunk table) | 1.937–1.969 | 0.03–0.05 | neutral at 1000; 0.02–0.04 ahead at 1500 |
| EXP-017 `vocab_pred` (w 0.5) / `vocab_pred_w1` | +0.04 / +0.08 vs `quant_fix` | ~0 | harmful: predicting 2.6-byte chunk buckets is noise for the decoder |
| EXP-019 `local` (context-free router) | 1.965–1.969 | 0.015 | identical bpb to contextual routers |
| EXP-018 `h1_quant` (1-stage, ratio 6) | 1.977–1.998 | 0.13–0.17 | final 1.845 / 1.861 / 1.845 |
| EXP-018 `h1_vocab_pred` (word-scale chunks + table + prediction) | **1.901 / 2.020 / 1.926** | **0.56 / 0.06 / 0.32** | bimodal across seeds |

- The 2-stage model's stage-0 chunks are character pairs: a table of them is worthless, and training the main
  network to predict them is harmful. At word scale (ratio 6) the same component gives −0.09 and −0.05 bpb at step
  1000 in two seeds and the main network's contribution rises from 0.13 to 0.56 bpb: the hierarchy carries the model
  for the first time in this project.
- Seed 1 did not adopt it. Its metrics show why: by step 230–560 its raw router scores were heavy-tailed
  (`score_std` 13–15 vs 6–8; `p_margin` 0.10 vs 0.3–0.4). Standardizing by a std that a few outliers dominate puts
  every other confidence at ≈ 0.5; the EMA dechunker then smooths with P ≈ 0.5, the hierarchy's output stays weak
  (`dechunk_rms` 0.65–0.93 vs 1.4–1.6 while `residual_rms` is 2.1) and the decoder commits to the residual path: the
  same "residual wins the race" failure as in EXP-002, now with a measurable trigger. Fix: median/MAD standardization
  with clipped z (default of the rank decider, unit test reproduces the failure mode) → EXP-021.

### EXP-017/018 finals (step 1763, honest compute, 3 seeds)
| arm | val bpb | main-network contribution |
|---|---|---|
| EXP-017 `quant_fix` (2-stage, end_mlp, fixed rank decider) | 1.7944 / 1.8043 / 1.7982 → **1.799 ± 0.005** | 0.03–0.04 |
| EXP-017 `vocab` (+ stage-0 chunk table) | 1.8015 / 1.7743 / 1.7770 → 1.784 ± 0.015 | 0.07–0.09 |
| EXP-017 `vocab_pred_w1` | 1.895 / 1.857 / 1.834 → 1.862 | 0.005–0.05 |
| EXP-018 `h1_quant` (1-stage, ratio 6, rank decider) | 1.854 / 1.861 / 1.845 → 1.853 | 0.13–0.17 |
| EXP-018 `h1_vocab_pred` (word-scale chunks + table + next-chunk prediction) | **1.720 / 1.863 / 1.737** | **0.75 / 0.13 / 0.55** |

- The fixed rank decider alone is worth as much as the paper router's 11% extra compute: 1.799 at budget vs 1.796
  at +11% (EXP-008 `base`), and 0.02 better than the previous honest-compute best (`fixed_aux`, 1.821).
- Word-scale learned chunks with a vocabulary and a chunk-level objective: −0.13 / −0.11 bpb vs the same model
  without them, at identical FLOPs, and the main network goes from decorative (0.13) to carrying the model (0.75 bpb:
  zeroing it gives 2.47). 1.72 equals the n-gram-table arms (1.714–1.722), which cheat the hierarchy by giving word
  identity to the byte stage; here the hierarchy earns it. The gap to BPE (1.634) is 0.09 for the adopted seeds.
- Seed 1 (1.863, contribution 0.13) is the heavy-tailed-scores failure diagnosed above; EXP-021 (robust decider,
  plus an arm that adds the n-gram tables as well) is running.
- Interpretation for learned tokenization: the paper's hierarchy was starved of *types*. Give the router's chunks an
  identity (a table keyed by their content) and a token-level objective, and the second level has something to
  learn; compression pinning (rank decisions) is what makes the comparison honest.

### What tokenizer did EXP-018 learn? (`chunk_stats`, final checkpoints, `logbook/diagnostics/cs_exp018_*.json`)
| model | bytes/chunk | singletons | boundaries on word boundaries | word boundaries hit | words cut consistently | top chunks |
|---|---|---|---|---|---|---|
| whitespace tokenizer | 6.1 | 25% | 100% | 100% | 100% | the, of, and |
| `h1_quant_s0` (no vocabulary) | 6.0 | 32% | 84% | 86% | 88% | `of `, `and `, `to `, ` `, `(`, `m `, `fro` |
| `h1_vocab_pred_s0` (adopted, 1.720) | 6.0 | **24%** | **97.7%** | **100%** | **99.9%** | `the `, `of `, `and `, `to `, `a `, `in `, `is ` |
| `h1_vocab_pred_s1` (not adopted, 1.863) | 6.0 | 37% | 0.6% | 0.6% | 87% | `s`, `d`, `e`, `ing`, `ed`, `er`, `ly`, `ion`, `ment`, `ation` |

- Given a vocabulary and a chunk-level objective, the router converged to a *word tokenizer* by itself: every word
  boundary is a chunk boundary and the same word is always one token. That is the learned tokenization H-Net
  promised, obtained without heuristics, and it is what let the main network take over 0.75 bpb of the work.
- The non-adopted seed learned a *morphological* tokenizer (suffixes `ing`/`ed`/`ly`/`ion`/`ment`/`ation` as
  tokens, cuts before the space): also a consistent segmentation (87%), but more fragmented (37% singletons) and
  slower to make the hierarchy useful; the residual path won the race first. Both outcomes are reachable from the
  same objective: adoption is decided early (step 200–600), by whether the hierarchy's signal reaches the decoder
  before the residual path consolidates. That is the next dynamics problem (EXP-021 removes one trigger, the
  heavy-tailed-score damping).

### EXP-021 (median/MAD decider) stopped: the MAD collapses under a word tokenizer
`h1r_vocab_pred_s0`: from step ~650 the robust scale `score_std/s0` falls 4.4 → 0.0 while `p_margin` pins at 0.50;
grad norm 0.3 → 4.6e3 (step 900) → 1.9e6 (step 1050), val bpb 3.25 at step 1000. With a word tokenizer 5/6 of the
positions are "not a boundary" and get near-identical logits, so the median and the MAD both sit inside that mode and
the MAD → 0; the ±4 clip protects the forward pass, not the gradient through 1/σ. Seeds 1–2 did not diverge but did
not adopt the hierarchy either (1.98 / 2.00 at step 1000 ≈ baseline). `h1r_quant` ≡ `h1_quant` (the decider change
is invisible without a vocabulary), `h1r_vocab_pred_ng` reached 1.82 at step 1000 (= `paper_ng`) before the stop.
Fix: mean/std over the central 98% of the scores (outlier-proof, cannot collapse unless 98% of scores coincide; unit
test with 5/6 identical scores). Rule, stated generally: a scale estimator for boundary scores must straddle both
modes of a bimodal distribution; medians do not. → EXP-023 (same arms + residual warmup; EXP-022 draft superseded).

### EXP-023 tokenizer statistics: adoption and the kind of tokenizer go together (`cs_exp023_*.json`)
| checkpoint | bpb | main contribution | boundaries on word boundaries | words cut consistently | top chunks |
|---|---|---|---|---|---|
| `vp_ng_s2` (adopted) | 1.656 | 0.42 | 95% | 99.4% | `the `, `of `, `and `, `to `, `a ` |
| `vp_s1` (not adopted) | 1.855 | 0.18 | 4% | 89% | `s`, `d`, `ing`, `ed`, `er`, `ly`, `ion`, `ment` |

Same picture as EXP-018 with the same seed: the seed that ends up with a *suffix/morpheme* tokenizer is the one whose
hierarchy is not adopted; the seeds that find the *word* tokenizer are adopted and reach 1.66–1.73. Deterministic per
seed across two deciders, so the bifurcation is in the router's initialization or the first few hundred steps of
data, not in the decision rule. Open question whether the tokenizer causes the non-adoption (morpheme chunks are
less predictable, a weaker chunk-level signal) or both follow from the same early state; either way the fix is to
steer the first ~300 steps (residual warmup `vp_rwarm`, pending; or a brief whitespace-prior on the router).

### EXP-024: the tokenizer curriculum makes adoption reliable; H-Net matches BPE at equal FLOPs
`prior` (vocabulary + next-chunk objective + n-gram tables + a 10%-of-training whitespace warm start on the router):
**1.6337 / 1.6346 / 1.6364 → 1.635 ± 0.001 bpb**, BPE **1.634 ± 0.002**, same data, same training FLOPs (realized
compute +0.3% of budget for every arm, `logbook/sweep_summary.py`). All three seeds adopt the hierarchy by step 250
and the main network ends worth 0.56–0.60 bpb. `temp4` (sharper confidences) goes the other way: 1.723, hierarchy share
0.02–0.21.

Final checkpoint's tokenizer (`cs_exp024_prior_s0.json`), after 90% of training with no prior: 97% of boundaries on
word boundaries, 99.3% of word boundaries hit, 99.8% of recurring words cut the same way, 25% singleton chunks (the
whitespace tokenizer: 25%); the 3% of non-whitespace cuts are around punctuation and brackets. The router kept the
word tokenizer it was pointed at and refined it; it did not drift back to character pairs or suffixes.

### Where this leaves the project (honest-compute view, tiny scale, 50M token-eq)
| model | val bpb | hierarchy's share |
|---|---|---|
| BPE (tiny_bpe_2m) | 1.634 ± 0.002 | — |
| paper H-Net, 2-stage, as published (α 0.03) | 1.796 at **+11% compute**; ~1.87 ± 0.11 at budget (EXP-006) | 0.04 |
| + fixed rank decider (EXP-017 `quant_fix`) | 1.799 ± 0.005 | 0.03 |
| 1-stage ratio 6 + word-scale chunk vocabulary + next-chunk objective (EXP-023 `vp`) | 1.773 ± 0.071 (1.73 when adopted) | 0.5–0.7 |
| + byte n-gram tables (EXP-023 `vp_ng`) | 1.682 ± 0.034 (1.66 when adopted) | 0.4 |
| + whitespace curriculum (EXP-024 `prior`) | **1.635 ± 0.001** | 0.57 |

The path from 1.80 to 1.635 was: (1) pin compression with rank decisions so comparisons are honest and routers cannot
collapse; (2) notice that the paper's learned chunks are not types (half of them singletons) and that the gap is
word identity; (3) give the chunks a vocabulary and a token-level objective; (4) move the chunk scale to words;
(5) remove the adoption lottery with a 10% warm start. What is still open: the two-level hierarchy (the vocabulary at
a (6,3) second level), scale (10M → 125M, more data), downstream tasks, and whether the warm start can be replaced by
something that does not name whitespace (a surprise-based or consistency-based start).

### A100-40GB throughput (GCP spot, compiled blocks, 50 steps, `scripts/bench`)
| model | MFU | bytes/s | note |
|---|---|---|---|
| BPE L04 / L06 / L08 / L12 | 11% / 22% / 34% / 41% | 1.87M / 1.21M / 1.07M / 0.52M | L08 whole-model compile 40.5%, eager 20%; Muon = AdamW speed |
| H-Net L04 / L08 (mb 8) / L12 | 6.5% / 17% / 24% | 0.61M / 0.36M / 0.20M | ~3x slower per byte than BPE at equal FLOPs/byte |
| packing 3 L04 runs per GPU | | BPE 3x0.55M, H-Net 3x0.22M | no gain over one run per GPU |

Prime: no H100 offers, H200 pods need SSH (unavailable in this container), GPU sandboxes need an account grant;
ladder moved to GCP A100 spot (user decision). Ladder cost at these speeds: 79 A100-h (~$114 spot); the warm-start
H-Net arm is dropped at 1e18 to fit +$100 (~$73 + preemption overhead).

### mix_v1 full build
25.0 GB train (4.96M docs, 98 shards) + 41 val sets, 2906 s on an 8-core CPU VM; 4.126 bytes/token overall with the
32k in-corpus BPE (en 4.53, code 3.46, math 3.50, multilingual 3.15). The ladder configs use 4.145 (measured on the
small build); the 0.5% mismatch in the H-Net's target compression is kept for consistency across the ladder.

### Deep dive: the adoption race, step by step (`logbook/figures/adoption_dynamics.png`, `cs_exp023_vp_s{0,1}_step*.json`)
EXP-023 `vp` (1-stage ratio 6, vocabulary + next-chunk loss), adopted seeds 0/2 (1.73) vs non-adopted seed 1 (1.85),
plus the warm-started `prior_s1` (1.63):
1. **Steps 0–60, identical in every seed.** The hierarchy is the only path (zero-init residual): its output rms is
   1.5, the router starts confident (margin 0.36) and, per the init analysis, ~92% consistent.
2. **Steps 60–130, the escape hatch opens.** The residual projection comes alive (rms 0 → 1.0) and the decoder
   switches to it: hierarchy output rms crashes 1.6 → 0.7 in *all* seeds and the router's margin collapses 0.36 →
   0.15. The router's only training signal is the decoder's use of the dechunked output (through the STE), so when
   the decoder stops listening the router stops learning and drifts.
3. **Steps 130–500, the bifurcation.** The undecided router falls into a basin. Seeds 0/2: word boundaries (93% of
   cuts on whitespace by step 500); seed 1: suffix boundaries (0.7% on whitespace; top chunks `s`, `d`, `ed`, `ing`,
   `ion`). The next-chunk CE is *the same* in both (4.9 at step 500): a suffix sequence is as predictable as a word
   sequence, so the chunk-level objective does not prefer the useful tokenizer. Besides, that objective is computed on
   the inner outputs before dechunking and the STE sits on the dechunked path, so it sends the router no gradient at
   all; the tokenizer objective never trained the tokenizer.
4. **Steps 400–600, lock-in.** Word chunks carry identity the byte decoder can use, so the decoder returns to the
   hierarchy (rms 0.75 → 1.4), the STE gradient returns, the margin recovers to 0.4 and the segmentation hardens to
   99.9% consistency. Suffix chunks tell the decoder about the word that just ended, not the one coming; the decoder
   never returns, the router's raw logits go heavy-tailed (std 17), margin stays 0.10, residual grows to 2.3.
5. **The warm start short-circuits step 2–3:** the hierarchy's output never falls below 0.95, the margin never below
   0.29, and the next-chunk CE is lower from step 0.

So the failure is a positive-feedback race between two parallel paths into the decoder, started by the bypass opening
while the router is undecided, and the router has no objective of its own that favours word-like types. Three things
follow: (a) the router needs a training signal that does not pass through the decoder's adoption (`dc.chunk_ste`: the
next-chunk loss now reaches the router); (b) the bypass should not open while the tokenizer is undecided
(`dc.residual_gate=margin`, or no residual at all in the vocabulary regime, never tested); (c) at this scale the outer
networks hold 60% of the compute, so the bypass can do most of the job by itself; at L12 they hold 12% (docs/LADDER.md),
so the race may be a small-scale artefact — the ladder decides. The next-chunk objective's indifference between word
and suffix tokenizers is the deeper problem: predictability is not usefulness; a tokenizer objective must be tied to
the byte-level loss (BPE's "compress the corpus" criterion is also predictability-blind, which is why it needs the
pretokenizer). EXP-027 tests (a), (b) and a thin-outer/heavy-main allocation at equal FLOPs.

### Why the issues were found late: CLAUDE.md, a step-0 gradient audit, an invariance suite, a visibility tool
User request after the adoption deep dive: explain why these failures took weeks to find, and write the meta of
finding them into `CLAUDE.md`. The table there lists nine misses with how we found each and when we could have:
every one was derivable from the gradient graph or from the first 200 steps of a CPU run, and we found them from the
final bpb of paid sweeps instead. Twelve rules follow (gradient graph first, step-0 audit, every aux loss is a
controller, what the lazy optimizer does, statistics that survive bimodal scores, mechanism metrics from step 1, seeds
are experiments, cheap before expensive, the paper as a list of fragile components, a standing invariance suite, fail
loudly, post-mortem against the list). Three are now code:
- `hnet/analysis/grad_audit.py` (tests `test_grad_audit.py`): autograd of each objective term against each parameter
  group at init and after N SGD steps, plus the mechanism metrics and plain-language warnings. On the prior-free
  best configuration it prints at step 0 that the next-chunk CE reaches only the chunk table (zero to the router, zero
  to the main network while the table is zero-initialised) and that with `dc.chunk_ste` the path opens.
  Side finding: `residual_proj` is "zero-init" only in its weight; the default `nn.Linear` bias is live (rms 0.064 at
  init), so the "bypass closed at init" assumption was never exactly true.
- `tests/test_invariances.py`: causality (a change at byte 200+ leaves logits before 200 bit-identical) and packing
  equivalence (two documents packed with `cu_seqlens` ≡ run separately) for the paper router, end_mlp + rank decisions,
  and the chunk vocabulary, with and without n-gram tables. All pass. Documented rather than fixed: in training mode
  the rank decision ranks over the whole batch (BatchNorm-like coupling); in eval mode it is causal. No cached-decoding
  test yet because there is no incremental decoding path.
- `logbook/training_report.py` + `$S2/report.sh`: every status update is TL;DR + per-sweep table + val-bpb curves +
  stage-0 mechanism panels (margin, undecided fraction, score std, hierarchy vs bypass rms, next-chunk CE, grad norm) +
  LR plots, with computed alerts. First report: the bypass rms is 5–40× the hierarchy's in every H-Net run of EXP-025
  and the ladder LR sweep, i.e. the adoption race is the normal state of these runs, not an exception.

Also today: `dc.chunk_ste` and `dc.residual_gate` implemented (tests `test_adoption_options.py`; the first version of
the chunk-STE test was wrong twice, both times for reasons the audit now prints: the ratio loss does reach the router
through G, and the zero-initialised table kills the next-chunk gradient at init). EXP-027 launched (two L4 spot VMs).
`hnet-ladder-lr` finished its queue (hnet / hnet_ws LR grid, results mirrored) and was deleted; the BPE retries and the
LR extension run on `hnet-ladder-lr2`, which was preempted and resumed three times in 20 minutes.

### Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched
| Muon LR | 0.0025 | 0.005 | 0.01 | 0.02 | 0.04 | 0.08 |
|---|---|---|---|---|---|---|
| bpe | 1.796 | 1.736 | 1.706 | **1.622** | 1.784 | 2.008 |
| hnet | 1.934 (router undecided, margin 0.03) | 1.888 | **1.861** | 1.905 | 2.049 | 2.210 |
| hnet_ws | 1.878 (margin 0.07) | 1.822 | **1.820** | 1.845 | 2.028 | 2.209 |

BPE has a sharp optimum at 0.02 (0.08 bpb worse one grid step either side); both H-Nets prefer 0.01 and are flatter.
At the prior optimum rung the gap is large: H-Net trails BPE by 0.24 bpb prior-free and 0.20 with the warm start, at
equal FLOPs and data, against 0.00 at tiny scale with the curriculum (EXP-024). The mechanism panels say why: the
bypass rms ends 5-25× the hierarchy's in every H-Net run above LR 0.01, and at the lowest LRs the router never
decides. The iso-FLOP ladder follows the blog's protocol (BPE's best LR transferred to every run, `best_mlr.bpe=0.02`),
which costs the H-Nets ~0.04 relative to their own optimum; noted as a caveat in the write-up. 1e16 rungs L03-L06
(bpe / hnet / hnet_ws) launched on `hnet-ladder-1e16` (4×A100 spot, 12 runs); 1e17 held until EXP-027's first evals
decide whether the ladder's H-Net arm should carry the chunk-STE / gated-bypass change. Both LR VMs deleted
(hnet-ladder-lr $4.12, hnet-ladder-lr2 ~$3.5).

### Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers
Independent ideation (subagent, `$S2/ideas_first_principles.md`) proposed, among ten structural variants, a router
objective based on the hierarchy's measured gain, with a no-training test: on EXP-023 `vp` seed 0 (word tokenizer)
vs seed 1 (suffix tokenizer), bits saved per byte by the hierarchy (full model minus zero-main), by byte class
(`hnet.analysis.diagnose`, 64 val docs, `logbook/diagnostics/diag_exp023_vp_s{0,1}[_step500].json`):

| checkpoint | all bytes | chunk start | word start | space | word internal |
|---|---|---|---|---|---|
| step 500, word seed 0 | 0.165 | 0.172 | 0.164 | 0.269 | 0.127 |
| step 500, suffix seed 1 | 0.068 | 0.071 | 0.026 | 0.084 | 0.068 |
| final, word seed 0 | 0.460 | 0.651 | 0.595 | 0.529 | 0.366 |
| final, suffix seed 1 | 0.192 | 0.317 | 0.221 | 0.023 | 0.213 |

Already at step 500, in the middle of the bifurcation, the word tokenizer's hierarchy is worth 2.4× more per byte
and 6× more at word starts than the suffix tokenizer's. The next-chunk CE was identical for the two (4.9 at step 500).
So a *usefulness* objective for the router exists and is not blind: credit each byte's gain (CE with the hierarchy
zeroed minus CE with it) to the chunk that covers it, and train the router's decisions on that advantage
(REINFORCE with a baseline, or the chunk STE weighted by the gain). This is the first candidate router objective
that prefers the right basin for the right reason. Next: implement `dc.router_gain_weight` (a second decoder pass on
the residual-only input, cheap at 4 outer layers) and run it prior-free, 3 seeds.

Two offline segmentation probes (`scripts/analysis/segmentation_probe.py`, 128 val docs, ratio 6,
`logbook/diagnostics/segmentation_probe.json`) as candidate *teachers* replacing the whitespace prior:
| segmentation | on whitespace | whitespace hit | consistency | types | unigram bits/byte |
|---|---|---|---|---|---|
| whitespace | 1.00 | 1.00 | 1.00 | 11343 | 1.78 |
| 5-gram entropy spikes (Harris/BLT) | 0.66 | 1.02 | 0.60 | 15647 | 2.01 |
| 5-gram entropy, absolute | 0.29 | 1.02 | 0.41 | 13338 | 1.68 |
| unigram MDL (K 10, no length prior, 4 EM iters) | 0.13 | 0.64 | 0.06 | 21389 | 1.50 |
Entropy spikes are a plausible language-agnostic prior (two thirds of cuts on word ends, top chunks are words) but
weaker than whitespace on English; unigram MDL without a length prior does what the theory predicts (long rare units,
84% singletons) and is not a tokenizer as configured. Neither replaces the whitespace curriculum on this data; the
entropy prior is the candidate for code/multilingual rungs where whitespace is the wrong prior.

EXP-027 at step 500 (val / zero-main): `cste` 2.022, 1.947 / 2.054; `cste_rgate` 2.000, 1.962 / 2.140; `rgate`
1.999, 1.979 / 2.404; reference `vp_ng` 1.994, 1.953, 1.956 / 2.02-2.05. Same loss as the reference so far; the
margin gate makes the hierarchy load-bearing from the start (0.42 bpb of contribution at step 500 vs 0.03-0.10 for
the reference), which is the mechanism it was meant to change. Finals at step 1763 decide.

### EXP-027 reads out negative (seeds 0-1): refereeing the race does not fix the tokenizer
`cste` 1.778 / 1.868, `cste_rgate` 1.742 / 1.882, `rgate` 1.923 / 1.699 against `vp_ng` 1.720 / 1.669 / 1.656. The
seed lottery survives all three interventions and the means are worse. The margin gate makes the hierarchy carry
0.3-0.4 bpb from step 500 (the reference carries 0.03-0.10 at that point), which is exactly the mechanism change it
was designed for, and the loss is no better: forcing adoption of a bad tokenizer is not progress. Together with the
gain measurement above this settles the diagnosis: the problem is not *when* the decoder adopts the hierarchy but
*which* tokenizer the router falls into, and only an objective that scores usefulness (EXP-028) addresses that.
Operational: `hx-exp025-muon-1of4` (bpe_muon 0.025) shut itself down after a mid-install preemption; deleted and
recreated in europe-west4-a (runs resume from the bucket checkpoints).

### Ladder 1e16 done (L03-L06, one seed, BPE's Muon LR 0.02 for every arm, 10 MB mix_v1 val)
| rung (D) | BPE | H-Net prior-free | H-Net warm start | gap prior-free | gap warm start |
|---|---|---|---|---|---|
| L03 | 1.695 | 1.879 | 1.823 | +0.18 | +0.13 |
| L04 (256) | **1.622** | 1.907 | 1.864 | +0.29 | +0.24 |
| L05 | 1.756 | 1.985 | 1.939 | +0.23 | +0.18 |
| L06 | 1.841 | 2.083 | 2.035 | +0.24 | +0.19 |
At 1e16 FLOPs the BPE iso-FLOP optimum is L04 and both H-Nets are best at L03 (the smallest rung): at this budget
the H-Net wants a smaller model and more bytes. (Corrected later: this is *consistent* with the blog, whose H-Net
optimum is also one rung below BPE's; see docs/BLOG_GAP_ANALYSIS.md.) The gap is 0.18-0.29 bpb prior-free and 0.13-0.24 with the warm start, at
equal FLOPs and data; the bypass dominates (rms 5× the hierarchy's) in every prior-free run. One seed each; the L04
BPE run is byte-identical to the LR-sweep run (reproducible pipeline). Caveat: BPE's LR transferred to the H-Nets
costs them ~0.04 (their own optimum is 0.01). → 1e17 rungs next (decided after checking the cost table).

### Correction: EXP-027 is neutral, not negative
The "negative" readout above used per-seed values of runs that were still at step 1500 as if final (the report's
step column showed the sweep maximum, not the seed's). Finals: `cste` 1.778 / 1.661 / 1.640 (1.693), `cste_rgate`
1.742 / 1.679 / 1.648 (1.690), `rgate` 1.746 / 1.699 / running, against `vp_ng` 1.682: same mean, same lottery.
The conclusion stands (the interventions change the mechanism, not the tokenizer the seed falls into) but the
magnitude claim was wrong. `training_report.py` now prints interim seeds with their step and leaves them out of the
mean. The EXPERIMENTS.md entry is corrected.


### Blog gap analysis (docs/BLOG_GAP_ANALYSIS.md): we have been testing below the blog's smallest run, with a smaller outer network
An independent agent re-read both posts and extracted the per-run data embedded in their plot bundles (30 BPE +
30 H-Net iso-FLOP runs). Findings that change the plan:
- **The blog's ladder starts at 1e18 FLOPs**, every run sees >= ~0.6B tokens, and its two most data-starved 1e18 runs
  (3-8 tokens/param) *lose* to BPE by 0.015-0.038 bpb; the H-Net lead grows with tokens/param (up to -0.063 on its
  D=256 rung at ~4.4B tokens). Our runs (0.03-0.5B tokens) are all below the blog's smallest. Its 1.28x multiplier is
  constant by construction of a shared-floor/shared-exponent fit and says nothing about 1e16-1e17. A 0.2-0.3 deficit is
  still far beyond the <= 0.04 small-data effect in the blog's own data, so protocol and implementation matter too.
- **Vocabulary size sets the H-Net's outer budget, and I ruled this out wrongly earlier.** The outer encoder/decoder are
  FLOP-matched to the BPE embedding + head. Verified from configs/ladder/ladder.json (share of BPE FLOPs in the head,
  ours with 32k vs the same rung with the blog's 128k): L03 0.60 -> 0.86, L04 0.48 -> 0.79, L05 0.39 -> 0.72,
  L08 0.22 -> 0.53. At our rungs the H-Net's byte-level networks get 2.4-4x less compute relative to the main network
  than the blog's, and our BPE wastes less on its head. My earlier dismissal compared parameter ratios across different
  rungs, which is the wrong comparison.
- Our LR transfer stacks the paper's Adam-derived `auto` per-stage multipliers (x2.35 outer at L04) on top of BPE's Muon
  LR; the blog describes no such multipliers. The H-Net's own best Muon LR is half of BPE's.
- The H-Net's optimum one rung below BPE's matches the blog and dnaHNet (arXiv 2602.10603); the 1e16 grid should add L02.
- Code is the strongest cheap signal in the blog (per-language gaps 2-4x the overall gap at 1e18); our per-source
  validation sets exist and are now being scored on the 1e16 checkpoints.
Ranked missing ingredients (report section 3): training length/steps; 128k-vocab outer budget; LR transfer under Muon;
compression control / router form (strong ratio loss instead of the rank decider, cosine-style end router, compression
ramp per ATDC arXiv 2605.30080); evaluation protocol (per source, 2 seeds per frontier point, L02 rung). First, a
cheap one-factor-at-a-time bridge from EXP-024 (BPE parity) to the 1e16 L04 rung (-0.24): six factors change between
them (steps 1763->728, batch 131k->545k bytes, AdamW->Muon+auto multipliers, FineWeb-Edu->mix, ratio 6->4.145,
GPT-2 50k -> in-corpus 32k baseline).

### Implementation audit (docs/IMPL_AUDIT.md, tests/audit/): the paper path is correct; two small fixes
An independent audit transcribed the official dynamic-chunking code (goombalab/hnet) into a CPU test and compared our
default model against it: forward pass and every parameter's gradient agree to 6e-8 (fp32, 1- and 2-stage, packed
batch with a 1-byte segment). Data and eval are correct: byte and BPE models score identical text, BPE bpb = token
NLL in bits / UTF-8 bytes of the same text, train and val disjoint, same documents in the same order. The residual
projection's live bias is as in the official code (only the weight is zero-initialized). Findings:
- B1 (may change results, tiny scale only): equal-FLOPs pairs are equal at the *target* compression; the paper's
  threshold router realizes 2.6-2.9 instead of 3 at stage 0, so EXP-002-era H-Nets ran 7-12% over BPE's FLOPs (favours
  the H-Net; "trails BPE" results are conservative). The ladder and rank-decider runs are exact (within 0.5%).
- B2 fixed: learned end routers computed their score in bf16 under autocast (~800 distinct values per 32k-byte batch
  for the rank decider). Now fp32 (`hnet/model/routing/end.py`). Effect on past runs judged negligible (one tie at the
  cut on the checked checkpoint), but runs from now on are not bit-identical to earlier ones in bf16.
- B3 fixed: casting a whole H-Net to bf16 crashed in residual_proj (eval only). B4: duplicate `set_progress` removed.
- Deviations that may matter (not bugs): no weight decay on norms / A_log / D / dt_bias (official decays them); the
  rank decider's batch coupling and eval-time running threshold; the paper's AdamW LR multipliers applied under Muon.

### 1e17 L05 finished: more data per run halves the gap at fixed model size
Same L05 model (D 384), 1e16 vs 1e17 FLOPs (0.10B vs 0.52B BPE tokens), one seed, BPE's Muon LR for all:
| budget | BPE | H-Net prior-free | gap | H-Net warm start | gap |
|---|---|---|---|---|---|
| 1e16 | 1.756 | 1.985 | +0.23 | 1.939 | +0.18 |
| 1e17 | 1.224 | 1.360 | +0.14 | 1.353 | +0.13 |
At matched steps inside the 1e17 run the gap went +0.25 (step 394) → +0.41 (788) → +0.34 (985) → +0.20 (1379) → +0.14
(final): the H-Net is slower early and catches up late, and the warm start's advantage shrinks to 0.007 with data.
Consistent with the blog's per-run data (H-Net loses in its most data-starved runs, lead grows with tokens/param).

EXP-028 interim (step 1500): gain 1.804 / 1.797, gain_cste 1.801 / 1.851, gain_rgate 1.819 / 1.823, against `vp_ng`
1.757 / 1.710 / 1.699 at the same step: 0.05-0.10 worse. The two `gain` seeds agree to 0.007 (the reference's
spread is 0.06), so the objective does remove the lottery, but into a worse basin; tokenizer statistics of the
finals will say which.

### Per-source evaluation of the 1e16 ladder (L03, L04; `logbook/tables/per_source/1e16_*.json`; 200 kB per source)
| group | L03 BPE | L03 H-Net | L03 H-Net ws | L04 BPE | L04 H-Net | L04 H-Net ws |
|---|---|---|---|---|---|---|
| English | 1.591 | 1.777 | 1.707 | 1.536 | 1.803 | 1.738 |
| code (14 langs) | 2.067 | 2.246 | 2.221 | 1.777 | 2.259 | 2.322 |
| held-out code (10 langs) | 2.618 | 2.706 | 2.709 | 2.152 | 2.706 | 2.767 |
| math | 2.031 | 2.228 | 2.163 | 1.885 | 2.233 | 2.195 |
| multilingual | 2.441 | 2.729 | 2.676 | 2.430 | 2.790 | 2.756 |
- At L03 the H-Net's deficit is smallest on held-out code: +0.02 to +0.08 bpb for Lua, Haskell, Visual Basic, Scala,
  Powershell, Perl (prior-free H-Net), and Go is at parity (-0.003, warm start), against +0.12 to +0.19 on English. This
  is the direction the blog reports (the H-Net transfers to unseen code better than BPE does), at 1e16 and one seed.
- At L04 the picture flips: BPE improves 0.29 bpb on code and 0.47 on held-out code going from L03 to L04 while
  improving 0.055 on English, and the H-Nets do not improve at all. BPE's L04 gain on code is suspiciously large and
  one seed; worth a second seed before drawing conclusions. The H-Net's realized compression on code is 2.8-3.8
  bytes/position (BPE 2.3-4.3 bytes/token): it does not compress code more than BPE at this scale, unlike the blog.
- Largest deficits: Chinese (+0.52), Russian (+0.36 to +0.45), assembly and SQL; smallest: Arabic (+0.14), English.

### Why the bypass wins under Muon: its size follows the LR schedule (evolve g1, optimization proposer)
CPU probe through the real trainer (L03 hnet_ws, benchmark schedule and LRs, one 2048-byte window per step, 150 steps):
the zero-initialized bypass matrix `residual_proj` sits in the stage-0 Muon group at 0.02 × 2.35 (`auto` multiplier) =
0.047 with no weight decay. Muon's orthogonalized step has a fixed size per step, so a zero-init matrix with a
consistent gradient grows its singular values by ≈ LR per step: its top singular value tracked 0.8 × the cumulative LR
at every step to 150 (3.90 at step 150, still linear), effective rank ~150/192. The bypass rms crossed the hierarchy's at
step ~85 and was 2.8× by step 150 (the GPU run: 3.8× at step 300, 5.1× at the end), while the encoder-output and
main-output gains barely moved (1.15, 1.04). So the "bypass opens at steps 60-130" race is, under Muon, mostly a clock:
the bypass's scale is set by the LR schedule, not by what it is learned to need, and nothing anchors it (the decoder
normalizes its input). Under AdamW the same zero-init matrix grows too (EXP-023), but this is the first measurement of
the rate. Candidates g1-opt-01 (drop the Adam-derived ×2.35 on stage-0 Muon matrices) and g1-opt-02 (decoupled weight
decay 0.5 on `residual_proj` only, which bounds its size at ~1/wd; probe: plateau at 1.50, bypass ratio held at 1.2)
test whether anchoring the bypass is a fix or only a symptom.
Also: my hand-written note in the generated docs/LADDER.md broke `test_committed_files_match_the_generator`, which the
VM test gate would have caught at launch. Moved to docs/ANALYSIS_LADDER.md; CLAUDE.md now says not to edit generated files.

### Interpretability audit (logbook/interp/REPORT.md; 281 runs' metrics, 20 checkpoints' activations): corrections
Ten findings, the ones that change the record:
1. **The seed lottery is the initialization.** Data order is identical across seeds (`shuffle_buffer: 0`); arms with the
   same parameter layout get the same starting router per seed. In all 5 prior-free arms of the n-gram layout seeds
   finish in the same order (s2 best, s0 worst; p = 1/1296); no intervention except the whitespace prior ever changed
   which seed loses (decider, temperature, residual warmup, chunk STE, margin gate).
2. **The two basins are "cut at the space" and "guess the space".** Word seeds put a boundary on every whitespace byte;
   suffix seeds never do: 55-64% of their cuts are on the last letter of a word (the router predicts "next byte is not
   a letter", AUC 0.75-0.80; under end semantics it cannot see that byte). "Morphological tokenizer" was wrong.
3. **The zero-main probe overstates the hierarchy 1.4-26×** (it also removes a constant offset the decoder relies on).
   Mean-ablation: EXP-006 hnet1 "0.63 bpb" is 0.025 bits of information; EXP-002/008 "0.69" is 0.18-0.27.
4. **The bypass/hierarchy rms ratio is not an adoption signal**: under Muon it tracks the LR (log-log r = 0.96, n = 28).
   The best tiny H-Net (EXP-025 prior_muon 0.025, 1.630) runs at ratio 7.6 with a hierarchy worth 0.38 bpb. The CLAUDE.md
   rule and the report's alert built on it are retired; the basin is now logged directly (`bnd_on_ws/s0`).
5. **There was never a switch away from the hierarchy**: it carries ≤0.005 bits for the first ~100 steps while the
   bypass is informative from step ~35. Gating or delaying the bypass cannot create adoption (consistent with EXP-027).
6. **The ladder's prior-free H-Nets (L03/L04) are in the losing basin** (≤8% of cuts on whitespace, hierarchy worth
   0.03-0.04 bits); the warm-start ones work (0.24-0.27 bits), but the 4.15 bytes/chunk target forces ~30% of their
   cuts off whitespace because spaces occur every ~6 bytes.
7. **The losing seed is readable at step ~210 from `dechunk_rms`** (within-arm |r| ≥ 0.5; AUC 0.88 at step 250).
8. **Tiny "BPE parity" is against BPE at an untuned LR** (BPE 1.564 at 6e-3 vs 1.634 at 1.5e-3; H-Nets flat in LR);
   tuned gap ≥ 0.066. At parity the H-Net wins on 2-4-byte tokens and loses on long whole words.
9. Where the hierarchy helps it helps on sentence structure: punctuation 1.6-2.1 bits/byte, newlines 1.1-1.2, digits ~1;
   the next word's first letter ~0.5, falling to ~0.1 by the fifth letter.
10. Other corrections: the adoption deep dive's margin timing (it rises over steps 60-100, collapses at 150-300); EXP-020
   "neutral" hides an undecided router; EXP-027's gate was undone by the optimizer (the pre-gate bypass grew to 4.6-6.8);
   the chunk table gives word identity only without n-gram tables; the race is decided the same way at L03/L04.
Recommendations adopted: log the basin and use mean-ablation; land the router in the word basin directly (g1-router-01's
self-surprise teacher is exactly the suggested next-byte-entropy prior); LR-tune before claiming parity; bound
`residual_proj` under Muon (g1-opt-02). Evolve consequence: with deterministic data order, prior-free candidates'
seed outcomes are partly a function of their parameter layout's init; generation 1 compares candidates with controls at
the same seeds and now logs `bnd_on_ws/s0` so each run's basin is visible.

### 1e17 ladder L05-L07 finals (one seed; BPE's Muon LR 0.02 for all)
| rung | BPE | H-Net | gap | H-Net ws | gap |
|---|---|---|---|---|---|
| L05 | 1.224 | 1.360 | +0.14 | 1.353 | +0.13 |
| L06 | 1.246 | 1.474 | +0.23 | 1.567 | +0.32 |
| L07 | 1.267 | 1.624 | +0.36 | 1.627 | +0.36 |
The H-Net degrades much faster than BPE as tokens/param fall; both are best at the smallest rung run (L05). Add L03/L04
at 1e17 before fitting multipliers (the iso-FLOP minimum is at the grid edge).

Evolve generation 1, seed-1 finals so far: g1-arch-02 (offset-resolved read-back of chunk vectors) 1.798 vs the warm-start
control 1.827 (-0.029); g1-arch-01 and g1-alloc-01 tie the control (1.826); g1-alloc-02 (8 bytes/chunk) 1.867. At 6-8
bytes/chunk 90-98% of boundaries land on whitespace (controls at 4.15: 63-69%), so word alignment alone does not help.

### 1e17 ladder complete; first two-budget fit (`logbook/tables/ladder_1e16_1e17_*_fit.json`, figures `ladder2_*`)
L08: BPE 1.312, H-Net 1.789 (+0.48), ws 1.683 (+0.37); L09: BPE 1.333, H-Net 1.853 (+0.52), ws 1.753 (+0.42).
Compute-optimal losses (iso-FLOP minima at the small edge of the grid for every arm and budget):
| | 1e16 | 1e17 | exponent α | compute multiplier vs BPE (95% CI) |
|---|---|---|---|---|
| BPE | 1.660 | 1.224 | 0.132 | 1 |
| H-Net prior-free | 1.887 | 1.355 | 0.144 | 0.43× [0.30, 0.63] |
| H-Net warm start | 1.836 | 1.353 | 0.133 | 0.47× [0.31, 0.60] |
At 1e16-1e17 the H-Net needs ~2.1-2.3× BPE's compute for the same loss. The prior-free arm's steeper exponent narrows the
gap (+0.23 → +0.13); a naive extrapolation crosses BPE near 6e20 FLOPs, which two budgets and one seed cannot support.
Caveats: BPE's Muon LR transferred to the H-Nets (costs them ~0.04), the `auto` LR multipliers stacked on Muon (g1-opt-01
shows dropping them is worth ~0.03 at L03), prior-free arms in the losing basin at L03/L04 (interp audit), iso-FLOP
minima at the grid edge (add L02/L03 at 1e17). The ladder VM shut down after finishing its queue (4.1 h, $24.3).
Generation 1 of the evolve loop was preempted at 23:13 UTC and stayed down during a session pause; resumed 05:10 UTC with
a 3.5-hour cap. EXP-026 KDA abandoned (its VM stopped at its time cap; the kernel is 10× slower and it trailed SWA).

### Correction: with Muon, tiny BPE is far ahead (EXP-025 bpe_muon 0.025 finished during the pause)
BPE + Muon at LR 0.025: 1.5257 / 1.5237 / 1.5259 (1.525 ± 0.001), against BPE + AdamW 1.634 and the best tiny H-Net
(warm start + Muon 0.025) 1.631. Every "the H-Net matches BPE at tiny scale" statement in this log compared against
AdamW BPE at an untuned LR; against the Muon-trained BPE the tiny H-Net trails by 0.106. EXP-027 rgate seed 2: 1.653
(arm mean 1.700 over three seeds, still a lottery). The evolve benchmark (L03, mix_v1) already uses Muon for both.

### Evolve g1: the match model beats BPE on seed 1, and the leak checks pass; generation 2 launched
g1-wild-01 (in-context match model: copy candidates from the most recent earlier occurrence of the last 4/8/16/32
bytes in the same document, entering as encoder/decoder embeddings and a gated logit bonus): **1.502** at step 908
(seed 1), against BPE 1.656 (-0.154) and its warm-start parent 1.827 (-0.325). Leak checks on the trained checkpoint
(`$S2/leak_check.py`, real validation text): changing every byte after position 1000 moves earlier logits by at most
7.6e-6; the same document packed after a copy of itself in a separate segment differs by at most 1.8e-5; on 12 English
FineWeb-Edu docs it scores 1.480 vs 1.657 for the parent; a document repeated inside one segment drops to 0.815 (parent
1.742). So the gain is causal, document-local, and present on English, not only code.
Other seed-1 finals: g1-mem-01 1.737, g1-mem-02 1.779, g1-opt-01 1.797, g1-arch-02 1.798, g1-arch-01 1.826,
g1-alloc-01 1.826, g1-opt-02 1.836, g1-router-02 1.852, g1-alloc-02 1.867 (controls hnet 1.825, hnet_ws 1.827).
Two GPU-only failures fixed (autocast-unsafe BCE in g1-router-01; resume crashed on a half-written checkpoint after a
preemption, killing bpe_s2). Generation-1 VM preempted twice in us-central1-a (the watcher's gcloud filter used a regex
alternation that matched nothing; fixed).
Generation 2 (branch evolve/g2 @ 19043df, us-central1-f): the decisive fairness control **g2-bpe-match** (BPE with the
same copy mechanism at token level: orders 1/2/4/8 tokens ≈ 4-33 bytes; +0.96% FLOPs, +0.69% params; offline the token
index correctly proposes 81-87% of the tokens the byte index copies whole), the match model combined with g1-arch-02 /
g1-mem-02 / g1-opt-01 and on the prior-free parent, the fixed g1-router-01, and mem-02 × arch-02; seeds 1-2.
