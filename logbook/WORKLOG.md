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
