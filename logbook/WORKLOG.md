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
