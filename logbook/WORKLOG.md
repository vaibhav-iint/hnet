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
- Tested integration points get a normal commit message and a `milestone-*` tag.

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
