# H-Net implementation audit (vs goombalab/hnet and arXiv 2507.07955)

Scope: model (`hnet/model/**`), losses, optimizer grouping, trainer loss / bpb / tokens_eq, data and val sets,
FLOPs accounting and the BPE<->H-Net matching (tiny configs, ladder). References: the official repository
(github.com/goombalab/hnet, HEAD cloned 2026-10-29; read, not executed: it needs CUDA/Triton kernels) and the paper's
LaTeX source (`export.arxiv.org/e-print/2507.07955`; the arxiv.org HTML host is blocked from this session).
New tests: `tests/audit/` (CPU, ~6 s). Tests that demonstrate a bug are `xfail(strict=True)` so the full-suite launch
gate stays green; `pytest tests/audit --runxfail` shows them failing.

## TL;DR

**The paper path is a faithful port.** With the default `DCConfig` (all project extensions off), our model reproduces
a line-by-line transcription of the official dynamic-chunking code (router, chunk, EMA dechunk with packed-segment
resets, plug-back, fp32 residual, STE, per-stage ratio loss with alpha) in the forward pass **and in every
parameter gradient** (max abs gradient difference 6e-8 in fp32, 1- and 2-stage models, packed batch with a 1-byte
segment): `tests/audit/test_official_parity.py`. The eval protocol (same text for byte and BPE models, BPE bpb = token
NLL in bits / UTF-8 bytes of that text) is correct and verified on the real val shard.

Bugs, by severity:

* **Changes results:** none found.
* **May change results**
  * **B1. The "equal FLOPs" pairs are equal only at the target compression.** The fairness gate
    (`scripts/search/fairness.py`, `configs/MATCHING.md`) checks configured FLOPs at the target ratios. With the paper's
    threshold decision nothing pins the realized compression, and the committed EXP-002 metrics show the tiny H-Nets
    running at stage-0 compression 2.6-2.9 instead of 3, i.e. **+7% to +12% training FLOPs/byte over the BPE partner**
    (paper router +7.1 / +9.1%, end_cosine_shift +12.2%, paper_nolrmult +11.9%; second half of training). Known in the
    logbook (EXP-002/006/010 "realized compute") but not enforced anywhere. The bias favours the H-Nets, so "H-Net trails
    BPE" conclusions are, if anything, conservative; any "H-Net matches/beats BPE" claim from a threshold-router run
    is not iso-FLOP. The ladder is not affected (rank decider pins compression; pairs within 0.5%).
  * **B2 (low). Learned-router scores are bf16-quantized under autocast.** `end_linear` / `end_mlp`
    (`hnet/model/routing/end.py:63,90`) compute the logit with an autocast matmul, then `.float()`. On a trained 1-stage
    checkpoint (EXP-024 prior_s0) a 32768-byte batch has 32625 distinct fp32 scores but **782** distinct bf16 scores.
    The rank decider sorts these; ties at the k-th score are split arbitrarily by `topk` and land at p' = 0.5
    (at this checkpoint only 1 tie at the cut, because the threshold sits in the gap of a bimodal distribution, so the
    impact is small; earlier in training it can be larger). Fix: run the router head under `autocast(enabled=False)`
    in fp32 (it is d x d/4 x 1, negligible).
* **Cosmetic**
  * **B3. `python -m hnet.eval.run --dtype bf16` crashes on our H-Net checkpoints**: `load_run_checkpoint`
    (`hnet/eval/run.py:57`) casts the whole model to bf16 incl. `residual_proj`, which `HNet.forward`
    (`hnet/model/hnet.py:133-134`) feeds `h.float()` with autocast off -> `RuntimeError: mat1 and mat2 must have the same
    dtype`. Loud, eval-only. (`load_official_checkpoint` keeps residual_proj fp32 and is fine.)
  * B4. `HNetForCausalLM.set_progress` is defined twice (`hnet/model/lm.py:378-386`).
  * B5. `hnet/eval/bpb.py:27,35` masks `<|endoftext|>` targets with the hard-coded GPT-2 id 50256; for data mixes
    (own BPE, EOT id 0) the guard is a no-op. Harmless because `ValSet` already sets every doc-end target to IGNORE.
  * B6. `DCConfig.target_ratios` defaults to `[4.0, 2.0]` (the blog's) and `AttnConfig.rotary_emb_dim` to the full head
    dim; the paper / official use (3,3) or 6 and head_dim/2. Every committed config sets both explicitly
    (`from_official_json` sets the paper's), so only ad-hoc configs are affected.

Not a bug (the suspicion in the brief): **`residual_proj` has a live bias, exactly as official.** Official
`HNet.__init__` builds `nn.Linear(d, d, dtype=float32)` (default `bias=True`), zero-inits only the weight and flags it
`_no_reinit`; `_init_weights` never touches it; the released checkpoints contain `backbone.residual_proj.bias`
(`tests/test_model_official_keys_hnet_2stage_L.json`). Ours is identical (`hnet/model/hnet.py:105-111`). At init the
residual path is therefore a constant vector b ~ U(-1/sqrt d, 1/sqrt d) (|b| <= 0.10 at d0 = 96), not exactly 0; the
paper says "initialized close to 0", consistent.

## Checks

| # | Check | Verdict | Evidence |
|---|---|---|---|
| 1a | Router projections | correct | `routing/cosine.py:34-40`: q/k `Linear(d, d, bias=False)`, identity init, `_no_reinit` = official `RoutingModule` |
| 1b | Cosine and p_t | correct | `cosine.py:43-46`: p = clamp((1 - cos(W_q h_{t-1}, W_k h_t))/2, 0, 1). Paper writes cos(q_t, k_{t-1}); official (and we) apply q to t-1 and k to t: a relabelling of two identically initialized learned matrices, equivalent |
| 1c | p_0 = 1 at every segment start | correct | `routing/base.py:63-64` (position_ids == 0, all stages, incl. inner segments); official `boundary_prob[cu_seqlens[:-1]] = 1`; gradient there is 0 in both |
| 1d | Boundary threshold | deviation (follows official) | `base.py:66`: p > 1 - p (ties not selected) = official argmax; paper says b = 1{p >= 0.5}. Immaterial |
| 1e | End-semantics variants causal / consistent | correct (our extension) | `routing/end.py`, `end_cosine_shift`; plug-back `cumsum - 1` reads the latest chunk ended at or before t; existing `tests/test_model_hnet.py::test_causality` (all routers x semantics) and `tests/test_invariances.py` pass. `end_cosine_shift` decides "t ends a chunk" from cos(h_{t-1}, h_t): causal, but the break lands one byte after the dissimilarity the paper keys on (learnable, intentional) |
| 2a | Chunk selection | correct | `chunking/select.py:25-32`: `h[mask]`, `next_cu = pad(cumsum(mask)[cu[1:]-1])` = official `ChunkLayer` |
| 2b | EMA smoothing | correct | `chunking/dechunk.py:51-53`: P = clamp(p, 1e-4, 1-1e-4) at selected positions, zbar_j = P_j z_j + (1-P_j) zbar_{j-1}, reset at inner segment starts = official Mamba-kernel EMA (dt = log 1/(1-p), A = -1, B = p, C = 1, seq_idx). Ours in fp32, official in bf16 |
| 2c | Plug-back | correct | `dechunk.py:25` cumsum(mask) - 1 = official |
| 2d | STE / confidence | correct | `hnet.py:35-44,152-153`; c_t = p or 1-p (`base.py:67`) = official `selected_probs`; forward 1, gradient identity |
| 2e | Detached tensors | correct | nothing detached on the paper path; gradient parity in all params incl. q/k (through STE, EMA P and G) |
| 2f | Packed-batch behaviour | correct | parity test uses 4 segments incl. a 1-byte one; `test_packed_equals_separate` passes |
| 3 | Ratio loss | correct | `losses/ratio.py:37-44` N/(N-1)((N-1)FG + (1-F)(1-G)), F = mean(mask) (no grad), G = mean(p); per micro-batch ("local", official); alpha 0.03 x sum over stages (`lm.py:236-237`, paper L = L_AR + alpha sum_s L_ratio^s); divided by grad_accum in the trainer (`train.py:560`) |
| 4 | Residual path | correct | fp32 Linear, zero weight, live default bias (= official, see above); computed from the encoder output before routing; added after the STE-scaled dechunk, cast back to h's dtype (`hnet.py:133-134,170`) |
| 5a | Network norms | correct | RMSNorm at the end of every encoder / decoder / main (`isotropic.py:120-121`), pre-norm blocks with fp32 residual (`block.py`) = flash-attn prenorm semantics |
| 5b | pad_dimension | correct | `hnet.py:113-114,127-128,172`: zero-init shared vector appended, output truncated to D_in (paper SpaceByte rule; official) |
| 5c | LR multipliers | correct | `optim/groups.py:40-50` = paper App. C: lambda^s = sqrt(N_GPT D^S/D^s / prod_{i<s} N^i); stage map puts embeddings, lm_head, router, residual_proj of stage s in stage s (`lm.py:200-205`) as official. tiny 2-stage: [3.51, 1.76, 0.72] (N_GPT 4.62) |
| 5d | Init | correct | `lm.py:294-309`, `hnet.py:186-197`, `isotropic.py:144-150`: lm_head N(0, .02), embeddings N(0, 1), Linear N(0, .02), out_proj/fc2 scaled by 1/sqrt(accumulated height), q/k/residual weight kept = official `_init_weights` |
| 5e | Weight decay | deviation (intentional) | ours: no WD on biases, all norms, every 1-D param (A_log, D, dt_bias, pad_dimension). Official `group_params` excludes only `*.bias` and `.norm.` (so it decays block norm1/norm2/rmsnorm, A_log, D, dt_bias, pad_dimension). The released utility may not be what was used for the paper |
| 6 | Mamba-2 config | correct | `mixers/mamba2.py:86-149` = mamba_ssm defaults (headdim 64, ngroups 1, d_conv 4, expand 2, chunk 256, A ~ U[1,16], dt in [1e-3, 0.1], floor 1e-4, D = 1, gated RMSNorm, no biases but conv). `SSMConfig` defaults = official configs (d_state 128). Tiny configs scale d_state/headdim down (32/32) intentionally |
| 7 | Extensions off == paper | correct | `tests/audit/test_official_parity.py` (default DCConfig: threshold decision, no chunk vocab, no n-gram, no prior / consistency / gain / entropy / main_aux / chunk_pred, `chunk_ste` False, `residual_gate` none, no dropout/warmup, `attn_res` 0) |
| 8a | Byte CE | correct | `lm.py:153-174` sum CE / #valid targets; trainer loss = sum CE / #targets of the step + aux/accum (`train.py:544-561`) |
| 8b | Ignore index at segment ends / BOS | correct | `data/loader.py:101-110`: no target across a document (bytes: from EOS and into BOS; tokens: into EOT); first byte of a doc predicted from BOS, first token from EOT. Train-only asymmetry: the byte model trains on predicting EOS, the BPE model never predicts EOT (~1 target per doc; both excluded from val bpb) |
| 8c | BPE bpb | correct | `data/valsets.py` + `train.py:383-414`: bpb = sum token NLL / ln 2 / UTF-8 bytes of the scored tokens; `tests/audit/test_val_alignment.py` verifies on the real val shard that both kinds score the same bytes (n_bytes equal, byte histogram of scored targets equal, no BOS/EOS/EOT targets) |
| 8d | Val sets identical | correct | same docs, same token-boundary chunks for both kinds; minor asymmetry for docs longer than one window: a continuation chunk gets 1 element of context on each side (1 token ~ 4.6 bytes vs 1 byte) |
| 8e | tokens_eq | correct (negligible offsets) | H-Net: text bytes / measured bytes-per-text-token (4.62); BPE: input positions incl. EOT (+~0.02%); H-Net step count from positions incl. BOS/EOS (~0.05% fewer text bytes). Train-log BPE bpb divides by input-token bytes (log-only) |
| 9a | FLOPs accounting | correct | `flops.py`: H-Net E/D/router/residual_proj at stage-s positions, main at innermost positions, byte LM head counted, embeddings/lookups 0, dechunk EMA ignored (O(d)); BPE embedding 0, LM head counted; App. B Mamba terms (out_proj counted exactly 2 d_inner d) |
| 9b | Pairs at target | correct | tiny within 2%, ladder within 0.5% (`test_flops_matching.py`) |
| 9c | Pairs at realized compression | **BUG B1** | threshold-router tiny runs +7..+12% (`test_threshold_router_runs_stay_within_budget_at_realized_compression`, xfail); ladder pinned by the rank decider (`test_quantile_decider_pins_realized_compression`) |
| 10a | Same documents / order | correct | byte shards are the exact GPT-2 decode of the same token shards (`convert_report.json`: 0 marker collisions, decode exact), no shuffle (`shuffle_buffer: 0`); per step the tiny BPE reads 64 x 448 tokens = 132.4 KB vs H-Net 131.1 KB (1% drift, same 231 MB in total) |
| 10b | Train/val overlap | correct | FineWeb-Edu: train shard 1+ vs val shard 0; data mixes: hash split + val-hash dedup (`data/mix.py:600`) |
| 10c | Document boundaries | correct | segments at BOS / after EOS / at EOT and window starts; Mamba conv+SSM, attention, rotary, router and EMA all reset per segment |

## Details

### B1. Equal-FLOPs pairs hold only at the target compression

`FlopsEstimator` (`hnet/train/metrics.py:130-175`) logs realized FLOPs/byte from the batch compression, but the gate
(`scripts/search/fairness.py`, "GFLOPs/byte at target compression") and `configs/MATCHING.md` use targets. EXP-002
(committed `runs/gcp/exp002_tiny2m/*/metrics.jsonl`, mean over the second half of training, BPE = 1.189e7 FLOPs/byte):

| arm | stage-0 / stage-1 compression (target 3 / 3) | realized FLOPs/byte vs BPE |
|---|---|---|
| paper_s0 / s1 | 2.66 / 3.04, 2.82 / 2.88 | +9.1% / +7.1% |
| end_cosine_shift_s0 | 2.57 / 3.02 | +12.2% |
| end_mlp_s1 | 2.64 / 3.04 | +8.2% |
| paper_nolrmult_s0 | 2.59 / 3.00 | +11.9% |
| paper_r42_s0 (4 / 2) | 3.62 / 2.00 | +9.2% |

Repro: `pytest tests/audit/test_flops_matching.py --runxfail`. Fix options: gate on realized FLOPs (re-match main depth
at the measured compression, or report realized-FLOPs-adjusted bpb), or pin compression (rank decider / stronger
ratio loss) for every arm that enters an iso-FLOP comparison.

### B2. bf16 router scores

`tests/audit/test_dtype_paths.py::test_learned_router_scores_are_not_bf16_quantized_under_autocast` (CPU bf16 autocast:
logits are exactly bf16-representable). Measured on `runs/gcp/exp024_adoption_curriculum/prior_s0` (end_mlp + quantile,
1 MB val text, 16 x 2048 bytes, train-mode decision): 32625 distinct fp32 scores vs 782 under bf16 autocast; 1 tie at
the k-th score. The cosine router computes cos in fp32 from bf16 q/k, like official.

### B3. bf16 eval load

`tests/audit/test_dtype_paths.py::test_model_cast_to_bf16_still_runs`. Fix: in `load_run_checkpoint`, cast with
`model.to(dtype)` then `residual_proj.float()` for every stage (as `load_official_checkpoint` does), or rely on autocast.

## Intentional deviations that might matter

* Weight decay exclusions (5e): we never decay norms / 1-D params; the official utility decays most of them.
* Rank ("quantile") decider (ladder, later experiments): selection and normalization statistics are batch-coupled in
  training (a scalar-level causal coupling across the batch, tested in `tests/test_invariances.py`), and eval uses
  running statistics, so train-time and eval-time chunkings differ; the ratio loss is inert under it.
* LR multipliers ("auto", AdamW-derived heuristic of the paper) are also applied under Muon in the ladder: outer-stage
  parameters get 2.35-3.15x the main network's LR (Muon and the Adam part alike), e.g. L07 [2.88, 1.0]. Untested
  whether the heuristic transfers to Muon's orthogonalized updates.
* EMA dechunk and residual in fp32 (official: EMA in bf16 inside the Mamba kernel).
* "End" boundary semantics, pool chunker, chunk vocabulary, n-gram tables, router prior / gain / consistency: off in
  the paper path (verified), on in several experiments.
* `hnet.eval.run --suite bpb` builds the val set at 8192 bytes / 1792 tokens regardless of the training window
  (tiny models train at 2048 / 448); the trainer's own eval uses the configured lengths. Both kinds extrapolate
  equally, but numbers from the two paths are not comparable.

## Not verified

* GPU kernel paths (mamba_ssm fused scan, causal_conv1d, flash-attn, flex) against the torch paths: their tests are
  CUDA-only and skipped on CPU; the parity test runs the torch paths.
* End-to-end against the released `cartesia-ai/hnet_2stage_L` weights: `tests/test_model_official.py` needs the 3.5 GB
  checkpoint, not present locally (only the state-dict key list is checked).
* Paper hyperparameters the paper does not state (AdamW betas, weight decay, grad clip) and the official training
  loop (not released); `aux_loss_scope: global` under multi-GPU.
* The paper HTML (arxiv.org blocked); equations were read from the LaTeX source instead.

## Audit tests

| file | what | status |
|---|---|---|
| `tests/audit/test_official_parity.py` | default config == official DC forward/backward (1 & 2 stages); residual_proj bias live as official | pass |
| `tests/audit/test_flops_matching.py` | ladder within 1% at target, tiny within 2% at target, rank decider pins compression | pass |
| | threshold-router runs at realized compression (B1) | xfail (strict) |
| `tests/audit/test_val_alignment.py` | byte and token val sets score identical text, bpb denominators equal | pass (needs val shard) |
| `tests/audit/test_dtype_paths.py` | bf16 cast crash (B3), bf16 router scores (B2) | xfail (strict) |
