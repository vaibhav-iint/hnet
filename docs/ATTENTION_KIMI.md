# Kimi attention variants, gated attention and Attention Residuals (WS-I2)

Code: `hnet/model/mixers/kimi.py` (mixers `kda`, `moba`, `gated_attn`), `hnet/model/isotropic.py` (optional
Attention Residuals, off by default). Tests: `tests/test_mixers_kimi.py`. Configs: `configs/models/tiny_bpe_2m_*`
and `configs/models/tiny_hnet2_2m_*` with suffixes `kda`, `kda_hybrid`, `moba`, `gated`, `gated_headwise`;
FLOPs table in `configs/MATCHING.md` ("Attention variants").

## How to use

Every mixer takes `(d_model, layer_idx=0, **attention kwargs)` with the reference `attn` keys (`num_heads,
num_kv_heads, rotary_emb_dim, rope_theta, window_size, nope_every, qkv_bias, out_bias, softmax_scale`) plus its
own keys, and reads document boundaries from `SeqCtx`: no state, conv window, block or attention edge crosses a
document.

```yaml
# dense baseline: every layer
model: {type: dense, ..., mixer: kda, mixer_kwargs: {expand_v: 1.25}}
# H-Net: swap only the main network's attention ("t") and configure the mixer
model:
  type: hnet
  mixer_codes: [{m: mamba2, t: attn}, {m: mamba2, t: attn}, {m: mamba2, t: moba}]
  mixer_cfg: {moba: {block_size: 32, top_k: 3}}
# Kimi Linear hybrid (3 KDA : 1 full attention, full attention without RoPE) as a layout pattern
  arch_layout: ["m4", ["T1m4", ["K3T1K2T1"], "m4T1"], "m4"]
  mixer_codes: [{m: mamba2, t: attn}, {m: mamba2, t: attn}, {m: mamba2, t: attn, k: kda}]
  mixer_cfg: {attn: [{}, {}, {rotary_emb_dim: 0}]}
```

FLOPs: each mixer has `flops_per_position(d_model, L, cv, **kwargs)` (forward FLOPs per position, `hnet.flops`
conventions; the caller applies the backward factor); parameters are counted from the class built on the meta
device.

## 1. `kda`: Kimi Delta Attention

**Source.** Kimi Team, "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv 2510.26692 (2025),
Sec. 3; parameterization and init follow fla `fla/layers/kda.py` (flash-linear-attention 0.6.0).

**What it is.** A gated delta rule (Gated DeltaNet) whose scalar per-head forget gate is replaced by a
channel-wise (per key dimension) gate: a linear-time recurrent mixer with a matrix state per head.

**Equations** (per head; K = V = head_dim by default):

* Eq. 1: `S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T`, `o_t = S_t^T q_t / sqrt(K)`.
* Eq. 9: `q, k, v = SiLU(ShortConv(W x))` (kernel 4, depthwise, no bias), `q, k <- L2Norm(q), L2Norm(k)`,
  `alpha_t = exp(g_t)`, `g_t = -exp(A_log) * softplus(W_f^up W_f^down x_t + dt_bias)` (low rank = head dim),
  `beta_t = sigmoid(W_b x_t)`.
* Eq. 10: `y_t = W_o (sigmoid(W_g^up W_g^down x_t) * RMSNorm_head(o_t))` (low-rank output gate with bias on the
  up-projection, as fla).
* Chunkwise training form (Eqs. 2-8, WY / UT transform), implemented in `kda_chunk_torch`: within a chunk with
  entry state `S_0` and in-chunk cumulative decay `Gamma`,
  `(I + A) U = Diag(beta)(V - (Gamma * K) S_0)` with `A_ri = beta_r k_r^T Diag(Gamma_r / Gamma_i) k_i` (i < r),
  solved with one batched triangular solve for all chunks; `O = scale ((Gamma * Q) S_0 + P U)` with
  `P_ri = q_r^T Diag(Gamma_r / Gamma_i) k_i` (i <= r); `S_C = Diag(Gamma_C) S_0 + ((Gamma_C / Gamma) * K)^T U`.
  Only the matrix-valued chunk-to-chunk transition is a Python loop (T / 64 iterations).
* Overflow-free decays: `Gamma_r / Gamma_i` is never formed as a ratio of two exponentials. Inside sub-chunks of 16
  the [16, 16, K] pairwise decay is explicit; across sub-chunks the decay factorizes through the end of the source
  sub-chunk, so every exponent is <= 0 (the secondary chunking of the fla kernels). Tests use per-step log-decays
  down to -80.
* Documents: the state is reset at every segment start by masking (same-segment pairs only in A and P; the entry
  state reaches only the segment that was running at the chunk start; only the last segment writes the exit
  state); the short convs reset too.

**Fast path.** CUDA + `fla` installed (and `HNET_FORCE_TORCH != 1`): `fla.ops.kda.chunk_kda(q, k, v, g, beta,
scale, cu_seqlens)` (q/k already normalized, g/beta precomputed). Only a missing `fla` package falls back; a broken
install raises. Parity test `test_kda_fla_parity` (skipped without CUDA + fla). `fla` is not installed on the dev
box and is not added to requirements.txt (Triton/CUDA only).

**Ignored attention keys** (documented in the class): `num_kv_heads` (no GQA), `rotary_emb_dim`, `rope_theta`,
`nope_every` (KDA has no RoPE: the decay and the short convs carry position), `window_size`, `qkv_bias`,
`out_bias`, `softmax_scale`. Own keys: `head_dim`, `expand_v`, `conv_size`, `conv_bias`, `gate_rank`,
`chunk_size` (64), `sub_chunk` (16), `allow_neg_eigval`, `norm_eps`. Output projection is named `out_proj` so
the depth-scaled init of the residual writers applies to it.

**Tiny-scale hyperparameters.** Dense `tiny_bpe_2m_kda`: 4 heads, K = 32, `expand_v` 1.25 (V = 40) to match FLOPs
(-0.6%; with V = 32 KDA is 2.6% cheaper than attention at 448 tokens). H-Net `tiny_hnet2_2m_kda`: main network
(d 256, 8 heads, K = V = 32) with defaults (+0.4%). Hybrid `tiny_hnet2_2m_kda_hybrid`: main `K3T1K2T1` (5 KDA +
2 full attention, as close to 3:1 as 7 layers allow, each group ending in full attention), main-stage attention
NoPE as in Kimi Linear (+0.3%).

**Deviations.** Kimi Linear's full-attention layers are MLA; the hybrid here uses the reference `attn` (GQA/MHA)
— swap `t` for WS-I1's `mla` to get the exact pairing. Initial state / caching for decoding are not implemented
(training and scoring only). FLOPs count the recurrence in its per-token form (`8 H K V` per position, like the
Mamba-2 SSD term), not the chunkwise algorithm's extra intra-chunk work.

**Expected relevance at small scale.** Kimi report KDA beating full attention at 1.4T tokens (compute multiplier
~1.16 cited by the Cartesia blog). At 448-token / ~227-position contexts the quadratic cost is small, so the
comparison is mostly about modelling quality per parameter; the delta rule's associative-recall ability and the
channel-wise decay should matter more for the H-Net main network (where each position is a ~9-byte chunk) than
the speed. Expect the pure-KDA models to be weaker on exact long-range copying and the hybrid to be the stronger
configuration, as in the paper.

## 2. `moba`: Mixture of Block Attention

**Source.** Lu et al. (Moonshot AI / Kimi), "MoBA: Mixture of Block Attention for Long-Context LLMs",
arXiv 2502.13189 (2025), Sec. 2.2.

**What it is.** Sparse softmax attention: KV is split into blocks; each query attends to its own block (causally)
plus the top-k-1 past blocks whose mean-pooled key has the largest inner product with the query. Parameter-free
gate; same projections as attention.

**Equations.** Eq. 2: `MoBA(q, K, V) = Softmax(q K[I]^T) V[I]`, `I = union of selected blocks` (Eq. 4).
Eq. 6: `s_i = <q, mean_pool(K[I_i])>`; Eq. 5: `g_i = 1` iff `s_i` is in the top-k. Current-block rule (Sec. 2.2):
the block containing the query is always selected (counted in k), causal mask inside it; future blocks are never
candidates. Blocks are aligned to each document's start (positions `[jB, (j+1)B)` of the document), scores use
the RoPE'd q and k, each query head selects its own blocks (GQA: kv heads shared), and the top-k gate passes no
gradient (as in the paper). `window_size >= 0` is honoured as an extra constraint (keys in `[t-w, t]`).

**Backends.** CUDA: `flex_attention` with a `mask_mod` reading the per-(head, query) block selection (block mask
built per layer call). CPU / fallback: SDPA with an explicit [H, T, T] mask (O(H T^2) memory). `backend=` forces
one; `test_moba_flex_equals_sdpa` checks they agree. No dependency on flash-attn (fla's `parallel_moba` needs
flash-attn and MHA).

**Tiny-scale hyperparameters.** Defaults `block_size` 32, `top_k` 3 (current + 2 past blocks: at most 96 keys).
Dense `tiny_bpe_2m_moba`: block 64, top-4 (at most 256 of 448 keys; sparsity active for t >= 256), -1.5% FLOPs
(block 32 / top-3 would be -4.2%, outside +-2%). H-Net main `tiny_hnet2_2m_moba`: block 32, top-3 (at most 96 of
~227 positions), -1.0%. Paper: block 512, top-3 at 8K (81% sparsity); 4096 / 12 at 32K+.

**Deviations.** The paper trains with a hybrid schedule (MoBA then full attention, or full attention in the last
layers); here every swapped layer is MoBA. FLOPs: exact average number of attended keys over a causal document of
L positions + gate scores (`2 q n_candidates`) + pooling.

**Expected relevance.** MoBA is a long-context efficiency method; at a few hundred positions its compute saving is
small and the sparsity mostly acts as a regularizer / restriction. With top-k covering all blocks it is exactly
causal attention (tested). Expect at best parity with attention at tiny scale; it becomes interesting for the
H-Net main network at long byte contexts (e.g. 8K+ bytes -> ~1K main positions).

## 3. `gated_attn`: gated attention (Qwen)

**Source.** Qiu et al. (Qwen), "Gated Attention for Large Language Models: Non-linearity, Sparsity, and
Attention-Sink-Free", arXiv 2505.06708 (2025).

**What it is.** The reference attention with a sigmoid gate on the SDPA output: Eq. 5, `Y' = Y * sigmoid(X W_theta)`,
at position G1 (after SDPA, before `W_O`), head-specific, multiplicative, computed from the block's normed input X.
The paper's best variant is elementwise (Table 1 row 5: PPL 6.026 -> 5.761 at 15B MoE); the headwise variant
(one scalar per head, row 10: 5.792) is nearly as good with H*d instead of d^2 parameters. `gate: elementwise |
headwise`; `gate_bias` (default False) exists so tests can force the gate open (then it equals `attn` exactly).

**Tiny-scale hyperparameters.** Dense `tiny_bpe_2m_gated`: elementwise (+1.8%); `tiny_bpe_2m_gated_headwise`
(+0.1%). H-Net `tiny_hnet2_2m_gated`: headwise (+0.1%) because the elementwise gate (a d x d projection per main
layer) is +2.5% there, outside the +-2% band; compare H-Net vs dense headwise for the like-for-like pairing.

**Deviations.** The paper fuses W_theta into the q projection; here it is a separate `gate_proj` (same function).
**Expected relevance.** Small but consistent gains are reported across scales (better stability, no attention
sink); one of the cheapest things to try, also inside the H-Net encoder/decoder attention.

## 4. Attention Residuals (Kimi): findings and implementation

**Primary source.** Kimi Team (G. Chen, Y. Zhang, J. Su, ...), "Attention Residuals", technical report,
arXiv 2603.15031 (16 Mar 2026); code/README: https://github.com/MoonshotAI/Attention-Residuals. The Cartesia blog's
"1.25x" is this paper's headline: Block AttnRes matches a baseline trained with 1.25x more compute (scaling-law fit,
Sec. 5.1; KDA's 1.16x is from Kimi Linear). Kimi K3 reportedly uses Block AttnRes (secondary coverage only).

**Method (well specified).** A residual-stream change, not a mixer. With PreNorm, `h_l = h_1 + sum_i f_i(h_i)`
(fixed unit weights). AttnRes replaces it with softmax attention over depth (Eqs. 1-4):
`h_l = sum_i alpha_{i->l} v_i`, `alpha_{i->l} = softmax_i(w_l^T RMSNorm(v_i))`, `v_0 = h_1` (embedding),
`v_i = f_i(h_i)`, with one learned pseudo-query `w_l in R^d` (zero-initialized, so uniform weights at init) and one
RMSNorm per layer, where every attention and every MLP is a "layer". Block AttnRes (Eqs. 5-6, Fig. 2): layers are
grouped into N blocks; sources are the embedding, the completed block sums `b_n = sum_{j in B_n} f_j(h_j)` and,
after a block's first layer, the block's partial sum. The output layer aggregates all block representations with
its own query. N ~ 8 blocks recovers most of the gain; ablations: input-dependent queries slightly better but
costlier, sigmoid instead of softmax worse, multi-head depth attention worse, RMSNorm on keys needed.

**Implemented** as an option of `Isotropic` (`hnet/model/isotropic.py`), **default off** (no new parameters, same
outputs, same code path as before):

* `Isotropic(..., attn_res: int = 0)`: 0 = standard residuals; `S > 0` = Block AttnRes with blocks of S
  sub-layers (mixer and MLP each count as one; S = 1 is Full AttnRes). Adds `attn_res` = ModuleList of
  `DepthAttention` (pseudo-query `query` [d], zero init, `_no_weight_decay`; RMSNorm `norm`) with one entry per
  sub-layer plus one for the output, i.e. `2 d (height + 1)` parameters per stack.
* Forward with AttnRes calls each `Block`'s `norm1`/`mixer`/`norm2`/`mlp` directly (`Block.forward` is not used;
  `block.py` is unchanged). The stack input plays the embedding `b_0`; sources are kept in fp32 like the residual
  stream; the final RMSNorm is applied to the output AttnRes.
* Tested against an independent transcription of Eqs. 2-6 (`test_attn_res_matches_paper_equations`, S = 1, 2, 3,
  100), plus per-document independence and gradients to every pseudo-query.

**Not plumbed** (files owned elsewhere), so no config can turn it on yet. Needed: a config field (e.g.
`HNetConfig.attn_res` per stage and `DenseConfig.extra["attn_res"]`) passed to `Isotropic` in `hnet/model/hnet.py`
(`iso()`) and `hnet/model/dense.py`; `hnet.flops` must add `2 d (height + 1)` params per stack (the depth attention
FLOPs are O(N d) per position, negligible, like norms). Activation-checkpointing wrappers that wrap
`Block.forward` would not apply on the AttnRes path. In an H-Net, each Isotropic (encoder, main, decoder) gets its
own depth attention whose `b_0` is that stack's input; the paper only covers flat Transformers, so whether the
dechunked/residual inputs of the decoder should be sources is open.

**Expected relevance.** The paper's gains come from deep models (dilution of each layer's contribution grows with
depth); our tiny stacks are 4-10 layers deep (8-20 sub-layers), so effects should be small but measurable; Full
AttnRes (S = 1) is affordable at our scale.

## Dense hybrid: not provided

`DenseConfig` builds `T{n_layers}` with one mixer, and `hnet.flops._dense` counts one layer x `n_layers`, so a
dense 3:1 KDA/attention hybrid cannot be expressed with exact param/FLOP accounting from the files this workstream
owns. Proposed change (owners of `hnet/config.py`, `hnet/model/dense.py`, `hnet/flops.py`): optional
`DenseConfig.layout` (e.g. `"K3T1K3T1K1T1"`) + `mixer_codes`, used by `LlamaLM` for `Isotropic` and by `_dense`
via `_iso`-style per-layer counting.
