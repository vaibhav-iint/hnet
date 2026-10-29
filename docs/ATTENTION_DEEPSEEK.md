# DeepSeek attention variants: MLA, NSA, DSA

Code: `hnet/model/mixers/deepseek.py` (registered mixers `mla`, `nsa`, `dsa`). Tests: `tests/test_mixers_deepseek.py`.
Matched configs: `configs/models/tiny_{bpe,hnet2}_2m_{mla,nsa,dsa}.yaml`. Matching table: `configs/MATCHING.md`
("Attention variants").

All three mixers are drop-in replacements for `attn`:

* **H-Net:** swap one stage's `t` code, e.g. only the main network:
  ```yaml
  mixer_codes: [{m: mamba2, t: attn}, {m: mamba2, t: attn}, {m: mamba2, t: mla}]
  mixer_cfg: {mla: [{}, {}, {kv_lora_rank: 200, q_lora_rank: 128}]}
  ```
* **Dense BPE baseline:** `mixer: mla` and `mixer_kwargs: {...}`.

Shared behaviour:

* The constructor takes `(d_model, layer_idx=0, **kw)` with the same attention keys as `attn` (`num_heads`,
  `num_kv_heads`, `rotary_emb_dim`, `rope_theta`, `window_size`, `nope_every`, `qkv_bias`, `out_bias`,
  `softmax_scale`) plus the variant's own keys. Unknown keys raise.
* `config_kwargs(cfg, stage)` starts from `cfg.attn_for_stage(stage)`; `cfg.mixer_cfg[name]` is merged on top.
* `flops_per_position(d_model, L, cv, **kw)` returns **forward** FLOPs per position, following `hnet.flops`
  conventions:
  * matmuls cost 2·in·out;
  * the quadratic terms are logits 2·ctx·H·d_qk, probs·V 2·ctx·H·d_v and softmax 3·H·ctx;
  * ctx is the exact causal average (or the last position under the `"full"` convention);
  * windows are honoured, and `attn_quadratic_scale` is applied;
  * norms, RoPE and top-k/sort are not counted.

  `model_flops` multiplies the result by 1 + backward_factor, exactly as it does for `attn_flops`.
* `forward(x[1,T,D], ctx)`: document masking from `ctx` means no attention across segments. RoPE positions and every
  block structure (compression and selection blocks) restart at each segment.
* Pure-PyTorch paths run on CPU. Fast paths are chosen automatically:
  * `mla` and the NSA window branch use `attn.attention` (flash / flex / SDPA);
  * the NSA selection branch and DSA top-k attention use `flex_attention` with a data-dependent `mask_mod` on CUDA,
    and a gather-based reference elsewhere (`HNET_SPARSE_BACKEND` overrides the choice).
* Optional imports fall back only when the package is missing.

## 1. MLA: Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434; DeepSeek-V3, arXiv 2412.19437)

**What it is.** Keys and values are produced from a shared low-rank latent c^KV per token, and only that latent is
cached at inference. RoPE cannot be applied to keys reconstructed from the latent, because it would block weight
absorption. MLA therefore adds a small decoupled RoPE query/key pair, with the RoPE key shared by all heads.

**Equations** (V2 Eqs. 9–19 / App. C Eqs. 37–47; V3 Eqs. 1–11 add RMSNorm on both latents):

```
c^Q_t  = RMSNorm(W^DQ h_t)                q^C_t = W^UQ c^Q_t       q^R_t = RoPE(W^QR c^Q_t)
c^KV_t = RMSNorm(W^DKV h_t)               k^C_t = W^UK c^KV_t      v^C_t = W^UV c^KV_t
k^R_t  = RoPE(W^KR h_t)                   (one key, shared by all heads)
q_t,i = [q^C_t,i ; q^R_t,i]   k_t,i = [k^C_t,i ; k^R_t]
o_t,i = Σ_{s≤t} softmax_s(q_t,i·k_s,i / sqrt(d_nope + d_rope)) v^C_s,i        u_t = W^O [o_t,1 ; … ; o_t,H]
```

`q_lora_rank = 0` means a full-rank query, `q_t = W^Q h_t` (DeepSeek-V2-Lite).

We implement the training-time, **non-absorbed** form: per-head K/V are materialized and fed to the standard
attention kernels. The absorbed inference form is documented in the docstring only: W^UK folds into the query and
W^UV into W^O, so all heads attend over the shared cache [c^KV; k^R] (MQA over a latent).

**Tiny-scale hyperparameters** (`_mla` configs):

* Head dims: qk = 16 nope + 16 rope, v = 32. This is the base's qk head dim; the H-Net base also uses rotary 16 of 32.
* `q_lora_rank` = d/2 (64 dense, 128 H-Net main). This costs the same as a full-rank query.
* `kv_lora_rank` = 96 (dense, d = 128) and 200 (H-Net main, d = 256). This is the FLOPs-matching knob: mixer FLOPs
  are within +0.35% of `attn`.

Module defaults without a config (paper ratios): d_rope = d_h/2, d_nope = d_v = d_h, kv_lora_rank = 4·d_h.

**Deviations.**
* `num_kv_heads` is accepted and ignored. MLA has no KV-head grouping.
* NoPE layers keep the rope channels but do not rotate them, so the parameter count does not depend on depth.
* No YaRN mscale. RoPE uses our non-interleaved layout; the open-source MLA code is interleaved, which is equivalent
  up to a channel permutation.
* RMSNorm eps is 1e-6.

**Relevance at small scale.** MLA's benefit is inference memory: the KV cache shrinks from 2·H·d_h to
d_c + d_rope per token. At 2M params with 4–8 heads the cache is tiny and training FLOPs are matched, so any
difference is a quality question: does the low-rank KV bottleneck plus decoupled RoPE help or hurt? Our
kv_lora_rank is 0.75·d. DeepSeek's d_c ≈ 0.07·d is not meaningful at d = 128.

## 2. NSA: Native Sparse Attention (Yuan et al., arXiv 2502.11089)

**What it is.** Each query combines three attention branches with learned sigmoid gates:
* compressed block summaries (coarse, global);
* the full tokens of a few top-scoring blocks (fine, selected);
* a sliding window (local).

Selection reuses the compressed branch's probabilities, so it costs almost nothing extra and is shared across a
GQA group.

**Equations** (Sec. 3.2–3.3):

```
o*_t = Σ_{c∈{cmp,slc,win}} g^c_t · Attn(q_t, K^c_t, V^c_t),   g^c_t = sigmoid(W_g h_t)  (per head)          (5)
K^cmp_t = { φ(k_{id+1 : id+l}) | 0 ≤ i ≤ ⌊(t−l)/d⌋ }   (block complete at or before t)                    (7)
p^cmp_t = softmax(q_t^T K^cmp_t)                                                                            (8)
p^slc_t[j] = Σ_i p^cmp_t[i] · |[id, id+l) ∩ [jl', (j+1)l')| / d      (overlap form of Eq. 9, see below)      (9)
p^slc'_t = Σ_{h in group} p^slc,(h)_t                                                                        (10)
I_t = top-n blocks of p^slc'_t (+ forced initial / local blocks);  K^slc_t = Cat{ k_{il'+1:(i+1)l'} | i∈I_t } (11–12)
K^win_t = k_{t−w : t}
```

φ is "a learnable MLP with intra-block position encoding". Ours is
`φ(B) = mean(B) + W2·SiLU(W1·vec(B + P))`, with learned P ∈ R^{l×d_h}, separate for K and V, shared by the KV heads.
K/V are independent per branch (`separate_kv`, as the paper).

**Causality.** At every level, a query sees:
* compressed blocks that end at or before t (queries with t < l − 1 see none, and that branch contributes 0);
* selected blocks with start ≤ t, with token-level s ≤ t inside them. The current, partially filled block is
  always selected (local);
* window keys s ∈ [t − w, t].

All block indices are segment-local.

**Tiny-scale hyperparameters.** The module defaults match the configs. The paper uses l=32, d=16, l'=64, n=16,
w=512 at 8k–64k.
* Compression: block l = 8, stride d = 4.
* Selection: block l' = 8, n = 8 blocks (1 initial + 2 local + 5 ranked = 64 keys). Selection is strictly sparse
  for t ≥ 64, i.e. most queries at L = 228 (H-Net main) and L = 448 (BPE).
* Window: w = 32.
* GQA: 2 KV heads (G = 2 dense, 4 H-Net), with selection shared per group.
* Compression-MLP hidden width: 12 (dense) and 96 (H-Net). This is the FLOPs-matching knob: mixer FLOPs within ±0.35%.

**Deviations.**
* Eq. (9), read literally, sums p^cmp at indices l'/d·j − m − n. Those are compression blocks that mostly *precede*
  selection block j, which looks like an indexing-convention artefact. We use the spatial-overlap weighting that the
  text describes; it has the same multiplicities when d | l and d | l'.
* Top-n ties go to the more recent block, which keeps the result independent of packing.
* RoPE is applied to keys before compression.
* The shared `window_size` key, when ≥ 0, additionally limits every branch to keys in [t − window_size, t].
* No Triton kernel. The reference path gathers the selected keys, using O(T·n·l'·Hk·d) memory; the CUDA path uses
  flex_attention with a block-selection `mask_mod`.
* The compression branch is computed densely over a padded per-segment layout, because it needs the probabilities.

**Relevance at small scale.** NSA is designed for 8k–64k contexts. At our lengths the three branches together still
attend to 64% (BPE, 448 tokens) or 97% (H-Net main, 228 positions) of the keys of dense causal attention. The
comparison therefore tests NSA as an *architecture*: gated multi-resolution attention with GQA, not as a speed-up.
Gains are plausible from the compression branch, which works like a learned pooling summary of the past.
Selection mostly overlaps the window at these lengths.

## 3. DSA: DeepSeek Sparse Attention (DeepSeek-V3.2-Exp tech report; github.com/deepseek-ai/DeepSeek-V3.2-Exp)

**What it is.** A lightweight "lightning indexer" scores every past token with a few ReLU heads. Each query then
attends only to its top-k tokens, shared by all heads (V3.2 instantiates it on MLA in MQA mode). The indexer is
trained to imitate the main attention distribution with a KL loss.

**Equations.**

```
I_t,s = Σ_{j=1}^{H^I} w^I_t,j · ReLU(q^I_t,j · k^I_s)                                                      (1)
u_t   = Attn(h_t, { c_s | I_t,s ∈ Top-k(I_t,:) })                                                          (2)
L^I   = Σ_t KL( p_t,: || softmax(I_t,:) )                    dense warm-up stage                           (3)
L^I   = Σ_t KL( p_t,S_t || softmax(I_t,S_t) )                sparse stage, S_t = selected set               (4)
```

* p_t is the main attention probabilities summed over heads and L1-normalized.
* Following the reference `Indexer`:
  * q^I = W^Q_I c^Q (the MLA query latent);
  * k^I = LayerNorm(W^K_I h);
  * w^I = W^w h · (H^I)^{-1/2} · (d^I)^{-1/2};
  * RoPE is applied to the first `index_rope_dim` channels of q^I and k^I.
* The indexer input is **detached**: the indexer learns only from L^I, and the main model only from the LM loss.

**Training signal, wired without a trainer change.** Each training forward (grad enabled, `index_loss != "off"`)
computes L^I as the per-token mean of Eq. 4, times `index_loss_coef`.
* `index_loss: inject` (default): an identity autograd function on the mixer output adds d(coef·L^I)/dθ to the
  backward pass. The indexer is therefore trained whenever the LM loss is back-propagated, even though the trainer
  only adds `out.aux_loss`. Because the indexer input is detached, no other parameter is affected; a test checks that
  main-model gradients equal those of `index_loss: off`. The KL value is stored in `mixer.index_kl` for
  logging, and it also works under the trainer's non-reentrant activation checkpointing (tested).
  * Caveat: the injected gradient is not divided by `grad_accum` (nor multiplied by an fp16 loss scale; bf16 needs
    none). With Adam this only changes the indexer's share of the global grad norm used for clipping. Set
    `index_loss_coef = 1/grad_accum` to match exactly.
* `index_loss: attr`: nothing is injected. The caller must add `hnet.model.mixers.deepseek.mixer_aux_loss(model)` to
  the loss it back-propagates, scaled like `out.aux_loss` (divided by grad_accum).
  * Proper trainer wiring would be a 3-line change in `lm.py`/`dense.py` (WS-A2): `out.aux_loss += mixer_aux_loss(self)`
    and `out.stats["index_kl"] = mean(index_kl)`. We did not edit those files.

**Tiny-scale hyperparameters.**
* `base: mla`, with exactly the `_mla` configs' attention (kv_lora 96 / 200, q_lora 64 / 128), so that `dsa` vs `mla`
  isolates sparsity.
* Indexer: H^I = 4 heads, d^I = 32 (dense) and 24 (H-Net). d^I is the FLOPs-matching knob: mixer FLOPs within
  ±0.4%. `index_rope_dim` = d^I / 2.
* `index_topk` = 64: density about 14% (dense) and 28% (H-Net) at the end of the context. V3.2 uses k = 2048 at 128K,
  with H^I = 64, d^I = 128.

**Deviations.**
* We train from scratch with the sparse-stage objective (Eq. 4) from step 0. There is no dense warm-up (Eq. 3),
  because there is no dense checkpoint to start from. Early positions (t < k) are dense anyway.
* L^I is averaged over tokens (paper: summed).
* No FP8 and no Hadamard rotation (they serve the FP8 indexer).
* Top-k ties go to the most recent key.
* `base: attn` (plain GQA attention) is also available.
* The shared `window_size`, when ≥ 0, restricts candidates to the window.
* The indexer is scored in query chunks, with keys limited to the last `max_seqlen` positions.

**Relevance at small scale.** The indexer is O(L²) like attention but cheap per key, about (2·d^I + 3)·H^I FLOPs.
Top-k saves (1 − k/ctx) of the attention quadratic term. At L ≈ 228–448 the two roughly cancel, so the configs are
FLOPs-matched with near-identical MLA ranks. The experiment asks whether learned token-level sparsity costs quality
at tiny scale. For the H-Net it also asks whether the main network, which already operates on compressed chunks,
can afford to drop most past chunks.

## Tests (`tests/test_mixers_deepseek.py`, CPU, ~5–10 s total)

* Causality and packed == separate documents, for every variant (MLA ± q-LoRA ± window; NSA GQA, and MHA with shared
  K/V and a window; DSA on MLA and on GQA attention).
* Gradient flow to all parameters, including the indexer through the injected KL.
* MLA vs its written-out equations.
* NSA vs a per-query loop implementation of Eqs. 5–12, for both backends.
* NSA reduces to dense causal attention when every block is selected and only the selection gate is open, and to
  `attn` with a sliding window when only the window gate is open.
* DSA vs a loop top-k reference, for both backends. With top-k ≥ context, DSA equals its dense base (MLA / attn).
* DSA indexer loss:
  * `inject` gives the same gradients as an explicit `attr` loss;
  * main-model gradients are unchanged;
  * KL = 0 for top-1;
  * no aux loss in eval mode or under no_grad;
  * gradients are unchanged under activation checkpointing.
* Flex vs gather on CPU (forward; flex has no CPU backward).
* CUDA parity tests (bf16, flash/flex vs SDPA/gather, with gradients), skipped without a GPU.
* FLOPs: closed forms; meta-device params == `mixer_counts`; `model_flops` training multiplier.
* Tiny H-Net and dense models with each mixer: forward/backward, `count_params` == built params, packed == separate.
* Every variant config differs from its base only in mixer fields, and is within 2% of `tiny_bpe_2m` and 1% of its
  base.
