# Reproducing "From Tokens to Bytes" I & II (Cartesia, Oct 28 2026): claims, reachability, plan

Sources: https://www.cartesia.ai/blog/from-tokens-to-bytes-part-1 and `-part-2` (read in full; local text copies in the
session scratchpad). The 2025 paper's H-Net ("LH-Net": lookahead/start-of-chunk boundaries) is explicitly superseded
by an unreleased "much improved" internal H-Net; the blog does not publish its architecture details beyond the items
below.

## What the blog states about its H-Net (usable as constraints)

- Boundaries are *not* lookahead: the LH-Net "draws chunk boundaries only after it sees the start of the next chunk",
  which "negatively impacts next-byte prediction and multiple hierarchies". Our end-of-chunk routers match this.
- 1-stage ladder: compression is matched to BPE (4.57 bytes/position) "by using a strong auxiliary loss" (we pin it
  exactly with rank decisions). The inner network is the BPE rung's Transformer, unchanged; the outer encoder/decoder
  are FLOP- *and parameter*-matched to the embedding + LM head; BPE hyperparameters (batch, LR) are transferred.
- 2-stage model: stage 0 = 4 Mamba-2 encoder + 4 decoder layers; stage 1 = one sliding-window attention block + 4
  Mamba-2 each side; main = 40-layer global-attention Transformer; target compressions 4x and 2x (trained ~7.6x);
  outer stages 43% of FLOPs; FLOPs/byte matched to SmolLM3 within 2%.
- Footnote: gated attention, MoE, hybrid SSM / linear attention are orthogonal and "synergize" with H-Nets.

## Claims

### Part I
| # | claim | numbers |
|---|---|---|
| I.1 | 1-stage H-Net beats the matched BPE rung at every size | all ladder rungs |
| I.2 | compute multiplier, validation bpb | **1.28x** (1e18–1e21 FLOPs) |
| I.3 | multiplier on downstream bpb suites | Paloma 1.83x, OLMES **2.06x** |
| I.4 | code, per language, in distribution | 1.49x (Markdown) … 2.40x (Go), median **2.04x** |
| I.5 | code, 10 languages not in the mix | 2.19x (Haskell) … 5.37x (Prolog), median **3.10x** |
| I.6 | H-Net compresses almost every code language *more* than BPE (spends less compute on code) | Fig. 6 |
| I.7 | in-corpus BPE is 1.1x better than Llama-3 BPE | |
| I.8 | 2-stage H-Net reaches SmolLM3-3B's final 6-task average with ~1T vs 11T tokens (>10x) | 3.67B params, 4.44 GFLOPs/byte |
| I.9 | qualitative: multi-word chunks, context-dependent (not pattern-based) boundaries, CJK correct, code chunks spanning newlines/indentation, **in-context tokenization** (recurring new words get merged), stage-0 chunks = morphemes (`track ers`) | |
| I.10 | DNA: 6–7x compute multiplier; codon-aligned chunks emerge late in training | |

Ladder recipe: depth L defines the model, D = multiple of 128 with D/L ≈ 75, heads = L, head dim 128, SwiGLU 3.6D,
MHA; Muon + WSD (fixed-fraction warmup/decay); batch/LR scaling laws fitted on selected rungs; iso-FLOP fits; causal
attention FLOPs only; LM head counted. Rungs L08 (221M total / 56M backbone) … L28 (2.62B). Data 70% English
(FineWeb-Edu) / 20% code (Stack v2, 15 languages) / 5% math (FineMath, InfiWebMath) / 5% multilingual (FineWeb2-HQ),
by bytes; documents split identically for bytes and BPE; in-corpus 128k BPE (4.57 bytes/token).

### Part II (loss-to-loss: validation bpb → OLMES bpb, per architecture)
| method | compute→val | val→OLMES | compute→OLMES (multiplier H-Net has over it) |
|---|---|---|---|
| H-Net | 1x | 1x | 1x |
| BLT-like (entropy-proxy boundaries, same arch) | 1.40x | 1.32x | 1.84x |
| BPE (with pretokenization) | 1.28x | 1.61x | 2.06x |
| BPE-NP (no pretokenization, 6.03 bytes/token) | 1.89x | 2.02x | 3.82x |
| Byte Transformer (isotropic) | 9.19x | 1.05x | 9.60x |

- II.1 Loss-to-loss laws depend strongly on the input encoding (contradicting "LLMs on the Line"); all runs of a
  ladder, on- or off-frontier, fall on one curve (~1% RMSE).
- II.2 Byte-level models transfer best; BPE-NP worst; pretokenization buys 1.5x compute→val and 1.26–1.3x val→OLMES.
- II.3 H-Net = BPE-level compute efficiency **and** byte-level transfer.
- II.4 End-to-end matters (Table 2, same architecture and compression): at 1e19/1e19.5/1e20 FLOPs, entropy-proxy
  boundaries 0.8160/0.7623/0.7138, frozen boundaries from a smaller H-Net 0.8139/0.7504/0.7057, end-to-end
  **0.8001/0.7488/0.7027**.
- II.5 Prompt-boundary problem: BPE-NP's OLMES bpb is unusable without marginalization correction.

## Where we are (tiny scale, ~3e15 FLOPs per run, FineWeb-Edu only)
1-stage ratio-6 H-Net with rank decisions + chunk vocabulary + next-chunk objective + byte n-gram tables + a 10%
whitespace warm start = BPE (1.635 vs 1.634 bpb) at equal FLOPs — but with **1.8x BPE's parameters** (27M vs 15M)
and a warm start that is a pretokenization-like prior (exactly what Part II calls a hand-engineered hack). Without the
warm start: 1.682 (adoption lottery, 1 of 3 seeds fails). The blog's ladder is 300–3e5x more compute per run.

## Reachability

| claim | reachable? | how |
|---|---|---|
| I.1, I.2 (val multiplier) | **yes, at 1e16–1e18** | mini iso-FLOP ladder L04–L12, blog recipe, BPE vs 1-stage H-Net |
| I.3 (downstream multiplier) | partly | bpb of correct answers on our paper7 suite (+ a few Paloma-like domains); OLMES itself is large |
| I.4–I.6 (code, OOD code, compression per source) | **yes** | 70/20/5/5 mix; code from `codeparrot/github-code-clean` (32 languages, HF-hosted) — hold ~8 out as OOD (Haskell, Fortran, Lua, Julia, Perl, Scala, …) |
| I.7 | yes, cheap | train in-corpus BPE; compare with GPT-2/Llama-3 tokenizers on one rung |
| I.8 (SmolLM3, 10x) | **no** (3B params, 1T tokens) | — |
| I.9 qualitative | **yes** | probe texts + chunk_stats on ladder checkpoints (in-context tokenization, code chunks) |
| I.10 DNA | possible later, cheap | not today |
| II.1–II.3 loss-to-loss | **yes, reduced** | 3–4 ladders (BPE, H-Net, Byte Transformer, BPE-NP) at 1e16–1e17; all runs of the iso-FLOP sweeps give the loss-to-loss scatter |
| II.4 end-to-end vs proxies | **yes, cheap** | existing `external` router: entropy boundaries from a small byte LM; frozen boundaries from a smaller H-Net; same arch/compression |
| II.5 prompt boundary | yes, cheap | BPE-NP tokenizer + marginalized scoring on the downstream suite |

## Scaling laws we can show
1. Compute→validation bpb (iso-FLOP frontier, Chinchilla-style fit L(C) = E + A·C^-α) for BPE and H-Net; compute
   multiplier = C_BPE(L)/C_HNet(L) at matched loss.
2. The same per domain (English / code / math / multilingual / per code language / held-out code languages).
3. Compression per source (bytes per main-network position vs bytes per BPE token).
4. Loss-to-loss: validation bpb → downstream bpb, across all runs of each ladder.
5. Optimal model size and tokens vs compute for each architecture (N_opt(C), D_opt(C)).

## Compute budget estimate
Current throughput is 1.2% MFU (tiny models, 3 per GPU, no compile): unusable at 1e17+. Expected at d ≥ 512 with
compile ≈ 25–35% MFU (L4: ~35 TFLOP/s; H100: ~350 TFLOP/s). A 3-budget iso-FLOP ladder (1e16, 1e17, 1e18, 4 sizes,
2 architectures) ≈ 1e19 FLOPs ≈ 80 L4-hours (~$25 spot / $70 on-demand) or ≈ 8 H100-hours (~$20). Adding a byte
Transformer and BPE-NP ladder at 1e16–1e17 roughly +30%. Data: ~30 GB bytes for the mix (download/prep on CPU).
