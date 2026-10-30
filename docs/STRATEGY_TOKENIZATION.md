# Fixing tokenization for real: what the H-Net author's video hints at, and what to try next

Written 2026-10-30, at the pause after the evolve loop (docs/EVOLVE.md, docs/SCALING_CRITERIA.md). Nothing in the
"ideas" section has been run; costs are estimates on 4xA100 spot at ~$5.9/h.

## 1. The video (Albert Gu / Cartesia, Chinese H-Net chunking)

A Chinese translation of the H-Net abstract, chunked by a trained H-Net. Each coloured block is one chunk. The red
boxes mark **split characters**: Chinese characters (3 bytes in UTF-8) with a chunk boundary inside them. The slider is
training data seen. Frames: `docs/figures/hnet_cjk_video_68B.png` (11 split characters) and
`docs/figures/hnet_cjk_video_500B.png` (0).

| training bytes | 56B | 62B | 68B | 75B | 127B | 218B | 343B | 459B | 500B |
|---|---|---|---|---|---|---|---|---|---|
| split characters | 2 | 7 | 11 | 2 | 2 | 2 | 2 | 2 | **0** |

At 500B bytes the chunks are characters and short words, and the embedded Latin words ("Tokenization", "Transformer",
"H-Net") are each one chunk. The counter, not the colours, is the message.

## 2. What he is hinting at

1. **The model discovers the atomic units of the data by itself.** Nobody told it what UTF-8 is. By 500B bytes it never
   cuts inside a character. A BPE vocabulary has to split its 32k slots between scripts (CJK gets the worst of that
   deal); a learned segmentation has no such budget to split, so mixed-script text costs nothing extra.
2. **Learned tokenization keeps improving with data; a fixed tokenizer never does.** The counter reaches zero only at
   the end of training. This is the bitter-lesson argument in one number: whatever structure a human engineers into a
   tokenizer, the model finds given enough data, and finds a better version.
3. **The segmentation is a slow, non-monotonic learner.** Split characters go 2 -> 11 -> 2 between 56B and 75B bytes
   before settling. The chunker is still reorganising tens of billions of bytes into training. Judging a byte-level
   model at an early checkpoint measures a tokenizer that is still being learned, not the architecture's ceiling.
4. **Show it where tokenizers are worst.** He chose Chinese, not English. The paper's largest margins were Chinese,
   code and DNA.

Caution: it is one passage, and we do not know whether the BPE it was compared against was trained on Chinese. Our
in-corpus-BPE fairness rule stands.

## 3. What this says about our results

* **We were far inside the unstable phase.** Their chunker was still unstable at 56-75B bytes. Our small benchmark saw
  0.5B bytes (L03 1e16), the bigger one 2.1B (L05 1e17): 30-150x less than where their segmentation settled. Our
  "word basin vs suffix basin" bifurcations (logbook/interp/REPORT.md) are plausibly that phase, caught at a scale
  where it never resolves.
* **Our own data-scaling evidence points the same way.** At a fixed 38M model (L05), 10x the data (1e16 -> 1e17 FLOPs):

  | | 1e16 | 1e17 |
  |---|---|---|
  | H-Net + space hint, gap to BPE (bpb) | +0.183 | +0.129 |
  | H-Net, no hint, gap to BPE | +0.229 | +0.136 |
  | cost of having no hint | 0.046 | **0.007** |

  With 10x the data, the model given no hint about spaces caught up with the one that had the hint. That is the
  video's message at a tiny scale. (Numbers: logbook/tldr/data.json, `ladder`.)
* **The evolve loop went the other way.** We added hand-built lookups (copying, tries, suffix memories) that help a
  small model early and fade with scale (docs/SCALING_CRITERIA.md). The video says the winning move is the opposite:
  remove priors, add data, and make the learned segmentation converge sooner.

## 4. Ideas to try, in order of cost

1. **Measure segmentation convergence, not just loss. Free (CPU, existing checkpoints).**
   Add the video's metric to evaluation: the fraction of chunk boundaries that fall inside a multi-byte UTF-8 sequence
   (`split_char_frac`), plus boundary alignment with words for English (we log `bnd_on_ws/s0`) and with a reference
   word segmenter for Chinese (measurement only, never a training signal). Plot against training bytes for every
   checkpoint we have (`$S2/ck/`, the bucket mirrors). Add `split_char_frac` to `logbook/training_report.py` as a
   mechanism panel and an alert (CLAUDE.md rule 6). If our chunkers never stabilised, that explains the deficit better
   than anything the loop tried, and it becomes the quantity to optimise.
2. **Benchmark where tokenizers fail. ~$10.** Rerun the L03 1e16 / L05 1e16 pair on a Chinese-heavy (or code-heavy)
   mix, with a 32k BPE trained on that same corpus (`scripts/data/` has the in-corpus tokenizer path for mix_v1).
   Our mix_v1 is 70% English, the one place a 32k BPE is near-optimal; multilingual was 5%. This is the cheapest test
   of whether the published H-Net (no lookups, no hint) wins anywhere at our scale. Also report per-source bpb
   (`scripts/eval/per_source.py`).
3. **Scale data, not model. ~$10.** Train L03 (20M) to 1e17 FLOPs (10x the data; H-Net ~3.5 h, BPE ~2 h on one A100).
   Compare the compute multiplier at fixed size across 1e16 and 1e17. Our transfer check (g4x) scaled model and data
   together and confused the two; the L05 table above suggests data alone helps the prior-free H-Net most.
4. **Make the learned segmentation converge sooner, with no priors.** The scalable, modality-agnostic research question
   hiding in the video. Our self-taught chunking (g3-route-02: a causal next-byte probe whose entropy leads the rank
   decision early, plus a probe-to-router teacher, both decaying to zero) is one such method: every seed reached the
   word basin at 0.5B bytes, where the published design needed a space hint. Pursue that line: self-supervised router
   objectives, lower-variance boundary estimation, router LR transfer (g1-opt-01). Judge candidates by
   data-to-convergence of the segmentation (idea 1's metric), with asymptotic loss as the tie-breaker. This replaces
   "fitness = bpb at 908 steps" as the evolve objective.
5. **The image analogue of the video. ~$5.** Feed raw RGB bytes of small images (e.g. 32x32 CIFAR, 3072 bytes each,
   concatenated as a byte stream) to the unchanged model and ask the video's question: do chunk boundaries land on
   pixel boundaries (multiples of 3 bytes) and on row boundaries (multiples of 96), without being told the image
   width? Baseline: a patch Transformer at equal FLOPs. This is "discovering UTF-8" for pixels: no new architecture,
   and the first multimodal evidence we could show. Extend to audio bytes (16-bit samples: boundaries on even bytes).
6. **Two-stage H-Net. ~$10 for a probe.** The paper's strongest results used two levels (bytes -> words -> phrases); we
   only ran one stage. Two levels is the design that maps to video (frames -> shots -> scenes). Likely worse at our
   scale, but it is the version that could scale. The stage-1 cosine-router failure (CLAUDE.md miss table) is fixed
   by the rank decider; run a step-0 audit and the invariance suite first (rules 1-2, 10).
7. **SmolLM2 reference line. Free (CPU).** Score SmolLM2-135M / 360M on the mix_v1 validation set in bits per byte
   (`hnet/eval/scorers.py` handles pretrained models) and add it to the TL;DR page, so readers see how far
   small-scale research sits from a trained model. Not a comparison of tokenizers, a scale reference.

## 5. Strategy: what "fixing tokenization" should mean in 2026

If the bitter lesson wins, the only lasting contributions are things that remove human design or make end-to-end
learning converge faster. By that test:

* **Dead ends:** vocabulary tricks, copy indexes, memorised tables, hand-written boundary hints. Each helps a small
  model and is absorbed by a large one. The evolve loop spent $47 showing this.
* **Worth pursuing:** a learned hierarchy whose segmentation also acts as adaptive compute, judged on
  (a) how fast its segmentation converges with data, (b) where fixed tokenizers structurally fail (non-English, code,
  numbers, character-level tasks, mixed scripts), and (c) whether one model handles bytes of any modality.
* **What a small budget can honestly produce:** not the crossover (that needs 1e20+ FLOPs), but the tooling and
  diagnostics (iso-FLOP fairness, in-corpus BPE, segmentation-convergence curves, the image-bytes probe) and the
  components that bring the crossover earlier. Those compound; another small-scale bpb win does not.

## 6. Order of work when resuming

Ideas 1 and 7 (free) first; then 2 and 3 (~$20, and they decide whether the published design has a path at our
scale); then 5 (multimodal evidence); 4 and 6 become the next evolve round's search space under the
docs/SCALING_CRITERIA.md filter.
