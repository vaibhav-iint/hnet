# Experiment log

One entry per experiment: what was tried, the exact config, hardware, cost, results and figures. Newest entries at
the bottom. Figures are in `logbook/figures/`; the script that made each figure is named in its entry.

Status legend: **planned** · **running** · **done** · **failed** · **abandoned**

| ID | Title | Status | Cost | Headline |
|---|---|---|---|---|
| EXP-000 | Cartesia Fig. 7 re-analysis (reference data, no training) | done | $0 | Stable-phase data multiplier ≈ 1.7x at 94B tokens and ≈ 3.0x at 189B; the 10x needs the H-Net cooldown |
| EXP-001 | What the blog's chunk-viewer data says about the improved router | done | $0 | Boundaries mark chunk ENDS (causal); stage-1 chunks are 'word␣' and almost never split words or UTF-8 characters |
| EXP-002 | Tiny (~2M) architecture search: routers and ablations vs BPE | running | ~$2 est. | (in progress) |

---

## EXP-000 · Cartesia Fig. 7 re-analysis

*2026-10-28 · status: done · cost: $0 · script: `logbook/exp000_reference.py`*

**Question.** How much of the blog's ">10x fewer tokens" comes from the architecture, and how much from comparing a
cooled-down H-Net with a stable-phase SmolLM3?

**Data.** The per-checkpoint scores behind blog I Fig. 7, extracted from the page's chart bundle into
`configs/reference/blog1_fig7_{hnet,smollm3}.csv`:
* the H-Net has 154 points to 0.99T tokens (cooldown from 0.855T);
* SmolLM3 has 117 points to 11.04T (stage 1 up to 7.93T).

The six tasks are HellaSwag, ARC, PIQA, WinoGrande, CSQA and MMLU.

**Method.** Fit SmolLM3's stage-1 (stable-LR) average as a quadratic in log tokens. For each stable-phase H-Net point,
find the SmolLM3 token count that reaches the same score.

**Results.**

| H-Net tokens | H-Net avg | SmolLM3 avg at same tokens | SmolLM3 tokens to match (smoothed) | Data multiplier |
|---|---|---|---|---|
| 94B | 55.66 | 53.87 | 162B | **1.7x** |
| 189B | 58.11 | 56.86 | 560B | **3.0x** |
| 301B | 58.74 | — | 822B | 2.7x |
| 0.99T (after cooldown) | 63.82 | — | 11.04T (SmolLM3 final, after stages 2–3) | 11x |

![Cartesia Fig. 7 re-analysis](figures/exp000_cartesia_fig7.png)

**Takeaways.**
* The architecture effect at matched stable phase is real but ~2–3x, not 10x. The H-Net's cooldown adds ~2.3 points
  (61.5 → 63.8).
* Our pass criteria for any H-Net vs baseline comparison use stable-vs-stable multipliers. Cooldown is a separate,
  explicitly labelled branch.
* Cartesia's SmolLM3 Winogrande (63.69 at the end) differs from the model card (58.88), so their eval setup differs from
  the card. Our harness is validated against these CSV numbers, not the card.

---

## EXP-001 · What the blog's chunk-viewer data says about the improved router

*2026-10-28 · status: done · cost: $0 · scripts: `scripts/reference/parse_chunk_viewer.py`,
`scripts/reference/analyze_chunk_viewer.py`, `logbook/exp001_chunk_viewer.py`*

**Question.** The blog's H-Net is unreleased and differs from the paper's (which the blog calls "LH-Net",
lazy/lookahead). What does its router actually do?

**Data.** Blog I Figs. 10–11 are interactive viewers. Their data files hold 96 texts (45 natural-language, 51 code)
with, for each text:
* the improved H-Net's stage-0/1 boundary positions and per-byte boundary probabilities p;
* LH-Net's boundaries on the same texts;
* Llama-3 and in-corpus BPE tokenizations.

These were parsed as data (no page code was executed) into `configs/reference/chunk_viewer.json`. The viewer code
labels an improved-H-Net boundary byte "ends here" (`ends: true` on every example) and a trailing chunk without one
"unfinished".

**Results.**
* **End-of-chunk boundaries.** p ≥ 0.5 coincides with "this byte is the last byte of a stage-0 chunk" on 99.76% of
  327,081 bytes. The decision at byte t can only use bytes ≤ t, so there is no lookahead.
* **Spaces go at the end of chunks.** 92% (English), 78% (other languages) and 68% (code) of stage-1 chunks end right
  after a space. LH-Net and BPE attach the space to the next word instead (≤ 20%).
* **Clean stage-1 units.**

  | Stage-1 cuts | English | Other languages | CJK |
  |---|---|---|---|
  | Inside a word: improved H-Net | 0% | 0.5% | — |
  | Inside a word: LH-Net | 3% | 3% | — |
  | Inside a UTF-8 character: improved H-Net | 0% | 0.01% | 0.5% |
  | Inside a UTF-8 character: LH-Net | 3% | 24% | 81% |
  | Inside a UTF-8 character: Llama-3 BPE | 0.2% | 0.07% | 4% |

  This confirms the blog's CJK claim quantitatively.
* **Compression (bytes per stage-1 chunk):**

  | Model | English | Other languages | CJK | Code |
  |---|---|---|---|---|
  | Improved H-Net | 7.0 | 9.7 | 4.3 | 7.0 |
  | LH-Net | 7.4 | 7.0 | 3.6 | 7.4 |
  | Llama-3 BPE | 4.5 | 4.5 | 3.6 | 4.0 |

* **Boundary probabilities are bimodal** (most mass below 0.1 or above 0.9), with a tail in between.

Example (first 96 bytes of the English example, ␣ = space):

```
H-Net s0 : Ever␣|wonder|ed␣|if␣|you|␣|could␣|beat|␣|a␣|bear|␣|or␣|a␣|horse|␣|in␣|a␣|race|?|␣|What|␣|about␣|a␣|rat␣|or␣|a␣|rh|ino|?|␣|The␣|an
H-Net s1 : Ever␣|wondered␣|if␣|you␣could␣|beat␣|a␣bear␣|or␣|a␣horse␣|in␣|a␣race?␣|What␣|about␣|a␣rat␣|or␣|a␣rhino?␣|The␣|an
LH-Net s0: Ever|␣wondered|␣i|f␣you|␣could|␣be|at|␣a|␣be|ar|␣or|␣a|␣ho|rse|␣i|n␣a|␣ra|ce|?␣|What|␣about|␣a|␣rat|␣or|␣a|␣r|hino|?␣|The|␣an
LH-Net s1: Ever|␣wondered|␣i|f␣you␣could␣beat|␣a␣bear|␣or|␣a␣horse|␣in␣a|␣race|?␣What|␣about|␣a␣rat|␣or|␣a␣rhino|?␣The␣an
Llama-3  : Ever|␣wondered|␣if|␣you|␣could|␣beat|␣a|␣bear|␣or|␣a|␣horse|␣in|␣a|␣race|?|␣What|␣about|␣a|␣rat|␣or|␣a|␣rh|ino|?|␣The|␣an
```

![Chunk-viewer analysis](figures/exp001_chunk_viewer.png)

**What we implement from this.**
* Routers declare start or end boundary semantics.
* With end semantics, the chunk vector is the last byte's state, and byte t reads the most recent chunk that ended at
  or before t.
* The router function is still a guess. Candidates `end_linear`, `end_mlp` and `end_cosine_shift` go through the
  1–10M architecture search (PLAN §2b), with the chunk statistics above as the targets for our trained models.

---

## EXP-002 · Tiny architecture search: routers and ablations vs BPE

*2026-10-29 · status: running · hardware: 1× L4 spot (GCP us-east1-c), 3 runs in parallel · spec:
`configs/search/exp002_tiny2m.yaml` · commit f988ba8 · fairness: `runs/search/exp002_tiny2m/fairness.md`*

**Question.** At tiny scale and equal FLOPs/byte and data:
* does the paper's 2-stage H-Net beat the BPE baseline;
* does an end-of-chunk router (the blog's design, EXP-001) beat the paper's lookahead router;
* which paper components matter?

**Setup.** Nine variants, each with 2 seeds.
* **Baseline:** BPE (GPT-2 tokenizer, 15M params incl. a 50K-vocab head).
* **H-Nets** (~7.9M params, 2-stage, (3,3)), with these routers:
  * the paper's `cosine_lookahead`;
  * three end-of-chunk guesses: `end_linear`, `end_mlp` and `end_cosine_shift`.
* **Ablations of the paper router:**
  * targets (4,2);
  * no smoothing;
  * no per-stage LR multipliers;
  * ratio-loss α = 0.1.

All runs share:
* **Data:** FineWeb-Edu train shard 1 in the same window order; 231M bytes per run (50M token-equivalents).
* **Optimizer:** AdamW, peak LR 1.5e-3, WSD (10% warmup, 20% inverse-sqrt decay), bf16, no compile.
* **Evaluation:** the same 1 MB val text for bpb, plus paper7 task metrics every 500 steps.

**Results.** Training in progress; live figure below (re-rendered by `logbook/exp002_progress.py`).

![EXP-002 progress](figures/exp002_progress.png)
