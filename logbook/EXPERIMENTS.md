# Experiment log

One entry per experiment: what was tried, the exact config, hardware, cost, results and figures. Newest entries at
the bottom. Figures are in `logbook/figures/`; the script that made each figure is named in its entry.

Status legend: **planned** · **running** · **done** · **failed** · **abandoned**

| ID | Title | Status | Cost | Headline |
|---|---|---|---|---|
| EXP-000 | Cartesia Fig. 7 re-analysis (reference data, no training) | done | $0 | Stable-phase data multiplier ≈ 1.7x at 94B tokens and ≈ 3.0x at 189B; the 10x needs the H-Net cooldown |

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
