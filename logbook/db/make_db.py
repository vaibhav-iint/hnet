#!/usr/bin/env python3
"""Build logbook/db/*.csv for the H-Net vs BPE project.

Sources, in priority order: logbook/EXPERIMENTS.md, logbook/WORKLOG.md, configs/search/*.yaml, run directories
(runs/gcp/<exp>/<arm>_s<seed>/{eval,metrics}.jsonl, done.json) and the live mirrors in the scratchpad.
Every number in runs.csv is read from eval.jsonl where a run directory exists (source = eval_jsonl); numbers typed in
from the logbook carry source = worklog / experiments_md.
"""
import csv, glob, json, math, os, re, statistics, datetime
from collections import OrderedDict, defaultdict

REPO = "/home/user/hnet-smollm3"
LIVE = "/tmp/claude-0/-home-user-hnet-smollm3/e0ec0617-4cec-5bc8-abcc-216d4a655704/scratchpad/live"
OUT = os.path.join(REPO, "logbook", "db")
NULL = "NULL"
BPE_REF = 1.634  # EXP-006 bpe, 3 seeds, 50M token-eq, finewebedu10B


def r4(x):
    return NULL if x is None else f"{x:.4f}"


# ----------------------------------------------------------------------------------------------------------------
# 1. experiments
# ----------------------------------------------------------------------------------------------------------------
# category: baselines & calibration | router / decision rule | hierarchy use & bypass | chunk vocabulary & memory |
#           curricula & priors | optimizer & mixers | ladder / scaling
TINY = dict(scale="tiny_2m", data="finewebedu10B", budget_tokens_eq="50M", gpu="L4")
E = []


def exp(exp_id, name, spec, date, question, status, outcome, section, category, seeds=3, cost=None, notes="", **kw):
    row = dict(exp_id=exp_id, name=name, spec_file=spec, date=date, question=question, status=status,
               scale=kw.get("scale", NULL), data=kw.get("data", NULL), budget_tokens_eq=kw.get("budget_tokens_eq", NULL),
               seeds=seeds, gpu=kw.get("gpu", NULL), cost_usd=NULL if cost is None else cost, outcome=outcome,
               worklog_section=section, category=category, notes=notes)
    E.append(row)


exp("EXP-000", "Cartesia Fig. 7 re-analysis", "logbook/exp000_reference.py", "2026-10-28",
    "How much of the blog's '>10x fewer tokens' is architecture and how much is comparing a cooled-down H-Net with a stable-phase SmolLM3?",
    "done", "Stable-vs-stable data multiplier is 1.7x at 94B tokens and 3.0x at 189B; the 10x needs the H-Net cooldown.",
    "21:39 · References read, including rendered charts", "baselines & calibration", seeds=0, cost=0,
    scale="reference data", data="blog I Fig. 7 CSVs", budget_tokens_eq=NULL, gpu="none",
    notes="No training; re-analysis of configs/reference/blog1_fig7_*.csv.")
exp("EXP-001", "What the blog's chunk-viewer data says about the improved router", "logbook/exp001_chunk_viewer.py", "2026-10-28",
    "What does the blog's (unreleased) improved H-Net router actually do?",
    "done", "Boundaries mark chunk ENDS causally (p>=0.5 = last byte of a stage-0 chunk on 99.76% of bytes); stage-1 chunks are 'word+space' and almost never split words or UTF-8 characters.",
    "22:30 · Reverse-engineering the blog's router (user: \"the blog architecture differs; we'll have to guess\")",
    "baselines & calibration", seeds=0, cost=0, scale="reference data", data="blog I Figs. 10-11 chunk-viewer bundles",
    gpu="none", notes="Led to end-of-chunk router semantics (end_linear, end_mlp, end_cosine_shift).")
exp("EXP-002", "Tiny architecture search: routers and ablations vs BPE", "configs/search/exp002_tiny2m.yaml", "2026-10-29",
    "At ~2M params and equal FLOPs/byte and data, does the paper's 2-stage H-Net beat BPE, does an end-of-chunk router beat the lookahead router, and which paper components matter?",
    "done", "Every H-Net trails BPE (1.635) by 0.15-0.21 bpb; variants differ by less than the seed spread (up to 0.05), so EXP-002 cannot rank routers or ablations.",
    "EXP-002 at 10/18; EXP-003 launched (commit df6ad43)", "baselines & calibration", seeds=2, cost=3.3,
    notes="Cost = ~$2 estimate + ~$1.3 for the on-demand L4 after 5 preemptions. 16/18 runs finished: paper_s1 and end_linear_s0/s1 were not re-run after the crashes.", **TINY)
exp("EXP-003", "AdamW learning-rate sweep (tuned fairness)", "configs/search/exp003_lr_adamw.yaml", "2026-10-29",
    "What is each architecture's best AdamW LR, and what is the H-Net vs BPE gap at tuned settings?",
    "stopped", "BPE's best LR is 6e-3 at the grid edge (1.564 +- 0.012); every H-Net peaks at 1.5e-3 (1.78-1.80) and degrades above it because of the stage-1 limit cycle; the tuned gap is 0.235 bpb.",
    "Where the n-gram gain lands; LR sweep; mid-training router collapses", "baselines & calibration", seeds=3, cost=9.26,
    notes="Cost = launch worst case. 55/60 runs finished in the run dirs (EXPERIMENTS.md says ~40/60); the 5 unfinished are seed-2 runs of paper/end_mlp/end_cosine_shift at lr 1.5e-3 and 3e-3. Partial report logbook/reports/exp003_partial/.", **TINY)
exp("EXP-004", "Muon vs AdamW (tiny, 4 architectures)", "configs/search/exp004_muon.yaml", "2026-10-29",
    "Does Muon on the hidden matrices change the H-Net vs BPE gap at tiny scale?",
    "superseded", "Never launched (params best_lr.* never filled); the Muon question was answered by EXP-025 on the best H-Net instead.",
    "WS-J done: trainer flags, resume hardening, sweep spec v2, launcher, report", "optimizer & mixers", seeds=3,
    notes="Spec only. 4 architectures x Muon LR {0.0125, 0.025, 0.05} x 3 seeds = 36 runs planned.", **TINY)
exp("EXP-005", "Attention variants inside BPE and H-Net", "configs/search/exp005_attn_variants.yaml", "2026-10-29",
    "Do DeepSeek (MLA, NSA, DSA) or Kimi (KDA, MoBA, gated) attention variants change the tiny-scale picture?",
    "superseded", "Never launched; the router-dynamics and vocabulary lines took the budget. Mixer swaps were tested on the H-Net's outer networks in EXP-010 and EXP-026 instead.",
    "WS-I1 done: DeepSeek MLA, NSA, DSA", "optimizer & mixers", seeds=3,
    notes="Spec only. {bpe, hnet} x {attn, mla, nsa, dsa, kda, kda_hybrid, moba, gated_attn} x 3 seeds = 48 runs planned; bpe_kda_hybrid model file does not exist.", **TINY)
exp("EXP-006", "Parameter matching with hashed n-gram embeddings", "configs/search/exp006_close_gap_tiny.yaml", "2026-10-29",
    "Is the gap to BPE its FLOP-free vocabulary table? Add FLOP-free hashed byte n-gram tables matching BPE's 15M params.",
    "done", "~40% of the gap is BPE's free table: 2-stage + n-gram 1.714 +- 0.016, 1-stage + n-gram 1.705 (1 seed) vs BPE 1.634; the gain lands in the byte stage (main-network contribution drops 0.63 -> 0.38).",
    "First results of the fix sweeps (val bpb, mean ± sd over seeds; BPE = 1.634 ± 0.002, 3 seeds)", "chunk vocabulary & memory", seeds=3,
    notes="14/17 runs finished (paper_s0 stopped at step 1180, end_mlp_ng_s1 at 1220, hnet1_ng_s0 at 1680; hnet1_ng_s2 never started). Its bpe arm is THE reference baseline (1.634 +- 0.002).", **TINY)
exp("EXP-007", "(number never used)", NULL, NULL,
    NULL, "voided", "No spec, no run directory and no logbook entry carry this number; the numbering jumps from EXP-006 to EXP-008.",
    NULL, "baselines & calibration", seeds=0, notes="Placeholder row so the sequence EXP-001..028 is complete.")
exp("EXP-008", "Training signals for the hierarchy (paper router)", "configs/search/exp008_hierarchy_use.yaml", "2026-10-29",
    "Can residual path-drop, a main-path auxiliary head or a router-entropy penalty make the paper 2-stage H-Net use its hierarchy?",
    "done", "Path-drop hurts and inflates compression (+31% compute); the aux head gives ~0.01 but collapses stage-0 compression to 1.25 bytes/position; the entropy penalty is harmful (combo_end 1.652 at +379% FLOPs, invalid). base = 1.796 +- 0.001 at +11% compute.",
    "Interim readouts of EXP-008/010/011 (honest-compute view; `logbook/sweep_summary.py`)", "hierarchy use & bypass", seeds=3,
    notes="combo and combo_end were dropped from the spec after the routers collapsed to select-all; their run dirs remain (voided). rent_s2 and mainaux_s2 did not finish.", **TINY)
exp("EXP-009", "Oracle chunking: is routing the bottleneck?", "configs/search/exp009_oracle_chunks.yaml", "2026-10-29",
    "With hand-made word boundaries at identical FLOPs, how much of the gap to BPE does perfect routing close?",
    "done", "Chunk semantics decide how much the hierarchy is used (end 0.56 > start 0.45 > fixed 0.25 bpb of main contribution), but oracle word chunks are only 0.04 better than the learned router and 0.17 behind BPE: routing is not the main gap.",
    "First results of the fix sweeps (val bpb, mean ± sd over seeds; BPE = 1.634 ± 0.002, 3 seeds)", "router / decision rule", seeds=3, **TINY)
exp("EXP-010", "Dynamics fixes and Mamba-free outer networks", "configs/search/exp010_dynamics_and_mixers.yaml", "2026-10-29",
    "Do a strong ratio loss (alpha 1) or an identity-init residual fix the early starvation of the hierarchy, and can the outer Mamba-2 be replaced by SWA or KDA?",
    "stopped", "alpha 1 holds compression (+0.8%) at 1.810 +- 0.012 (2 seeds): the paper router's apparent edge over fixed chunks was extra compute; identity residual 1.783 but at +15.6% compute and contribution 0; SWA/KDA arms did not finish (KDA crashed on a kernel channel constraint, SWA stopped at step 250).",
    "Interim readouts of EXP-008/010/011 (honest-compute view; `logbook/sweep_summary.py`)", "hierarchy use & bypass", seeds=3,
    notes="8/14 runs finished. The EXP-026 spec cites 'EXP-010's SWA arm ended at 2.005 bpb' but the only SWA record in the run dirs is 2.8607 at step 250 (swa_s0); the 2.005 could not be located.", **TINY)
exp("EXP-011", "Router-free fixed (3,3) hourglass + training signals", "configs/search/exp011_fixed_hourglass.yaml", "2026-10-29",
    "With fixed pooling (no learned router) at honest compute, which training signal restores use of the hierarchy?",
    "done", "fixed 1.832 +- 0.008; + aux head 1.821 +- 0.002 (best honest-compute H-Net before the vocabulary line); residual warmup 1.844 neutral; identity residual 1.854 (contribution 0.10); path-drop 1.889 harmful.",
    "Harvest of the finished sweeps (`logbook/sweep_summary.py --all`, honest-compute view)", "hierarchy use & bypass", seeds=3,
    notes="fixed_rdrop_s2 failed at step 50 (no eval).", **TINY)
exp("EXP-012", "Chunk vectors that summarize the chunk (pooling chunker)", "configs/search/exp012_pool_chunker.yaml", "2026-10-29",
    "Does a chunk vector that pools over the chunk (mean / attention) instead of the boundary state let the main network do word-level LM?",
    "stopped", "Incomplete: select 1.917 / 1.820 (2 seeds, large spread), pool_mean 1.789 (1 seed); attention-pool arms never finished (first attempt crashed on a bf16 dtype mismatch, relaunch left at step <=940). No write-up; the budget moved to the rank-decision/vocabulary line.",
    "Finding: the chunk vector is a boundary *state*, not a chunk *summary* (`probe_main.py --identity`)", "hierarchy use & bypass", seeds=3,
    notes="EXPERIMENTS.md still says 'running'; the run dirs show 3/10 finished and nothing after step 970. Status 'stopped' is inferred.", **TINY)
exp("EXP-013", "Can the H-Net use BPE's learning rate?", "configs/search/exp013_lr_allocation.yaml", "2026-10-29",
    "At base LR 6e-3, do a strong ratio loss, an entropy penalty, no multipliers or main-heavy multipliers let the H-Net train at BPE's LR?",
    "stopped", "Incomplete (6/13 finished): lr6 1.867 (1 seed), lr6_alpha1 1.927/1.940, lr6_nomult 1.785 (1 seed), lr6_rent 1.907/1.781; no arm approaches the 1.5e-3 results. Superseded by rank decisions (EXP-014/017), which remove the LR ceiling's mechanism.",
    "The stage-1 router is a limit cycle at high LR (`logbook/figures/exp003_collapses.png`)", "optimizer & mixers", seeds=3,
    notes="EXPERIMENTS.md says 'running'; never written up. lr6_nomult_s0 was used for chunk_stats (character-pair tokenizer at stage 0).", **TINY)
exp("EXP-014", "Rank-based (quantile) boundary decisions, first version", "configs/search/exp014_quantile_decision.yaml", "2026-10-29",
    "Do rank (top 1/N) boundary decisions remove the stage-1 limit cycle and let the H-Net use lr 6e-3?",
    "voided", "Invalid: the p-space quantile saturated (tau -> 1.000 within 70-90 steps, sigmoid exactly 1 for > 1/N positions) and collapsed the selection (compression 440-1800) in most runs; stopped at step ~300. Rewritten to rank batch-normalized fp32 logits (validated as EXP-017 quant_fix).",
    "EXP-014 post-mortem: the p-space quantile rule saturates and collapses", "router / decision rule", seeds=3,
    notes="quant_pool at lr 1.5e-3 never started. Rows are the collapsed runs, recorded for the post-mortem only.", **TINY)
exp("EXP-015", "Hierarchy training signals at honest compute (factorial)", "configs/search/exp015_hierarchy_factorial.yaml", "2026-10-29",
    "With compression pinned by rank decisions, do path-drop, the aux head, both, or both + mean pooling make the hierarchy carry the model?",
    "stopped", "Never launched: built on the EXP-014 decision rule and stopped with it; to be re-run on the fixed rule only if EXP-017 says the hierarchy is worth more training signals (it said: give it a vocabulary instead).",
    "EXP-014 post-mortem: the p-space quantile rule saturates and collapses", "hierarchy use & bypass", seeds=3, notes="Spec only.", **TINY)
exp("EXP-016", "Chunk-level multi-byte prediction", "configs/search/exp016_chunk_pred.yaml", "2026-10-29",
    "Does a head on the innermost main network's output predicting the next chunk's first k bytes give the hierarchy a direct training signal?",
    "stopped", "Never launched (same reason as EXP-015). The idea survived as the next-chunk objective of the chunk vocabulary (EXP-017).",
    "Interim readouts of EXP-008/010/011 (honest-compute view; `logbook/sweep_summary.py`)", "hierarchy use & bypass", seeds=3, notes="Spec only.", **TINY)
exp("EXP-017", "A vocabulary for the learned chunks (2-stage, stage-0 table)", "configs/search/exp017_chunk_vocab.yaml", "2026-10-29",
    "Does a hashed chunk vocabulary (parameter-matched to BPE's table) plus a sampled-softmax next-chunk objective close the gap at (3,3)?",
    "done", "quant_fix 1.799 +- 0.005 (the fixed rank decider alone equals the paper router's +11% compute); vocab 1.784 +- 0.015 (small gain); vocab_pred 1.836 and vocab_pred_w1 1.862 harmful: stage-0 chunks are character pairs, a table of them is nearly worthless and predicting them is noise.",
    "EXP-017/018 finals (step 1763, honest compute, 3 seeds)", "chunk vocabulary & memory", seeds=3, **TINY)
exp("EXP-018", "Where should the chunk vocabulary sit? (1-stage, word-scale chunks)", "configs/search/exp018_vocab_shapes.yaml", "2026-10-29",
    "Does the vocabulary + next-chunk objective work at word scale (1-stage, ratio 6)?",
    "done", "h1_quant 1.853; h1_vocab_pred 1.720 / 1.863 / 1.737 with main contribution 0.75 / 0.13 / 0.55: -0.13 / -0.11 bpb at identical FLOPs in two seeds (best honest-compute H-Net so far); seed 1 did not adopt the hierarchy (suffix tokenizer).",
    "EXP-017/018 finals (step 1763, honest compute, 3 seeds)", "chunk vocabulary & memory", seeds=3, **TINY)
exp("EXP-019", "A router that is a tokenizer: context-free boundary scores", "configs/search/exp019_local_router.yaml", "2026-10-29",
    "Does a context-free router (logit from the last 8 bytes) give consistent chunks, and does the vocabulary then help more than with the contextual router?",
    "done", "Finals (run dirs; never written up): local 1.813 +- 0.001, local_vocab 1.838 +- 0.072, local_vocab_pred 1.836 +- 0.031: identical bpb to the contextual router at (3,3); the windowed design is not more consistent at init (8% at k=8).",
    "Consistency at initialization vs after training: the router objective is anti-tokenizer", "router / decision rule", seeds=3,
    notes="EXPERIMENTS.md says 'running'; all 9 runs are done in the run dirs. Final numbers appear only here.", **TINY)
exp("EXP-020", "Router self-consistency penalty (second launch, scale-free)", "configs/search/exp020_router_consistency.yaml", "2026-10-29",
    "Does penalizing the within-4-gram variance of the stage-0 logit make the learned tokenizer consistent and the second level useful?",
    "done", "cons1 1.798, cons10 1.799 vs quant_fix 1.799 +- 0.005: neutral on average with a larger seed spread; cons10_vocab_pred 1.859 (vocab_pred without the penalty 1.836). Consistency can be imposed without cost but does not make the second level useful at (3,3).",
    "EXP-020 first launch invalid: an unnormalized consistency penalty is satisfied by shrinking the logits", "router / decision rule", seeds=3,
    notes="First launch voided: the unnormalized penalty was satisfied by shrinking the logits (score_std 1.2 -> 0.15); no run dirs of it remain.", **TINY)
exp("EXP-021", "Word-scale vocabulary with the median/MAD (robust) decider", "configs/search/exp021_word_vocab_robust.yaml", "2026-10-29",
    "Does standardizing router scores by median/MAD remove the heavy-tailed-score non-adoption of EXP-018 seed 1?",
    "stopped", "Stopped at step ~1000: with a word tokenizer 5/6 of the scores are identical, the MAD -> 0, 1/sigma blew up the gradient (grad norm 1.9e6) and h1r_vocab_pred_s0 diverged (3.25 bpb); seeds 1-2 did not adopt (1.98 / 2.00 at step 1000). Replaced by EXP-023.",
    "EXP-021 (median/MAD decider) stopped: the MAD collapses under a word tokenizer", "chunk vocabulary & memory", seeds=3,
    notes="No run dirs under runs/gcp; numbers are from the work log at step 1000.", **TINY)
exp("EXP-022", "Levers on the adoption race (draft)", "configs/search/exp022_adoption.yaml", "2026-10-29",
    "Do residual warmup 0.3, confidence temperature 4 or prediction weight 1.0 make adoption reliable (on the robust decider)?",
    "superseded", "Draft superseded by EXP-023 (trimmed decider; rwarm arm carried over as vp_rwarm) and EXP-024 (temp4 arm carried over).",
    "EXP-021 (median/MAD decider) stopped: the MAD collapses under a word tokenizer", "curricula & priors", seeds=3, notes="Spec only.", **TINY)
exp("EXP-023", "Word-scale chunk vocabulary, trimmed-statistics rank decider (+ n-gram tables, + residual warmup)", "configs/search/exp023_word_vocab_trimmed.yaml", "2026-10-29",
    "With an outlier-proof decider, does the word-scale vocabulary replicate EXP-018, and do byte n-gram tables or residual warmup help?",
    "done", "vp 1.732 / 1.855 / 1.733 replicates EXP-018 seed by seed (seed 1 not adopted: adoption is decided by init/data order, not the decider); vp_ng 1.720 / 1.669 / 1.656: adopted seeds 0.02-0.035 from BPE at identical FLOPs; vp_rwarm 1.810 / 1.875 / 1.746 worse in every seed.",
    "EXP-023 tokenizer statistics: adoption and the kind of tokenizer go together (`cs_exp023_*.json`)", "chunk vocabulary & memory", seeds=3, **TINY)
exp("EXP-024", "Reliable adoption: a whitespace tokenizer curriculum", "configs/search/exp024_adoption_curriculum.yaml", "2026-10-29",
    "Does training the router toward 'a chunk ends at every whitespace byte' for the first 10% of steps make adoption reliable?",
    "done", "prior 1.6337 / 1.6346 / 1.6364 = 1.635 +- 0.001: H-Net matches BPE (1.634 +- 0.002) at identical FLOPs, every seed adopts by step 250 (main contribution 0.56-0.60); temp4 1.723 goes the other way (contribution 0.02-0.21).",
    "EXP-024: the tokenizer curriculum makes adoption reliable; H-Net matches BPE at equal FLOPs", "curricula & priors", seeds=3, **TINY)
exp("EXP-025", "Muon vs AdamW on the BPE baseline and the best H-Net (EXP-024 prior)", "configs/search/exp025_muon.yaml", "2026-10-29",
    "Does Muon (the ladder optimizer) change the BPE vs H-Net picture at tiny scale?",
    "running", "From the run dirs (not yet in the logbook): prior_muon mlr 0.025 1.6299 / 1.6347 / 1.6297 (1.631, slightly below AdamW's 1.635); mlr 0.05 1.790 +- 0.023; bpe_muon mlr 0.05 1.710 +- 0.003 (worse than AdamW 1.634); bpe_muon mlr 0.025 still running (1.715 at step 750).",
    "Why the issues were found late: CLAUDE.md, a step-0 gradient audit, an invariance suite, a visibility tool", "optimizer & mixers", seeds=3,
    notes="GPU inferred (tiny sweeps run on spot L4s). The work log mentions EXP-025 only in the training-report note (bypass rms 5-40x the hierarchy's in every H-Net run).", **TINY)
exp("EXP-026", "Mamba-free outer networks for the best H-Net (SWA, KDA)", "configs/search/exp026_mamba_free.yaml", "2026-10-29",
    "Can the byte-level Mamba-2 encoder/decoder of the EXP-024 prior model be replaced by sliding-window attention or Kimi Delta Attention?",
    "done", "From the run dirs (not yet in the logbook): swa 1.7422 / 1.7437 / 1.7408 = 1.742 +- 0.001, 0.11 behind the Mamba-2 reference 1.635; kda failed (rc=1, all 3 seeds).",
    "(not yet in the work log)", "optimizer & mixers", seeds=3,
    notes="GPU inferred. Status 'done' = the sweep finished (swa done, kda failed); no logbook entry yet.", **TINY)
exp("EXP-027", "Breaking the adoption race at its mechanism (chunk STE, margin-gated bypass)", "configs/search/exp027_adoption_mechanism.yaml", "2026-10-29",
    "Does giving the router its own gradient (chunk STE) or gating the bypass by router margin remove the seed lottery without the whitespace prior?",
    "running", "Work log interim at step 500: same loss as the reference, margin gate makes the hierarchy load-bearing from the start (0.42 bpb vs 0.03-0.10). First finals in the live mirror (not yet in the logbook): cste_s0 1.778, cste_rgate_s0 1.742, rgate_s1 1.699 vs vp_ng 1.720 / 1.669 / 1.656 and prior 1.635: no arm below the prior-free reference so far; 3 of 6 started runs still running, seed 2 queued.",
    "Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers", "hierarchy use & bypass", seeds=3, cost=2.22,
    notes="Cost = worst case for 2 L4 spot VMs. Only seeds 0 and 1 have started (live mirror); seed 2 queued. Launched 18:30 UTC.", **TINY)
exp("EXP-028", "A usefulness objective for the router (REINFORCE on the hierarchy's gain)", "configs/search/exp028_router_gain.yaml", "2026-10-29",
    "Does training the router's decisions on the hierarchy's measured per-byte gain prefer the word tokenizer prior-free?",
    "planned", "Spec written today, not launched. Not iso-FLOP (+~15% for the second decoder pass): a mechanism test.",
    "Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers", "router / decision rule", seeds=3,
    notes="Base = EXP-023 vp_ng; arms gain, gain_rgate, gain_cste.", **TINY)

LADDER_RUNGS = {  # rung: (D, layout, d0, chunk-vocab buckets, n-gram buckets)  from docs/LADDER.md
    "L03": (256, "m2 / T3 / m2", 192, 25733, 4817), "L04": (256, "m2 / T4 / m2", 192, 25733, 4817),
    "L05": (384, "m3 / T5 / m3", 224, 32621, 6113), "L06": (512, "m3 / T6 / m2", 256, 38501, 7229),
    "L07": (512, "m3 / T7 / m2", 256, 38501, 7229), "L08": (640, "m2 / T8 / m2", 320, 38461, 7219),
    "L09": (640, "m2 / T9 / m2", 320, 38461, 7219), "L10": (768, "m3 / T10 / m3", 320, 45697, 8563),
    "L11": (768, "m3 / T11 / m3", 320, 45697, 8563), "L12": (896, "m2 / T12 / m2", 384, 44843, 8389),
}


def ladder_spec_info(path):
    """steps / BPE tokens / H-Net bytes from the generated spec's comment and set lines."""
    txt = open(path).read()
    steps = re.search(r"same steps \((\d+)\)", txt)
    tok = re.search(r"BPE ([\d.]+)B tokens", txt)
    byt = re.search(r"H-Net ([\d.]+)B bytes", txt)
    return (int(steps.group(1)) if steps else None, tok.group(1) + "B tokens" if tok else None, byt.group(1) + "B bytes" if byt else None)


LADDER = dict(scale=None, data="mix_v1", gpu="A100")
exp("LADDER-LR-1e16", "Ladder Muon-LR tuning, rung L04 at 1e16 FLOPs", "configs/search/ladder_lr_1e16.yaml", "2026-10-29",
    "What is the best Muon LR for BPE L04 and its two H-Net conversions at 1e16 FLOPs on mix_v1?",
    "done", "BPE has a sharp optimum at 0.02 (1.622; 0.08 worse one grid step either side); both H-Nets prefer 0.01 (hnet 1.861, hnet_ws 1.820) and are flatter. At BPE's optimum the H-Net trails by 0.24 prior-free / 0.20 with the warm start.",
    "Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched", "ladder / scaling", seeds=1, cost=4.12,
    scale="ladder_L04", budget_tokens_eq="0.095B tokens (1e16 FLOPs)", data="mix_v1", gpu="A100",
    notes="VM hnet-ladder-lr ($4.12); BPE retries ran on hnet-ladder-lr2. 728 steps x 131072 tokens; H-Nets 0.397B bytes.")
exp("LADDER-LREXT-1e16", "Ladder Muon-LR grid extended downward (0.0025, 0.005)", "configs/search/lrext_1e16.yaml", "2026-10-29",
    "Is the H-Net's LR optimum below 0.01? (the first grid was monotone)",
    "done", "No: hnet 1.934 (margin 0.03, router undecided) / 1.888 and hnet_ws 1.878 / 1.822 at 0.0025 / 0.005; the optimum stays at 0.01. bpe 1.796 / 1.736.",
    "Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched", "ladder / scaling", seeds=1, cost=3.5,
    scale="ladder_L04", budget_tokens_eq="0.095B tokens (1e16 FLOPs)", data="mix_v1", gpu="A100",
    notes="VM hnet-ladder-lr2 (~$3.5, preempted and resumed three times in 20 minutes).")
for budget in ("1e16", "1e17", "1e18"):
    for rung in ("L03", "L04", "L05", "L06", "L07", "L08", "L09", "L10", "L11"):
        spec = f"configs/search/ladder_{budget}_{rung}.yaml"
        if not os.path.exists(os.path.join(REPO, spec)):
            continue
        steps, tok, byt = ladder_spec_info(os.path.join(REPO, spec))
        if budget == "1e16":
            status = {"L03": "done", "L04": "done"}.get(rung, "running")
            outcome = {"L03": "bpe 1.695, hnet 1.879, hnet_ws 1.823 at the final step 908 (no done.json in the mirror): H-Net trails BPE by 0.18 prior-free, 0.13 with the warm start.",
                       "L04": "bpe 1.622 (identical records to ladder_lr_1e16 bpe mlr0.02), hnet 1.907, hnet_ws 1.864 at the final step 728: H-Net trails BPE by 0.28 / 0.24 at BPE's LR (0.24 / 0.20 at its own LR in the LR sweep).",
                       }.get(rung, "Launched on hnet-ladder-1e16 (4xA100 spot); no eval records mirrored yet.")
            note = "1e16 rungs L03-L06 launched on hnet-ladder-1e16; 1e17 held until EXP-027's first evals decide whether the ladder's H-Net arm carries chunk-STE / gated bypass."
        else:
            status, outcome = "planned", "Spec generated; held until EXP-027's first evals (1e17) / budget (1e18, warm-start arm dropped to fit +$100)."
            note = "Blog protocol: BPE's best Muon LR (0.02) transferred to every run."
        exp(f"LADDER-{budget}-{rung}", f"Iso-FLOP ladder {budget} FLOPs, rung {rung}", spec, "2026-10-29",
            f"At {budget} training FLOPs and the same data, does the 1-stage H-Net conversion of BPE {rung} (prior-free / whitespace warm start) match BPE?",
            status, outcome, "Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched",
            "ladder / scaling", seeds=1, scale=f"ladder_{rung}", budget_tokens_eq=f"{tok} ({budget} FLOPs)", data="mix_v1", gpu="A100",
            notes=f"{steps} steps; H-Net {byt}. {note}")

# ----------------------------------------------------------------------------------------------------------------
# 2. arms + settings
# ----------------------------------------------------------------------------------------------------------------
BASE = {
    "configs/runs/tiny_bpe_2m.yaml": OrderedDict([
        ("tokenizer", "bpe_gpt2_50k"), ("vocab_size", "50304"), ("n_stages", "0"), ("arch_layout", "dense llama, 10 layers"),
        ("d_model", "128"), ("d_intermediate", "384"), ("mixer", "attn"), ("optimizer", "adamw"), ("lr", "1.5e-3"),
        ("schedule", "wsd (warmup 0.1, decay 0.2 inverse_sqrt)"), ("train_tokens", "5.0e7"), ("seq_len", "448"),
        ("batch_tokens", "28672"), ("dataset", "finewebedu10B (shard 1)")]),
    "configs/runs/tiny_hnet2_2m.yaml": OrderedDict([
        ("tokenizer", "bytes"), ("vocab_size", "256"), ("n_stages", "2"), ("arch_layout", '["m4", ["T1m4", ["T7"], "m4T1"], "m4"]'),
        ("d_model", "[96, 128, 256]"), ("d_intermediate", "[0, 384, 768]"), ("target_ratios", "[3.0, 3.0]"),
        ("ratio_loss_alpha", "0.03"), ("router", "cosine_lookahead"), ("decision", "threshold"), ("chunker", "select"),
        ("dechunker", "ema"), ("ste", "true"), ("lr_multiplier", "auto"), ("mixer", "mamba2+attn"), ("optimizer", "adamw"),
        ("lr", "1.5e-3"), ("schedule", "wsd (warmup 0.1, decay 0.2 inverse_sqrt)"), ("train_tokens", "5.0e7"),
        ("seq_len", "2048"), ("batch_bytes", "131072"), ("dataset", "finewebedu10B (shard 1)")]),
}
BASE["configs/runs/tiny_hnet1_2m.yaml"] = OrderedDict(BASE["configs/runs/tiny_hnet2_2m.yaml"])
BASE["configs/runs/tiny_hnet1_2m.yaml"].update({"n_stages": "1", "arch_layout": '["m4", ["T8"], "m4"]', "d_model": "[96, 256]",
                                                 "d_intermediate": "[0, 768]", "target_ratios": "[6.0]"})
BASE["configs/runs/tiny_hnet2_2m_ratios42.yaml"] = OrderedDict(BASE["configs/runs/tiny_hnet2_2m.yaml"])
BASE["configs/runs/tiny_hnet2_2m_ratios42.yaml"].update({"target_ratios": "[4.0, 2.0]", "d_intermediate": "[0, 384, 896]"})


def ladder_base(kind, rung):
    D, layout, d0, cvb, ngb = LADDER_RUNGS[rung]
    L = int(rung[1:])
    if kind == "bpe":
        return OrderedDict([("tokenizer", "bpe32k"), ("vocab_size", "32768"), ("n_stages", "0"),
                            ("arch_layout", f"dense llama, {L} layers, {L} heads x 128"), ("d_model", str(D)), ("mixer", "attn"),
                            ("optimizer", "muon"), ("schedule", "wsd (warmup 0.1, decay 0.2 inverse_sqrt)"), ("seq_len", "2048"),
                            ("dataset", "mix_v1")])
    return OrderedDict([("tokenizer", "bytes"), ("vocab_size", "256"), ("n_stages", "1"), ("arch_layout", layout),
                        ("d_model", f"[{d0}, {D}]"), ("target_ratios", "[4.145]"), ("ratio_loss_alpha", "0.03"),
                        ("router", "end_mlp"), ("decision", "quantile"), ("chunker", "select"), ("dechunker", "ema"),
                        ("chunk_vocab", f"[{{buckets: {cvb}, n_hash: 2, max_len: 16, pred_weight: 0.5, n_neg: 1024}}]"),
                        ("ngram_embed", f"{{ns: [3, 4, 5, 6, 7, 8], buckets: {ngb}}}"),
                        ("router_prior_weight", "1.0" if kind == "hnet_ws" else "0.0"), ("router_prior_frac", "0.1"),
                        ("lr_multiplier", "auto"), ("mixer", "mamba2+attn"), ("optimizer", "muon"),
                        ("schedule", "wsd (warmup 0.1, decay 0.2 inverse_sqrt)"), ("seq_len", "8512"), ("dataset", "mix_v1")])


SKIP_KEYS = ("train.compile", "eval.", "train.out_dir", "train.seed", "train.ckpt_every")
RENAME = {"train.lr": "lr", "train.muon.lr": "muon_lr", "train.optimizer": "optimizer", "train.muon.adam_betas": "muon_adam_betas",
          "train.muon.adam_weight_decay": "muon_adam_weight_decay", "train.muon.momentum_warmup_steps": "muon_momentum_warmup_steps",
          "train.train_tokens": "train_tokens", "train.train_bytes": "train_bytes", "data.micro_batch": "micro_batch",
          "data.batch_tokens": "batch_tokens", "data.batch_bytes": "batch_bytes", "model.mixer_codes": "mixer_codes",
          "model.mixer_cfg": "mixer_cfg", "model.attn.window_size": "attn_window_size", "model.lr_multiplier": "lr_multiplier",
          "model.ngram_embed": "ngram_embed"}


def parse_set(s):
    k, v = s.split("=", 1)
    k = k.strip()
    if any(k.startswith(p) for p in SKIP_KEYS):
        return None
    if k in ("train.muon.lr_embed", "train.muon.lr_head", "train.muon.lr_other"):
        return "adam_part_lr", v
    if k in RENAME:
        return RENAME[k], v
    if k.startswith("model.dc."):
        return k[len("model.dc."):], v
    if k.startswith("model."):
        return k[len("model."):], v
    return k, v


ARMS = []          # dict rows
SETTINGS = []      # (exp_id, arm, setting, value, origin)


def arm(exp_id, name, label, base, parent=None, notes="", spec_sets=(), run_sets=(), extra=None):
    ARMS.append(dict(exp_id=exp_id, arm=name, label=label, base_config=base, parent_arm=parent or NULL, notes=notes))
    st = OrderedDict()
    basevals = BASE.get(base) if isinstance(base, str) else None
    if basevals is None and extra is not None:
        basevals = extra
    for k, v in (basevals or {}).items():
        st[k] = (v, "base_config")
    for origin, sets in (("spec_set", spec_sets), ("run_set", run_sets)):
        for s in sets:
            p = parse_set(s)
            if p is None:
                continue
            k, v = p
            st[k] = (v, origin)
    # derived: mixer from mixer_codes
    if "mixer_codes" in st:
        mc = st["mixer_codes"][0]
        st["mixer"] = ("swa+attn" if "m: attn" in mc else "kda+attn" if "m: kda" in mc else mc, "derived")
    if "muon_lr" in st and "lr" in st and st["lr"][1] == "base_config":
        del st["lr"]  # the AdamW base LR does not apply under Muon (the Adam part uses adam_part_lr)
    if st.get("decision", ("", ""))[0] == "quantile" and "decision_version" not in st and exp_id not in ("EXP-014",):
        pass
    for k, (v, o) in st.items():
        SETTINGS.append((exp_id, name, k, v, o))


H2, H1, BPE, H42 = ("configs/runs/tiny_hnet2_2m.yaml", "configs/runs/tiny_hnet1_2m.yaml", "configs/runs/tiny_bpe_2m.yaml",
                    "configs/runs/tiny_hnet2_2m_ratios42.yaml")
CV2 = "model.dc.chunk_vocab=[{buckets: 65536, n_hash: 2, max_len: 16}, null]"
CVP2 = "model.dc.chunk_vocab=[{buckets: 65536, n_hash: 2, max_len: 16, pred_weight: 0.5, n_neg: 1024}, null]"
CVP2_W1 = "model.dc.chunk_vocab=[{buckets: 65536, n_hash: 2, max_len: 16, pred_weight: 1.0, n_neg: 1024}, null]"
CVP1 = "model.dc.chunk_vocab=[{buckets: 65536, n_hash: 2, max_len: 16, pred_weight: 0.5, n_neg: 1024}]"
CVP1_W1 = "model.dc.chunk_vocab=[{buckets: 65536, n_hash: 2, max_len: 16, pred_weight: 1.0, n_neg: 1024}]"
NG = "model.ngram_embed={ns: [3, 4, 5, 6, 7, 8], buckets: 12289}"
NG1 = "model.ngram_embed={ns: [3, 4, 5, 6, 7, 8], buckets: 13200}"
Q = ["model.dc.router=end_mlp", "model.dc.decision=quantile"]

# EXP-002
arm("EXP-002", "bpe", "BPE baseline", BPE)
arm("EXP-002", "paper", "paper router (LH-Net), cosine_lookahead, (3,3)", H2)
arm("EXP-002", "end_linear", "end-of-chunk, linear", H2, "EXP-002/paper", run_sets=["model.dc.router=end_linear"])
arm("EXP-002", "end_mlp", "end-of-chunk, MLP", H2, "EXP-002/paper", run_sets=["model.dc.router=end_mlp"])
arm("EXP-002", "end_cosine_shift", "end-of-chunk, cosine", H2, "EXP-002/paper", run_sets=["model.dc.router=end_cosine_shift"])
arm("EXP-002", "paper_r42", "targets (4,2)", H42, "EXP-002/paper")
arm("EXP-002", "paper_nosmooth", "no smoothing", H2, "EXP-002/paper", run_sets=["model.dc.dechunker=repeat"])
arm("EXP-002", "paper_nolrmult", "no LR multipliers", H2, "EXP-002/paper", run_sets=["model.lr_multiplier=null"])
arm("EXP-002", "paper_alpha01", "ratio loss α=0.1", H2, "EXP-002/paper", run_sets=["model.dc.ratio_loss_alpha=0.1"])
# EXP-003 grid
for arch, base, rs, lab in (("bpe", BPE, [], "BPE baseline"), ("paper", H2, [], "paper router (LH-Net)"),
                            ("end_mlp", H2, ["model.dc.router=end_mlp"], "end-of-chunk, MLP"),
                            ("end_cosine_shift", H2, ["model.dc.router=end_cosine_shift"], "end-of-chunk, cosine")):
    for lr in ("0.0005", "0.001", "0.0015", "0.003", "0.006"):
        arm("EXP-003", f"{arch}__lr{lr}", f"{lab}, lr {lr}", base, f"EXP-002/{arch}", run_sets=rs + [f"train.lr={lr}"])
# EXP-004 (planned)
for arch, base, rs, lab in (("bpe", BPE, [], "BPE baseline"), ("paper", H2, [], "paper router (LH-Net)"),
                            ("end_mlp", H2, ["model.dc.router=end_mlp"], "end-of-chunk, MLP"),
                            ("end_cosine_shift", H2, ["model.dc.router=end_cosine_shift"], "end-of-chunk, cosine")):
    arm("EXP-004", arch, f"{lab}, Muon (grid mlr 0.0125 / 0.025 / 0.05; never launched)", base, f"EXP-003/{arch}__lr0.0015" if arch != "bpe" else "EXP-003/bpe__lr0.006",
        notes="Adam-part LR = best EXP-003 LR (never filled in)", spec_sets=["train.optimizer=muon", "train.muon.adam_betas=[0.9, 0.95]", "train.muon.adam_weight_decay=0.1"], run_sets=rs)
# EXP-005 (planned)
for fam, base in (("bpe", BPE), ("hnet", H2)):
    for var in ("attn", "mla", "nsa", "dsa", "kda", "kda_hybrid", "moba", "gated_attn"):
        mf = {"gated_attn": "gated_headwise" if fam == "bpe" else "gated"}.get(var, var)
        model = f"configs/models/tiny_{'bpe' if fam=='bpe' else 'hnet2'}_2m{'' if var=='attn' else '_'+mf}.yaml"
        arm("EXP-005", f"{fam}_{var}", f"{'BPE' if fam=='bpe' else 'H-Net (best router)'} with {var} mixer (never launched)", base,
            f"EXP-002/{'bpe' if fam=='bpe' else 'paper'}", notes=f"model file {model}" + (" (does not exist)" if (fam, var) == ("bpe", "kda_hybrid") else ""),
            run_sets=[f"mixer={var}"] if var != "attn" else [])
# EXP-006
arm("EXP-006", "bpe", "BPE baseline", BPE, "EXP-002/bpe", notes="THE reference baseline: 1.634 ± 0.002 (3 seeds)")
arm("EXP-006", "paper", "2-stage, paper router", H2, "EXP-002/paper")
arm("EXP-006", "paper_ng", "2-stage + n-gram emb", H2, "EXP-006/paper", run_sets=[NG])
arm("EXP-006", "end_mlp_ng", "2-stage end-of-chunk MLP + n-gram emb", H2, "EXP-002/end_mlp", run_sets=["model.dc.router=end_mlp", NG])
arm("EXP-006", "hnet1", "1-stage, paper router (ratio 6)", H1)
arm("EXP-006", "hnet1_ng", "1-stage + n-gram emb", H1, "EXP-006/hnet1", run_sets=[NG1])
# EXP-008
S8 = ["eval.zero_main_probe=true"]
arm("EXP-008", "base", "paper 2-stage", H2, "EXP-002/paper", notes="3-seed re-run of EXP-002 paper with the zero-main probe")
arm("EXP-008", "rdrop", "residual path-drop 0.3", H2, "EXP-008/base", run_sets=["model.dc.residual_dropout=0.3"])
arm("EXP-008", "mainaux", "main-path aux head 0.3", H2, "EXP-008/base", run_sets=["model.dc.main_aux_weight=0.3"],
    notes="stage-0 compression collapsed to 1.25 bytes/position (far over the FLOPs budget)")
arm("EXP-008", "rent", "router entropy 0.05", H2, "EXP-008/base", run_sets=["model.dc.router_entropy_weight=0.05"])
arm("EXP-008", "combo", "path-drop + aux head + entropy penalty (paper router); dropped", H2, "EXP-008/base",
    run_sets=["model.dc.residual_dropout=0.3", "model.dc.main_aux_weight=0.3", "model.dc.router_entropy_weight=0.05"],
    notes="Dropped from the spec: routers collapsed to select-all, ran out of memory")
arm("EXP-008", "combo_end", "path-drop + aux head + entropy penalty, end_mlp router; dropped", H2, "EXP-008/combo",
    run_sets=["model.dc.router=end_mlp", "model.dc.residual_dropout=0.3", "model.dc.main_aux_weight=0.3", "model.dc.router_entropy_weight=0.05"],
    notes="Dropped from the spec: 1.652 at +379% FLOPs (compression 1.15 / 1.54), invalid")
# EXP-009
arm("EXP-009", "space_start", "word-start chunks (oracle)", H1, "EXP-006/hnet1", run_sets=["model.dc.router=space_like"])
arm("EXP-009", "space_end", "word-end chunks (oracle)", H1, "EXP-006/hnet1", run_sets=["model.dc.router=space_like", "model.dc.router_kwargs={semantics: end}"])
arm("EXP-009", "fixed6", "fixed 6-byte chunks", H1, "EXP-006/hnet1", run_sets=["model.dc.router=fixed_pool"])
# EXP-010
arm("EXP-010", "alpha1", "strong ratio loss (alpha 1.0)", H2, "EXP-008/base", run_sets=["model.dc.ratio_loss_alpha=1.0"])
arm("EXP-010", "resid_id", "identity-init residual", H2, "EXP-008/base", run_sets=["model.dc.residual_init=identity"])
arm("EXP-010", "alpha1_resid_id", "alpha 1.0 + identity residual", H2, "EXP-008/base", run_sets=["model.dc.ratio_loss_alpha=1.0", "model.dc.residual_init=identity"])
arm("EXP-010", "swa", "outer Mamba -> sliding-window attention", H2, "EXP-008/base", run_sets=["model.mixer_codes={m: attn, t: attn}", "model.attn.window_size=[256, 384, -1]"])
arm("EXP-010", "kda", "outer Mamba -> Kimi Delta Attention", H2, "EXP-008/base", run_sets=["model.mixer_codes={m: kda, t: attn}", "model.mixer_cfg={kda: [{expand_v: 1.5}, {expand_v: 1.25}, {}]}"],
    notes="first launch failed on a causal_conv1d channel-count constraint; no eval records")
# EXP-011
S11 = ["model.dc.router=fixed_pool"]
arm("EXP-011", "fixed", "fixed (3,3) pooling", H2, "EXP-008/base", spec_sets=S11)
arm("EXP-011", "fixed_rwarm", "+ residual warmup (first 20%)", H2, "EXP-011/fixed", spec_sets=S11, run_sets=["model.dc.residual_warmup=0.2"])
arm("EXP-011", "fixed_rid", "+ identity-init residual", H2, "EXP-011/fixed", spec_sets=S11, run_sets=["model.dc.residual_init=identity"])
arm("EXP-011", "fixed_rdrop", "+ residual path-drop 0.3", H2, "EXP-011/fixed", spec_sets=S11, run_sets=["model.dc.residual_dropout=0.3"])
arm("EXP-011", "fixed_aux", "+ main-path aux head 0.3", H2, "EXP-011/fixed", spec_sets=S11, run_sets=["model.dc.main_aux_weight=0.3"])
# EXP-012
S12 = ["model.dc.router=end_mlp"]
arm("EXP-012", "select", "end-of-chunk MLP router, select chunker", H2, "EXP-002/end_mlp", spec_sets=S12, notes="= EXP-002 end_mlp re-run with 3 seeds")
arm("EXP-012", "pool_mean", "+ mean-pool chunk vectors", H2, "EXP-012/select", spec_sets=S12, run_sets=["model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: mean}"])
arm("EXP-012", "pool_attn", "+ attention-pool chunk vectors", H2, "EXP-012/select", spec_sets=S12, run_sets=["model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: attn}"])
arm("EXP-012", "pool_attn_rwarm", "+ attention pool + residual warmup 20%", H2, "EXP-012/pool_attn", spec_sets=S12,
    run_sets=["model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: attn}", "model.dc.residual_warmup=0.2"])
# EXP-013
S13 = ["train.lr=6.0e-3", "model.dc.router=end_mlp"]
arm("EXP-013", "lr6", "lr 6e-3, paper multipliers", H2, "EXP-003/end_mlp__lr0.006", spec_sets=S13)
arm("EXP-013", "lr6_alpha1", "lr 6e-3 + strong ratio loss (alpha 1)", H2, "EXP-013/lr6", spec_sets=S13, run_sets=["model.dc.ratio_loss_alpha=1.0"])
arm("EXP-013", "lr6_rent", "lr 6e-3 + router entropy 0.05", H2, "EXP-013/lr6", spec_sets=S13, run_sets=["model.dc.router_entropy_weight=0.05"])
arm("EXP-013", "lr6_nomult", "lr 6e-3, no multipliers", H2, "EXP-013/lr6", spec_sets=S13, run_sets=["model.lr_multiplier=null"])
arm("EXP-013", "lr6_mainheavy", "lr 6e-3, multipliers [1, 1, 4]", H2, "EXP-013/lr6", spec_sets=S13, run_sets=["model.lr_multiplier=[1, 1, 4]"])
# EXP-014 (p-space quantile, voided)
S14 = Q + ["model.dc.decision_version=p-space quantile (saturating; replaced)"]
for lr in ("0.0015", "0.006"):
    arm("EXP-014", f"quant__lr{lr}", f"quantile decisions (p-space), lr {lr}", H2, "EXP-012/select", spec_sets=S14, run_sets=[f"train.lr={lr}"])
    arm("EXP-014", f"quant_pool__lr{lr}", f"quantile decisions + attention-pool chunk vectors, lr {lr}", H2, f"EXP-014/quant__lr{lr}", spec_sets=S14,
        run_sets=["model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: attn}", f"train.lr={lr}"],
        notes="never started" if lr == "0.0015" else "")
# EXP-015 / 016 (never launched)
arm("EXP-015", "q_rdrop", "quantile + residual path-drop 0.3", H2, "EXP-014/quant__lr0.0015", spec_sets=Q, run_sets=["model.dc.residual_dropout=0.3"])
arm("EXP-015", "q_aux", "quantile + main-path aux head 0.3", H2, "EXP-014/quant__lr0.0015", spec_sets=Q, run_sets=["model.dc.main_aux_weight=0.3"])
arm("EXP-015", "q_rdrop_aux", "quantile + path-drop + aux head", H2, "EXP-014/quant__lr0.0015", spec_sets=Q, run_sets=["model.dc.residual_dropout=0.3", "model.dc.main_aux_weight=0.3"])
arm("EXP-015", "q_rdrop_aux_pool", "quantile + path-drop + aux head + mean-pool chunk vectors", H2, "EXP-014/quant__lr0.0015", spec_sets=Q,
    run_sets=["model.dc.residual_dropout=0.3", "model.dc.main_aux_weight=0.3", "model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: mean}"])
arm("EXP-016", "cp4", "chunk-pred k=4, w=0.5", H2, "EXP-014/quant__lr0.0015", spec_sets=Q, run_sets=["model.dc.chunk_pred_k=4", "model.dc.chunk_pred_weight=0.5"])
arm("EXP-016", "cp4_w1", "chunk-pred k=4, w=1.0", H2, "EXP-014/quant__lr0.0015", spec_sets=Q, run_sets=["model.dc.chunk_pred_k=4", "model.dc.chunk_pred_weight=1.0"])
arm("EXP-016", "cp4_pool", "chunk-pred k=4 + mean-pool chunk vectors", H2, "EXP-014/quant__lr0.0015", spec_sets=Q,
    run_sets=["model.dc.chunk_pred_k=4", "model.dc.chunk_pred_weight=0.5", "model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: mean}"])
arm("EXP-016", "cp8_pool", "chunk-pred k=8 + mean-pool chunk vectors", H2, "EXP-014/quant__lr0.0015", spec_sets=Q,
    run_sets=["model.dc.chunk_pred_k=8", "model.dc.chunk_pred_weight=0.5", "model.dc.chunker=pool", "model.dc.chunker_kwargs={mode: mean}"])
# EXP-017
Q17 = Q + ["model.dc.decision_version=rank of batch-normalized fp32 logits (mean/std)"]
arm("EXP-017", "quant_fix", "rank decisions (fixed), no vocabulary", H2, "EXP-014/quant__lr0.0015", spec_sets=Q17, notes="honest-compute 2-stage baseline 1.799 ± 0.005")
arm("EXP-017", "vocab", "+ hashed chunk vocabulary (input side)", H2, "EXP-017/quant_fix", spec_sets=Q17, run_sets=[CV2])
arm("EXP-017", "vocab_pred", "+ vocabulary + next-chunk prediction (w 0.5)", H2, "EXP-017/vocab", spec_sets=Q17, run_sets=[CVP2])
arm("EXP-017", "vocab_pred_w1", "+ vocabulary + next-chunk prediction (w 1.0)", H2, "EXP-017/vocab", spec_sets=Q17, run_sets=[CVP2_W1])
# EXP-018
arm("EXP-018", "h1_quant", "1-stage (ratio 6), rank decisions, no vocabulary", H1, "EXP-006/hnet1", spec_sets=Q17)
arm("EXP-018", "h1_vocab_pred", "1-stage + vocabulary + next-chunk prediction", H1, "EXP-018/h1_quant", spec_sets=Q17, run_sets=[CVP1])
# EXP-019
S19 = ["model.dc.router=local_mlp", "model.dc.router_kwargs={k: 8, d_hidden: 64}", "model.dc.decision=quantile", "model.dc.decision_version=rank of batch-normalized fp32 logits (mean/std)"]
arm("EXP-019", "local", "context-free router (last 8 bytes), rank decisions", H2, "EXP-017/quant_fix", spec_sets=S19)
arm("EXP-019", "local_vocab", "+ hashed chunk vocabulary", H2, "EXP-019/local", spec_sets=S19, run_sets=[CV2])
arm("EXP-019", "local_vocab_pred", "+ vocabulary + next-chunk prediction (w 0.5)", H2, "EXP-019/local", spec_sets=S19, run_sets=[CVP2])
# EXP-020
S20 = Q17 + ["model.dc.router_consistency_n=4"]
arm("EXP-020", "cons1", "consistency penalty w 1 (scale-free: unexplained variance share)", H2, "EXP-017/quant_fix", spec_sets=S20, run_sets=["model.dc.router_consistency_weight=1.0"])
arm("EXP-020", "cons10", "consistency penalty w 10", H2, "EXP-017/quant_fix", spec_sets=S20, run_sets=["model.dc.router_consistency_weight=10.0"])
arm("EXP-020", "cons10_vocab_pred", "consistency w 10 + vocabulary + next-chunk prediction", H2, "EXP-017/vocab_pred", spec_sets=S20,
    run_sets=["model.dc.router_consistency_weight=10.0", CVP2])
# EXP-021
Q21 = Q + ["model.dc.decision_version=rank on median/MAD-standardized scores, clipped z (collapsed)"]
arm("EXP-021", "h1r_quant", "1-stage (ratio 6), robust rank decisions", H1, "EXP-018/h1_quant", spec_sets=Q21, notes="identical to h1_quant without a vocabulary")
arm("EXP-021", "h1r_vocab_pred", "+ chunk vocabulary + next-chunk prediction", H1, "EXP-018/h1_vocab_pred", spec_sets=Q21, run_sets=[CVP1])
arm("EXP-021", "h1r_vocab_pred_ng", "+ chunk vocabulary + prediction + byte n-gram tables", H1, "EXP-021/h1r_vocab_pred", spec_sets=Q21, run_sets=[CVP1, NG])
# EXP-022 (superseded draft)
S22 = Q21 + [CVP1]
arm("EXP-022", "rwarm", "vocabulary + prediction, residual warmup 0.3", H1, "EXP-021/h1r_vocab_pred", spec_sets=S22, run_sets=["model.dc.residual_warmup=0.3"])
arm("EXP-022", "temp4", "vocabulary + prediction, confidence temperature 4", H1, "EXP-021/h1r_vocab_pred", spec_sets=S22, run_sets=["model.dc.decision_kwargs={temp: 4.0}"])
arm("EXP-022", "w1", "vocabulary + prediction weight 1.0", H1, "EXP-021/h1r_vocab_pred", spec_sets=S22, run_sets=[CVP1_W1])
# EXP-023
Q23 = Q + ["model.dc.decision_version=rank on trimmed (central 98%) mean/std-standardized scores", CVP1]
arm("EXP-023", "vp", "vocabulary + next-chunk prediction (trimmed decider)", H1, "EXP-018/h1_vocab_pred", spec_sets=Q23)
arm("EXP-023", "vp_ng", "+ byte n-gram tables", H1, "EXP-023/vp", spec_sets=Q23, run_sets=[NG], notes="prior-free best configuration; base of EXP-027/028")
arm("EXP-023", "vp_rwarm", "+ residual warmup 0.3", H1, "EXP-023/vp", spec_sets=Q23, run_sets=["model.dc.residual_warmup=0.3"])
# EXP-024
Q24 = Q23 + [NG]
arm("EXP-024", "prior", "vp_ng + whitespace curriculum (10% of steps)", H1, "EXP-023/vp_ng", spec_sets=Q24, run_sets=["model.dc.router_prior_weight=1.0", "model.dc.router_prior_frac=0.1"],
    notes="best H-Net: matches BPE at equal FLOPs")
arm("EXP-024", "temp4", "vp_ng + confidence temperature 4", H1, "EXP-023/vp_ng", spec_sets=Q24, run_sets=["model.dc.decision_kwargs={temp: 4.0}"])
# EXP-025
S25 = ["train.optimizer=muon", "train.muon.adam_betas=[0.9, 0.95]", "train.muon.adam_weight_decay=0.1"]
for mlr in ("0.025", "0.05"):
    arm("EXP-025", f"bpe_muon__mlr{mlr}", f"BPE, Muon lr {mlr} (Adam part 6e-3)", BPE, "EXP-006/bpe", spec_sets=S25,
        run_sets=["train.muon.lr_embed=6.0e-3", f"train.muon.lr={mlr}"])
    arm("EXP-025", f"prior_muon__mlr{mlr}", f"H-Net (EXP-024 prior), Muon lr {mlr} (Adam part 1.5e-3)", H1, "EXP-024/prior", spec_sets=S25,
        run_sets=["train.muon.lr_embed=1.5e-3"] + Q24 + ["model.dc.router_prior_weight=1.0", "model.dc.router_prior_frac=0.1", f"train.muon.lr={mlr}"])
# EXP-026
S26 = Q24 + ["model.dc.router_prior_weight=1.0", "model.dc.router_prior_frac=0.1"]
arm("EXP-026", "swa", "outer Mamba-2 -> sliding-window attention (256)", H1, "EXP-024/prior", spec_sets=S26, run_sets=["model.mixer_codes={m: attn, t: attn}", "model.attn.window_size=[256, -1]"])
arm("EXP-026", "kda", "outer Mamba-2 -> Kimi Delta Attention", H1, "EXP-024/prior", spec_sets=S26, run_sets=["model.mixer_codes={m: kda, t: attn}", "model.mixer_cfg={kda: [{expand_v: 1.5}, {}]}"],
    notes="failed rc=1 in all 3 seeds (sweep_status); second KDA failure after EXP-010")
# EXP-027
arm("EXP-027", "cste", "router trained by the next-chunk loss (chunk STE)", H1, "EXP-023/vp_ng", spec_sets=Q24, run_sets=["model.dc.chunk_ste=true"])
arm("EXP-027", "rgate", "bypass gated by router margin", H1, "EXP-023/vp_ng", spec_sets=Q24, run_sets=["model.dc.residual_gate=margin"])
arm("EXP-027", "cste_rgate", "both", H1, "EXP-023/vp_ng", spec_sets=Q24, run_sets=["model.dc.chunk_ste=true", "model.dc.residual_gate=margin"])
# EXP-028
S28 = Q24 + ["model.dc.router_gain_weight=1.0"]
arm("EXP-028", "gain", "router trained by REINFORCE on the hierarchy's gain", H1, "EXP-023/vp_ng", spec_sets=S28)
arm("EXP-028", "gain_rgate", "+ bypass gated by router margin", H1, "EXP-027/rgate", spec_sets=S28, run_sets=["model.dc.residual_gate=margin"])
arm("EXP-028", "gain_cste", "+ chunk STE", H1, "EXP-027/cste", spec_sets=S28, run_sets=["model.dc.chunk_ste=true"])
# Ladder
LADDER_SETS = ["train.optimizer=muon", "train.muon.lr_embed=0.003"]
for exp_id, mlrs in (("LADDER-LR-1e16", ("0.01", "0.02", "0.04", "0.08")), ("LADDER-LREXT-1e16", ("0.0025", "0.005"))):
    for kind, lab in (("bpe", "BPE L04"), ("hnet", "H-Net L04 (prior-free)"), ("hnet_ws", "H-Net L04 + whitespace warm start")):
        for mlr in mlrs:
            cfg = "configs/ladder/runs/ladder_bpe.yaml" if kind == "bpe" else "configs/ladder/runs/ladder_hnet.yaml"
            rs = (["data.micro_batch=16", "data.batch_tokens=131072", "train.train_tokens=95420416"] if kind == "bpe"
                  else ["data.micro_batch=8", "data.batch_bytes=544768", "train.train_bytes=396591104"])
            arm(exp_id, f"{kind}__mlr{mlr}", f"{lab}, Muon lr {mlr}", cfg, None if kind == "bpe" else f"{exp_id}/bpe__mlr{mlr}",
                notes=f"model configs/ladder/models/{kind}_L04.yaml", spec_sets=LADDER_SETS + ["train.muon.momentum_warmup_steps=72"],
                run_sets=rs + [f"train.muon.lr={mlr}"], extra=ladder_base(kind, "L04"))
for e in E:
    if not e["exp_id"].startswith("LADDER-1e"):
        continue
    budget, rung = e["exp_id"].split("-")[1], e["exp_id"].split("-")[2]
    spec = os.path.join(REPO, e["spec_file"])
    txt = open(spec).read()
    sets = {k: re.search(rf"{k}=(\S+?)[\],]", txt) for k in ("data.batch_tokens", "train.train_tokens", "data.batch_bytes", "train.train_bytes", "train.muon.momentum_warmup_steps")}
    sets = {k: (m.group(1) if m else None) for k, m in sets.items()}
    for kind, lab in (("bpe", f"BPE {rung}"), ("hnet", f"H-Net {rung} (prior-free)"), ("hnet_ws", f"H-Net {rung} + whitespace warm start")):
        cfg = "configs/ladder/runs/ladder_bpe.yaml" if kind == "bpe" else "configs/ladder/runs/ladder_hnet.yaml"
        rs = ([f"data.batch_tokens={sets['data.batch_tokens']}", f"train.train_tokens={sets['train.train_tokens']}"] if kind == "bpe"
              else [f"data.batch_bytes={sets['data.batch_bytes']}", f"train.train_bytes={sets['train.train_bytes']}"])
        arm(e["exp_id"], kind, lab, cfg, None if kind == "bpe" else f"{e['exp_id']}/bpe",
            notes=f"model configs/ladder/models/{kind}_{rung}.yaml; Muon LR = best_mlr.bpe = 0.02 from LADDER-LR-1e16",
            spec_sets=LADDER_SETS + ["train.muon.lr=0.02", f"train.muon.momentum_warmup_steps={sets['train.muon.momentum_warmup_steps']}"],
            run_sets=rs, extra=ladder_base(kind, rung))

ARM_KEYS = {(a["exp_id"], a["arm"]) for a in ARMS}

# ----------------------------------------------------------------------------------------------------------------
# 3. runs (harvested)
# ----------------------------------------------------------------------------------------------------------------
RUN_ROOTS = {  # exp_id -> list of candidate roots (first = preferred when equally advanced)
    "EXP-002": ["runs/gcp/exp002_tiny2m"], "EXP-003": ["runs/gcp/exp003_lr_adamw"], "EXP-006": ["runs/gcp/exp006_close_gap_tiny"],
    "EXP-008": ["runs/gcp/exp008_hierarchy_use"], "EXP-009": ["runs/gcp/exp009_oracle_chunks"],
    "EXP-010": ["runs/gcp/exp010_dynamics_and_mixers"], "EXP-011": ["runs/gcp/exp011_fixed_hourglass"],
    "EXP-012": ["runs/gcp/exp012_pool_chunker"], "EXP-013": ["runs/gcp/exp013_lr_allocation"],
    "EXP-014": ["runs/gcp/exp014_quantile_decision"], "EXP-017": ["runs/gcp/exp017_chunk_vocab"],
    "EXP-018": ["runs/gcp/exp018_vocab_shapes"], "EXP-019": ["runs/gcp/exp019_local_router"],
    "EXP-020": ["runs/gcp/exp020_router_consistency"], "EXP-023": ["runs/gcp/exp023_word_vocab_trimmed"],
    "EXP-024": ["runs/gcp/exp024_adoption_curriculum"],
    "EXP-025": [LIVE + "/exp025_muon"], "EXP-026": [LIVE + "/exp026_mamba_free"], "EXP-027": [LIVE + "/exp027_adoption_mechanism"],
    "LADDER-LR-1e16": [LIVE + "/ladder_lr_1e16", "runs/gcp/hnet-ladder-lr/search/ladder_lr_1e16", "runs/gcp/hnet-ladder-lr2/search/ladder_lr_1e16"],
    "LADDER-LREXT-1e16": [LIVE + "/lrext_1e16", "runs/gcp/hnet-ladder-lr2/search/lrext_1e16"],
    "LADDER-1e16-L03": [LIVE + "/ladder_1e16_L03", "runs/gcp/hnet-ladder-1e16/search/ladder/1e16/L03"],
    "LADDER-1e16-L04": [LIVE + "/ladder_1e16_L04", "runs/gcp/hnet-ladder-1e16/search/ladder/1e16/L04"],
    "LADDER-1e16-L05": [LIVE + "/ladder_1e16_L05", "runs/gcp/hnet-ladder-1e16/search/ladder/1e16/L05"],
    "LADDER-1e16-L06": [LIVE + "/ladder_1e16_L06", "runs/gcp/hnet-ladder-1e16/search/ladder/1e16/L06"],
}
STEPS_TOTAL = {"configs/runs/tiny_bpe_2m.yaml": 1744, "configs/runs/tiny_hnet2_2m.yaml": 1763, "configs/runs/tiny_hnet1_2m.yaml": 1763,
               "configs/runs/tiny_hnet2_2m_ratios42.yaml": 1763}
for e in E:
    if e["exp_id"].startswith("LADDER"):
        steps = re.match(r"(\d+) steps", e["notes"])
        if steps:
            STEPS_TOTAL[e["exp_id"]] = int(steps.group(1))
STEPS_TOTAL["LADDER-LR-1e16"] = STEPS_TOTAL["LADDER-LREXT-1e16"] = 728


def steps_total_for(exp_id, base):
    return STEPS_TOTAL.get(exp_id) or STEPS_TOTAL.get(base)


def last_with(path, key):
    last, n = None, 0
    if not os.path.exists(path):
        return None, 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n += 1
            if r.get(key) is not None:
                last = r
    return last, n


def sweep_status(root):
    st = {}
    for p in glob.glob(os.path.join(root, "sweep_status*.json")) + glob.glob(os.path.join(root, "queue_status.json")):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        for k, v in d.get("runs", {}).items():
            st[k] = v if isinstance(v, str) else v.get("status")
    return st


# manual annotations: (exp_id, run) -> dict(adopted=, tokenizer_kind=, notes=)
ANN = {
    ("EXP-018", "h1_vocab_pred_s0"): dict(adopted="yes", tokenizer_kind="word", notes="chunk_stats: 97.7% boundaries on word boundaries, 99.9% consistency, top chunks 'the ', 'of ', 'and '"),
    ("EXP-018", "h1_vocab_pred_s1"): dict(adopted="no", tokenizer_kind="suffix", notes="heavy-tailed router scores (std 13-15), confidences ~0.5, residual path won; suffix/morpheme tokenizer ('ing', 'ed', 'ly', 'ion')"),
    ("EXP-018", "h1_vocab_pred_s2"): dict(adopted="yes", notes="main contribution 0.55"),
    ("EXP-018", "h1_quant_s0"): dict(tokenizer_kind="mixed", notes="chunk_stats: 84% of boundaries on word boundaries, 32% singletons, top chunks 'of ', 'and ', ' ', '(', 'fro'"),
    ("EXP-023", "vp_s0"): dict(adopted="yes", tokenizer_kind="word", notes="93% of cuts on whitespace by step 500 (cs_exp023_vp_s0_step*)"),
    ("EXP-023", "vp_s1"): dict(adopted="no", tokenizer_kind="suffix", notes="same seed as EXP-018 h1_vocab_pred_s1: suffix tokenizer, 4% boundaries on word boundaries, main contribution 0.18"),
    ("EXP-023", "vp_s2"): dict(adopted="yes"),
    ("EXP-023", "vp_ng_s0"): dict(adopted="no", notes="the seed that 'stays at 1.72' (main contribution 0.09). NOTE: the EXP-027 entry/spec say 'seed 1 not adopted' for vp_ng; the eval numbers show seed 0 is the non-adopted one"),
    ("EXP-023", "vp_ng_s1"): dict(adopted="yes"),
    ("EXP-023", "vp_ng_s2"): dict(adopted="yes", tokenizer_kind="word", notes="chunk_stats: 95% boundaries on word boundaries, 99.4% consistency"),
    ("EXP-023", "vp_rwarm_s0"): dict(adopted="yes", notes="contribution 0.32; worse than vp"),
    ("EXP-023", "vp_rwarm_s1"): dict(adopted="no", notes="'seed 1 still not adopted' (EXPERIMENTS.md)"),
    ("EXP-023", "vp_rwarm_s2"): dict(adopted="yes"),
    ("EXP-024", "prior_s0"): dict(adopted="yes", tokenizer_kind="word", notes="chunk_stats after 90% prior-free training: 97% boundaries on word boundaries, 99.3% hit, 99.8% consistency, 25% singletons"),
    ("EXP-024", "prior_s1"): dict(adopted="yes", notes="adopted by step 250"),
    ("EXP-024", "prior_s2"): dict(adopted="yes", notes="adopted by step 250"),
    ("EXP-024", "temp4_s0"): dict(adopted="no", notes="hierarchy share 0.02"),
    ("EXP-003", "end_mlp__lr0.0015_s0"): dict(tokenizer_kind="mixed", notes="chunk_stats: stage 0 = character-pair tokenizer (3% of boundaries on word boundaries), stage 1 not a tokenizer (55% singletons, 30% consistency)"),
    ("EXP-008", "base_s0"): dict(tokenizer_kind="mixed", notes="chunk_stats: stage 0 character pairs (17% on word boundaries), stage 1 48% singletons, 53% consistency"),
    ("EXP-013", "lr6_nomult_s0"): dict(tokenizer_kind="mixed", notes="chunk_stats: stage 0 character pairs (19% on word boundaries), stage 1 52% singletons, 23% consistency"),
    ("EXP-008", "combo_end_s0"): dict(notes="invalid: routers collapsed to select-all, compression 1.15 / 1.54, +379% FLOPs"),
    ("EXP-008", "combo_end_s2"): dict(notes="invalid: routers collapsed to select-all"),
    ("EXP-008", "combo_s0"): dict(notes="ran out of memory (stage-1 router selected everything)"),
    ("EXP-008", "combo_s1"): dict(notes="ran out of memory"),
    ("EXP-008", "combo_s2"): dict(notes="ran out of memory"),
    ("EXP-008", "rdrop_s0"): dict(notes="realized compute +31% (compression 2.1 / 2.7)"),
    ("EXP-010", "swa_s0"): dict(notes="stopped at step 250 (2.86). The EXP-026 spec cites 'EXP-010 SWA ended at 2.005', which no record supports"),
    ("EXP-010", "kda_s0"): dict(notes="crashed before the first eval (causal_conv1d channel-count constraint)"),
    ("EXP-010", "kda_s1"): dict(notes="crashed before the first eval"),
    ("EXP-011", "fixed_rdrop_s2"): dict(notes="failed at step 50, no eval"),
    ("EXP-012", "pool_attn_s2"): dict(notes="crashed at step 20 (bf16 dtype mismatch under autocast, fixed later)"),
    ("EXP-014", "quant__lr0.0015_s1"): dict(notes="stage-1 collapsed: compression/s1 529"),
    ("EXP-014", "quant__lr0.0015_s2"): dict(notes="stage-0 collapsed at step ~290: compression/s0 1628, stage 1 selects everything"),
    ("EXP-014", "quant__lr0.006_s0"): dict(notes="stage-1 collapsed: compression/s1 426"),
    ("EXP-014", "quant__lr0.006_s1"): dict(notes="stage-0 collapsed: compression/s0 1590"),
    ("EXP-014", "quant__lr0.006_s2"): dict(notes="stage-0 collapsed: compression/s0 1628"),
    ("EXP-014", "quant_pool__lr0.006_s0"): dict(notes="stage-1 collapsed: compression/s1 530"),
    ("EXP-014", "quant_pool__lr0.006_s1"): dict(notes="stage-1 collapsed: compression/s1 427"),
    ("EXP-014", "quant_pool__lr0.006_s2"): dict(notes="the one healthy-looking run (tau/s0 = 1.000 too)"),
    ("EXP-006", "hnet1_ng_s0"): dict(notes="stopped at step 1680 (1.7376 at the step-1500 eval); the worklog's '1.722 ± 0.023 (2 seeds)' used this unfinished run"),
    ("LADDER-1e16-L04", "bpe_s0"): dict(notes="identical records to LADDER-LR-1e16 bpe__mlr0.02_s0 (the LR-sweep run reused as the rung's BPE run)"),
    ("LADDER-LREXT-1e16", "hnet__mlr0.0025_s0"): dict(notes="router undecided: p_margin/s0 0.03"),
    ("LADDER-LREXT-1e16", "hnet_ws__mlr0.0025_s0"): dict(notes="p_margin/s0 0.07"),
    ("EXP-027", "cste_s0"): dict(notes="interim; step 500 val 2.022 (worklog)"),
    ("EXP-026", "kda_s0"): dict(notes="one eval record (step 250) before the failure"),
    ("EXP-027", "rgate_s0"): dict(notes="interim; margin gate makes the hierarchy load-bearing from the start (0.42 bpb contribution at step 500)"),
}

RUNS = []
for exp_id, roots in RUN_ROOTS.items():
    exp_status = next(e["status"] for e in E if e["exp_id"] == exp_id)
    found = defaultdict(list)  # run name -> [(root, dir)]
    stat = {}
    for root in roots:
        absroot = root if root.startswith("/") else os.path.join(REPO, root)
        stat.update(sweep_status(absroot))
        for d in sorted(glob.glob(os.path.join(absroot, "*"))):
            name = os.path.basename(d)
            if not os.path.isdir(d) or name.startswith("_") or not re.search(r"_s\d+$", name):
                continue
            found[name].append((root, d))
    for name, cands in sorted(found.items()):
        # pick the most advanced copy (max eval step, then done.json)
        best = None
        for root, d in cands:
            ev, nev = last_with(os.path.join(d, "eval.jsonl"), "val/bpb")
            score = ((ev or {}).get("step") or -1, os.path.exists(os.path.join(d, "done.json")))
            if best is None or score > best[0]:
                best = (score, root, d, ev)
        _, root, d, ev = best
        others = [r for r, _ in cands if r != root]
        arm_name, seed = re.match(r"(.*)_s(\d+)$", name).groups()
        if (exp_id, arm_name) not in ARM_KEYS:
            raise SystemExit(f"run {exp_id}/{name}: arm {arm_name} not declared")
        base = next(a["base_config"] for a in ARMS if a["exp_id"] == exp_id and a["arm"] == arm_name)
        mt, _ = last_with(os.path.join(d, "metrics.jsonl"), "step")
        done = os.path.exists(os.path.join(d, "done.json"))
        rc = None
        if done:
            try:
                rc = json.load(open(os.path.join(d, "done.json"))).get("rc")
            except Exception:
                rc = "?"
        diverged = os.path.exists(os.path.join(d, "diverged.json"))
        total = steps_total_for(exp_id, base)
        ev_step = ev["step"] if ev else None
        m_step = mt["step"] if mt else None
        steps_done = max([s for s in (ev_step, m_step) if s is not None], default=0)
        notes = []
        a_notes = ANN.get((exp_id, name), {}).get("notes", "")
        sw = stat.get(name)
        if exp_id == "EXP-014":
            status = "voided"
        elif exp_id == "EXP-008" and arm_name in ("combo", "combo_end"):
            status = "voided"
        elif diverged:
            status = "failed"
        elif done and rc == 0:
            status = "done"
        elif sw and sw.startswith("failed"):
            status = "failed"
            notes.append(f"sweep_status: {sw}")
        elif total and ev_step == total:
            status = "done"
            notes.append("final step reached; no done.json in the mirror")
        elif ev is None and mt is None:
            status = "failed" if "crash" in a_notes else "stopped"
            notes.append("empty run dir (no metrics.jsonl / eval.jsonl)")
        elif exp_status == "running":
            status = "running"
        elif steps_done == 0 or (ev is None and (m_step or 0) < 300):
            status = "failed"
            notes.append(f"no eval record; last metrics step {m_step}")
        else:
            status = "stopped"
            notes.append(f"unfinished: last eval at step {ev_step}")
        if others:
            notes.append("also mirrored at " + "; ".join(others))
        zm = ev.get("val/zero_main_s0_bpb") if ev else None
        val = ev["val/bpb"] if ev else None
        a = ANN.get((exp_id, name), {})
        if a.get("notes"):
            notes.append(a["notes"])
        RUNS.append(dict(exp_id=exp_id, arm=arm_name, seed=int(seed), run_dir=os.path.join(root, name), status=status,
                         steps_done=steps_done, steps_total=total or NULL, final_step=ev_step if ev_step is not None else NULL,
                         val_bpb_final=r4(val), zero_main_bpb_final=r4(zm),
                         main_contribution=r4(zm - val) if (zm is not None and val is not None) else NULL,
                         train_bpb_final=r4(mt.get("bpb")) if mt and mt.get("bpb") is not None else NULL,
                         adopted=a.get("adopted", NULL), tokenizer_kind=a.get("tokenizer_kind", NULL),
                         source="eval_jsonl", notes="; ".join(notes)))

# worklog-only runs (no run dirs): EXP-021 at step ~1000
for arm_name, seed, val, status, note in (
        ("h1r_vocab_pred", 0, 3.25, "failed", "diverged: MAD -> 0, grad norm 0.3 -> 4.6e3 (step 900) -> 1.9e6 (step 1050); val bpb 3.25 at step 1000"),
        ("h1r_vocab_pred", 1, 1.98, "stopped", "did not adopt the hierarchy (≈ baseline at step 1000)"),
        ("h1r_vocab_pred", 2, 2.00, "stopped", "did not adopt the hierarchy (≈ baseline at step 1000)"),
        ("h1r_vocab_pred_ng", None, 1.82, "stopped", "1.82 at step 1000 (= paper_ng level) before the stop; seed not stated"),
        ("h1r_quant", None, None, "stopped", "≡ h1_quant (the decider change is invisible without a vocabulary); no number recorded")):
    RUNS.append(dict(exp_id="EXP-021", arm=arm_name, seed=seed if seed is not None else NULL, run_dir=NULL, status=status,
                     steps_done=1000, steps_total=1763, final_step=1000, val_bpb_final=r4(val), zero_main_bpb_final=NULL,
                     main_contribution=NULL, train_bpb_final=NULL, adopted="no" if arm_name == "h1r_vocab_pred" and seed else NULL,
                     tokenizer_kind=NULL, source="worklog", notes=note))

# ladder 1e16 rungs: status and outcome from the harvested runs (these sweeps move while the logbook is written)
for e in E:
    if not e["exp_id"].startswith("LADDER-1e16-"):
        continue
    rr = {r["arm"]: r for r in RUNS if r["exp_id"] == e["exp_id"]}
    if not rr:
        continue
    parts, n_done = [], 0
    for kind in ("bpe", "hnet", "hnet_ws"):
        r = rr.get(kind)
        if r is None:
            parts.append(f"{kind} not started")
            continue
        n_done += r["status"] == "done"
        parts.append(f"{kind} {r['val_bpb_final']}" + ("" if r["status"] == "done" else f" (step {r['final_step']}/{r['steps_total']}, {r['status']})"))
    e["status"] = "done" if n_done == 3 else "running"
    extra = ""
    if n_done == 3 and all(rr[k]["val_bpb_final"] != NULL for k in rr):
        b = float(rr["bpe"]["val_bpb_final"])
        extra = f": H-Net trails BPE by {float(rr['hnet']['val_bpb_final'])-b:.2f} prior-free, {float(rr['hnet_ws']['val_bpb_final'])-b:.2f} with the warm start, at BPE's Muon LR 0.02"
    e["outcome"] = ", ".join(parts) + extra + ("; no done.json in the mirror, final step reached" if n_done == 3 else "") + "."

# ----------------------------------------------------------------------------------------------------------------
# 4. arm_results, references
# ----------------------------------------------------------------------------------------------------------------
TINY_EXPS = {e["exp_id"] for e in E if e["scale"] == "tiny_2m"}
VERDICT = {  # manual verdicts (exp, arm) -> phrase; others computed
    ("EXP-002", "bpe"): "baseline", ("EXP-006", "bpe"): "reference baseline", ("EXP-003", "bpe__lr0.006"): "tuned BPE best (grid edge)",
    ("EXP-002", "paper"): "trails BPE 0.15; over budget +7%", ("EXP-002", "end_mlp"): "trails BPE; learns word-like chunks",
    ("EXP-006", "paper_ng"): "n-gram tables close ~40% of gap", ("EXP-006", "hnet1_ng"): "n-gram tables close ~40% of gap",
    ("EXP-006", "end_mlp_ng"): "n-gram tables help", ("EXP-006", "hnet1"): "1-stage learned-router baseline",
    ("EXP-008", "base"): "paper baseline, +11% compute", ("EXP-008", "rdrop"): "harmful; +31% compute",
    ("EXP-008", "mainaux"): "invalid: compression collapsed to 1.25", ("EXP-008", "rent"): "harmful; hierarchy dead",
    ("EXP-008", "combo"): "invalid (OOM, select-all)", ("EXP-008", "combo_end"): "invalid (+379% FLOPs)",
    ("EXP-009", "space_end"): "oracle: only 0.04 better than learned", ("EXP-009", "space_start"): "end > start", ("EXP-009", "fixed6"): "no semantics: worst",
    ("EXP-010", "alpha1"): "holds compression; neutral", ("EXP-010", "resid_id"): "+15.6% compute; hierarchy unused",
    ("EXP-010", "alpha1_resid_id"): "incomplete (1 seed)", ("EXP-010", "swa"): "no result (stopped)", ("EXP-010", "kda"): "crashed",
    ("EXP-011", "fixed"): "router-free baseline", ("EXP-011", "fixed_aux"): "best honest-compute pre-vocabulary (+0.01)",
    ("EXP-011", "fixed_rwarm"): "neutral", ("EXP-011", "fixed_rid"): "harmful; contribution 0.10", ("EXP-011", "fixed_rdrop"): "harmful",
    ("EXP-012", "select"): "incomplete; seed spread 0.10", ("EXP-012", "pool_mean"): "incomplete (1 seed)", ("EXP-012", "pool_attn"): "no result",
    ("EXP-012", "pool_attn_rwarm"): "no result",
    ("EXP-013", "lr6"): "incomplete; worse than lr 1.5e-3", ("EXP-013", "lr6_alpha1"): "harmful", ("EXP-013", "lr6_nomult"): "incomplete (1 seed)",
    ("EXP-013", "lr6_rent"): "incomplete; seed spread 0.13", ("EXP-013", "lr6_mainheavy"): "no result",
    ("EXP-017", "quant_fix"): "honest-compute 2-stage baseline", ("EXP-017", "vocab"): "small gain (-0.015)",
    ("EXP-017", "vocab_pred"): "harmful at (3,3)", ("EXP-017", "vocab_pred_w1"): "harmful at (3,3)",
    ("EXP-018", "h1_quant"): "1-stage honest-compute baseline", ("EXP-018", "h1_vocab_pred"): "seed lottery: -0.13/-0.11 when adopted",
    ("EXP-019", "local"): "neutral vs contextual router", ("EXP-019", "local_vocab"): "neutral; seed spread 0.14", ("EXP-019", "local_vocab_pred"): "harmful at (3,3)",
    ("EXP-020", "cons1"): "neutral, larger spread", ("EXP-020", "cons10"): "neutral, larger spread", ("EXP-020", "cons10_vocab_pred"): "harmful at (3,3)",
    ("EXP-021", "h1r_vocab_pred"): "diverged (MAD collapse)", ("EXP-021", "h1r_vocab_pred_ng"): "stopped", ("EXP-021", "h1r_quant"): "stopped (= h1_quant)",
    ("EXP-023", "vp"): "seed lottery; replicates EXP-018", ("EXP-023", "vp_ng"): "prior-free best; 1 of 3 seeds not adopted",
    ("EXP-023", "vp_rwarm"): "harmful in every seed",
    ("EXP-024", "prior"): "matches BPE", ("EXP-024", "temp4"): "harmful; lowers hierarchy share",
    ("EXP-025", "prior_muon__mlr0.025"): "matches BPE; slightly below AdamW prior", ("EXP-025", "prior_muon__mlr0.05"): "Muon LR too high",
    ("EXP-025", "bpe_muon__mlr0.05"): "worse than AdamW BPE (1.634)", 
    ("EXP-026", "swa"): "harmful (-0.11 vs Mamba-2 outer)", ("EXP-026", "kda"): "failed (rc=1, all seeds)",
    ("LADDER-LR-1e16", "bpe__mlr0.02"): "BPE L04 optimum", ("LADDER-LR-1e16", "hnet__mlr0.01"): "H-Net optimum; trails BPE 0.24",
    ("LADDER-LR-1e16", "hnet_ws__mlr0.01"): "H-Net+ws optimum; trails BPE 0.20",
}


def stats(vals):
    if not vals:
        return None, None, None, None
    m = statistics.mean(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else None
    return m, sd, min(vals), max(vals)


ARM_RESULTS = []
REF_ROWS = []
# references computed from runs
def arm_vals(exp_id, arm_name, statuses=("done",)):
    return [float(r["val_bpb_final"]) for r in RUNS if r["exp_id"] == exp_id and r["arm"] == arm_name and r["status"] in statuses and r["val_bpb_final"] != NULL]


REFS = [  # id, exp, arm, description
    ("REF-BPE-TINY", "EXP-006", "bpe", "BPE baseline at tiny scale: tiny_bpe_2m, AdamW 1.5e-3, 50M token-eq, finewebedu10B, 3 seeds (1.634 ± 0.002). The reference for every tiny-scale H-Net."),
    ("REF-BPE-TINY-EXP002", "EXP-002", "bpe", "Same BPE baseline, EXP-002's 2 seeds (1.635)."),
    ("REF-BPE-TUNED", "EXP-003", "bpe__lr0.006", "BPE at its best AdamW LR 6e-3 (grid edge), tuned-fairness reference (1.564 ± 0.012)."),
    ("REF-PAPER-2STAGE", "EXP-008", "base", "Paper 2-stage (3,3) H-Net as published (alpha 0.03), 1.796 ± 0.001 at +11% realized compute."),
    ("REF-HNET1", "EXP-006", "hnet1", "1-stage ratio-6 H-Net with the paper's learned router, 1.845 ± 0.002."),
    ("REF-FIXED", "EXP-011", "fixed", "Router-free fixed (3,3) hourglass at honest compute, 1.832 ± 0.008."),
    ("REF-QUANT-FIX", "EXP-017", "quant_fix", "2-stage end_mlp with the fixed rank decider, honest-compute 2-stage baseline (1.799 ± 0.005)."),
    ("REF-H1-QUANT", "EXP-018", "h1_quant", "1-stage ratio 6, end_mlp, rank decider, no vocabulary (1.853): baseline of the word-scale line."),
    ("REF-VP-NG", "EXP-023", "vp_ng", "Prior-free best: 1-stage ratio 6 + chunk vocabulary + next-chunk objective + n-gram tables (1.682 mean; 1.66 when adopted). Base of EXP-027/028."),
    ("REF-PRIOR", "EXP-024", "prior", "Best H-Net: vp_ng + 10% whitespace curriculum (1.635 ± 0.001), matches BPE at equal FLOPs. Base of EXP-025/026."),
    ("REF-LADDER-BPE-L04-1e16", "LADDER-LR-1e16", "bpe__mlr0.02", "Ladder BPE L04 at 1e16 FLOPs, Muon LR 0.02 (its optimum): 1.622."),
    ("REF-LADDER-HNET-L04-1e16", "LADDER-LR-1e16", "hnet__mlr0.01", "Ladder H-Net L04 prior-free at its own optimum (Muon 0.01): 1.861."),
    ("REF-LADDER-HNETWS-L04-1e16", "LADDER-LR-1e16", "hnet_ws__mlr0.01", "Ladder H-Net L04 + whitespace warm start at its optimum (Muon 0.01): 1.820."),
]
for e in E:
    if e["exp_id"].startswith("LADDER-1e16-") and any(r["exp_id"] == e["exp_id"] and r["arm"] == "bpe" for r in RUNS):
        rung = e["exp_id"].split("-")[2]
        REFS.append((f"REF-LADDER-BPE-{rung}-1e16", e["exp_id"], "bpe", f"Ladder BPE {rung} rung run at 1e16 FLOPs (Muon LR 0.02 transferred from the L04 tuning sweep); the reference for that rung's H-Net arms."
                     + (" Identical records to LADDER-LR-1e16 bpe__mlr0.02." if rung == "L04" else "")))
for rid, ex, ar, desc in REFS:
    vals = arm_vals(ex, ar)
    m = statistics.mean(vals) if vals else None
    REF_ROWS.append(dict(reference_id=rid, exp_id=ex, arm=ar, val_bpb=r4(m), n_seeds=len(vals), description=desc))

REF_FOR_EXP = {"LADDER-LR-1e16": "REF-LADDER-BPE-L04-1e16", "LADDER-LREXT-1e16": "REF-LADDER-BPE-L04-1e16"}
for rid, ex, ar, _ in REFS:
    if ex.startswith("LADDER-1e16-"):
        REF_FOR_EXP[ex] = rid
REF_VAL = {r["reference_id"]: (float(r["val_bpb"]) if r["val_bpb"] != NULL else None) for r in REF_ROWS}

for a in ARMS:
    ex, ar = a["exp_id"], a["arm"]
    vals = arm_vals(ex, ar)
    m, sd, lo, hi = stats(vals)
    all_runs = [r for r in RUNS if r["exp_id"] == ex and r["arm"] == ar]
    if ex in TINY_EXPS:
        ref = "REF-BPE-TINY"
        dbpe = (m - BPE_REF) if m is not None else None
    else:
        ref = REF_FOR_EXP.get(ex, NULL)
        dbpe = None
    rv = REF_VAL.get(ref)
    dref = (m - rv) if (m is not None and rv is not None) else None
    v = VERDICT.get((ex, ar))
    exp_status = next(e["status"] for e in E if e["exp_id"] == ex)
    n_unfinished = len([r for r in all_runs if r["status"] == "running"])
    if v is None and exp_status == "running" and all_runs and n_unfinished:
        v = f"running: {len(vals)}/{len(all_runs)} started seeds done" + (f" ({', '.join(f'{x:.3f}' for x in sorted(vals))})" if vals else "")
    if v is None:
        if not all_runs:
            v = "not run"
        elif m is None:
            v = "no finished seed"
        elif ex == "EXP-003":
            v = "LR grid point"
        elif ex.startswith("LADDER"):
            if "mlr" in ar:
                v = "LR grid point" if dref is None else f"LR grid point ({dref:+.2f} vs BPE optimum)"
            elif ar == "bpe":
                v = "rung baseline"
            else:
                v = "rung run" if dref is None else f"trails BPE {dref:+.2f} at BPE's LR"
        elif sd is not None and sd > 0.05:
            v = "seed lottery"
        elif dbpe is not None and abs(dbpe) < 0.01:
            v = "matches BPE"
        else:
            v = "trails BPE"
    note = ""
    nf = len([r for r in all_runs if r["status"] in ("running", "stopped")])
    interim = [r for r in all_runs if r["status"] == "running" and r["val_bpb_final"] != NULL]
    if interim:
        note = "interim: " + ", ".join(f"s{r['seed']} {r['val_bpb_final']}@{r['final_step']}" for r in interim)
    elif nf:
        note = f"{nf} run(s) unfinished"
    ARM_RESULTS.append(dict(exp_id=ex, arm=ar, n_seeds=len(vals), val_bpb_mean=r4(m), val_bpb_sd=r4(sd),
                            best_seed_bpb=r4(lo), worst_seed_bpb=r4(hi), delta_vs_bpe=r4(dbpe), reference=ref,
                            delta_vs_reference=r4(dref), verdict=v, notes=note))

# ----------------------------------------------------------------------------------------------------------------
# 5. findings
# ----------------------------------------------------------------------------------------------------------------
FINDINGS = []


def finding(exp_id, date, cat, statement, evidence, consequence, section):
    FINDINGS.append(dict(finding_id=f"F-{len(FINDINGS)+1:03d}", exp_id=exp_id, date=date, category=cat, statement=statement,
                         evidence=evidence, consequence=consequence, worklog_section=section))


D = "2026-10-29"
finding("EXP-000", "2026-10-28", "scaling", "The blog's '>10x fewer tokens' compares a cooled-down H-Net with a stable-phase SmolLM3; stable-vs-stable the architecture effect is 2-3x.",
        "data multiplier 1.7x at 94B tokens, 3.0x at 189B, 11x only after the H-Net cooldown (+2.3 points)", "Pass criteria for every H-Net vs baseline comparison use stable-vs-stable multipliers; cooldown is a separate labelled branch.",
        "21:39 · References read, including rendered charts")
finding("EXP-001", "2026-10-28", "architecture", "The blog's improved router marks causal end-of-chunk boundaries and its stage-1 chunks are whole words with the trailing space.",
        "p >= 0.5 coincides with the last byte of a stage-0 chunk on 99.76% of 327,081 bytes; 92% of English stage-1 chunks end right after a space; 0% word-internal cuts", "Routers declare start/end semantics; end_linear / end_mlp / end_cosine_shift added to the architecture search.",
        "22:30 · Reverse-engineering the blog's router (user: \"the blog architecture differs; we'll have to guess\")")
finding("EXP-002", D, "infra", "Intermittent CUDA illegal memory access came from flex attention's compiled block-mask builder on per-step ragged packed lengths, not from mamba_ssm.",
        "repro with CUDA_LAUNCH_BLOCKING=1: faulting kernel = inductor-compiled create_block_mask at attn.py:98; inner-stage lengths change every step and are rarely a multiple of 128",
        "attn.packed_block_mask (eager, index-clamped, later chunked to avoid the [H,T,T] OOM); regression test tests/test_flex_mask.py", "Root cause of the EXP-002 crashes: flex attention mask builder, not mamba_ssm (commit after 88237a3)")
finding("EXP-002", D, "infra", "Triton's autotuner is not thread-safe: two post-run chunk diagnostics running at once crashed with TypeError.",
        "paper_nolrmult_s0 and paper_nosmooth_s0 finished together on VM c; 'NoneType object is not a mapping'", "diag_lock in scripts/search/sweep.py serializes the diagnostics.",
        "Second failure mode: post-run diagnostics raced in Triton's autotuner (commit 9e9f8c4)")
finding("EXP-002", D, "infra", "On resume MetricsLogger appended, so steps between the last checkpoint and the preemption were logged twice.",
        "found while planning the BPE resume", "Resume truncates metrics.jsonl / eval.jsonl to step <= checkpoint; test_resume_after_crash_drops_records_past_checkpoint.",
        "EXP-002 split across three spot L4s (commits be2fa47, 950576d)")
finding("EXP-002", D, "architecture", "At ~2M params every H-Net trails BPE by 0.15-0.21 bpb and the variants differ by less than the seed spread.",
        "H-Net 1.78-1.85 vs BPE 1.635; seed spread up to 0.05 bpb", "Three seeds from EXP-003 on; EXP-002 cannot rank routers or ablations.",
        "EXP-002 at 10/18; EXP-003 launched (commit df6ad43)")
finding("EXP-002", D, "training-dynamics", "The innermost main network is barely used: the decoders rely on the zero-init residual bypass almost immediately.",
        "zeroing the main network (6.0M of 7.9M params, 38% of FLOPs) costs 0.04 bpb; its gradient norm drops to 1-3% of stage 0's by step 20 in every variant",
        "Hypotheses and fix sweeps EXP-008/011 (training signals for the hierarchy), per-step probe run.", "Deep dive: what the tiny H-Net learns, and why it trails BPE")
finding("EXP-002", D, "router-objective", "The stage-1 paper router never leaves its initialization because the ratio loss gives zero gradient at target and the remaining signals scale with the main network's usefulness.",
        "46-70% of stage-1 p in 0.4-0.6, 2% confident; dL/dG = 0 once F is on target; 13% of stage-1 starts after whitespace (blog: 92%)", "Self-reinforcing equilibrium identified; router confidence logged; later addressed by rank decisions.",
        "Deep dive: what the tiny H-Net learns, and why it trails BPE")
finding("EXP-002", D, "data", "84% of the gap to BPE sits in the letters of 'space+word' tokens: word identity.",
        "token-aligned comparison on identical text: 1.78 bits/byte on the letters after a space vs 0.45 on the space", "The vocabulary line (EXP-017/018/023): give the hierarchy word identity.",
        "Deep dive: what the tiny H-Net learns, and why it trails BPE")
finding("EXP-002", D, "scaling", "Neither model copies in context at ~50M tokens, so the tiny regime is dominated by local statistics and is for mechanisms, not the final verdict.",
        "repeated phrases still cost 2.1-2.4 bits/byte on the 4th-6th repetition for BPE and H-Net alike", "Tiny runs are used to debug mechanisms; the verdict moves to the ladder.",
        "Deep dive: what the tiny H-Net learns, and why it trails BPE")
finding("EXP-002", D, "architecture", "The main network is not a decoder bottleneck: its output is less predictive of the next byte than its own input, and its gradient SNR is normal.",
        "linear probes: 2.62 bits (output) vs 2.08 (input) for paper_s0; SNR 2.4 at step 500 vs 1.9 for stage 0", "Fixes must deliver gradient to the main network early, not widen the decoder.",
        "Linear probes on the main network: it is not a decoder bottleneck (`hnet/analysis/probe_main.py`)")
finding("EXP-002", D, "architecture", "The chunk vector is a boundary state, not a chunk summary: the main network does not know which word it is looking at.",
        "end_mlp: 0.18 bits left about the chunk's last byte, 3.3-3.6 of 4.8 bits left about its first byte even with an MLP probe; start semantics encode the first byte and nothing of the content",
        "Pooling chunker (EXP-012) proposed; later the chunk vocabulary gives the chunk an identity directly.", "Finding: the chunk vector is a boundary *state*, not a chunk *summary* (`probe_main.py --identity`)")
finding("EXP-006", D, "architecture", "FLOP-free hashed n-gram tables matching BPE's 15M parameters remove ~40% of the gap, but the gain lands in the byte stage, not in the hierarchy.",
        "hnet1 1.845 -> hnet1_ng 1.705 (1 seed), paper 1.795 -> paper_ng 1.714 ± 0.016; main contribution falls 0.63 -> 0.38; punctuation gap 0.56 -> 0.16",
        "Parameter matching is part of every later configuration (vp_ng, prior, ladder tables); the hierarchy itself still needs identity.", "Where the n-gram gain lands; LR sweep; mid-training router collapses")
finding("EXP-009", D, "router-objective", "Routing is not the main gap: oracle word chunks are only 0.04 better than the learned router and 0.17 behind BPE, though chunk semantics decide how much the hierarchy is used.",
        "space_end 1.806 ± 0.026 (main 0.56) > space_start 1.827 (0.45) > fixed6 1.859 (0.25); learned hnet1 1.845", "Attention moves to what the main network receives and how early it is trained.",
        "First results of the fix sweeps (val bpb, mean ± sd over seeds; BPE = 1.634 ± 0.002, 3 seeds)")
finding("EXP-003", D, "optimizer", "BPE's best AdamW LR is 6e-3 at the grid edge while every H-Net peaks at 1.5e-3, so the tuned gap is 0.235 bpb, not 0.17.",
        "BPE 1.564 ± 0.012 at 6e-3; H-Nets 1.78-1.80 at 1.5e-3 and worse above", "EXP-013 (LR allocation) and the rank decider target the H-Net's LR ceiling; paper multipliers noted (stage 0 at 3.5x base).",
        "Where the n-gram gain lands; LR sweep; mid-training router collapses")
finding("EXP-003", D, "training-dynamics", "At lr >= 3e-3 the stage-1 router is a limit cycle between select-all and select-none: the ratio loss constrains only the mean of p, so a uniform bias shift flips every position.",
        "compression spikes to 18-631 bytes/position for tens of steps (end_mlp 6e-3 s0 at step 550); at 6e-3 the oscillation lasts the whole run", "Rank-based (quantile) boundary decisions (EXP-014) that pin compression and are immune to uniform shifts.",
        "The stage-1 router is a limit cycle at high LR (`logbook/figures/exp003_collapses.png`)")
finding("EXP-008", D, "router-objective", "The paper router runs 10-16% over its FLOPs budget at alpha 0.03; its apparent edge over fixed chunks was extra compute.",
        "realized compute +10.5% to +15.6% for paper/base/resid_id; alpha 1 holds +0.8% at 1.818 vs fixed 1.823; combo_end +379%", "Policy: H-Net arms use rank decisions or alpha >= 1; every summary reports realized FLOPs vs budget and flags > 2%.",
        "Realized compute: the paper router runs 10–16% over its FLOPs budget (`logbook/sweep_summary.py`)")
finding("EXP-008", D, "training-dynamics", "The router-entropy penalty is harmful: it drives the routers to select almost everything or kills the stage-1 hierarchy outright.",
        "combo/combo_end collapsed to compression 1.15 / 1.54 (+379% FLOPs, OOM); rent contribution 0.000 at 1.921", "Entropy penalty dropped from further use; combo arms removed from the spec.",
        "Interim readouts of EXP-008/010/011 (honest-compute view; `logbook/sweep_summary.py`)")
finding("EXP-011", D, "training-dynamics", "Every 'training signal' intervention on the bypass is neutral or harmful: path-drop inflates compression, identity-init starves the hierarchy for good, warmup is neutral, the aux head gives 0.01.",
        "rdrop +31% compute; fixed_rid contribution 0.10 at 1.854; fixed_rwarm 1.844 vs fixed 1.832; fixed_aux 1.821", "The gap is memory for identity, not signal routing: pivot to the chunk vocabulary (EXP-017).",
        "Harvest of the finished sweeps (`logbook/sweep_summary.py --all`, honest-compute view)")
finding("EXP-014", D, "router-objective", "The p-space quantile rule saturates: with the threshold detached nothing anchors the logit scale, sigmoid becomes exactly 1.0 for more than 1/N positions and the stage dies.",
        "tau/s1 -> 1.000 within 70-90 steps, then compression/s1 440-600; quant__lr0.0015_s2 compression/s0 1300-1800", "Decider rewritten to rank batch-normalized fp32 logits with a midpoint threshold; rule: every decision rule needs a scale anchor.",
        "EXP-014 post-mortem: the p-space quantile rule saturates and collapses")
finding("EXP-017", D, "router-objective", "The paper's learned stage-1 chunks are not types: half are singletons and recurring words are segmented consistently only 23-53% of the time; stage 0 is a character-pair tokenizer.",
        "chunk_stats on 127 kB: stage 1 48-55% singletons, 7-17% boundaries on word boundaries; stage 0 2.6-2.7 bytes/chunk, 3-19% on word boundaries", "Vocabulary moved to word scale (EXP-018); context-free router (EXP-019); consistency penalty (EXP-020).",
        "What tokenizer did the router learn? (`hnet/analysis/chunk_stats.py`, 128 val docs, 127 kB)")
finding("EXP-019", D, "router-objective", "Training removes consistency: the contextual router starts 92-95% consistent and ends at 30-73%, because CE-through-STE never rewards agreement across contexts; the 8-byte windowed router is less consistent at init.",
        "end_mlp init 92-95% vs trained 73% (stage 0) / 30% (stage 1); local_mlp 8% at k=8, 32% at k=4", "Self-consistency regularizer dc.router_consistency_weight (EXP-020); EXP-019 kept as a cheap test.",
        "Consistency at initialization vs after training: the router objective is anti-tokenizer")
finding("EXP-020", D, "router-objective", "An unnormalized consistency penalty is satisfied by shrinking the logits, because rank decisions on normalized scores are scale-invariant.",
        "cons1_s0: router_consistency/s0 -> 0.000 while score_std/s0 falls 1.2 -> 0.15", "Penalty made scale-free (within-group / total variance = 1 - R^2); EXP-020 relaunched; rule: once decisions are scale-free every router aux term must be too.",
        "EXP-020 first launch invalid: an unnormalized consistency penalty is satisfied by shrinking the logits")
finding("EXP-017", D, "architecture", "The chunk vocabulary works only at word scale: at (3,3) the stage-0 chunk table is nearly worthless and predicting character-pair buckets is harmful; at ratio 6 the hierarchy carries the model for the first time.",
        "EXP-017 vocab_pred +0.04 / vocab_pred_w1 +0.08 vs quant_fix; EXP-018 h1_vocab_pred -0.13 / -0.11 with main contribution 0.75 (zeroing it gives 2.47)", "The word-scale 1-stage shape becomes the main line (EXP-021/023/024).",
        "EXP-017/018 finals (step 1763, honest compute, 3 seeds)")
finding("EXP-017", D, "training-dynamics", "The fixed rank decider alone is worth as much as the paper router's 11% extra compute.",
        "quant_fix 1.799 ± 0.005 at budget vs EXP-008 base 1.796 at +11%; 0.02 better than fixed_aux 1.821", "Rank decisions become the default for every H-Net arm.",
        "EXP-017/018 finals (step 1763, honest compute, 3 seeds)")
finding("EXP-018", D, "training-dynamics", "A seed fails to adopt the hierarchy when its raw router scores go heavy-tailed: standardizing by an outlier-dominated std puts every confidence at 0.5 and the residual path wins.",
        "seed 1 score_std 13-15 vs 6-8, p_margin 0.10 vs 0.3-0.4, dechunk_rms 0.65-0.93 vs residual_rms 2.1", "Median/MAD standardization with clipped z (EXP-021).",
        "EXP-017/018/019 interim (step 1000–1250, 3 seeds): the vocabulary works only at word scale")
finding("EXP-018", D, "router-objective", "Given a vocabulary and a chunk-level objective the router converged to a word tokenizer by itself; the non-adopted seed learned a suffix/morpheme tokenizer.",
        "h1_vocab_pred_s0: 97.7% boundaries on word boundaries, 99.9% consistency, top chunks 'the ', 'of '; s1: 0.6% on word boundaries, tokens 'ing', 'ed', 'ly', 'ion'", "Adoption is decided early (steps 200-600); the next dynamics problem is the race between hierarchy and bypass.",
        "What tokenizer did EXP-018 learn? (`chunk_stats`, final checkpoints, `logbook/diagnostics/cs_exp018_*.json`)")
finding("EXP-021", D, "training-dynamics", "Median/MAD standardization collapses under a word tokenizer: 5/6 of the scores are identical, the MAD -> 0 and 1/sigma blows up the gradient.",
        "score_std/s0 4.4 -> 0.0 from step ~650, grad norm 0.3 -> 1.9e6, val bpb 3.25 at step 1000 (h1r_vocab_pred_s0)", "Trimmed mean/std over the central 98% (EXP-023); rule: a scale estimator for bimodal scores must straddle both modes.",
        "EXP-021 (median/MAD decider) stopped: the MAD collapses under a word tokenizer")
finding("EXP-023", D, "training-dynamics", "Adoption and the kind of tokenizer go together and are deterministic per seed across deciders: the bifurcation is in the router's initialization or first few hundred steps of data, not the decision rule.",
        "vp 1.732 / 1.855 / 1.733 replicates EXP-018 1.720 / 1.863 / 1.737 seed by seed; vp_s1 suffix tokenizer (4% on word boundaries), vp_ng_s2 word (95%)", "Steer the first ~300 steps: residual warmup (failed) or a brief whitespace prior (EXP-024).",
        "EXP-023 tokenizer statistics: adoption and the kind of tokenizer go together (`cs_exp023_*.json`)")
finding("EXP-024", D, "training-dynamics", "A 10%-of-training whitespace warm start on the router makes adoption reliable and the H-Net matches BPE at equal FLOPs; a sharper confidence temperature does the opposite.",
        "prior 1.6337 / 1.6346 / 1.6364 = 1.635 ± 0.001 vs BPE 1.634 ± 0.002, realized compute +0.3%, all seeds adopt by step 250; temp4 1.723 with hierarchy share 0.02-0.21", "prior becomes the best configuration (base of EXP-025/026); the ladder gets an hnet_ws arm.",
        "EXP-024: the tokenizer curriculum makes adoption reliable; H-Net matches BPE at equal FLOPs")
finding("EXP-024", D, "router-objective", "After the prior is released the router keeps and refines the word tokenizer; it does not drift back to character pairs or suffixes.",
        "final checkpoint: 97% of boundaries on word boundaries, 99.3% hit, 99.8% consistency, 25% singletons (= whitespace tokenizer)", "The curriculum is a warm start, not a constraint.",
        "EXP-024: the tokenizer curriculum makes adoption reliable; H-Net matches BPE at equal FLOPs")
finding("LADDER-LR-1e16", D, "scaling", "On A100-40GB the H-Net is ~3x slower per byte than BPE at equal FLOPs/byte, and packing several runs per GPU gains nothing.",
        "BPE L08 34% MFU (40.5% whole-model compile) vs H-Net L08 17%; 3 L04 runs per GPU: 3x0.55M vs 1.87M bytes/s", "Ladder moved to GCP A100 spot (~$114 for 79 A100-h); the 1e18 warm-start arm dropped to fit +$100.",
        "A100-40GB throughput (GCP spot, compiled blocks, 50 steps, `scripts/bench`)")
finding("LADDER-LR-1e16", D, "data", "mix_v1 has 4.126 bytes/token under the 32k in-corpus BPE; the ladder keeps 4.145 (measured on the small build) for consistency.",
        "25.0 GB train, 4.96M docs, 98 shards + 41 val sets; en 4.53, code 3.46, math 3.50, multilingual 3.15", "0.5% mismatch in the H-Net's target compression accepted across the ladder.",
        "mix_v1 full build")
finding("EXP-023", D, "training-dynamics", "The adoption failure is a positive-feedback race: the bypass opens at steps 60-130 while the router is undecided, the decoder switches to it, the router loses its only gradient and drifts into a basin; the next-chunk objective is indifferent between word and suffix tokenizers and never reaches the router.",
        "hierarchy rms 1.6 -> 0.7 and margin 0.36 -> 0.15 in all seeds at steps 60-130; next-chunk CE 4.9 at step 500 for both basins; warm-started prior_s1 never drops below 0.95", "dc.chunk_ste and dc.residual_gate=margin implemented and tested in EXP-027; a usefulness objective for the router sought (EXP-028).",
        "Deep dive: the adoption race, step by step (`logbook/figures/adoption_dynamics.png`, `cs_exp023_vp_s{0,1}_step*.json`)")
finding("EXP-027", D, "infra", "residual_proj is zero-init only in its weight; the default nn.Linear bias is live, so 'bypass closed at init' was never exactly true.",
        "bias rms 0.064 at init (grad_audit side finding)", "Documented; grad_audit prints it.",
        "Why the issues were found late: CLAUDE.md, a step-0 gradient audit, an invariance suite, a visibility tool")
finding("EXP-027", D, "infra", "Nine failures found late were all derivable from the gradient graph or the first 200 CPU steps, yet were found from the final bpb of paid sweeps.",
        "list of nine misses in CLAUDE.md; grad_audit shows at step 0 that the next-chunk CE reaches only the chunk table", "Twelve rules in CLAUDE.md; hnet/analysis/grad_audit.py, tests/test_invariances.py and logbook/training_report.py added.",
        "Why the issues were found late: CLAUDE.md, a step-0 gradient audit, an invariance suite, a visibility tool")
finding("EXP-025", D, "training-dynamics", "The adoption race is the normal state of these runs, not an exception: the bypass rms is 5-40x the hierarchy's in every H-Net run of EXP-025 and the ladder LR sweep.",
        "training_report mechanism panels (hierarchy vs bypass rms)", "Mechanism panels and alerts in every status report.",
        "Why the issues were found late: CLAUDE.md, a step-0 gradient audit, an invariance suite, a visibility tool")
finding("LADDER-LR-1e16", D, "optimizer", "BPE L04 has a sharp Muon-LR optimum at 0.02 while both H-Nets prefer 0.01 and are flatter; transferring BPE's LR (blog protocol) costs the H-Nets ~0.04.",
        "bpe 1.622 at 0.02 (1.706 / 1.784 either side); hnet 1.861 at 0.01, hnet_ws 1.820; at 0.0025 the router never decides (margin 0.03)", "Iso-FLOP ladder runs with best_mlr.bpe = 0.02 for every arm; caveat noted for the write-up.",
        "Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched")
finding("LADDER-LR-1e16", D, "scaling", "At the ladder's prior-optimum rung the H-Net trails BPE by 0.24 bpb prior-free and 0.20 with the warm start at equal FLOPs and data, against 0.00 at tiny scale with the curriculum.",
        "L04 1e16: bpe 1.622, hnet 1.861, hnet_ws 1.820; bypass rms 5-25x the hierarchy's above LR 0.01", "1e17 held until EXP-027's first evals decide whether the ladder's H-Net arm carries chunk-STE / gated bypass.",
        "Ladder LR tuning done (L04, 1e16, one seed each; `ladder_lr_1e16` + `lrext_1e16`), 1e16 rungs launched")
finding("EXP-028", D, "router-objective", "The hierarchy's measured per-byte gain separates word from suffix tokenizers while the next-chunk CE does not: a usefulness objective for the router exists.",
        "step 500: word seed 0 saves 0.165 bits/byte (0.164 at word starts) vs suffix seed 1 0.068 (0.026): 2.4x / 6x; next-chunk CE identical (4.9)", "dc.router_gain_weight (REINFORCE on the standardized gain) implemented; EXP-028 spec written.",
        "Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers")
finding("EXP-028", D, "data", "Offline segmentation probes: 5-gram entropy spikes are a plausible language-agnostic prior but weaker than whitespace on English; unigram MDL without a length prior is not a tokenizer.",
        "entropy spikes 0.66 on whitespace, 0.60 consistency, 2.01 unigram bits/byte; MDL 0.13 on whitespace, 84% singletons", "Entropy prior kept as the candidate for code/multilingual rungs; whitespace curriculum stays on English.",
        "Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers")
finding("EXP-027", D, "training-dynamics", "At step 500 the chunk-STE and margin-gate arms have the same loss as the prior-free reference, but the margin gate makes the hierarchy load-bearing from the start.",
        "val/zero-main at step 500: cste 2.022 / 2.054, cste_rgate 2.000 / 2.140, rgate 1.999 / 2.404 vs vp_ng 1.994-1.956 / 2.02-2.05; rgate contribution 0.42 vs 0.03-0.10", "Finals at step 1763 decide; the ladder's H-Net arm waits on them.",
        "Falsifiers from the first-principles ideas (no training): the hierarchy's gain *does* separate word from suffix tokenizers")
finding("EXP-002", D, "infra", "With the fused Mamba kernels, compile: blocks makes the tiny model ~4.6x slower (recompiles around torch.compiler.disable'd kernels).",
        "2.6 s/step compiled vs 0.57 s/step eager on a spot L4", "All tiny sweeps run with train.compile=false; to investigate before the 125M runs (later: whole-model compile + fused LM-head loss for the ladder).",
        "WS-H done: GCP runner `python -m infra.gcp`")
finding("EXP-002", D, "data", "GPT-2 bytes/token on FineWeb-Edu is 4.621 (FineWeb 4.454); the paper's 1792 tokens <-> 8192 bytes holds.",
        "measured on the converted shards; speedrun targets 1.0667 / 0.9496 bpb", "Tiny configs matched at 4.6 bytes/token; val protocol scores every byte once on identical text.",
        "22:55 · WS-C (data pipeline) finished")

# ----------------------------------------------------------------------------------------------------------------
# 6. components
# ----------------------------------------------------------------------------------------------------------------
COMPONENTS = []


def comp(name, path, key, intro, desc, status):
    COMPONENTS.append(dict(component=name, module_path=path, config_key=key, introduced_in_exp=intro, description=desc, status=status))


comp("paper cosine lookahead router", "hnet/model/routing/cosine.py (CosineLookaheadRouter)", "model.dc.router=cosine_lookahead", "EXP-002",
     "The paper's router: boundary p from the cosine distance of consecutive states (start semantics, one-chunk lookahead).", "harmful")
comp("end-of-chunk MLP router", "hnet/model/routing/end.py (EndMLPRouter)", "model.dc.router=end_mlp", "EXP-002",
     "Causal end-of-chunk boundary logit from an MLP on the encoder state (blog semantics, EXP-001); the router of every later arm.", "kept")
comp("end-of-chunk linear / cosine-shift routers", "hnet/model/routing/end.py, routing/cosine.py (EndCosineShiftRouter)", "model.dc.router=end_linear | end_cosine_shift", "EXP-002",
     "Alternative end-of-chunk guesses; indistinguishable from end_mlp within seed noise.", "neutral")
comp("quantile (rank) decider", "hnet/model/routing/decide.py (QuantileDecider)", "model.dc.decision=quantile; model.dc.decision_kwargs={temp, normalize, momentum}", "EXP-014",
     "Select the top 1/N positions by rank of the batch-standardized fp32 logit (trimmed mean/std since EXP-023); pins compression exactly, makes the ratio loss inert, immune to uniform shifts. EXP-014's p-space version collapsed; EXP-021's median/MAD version diverged.", "kept")
comp("confidence temperature", "hnet/model/routing/decide.py", "model.dc.decision_kwargs={temp: 4.0}", "EXP-024",
     "Sharper STE/EMA confidences from the standardized score margin.", "harmful")
comp("chunk vocabulary (hashed chunk table)", "hnet/model/chunk_vocab.py (ChunkVocab, hash_bank)", "model.dc.chunk_vocab=[{buckets, n_hash, max_len, ...}]", "EXP-017",
     "Each chunk's byte content hashed by n_hash hashes into FLOP-free embedding tables added to the chunk vector: a token table for the learned tokenizer. Worthless at (3,3) character pairs, decisive at word scale.", "kept")
comp("next-chunk objective (sampled softmax)", "hnet/model/chunk_vocab.py", "model.dc.chunk_vocab[...].pred_weight / n_neg", "EXP-017",
     "Trains the main network to predict the next chunk's bucket (1024 negatives, training only, +1.6% FLOPs). Harmful at (3,3), part of every word-scale arm; does not reach the router unless chunk_ste.", "kept")
comp("byte n-gram tables", "hnet/model/ngram.py (HashNgramEmbedding)", "model.ngram_embed={ns: [3..8], buckets}", "EXP-006",
     "BLT-style hashed byte n-gram embeddings (FLOP-free) added to the byte stage; parameter-matches BPE's table and closes ~40% of the gap.", "kept")
comp("whitespace prior curriculum", "hnet/model/lm.py (router_prior)", "model.dc.router_prior_weight, model.dc.router_prior_frac", "EXP-024",
     "For the first router_prior_frac of training the stage-0 router's logit is also trained (BCE, weight -> 0) toward 'a chunk ends at every whitespace byte'; then released.", "kept")
comp("chunk STE", "hnet/model/hnet.py", "model.dc.chunk_ste=true", "EXP-027",
     "Chunk vectors entering the main network are multiplied by STE(confidence), so chunk-level losses train the router independently of the decoder's adoption.", "untested")
comp("residual gate (margin)", "hnet/model/hnet.py", "model.dc.residual_gate=margin", "EXP-027",
     "The residual bypass into each decoder is scaled by the router's mean margin 2*mean|p-0.5|, so it opens only once the tokenizer is decided.", "untested")
comp("router gain objective", "hnet/model/lm.py, hnet/analysis/grad_audit.py", "model.dc.router_gain_weight, model.dc.router_gain_ema", "EXP-028",
     "Second decoder + head pass on the bypass alone; each byte's gain is credited to its chunk, standardized, and the router's decisions are trained by REINFORCE on that advantage (+~15% compute at tiny scale).", "untested")
comp("local (context-free) router", "hnet/model/routing/local.py (LocalMLPRouter)", "model.dc.router=local_mlp; model.dc.router_kwargs={k, d_hidden}", "EXP-019",
     "Boundary logit from the last k bytes only, so the same string is always cut the same way.", "neutral")
comp("router consistency penalty", "hnet/model/lm.py (_router_consistency)", "model.dc.router_consistency_weight, model.dc.router_consistency_n", "EXP-020",
     "Share of the stage-0 logit variance not explained by the last n bytes (1 - R^2 of the hashed n-gram group means); scale-free since the second launch.", "neutral")
comp("router entropy penalty", "hnet/model/lm.py", "model.dc.router_entropy_weight", "EXP-008",
     "Mean binary entropy of boundary probabilities, pushing toward confident decisions; amplifies the all-flip mode.", "harmful")
comp("residual path-drop", "hnet/model/hnet.py", "model.dc.residual_dropout", "EXP-008",
     "Training-time probability of zeroing a whole segment's residual path so the decoder must rely on the hierarchy; inflates compression by 31%.", "harmful")
comp("identity-init residual", "hnet/model/hnet.py", "model.dc.residual_init=identity", "EXP-010",
     "Residual projection initialized to identity instead of zero; the bypass is strong from step 0 and the hierarchy is never used.", "harmful")
comp("residual warmup", "hnet/model/hnet.py (progress hook)", "model.dc.residual_warmup", "EXP-011",
     "Residual path into each decoder scaled from 0 to 1 over the first fraction of training; neutral at (3,3), harmful at word scale (vp_rwarm).", "harmful")
comp("main-path auxiliary head", "hnet/model/hnet.py (r.main_out) / lm.py", "model.dc.main_aux_weight", "EXP-008",
     "Auxiliary next-byte CE on the stage-0 dechunked main-network output alone; +0.01 on the fixed hourglass, collapsed compression with the learned router.", "neutral")
comp("chunk-level multi-byte prediction", "hnet/model/lm.py", "model.dc.chunk_pred_k, model.dc.chunk_pred_weight", "EXP-016",
     "Head on the innermost main network's output predicting the first k bytes of the next chunk (bypasses dechunker, residual and decoder). Spec only.", "untested")
comp("pooling chunker", "hnet/model/chunking/pool.py (PoolChunker)", "model.dc.chunker=pool; model.dc.chunker_kwargs={mode: mean|attn}", "EXP-012",
     "Chunk vector = learned summary (mean or attention pooling) of the encoder states within the chunk plus the boundary state.", "untested")
comp("select chunker / EMA dechunker", "hnet/model/chunking/select.py, chunking/dechunk.py (EMADechunker, RepeatDechunker)", "model.dc.chunker=select; model.dc.dechunker=ema|repeat", "EXP-002",
     "Paper's select-and-plug-back with EMA smoothing; 'repeat' (no smoothing) was the paper_nosmooth ablation (neutral).", "kept")
comp("ratio loss / strong ratio loss", "hnet/model/losses/ratio.py", "model.dc.ratio_loss_alpha, model.dc.aux_loss=ratio", "EXP-002",
     "Paper's compression ratio loss; at alpha 0.03 it cannot hold the budget (max penalty ~0.09 nats), alpha 1.0 holds it. Inert under rank decisions.", "neutral")
comp("oracle / fixed routers", "hnet/model/routing/heuristic.py (SpaceLikeRouter, FixedPoolRouter)", "model.dc.router=space_like | fixed_pool; router_kwargs={semantics}", "EXP-009",
     "Hard boundaries at word starts/ends or every N bytes, for diagnosis of routing vs capacity.", "kept")
comp("per-stage LR multipliers", "hnet/optim/groups.py (auto_lr_multipliers)", "model.lr_multiplier=auto | null | [..]", "EXP-002",
     "Paper App. C modulation (stage 0 at ~3.5x base, main at 0.72x); removing them is neutral at tiny scale.", "neutral")
comp("Muon optimizer", "hnet/optim/muon.py (MuonAdamW, build_muon_optimizer)", "train.optimizer=muon; train.muon.lr, lr_embed/lr_head/lr_other, momentum_warmup_steps", "EXP-025",
     "modded-nanogpt record-18 Muon on hidden 2-D matrices, Adam on embeddings/heads/norms/router. Ladder optimizer; on the tiny prior H-Net 1.631 vs AdamW 1.635.", "kept")
comp("Kimi Delta Attention (KDA) mixer", "hnet/model/mixers/kimi.py (KimiDeltaAttention)", "model.mixer_codes={m: kda}; model.mixer_cfg={kda: [{expand_v}]}", "EXP-010",
     "Chunkwise gated delta-rule linear attention replacing the outer Mamba-2; crashed in EXP-010 (channel constraint) and failed again in EXP-026 (rc=1).", "untested")
comp("sliding-window attention (SWA) outer mixer", "hnet/model/mixers/attn.py (CausalAttention, window_size)", "model.mixer_codes={m: attn, t: attn}; model.attn.window_size", "EXP-010",
     "Outer Mamba-2 replaced by windowed softmax attention (window 256); 1.742 vs 1.635 on the prior model.", "harmful")
comp("DeepSeek / Kimi attention variants (MLA, NSA, DSA, MoBA, gated)", "hnet/model/mixers/deepseek.py, hnet/model/mixers/kimi.py", "model: configs/models/tiny_*_2m_<variant>.yaml", "EXP-005",
     "Implemented and FLOPs-matched with tests; never trained (EXP-005 not launched).", "untested")
comp("attention residuals", "hnet/model/isotropic.py (Isotropic attn_res)", "model.attn_res", "EXP-005",
     "arXiv 2603.15031 block-wise attention over earlier layer outputs; wired into configs and FLOPs, never trained.", "untested")
comp("packed block mask", "hnet/model/mixers/attn.py (packed_block_mask)", "(always on for flex attention)", "EXP-002",
     "Eager, index-clamped, row-chunked BlockMask for ragged packed lengths; fixed the CUDA illegal-memory-access crashes and the [H,T,T] OOM.", "kept")
comp("zero-main eval probe", "hnet/model/probes.py (zero_main)", "eval.zero_main_probe=true", "EXP-008",
     "val bpb with each stage's main network silenced; val/zero_main_s0_bpb - val/bpb = the hierarchy's contribution.", "kept")
comp("fairness gate + realized-compute check", "scripts/search/fairness.py, logbook/sweep_summary.py", "fairness: strict | tuned", "EXP-002",
     "Holds data, bytes/step, FLOPs/byte (±2%), optimizer and eval equal across arms; realized FLOPs from logged compression flagged above 2%.", "kept")
comp("chunk_stats tokenizer statistics", "hnet/analysis/chunk_stats.py", "(analysis)", "EXP-017",
     "Bytes/chunk, singleton share, boundaries on word boundaries, per-word consistency and top chunks of a trained router vs the whitespace tokenizer.", "kept")
comp("step-0 gradient audit", "hnet/analysis/grad_audit.py", "(analysis)", "EXP-027",
     "Autograd of each objective term against each parameter group at init and after N SGD steps, with mechanism metrics and plain-language warnings.", "kept")
comp("invariance suite", "tests/test_invariances.py", "(tests)", "EXP-027",
     "Causality and packing equivalence for the paper router, end_mlp + rank decisions and the chunk vocabulary (train-mode rank decisions couple the batch; eval mode is causal).", "kept")

# ----------------------------------------------------------------------------------------------------------------
# write
# ----------------------------------------------------------------------------------------------------------------
os.makedirs(OUT, exist_ok=True)
SNAPSHOT = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
with open(os.path.join(OUT, "SNAPSHOT.txt"), "w") as f:
    f.write(f"runs.csv harvested from the run directories and live mirrors at {SNAPSHOT} (UTC)\n")


def write(name, rows, fields):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    print(f"{name}: {len(rows)} rows")


write("experiments.csv", E, ["exp_id", "name", "spec_file", "date", "question", "status", "scale", "data", "budget_tokens_eq", "seeds", "gpu",
                             "cost_usd", "outcome", "worklog_section", "category", "notes"])
write("arms.csv", ARMS, ["exp_id", "arm", "label", "base_config", "parent_arm", "notes"])
write("arm_settings.csv", [dict(exp_id=a, arm=b, setting=c, value=d, origin=e) for a, b, c, d, e in SETTINGS], ["exp_id", "arm", "setting", "value", "origin"])
write("runs.csv", RUNS, ["exp_id", "arm", "seed", "run_dir", "status", "steps_done", "steps_total", "final_step", "val_bpb_final", "zero_main_bpb_final",
                         "main_contribution", "train_bpb_final", "adopted", "tokenizer_kind", "source", "notes"])
write("arm_results.csv", ARM_RESULTS, ["exp_id", "arm", "n_seeds", "val_bpb_mean", "val_bpb_sd", "best_seed_bpb", "worst_seed_bpb", "delta_vs_bpe", "reference",
                                       "delta_vs_reference", "verdict", "notes"])
write("references.csv", REF_ROWS, ["reference_id", "exp_id", "arm", "val_bpb", "n_seeds", "description"])
write("findings.csv", FINDINGS, ["finding_id", "exp_id", "date", "category", "statement", "evidence", "consequence", "worklog_section"])
write("components.csv", COMPONENTS, ["component", "module_path", "config_key", "introduced_in_exp", "description", "status"])

# integrity checks
exp_ids = {e["exp_id"] for e in E}
assert all(a["exp_id"] in exp_ids for a in ARMS)
assert all((r["exp_id"], r["arm"]) in ARM_KEYS for r in RUNS)
assert all((s[0], s[1]) in ARM_KEYS for s in SETTINGS)
for a in ARMS:
    if a["parent_arm"] != NULL:
        pe, pa = a["parent_arm"].split("/", 1)
        assert (pe, pa) in ARM_KEYS, a["parent_arm"]
assert all(f["exp_id"] in exp_ids for f in FINDINGS)
assert all(c["introduced_in_exp"] in exp_ids for c in COMPONENTS)
assert all((r["exp_id"], r["arm"]) in ARM_KEYS for r in REF_ROWS)
assert len({(r["exp_id"], r["arm"], r["seed"]) for r in RUNS}) == len(RUNS)
print("integrity OK")
