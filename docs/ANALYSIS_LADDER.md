# Ladder evaluation and analysis: per-source bpb, downstream bpb, iso-FLOP fits, compute multipliers

Tooling for the BPE-vs-H-Net iso-FLOP ladder (1e16/1e17/1e18), reproducing the analysis of "From Tokens to Bytes" I/II
(docs/BLOG_REPRO_PLAN.md). Library: `hnet/analysis/scaling.py`. Tests: `tests/test_scaling.py`, `tests/test_per_source.py`.

| Step | Command | Output |
|---|---|---|
| Per-source bpb + compression | `python scripts/eval/per_source.py --model RUN --mix mix_v1` (or `--sources MANIFEST`) | `RUN/per_source.json` |
| Downstream soft bpb | `python scripts/eval/downstream_bpb.py --model RUN [--domains paloma_like]` | `RUN/downstream_bpb.json` |
| Fits, multipliers, figures | `python scripts/analysis/fit_ladder.py --runs 'runs/search/ladder/*/*/*_s*' [--out X.json]` | report, JSON, `logbook/figures/ladder_*.png` |

## Sources
`--mix <dataset>`: one source per leaf val set of a built mix (docs/DATA_MIX.md): group english/code/math/multilingual,
code `lang`, held-out languages flagged `held_out`, weight = the leaf's target share. Manifest (YAML/JSON; paths relative
to the manifest dir, then repo root): `sources: [{name, paths|path, group, lang, held_out, weight, max_docs, max_bytes,
text_field, doc_split}]`. Formats: `<stem>.bytes.bin` (+ `.idx.npy`, complete docs), GPT-2 token shards `*.bin`,
`*.jsonl[.gz]`, `*.json.gz`, `*.parquet`, `*.txt`, `hf://<repo>/<file>[@rev]`. Default: FineWeb-Edu val.

## Definitions
- Per-source bpb: docs in file order (valid UTF-8) until max_bytes (whole docs); windows cut greedily at token boundaries
  of one split tokenizer (`--split-tokenizer auto` = the run's dataset tokenizer, so BPE and H-Net runs of a ladder score
  identical windows) that are UTF-8 character starts, <= byte_seq_len-1 bytes and token_seq_len-1 tokens (defaults: run's
  eval.val_byte_seq_len / val_token_seq_len); window c conditioned on one element (BOS/EOT, else the last element of
  window c-1). Byte models: bits of every byte; BPE: summed token log-probs / UTF-8 bytes. No end-of-doc target.
  Reproduces `valset_bpb` exactly on FineWeb-Edu. Groups pooled by bytes; held-out -> `<group>_held_out` (not in `all`);
  `mix` = weight-averaged.
- Compression: H-Net positions per stage from the routers' boundary masks in eval mode; `bytes_per_main_position` =
  stage-0 positions / innermost positions (trainer convention). BPE: `bytes_per_token`.
- Downstream soft bpb: -log2 p(gold continuation | context) / continuation bytes, pooled per task (LAMBADA: target word),
  `average` = unweighted mean of the 7 paper7 tasks (item-bootstrap CI). Gold choice only unless `--all-choices`.
  `--limit N`: seeded subset (`--subset-seed -1`: first N, the trainer convention). `--domains paloma_like`: WikiText-103
  test + C4-en val shard 0 (Paloma is gated).
- BPE prompt boundary: `--boundary canonical` (default) tokenizes context+continuation jointly and scores from the first
  token containing any answer byte (straddling token charged to the answer; denominator = answer bytes);
  `lmeval` = lm-eval `_encode_pair`. `boundary_stats` counts straddles / non-canonical items (0/2100 with GPT-2
  pretokenization). No marginalization (needed for BPE without pretokenization).
- Run metadata: C = sum flops_per_byte x bytes over metrics.jsonl intervals; N = total params (`--param-key
  params_nonembed`); D = bytes. Arch: ladder.json > CLI `ARCH=GLOB` > sweep arm > model.type. Budget: ladder.json > sweep
  point > spec name (`ladder_1e17_L05`) > clustering (0.15 dex).

## Fits (per metric: val, group:*, src:*, down:*, dom:*)
1. Seeds averaged per (arch, budget, N). Iso-FLOP: quadratic in ln N; vertex outside the sampled sizes -> `edge`
   (clamped); < 3 sizes / concave -> `min_observed`. D_opt from log-linear D(N). Each metric gets its own frontier.
2. L(C) = E + A (C/1e18)^-alpha; `--form auto`: >=4 budgets free E, 3 shared E, 2 pure, 1 none. E profiled; A, alpha
   closed form.
3. Multiplier m(L) = C_BPE(L)/C_HNet(L) over the loss range covered by both frontiers (else H-Net's, `extrapolated`);
   geometric mean, ends, per budget; `multiplier_interp` = law-free check.
4. Bootstrap: seeds (replicate seeds) or residual (pooled dof-corrected iso-FLOP residuals); full refit per replicate.
5. N_opt(C), D_opt(C) log-log fits.
6. Loss-to-loss: y = K x^kappa over all runs per arch; m_l2l(L) = C_BPE(L')/C_BPE(L), g_BPE(L') = g_HNet(L);
   compute->down = m_val x m_l2l (Part II: 1.28 x 1.61 = 2.06); case-bootstrap CI.

Figures: loss_vs_compute, isoflop, domain_multipliers, code_multipliers (in-dist vs held-out medians), compression,
loss_to_loss, allocation.

## Degenerate inputs
One budget: "single compute budget ... only matched-budget loss gaps are reported"; one size -> min_observed; narrow
val spread / non-increasing relation -> loss-to-loss notes. Runs without per_source/downstream JSON use the trainer's
eval.jsonl soft bpb as down:* (lm-eval boundary, first-N; `info.down_source`).

## Validation and limitations
Synthetic Chinchilla ladders (H-Net(N,D) = BPE(N sqrt m, D sqrt m)): multipliers within 3% (2/3/4 budgets), per-language
within 4%, N_opt/D_opt exponents within 0.02, CIs cover the truth, l2l matches its closed form. Quadratic iso-FLOP bias
~1-2% low with 4 sizes; E not identifiable per arch with 3 budgets (`--form fixed --E`); paper7 not OLMES; FLOPs drift
within a budget ignored beyond the geometric mean (`flops_spread_dex` reported).

Known gap: the trainer's in-training `eval.downstream` uses the GPT-2 scorer for token models; for BPE runs on `mix_v1`
use `scripts/eval/downstream_bpb.py` (dataset tokenizer, end-of-text id 0) instead.

## LR tuning result (2026-10-29)
L04 at 1e16, one seed: BPE best Muon LR 0.02 (1.622 bpb; 0.01 → 1.706, 0.04 → 1.784); hnet best 0.01 (1.861);
hnet_ws best 0.01 (1.820). The iso-FLOP specs use `--param best_mlr.bpe=0.02` for every arm (blog protocol).
