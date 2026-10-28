# hnet

A from-scratch, modular implementation of **H-Net** (dynamic-chunking hierarchical byte-level language models;
Hwang, Wang & Gu, arXiv 2507.07955) and a controlled comparison against tokenized (BPE) Transformers at the
100M–500M scale. The experiments follow Cartesia's "From Tokens to Bytes" posts on a ~$500 budget.

* `PLAN.md`: the experiment plan in effect. `PLAN_full_scale.md`: the original 3.7B-vs-SmolLM3 plan and the
  verified reference numbers.
* `SPEC.md`: interfaces between components (model, data, trainer, eval, infra).
* `logbook/`: `WORKLOG.md` (everything done, chronologically) and `EXPERIMENTS.md` (every run, with results and
  loss curves).
* `hnet/`: the package. Components (mixers, routers, chunkers, dechunkers, losses, optimizers) are registered by name
  in `hnet/registry.py` and selected in YAML configs.
* `configs/reference/`: data extracted from the blog's figures, used as validation targets.

Status: under construction. See the logbook.
