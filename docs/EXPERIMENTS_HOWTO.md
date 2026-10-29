# Running experiments: define, check, launch, monitor, resume, report

The workflow for a sweep of short training runs (the tiny architecture search, PLAN.md §2b), from a YAML spec to a
report. Everything here runs locally on CPU except `launch.py create --execute`, which creates paid VMs.

| Step | Command |
|---|---|
| Define | `configs/search/<exp>.yaml` (format below) |
| Inspect jobs | `python scripts/search/sweep.py SPEC --list [--shard i/n]` |
| Check fairness | `python scripts/search/fairness.py SPEC` |
| Launch on GCP | `python infra/sweeps/launch.py create SPEC --vms N [--execute]` |
| Monitor | `python infra/sweeps/launch.py status SPEC` |
| Report | `python scripts/search/report.py SPEC --gcs gs://... --runs runs/gcp/<name>` |
| Tear down | `python infra/sweeps/launch.py teardown SPEC --execute` |

## 1. Single runs: the trainer CLI

```bash
python -m hnet.train.train --config configs/runs/tiny_hnet2_2m.yaml --lr 3e-3 --seed 1 --out runs/x
python -m hnet.train.train --config configs/runs/tiny_bpe_2m.yaml --model configs/models/tiny_bpe_2m_kda.yaml \
    --train-tokens 1e7 --no-compile --print-config      # print the composed config and exit
python -m hnet.train.train --help                       # every flag and the config key it sets
```

* **Flags** (`hnet/train/cli.py`, table `FLAGS`): `--out --seed --optimizer --lr --muon-lr --weight-decay --warmup
  --decay --steps --train-tokens --train-bytes --stop-at-step --micro-batch --seq-len --precision --device
  --eval-every --log-every --ckpt-every --ckpt-keep --ckpt-steps --ckpt-tokens --resume --max-hours
  --compile [blocks|model] / --no-compile`. A budget flag clears the other budgets (`--train-tokens` sets
  `train.steps=null`), so it is never silently ignored.
* **Precedence** (later wins): `--config` with its includes (`--model` swaps the run file's `model: {include:}`
  and keeps its other model keys, e.g. `lr_multiplier_n_gpt: data`) < flags < `--set a.b=c` (left to right) <
  defaults fill whatever is still unset. If a flag and a `--set` touch the same key (or one is a prefix of the
  other, e.g. `--warmup` and `--set train.schedule=...`) with different values, the command fails.
* Adding a flag: append a `Flag(...)` to `FLAGS`; the parser, `--help` and `tests/test_train_cli.py` pick it up.

## 2. Checkpoints and resume

* `<out>/ckpt/step_XXXXXXX/` is written atomically (temp dir + rename). `train.ckpt_every` steps → rolling
  checkpoints, the newest `ckpt_keep` (≥ 1) are kept. Permanent ones (marked `KEEP`): `ckpt_steps: [..]`,
  `ckpt_tokens: [..]` (first step at or past each tokens_eq value), `ckpt_every_tokens`, and the final one.
* SIGTERM/SIGINT (e.g. spot preemption) finishes the current step, writes a checkpoint and exits 0.
* `train.resume` (`--resume`): `auto` (default; newest checkpoint in `--out`, else start fresh), `none`, or a
  path to a checkpoint dir or a run dir (its newest checkpoint; branching into a new `--out` is fine). An
  explicit path that holds no checkpoint is an error.
* **Exactness.** A resumed run reproduces the uninterrupted one bit for bit on CPU: model, optimizer (AdamW
  moments; Muon momentum buffers and step counters), LR schedule, data position (a pure function of seed and
  step), RNG streams, token/byte counters and the metric sums of a log interval cut by the kill. After a hard
  kill (SIGKILL) the run restarts from the last checkpoint and `metrics.jsonl`/`eval.jsonl` are truncated to that
  step, so curves have no duplicates. Tested by `tests/test_train_resume_kill.py` (subprocess killed with
  SIGTERM and SIGKILL at an arbitrary step, AdamW and Muon) and `tests/test_train_resume.py`.
* `train.resume_strict` (default true) refuses to resume if model, data, optimizer, LR, schedule, budget, seed or
  precision differ from the checkpoint's config (the error lists the differences). Set it false to branch on
  purpose (e.g. a cooldown with another schedule).
* `train.stop_on_nonfinite: N` (default 3): after N consecutive non-finite logged losses the run stops, writes
  `diverged.json` and no checkpoint. Sweeps record such runs as `diverged`, a result, not a failure.

## 3. Defining a sweep (`configs/search/*.yaml`)

```yaml
name: exp003_lr_adamw                    # id: GCS prefix, VM names, report title
out_root: runs/search/exp003_lr_adamw    # run dirs: <out_root>/<job name>
gcs: gs://neat-dispatch-501718-a5-hnet/runs/exp003_lr_adamw
seeds: [0, 1, 2]
fairness: tuned                          # strict (default) | tuned | {mode: tuned, per_arch: [train.lr]}
set: [train.compile=false]               # --set for every job
grid: {train.lr: [5.0e-4, 1.0e-3]}       # crossed with every run and seed
grid_alias: {train.lr: lr}               # short name in job names
params: {best_lr: {bpe: null}}           # ${best_lr.bpe} in set/grid/model/config; null = must be supplied
params_note: where the params come from
report: {baseline: bpe, groups: {Routers: [paper, end_mlp]}, labels: {bpe: BPE baseline}}
runs:                                    # one entry per architecture
  - {name: bpe, config: configs/runs/tiny_bpe_2m.yaml}
  - {name: end_mlp, config: configs/runs/tiny_hnet2_2m.yaml, set: [model.dc.router=end_mlp]}
  - {name: hnet_kda, config: configs/runs/tiny_hnet2_2m.yaml, model: configs/models/tiny_hnet2_2m_kda.yaml}
```

* Jobs are named `<run>[__<alias><value>...]_s<seed>` (`end_mlp__lr0.0015_s1`; `<run>_s<seed>` without a grid).
  Overrides per job, in order: spec `set`, run `set`, the grid point, `train.seed`, `train.out_dir`.
* Order: seeds outermost, then runs, then grid points (seed 0 of everything finishes first).
  `--shard i/n` (0-based) takes jobs `i, i+n, ...`; every VM computes the same split from the spec.
* Write YAML floats with a dot (`5.0e-4`): YAML 1.1 reads `5e-4` as a string (the loader converts it anyway).
  Quote list items that contain `${...}` inside `[...]`.
* Unknown keys, duplicate names, unset parameters (all listed at once) and missing config files are errors.
  `--allow-missing` (sweep, fairness, launch) runs/checks the other runs and lists the missing ones.
* Parameters can also come from `--param name=value` or `--params-file F` (precedence spec < file < `--param`),
  but VMs read the spec from git: **paste values into the spec and commit before launching.**
* Look at what a spec expands to: `python scripts/search/sweep.py SPEC --list` or `--dry-run` (trainer commands).

## 4. Fairness gate

`python scripts/search/fairness.py SPEC [--mode strict|tuned] [--param ...] [--allow-missing]` composes every
(architecture, grid point) config and writes `<out_root>/fairness.md`. `sweep.py` and `launch.py create` refuse
to start if it fails.

* **strict** (default): data, optimizer, schedule, precision, val set and eval cadence identical; bytes/step,
  total bytes and FLOPs/byte within ±2%. Use it for ablations at one fixed recipe (EXP-002).
* **tuned**: runs may differ in the swept keys. Every architecture must have the **same grid** (keys and values)
  and the same seeds, everything else is held as in strict mode, and the report compares each architecture at its
  best value. `per_arch` keys may differ between architectures only when each run sets them from a spec
  parameter (`train.lr=${lr.bpe}`), i.e. values tuned in an earlier tuned sweep (EXP-004's Adam LR, EXP-005's LR).

## 5. Launching on GCP (coordinator)

`infra/sweeps/launch.py` uses infra/gcp's isolated gcloud config and never prints credentials.

```bash
git push                                                   # the VMs clone the commit from GitHub
python infra/sweeps/launch.py create configs/search/exp003_lr_adamw.yaml --vms 5 --max-hours 5   # review
python infra/sweeps/launch.py create configs/search/exp003_lr_adamw.yaml --vms 5 --max-hours 5 --execute
```

* `create` checks: spec expands (params set, configs present), fairness gate, the commit is on a remote branch
  and holds the working tree's spec/config files, worst-case cost (VMs x $/h x hours) ≤ `--budget-cap` ($25).
* VM `i` is `hx-<name>-<i>of<n>`: spot `g2-standard-8` (1× L4), stopped (not deleted) on preemption,
  `--max-run-duration`, labels `hnet-sweep=<name>`, `hnet-shard=<i>of<n>`, metadata `hnet-commit`, `hnet-spec`,
  `hnet-gcs`, `hnet-parallel` (default 3 runs per GPU), `hnet-shard=i/n`. `--execute` tries zones in order and
  moves on only on capacity/quota errors; an existing VM of the same name is left alone (safe to re-run).
* On the VM (`infra/sweeps/sweep_vm_startup.sh`): clone at the commit, install, check fused kernels and CUDA
  tests, fetch data, then `sweep.py SPEC --shard i/n --gcs ... --restore --mirror-every 120`. It restores the
  shard's run dirs from GCS (ones already on disk are kept), mirrors them every 120 s, uploads each run when it
  ends, uploads its log to `<gcs>/_logs/`, and shuts down. The older `hnet-only` metadata still works.

Sizing: a tiny run takes roughly 35-45 min with 3 runs sharing an L4, so 12 jobs per VM fit in ~3.5 h.

## 6. Monitoring and resuming

```bash
python infra/sweeps/launch.py status configs/search/exp003_lr_adamw.yaml
```

prints each VM's state and its shard's progress from `<gcs>/sweep_status.shard-<i>-of-<n>.json`, lists failed
runs (logs: `<gcs>/<job>/stdout.log`, `<gcs>/_logs/`), and prints the `gcloud compute instances start` command for
VMs that stopped before finishing (preemption or the run-duration cap). A restarted VM reruns the startup
script; finished jobs are skipped and unfinished ones resume from their newest checkpoint. Local equivalent:
re-run `python scripts/search/sweep.py SPEC [--shard i/n]`. Live curves while it runs: `report.py` (below) on
the pulled runs, or `logbook/exp002_progress.py` for EXP-002.

## 7. Report

```bash
python scripts/search/report.py configs/search/exp003_lr_adamw.yaml \
    --gcs gs://neat-dispatch-501718-a5-hnet/runs/exp003_lr_adamw --runs runs/gcp/exp003_lr_adamw
```

Writes `<runs>/report/`: `report.md` (final val bpb mean ± 95% t-interval over seeds; Δ vs the baseline with a
paired interval over seeds; best setting per architecture with a grid-edge warning; data multiplier vs the
baseline, stable phase vs stable phase; paper7 soft bpb and acc_norm; bytes per position; chunk diagnostics vs
the blog's targets), `summary.json`, `best_params.yaml` (`best_lr: {arch: value}`) and `figures/` (overview,
val/train bpb small multiples, LR U-curves, compression). Diverged seeds count as +inf for the best setting.

Chaining: EXP-004/005 take EXP-003's best LRs as parameters. Copy `best_params.yaml`'s values into their
`params:` (EXP-005 also needs `hnet_router`), commit, re-run the fairness gate, launch.

## 8. Current experiment specs

| Spec | Runs | Fairness | Notes |
|---|---|---|---|
| `exp002_tiny2m.yaml` | 9 variants × 2 seeds | strict | routers and ablations at LR 1.5e-3 |
| `exp003_lr_adamw.yaml` | {bpe, paper, end_mlp, end_cosine_shift} × LR {5e-4, 1e-3, 1.5e-3, 3e-3, 6e-3} × 3 seeds = 60 | tuned | |
| `exp004_muon.yaml` | same 4 × Muon LR {0.0125, 0.025, 0.05} × 3 seeds = 36 | tuned, per_arch Adam LR | needs `best_lr.*` from EXP-003 |
| `exp005_attn_variants.yaml` | {bpe, hnet} × {attn, mla, nsa, dsa, kda, kda_hybrid, moba, gated_attn} × 3 seeds = 48 | tuned, per_arch LR | needs `lr.*`, `hnet_router`; lists missing model files |
