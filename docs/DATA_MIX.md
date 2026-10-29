# mix_v1: the "From Tokens to Bytes" training mixture at small scale

`mix_v1` reproduces the data setup of the Cartesia blog (part I, "Data" and Figure 2; summary in
`docs/BLOG_REPRO_PLAN.md`): a 70 / 20 / 5 / 5 **byte** mixture of English web text, code, math and multilingual web
text. It is stored twice over exactly the same documents, as byte shards for byte-level models and as token shards
from an **in-corpus 32k byte-level BPE** for the BPE baseline, plus validation sets per source, per code language
(held-out languages included), per category and for the whole mixture.

Code: `hnet/data/mix.py` (pipeline), `hnet/data/hf_sources.py` (HF parquet reader), `hnet/data/tokenizer.py`
(BPE training and decoding tables), `scripts/data/build_mix.py` (CLI). Spec: `configs/data/mix_v1.spec.yaml`.
Tokenizer: `configs/tokenizers/mix_v1_bpe32k.json`. Tests: `tests/test_data_mix.py`, `tests/test_data_mix_train.py`.

## Sources and shares

Shares are fractions of training UTF-8 bytes (as in the blog). All sources are public HF parquet datasets, read
anonymously at pinned revisions (see the spec).

| category | leaf | HF source | share of mix | blog source (share) |
|---|---|---|---|---|
| English 70% | `en.fineweb_edu` | `HuggingFaceFW/fineweb-edu`, `sample/10BT` | 70.00% | fineweb-edu (70.00%) |
| Code 20% | `code.c` | `codeparrot/github-code-clean`, language `C` | 3.33% | stack-v2-c (3.29%) |
| | `code.cpp` | `C++` | 2.53% | stack-v2-cpp (2.50%) |
| | `code.javascript` | `JavaScript` | 2.36% | (2.33%) |
| | `code.python` | `Python` | 2.32% | (2.29%) |
| | `code.java` | `Java` | 2.30% | (2.27%) |
| | `code.php` | `PHP` | 2.03% | (2.00%) |
| | `code.csharp` | `C#` | 1.94% | (1.92%) |
| | `code.markdown` | `Markdown` | 1.30% | (1.28%) |
| | `code.typescript` | `TypeScript` | 0.59% | (0.58%) |
| | `code.sql` | `SQL` | 0.41% | (0.40%) |
| | `code.go` | `GO` | 0.31% | (0.31%) |
| | `code.ruby` | `Ruby` | 0.27% | (0.27%) |
| | `code.shell` | `Shell` | 0.16% | (0.16%) |
| | `code.rust` | `Rust` | 0.15% | (0.15%) |
| | — | (Swift is not in github-code-clean) | — | stack-v2-swift (0.26%) |
| Math 5% | `math.finemath` | `HuggingFaceTB/finemath`, `finemath-3plus` | 3.07% | finemath-3plus (3.07%) |
| | `math.infiwebmath` | `HuggingFaceTB/finemath`, `infiwebmath-3plus` | 1.93% | infiwebmath-3plus (1.93%) |
| Multilingual 5% | `ml.rus_Cyrl`, `ml.cmn_Hani`, `ml.deu_Latn`, `ml.fra_Latn`, `ml.spa_Latn`, `ml.ita_Latn`, `ml.por_Latn`, `ml.arb_Arab` | `HuggingFaceFW/fineweb-2` (`data/<lang>/train`) | 0.625% each | fineweb2-hq-* (0.67% each, Arabic 0.33%) |

Code weights are the blog's Figure 2 byte shares of its 15 Stack v2 languages; without Swift the other 14 are
renormalized to the 20% code share (factor 20 / 19.75).

**Held-out code languages** (evaluation only, never in train; the blog used 10 Stack v2 languages outside its mix,
from Haskell to Prolog): `heldout.haskell`, `heldout.fortran`, `heldout.lua`, `heldout.julia`, `heldout.perl`,
`heldout.scala`, `heldout.assembly`, `heldout.visual_basic`, `heldout.powershell`, `heldout.tex` (github-code-clean
values Haskell, FORTRAN, Lua, Julia, Perl, Scala, Assembly, Visual Basic, PowerShell, TeX). The other
github-code-clean languages (HTML, CSS, Makefile, Dockerfile, CMake, Batchfile) are not used at all.

### Deviations from the blog

* **Code is `codeparrot/github-code-clean`, not Stack v2** (Stack v2 contents need Software Heritage / AWS access).
  It has 30 languages: 14 of the blog's 15 in-mix languages (no Swift) and none of OCaml, Prolog, Lean among the
  blog's out-of-distribution ones. Files are GitHub files (2022 snapshot, deduplicated and filtered by codeparrot).
  Our held-out set therefore differs from the blog's (6 of our 10 are plausible members of its 10: Haskell, Fortran,
  Lua, Julia, Perl, Scala; it named Haskell, Fortran, OCaml, Prolog, Lean).
* **Multilingual is FineWeb-2, not FineWeb2-HQ**, with equal bytes for the 8 languages (the blog's Arabic share is
  half the others'). `epfml/FineWeb2-HQ` is reachable and has all 8 languages; switch by editing `repo`/`files` of
  the `fw2_*` sources in the spec (the column projection skips its embeddings).
* **Tokenizer vocab 32k instead of 128k** (small models: the embedding / LM head would otherwise dominate);
  same Llama-3 pretokenization regex, trained on the first 500 MB of this mix's train stream.
* Documents longer than 1 MB (UTF-8) are dropped (rare; mostly generated code).

## Splits and val sets

* A document is **val** iff `hash(seed, key) < p_val`; `key` = the document text for web/math (so exact duplicates
  land on the same side), the **repository name** for code (whole repos go to val, so a val file never shares a
  repo with train files). `p_val` = 2% (English, math), 5% (multilingual), 10% (code repos). Held-out languages:
  every document is val.
* Train documents whose exact text equals any val-assigned document (any leaf) are dropped.
* Documents are never split, neither across train/val nor across shards.
* Val sets (whole documents, prefixes of each leaf's val stream, so they are nested):
  * `mix` (`<dataset>_val_000000.bin`, what `eval.val: standard` uses): the train mixture's byte shares, 10 MB;
  * one per category: `en`, `code`, `math`, `ml` (leaf shares as in the mix), `heldout` (equal shares), 5 MB;
  * one per leaf: `en.fineweb_edu`, `code.<lang>` (14), `heldout.<lang>` (10), `math.*` (2), `ml.*` (8), 2 MB.
  (Small preset: 512 KB / 128 KB / 64 KB.)

## Mixture order in the train stream

Every **prefix** of the train stream has the target byte shares: the leaves' documents are merged in the order of
(bytes of the leaf before the document's midpoint) / (leaf share), i.e. the leaf furthest behind its share goes next (deterministic
ties). So a run that reads only the first 10% of the data (small rungs of an iso-FLOP ladder) still sees the
70/20/5/5 mixture (to within one document per leaf); the loader's `shuffle_buffer` can stay 0. Within a leaf,
documents follow the source's order (files in a seeded permutation, row groups in order).

## On-disk layout (data root = `--root`, `$HNET_DATA` or `<repo>/data`)

```
<root>/mix_v1/manifest.json                         sizes, shares, bytes/token per leaf/category/val set, sources
<root>/mix_v1/tokenizer.json                        exact tokenizer of the token shards (copy)
<root>/mix_v1/mix_v1_train_XXXXXX.bin               token shards (modded-nanogpt header, uint16, [EOT=0] + ids per doc)
<root>/mix_v1/bytes/mix_v1_train_XXXXXX.bytes.bin   byte shards ([BOS] + UTF-8 + [EOS] per doc, hnet.data.shards header)
<root>/mix_v1/bytes/mix_v1_train_XXXXXX.idx.npy     doc index (BYTE_OFF, N_BYTES, TOK_OFF, N_TOKENS, FLAGS)
<root>/mix_v1/bytes/mix_v1_train_XXXXXX.src.npy     int16 leaf index per doc (manifest leaves[*].index)
<root>/mix_v1/mix_v1_val_000000.bin (+ bytes/...)   combined val set; mix_v1_val-<name>_000000.* named val sets
<root>/mix_pools/mix_v1-s0/<leaf>/<unit>.{bin,npy} fetched raw documents (shared by all builds of spec+seed)
<root>/mix_pools/mix_v1-s0/_state/<source>.json     fetch progress (resume point, files order, pinned revision)
```

Shards are ~256 MB of text each (small: 8 MB). Every token shard document starts with `<|endoftext|>` = id 0, and
`bytes/<stem>.json` stores the shard's statistics and resume key. The build checks that every document's tokens
decode to exactly its bytes.

## Commands

```bash
# local end-to-end test build (~50 MB of train text; streams ~0.3 GB from HF, ~1 min + tokenization)
python scripts/data/build_mix.py --preset small
python scripts/data/build_mix.py --preset small --report          # bytes/token table from the manifest

# full build (25 GB of train text) on the GPU box, straight from HF (no credentials; HF_TOKEN optional):
python scripts/data/build_mix.py --preset full --fetch-mode download --fetch-workers 12 --workers 8 --delete-pools
#   override the size with --train-bytes 50e9 etc.; re-run the same command to resume after any interruption

# retrain the in-corpus tokenizer (default sample: first 1 GB of the train stream) and build with it
python scripts/data/build_mix.py --preset full --stages fetch,tokenizer,assemble \
    --tokenizer-out configs/tokenizers/mix_v1_bpe32k.json
# BPE-NP (no pretokenization, blog part II): its own tokenizer and token shards under another dataset name
python scripts/data/build_mix.py --preset full --stages fetch,tokenizer,assemble --no-pretokenize \
    --tokenizer-sample-bytes 2e8 --tokenizer-out configs/tokenizers/mix_v1_bpe32k_np.json --dataset mix_v1_np
```

The committed tokenizer was trained with
`python scripts/data/build_mix.py --preset small --stages fetch,tokenizer --tokenizer-sample-bytes 5e8 --tokenizer-out configs/tokenizers/mix_v1_bpe32k.json`
(the first 500 MB of the seed-0 train stream; the full build's train stream starts with the same documents, and
no val-assigned document is in it).

Full-build resources (estimates from the local runs): HF reads ~25 GB (fineweb-edu ~12 GB of parquet, code ~8 GB:
the rarest in-mix language, Rust, needs ~270 code row groups); pools ~32 GB; output ~25 GB byte shards + ~11 GB
token shards (+ val sets); fetching is network-bound (10 MB/s per streamed source here; download mode is faster),
tokenization ~4–6 MB/s per core with `--workers`.

Determinism and resumption: a build is a function of (spec, seed, preset sizes, tokenizer). Fetch progress is saved
after every unit (row groups), finished shards are kept if their document plan and tokenizer are unchanged, so
re-running the same command after a crash resumes. Pools are prefixes of each leaf's deterministic document stream:
a small build's pools are a prefix of a full build's. Multi-leaf sources (code) stop storing a language once it has
its quota; such a pool cannot later grow for a bigger build in the same data root (the fetch says so; use a fresh
root or delete that source's pool).

## Using it in runs

The data section (`configs/data/mix_v1.yaml`, or `mix_v1_small.yaml`):

```yaml
# byte / H-Net run
include: _core.yaml
model: {include: ../models/hnet1_small.yaml, lr_multiplier_n_gpt: data}
data: {include: ../data/mix_v1.yaml, seq_len: 8192, micro_batch: 4, batch_bytes: 2097152}
# BPE run: same data section; the model's vocab must be the mix tokenizer's
model: {include: ../models/bpe_small.yaml, vocab_size: 32768}
data: {include: ../data/mix_v1.yaml, seq_len: <matched>, batch_tokens: ...}
```

* The token loader's document-start id (0), the per-token byte counts (`n_bytes`, bpb) and `bytes_per_token`
  (tokens_eq, `lr_multiplier_n_gpt: data`) come from the mix's manifest/tokenizer automatically
  (`hnet.data.tokenizer.dataset_tokenizer`, `hnet.data.loader.bytes_per_token`).
* Matched windows: `hnet.data.loader.matched_token_seq_len(8192, dataset="mix_v1")` (uses the measured stream
  bytes/token of the mix instead of GPT-2's 4.62).
* Validation: `eval.val: standard` = the combined set; any named set works too (`eval.val: code.python`,
  `heldout`, ...). For all sets at once on a checkpoint: `hnet.data.mix.mix_valsets("mix_v1")` returns
  `{name: ValSet}` (same text for byte and token models, as for FineWeb-Edu).
* `data.n_train_shards` limits a run to the first shards, which still have the full mixture.

## Bytes per token (small build, `mix_v1_small`, 32k in-corpus BPE)

MEASURED_TABLE

The blog's 128k in-corpus tokenizer gets 4.57 overall (fineweb-edu 4.95, code 2.88–4.79, math 3.69–3.86,
multilingual 3.75–5.43); a 32k vocabulary compresses less.
