"""Task/item types and the dataset loader shared by every prompt suite."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Union


@dataclass
class MCItem:
    """One multiple-choice question as scoring requests.

    contexts[i] + continuations[i] is choice i's full text; ``norm_texts[i]`` is the string whose ``len()`` is the
    char-normalization denominator (lm-eval: the choice without the target delimiter; lighteval: the continuation
    with its leading space). ``prompt``/``choices``/``delimiter`` keep the raw parts so perturbations can be
    applied to the prompt and/or the completion (robustness suite)."""

    doc_id: Union[int, str]
    contexts: List[str]
    continuations: List[str]
    gold: List[int]
    norm_texts: List[str]
    group: Optional[str] = None  # e.g. MMLU subject (macro averaging)
    prompt: Optional[str] = None
    choices: Optional[List[str]] = None
    delimiter: str = " "

    @property
    def requests(self):
        return list(zip(self.contexts, self.continuations))


@dataclass
class LMItem:
    """Single-continuation log-likelihood item (LAMBADA): acc = greedy, perplexity over items."""

    doc_id: Union[int, str]
    context: str
    continuation: str
    group: Optional[str] = None

    @property
    def requests(self):
        return [(self.context, self.continuation)]


Item = Union[MCItem, LMItem]


@dataclass
class DataSource:
    repo: str
    filename: str
    revision: str = "main"  # "refs/convert/parquet" for script-only repos


@dataclass
class Task:
    name: str  # e.g. "hellaswag"
    suite: str  # "paper7" | "smollm3_cf"
    source: DataSource
    build: Callable[[dict, int], Optional[Item]]
    kind: str = "mc"  # "mc" | "lm"
    primary: str = "acc_norm_char"  # metric reported in suite tables
    macro_by_group: bool = False  # MMLU: average over subjects
    rows: Optional[List[dict]] = None  # injected rows (tests); else downloaded
    meta: dict = field(default_factory=dict)

    @property
    def full_name(self):
        return f"{self.suite}/{self.name}"

    def load_rows(self) -> List[dict]:
        if self.rows is None:
            self.rows = load_parquet_rows(self.source)
        return self.rows

    def items(self, limit: Optional[int] = None, subset_seed: Optional[int] = None) -> List[Item]:
        """First ``limit`` valid items (lm-eval ``--limit``), or a seeded random subset of ``limit`` rows when
        ``subset_seed`` is given (unbiased; e.g. HellaSwag val starts with 3243 ActivityNet items)."""
        rows = list(enumerate(self.load_rows()))
        if subset_seed is not None and limit is not None:
            import random

            random.Random(subset_seed).shuffle(rows)
        out = []
        for i, row in rows:
            it = self.build(row, i)
            if it is not None:
                out.append(it)
                if limit is not None and len(out) >= limit:
                    break
        return out


_ROW_CACHE: Dict[tuple, List[dict]] = {}


def load_parquet_rows(src: DataSource) -> List[dict]:
    """Download one parquet file of a HF dataset repo (cached by huggingface_hub) and return its rows."""
    key = (src.repo, src.filename, src.revision)
    if key not in _ROW_CACHE:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq

        path = hf_hub_download(src.repo, src.filename, repo_type="dataset", revision=src.revision,
                               cache_dir=os.environ.get("HNET_EVAL_CACHE"))
        _ROW_CACHE[key] = pq.read_table(path).to_pylist()
    return _ROW_CACHE[key]


# Parquet files (the lighteval/lm-eval dataset repos; script-only repos read from the hub's parquet conversion).
SOURCES = {
    "hellaswag": DataSource("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"),
    "arc_easy": DataSource("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
    "arc_challenge": DataSource("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    "piqa": DataSource("ybisk/piqa", "plain_text/validation/0000.parquet", "refs/convert/parquet"),
    "winogrande": DataSource("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet"),
    "commonsenseqa": DataSource("tau/commonsense_qa", "data/validation-00000-of-00001.parquet"),
    "openbookqa_test": DataSource("allenai/openbookqa", "main/test-00000-of-00001.parquet"),
    "openbookqa_validation": DataSource("allenai/openbookqa", "main/validation-00000-of-00001.parquet"),
    "mmlu": DataSource("cais/mmlu", "all/test-00000-of-00001.parquet"),
    "boolq": DataSource("google/boolq", "data/validation-00000-of-00001.parquet"),
    "lambada_openai": DataSource("EleutherAI/lambada_openai", "default/test/default.parquet"),
}
