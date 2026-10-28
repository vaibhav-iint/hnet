"""Task registry: ``get_task("paper7/hellaswag")``, ``suite_tasks("smollm3_cf")``.

Suites
* ``paper7``     lm-eval-harness prompts/metrics of the H-Net paper's Table 2 (``lmeval.py``);
* ``smollm3_cf`` SmolLM3 lighteval CF prompts (``lighteval_cf.py``);
* ``robustness`` HellaSwag (paper7 prompt) under the 5 BLT perturbations (built in ``hnet.eval.harness``).
"""

from __future__ import annotations

import copy
from typing import Dict, List

from hnet.eval.tasks import lighteval_cf, lmeval
from hnet.eval.tasks.base import SOURCES, DataSource, Item, LMItem, MCItem, Task, load_parquet_rows  # noqa: F401

REGISTRY: Dict[str, Task] = {t.full_name: t for t in lmeval.TASKS + lighteval_cf.TASKS}

SUITES = {
    "paper7": [t.name for t in lmeval.TASKS],
    "smollm3_cf": [t.name for t in lighteval_cf.TASKS],
}

# Averages reported per suite. smollm3 "average6" = Cartesia's blog Fig. 7 six-task average
# (HellaSwag, ARC = mean(easy, challenge), PIQA, Winogrande, CSQA, MMLU).
SUITE_AVERAGES = {
    "paper7": {"average": ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande",
                           "openbookqa"]},
    "smollm3_cf": {"average6": ["hellaswag", "arc", "piqa", "winogrande", "commonsenseqa", "mmlu"]},
}
DERIVED = {"smollm3_cf": {"arc": ["arc_easy", "arc_challenge"]}}


def get_task(name: str) -> Task:
    """``suite/task``; returns a fresh copy (rows can be injected without touching the registry)."""
    return copy.copy(REGISTRY[name])


def suite_tasks(suite: str, only: List[str] = None) -> List[Task]:
    names = SUITES[suite] if not only else [n for n in SUITES[suite] if n in only]
    return [get_task(f"{suite}/{n}") for n in names]
