"""``paper7`` suite: lm-evaluation-harness zero-shot prompts for the H-Net paper's Table 2.

Reproduces the lm-eval task yamls (EleutherAI/lm-evaluation-harness, ``lm_eval/tasks/{lambada,hellaswag,piqa,
arc,winogrande,openbookqa}``): requests are ``(doc_to_text, target_delimiter + choice)`` with
``target_delimiter = " "``; ``acc_norm`` divides by ``len(choice)`` (without the delimiter), as in
``ConfigurableTask.process_results``.

| task           | split            | context                                 | continuation            | Table 2 |
|----------------|------------------|-----------------------------------------|-------------------------|---------|
| lambada_openai | test (5153)      | text minus last word                    | " " + last word         | acc     |
| hellaswag      | validation       | preprocess(label + ": " + ctx_a + " " + ctx_b.capitalize()) | " " + preprocess(ending) | acc_norm |
| piqa           | validation       | "Question: {goal}\\nAnswer:"            | " " + sol               | acc     |
| arc_easy       | test             | "Question: {question}\\nAnswer:"        | " " + choice            | acc     |
| arc_challenge  | test             | same                                    | same                    | acc_norm|
| winogrande     | validation (xl)  | sentence[:blank] + option   (partial)   | " " + sentence[blank+1:].strip() | acc |
| openbookqa     | test             | question_stem                           | " " + choice            | acc_norm|
"""

from __future__ import annotations

import re

from hnet.eval.tasks.base import SOURCES, LMItem, MCItem, Task

SUITE = "paper7"


def hellaswag_preprocess(text: str) -> str:
    """lm_eval/tasks/hellaswag/utils.py::preprocess."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def _mc(doc_id, ctx, choices, gold, group=None, delim=" "):
    return MCItem(doc_id=doc_id, contexts=[ctx] * len(choices), continuations=[delim + c for c in choices],
                  gold=[int(gold)], norm_texts=list(choices), group=group, prompt=ctx, choices=list(choices),
                  delimiter=delim)


def build_hellaswag(row, i):
    ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
    query = hellaswag_preprocess(row["activity_label"] + ": " + ctx)
    choices = [hellaswag_preprocess(e) for e in row["endings"]]
    return _mc(row.get("ind", i), query, choices, int(row["label"]))


def build_arc(row, i):
    ctx = f"Question: {row['question']}\nAnswer:"
    labels = list(row["choices"]["label"])
    return _mc(row.get("id", i), ctx, list(row["choices"]["text"]), labels.index(row["answerKey"]))


def build_piqa(row, i):
    return _mc(i, f"Question: {row['goal']}\nAnswer:", [row["sol1"], row["sol2"]], int(row["label"]))


def build_openbookqa(row, i):
    labels = list(row["choices"]["label"])
    return _mc(row.get("id", i), row["question_stem"], list(row["choices"]["text"]),
               labels.index(row["answerKey"].lstrip()))


def build_winogrande(row, i):
    """Partial evaluation (lm-eval preprocess_winogrande.py): the two contexts differ (sentence up to the blank
    with each option filled in); the shared continuation after the blank is scored."""
    s = row["sentence"]
    idx = s.index("_")
    target = s[idx + 1:].strip()
    contexts = [s[:idx] + row["option1"], s[:idx] + row["option2"]]
    gold = {"1": 0, "2": 1}[str(row["answer"])]
    return MCItem(doc_id=i, contexts=contexts, continuations=[" " + target] * 2, gold=[gold],
                  norm_texts=[target, target])


def build_lambada(row, i):
    words = row["text"].split(" ")
    return LMItem(doc_id=i, context=" ".join(words[:-1]), continuation=" " + words[-1])


TASKS = [
    Task("lambada_openai", SUITE, SOURCES["lambada_openai"], build_lambada, kind="lm", primary="acc"),
    Task("hellaswag", SUITE, SOURCES["hellaswag"], build_hellaswag, primary="acc_norm_char"),
    Task("piqa", SUITE, SOURCES["piqa"], build_piqa, primary="acc"),
    Task("arc_easy", SUITE, SOURCES["arc_easy"], build_arc, primary="acc"),
    Task("arc_challenge", SUITE, SOURCES["arc_challenge"], build_arc, primary="acc_norm_char"),
    Task("winogrande", SUITE, SOURCES["winogrande"], build_winogrande, primary="acc"),
    Task("openbookqa", SUITE, SOURCES["openbookqa_test"], build_openbookqa, primary="acc_norm_char"),
]
