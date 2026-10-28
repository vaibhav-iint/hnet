"""``smollm3_cf`` suite: SmolLM3's lighteval cloze-formulation (CF) prompts, re-implemented without lighteval.

Source: huggingface/smollm ``text/evaluation/smollm3/tasks.py`` (tasks ``hellaswag_cf``, ``arc_cf:{easy,challenge}``,
``piqa_cf``, ``winogrande_cf``, ``commonsenseqa_cf``, ``openbookqa_cf``, ``mmlu_cf:<subject>``, ``boolq_cf``) and
lighteval @ 2788496c (the commit pinned in SmolLM3's requirements.txt):
``tasks/templates/{multichoice,hellaswag,continuation,boolq}.py``, ``templates/utils/{formulation,formatting_utils}.py``,
``tasks/multilingual/adapters.py::winogrand_adapter``, ``tasks/default_prompts.py::hellaswag_preprocess``,
English ``TranslationLiterals`` (question "question", answer "answer", yes/no, all spaces " ").

MCQ CF (ARC, PIQA, CSQA, OBQA, MMLU, BoolQ)::

    "{Context}\\n" (BoolQ passage only) + "Question: {Question}\\nAnswer:"  |  " {Choice}"

where Question/Choice/Context go through ``capitalize(fix_ending_punct(.))`` (rstrip, first letter upper-cased).
HellaSwag CF: ``"{Activity label}:\\n{ctx_a} {ctx_b}"`` with lighteval's preprocessing, continuation
``" " + ending`` re-capitalized from the context (lower-cased unless the context ends a sentence).
Winogrande CF (continuation template): context = sentence before ``_`` (rstripped), continuation =
``" " + option + " " + rest`` (rest keeps its own leading space, so there are two spaces; the option's first letter is
lower-cased when the context does not end a sentence). This is lighteval's behaviour, reproduced verbatim; it is *not*
lm-eval's partial scoring.

Metrics (``qa_metrics``): ``acc`` with ``LogProbTokenNorm`` (logprob / #continuation tokens) and
``LogProbCharNorm`` (logprob / len(continuation incl. its leading space)); here ``acc_norm_token`` and
``acc_norm_char``. Datasets/splits as in tasks.py: HellaSwag val, ARC test, PIQA val, Winogrande xl val, CSQA val,
OBQA *validation*, MMLU test (all 57 subjects, macro average), BoolQ val.
"""

from __future__ import annotations

import re

from hnet.eval.tasks.base import SOURCES, MCItem, Task

SUITE = "smollm3_cf"
LETTER_INDICES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# English TranslationLiterals
FULL_STOP, COMMA, QMARK, EXCL, COLON, SEMICOLON = ".", ",", "?", "!", ":", ";"
WORD_SPACE = SENTENCE_SPACE = " "


def capitalize(w: str) -> str:
    return w[:1].upper() + w[1:]


def decapitalize(w: str) -> str:
    return w[:1].lower() + w[1:]


def fix_ending_punct(ctx: str) -> str:
    # English literals map every ending to itself, so this reduces to rstrip().
    return ctx.rstrip()


def punctuation_ends_sentence(text: str) -> bool:
    return text.rstrip().endswith((QMARK, FULL_STOP, EXCL, COLON, SEMICOLON))


def fix_capitalization(prefix: str, text: str) -> str:
    if len(prefix) == 0 or prefix.endswith("\n"):
        return capitalize(text)
    return capitalize(text) if punctuation_ends_sentence(prefix) else decapitalize(text)


def hellaswag_preprocess(text, wikihow_artifacts=(" [title]",), truncate_dots=False, strip_text=False,
                         dot_replacement=". "):
    """lighteval default_prompts.hellaswag_preprocess (note: ``truncate_dots`` is a literal-string no-op there)."""
    for a in wikihow_artifacts:
        text = text.replace(a, dot_replacement)
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    if truncate_dots:
        text = text.replace(r"\.+", r"\.")
    if strip_text:
        text = text.strip()
    return text


def mcq_cf(question, choices, gold, context=None, instruction=None):
    """lighteval get_mcq_prompt_function with CFFormulation -> (query, continuations, gold)."""
    instruction = f"{instruction}\n" if instruction else ""
    context = f"{capitalize(fix_ending_punct(context))}\n" if context else ""
    question = capitalize(fix_ending_punct(question))
    answers = [SENTENCE_SPACE + capitalize(fix_ending_punct(str(a))) for a in choices]
    query = f"{instruction}{context}{capitalize('question')}{COLON}{SENTENCE_SPACE}{question}\n{capitalize('answer')}{COLON}"
    return query, answers, gold if isinstance(gold, list) else [gold]


def continuation_cf(context, continuations, gold, instruction=None):
    """lighteval get_continuation_prompt_function with CFFormulation (fix_formatting=True)."""
    instruction = f"{instruction}\n" if instruction else ""
    context = capitalize(fix_ending_punct(context))
    conts = [fix_capitalization(context, fix_ending_punct(c)) for c in continuations]
    space = SENTENCE_SPACE if punctuation_ends_sentence(context) else WORD_SPACE
    return f"{instruction}{context}", [space + c for c in conts], gold if isinstance(gold, list) else [gold]


def _item(doc_id, query, answers, gold, group=None):
    return MCItem(doc_id=doc_id, contexts=[query] * len(answers), continuations=answers, gold=list(gold),
                  norm_texts=list(answers), group=group)


# ---------------------------------------------------------------------------------------------- tasks
def build_hellaswag(row, i):
    def process_context(ctx):
        if ctx == "":
            return ""
        return capitalize(fix_ending_punct(hellaswag_preprocess(ctx, truncate_dots=True, strip_text=True)))

    label = row["activity_label"]
    label = f"{capitalize(label)}:\n" if label else ""
    ctx_a, ctx_b = process_context(row["ctx_a"]), process_context(row.get("ctx_b", ""))
    if ctx_b:
        space = SENTENCE_SPACE if punctuation_ends_sentence(ctx_a) else WORD_SPACE
        ctx_a = f"{ctx_a.rstrip()}{space}{fix_capitalization(ctx_a, ctx_b)}"
    full = f"{label}{ctx_a}"
    choices = [hellaswag_preprocess(c, truncate_dots=True, strip_text=True, dot_replacement=FULL_STOP + SENTENCE_SPACE)
               for c in row["endings"]]
    if any(len(c.strip()) == 0 for c in choices):
        return None
    q, a, g = continuation_cf(full, choices, int(row["label"]))
    return _item(row.get("ind", i), q, a, g)


def build_arc(row, i):
    key = row["answerKey"]
    gold = int(key) - 1 if key.isdigit() else LETTER_INDICES.index(key)
    q, a, g = mcq_cf(row["question"], list(row["choices"]["text"]), gold)
    return _item(row.get("id", i), q, a, g)


def build_piqa(row, i):
    q, a, g = mcq_cf(row["goal"], [row["sol1"], row["sol2"]], int(row["label"]))
    return _item(i, q, a, g)


def build_commonsenseqa(row, i):
    gold = list(row["choices"]["label"]).index(row["answerKey"].strip())
    q, a, g = mcq_cf(row["question"], list(row["choices"]["text"]), gold)
    return _item(row.get("id", i), q, a, g)


def build_openbookqa(row, i):
    q, a, g = mcq_cf(row["question_stem"], list(row["choices"]["text"]), LETTER_INDICES.index(row["answerKey"]))
    return _item(row.get("id", i), q, a, g)


def build_mmlu(row, i):
    q, a, g = mcq_cf(row["question"], list(row["choices"]), int(row["answer"]))
    return _item(i, q, a, g, group=row.get("subject"))


def build_boolq(row, i):
    q, a, g = mcq_cf(row["question"], ["yes", "no"], 0 if row["answer"] else 1, context=row["passage"])
    return _item(i, q, a, g)


def build_winogrande(row, i):
    s = row["sentence"]
    if s.count("_") != 1:
        return None
    query, end = s.split("_")
    if len(query.strip()) == 0:
        return None
    conts = [f"{o}{WORD_SPACE}{end}" for o in (row["option1"], row["option2"])]
    q, a, g = continuation_cf(query, conts, int(row["answer"]) - 1)
    return _item(i, q, a, g)


TASKS = [
    Task("hellaswag", SUITE, SOURCES["hellaswag"], build_hellaswag),
    Task("arc_easy", SUITE, SOURCES["arc_easy"], build_arc),
    Task("arc_challenge", SUITE, SOURCES["arc_challenge"], build_arc),
    Task("piqa", SUITE, SOURCES["piqa"], build_piqa),
    Task("winogrande", SUITE, SOURCES["winogrande"], build_winogrande),
    Task("commonsenseqa", SUITE, SOURCES["commonsenseqa"], build_commonsenseqa),
    Task("openbookqa", SUITE, SOURCES["openbookqa_validation"], build_openbookqa),
    Task("mmlu", SUITE, SOURCES["mmlu"], build_mmlu, macro_by_group=True),
    Task("boolq", SUITE, SOURCES["boolq"], build_boolq),
]
