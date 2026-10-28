"""Prompt-construction snapshots. Expected strings are derived by hand from the lm-eval task yamls/utils
(paper7) and lighteval @2788496c templates used by SmolLM3's tasks.py (smollm3_cf); rows are real dataset rows."""

import pytest

from hnet.eval.tasks import REGISTRY, SUITES, get_task

HS_ROOF = {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he",
           "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                       "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "label": "3"}
HS_WIKI = {"ind": 1, "activity_label": "Personal Care and Style",
           "ctx_a": "[header] How to become a fashion consultant [title] Obtain your high school diploma or ged. [step] "
                    "This job requires a high school diploma.",
           "ctx_b": "",
           "endings": ["[substeps] Many high schools have classes. [title] Search for positions.",
                       "Having an associate's degree helps.", "you should also be able to show the world.",
                       "[title] Earn a bachelor's degree in a fashion-related field. [step] Degrees include art."],
           "label": "3"}
ARC = {"id": "Mercury_417466", "question": "Which statement best explains why photosynthesis is the foundation of most food webs?",
       "choices": {"text": ["Sunlight is the source of energy for nearly all ecosystems.",
                            "Most ecosystems are found on land instead of in water.",
                            "Carbon dioxide is more available than other gases.", "The producers in all ecosystems are plants."],
                   "label": ["A", "B", "C", "D"]}, "answerKey": "A"}
ARC_NUM = {"id": "x", "question": "what is 1+1? ", "choices": {"text": ["one", "two"], "label": ["1", "2"]}, "answerKey": "2"}
PIQA = {"goal": "How do I ready a guinea pig cage for it's new occupants?", "sol1": "provide bedding made of paper.",
        "sol2": "Provide bedding made of jeans.", "label": 0}
WINO = {"sentence": "Sarah was a much better surgeon than Maria so _ always got the easier cases.", "option1": "Sarah",
        "option2": "Maria", "answer": "2"}
OBQA = {"id": "8-376", "question_stem": "Frilled sharks and angler fish live far beneath the surface of the ocean, which is why they are known as",
        "choices": {"text": ["Deep sea animals", "fish", "Long Sea Fish", "Far Sea Animals"], "label": ["A", "B", "C", "D"]},
        "answerKey": "A"}
CSQA = {"id": "1afa", "question": "A revolving door is convenient for two direction travel, but it also serves as a security measure at a what?",
        "choices": {"label": ["A", "B", "C", "D", "E"], "text": ["bank", "library", "department store", "mall", "new york"]},
        "answerKey": "A"}
MMLU = {"question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
        "subject": "abstract_algebra", "choices": ["0", "4", "2", "6"], "answer": 1}
BOOLQ = {"question": "does ethanol take more energy make that produces", "answer": False,
         "passage": "all biomass goes through at least some of these steps.  "}
LAMBADA = {"text": "He looked at her, feeling stunned. I don't care about signs"}


def build(name, row):
    return get_task(name).build(row, 0)


def test_registry_complete():
    assert SUITES["paper7"] == ["lambada_openai", "hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande",
                                "openbookqa"]
    assert set(SUITES["smollm3_cf"]) == {"hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande",
                                         "commonsenseqa", "openbookqa", "mmlu", "boolq"}
    assert all(name in REGISTRY for name in ["paper7/hellaswag", "smollm3_cf/mmlu"])


# ----------------------------------------------------------------------------------------- paper7 (lm-eval)
def test_paper7_hellaswag():
    it = build("paper7/hellaswag", HS_ROOF)
    assert it.contexts == ["Roof shingle removal: A man is sitting on a roof. He"] * 4
    assert it.continuations[0] == " is using wrap to wrap a pair of skis."
    assert it.norm_texts[0] == "is using wrap to wrap a pair of skis." and it.gold == [3]
    it = build("paper7/hellaswag", HS_WIKI)
    assert it.contexts[0] == ("Personal Care and Style: How to become a fashion consultant. Obtain your high school "
                              "diploma or ged. This job requires a high school diploma.")  # strip() removes the " " + ""
    # lm-eval strips *before* removing "[substeps]", leaving a leading space (double space after the delimiter)
    assert it.continuations[0] == "  Many high schools have classes.. Search for positions."
    assert it.continuations[3] == "  Earn a bachelor's degree in a fashion-related field. Degrees include art."
    assert it.continuations[2] == " you should also be able to show the world."


def test_paper7_arc_piqa_obqa():
    it = build("paper7/arc_easy", ARC)
    assert it.contexts[0] == ("Question: Which statement best explains why photosynthesis is the foundation of most "
                              "food webs?\nAnswer:")
    assert it.continuations[1] == " Most ecosystems are found on land instead of in water." and it.gold == [0]
    assert build("paper7/arc_challenge", ARC_NUM).gold == [1]
    assert build("paper7/arc_challenge", ARC_NUM).contexts[0] == "Question: what is 1+1? \nAnswer:"
    it = build("paper7/piqa", PIQA)
    assert it.contexts[0] == "Question: How do I ready a guinea pig cage for it's new occupants?\nAnswer:"
    assert it.continuations == [" provide bedding made of paper.", " Provide bedding made of jeans."]
    it = build("paper7/openbookqa", OBQA)
    assert it.contexts[0] == OBQA["question_stem"]
    assert it.continuations[1] == " fish" and it.norm_texts[1] == "fish" and it.gold == [0]


def test_paper7_winogrande_partial():
    it = build("paper7/winogrande", WINO)
    assert it.contexts == ["Sarah was a much better surgeon than Maria so Sarah",
                           "Sarah was a much better surgeon than Maria so Maria"]
    assert it.continuations == [" always got the easier cases."] * 2 and it.gold == [1]


def test_paper7_lambada():
    it = build("paper7/lambada_openai", LAMBADA)
    assert it.context == "He looked at her, feeling stunned. I don't care about"
    assert it.continuation == " signs"


# ----------------------------------------------------------------------------------------- smollm3_cf (lighteval)
def test_cf_hellaswag():
    it = build("smollm3_cf/hellaswag", HS_ROOF)
    assert it.contexts[0] == "Roof shingle removal:\nA man is sitting on a roof. He"
    assert it.continuations[2] == " is holding a rubik's cube."
    assert it.norm_texts == it.continuations  # LogProbCharNorm counts the leading space
    it = build("smollm3_cf/hellaswag", HS_WIKI)
    assert it.contexts[0] == ("Personal Care and Style:\nHow to become a fashion consultant. Obtain your high school "
                              "diploma or ged. This job requires a high school diploma.")
    assert it.continuations[0] == " Many high schools have classes.. Search for positions."
    assert it.continuations[2] == " You should also be able to show the world."  # capitalized after a full stop
    assert it.continuations[3] == " Earn a bachelor's degree in a fashion-related field. Degrees include art."


def test_cf_hellaswag_decapitalizes_mid_sentence():
    row = dict(HS_ROOF, endings=["Is using wrap.", "is ripping.", "x", "y"])
    it = build("smollm3_cf/hellaswag", row)
    assert it.continuations[0] == " is using wrap."
    assert build("smollm3_cf/hellaswag", dict(HS_ROOF, endings=["[title]", "a", "b", "c"])) is None  # empty ending


def test_cf_mcq_tasks():
    q = "Question: Which statement best explains why photosynthesis is the foundation of most food webs?\nAnswer:"
    it = build("smollm3_cf/arc_easy", ARC)
    assert it.contexts[0] == q and it.continuations[0] == " Sunlight is the source of energy for nearly all ecosystems."
    it = build("smollm3_cf/arc_challenge", ARC_NUM)
    assert it.contexts[0] == "Question: What is 1+1?\nAnswer:" and it.continuations == [" One", " Two"]
    assert it.gold == [1]
    it = build("smollm3_cf/piqa", PIQA)
    assert it.continuations == [" Provide bedding made of paper.", " Provide bedding made of jeans."]
    it = build("smollm3_cf/openbookqa", OBQA)
    assert it.contexts[0] == ("Question: Frilled sharks and angler fish live far beneath the surface of the ocean, "
                              "which is why they are known as\nAnswer:")
    assert it.continuations == [" Deep sea animals", " Fish", " Long Sea Fish", " Far Sea Animals"]
    it = build("smollm3_cf/commonsenseqa", CSQA)
    assert it.continuations == [" Bank", " Library", " Department store", " Mall", " New york"] and it.gold == [0]
    it = build("smollm3_cf/mmlu", MMLU)
    assert it.contexts[0] == ("Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) "
                              "over Q.\nAnswer:")
    assert it.continuations == [" 0", " 4", " 2", " 6"] and it.gold == [1] and it.group == "abstract_algebra"


def test_cf_boolq():
    it = build("smollm3_cf/boolq", BOOLQ)
    assert it.contexts[0] == ("All biomass goes through at least some of these steps.\n"
                              "Question: Does ethanol take more energy make that produces\nAnswer:")
    assert it.continuations == [" Yes", " No"] and it.gold == [1]


def test_cf_winogrande_lighteval_quirks():
    it = build("smollm3_cf/winogrande", WINO)
    assert it.contexts == ["Sarah was a much better surgeon than Maria so"] * 2
    # lighteval: option + word_space + rest (rest keeps its leading space), first letter de-capitalized
    assert it.continuations == [" sarah  always got the easier cases.", " maria  always got the easier cases."]
    assert it.gold == [1]
    assert build("smollm3_cf/winogrande", dict(WINO, sentence="no blank here")) is None


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_task_builds_requests(name):
    rows = {"hellaswag": HS_ROOF, "arc_easy": ARC, "arc_challenge": ARC, "piqa": PIQA, "winogrande": WINO,
            "openbookqa": OBQA, "commonsenseqa": CSQA, "mmlu": MMLU, "boolq": BOOLQ, "lambada_openai": LAMBADA}
    it = build(name, rows[name.split("/")[1]])
    for ctx, cont in it.requests:
        assert isinstance(ctx, str) and cont.startswith(" ") and len(cont) > 1
