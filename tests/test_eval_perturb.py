import math

import pytest

from hnet.eval.perturb import PERTURBATIONS, perturb, perturb_mc_item, robustness_score
from hnet.eval.tasks import get_task

TEXT = "A man is sitting on a roof. He starts pulling up roofing on a roof."


def test_deterministic_and_seeded():
    for p in PERTURBATIONS:
        assert perturb(TEXT, p, seed=0, key=1) == perturb(TEXT, p, seed=0, key=1)
    for p in ("drop", "randomcase", "repeat"):
        outs = {perturb(TEXT, p, seed=s, key=1) for s in range(5)}
        assert len(outs) > 1, p
        assert perturb(TEXT, p, 0, key=1) != perturb(TEXT, p, 0, key=2)


def test_definitions():
    assert perturb("an owl", "antspeak") == "A N   O W L"
    assert perturb(TEXT, "uppercase") == TEXT.upper()
    rc = perturb(TEXT * 20, "randomcase", seed=3)
    assert rc.lower() == (TEXT * 20).lower()
    letters = [c for c in rc if c.isalpha()]
    frac_upper = sum(c.isupper() for c in letters) / len(letters)
    assert 0.4 < frac_upper < 0.6
    long = TEXT * 50
    d = perturb(long, "drop", seed=1)
    assert 0.85 < len(d) / len(long) < 0.95  # ~10% dropped
    # dropped text is a subsequence of the original
    it = iter(long)
    assert all(c in it for c in d)
    r = perturb(long, "repeat", seed=1)
    # ~20% of characters get 1-3 (mean 2) extra copies -> ~ +40% length
    assert 1.3 < len(r) / len(long) < 1.5
    assert perturb("abc", "drop", seed=0, key="x") != ""


def test_never_more_than_four_repeats_added_by_repeat():
    s = "abcdefghij" * 30
    r = perturb(s, "repeat", seed=7)
    # every run of identical chars in the output comes from one input char (input has no repeats) => <= 4
    run, best = 1, 1
    for a, b in zip(r, r[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    assert best <= 4


def test_perturb_item_modes():
    row = {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he",
           "endings": ["is using wrap.", "is ripping tiles off.", "is holding a cube.", "starts pulling up roofing."],
           "label": "3"}
    it = get_task("paper7/hellaswag").build(row, 0)
    p = perturb_mc_item(it, "uppercase", "prompt")
    assert p.contexts[0] == it.contexts[0].upper() and p.continuations == it.continuations
    c = perturb_mc_item(it, "uppercase", "completion")
    assert c.contexts == it.contexts and c.continuations[0] == " IS USING WRAP." and c.norm_texts[0] == "IS USING WRAP."
    b = perturb_mc_item(it, "drop", "both", seed=0)
    assert b.contexts[0] == perturb_mc_item(it, "drop", "prompt", seed=0).contexts[0]
    assert b.continuations == perturb_mc_item(it, "drop", "completion", seed=0).continuations
    assert b.gold == it.gold


def test_robustness_score():
    assert robustness_score(0.25, 0.55) == pytest.approx(0.0)
    assert robustness_score(0.55, 0.55) == pytest.approx(100.0)
    assert robustness_score(0.309, 0.545) == pytest.approx(20.0, abs=0.01)  # paper Table 3, Transformer-L
    assert math.isnan(robustness_score(0.3, 0.24))
