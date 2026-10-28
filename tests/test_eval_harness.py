"""End-to-end harness on fake scorers and injected rows (no network)."""

import json
import math

import numpy as np
import pytest
import torch

from hnet.eval.harness import evaluate_robustness, evaluate_suite, score_items
from hnet.eval.report import results_tables
from hnet.eval.scorers import ByteModelScorer, LLResult
from hnet.eval.tasks import MCItem, get_task

torch.set_num_threads(1)


class ScriptedScorer:
    """logprob(ctx, cont) = -a * len(cont) + (b if 'GOOD' in cont else 0); n_units = word count."""

    unit = "token"

    def __init__(self, a=0.5, b=100.0):
        self.a, self.b = a, b

    def loglikelihood_detailed(self, reqs, progress=False):
        out = []
        for ctx, cont in reqs:
            lp = -self.a * len(cont) + (self.b if "GOOD" in cont else 0.0)
            out.append(LLResult(lp, "GOOD" in cont, len(cont.encode()), len(cont.split())))
        return out


def test_score_items_metrics():
    sc = ScriptedScorer(b=1.0)
    it = MCItem(0, ["q"] * 3, [" GOOD but very long answer", " short", " GOOD"], [0], [" GOOD but very long answer", " short", " GOOD"])
    (r,) = score_items(sc, [it])
    lp = np.array(r["lp"])
    assert r["pred"]["acc"] == int(np.argmax(lp)) == 2 and r["acc"] == 0.0
    # char norm: lp/len -> -0.5 + 1/len: choice 0 has the largest (least negative) value? compute explicitly
    exp = int(np.argmax(lp / np.array([26, 6, 5])))
    assert r["pred"]["acc_norm_char"] == exp
    assert r["acc_norm_char"] == float(exp == 0)
    assert r["gold_nll_bits"] == pytest.approx(-lp[0] / math.log(2)) and r["gold_bytes"] == 26
    assert "acc_norm_token" in r


def _hs_rows(n=12):
    rows = []
    for i in range(n):
        endings = ["ends badly here.", "GOOD ending.", "another bad ending that is long.", "meh."]
        rows.append({"ind": i, "activity_label": "Sport", "ctx_a": f"A person does thing {i}.", "ctx_b": "they",
                     "endings": endings, "label": str(1 if i % 3 else 0)})
    return rows


def test_evaluate_suite_paper7_subset_and_averages():
    t_hs = get_task("paper7/hellaswag")
    t_hs.rows = _hs_rows()
    t_lmb = get_task("paper7/lambada_openai")
    t_lmb.rows = [{"text": "the word is GOOD"}, {"text": "the word is bad"}]
    summary, recs = evaluate_suite(ScriptedScorer(), "paper7", task_objs=[t_lmb, t_hs], n_boot=50)
    hs = summary["tasks"]["hellaswag"]["metrics"]
    assert hs["acc"]["value"] == pytest.approx(8 / 12)  # gold == 1 for 8 of 12 rows
    assert summary["tasks"]["lambada_openai"]["metrics"]["acc"]["value"] == 0.5
    lp = [-0.5 * 5 + 100, -0.5 * 4]
    assert summary["tasks"]["lambada_openai"]["metrics"]["perplexity"]["value"] == pytest.approx(math.exp(-np.mean(lp)))
    assert summary["averages"] == {}  # incomplete suite -> no 7-task average
    assert len(recs["hellaswag"]) == 12


def test_smollm3_derived_and_macro():
    arc = {"question": "q?", "choices": {"text": ["GOOD", "bad"], "label": ["A", "B"]}, "answerKey": "A"}
    t_e, t_c = get_task("smollm3_cf/arc_easy"), get_task("smollm3_cf/arc_challenge")
    t_e.rows = [arc] * 4
    t_c.rows = [dict(arc, answerKey="B")] * 4
    t_m = get_task("smollm3_cf/mmlu")
    t_m.rows = [{"question": "q", "subject": "s1", "choices": ["GOOD", "x", "y", "z"], "answer": 0}] * 9 + \
               [{"question": "q", "subject": "s2", "choices": ["GOOD", "x", "y", "z"], "answer": 1}]
    summary, _ = evaluate_suite(ScriptedScorer(), "smollm3_cf", task_objs=[t_e, t_c, t_m], n_boot=30)
    assert summary["derived"]["arc"]["acc"]["value"] == pytest.approx(0.5)
    m = summary["tasks"]["mmlu"]["metrics"]
    assert m["acc"]["value"] == pytest.approx(0.5)  # macro: (1 + 0) / 2
    assert m["acc_micro"]["value"] == pytest.approx(0.9)
    md = results_tables([{"model": "fake", "suites": {"smollm3_cf": summary}}])
    assert "arc (mean)" in md and "| fake |" in md


def test_robustness_with_byte_model():
    from hnet.eval.fakes import SegmentSumModel

    t = get_task("paper7/hellaswag")
    t.rows = _hs_rows(6)
    sc = ByteModelScorer(SegmentSumModel(), device="cpu", batch_tokens=4096)
    s, recs = evaluate_robustness(sc, task=t, n_boot=20, modes=("prompt", "both"), perturbations=("uppercase", "drop"))
    assert set(s["variants"]) == {"clean", "uppercase/prompt", "uppercase/both", "drop/prompt", "drop/both"}
    avg = np.mean([s["perturbations"][p]["value"] for p in ("uppercase", "drop")])
    assert s["average"]["value"] == pytest.approx(avg)
    clean = s["clean"]["value"]
    exp = 100 * (avg - 0.25) / (clean - 0.25) if clean > 0.25 else float("nan")
    got = s["robustness_score"]["value"]
    assert (math.isnan(exp) and math.isnan(got)) or got == pytest.approx(exp)
    json.dumps(s)  # serializable


def test_trainer_hook_flat_floats(monkeypatch):
    import hnet.eval as he
    from hnet.eval import harness
    from hnet.eval.fakes import SegmentSumModel

    t = get_task("paper7/hellaswag")
    t.rows = _hs_rows(4)
    monkeypatch.setattr(harness, "suite_tasks", lambda suite, only=None: [t])
    out = he.evaluate_model(SegmentSumModel(), kind="bytes", device="cpu", suites=["paper7"], limit=4)
    assert out and all(isinstance(v, float) for v in out.values())
    assert "paper7/hellaswag/acc_norm_char" in out
