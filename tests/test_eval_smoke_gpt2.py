"""Real smoke test: GPT-2 small (HF) on 50 HellaSwag val items. Downloads ~550 MB once; marked slow.

    OMP_NUM_THREADS=4 python -m pytest -q -s tests/test_eval_smoke_gpt2.py -m slow

Reference: lm-eval reports GPT-2 small HellaSwag acc_norm = 31.1% (acc 28.9%) on the full 10,042-item val set;
on 50 items the 95% CI is roughly +-13 points.
"""

import pytest
import torch

pytestmark = pytest.mark.slow


def test_gpt2_hellaswag_50():
    pytest.importorskip("transformers")
    from hnet.eval.harness import evaluate_suite
    from hnet.eval.scorers import HFScorer, TokenModelScorer

    torch.set_num_threads(4)
    sc = HFScorer("gpt2", device="cpu", batch_tokens=8192)
    assert sc.add_bos is False and sc.prefix_id == 50256  # lm-eval GPT-2 behaviour
    out = {}
    for suite in ("paper7", "smollm3_cf"):
        s, recs = evaluate_suite(sc, suite, tasks=["hellaswag"], limit=50, n_boot=200)
        m = s["tasks"]["hellaswag"]["metrics"]
        out[suite] = {k: round(100 * m[k]["value"], 1) for k in ("acc", "acc_norm_char", "acc_norm_token")}
        out[suite]["soft_bpb"] = round(m["soft_bpb"]["value"], 3)
        assert len(recs["hellaswag"]) == 50
    print("\nGPT-2 small, HellaSwag first 50 val items:", out)
    assert 0.15 <= out["paper7"]["acc_norm_char"] / 100 <= 0.55
    # same tokenization through tiktoken (our TokenModelScorer path) with the HF weights
    tk = TokenModelScorer(sc.model, tokenizer="gpt2", add_bos=False, device="cpu", max_len=1024)
    tk._logits = sc._logits  # HF forward, tiktoken encoding
    reqs = [("Roof shingle removal: A man is sitting on a roof. He", " starts pulling up roofing on a roof.")]
    a, b = sc.loglikelihood(reqs)[0], tk.loglikelihood(reqs)[0]
    assert a[0] == pytest.approx(b[0], abs=1e-3)
