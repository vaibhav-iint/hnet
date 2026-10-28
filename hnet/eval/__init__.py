"""Evaluation harness (WS-E): scorers, tasks, perturbations, bpb, statistics, reports.

    from hnet.eval.scorers import ByteModelScorer, TokenModelScorer, HFScorer
    from hnet.eval.harness import evaluate_suite
    results = evaluate_suite(scorer, "paper7", limit=100)

CLI: ``python -m hnet.eval.run --model <spec> --suite paper7|smollm3_cf|robustness|bpb|all --out results.json``.
"""


def evaluate_model(model, model_cfg=None, kind: str = "bytes", device=None, **kw) -> dict:
    """Trainer hook (``eval.downstream`` config section): flat {metric: float}. See ``hnet.eval.run.evaluate_model``."""
    from hnet.eval.run import evaluate_model as _f

    return _f(model, model_cfg=model_cfg, kind=kind, device=device, **kw)
