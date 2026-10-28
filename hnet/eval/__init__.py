"""Evaluation harness (WS-E): scorers, tasks, perturbations, bpb, statistics, reports.

    from hnet.eval.scorers import ByteModelScorer, TokenModelScorer, HFScorer
    from hnet.eval.harness import evaluate_suite
    results = evaluate_suite(scorer, "paper7", limit=100)

CLI: ``python -m hnet.eval.run --model <spec> --suite paper7|smollm3_cf|robustness|bpb|all --out results.json``.
"""
