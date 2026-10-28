"""scripts/speedrun: target conversion and the time/FLOPs-to-target report (synthetic inputs, no downloads)."""

import importlib.util
import json
import math
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "speedrun", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_target_conversion_on_synthetic_shard(tmp_path):
    shards = pytest.importorskip("hnet.data.shards")
    target = load_script("target")
    # tokens: EOT, then 'hello' (31373) and ' world' (995) repeated; every target after position 0 counted
    n = 12
    toks = np.array([50256] + [31373, 995] * ((n + 1) // 2), dtype=np.uint16)[: n + 1]
    path = tmp_path / "val.bin"
    shards.write_token_shard(path, toks)
    res = target.compute(str(path), losses=(3.28,), n_targets=n)
    n_bytes = int(shards.token_byte_lengths(toks[1 : n + 1]).sum())
    assert n_bytes == 6 * 5 + 6 * 6  # "hello" + " world"
    assert res["n_bytes"] == n_bytes and res["n_eot_targets"] == 0
    assert res["target_bpb"]["3.28"] == pytest.approx(3.28 * n / math.log(2) / n_bytes)
    assert res["target_bpb_eot_free"]["3.28"] == pytest.approx(res["target_bpb"]["3.28"])


def write_run(d, bpbs, step_time=2.0, fpb=1e9, bytes_per_step=1e6):
    d.mkdir()
    with open(d / "metrics.jsonl", "w") as f:
        for s in range(10, 101, 10):
            f.write(json.dumps({"step": s, "step_time": step_time, "bytes_seen": s * bytes_per_step,
                                "tokens_eq": s * bytes_per_step / 4, "flops_per_byte": fpb}) + "\n")
    with open(d / "eval.jsonl", "w") as f:
        for s, b in zip(range(25, 101, 25), bpbs):
            f.write(json.dumps({"step": s, "tokens_eq": s * bytes_per_step / 4, "bytes_seen": s * bytes_per_step,
                                "val/bpb": b}) + "\n")
    return str(d)


def test_report_interpolates_crossing(tmp_path, capsys):
    rep = load_script("speedrun_report")
    a = write_run(tmp_path / "a", [1.4, 1.2, 1.0, 0.9])  # crosses 1.1 halfway between steps 50 and 75
    b = write_run(tmp_path / "b", [1.5, 1.4, 1.3, 1.2])  # never reaches it
    rows = rep.report([a, b], ["A", "B"], target=1.1)
    ra, rb = rows
    assert ra["reached"] and not rb["reached"] and rb["best_bpb"] == 1.2
    assert ra["step_to_target"] == pytest.approx(62.5)
    assert ra["time_s_to_target"] == pytest.approx(125.0)
    assert ra["bytes_to_target"] == pytest.approx(62.5e6)
    assert ra["flops_to_target"] == pytest.approx(62.5e6 * 1e9)
    rep.print_table(rows, 1.1)
    out = capsys.readouterr().out
    assert "| A | yes |" in out and "| B | no |" in out
    rep.main([a, b, "--target-bpb", "1.1", "--json", str(tmp_path / "r.json"), "--plot", str(tmp_path / "c.png")])
    assert json.load(open(tmp_path / "r.json"))[0]["run"] == "a"
    assert (tmp_path / "c.png").stat().st_size > 0
