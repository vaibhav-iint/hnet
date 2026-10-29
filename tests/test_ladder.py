"""Compute-optimal ladder (scripts/ladder/make_ladder.py): committed files are up to date, every generated model
config loads and builds (meta device), the BPE rungs follow the blog recipe, each H-Net conversion keeps the identical
main Transformer and matches its BPE rung in FLOPs/byte (total and outer vs embedding + head) and parameters, the
iso-FLOP runs spend their budget, and the fairness gate passes for every spec of every budget."""

import glob
import json
import os
import sys

import pytest
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts", "search"), os.path.join(REPO, "scripts", "ladder")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_ladder as ML  # noqa: E402

from hnet.config import DenseConfig, HNetConfig, model_config_from_yaml  # noqa: E402
from hnet.flops import count_params, model_flops  # noqa: E402

MANIFEST = json.load(open(os.path.join(REPO, ML.MANIFEST)))
OPTS = ML.opts_from(MANIFEST["options"])
RUNGS = {r["name"]: r for r in MANIFEST["rungs"]}
MODEL_FILES = sorted(glob.glob(os.path.join(REPO, ML.MODELS_DIR, "*.yaml")))
SPECS = sorted(glob.glob(os.path.join(REPO, ML.SEARCH_DIR, "ladder_*.yaml")))


def model_path(variant, rung):
    return os.path.join(REPO, ML.MODELS_DIR, f"{variant}_{rung}.yaml")


def test_committed_files_match_the_generator():
    files = ML.generate(OPTS)
    stale = [p for p, t in files.items() if open(os.path.join(REPO, p)).read() != t]
    extra = [p for p in ML.existing_generated(ML.Path(REPO)) if p not in files]
    assert not stale and not extra, f"rerun {ML.GEN}: stale {stale}, not generated {extra}"
    assert len(MODEL_FILES) == 3 * len(OPTS.rungs)
    assert len(SPECS) == sum(len(v) for v in MANIFEST["budgets"].values()) + 1  # + the LR sweep


@pytest.mark.parametrize("path", MODEL_FILES, ids=os.path.basename)
def test_model_configs_load_and_build_on_meta(path):
    from hnet.model import build_model

    cfg = model_config_from_yaml(path)
    with torch.device("meta"):
        model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == count_params(cfg)
    meta = yaml.safe_load(open(path))["ladder"]
    assert meta["bytes_per_token"] == OPTS.bytes_per_token and meta["vocab"] == OPTS.vocab


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_bpe_rung_follows_the_blog_recipe(rung):
    L = RUNGS[rung]["L"]
    c = model_config_from_yaml(model_path("bpe", rung))
    assert isinstance(c, DenseConfig) and c.n_layers == L and c.vocab_size == OPTS.vocab
    assert c.d_model % 128 == 0 and c.d_model == max(OPTS.d_floor, ML.nearest_multiple(75 * L))
    assert abs(c.d_model - 75 * L) <= 64 or c.d_model == OPTS.d_floor
    assert c.num_heads == L and c.head_dim == 128 and c.num_kv_heads is None
    assert c.d_intermediate % 128 == 0 and abs(c.d_intermediate - 3.6 * c.d_model) <= 64
    assert not c.tie_embeddings and c.mixer == "attn"


@pytest.mark.parametrize("rung", sorted(RUNGS))
def test_hnet_conversion_matches_its_bpe_rung(rung):
    b = model_config_from_yaml(model_path("bpe", rung))
    h = model_config_from_yaml(model_path("hnet", rung))
    ws = model_config_from_yaml(model_path("hnet_ws", rung))
    bpt, tok, byt = OPTS.bytes_per_token, OPTS.seq_tokens, OPTS.seq_bytes
    # identical main network
    assert isinstance(h, HNetConfig) and h.n_stages == 1 and h.stage_layout(1) == [f"T{b.n_layers}"]
    assert h.d_model[1] == b.d_model and h.d_intermediate[1] == b.d_intermediate
    a = h.attn_for_stage(1)
    assert a["num_heads"] == b.num_heads and a["head_dim"] == b.head_dim and a["num_kv_heads"] == b.num_heads
    assert a["window_size"] == -1 and a["rotary_emb_dim"] == b.head_dim
    rb = model_flops(b, seq_len=tok, bytes_per_token=bpt)
    rh = model_flops(h, seq_len=byt)
    s0, sm = rh["per_stage"]
    assert sm["params"] - sm["components"]["pad_dimension"]["params"] == rb["params_nonembed"]
    # compression pinned at bytes/token with the rank decider
    assert h.dc.target_ratios == [bpt] and h.dc.decision == "quantile" and h.dc.router == "end_mlp"
    assert h.dc.router_prior_weight == 0.0 and ws.dc.router_prior_weight == 1.0 and ws.dc.router_prior_frac == 0.1
    assert {**ws.to_dict(), "dc": None} == {**h.to_dict(), "dc": None}
    assert {**ws.to_dict()["dc"], "router_prior_weight": 0.0} == h.to_dict()["dc"]
    # outer (stage 0 incl. byte head) vs BPE embedding + head; total FLOPs/byte; parameters
    emb_head = sum(rb["per_stage"][0]["components"][k]["flops_per_byte"] for k in ("embedding", "lm_head"))
    assert s0["flops_per_byte"] == pytest.approx(emb_head, rel=0.02)
    assert rh["flops_per_byte"] == pytest.approx(rb["flops_per_byte"], rel=0.02)
    assert abs(rh["flops_per_byte"] / rb["flops_per_byte"] - 1) <= OPTS.total_tol + 1e-9
    assert rh["params"] == pytest.approx(rb["params"], rel=0.05)
    assert model_flops(ws, seq_len=byt)["flops_per_byte"] == rh["flops_per_byte"] and count_params(ws) == rh["params"]
    # the manifest's numbers are these
    n = RUNGS[rung]["numbers"]
    assert n["hnet"]["flops_per_byte"] == pytest.approx(rh["flops_per_byte"]) and n["bpe"]["params"] == rb["params"]


def test_smallest_rung_trains_one_step_on_cpu():
    """Forward + backward of the L03 models (real weights): the head_dim != D // heads attention and the ladder's
    H-Net components (rank decider, chunk vocabulary, n-gram tables, warm start) run end to end."""
    from hnet.model import build_model

    rung = sorted(RUNGS)[0]
    torch.manual_seed(0)
    for variant, vocab in (("bpe", OPTS.vocab), ("hnet_ws", 250)):
        cfg = model_config_from_yaml(model_path(variant, rung))
        m = build_model(cfg)
        if hasattr(m, "init_weights"):
            m.init_weights()
        ids = torch.randint(0, vocab, (1, 256))
        out = m(ids, labels=ids)
        loss = out.loss + out.aux_loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize("budget", OPTS.budgets)
def test_iso_flop_runs_spend_the_budget(budget):
    runs = MANIFEST["budgets"][budget]
    assert 3 <= len(runs) <= OPTS.n_rungs
    assert sum(x["optimum_prior"] for x in runs) == 1
    C = float(budget)
    for x in runs:
        n = RUNGS[x["rung"]]["numbers"]
        assert x["tokens"] == x["steps"] * x["batch_windows"] * OPTS.seq_tokens
        assert x["bytes"] == x["steps"] * x["batch_windows"] * OPTS.seq_bytes
        assert x["flops_bpe"] == pytest.approx(C, rel=0.02) and x["flops_hnet"] == pytest.approx(C, rel=0.03)
        assert x["flops_bpe"] == pytest.approx(x["tokens"] * n["bpe"]["flops_per_token"])
        sp = yaml.safe_load(open(os.path.join(REPO, ML.SEARCH_DIR, f"ladder_{budget}_{x['rung']}.yaml")))
        sets = {r["name"]: r["set"] for r in sp["runs"]}
        assert f"train.train_tokens={x['tokens']}" in sets["bpe"]
        assert f"train.train_bytes={x['bytes']}" in sets["hnet"] and sets["hnet"] == sets["hnet_ws"]


def test_iso_specs_require_the_tuned_lr():
    from spec import Spec, SpecError

    path = os.path.join(REPO, ML.SEARCH_DIR, f"ladder_{OPTS.budgets[-1]}_{MANIFEST['budgets'][OPTS.budgets[-1]][0]['rung']}.yaml")
    with pytest.raises(SpecError, match="best_mlr.bpe"):
        Spec.load(path).jobs()
    jobs = Spec.load(path, ["best_mlr.bpe=0.02"]).jobs()
    assert [j.arch for j in jobs] == list(ML.VARIANTS)
    assert all("train.muon.lr=0.02" in j.sets for j in jobs)


@pytest.fixture
def measured_bpt(monkeypatch):
    """The fairness gate reads bytes/token of the dataset from hnet/data/bpt_stats.json; until the ladder dataset is
    measured, stand in the generator's value (the gate then checks the configs are consistent with it)."""
    import hnet.data.loader as loader

    orig = loader.bytes_per_token

    def bpt(dataset="finewebedu10B", stream=False):
        return OPTS.bytes_per_token if dataset == OPTS.dataset else orig(dataset, stream)

    monkeypatch.setattr(loader, "bytes_per_token", bpt)


@pytest.mark.parametrize("path", SPECS, ids=os.path.basename)
def test_fairness_gate_passes_for_every_spec(path, tmp_path, measured_bpt):
    from fairness import main as fairness

    sp = yaml.safe_load(open(path))
    sp["out_root"] = str(tmp_path / "out")
    p = tmp_path / os.path.basename(path)
    p.write_text(yaml.safe_dump(sp, sort_keys=False))
    with torch.device("meta"):  # the gate builds every model to count parameters
        rc = fairness([str(p), "--param", "best_mlr.bpe=0.02"])
    res = json.loads((tmp_path / "out" / "fairness.json").read_text())
    assert rc == 0, res["problems"]
    archs = {k.split("|")[0] for k in res["rows"]}
    assert archs == set(ML.VARIANTS)
    if sp["fairness"] == "tuned":
        assert len(res["rows"]) == 3 * len(OPTS.lr_grid)
