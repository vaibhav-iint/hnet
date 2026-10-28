"""Tests for hnet.flops: calibration against published numbers + matched configs + exact param counts."""

import glob
import os

import pytest
import yaml

from hnet.config import DenseConfig, HNetConfig, load_yaml, model_config_from_yaml
from hnet.flops import (
    BLOG_FIG8,
    PAPER_APPB,
    SPEC,
    attn_context,
    count_params,
    model_flops,
)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OFF = os.path.join(ROOT, "configs", "official")
MODELS = sorted(glob.glob(os.path.join(ROOT, "configs", "models", "*.yaml")))
G = 1e9


def official(name):
    return HNetConfig.from_official_json(os.path.join(OFF, name + ".json"))


def gpt3(d, ffn):  # paper's BPE Transformer (Table 1): 24 layers, 16 heads, GPT-2 vocab
    return DenseConfig(vocab_size=50257, d_model=d, n_layers=24, d_intermediate=ffn, num_heads=16)


# --------------------------------------------------------------------------- helpers
def test_attn_context():
    cv = SPEC
    assert attn_context(100, -1, cv) == pytest.approx(50.5)
    # window w: keys [t-w, t] -> at most w+1 keys
    L, w = 5000, 1023
    exact = sum(min(t + 1, w + 1) for t in range(L)) / L
    assert attn_context(L, w, cv) == pytest.approx(exact)
    assert attn_context(L, w, PAPER_APPB) == w + 1
    assert attn_context(L, -1, PAPER_APPB) == L


def test_training_is_three_times_forward():
    cfg = official("hnet_2stage_L")
    f = model_flops(cfg, training=False)["flops_per_byte"]
    assert model_flops(cfg, training=True)["flops_per_byte"] == pytest.approx(3 * f)


# --------------------------------------------------------------------------- paper Table 1 params
@pytest.mark.parametrize(
    "name,ref",
    [("hnet_1stage_L", 680e6), ("hnet_1stage_XL", 1.3e9), ("hnet_2stage_L", 870e6), ("hnet_2stage_XL", 1.6e9)],
)
def test_paper_table1_params(name, ref):
    assert count_params(official(name)) == pytest.approx(ref, rel=0.03)


# --------------------------------------------------------------------------- paper Table 2 (eval GFLOPs/byte)
# Reproduced by App. B taken literally (PAPER_APPB: full non-causal attention, embedding counted as a matmul,
# LM head counted), forward only, 8192 bytes / 1792 tokens at 4.6 bytes/token, BPIC from Table 1.
@pytest.mark.parametrize(
    "cfg,bpp,ref",
    [
        (gpt3(1536, 4096), None, 0.42),
        (gpt3(2048, 5504), None, 0.69),
        ("hnet_1stage_L", [1, 4.8], 0.43),
        ("hnet_1stage_XL", [1, 4.7], 0.72),
        # 2-stage: intermediate stage at N0 = 2.93 bytes/position (stage-1 ratio = BPIC / 2.93 ~ 2.4)
        ("hnet_2stage_L", [1, 2.93, 7.0], 0.43),
        ("hnet_2stage_XL", [1, 2.93, 6.9], 0.69),
    ],
)
def test_paper_table2(cfg, bpp, ref):
    if isinstance(cfg, str):
        cfg = official(cfg)
    r = model_flops(cfg, bytes_per_position=bpp, seq_len=None, bytes_per_token=4.6, training=False,
                    convention=PAPER_APPB)
    assert round(r["flops_per_byte"] / G, 2) == ref


def test_paper_table2_sqrt_split():
    """sqrt split of BPIC reproduces 2-stage L but overshoots XL (0.709 vs 0.69)."""
    L = model_flops(official("hnet_2stage_L"), [1, 7.0 ** 0.5, 7.0], training=False, convention=PAPER_APPB)
    XL = model_flops(official("hnet_2stage_XL"), [1, 6.9 ** 0.5, 6.9], training=False, convention=PAPER_APPB)
    assert round(L["flops_per_byte"] / G, 2) == 0.43
    assert XL["flops_per_byte"] / G == pytest.approx(0.709, abs=0.002)


def test_spec_convention_is_cheaper_for_bpe():
    """Under SPEC (causal, no embedding) the paper's matched models are no longer matched: note for MATCHING.md."""
    t = model_flops(gpt3(1536, 4096), training=False)["flops_per_byte"] / G
    h = model_flops(official("hnet_2stage_L"), [1, 2.93, 7.0], training=False)["flops_per_byte"] / G
    assert t == pytest.approx(0.358, abs=0.002)
    assert h > 1.1 * t


# --------------------------------------------------------------------------- blog Fig. 8
SMOLLM3 = DenseConfig(vocab_size=128256, d_model=2048, n_layers=36, d_intermediate=11008, num_heads=16,
                      num_kv_heads=4, tie_embeddings=True)


def test_blog_smollm3():
    r = model_flops(SMOLLM3, seq_len=4096, bytes_per_token=4.44, convention=BLOG_FIG8)
    c = r["per_stage"][0]["components"]
    assert r["params"] / G == pytest.approx(3.08, abs=0.01)
    assert (c["embedding"]["params"] + c["lm_head"]["params"]) / G == pytest.approx(0.26, abs=0.005)
    assert c["body"]["params"] / G == pytest.approx(2.81, abs=0.005)
    assert c["lm_head"]["flops_per_byte"] / G == pytest.approx(0.35, abs=0.006)
    assert c["body"]["flops_per_byte"] / G == pytest.approx(4.00, abs=0.01)
    assert r["flops_per_byte"] / G == pytest.approx(4.36, abs=0.006)


def test_blog_smollm3_standard_causal_overshoots():
    """Standard causal counting cannot give body 4.00 together with LM head 0.35."""
    r = model_flops(SMOLLM3, seq_len=4096, bytes_per_token=4.5, convention=SPEC)
    c = r["per_stage"][0]["components"]
    assert c["lm_head"]["flops_per_byte"] / G == pytest.approx(0.35, abs=0.001)
    assert c["body"]["flops_per_byte"] / G > 4.1


def load_3p7b():
    d = dict(load_yaml(os.path.join(ROOT, "configs", "hnet_3p7b_smollm3.yaml"))["model"])
    d.pop("type", None)
    d["dc"] = {k: v for k, v in d["dc"].items() if k != "smoothing"}  # not a DCConfig field
    return HNetConfig.from_dict(d)


def test_blog_hnet_3p7b():
    # seq 18432 bytes (= 4096 Llama-3 tokens x 4.5), main at the realized 7.6 bytes/position, stage 1 at 3.82 (3.8-3.85 all fit)
    r = model_flops(load_3p7b(), bytes_per_position=[1, 3.82, 7.6], seq_len=18432, convention=BLOG_FIG8)
    p = [s["params"] / G for s in r["per_stage"]]
    f = [s["flops_per_byte"] / G for s in r["per_stage"]]
    assert r["params"] / G == pytest.approx(3.67, abs=0.005)
    assert p == pytest.approx([0.22, 0.32, 3.13], abs=0.006)  # main 3.125B (blog rounds to 3.13)
    assert f == pytest.approx([1.37, 0.52, 2.55], abs=0.006)
    assert r["flops_per_byte"] / G == pytest.approx(4.44, abs=0.006)
    # at exactly 4.0 bytes per stage-1 position the yaml gives 0.50, not 0.52
    r4 = model_flops(load_3p7b(), bytes_per_position=[1, 4.0, 7.6], seq_len=18432, convention=BLOG_FIG8)
    assert r4["per_stage"][1]["flops_per_byte"] / G == pytest.approx(0.499, abs=0.002)


# --------------------------------------------------------------------------- our matched configs
def _ref_flops(path):
    m = load_yaml(path).get("matching", {})
    cfg = model_config_from_yaml(path)
    if isinstance(cfg, DenseConfig):
        return model_flops(cfg, seq_len=m["seq_len_tokens"], bytes_per_token=m["bytes_per_token"])
    return model_flops(cfg, seq_len=m["seq_len_bytes"])


@pytest.mark.parametrize("path", MODELS, ids=[os.path.basename(p) for p in MODELS])
def test_models_matched(path):
    m = load_yaml(path)["matching"]
    r = _ref_flops(path)
    assert r["params"] == count_params(model_config_from_yaml(path))
    ref = m.get("reference")
    if not ref:
        return
    rr = _ref_flops(os.path.join(os.path.dirname(path), ref + ".yaml"))
    err = r["flops_per_byte"] / rr["flops_per_byte"] - 1
    if path.endswith("_ratios42.yaml"):  # deliberately not re-matched (MATCHING.md)
        assert abs(err) < 0.06
    else:
        assert abs(err) <= 0.02, f"{os.path.basename(path)}: {100 * err:+.2f}% vs {ref}"


def test_ablations_differ_only_in_dc_or_mixers():
    for path in MODELS:
        m = load_yaml(path)
        base = m["matching"].get("ablation_of")
        if not base:
            continue
        a = m["model"]
        b = load_yaml(os.path.join(os.path.dirname(path), base + ".yaml"))["model"]
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert diff <= {"dc", "arch_layout", "d_intermediate", "lr_multiplier"}, (path, diff)


# --------------------------------------------------------------------------- vs instantiated models
def _build_model():
    try:
        from hnet.model import build_model
    except Exception:  # model package not written yet (parallel workstream)
        return None
    return build_model


OFFICIAL = sorted(glob.glob(os.path.join(OFF, "hnet_*L.json")))


@pytest.mark.parametrize("path", MODELS, ids=[os.path.basename(p) for p in MODELS])
def test_params_match_model(path):
    build_model = _build_model()
    if build_model is None:
        pytest.skip("hnet.model.build_model not importable yet")
    import torch

    torch.set_num_threads(1)
    cfg = model_config_from_yaml(path)
    with torch.device("meta"):
        model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == count_params(cfg)


@pytest.mark.parametrize("path", OFFICIAL, ids=[os.path.basename(p) for p in OFFICIAL])
def test_params_match_model_official(path):
    build_model = _build_model()
    if build_model is None:
        pytest.skip("hnet.model.build_model not importable yet")
    import torch

    cfg = HNetConfig.from_official_json(path)
    with torch.device("meta"):
        model = build_model(cfg)
    assert sum(p.numel() for p in model.parameters()) == count_params(cfg)
