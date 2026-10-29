"""scripts/eval/per_source.py and scripts/eval/downstream_bpb.py on tiny CPU models (no network)."""

import gzip
import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch

torch.set_num_threads(1)
hmodel = pytest.importorskip("hnet.model")
tiktoken = pytest.importorskip("tiktoken")

from hnet.config import DenseConfig, model_config_from_yaml  # noqa: E402
from hnet.eval.scorers import ByteModelScorer, TokenModelScorer  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DENSE = dict(variant="llama", vocab_size=50304, d_model=64, n_layers=2, d_intermediate=128, num_heads=4)
TEXTS = {
    "english": ["The quick brown fox jumps over the lazy dog. " * 6, "Bits per byte is a tokenizer-free metric.\n" * 4],
    "code": ["def f(x):\n    return x * 2  # double\n" * 5, "for i in range(10):\n    print(i)\n" * 4],
    "haskell": ["main :: IO ()\nmain = putStrLn \"hi\"\n" * 5],
    "multi": ["Größe, café, naïve — 日本語のテキスト。" * 6],
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", "eval", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PS = _load("per_source")
DS = _load("downstream_bpb")


@pytest.fixture(scope="module")
def hnet_model():
    m = hmodel.build_model(model_config_from_yaml(os.path.join(ROOT, "configs/tiny_cpu.yaml")))
    m.init_weights()
    return m.eval()


@pytest.fixture(scope="module")
def dense_model():
    m = hmodel.build_model(DenseConfig(**DENSE))
    m.init_weights()
    return m.eval()


def _write_jsonl(path, texts, field="text", gz=False):
    op = gzip.open if gz else open
    with op(path, "wt", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({field: t}) + "\n")
    return str(path)


@pytest.fixture()
def sources(tmp_path):
    return [
        PS.Source("web", [_write_jsonl(tmp_path / "web.jsonl", TEXTS["english"])], group="english", weight=0.7),
        PS.Source("python", [_write_jsonl(tmp_path / "py.jsonl.gz", TEXTS["code"], "content", gz=True)],
                  group="code", lang="python", weight=0.2),
        PS.Source("haskell", [_write_jsonl(tmp_path / "hs.jsonl", TEXTS["haskell"])], group="code", lang="haskell",
                  held_out=True),
        PS.Source("multi", [str(tmp_path / "multi.txt")], group="multilingual", weight=0.1),
    ]


@pytest.fixture(autouse=True)
def _txt(tmp_path):
    (tmp_path / "multi.txt").write_text(TEXTS["multi"][0], encoding="utf-8")


def test_make_windows_respect_limits_and_utf8():
    tok = PS.Tokenizer("gpt2")
    doc = ("héllo wörld 日本語のテキスト, plain words and more words. " * 40).encode("utf-8")
    ids, lens = tok.encode_with_lens(doc.decode("utf-8"))
    assert lens.sum() == len(doc)
    wins = PS.make_windows(doc, ids, lens, byte_seq_len=97, token_seq_len=20)
    assert wins[0][0] == 0 and wins[-1][1] == len(doc) and wins[-1][3] == len(ids)
    for (a, b, ta, tb), nxt in zip(wins, wins[1:] + [None]):
        assert b - a <= 96 and tb - ta <= 19 and b > a
        doc[a:b].decode("utf-8")  # every window is valid UTF-8 (cut at a character start)
        if nxt:
            assert nxt[0] == b and nxt[2] == tb


def test_iter_docs_formats(tmp_path):
    from hnet.data import shards as S

    enc = tiktoken.get_encoding("gpt2")
    docs = ["first doc.", "second — doc ü", "third"]
    toks = []
    for d in docs:
        toks += [S.EOT_ID] + enc.encode_ordinary(d)
    S.write_token_shard(tmp_path / "val_000000.bin", np.array(toks, dtype=np.uint16))
    S.convert_shard(tmp_path / "val_000000.bin", tmp_path / "bytes", last_complete=True)
    got = list(PS.iter_docs(str(tmp_path / "bytes" / "val_000000.bytes.bin")))
    assert got == [d.encode() for d in docs]
    # token shard: complete docs only (the last one has no following <|endoftext|>)
    assert list(PS.iter_docs(str(tmp_path / "val_000000.bin"))) == [d.encode() for d in docs[:2]]
    p = _write_jsonl(tmp_path / "a.jsonl.gz", docs, "content", gz=True)
    assert list(PS.iter_docs(p)) == [d.encode() for d in docs]
    src = PS.Source("x", [p], max_bytes=12)
    assert len(PS.load_source_docs(src, None)["docs"]) == 2  # whole docs until >= max_bytes


def test_manifest_and_source_arg(tmp_path):
    p = _write_jsonl(tmp_path / "py.jsonl", TEXTS["code"])
    (tmp_path / "m.yaml").write_text("sources:\n  - {name: python, group: code, lang: python, paths: [py.jsonl]}\n"
                                     "  - {name: hs, group: code, held_out: true, path: py.jsonl, max_docs: 1}\n")
    srcs = PS.load_manifest(str(tmp_path / "m.yaml"))
    assert [s.name for s in srcs] == ["python", "hs"] and srcs[0].paths == [p] and srcs[1].held_out
    s = PS.parse_source_arg(f"py={p}@code")
    assert s.group == "code" and s.paths == [p]


def _segments_reference(model, kind, sources, bsl, tsl):
    loaded = [PS.load_source_docs(s, None)["docs"] for s in sources]
    tok = PS.Tokenizer("gpt2")
    segs, _ = PS.build_segments(loaded, kind, tok, tok if kind == "token" else None, bsl, tsl)
    sc = (ByteModelScorer(model, device="cpu", batch_tokens=1) if kind == "byte"
          else TokenModelScorer(model, device="cpu", batch_tokens=1))
    ref = sc.score_sequences([s.ids for s in segs], [len(s.ids) - 1 for s in segs])
    return segs, np.array([-lp / math.log(2) for lp, _ in ref])


@pytest.mark.parametrize("kind", ["byte", "token"])
def test_evaluate_sources(kind, sources, hnet_model, dense_model):
    model = hnet_model if kind == "byte" else dense_model
    res = PS.evaluate_sources(model, kind, sources, device="cpu", byte_seq_len=128, token_seq_len=40,
                              max_bytes=None, batch_tokens=512, n_boot=50)
    segs, ref_bits = _segments_reference(model, kind, sources, 128, 40)
    for si, s in enumerate(sources):
        r = res["sources"][s.name]
        m = [i for i, g in enumerate(segs) if g.src == si]
        assert r["n_bytes"] == sum(len(t.encode()) for t in TEXTS[{"web": "english", "python": "code",
                                                                      "haskell": "haskell", "multi": "multi"}[s.name]])
        # packed scoring == one-by-one scoring through the shared scorer machinery
        assert r["bits"] == pytest.approx(ref_bits[m].sum(), rel=2e-4)
        assert r["bpb"] == pytest.approx(r["bits"] / r["n_bytes"])
        assert r["bpb_ci95"] is None or r["bpb_ci95"][0] <= r["bpb_ci95"][1]
        if kind == "byte":
            assert len(r["positions_by_stage"]) == 3  # tiny_cpu is a 2-stage H-Net
            assert r["bytes_per_main_position"] >= 1 and r["bytes_per_position_by_stage"][0] == 1
        else:
            assert r["bytes_per_token"] == pytest.approx(r["split_bytes_per_token"])
    g = res["groups"]
    assert set(g) == {"english", "code", "code_held_out", "multilingual", "all", "mix"}
    assert g["code"]["sources"] == ["python"] and g["code_held_out"]["sources"] == ["haskell"]
    assert "haskell" not in g["all"]["sources"]
    assert g["all"]["bpb"] == pytest.approx(sum(res["sources"][n]["bits"] for n in g["all"]["sources"]) /
                                            sum(res["sources"][n]["n_bytes"] for n in g["all"]["sources"]))
    assert g["mix"]["weights"]["web"] == pytest.approx(0.7)


def test_reencode_path_for_other_model_tokenizer(sources):
    """A BPE model with a tokenizer other than the split tokenizer re-encodes each window's text."""
    split = PS.Tokenizer("gpt2")
    other = PS.Tokenizer("gpt2")
    other.spec = "gpt2-copy"
    docs = [PS.load_source_docs(s, None)["docs"] for s in sources]
    a, _ = PS.build_segments(docs, "token", split, split, 128, 40)
    b, _ = PS.build_segments(docs, "token", split, other, 128, 40)
    assert [s.n_bytes for s in a] == [s.n_bytes for s in b]
    same = sum(x.ids == y.ids for x, y in zip(a, b))
    assert same >= 0.8 * len(a)  # identical except where a window cut changes the pretokenization


def test_per_source_cli_and_load_run(tmp_path, sources):
    from hnet.analysis.scaling import load_run
    from hnet.train.train import build_model_from_section

    section = {"type": "dense", **DENSE}
    model, _ = build_model_from_section(section)
    run = tmp_path / "run"
    ck = run / "ckpt" / "step_0000010"
    ck.mkdir(parents=True)
    torch.save(model.state_dict(), ck / "model.pt")
    cfg = {"model": section, "eval": {"val_byte_seq_len": 256, "val_token_seq_len": 64}, "train": {"seed": 3}}
    (ck / "meta.json").write_text(json.dumps({"step": 10, "bytes_seen": 4600, "config": cfg}))
    import yaml

    (run / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run / "metrics.jsonl").write_text(json.dumps({"step": 10, "bytes_seen": 4600, "flops_per_byte": 1e6}) + "\n")
    (run / "eval.jsonl").write_text(json.dumps({"step": 10, "val/bpb": 9.0}) + "\n")
    (run / "done.json").write_text("{}")
    PS.main(["--model", str(run), "--device", "cpu", "--n-boot", "0",
             "--source", f"python={sources[1].paths[0]}@code", "--source", f"web={sources[0].paths[0]}@english"])
    out = json.load(open(run / "per_source.json"))
    assert out["settings"]["byte_seq_len"] == 256 and out["settings"]["token_seq_len"] == 64
    assert out["run"]["kind"] == "token" and set(out["sources"]) == {"python", "web"}
    r = load_run(str(run))
    assert r.arch == "bpe" and r.seed == 3 and r.flops == pytest.approx(4.6e9)
    assert r.metrics["src:python"] == pytest.approx(out["sources"]["python"]["bpb"])
    assert r.metrics["group:all"] == pytest.approx(out["groups"]["all"]["bpb"]) and r.metrics["val"] == 9.0
    assert r.info["compression"]["python"]["bytes_per_token"] > 1


# ============================================================================================ downstream bpb
def test_canonical_boundary(dense_model):
    sc = DS.CanonicalTokenScorer(dense_model, device="cpu")
    lm = TokenModelScorer(dense_model, device="cpu")
    enc = sc.tok
    # aligned boundary (leading space moved into the continuation): canonical == lm-eval
    assert sc.encode_pair("Question: what?\nAnswer:", " it is 5") == lm.encode_pair("Question: what?\nAnswer:",
                                                                                    " it is 5")
    assert sc.boundary_stats == {"pairs": 1, "straddle": 0, "noncanonical": 0}
    # boundary inside a token: "bro|wn" -> canonical scores the joint " brown" token as part of the answer
    ctx, cont, nb = sc.encode_pair("The quick bro", "wn fox")
    whole = enc.encode_ordinary("The quick brown fox")
    assert ctx[1:] + cont == whole and enc.decode(cont) == " brown fox" and nb == len("wn fox")
    assert sc.boundary_stats["straddle"] == 1 and sc.boundary_stats["noncanonical"] == 1
    # lm-eval: input = enc(context) + tail of the joint encoding (not a canonical tokenization)
    c2, x2, _ = lm.encode_pair("The quick bro", "wn fox")
    assert c2[1:] == enc.encode_ordinary("The quick bro") and enc.decode(x2) != "wn fox"
    sl = DS.CanonicalTokenScorer(dense_model, device="cpu", boundary="lmeval")
    assert sl.encode_pair("The quick bro", "wn fox") == (c2, x2, 6)
    (lp, _, n), = sc.loglikelihood([("The quick bro", "wn fox")])
    assert lp < 0 and n == 6


def _hs_rows(n=8):
    endings = ["ends badly here.", "GOOD ending.", "another bad ending that is long.", "meh."]
    return [{"ind": i, "activity_label": "Sport", "ctx_a": f"A person does thing {i}.", "ctx_b": "they",
             "endings": endings, "label": str(1 if i % 3 else 0)} for i in range(n)]


def _tasks(all_choices):
    tl = DS.paper7_tasks(["hellaswag", "lambada_openai"], all_choices=all_choices)
    tl[0].rows = _hs_rows()
    tl[1].rows = [{"text": "the word is GOOD"}, {"text": "the cat sat on the mat"}]
    return tl


@pytest.mark.parametrize("kind", ["byte", "token"])
def test_downstream_gold_only_matches_full_scoring(kind, hnet_model, dense_model):
    sc = (ByteModelScorer(hnet_model, device="cpu") if kind == "byte"
          else DS.CanonicalTokenScorer(dense_model, device="cpu"))
    gold = DS.evaluate_downstream(sc, task_objs=_tasks(False), n_boot=50)
    full = DS.evaluate_downstream(sc, task_objs=_tasks(True), n_boot=0, all_choices=True)
    for t in ("hellaswag", "lambada_openai"):
        assert gold["tasks"][t]["soft_bpb"] == pytest.approx(full["tasks"][t]["soft_bpb"], rel=1e-5)
        assert gold["tasks"][t]["n_bytes"] == full["tasks"][t]["n_bytes"]
    assert "acc_norm_char" in full["tasks"]["hellaswag"] and "acc_norm_char" not in gold["tasks"]["hellaswag"]
    avg = gold["average"]
    assert avg["soft_bpb"] == pytest.approx(np.mean([v["soft_bpb"] for v in gold["tasks"].values()]))
    assert avg["ci95"][0] <= avg["soft_bpb"] <= avg["ci95"][1] and avg["n_tasks"] == 2
    # identical bytes for byte and token models (same prompt and boundary convention)
    hs = gold["tasks"]["hellaswag"]["n_bytes"]
    assert hs == sum(len((" " + r["endings"][int(r["label"])]).encode()) for r in _hs_rows())
