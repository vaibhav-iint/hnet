"""Data mixes (hnet.data.mix): routing/splits, mixture proportions, resumable fetch, aligned byte/token shards,
val sets, BPE tokenizer exactness. All on small synthetic sources (no network)."""

import hashlib
import json

import numpy as np
import pytest

from hnet.config import BOS_ID, EOS_ID
from hnet.data import mix as M
from hnet.data import shards as S
from hnet.data.loader import WindowLoader, segment_starts

pytest.importorskip("tokenizers")
from hnet.data.tokenizer import BPETokenizer, LLAMA3_PATTERN, train_bpe  # noqa: E402

_WORDS = ["the", "data", "byte", "token", "model", "chunk", "Paris", "naïve", "café", "中文", "日本語", "😀",
          "def", "return", "x", "(", ")", ":", "\n", "    ", "{", "}", ";", "1234", "<|endoftext|>", "привет", "мир"]


def _text(rng, n):
    return "".join(_WORDS[i] + s for i, s in zip(rng.integers(0, len(_WORDS), n), rng.choice([" ", "", "\n"], n)))


SPEC = {
    "name": "synthmix", "seed": 3, "max_doc_bytes": 3000,
    "categories": {"en": 0.7, "code": 0.2, "ml": 0.1},
    "sources": {
        "web": {"repo": "fake/web", "revision": "r1", "files": ["web/*.parquet"], "text": "text", "rg_per_unit": 2,
                "p_val": 0.1, "leaves": {"en.web": {"weight": 1}}},
        "code": {"repo": "fake/code", "revision": "r1", "files": ["code/*.parquet"], "text": "code",
                 "select": "language", "split_key": "repo_name", "rg_per_unit": 1, "p_val": 0.2,
                 "leaves": {"code.a": {"value": "A", "weight": 2.0}, "code.b": {"value": "B", "weight": 1.0},
                            "heldout.h": {"value": "H", "heldout": True}}},
        "ml": {"repo": "fake/ml", "revision": "r1", "files": ["ml/*.parquet"], "text": "text", "rg_per_unit": 1,
               "p_val": 0.1, "leaves": {"ml.x": {"weight": 1}}},
    },
}


def _fake_files(seed=0):
    rng = np.random.default_rng(seed)
    files = {}
    for f in range(3):
        files[f"web/{f:03d}.parquet"] = [
            {"text": [_text(rng, int(rng.integers(5, 120))).encode() for _ in range(40)]} for _ in range(6)]
        rgs = []
        for _ in range(6):
            n = 60
            lang = list(rng.choice(["A", "B", "H", "Z"], n, p=[0.5, 0.3, 0.1, 0.1]))
            rgs.append({"code": [_text(rng, int(rng.integers(5, 80))).encode() for _ in range(n)], "language": lang,
                        "repo_name": [f"repo{int(r)}" for r in rng.integers(0, 40, n)]})
        rgs[0]["code"][0] = b"\xff\xfe invalid utf8"  # dropped
        rgs[0]["code"][1] = b"x" * 5000  # over max_doc_bytes: dropped
        rgs[0]["language"][:2] = ["A", "A"]
        files[f"code/{f:03d}.parquet"] = rgs
        files[f"ml/{f:03d}.parquet"] = [
            {"text": [("Привет мир " + _text(rng, int(rng.integers(3, 40)))).encode() for _ in range(30)]}
            for _ in range(4)]
    files["web/999.parquet"] = files.pop("web/002.parquet")
    files["README.md"] = []
    return files


class FakeReader:
    def __init__(self, files, prefix, fail_after=None):
        self.files, self.prefix, self.fail_after, self.n_reads = files, prefix, fail_after, 0

    def list_files(self):
        return sorted(self.files)

    def num_row_groups(self, f):
        return len(self.files[f])

    def read(self, f, a, b, columns):
        if self.fail_after is not None and self.n_reads >= self.fail_after:
            raise ConnectionError("simulated network failure")
        self.n_reads += 1
        rgs = self.files[f][a:b]
        return {c: sum((rg[c] for rg in rgs), []) for c in columns}

    def prefetch(self, f):
        pass

    def release(self, f):
        pass

    def close(self):
        pass


def _factory(files, fail_after=None):
    return lambda src: FakeReader(files, src.name, fail_after)


@pytest.fixture(scope="module")
def tokenizer_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    js = train_bpe([_text(rng, 100) for _ in range(300)], vocab_size=400)
    p = tmp_path_factory.mktemp("tok") / "tok.json"
    p.write_text(js)
    return p


CFG = dict(dataset="synthmix_b", train_bytes=60_000, shard_bytes=15_000, val_leaf_bytes=1500, val_group_bytes=3000,
           val_mix_bytes=4000)


@pytest.fixture(scope="module")
def built(tmp_path_factory, tokenizer_path):
    root = tmp_path_factory.mktemp("mixroot")
    spec = M.load_spec(SPEC)
    cfg = M.BuildConfig(**CFG)
    files = _fake_files()
    M.fetch(spec, cfg, root, _factory(files), workers=2, log=lambda s: None)
    man = M.assemble(spec, cfg, tokenizer_path, root, log=lambda s: None)
    return dict(root=root, spec=spec, cfg=cfg, man=man, files=files)


# --------------------------------------------------------------------------------------------- spec / routing
def test_spec_shares():
    spec = M.load_spec(SPEC)
    s = {lf.name: lf.share for lf in spec.leaves}
    assert s["en.web"] == pytest.approx(0.7) and s["ml.x"] == pytest.approx(0.1)
    assert s["code.a"] == pytest.approx(0.2 * 2 / 3) and s["code.b"] == pytest.approx(0.2 / 3)
    assert s["heldout.h"] == 0.0 and spec.leaf("heldout.h").category == "heldout"
    assert sum(s.values()) == pytest.approx(1.0)


def test_real_spec_loads():
    spec = M.load_spec("mix_v1")
    assert sum(lf.share for lf in spec.leaves) == pytest.approx(1.0)
    cats = {}
    for lf in spec.leaves:
        cats[lf.category] = cats.get(lf.category, 0) + lf.share
    assert cats == pytest.approx({"en": 0.7, "code": 0.2, "math": 0.05, "ml": 0.05, "heldout": 0.0})
    assert len([lf for lf in spec.leaves if lf.category == "code"]) == 14
    assert len([lf for lf in spec.leaves if lf.heldout]) >= 8
    held = {lf.value for lf in spec.leaves if lf.heldout}
    assert not held & {lf.value for lf in spec.leaves if lf.category == "code"}
    assert all(s.revision for s in spec.sources.values())  # pinned


def test_route_unit_split_rules():
    spec = M.load_spec(SPEC)
    src = spec.sources["code"]
    rng = np.random.default_rng(1)
    n = 400
    cols = {"code": [_text(rng, 20).encode() for _ in range(n)], "language": list(rng.choice(["A", "B", "H", "Z"], n)),
            "repo_name": [f"r{int(i)}" for i in rng.integers(0, 30, n)]}
    cols["code"][0], cols["language"][0] = b"\xc3\x28", "A"  # invalid UTF-8
    st = {}
    out = M.route_unit(spec, src, cols, st)
    assert st["dropped_utf8"] == 1
    assert set(out) == {"code.a", "code.b", "heldout.h"}  # Z is not in the spec
    assert (out["heldout.h"][1][:, M.META_VAL] == 1).all()  # held-out: val only
    # whole repositories go to one side
    side = {}
    for i in range(1, n):
        lf = {"A": "code.a", "B": "code.b"}.get(cols["language"][i])
        if lf is None:
            continue
        side.setdefault(cols["repo_name"][i], set()).add(M.split_u(cols["repo_name"][i].encode(), spec.seed) < 0.2)
    assert all(len(v) == 1 for v in side.values())
    assert any(next(iter(v)) for v in side.values()) and any(not next(iter(v)) for v in side.values())


# --------------------------------------------------------------------------------------------- proportions
def test_interleave_every_prefix_has_the_mixture():
    rng = np.random.default_rng(0)
    w = [0.7, 0.2, 0.06, 0.04]
    lens = [rng.integers(1, 2000, size=int(2000 * x) + 50) for x in w]
    li, pos = M.interleave(lens, w, None)
    sizes = np.array([lens[i][p] for i, p in zip(li, pos)])
    for i in range(len(w)):  # each list stays in order
        assert (np.diff(pos[li == i]) == 1).all()
    cum = np.cumsum(sizes)
    per = np.zeros((len(li), len(w)))
    per[np.arange(len(li)), li] = sizes
    per = np.cumsum(per, axis=0)
    usable = cum < min(np.sum(x) / ww for x, ww in zip(lens, w)) * 0.95  # before any list runs out
    for k in np.flatnonzero(usable)[::97]:
        for i in range(len(w)):
            assert abs(per[k, i] - w[i] * cum[k]) <= 2 * 2000, (k, i)  # within ~one document
    # the cut: first doc reaching the total, and prefix-stable
    li2, pos2 = M.interleave(lens, w, 300_000)
    assert sizes[: len(li2)].sum() >= 300_000 > sizes[: len(li2) - 1].sum()
    assert (li2 == li[: len(li2)]).all() and (pos2 == pos[: len(li2)]).all()


def test_built_mixture_proportions(built):
    man = built["man"]
    assert man["train"]["bytes"] >= CFG["train_bytes"]
    for n, v in man["leaves"].items():
        if v["heldout"]:
            assert v["train"]["bytes"] == 0
        else:
            assert abs(v["train"]["byte_share"] - v["train"]["target_share"]) < 3000 / CFG["train_bytes"], n


# --------------------------------------------------------------------------------------------- fetch
def _pool_digest(root, spec):
    h = hashlib.sha1()
    pd = M.pool_dir(spec, root)
    for p in sorted(pd.rglob("*")):
        if p.is_file() and "_state" not in p.parts:
            h.update(p.relative_to(pd).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_fetch_resume_is_exact(tmp_path):
    spec = M.load_spec(SPEC)
    cfg = M.BuildConfig(**CFG)
    files = _fake_files()
    M.fetch(spec, cfg, tmp_path / "a", _factory(files), workers=1, log=lambda s: None)
    with pytest.raises(ConnectionError):  # dies after 2 reads per source ...
        M.fetch(spec, cfg, tmp_path / "b", _factory(files, fail_after=2), workers=1, log=lambda s: None,
                prefetch=1)
    st = M.source_state(spec, "web", tmp_path / "b")
    assert 0 < st["units"] <= 2
    M.fetch(spec, cfg, tmp_path / "b", _factory(files), workers=1, log=lambda s: None)  # ... and resumes
    assert _pool_digest(tmp_path / "a", spec) == _pool_digest(tmp_path / "b", spec)
    # a bigger build extends the same pools (prefix property)
    big = M.BuildConfig(**{**CFG, "train_bytes": 120_000})
    M.fetch(spec, big, tmp_path / "b", _factory(files), workers=1, log=lambda s: None)
    pa, pb = M.load_pools(spec, tmp_path / "a"), M.load_pools(spec, tmp_path / "b")
    for n in pa:
        assert len(pb[n]) >= len(pa[n])
        assert (pb[n].meta[: len(pa[n])] == pa[n].meta).all()


# --------------------------------------------------------------------------------------------- shards
def test_train_shards_aligned(built):
    root, man = built["root"], built["man"]
    tok = BPETokenizer(M.mix_dir(CFG["dataset"], root) / "tokenizer.json")
    assert tok.eot_id == 0
    d = M.mix_dir(CFG["dataset"], root)
    n_docs = 0
    for stem in man["train"]["shards"]:
        toks = S.read_token_shard(d / f"{stem}.bin")
        data = S.read_byte_shard(d / "bytes" / f"{stem}.bytes.bin")
        idx = S.read_doc_index(d / "bytes" / f"{stem}.idx.npy")
        src = np.load(d / "bytes" / f"{stem}.src.npy")
        assert len(idx) == len(src) == int((toks == tok.eot_id).sum()) == int((data == BOS_ID).sum())
        assert int((data == EOS_ID).sum()) == len(idx) and data[0] == BOS_ID and data[-1] == EOS_ID
        assert toks[0] == tok.eot_id
        for row in idx:
            assert row[S.FLAGS] == S.HAS_BOS | S.HAS_EOS
            assert bytes(S.doc_text_bytes(data, row)) == tok.decode_bytes(S.doc_text_tokens(toks, row))
        n_docs += len(idx)
        heldout = [man["leaves"][n]["index"] for n in man["leaves"] if man["leaves"][n]["heldout"]]
        assert not np.isin(src, heldout).any()
    assert n_docs == man["train"]["docs"]


def test_val_disjoint_from_train(built):
    root, man = built["root"], built["man"]
    d = M.mix_dir(CFG["dataset"], root)
    tok = BPETokenizer(d / "tokenizer.json")

    def doc_hashes(stem):
        data = S.read_byte_shard(d / "bytes" / f"{stem}.bytes.bin")
        idx = S.read_doc_index(d / "bytes" / f"{stem}.idx.npy")
        return {hashlib.sha1(bytes(S.doc_text_bytes(data, r))).hexdigest() for r in idx}

    train = set().union(*[doc_hashes(s) for s in man["train"]["shards"]])
    assert {"mix", "en", "code", "ml", "heldout", "code.a", "heldout.h", "en.web"} <= set(man["val_sets"])
    for name, v in man["val_sets"].items():
        assert v["docs"] > 0
        assert not (doc_hashes(v["stem"]) & train), name
    # nested: the leaf, category and combined sets are prefixes of the same val pool of each leaf
    leaf_a = doc_hashes(man["val_sets"]["code.a"]["stem"])
    code = doc_hashes(man["val_sets"]["code"]["stem"])
    assert leaf_a <= code or (code & leaf_a) == code - doc_hashes(man["val_sets"]["code.b"]["stem"])
    assert set(man["val_sets"]["heldout"]["leaves"]) == {"heldout.h"}
    assert "heldout.h" not in man["val_sets"]["mix"]["leaves"]
    del tok


def test_assemble_deterministic_and_resumable(built, tokenizer_path):
    root, spec, cfg = built["root"], built["spec"], built["cfg"]
    d = M.mix_dir(cfg.dataset, root)

    def digest():
        h = hashlib.sha1()
        for p in sorted(d.rglob("*")):
            if p.is_file() and p.name != "manifest.json":
                h.update(p.name.encode())
                if not p.name.endswith(".json"):
                    h.update(p.read_bytes())
        return h.hexdigest()

    before = digest()
    stem = built["man"]["train"]["shards"][1]
    (d / f"{stem}.bin").unlink()
    (d / "bytes" / f"{stem}.json").unlink()
    logs = []
    man = M.assemble(spec, cfg, tokenizer_path, root, log=logs.append)
    assert any("1 shards to write" in s for s in logs)
    assert digest() == before
    assert man["train"]["bytes_per_token"] == built["man"]["train"]["bytes_per_token"]


def test_loader_and_valsets_on_mix(built):
    root, man = built["root"], built["man"]
    from hnet.data import standard_valset
    from hnet.data.download import byte_shard_paths, token_paths
    from hnet.data.loader import bytes_per_token

    ds = CFG["dataset"]
    tp, bp = token_paths(ds, "train", root), byte_shard_paths(ds, "train", root)
    assert len(tp) == len(bp) == man["train"]["n_shards"] > 1
    ld = WindowLoader(tp, "tokens", 64, n_windows=2, eot_id=0, prefetch=0, pin_memory=False)
    b = ld.next_batch()
    x = b["input_ids"].numpy().reshape(2, 64)
    assert (segment_starts(x, "tokens", 0).sum() + 0) == len(b["cu_seqlens"]) - 1
    assert (b["labels"].numpy() != 0).all()  # the document-start token (id 0) is never a target
    assert bytes_per_token(ds, root=root) == pytest.approx(man["train"]["bytes_per_token"])
    for sub in (None, "code.b", "heldout"):
        vs = standard_valset(ds, n_bytes_target=None, root=root, subset=sub, byte_seq_len=256, token_seq_len=64,
                             windows_per_batch=4)
        tb, bb = vs.token_batches(), vs.byte_batches()
        assert sum(x["n_bytes"] for x in tb) == sum(x["n_bytes"] for x in bb) == vs.n_bytes
        want = man["val_sets"][sub or "mix"]
        assert vs.n_docs == want["docs"] and vs.n_bytes == want["bytes"]
    vss = M.mix_valsets(ds, root, byte_seq_len=256, token_seq_len=64)
    assert set(vss) == set(man["val_sets"])


# --------------------------------------------------------------------------------------------- tokenizer
def test_bpe_roundtrip_and_special_text(tokenizer_path):
    tok = BPETokenizer(tokenizer_path)
    assert tok.pretokenize and tok.vocab_size == 400
    rng = np.random.default_rng(5)
    for _ in range(50):
        t = _text(rng, 60) + " <|endoftext|> \x00 \r\n\t ÿ"
        ids = tok.encode(t)
        assert (ids != tok.eot_id).all()
        assert tok.decode_bytes(ids) == t.encode()
    lens = tok.byte_lengths(np.arange(tok.vocab_size))
    assert lens[0] == 0 and (lens[1:] >= 1).all()


def test_bpe_np_variant_spans_words():
    rng = np.random.default_rng(0)
    docs = [_text(rng, 80) for _ in range(200)]
    tok = BPETokenizer(train_bpe(docs, vocab_size=500, pretokenize=False))
    assert not tok.pretokenize
    assert all(tok.decode_bytes(tok.encode(d)) == d.encode() for d in docs)
    pieces = [tok.decode_bytes([i]) for i in range(tok.vocab_size)]
    assert any(b" " in p.strip(b" ") for p in pieces)  # merges across word boundaries exist


def test_llama3_pattern_splits_like_tiktoken():
    import regex

    pat = regex.compile(LLAMA3_PATTERN)
    assert pat.findall("Hello world's 12345\n\n  x") == ["Hello", " world", "'s", " ", "123", "45", "\n\n", " ", " x"]


def test_manifest_is_json_serializable(built):
    p = M.mix_dir(CFG["dataset"], built["root"]) / "manifest.json"
    man = json.loads(p.read_text())
    assert man["format"] == M.MIX_FORMAT and man["tokenizer"]["eot_id"] == 0
    assert M.is_mix(CFG["dataset"], built["root"]) and not M.is_mix("finewebedu10B")
