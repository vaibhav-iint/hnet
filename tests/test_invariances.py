"""Standing invariance suite (CLAUDE.md rule 10): causality and packing equivalence of the whole H-Net forward pass,
for the routers / deciders / extras we actually train with. A cached-decoding test is missing because the model has
no incremental decoding path yet; add it with that path.
"""

import pytest
import torch

from hnet.config import model_config_from_yaml
from hnet.model import build_model

VARIANTS = {
    "paper": {},
    "end_mlp_quantile": {"router": "end_mlp", "decision": "quantile"},
    "end_mlp_quantile_vocab": {"router": "end_mlp", "decision": "quantile",
                               "chunk_vocab": [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5,
                                                "n_neg": 256}]},
    # model-level (non-dc) options go under "_model"
    "end_mlp_quantile_vocab_trie": {"router": "end_mlp", "decision": "quantile",
                                    "chunk_vocab": [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5,
                                                     "n_neg": 256}],
                                    "_model": {"trie_head": {"rows": 4099, "dim": 16, "n_hash": 2, "max_len": 8}}},
    "end_mlp_quantile_vocab_suffix": {"router": "end_mlp", "decision": "quantile",
                                      "chunk_vocab": [{"buckets": 4096, "n_hash": 2, "max_len": 16, "pred_weight": 0.5,
                                                       "n_neg": 256, "suffix_lens": [2, 3, 4, 6, 8, 12, 16]}]},
}


def _model(name, ngram=False):
    cfg = model_config_from_yaml("configs/models/tiny_hnet1_2m.yaml")
    for k, v in VARIANTS[name].items():
        if k == "_model":
            for mk, mv in v.items():
                setattr(cfg, mk, mv)
        else:
            setattr(cfg.dc, k, v)
    if ngram:
        cfg.ngram_embed = {"ns": [3, 4, 5], "buckets": 1024}
    torch.manual_seed(0)
    m = build_model(cfg)
    m.init_weights()
    # a few training steps so running statistics (quantile decider) and tables are not at their init values
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    ids = torch.randint(0, 256, (2, 256))
    for _ in range(3):
        o = m(ids, labels=ids)
        opt.zero_grad()
        (o.loss + o.aux_loss).backward()
        opt.step()
    return m.eval()


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("ngram", [False, True])
def test_future_bytes_do_not_change_earlier_logits(name, ngram):
    m = _model(name, ngram)
    ids = torch.randint(0, 256, (1, 300))
    alt = ids.clone()
    alt[0, 200:] = (alt[0, 200:] + 7) % 256  # change every byte from position 200 on
    with torch.no_grad():
        a = m(ids).logits[0, :200]
        b = m(alt).logits[0, :200]
    assert torch.allclose(a, b, atol=1e-5), f"{name}: logits before the change differ by {(a - b).abs().max():.2e}"


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("ngram", [False, True])
def test_packed_segments_equal_separate_sequences(name, ngram):
    m = _model(name, ngram)
    a = torch.randint(0, 256, (1, 180))
    b = torch.randint(0, 256, (1, 220))
    with torch.no_grad():
        la = m(a).logits[0]
        lb = m(b).logits[0]
        packed = torch.cat([a[0], b[0]])
        cu = torch.tensor([0, 180, 400], dtype=torch.int32)
        lp = m(packed, cu_seqlens=cu, max_seqlen=220).logits
    assert torch.allclose(lp[:180], la, atol=1e-5), f"{name}: segment 0 differs by {(lp[:180] - la).abs().max():.2e}"
    assert torch.allclose(lp[180:], lb, atol=1e-5), f"{name}: segment 1 differs by {(lp[180:] - lb).abs().max():.2e}"


def test_quantile_decider_is_batch_coupled_in_training_mode_only():
    """Documented, not a bug: in training mode the rank decision ranks scores over the whole batch (like BatchNorm),
    so a change to one sequence can move another's threshold; in eval mode the running threshold makes it causal."""
    m = _model("end_mlp_quantile")
    ids = torch.randint(0, 256, (2, 256))
    alt = ids.clone()
    alt[1] = (alt[1] + 3) % 256
    with torch.no_grad():
        e0 = m(ids).bpred[0].boundary_mask[:256]
        e1 = m(alt).bpred[0].boundary_mask[:256]
    assert torch.equal(e0, e1)
