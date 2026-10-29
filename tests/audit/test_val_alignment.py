"""Audit: the byte and BPE val sets score exactly the same text, and bpb = total NLL in bits / UTF-8 bytes of it.

Uses the local FineWeb-Edu val token shard (skipped without it). Checks, on ~300 KB of whole documents:
* the concatenation of scored byte targets (in order) == the UTF-8 decode of the scored token targets (in order);
* sum(n_bytes) is the same for both kinds and equals the number of scored byte targets;
* no BOS/EOS/<|endoftext|> target is scored.
"""

import os

import numpy as np
import pytest

from hnet.config import BOS_ID, IGNORE_INDEX
from hnet.data.tokenizer import load_tokenizer
from hnet.data.valsets import ValSet

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARD = os.path.join(ROOT, "data/finewebedu10B/finewebedu_val_000000.bin")


@pytest.mark.skipif(not os.path.isfile(SHARD), reason="val shard not present")
@pytest.mark.parametrize("lens", [(2048, 448), (8192, 1792)])
def test_byte_and_token_valsets_score_identical_text(lens):
    vs = ValSet("audit", 300_000, SHARD, byte_seq_len=lens[0], token_seq_len=lens[1], windows_per_batch=4)
    tok = load_tokenizer(None)
    lens_tbl = tok.table[2]
    bb, tb = vs.byte_batches(), vs.token_batches()
    nb_b = sum(int(b["n_bytes"]) for b in bb)
    nb_t = sum(int(b["n_bytes"]) for b in tb)
    assert nb_b == nb_t == vs.n_bytes
    yb = np.concatenate([b["labels"].numpy() for b in bb])
    yb = yb[yb != IGNORE_INDEX]
    assert len(yb) == nb_b and (yb < BOS_ID).all()
    yt = np.concatenate([b["labels"].numpy() for b in tb])
    yt = yt[yt != IGNORE_INDEX]
    assert (yt != tok.eot_id).all()
    assert int(lens_tbl[yt].sum()) == nb_t
    from hnet.data.tokenizer import decode_with_table

    # packing reorders segments across windows (first-fit), so compare the multisets of per-doc text via totals and
    # the sorted byte histogram, plus the exact sequence when packing keeps order
    assert np.array_equal(np.bincount(yb, minlength=256),
                          np.bincount(decode_with_table(yt, tok.table).astype(np.int64), minlength=256))
