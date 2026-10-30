# g3-copy-01 / g3-copy-02: offline probe of chunk-level copying (before any GPU)

`PYTHONPATH=. python scripts/evolve/copy_probe.py mix_v1_val mix_v1_val-code mix_v1_val-en` (first 2 MB of each
validation set, 8512-byte windows, no cross-window matches). The router's segmentation is replaced by a whitespace
**proxy** (chunks end at whitespace, at most 12 bytes; what a word-basin router does, bnd_on_ws ~0.9): analysis only,
no model sees it. Byte index = the match model's longest order>=8 candidate (g1-wild-01).

## Chunk pointer (g3-copy-01): what the main network would be told

| set | proxy bytes/chunk | chunks with an o>=8 pointer | next chunk equal (o8 / o16) | whole next chunk proposed correctly (o8 / o16) |
|---|---|---|---|---|
| mix | 5.42 | 27.8% | 47.7% / 70.6% | 13.3% / 7.6% of chunks |
| code | 3.48 | 58.0% | 73.0% / 81.4% | 42.3% / 27.5% |
| en | 6.00 | 18.6% | 20.8% / 37.6% | 3.9% / 1.3% |

Byte-level value of the same pointer (an offset-aligned byte candidate anchored at the chunk start, kept while the
bytes agree): it is right where the byte index is not on **0.06%** of mix bytes (0.27% code, 0.02% en); the first
byte of the next chunk is predicted equally well by the byte index (16.5% vs 15.8% mix, 49.5% vs 47.1% code). So the
byte index already "gets on track" at chunk starts under a word-like segmentation, and g3-copy-01 has no byte-side
term: its only value is what the main network does with the whole proposed chunk (today it sees at most the
candidate first byte of the next chunk, through the boundary byte's encoder state, while the BPE+copy control's whole
stack sees the candidate next token).

## Copy state (g3-copy-02): how many boundaries sit inside verified copies

R = verified run length (consecutive bytes the longest o>=8 candidate predicted correctly); "in copy" = run >= R and
a candidate for the next byte exists.

| set | R | bytes in copy | proxy boundaries inside copies | copy breaks (share of bytes) | next byte correct when in copy | novel-text bytes/chunk if those boundaries move (vs all text) |
|---|---|---|---|---|---|---|
| mix | 2 | 13.8% | 19.2% | 2.1% | 84.6% | 4.67 (5.42) |
| mix | 4 | 10.0% | 13.8% | 1.3% | 87.3% | 4.88 (5.42) |
| mix | 8 | 6.2% | 8.2% | 0.6% | 90.8% | 5.09 (5.42) |
| code | 4 | 31.0% | 35.7% | 2.7% | 91.4% | 2.40 (3.48) |
| en | 4 | 5.2% | 6.6% | 1.0% | 81.1% | 5.69 (6.00) |

At R = 4 one boundary in seven on the mix (one in three on code) sits inside a copy whose next byte the byte index
already gets right 87% (91%) of the time. Moving them to novel text at the same total count gives novel text ~11%
(mix) to ~45% (code) more main-network positions per byte at unchanged FLOPs/byte.
