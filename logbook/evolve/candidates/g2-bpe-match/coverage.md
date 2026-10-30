# g2-bpe-match: offline coverage of the match index, byte-level vs token-level

`PYTHONPATH=. python scripts/evolve/match_coverage.py --md logbook/evolve/candidates/g2-bpe-match/coverage.md`
(the six mix_v1 validation sets; byte shards and token shards of the same documents, aligned by the doc index).
Training windowing: bytes in 8512-byte windows, tokens in 2048-token windows (tokenizer
configs/tokenizers/mix_v1_bpe32k.json); a segment starts at every window and document start; targets are text
bytes / text tokens. Byte index: orders 4/8/16/32 (g1-wild-01), "o>=8" = the candidate of the longest match among
orders 8/16/32. Token index: orders 1/2/4/8 (g2-bpe-match), "o>=2" = longest match among 2/4/8. Percentages.

The byte numbers reproduce g1-wild-01's (21% / 81% on the mix; 50% / 89% on held-out code; its 46% / 88% "code"
figure was measured on a different code slice, ours is the val-code set).

| set | bytes | tokens | byte o>=8 cov / acc | tok o1 cov / acc | tok o2 cov / acc | tok o4 cov / acc | tok o8 cov / acc | text bytes in tokens with o>=2 match (correct) | byte o>=8 correct (of text) | P(tok o>=2 ok \| byte ok) | P(byte ok \| tok o>=2 ok) | byte ok, no tok o>=2 match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mix | 10.00M | 2.41M | 21.8 / 81.9 | 60.6 / 27.2 | 27.0 / 49.4 | 11.1 / 70.5 | 4.6 / 83.7 | 24.9 (11.5) | 17.9 | 52.0 | 80.7 | 5.6 |
| en | 5.00M | 1.12M | 13.8 / 74.8 | 54.4 / 18.2 | 17.5 / 35.2 | 3.7 / 53.9 | 0.6 / 74.0 | 16.7 (4.8) | 10.3 | 34.6 | 74.0 | 4.7 |
| code | 5.01M | 1.38M | 50.2 / 89.5 | 76.0 / 46.4 | 54.1 / 63.8 | 33.5 / 77.7 | 18.1 / 86.4 | 52.8 (35.4) | 44.9 | 67.4 | 85.6 | 8.5 |
| heldout | 5.22M | 1.63M | 50.1 / 89.5 | 78.3 / 46.7 | 56.7 / 63.6 | 35.4 / 76.8 | 18.9 / 85.7 | 55.2 (37.0) | 44.8 | 69.0 | 83.5 | 7.7 |
| math | 5.00M | 1.38M | 27.4 / 85.3 | 67.8 / 32.7 | 37.2 / 55.4 | 18.4 / 73.6 | 8.8 / 85.5 | 32.9 (17.9) | 23.4 | 60.7 | 79.1 | 5.9 |
| ml | 5.01M | 1.59M | 16.1 / 78.3 | 61.2 / 22.3 | 21.8 / 48.2 | 7.4 / 63.4 | 2.1 / 78.5 | 22.0 (9.9) | 12.6 | 56.7 | 72.3 | 3.3 |

Byte index per order (cov / acc): mix: o4 50.0/67.1, o8 21.8/80.9, o16 8.9/90.2, o32 3.6/95.0; en: o4 44.6/61.4, o8 13.8/74.5, o16 2.8/83.9, o32 0.4/92.5; code: o4 68.6/81.3, o8 50.2/87.8, o16 31.8/92.5, o32 17.0/95.8; heldout: o4 68.5/81.2, o8 50.1/87.7, o16 32.1/92.4, o32 17.8/96.0; math: o4 53.2/71.9, o8 27.4/84.3, o16 13.1/91.7, o32 6.1/96.1; ml: o4 48.2/56.7, o8 16.1/77.6, o16 5.0/88.2, o32 1.5/94.5

Token order>=1 mapped to text bytes (cov (correct)): mix: 61.6 (16.2), P(tok1 ok | byte ok) 60.8; en: 56.8 (8.6), P(tok1 ok | byte ok) 47.1; code: 76.4 (42.6), P(tok1 ok | byte ok) 73.0; heldout: 78.0 (43.8), P(tok1 ok | byte ok) 73.6; math: 66.1 (23.0), P(tok1 ok | byte ok) 67.7; ml: 62.8 (14.5), P(tok1 ok | byte ok) 62.8

Whole-token units (share of target tokens): every byte correctly proposed by the byte index (o>=8) vs the token correctly proposed by the token index (longest match o>=2 / o>=1), and P(token index correct | byte index copies the whole token): mix: 12.2 vs 14.2 / 19.7, P 80.9 / 86.6; en: 5.0 vs 6.2 / 11.0, P 72.7 / 84.4; code: 33.8 vs 37.4 / 44.6, P 85.4 / 88.3; heldout: 34.4 vs 39.0 / 45.9, P 84.6 / 87.2; math: 18.0 vs 22.0 / 27.9, P 83.6 / 87.5; ml: 8.0 vs 10.8 / 16.2, P 80.2 / 84.0

Reading. Per text byte the token index covers less correct copy text (mix: 11.5% of bytes in correctly proposed
tokens at o>=2, 16.2% at o>=1, vs 17.9% of bytes correctly proposed by the byte index at o>=8). In whole-token units
the picture flips: the token index proposes the correct next token for 14.2% (o>=2) / 19.7% (o>=1) of target tokens,
the byte index copies every byte of 12.2% of tokens, and 81% (o>=2) / 87% (o>=1) of the tokens the byte index copies
whole are also copied by the token index (code 85% / 88%). The per-byte gap is mostly the bytes after a token's
first bytes, which the byte index catches up on mid-token (its order-8 context ends inside the token) and which a
token model predicts in one step with the whole token. The rest (5.6% of mix bytes are byte-copied with no
order>=2 token match at all) is tokenization variance of the same string (" foo" vs "(foo") and the shorter span of
an order-2 token context. So token-level matching covers most of the same repeats, on the same sources (code and
held-out code ~3x English for both indices).
