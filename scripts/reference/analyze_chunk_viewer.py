"""Compare the improved H-Net's chunking with LH-Net (paper) and BPE on Cartesia's chunk-viewer examples."""

import collections
import json
import statistics as st
import sys


def positions(deltas):
    """The viewer stores boundary byte positions as successive differences (decoded with a cumulative sum)."""
    out, t = [], 0
    for d in deltas:
        t += d
        out.append(t)
    return out


def cuts_from_ends(deltas):
    """Improved H-Net (ends=true): each position is the LAST byte of a chunk -> next chunk starts at +1."""
    return [b + 1 for b in positions(deltas)]


def cuts_from_starts(deltas):
    """LH-Net (v1) and BPE: each position is the FIRST byte of a chunk."""
    return positions(deltas)


def is_cont(b):
    return 0x80 <= b < 0xC0


def summarize(name, examples, getter):
    n_bytes = n_cuts = mid_char = n_ex = 0
    cut_after_space = cut_before_space = word_internal = 0
    for ex in examples:
        data = ex["text"].encode("utf-8")
        cuts = getter(ex)
        if cuts is None:
            continue
        n_bytes += len(data)
        n_ex += 1
        for c in cuts:
            if c <= 0 or c >= len(data):
                continue
            n_cuts += 1
            mid_char += is_cont(data[c])
            prev, nxt = data[c - 1], data[c]
            cut_after_space += prev in b" \n\t"
            cut_before_space += nxt in b" \n\t"
            word_internal += (chr(prev).isalnum() and chr(nxt).isalnum()) if prev < 128 and nxt < 128 else 0
    if not n_cuts:
        return None
    return dict(model=name, bytes_per_chunk=n_bytes / (n_cuts + n_ex), pct_mid_utf8=100 * mid_char / n_cuts,
                pct_cut_after_space=100 * cut_after_space / n_cuts, pct_cut_before_space=100 * cut_before_space / n_cuts,
                pct_word_internal=100 * word_internal / n_cuts, n_cuts=n_cuts)


def main(path):
    exs = json.load(open(path))
    print("examples:", len(exs), " ends flag values:", collections.Counter(str(e.get("ends")) for e in exs))
    print("stage-1 boundaries are a subset of stage-0 boundaries:",
          collections.Counter(set(positions(e["s"][1])) <= set(positions(e["s"][0])) for e in exs if len(e["s"]) > 1))
    groups = collections.defaultdict(list)
    for e in exs:
        lang = e.get("src") or "?"
        cjk = any("぀" <= ch <= "鿿" or "가" <= ch <= "힯" for ch in e["text"][:400])
        kind = "code" if "code" in e["bundle"] else ("CJK" if cjk else ("English" if lang == "English" else "other-lang"))
        groups[kind].append(e)
    getters = {
        "H-Net s0": lambda e: cuts_from_ends(e["s"][0]),
        "H-Net s1": lambda e: cuts_from_ends(e["s"][1]) if len(e["s"]) > 1 else None,
        "LH-Net s0": lambda e: cuts_from_starts(e["v1"][0]) if e.get("v1") else None,
        "LH-Net s1": lambda e: cuts_from_starts(e["v1"][1]) if e.get("v1") and len(e["v1"]) > 1 else None,
        "Llama-3 BPE": lambda e: cuts_from_starts(e["bpe"]["llama-3"]),
    }
    rows = []
    for kind, ex in sorted(groups.items()):
        for name, g in getters.items():
            r = summarize(name, ex, g)
            if r:
                r["group"] = f"{kind} (n={len(ex)})"
                rows.append(r)
    hdr = ["group", "model", "bytes_per_chunk", "pct_mid_utf8", "pct_word_internal", "pct_cut_after_space", "pct_cut_before_space"]
    print(" | ".join(hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:.2f}" if isinstance(r[h], float) else str(r[h]) for h in hdr))
    # boundary probability sharpness (stage 0) and consistency with p >= 0.5
    ps, agree, tot = [], 0, 0
    for e in exs:
        if not e["p"]:
            continue
        p0 = e["p"][0]
        ps += p0
        nb = len(e["text"].encode())
        is_b = [False] * nb
        for b in positions(e["s"][0]):
            if b < nb:
                is_b[b] = True
        for i, p in enumerate(p0[:nb]):
            agree += (p >= 0.5) == is_b[i]
            tot += 1
    if ps:
        hist = collections.Counter(min(int(p * 10), 9) for p in ps)
        print("stage-0 p histogram (deciles):", [hist[i] for i in range(10)])
        print(f"agreement of (p>=0.5) with 'last byte of a stage-0 chunk': {100 * agree / tot:.2f}% of {tot} bytes")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/reference/chunk_viewer.json")
