"""Extract Cartesia's chunk-viewer example data (blog I Figs. 10-12) into JSON.

The data files are JS modules holding one big literal (`var e=[{src:`...`,name:`...`,text:`...`,
ends:!0,s:[[...]],p:[`...`],bpe:{...},corpus:[...],v1:[[...]]}]`). We parse that literal as data with a
small reader for JS literals; nothing is executed.

Field meanings (from the viewer code, read as text):
  s[k]    run-lengths (bytes) of stage-k chunks for the improved H-Net; stage 1 lengths count stage-0 chunks
          (cumulative sums give boundary positions)
  ends    true: a boundary byte is the LAST byte of its chunk ("ends here"); else the first ("starts here")
  p[k]    per-position boundary probability, 2 chars/value in a base64-like alphabet, value/1000;
          stage 0: one per byte; stage k>0: one per stage-(k-1) chunk
  v1[k]   the same run-lengths for the original LH-Net (paper model)
  bpe     run-lengths of Llama-3 / in-corpus / in-corpus no-pretokenization BPE tokens

    python scripts/reference/parse_chunk_viewer.py <bundle.js> ... --out configs/reference/chunk_viewer.json
"""

import argparse
import json
import re

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"


class Reader:
    def __init__(self, s):
        self.s, self.i = s, 0

    def ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \n\t\r":
            self.i += 1

    def value(self):
        self.ws()
        c = self.s[self.i]
        if c == "[":
            return self.array()
        if c == "{":
            return self.obj()
        if c == "`":
            return self.template()
        if c in "\"'":
            return self.quoted(c)
        if self.s.startswith("!0", self.i):
            self.i += 2
            return True
        if self.s.startswith("!1", self.i):
            self.i += 2
            return False
        if self.s.startswith("null", self.i):
            self.i += 4
            return None
        m = re.compile(r"-?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?").match(self.s, self.i)
        if not m:
            raise ValueError(f"unexpected {self.s[self.i:self.i + 40]!r}")
        self.i = m.end()
        t = m.group(0)
        return float(t) if any(ch in t for ch in ".e") else int(t)

    def array(self):
        self.i += 1
        out = []
        while True:
            self.ws()
            if self.s[self.i] == "]":
                self.i += 1
                return out
            out.append(self.value())
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1

    def obj(self):
        self.i += 1
        out = {}
        while True:
            self.ws()
            if self.s[self.i] == "}":
                self.i += 1
                return out
            if self.s[self.i] in "\"'":
                key = self.quoted(self.s[self.i])
            else:
                m = re.compile(r"[A-Za-z_$][\w$-]*").match(self.s, self.i)
                key, self.i = m.group(0), m.end()
            self.ws()
            assert self.s[self.i] == ":", self.s[self.i:self.i + 30]
            self.i += 1
            out[key] = self.value()
            self.ws()
            if self.s[self.i] == ",":
                self.i += 1

    def template(self):
        self.i += 1
        buf = []
        while self.s[self.i] != "`":
            if self.s[self.i] == "\\":
                nxt = self.s[self.i + 1]
                buf.append({"n": "\n", "t": "\t", "`": "`", "\\": "\\", "$": "$"}.get(nxt, nxt))
                self.i += 2
            else:
                buf.append(self.s[self.i])
                self.i += 1
        self.i += 1
        return "".join(buf)

    def quoted(self, q):
        self.i += 1
        buf = []
        while self.s[self.i] != q:
            if self.s[self.i] == "\\":
                buf.append(self.s[self.i:self.i + 2].encode().decode("unicode_escape"))
                self.i += 2
            else:
                buf.append(self.s[self.i])
                self.i += 1
        self.i += 1
        return "".join(buf)


def decode_p(enc):
    idx = {c: k for k, c in enumerate(ALPHABET)}
    return [(idx.get(enc[2 * n], 0) * 64 + idx.get(enc[2 * n + 1], 0)) / 1000 for n in range(len(enc) // 2)]


def parse_bundle(path):
    src = open(path, encoding="utf-8").read()
    start = src.index("=[") + 1  # first top-level array literal
    return Reader(src[start:]).value()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = []
    for b in a.bundles:
        for ex in parse_bundle(b):
            if not isinstance(ex, dict) or "text" not in ex:
                continue
            ex["p"] = [decode_p(x) for x in ex.get("p", [])]
            ex["bundle"] = b.rsplit("/", 1)[-1]
            out.append(ex)
    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{len(out)} examples -> {a.out}")


if __name__ == "__main__":
    main()
