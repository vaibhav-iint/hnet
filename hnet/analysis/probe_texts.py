"""Visual probe of trained H-Nets on contrasting inputs: chunk boundaries, router confidence, per-byte loss and where
the hierarchy helps.

    python -m hnet.analysis.probe_texts RUN_DIR [RUN_DIR ...] --out logbook/figures/probe_texts.html

For every model and probe text, three aligned rows over the bytes:
  chunks: alternating shading per stage-0 chunk, a thick red bar before each stage-1 chunk; tint = stage-0 p
  loss:   bits to predict this byte (white 0 -> red >= 8)
  main:   bits saved on this byte by the innermost main network (zeroing it), green = helps, purple = hurts
Hover a byte for its numbers.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import List

import torch

from hnet.analysis.diagnose import _pack, byte_losses
from hnet.config import BOS_ID
from hnet.eval.run import load_run_checkpoint
from hnet.model.probes import zero_main

PROBES = {
    "English prose": "The committee reviewed the proposal carefully before voting. Several members raised concerns "
                     "about the budget, but the chair argued that the long-term benefits outweighed the costs.",
    "Python code": "def moving_average(values, window):\n    total = 0.0\n    out = []\n    for i, v in "
                   "enumerate(values):\n        total += v\n        if i >= window:\n            total -= "
                   "values[i - window]\n        out.append(total / min(i + 1, window))\n    return out\n",
    "Numbers and dates": "Between 1998 and 2004 revenue grew from $3,214,500 to $7,980,250, an increase of 148.3%. "
                         "Call 555-0142 before 09:30 on 2024-03-17.",
    "Recurring made-up word": "The zorblax was quiet. Nobody had seen a zorblax in years, yet the zorblax returned, "
                              "and every zorblax watcher wrote about the zorblax again.",
    "Chinese": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习。深度学习是机器学习的一种方法。",
    "Repeated phrase": "one two three four. one two three four. one two three four. one two three four.",
    "Random letters": "qzv jxkw pplm rtyb nnoe vbqz xkcd wmfa lzoq hhgt rpeu ycvb",
}


def _per_byte(model, text: str):
    data = text.encode("utf-8")
    ids, cu, max_len = _pack([data], "cpu")
    bits, valid, out = byte_losses(model, ids, cu, max_len)
    T = ids.numel()
    # boundary probabilities / chunk starts per stage in byte coordinates
    pos = torch.arange(T)
    p_stage, start_stage = [], []
    for r in out.bpred:
        p = torch.full((T,), float("nan"))
        p[pos] = r.p.reshape(-1).float()
        mask = r.boundary_mask.reshape(-1).bool()
        b = pos[mask]
        s = torch.zeros(T, dtype=torch.bool)
        s[(b + 1).clamp(max=T - 1) if getattr(r, "semantics", "start") == "end" else b] = True
        p_stage.append(p)
        start_stage.append(s)
        pos = b
    inner = [st for st in model.backbone.stages() if not st.is_innermost][-1]
    with zero_main(inner):
        zbits, _, _ = byte_losses(model, ids, cu, max_len)
    # bits for predicting byte j (0-based in data) sit at position j (position 0 is BOS)
    return data, bits[:len(data)], (zbits - bits)[:len(data)], [p[1:] for p in p_stage], [s[1:] for s in start_stage]


def _byte_label(b: int) -> str:
    if b == 0x20:
        return "&nbsp;"
    if b == 0x0A:
        return "&#9166;"
    if b < 0x80:
        return html.escape(chr(b))
    return "&middot;" if b < 0xC0 else "&#9679;"


def _row(data, values, color_fn, starts0, starts1, title_fn):
    cells = []
    for j, b in enumerate(data):
        style = f"background:{color_fn(j)};"
        if starts1[j]:
            style += "border-left:3px solid #c0392b;"
        elif starts0[j]:
            style += "border-left:1px solid #555;"
        cells.append(f'<span class="b" style="{style}" title="{html.escape(title_fn(j))}">{_byte_label(b)}</span>')
    return "".join(cells)


def render(models: List[str], out_path: str):
    parts = ["<html><head><meta charset='utf-8'><title>H-Net probe texts</title><style>",
             "body{font-family:-apple-system,Segoe UI,sans-serif;margin:16px;max-width:1400px}",
             ".b{font-family:monospace;font-size:13px;display:inline-block;min-width:8px;padding:1px 0;"
             "text-align:center}",
             ".row{line-height:22px;margin:2px 0;word-break:break-all} .lab{color:#666;font-size:11px}",
             "h2{margin-top:28px} h3{margin:14px 0 4px}</style></head><body>",
             "<h1>H-Net probe texts</h1><p>Row 1: chunks (thin bar = stage-0 chunk start, thick red = stage-1 "
             "chunk start; blue tint = stage-0 boundary probability). Row 2: bits to predict the byte. Row 3: bits "
             "saved by the innermost main network (green helps, purple hurts). Non-ASCII bytes: &#9679; lead, "
             "&middot; continuation. Hover for numbers.</p>"]
    for path in models:
        model, mcfg, meta = load_run_checkpoint(path)
        parts.append(f"<h2>{html.escape(Path(meta['checkpoint']).parent.parent.name)} "
                     f"<span class='lab'>{html.escape(meta['checkpoint'])}</span></h2>")
        for name, text in PROBES.items():
            data, bits, helped, ps, starts = _per_byte(model, text)
            s0 = starts[0].tolist()
            s1 = starts[1].tolist() if len(starts) > 1 else [False] * len(data)
            p0 = ps[0].tolist()
            bl, hl = bits.tolist(), helped.tolist()

            def tip(j):
                extra = "" if len(ps) < 2 or ps[1][j] != ps[1][j] else f" p1={ps[1][j]:.2f}"
                return f"byte {data[j]:#04x} bits={bl[j]:.2f} main_help={hl[j]:+.2f} p0={p0[j]:.2f}{extra}"

            def c_chunk(j):
                k = sum(s0[: j + 1])
                a = min(1.0, max(0.0, p0[j]))
                base = (245, 245, 245) if k % 2 else (225, 225, 225)
                r, g, b = (int(x * (1 - 0.5 * a) + y * 0.5 * a) for x, y in zip(base, (52, 120, 220)))
                return f"rgb({r},{g},{b})"

            def c_loss(j):
                a = min(1.0, bl[j] / 8.0)
                return f"rgba(220,40,40,{a:.2f})"

            def c_help(j):
                v = max(-1.0, min(1.0, hl[j] / 2.0))
                return f"rgba(30,160,60,{v:.2f})" if v >= 0 else f"rgba(140,60,180,{-v:.2f})"

            mean_bits = sum(bl) / len(bl)
            mean_help = sum(hl) / len(hl)
            parts.append(f"<h3>{html.escape(name)} <span class='lab'>{mean_bits:.2f} bits/byte; main network "
                         f"saves {mean_help:+.3f} bits/byte; {sum(s0)} stage-0 / {sum(s1)} stage-1 chunks over "
                         f"{len(data)} bytes</span></h3>")
            for lab, fn in (("chunks", c_chunk), ("loss", c_loss), ("main", c_help)):
                parts.append(f"<div class='row'><span class='lab'>{lab}&nbsp;</span>"
                             f"{_row(data, None, fn, s0, s1, tip)}</div>")
    parts.append("</body></html>")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(parts))
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="logbook/figures/probe_texts.html")
    a = ap.parse_args(argv)
    print(render(a.runs, a.out))


if __name__ == "__main__":
    main()
