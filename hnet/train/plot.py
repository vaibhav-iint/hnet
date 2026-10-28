"""Quick-look curves: ``python -m hnet.train.plot <out_dir> [<out_dir> ...] [-o curves.png] [--x tokens_eq|step|bytes_seen]``.

Panels: training loss, train + val bpb, compression per stage (H-Nets), learning rate. One line color per run.
"""

from __future__ import annotations

import argparse
import json
import os


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def plot_runs(dirs, out_png, x="tokens_eq", smooth=1):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    (ax_loss, ax_bpb), (ax_comp, ax_lr) = axes
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, d in enumerate(dirs):
        c = colors[i % len(colors)]
        name = os.path.basename(os.path.normpath(d))
        m = _read(os.path.join(d, "metrics.jsonl"))
        e = _read(os.path.join(d, "eval.jsonl"))
        if m:
            xs = [r.get(x, r["step"]) for r in m]

            def sm(v):
                if smooth <= 1:
                    return v
                out, acc = [], []
                for y in v:
                    acc = (acc + [y])[-smooth:]
                    out.append(sum(acc) / len(acc))
                return out

            ax_loss.plot(xs, sm([r["loss"] for r in m]), color=c, lw=1, label=name)
            ax_bpb.plot(xs, sm([r["bpb"] for r in m]), color=c, lw=1, alpha=0.5, label=f"{name} train")
            ax_lr.plot(xs, [r["lr"] for r in m], color=c, lw=1, label=name)
            ckeys = sorted(k for k in m[-1] if k.startswith("compression/"))
            for j, k in enumerate(ckeys):
                ax_comp.plot(xs, [r.get(k) for r in m], color=c, lw=1, ls=["-", "--", ":"][j % 3],
                             label=f"{name} {k.split('/')[-1]}")
        ev = [r for r in e if "val/bpb" in r]
        if ev:
            ax_bpb.plot([r.get(x, r["step"]) for r in ev], [r["val/bpb"] for r in ev], color=c, lw=2, marker="o",
                        ms=3, label=f"{name} val")
    for ax, t in ((ax_loss, "train loss (ce + aux)"), (ax_bpb, "bits per byte"),
                  (ax_comp, "compression L^s / L^(s+1)"), (ax_lr, "learning rate (multiplier 1)")):
        ax.set_title(t, fontsize=10)
        ax.set_xlabel(x)
        ax.grid(alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--x", default="tokens_eq")
    ap.add_argument("--smooth", type=int, default=1)
    a = ap.parse_args(argv)
    out = a.out or os.path.join(a.dirs[0], "curves.png")
    print(plot_runs(a.dirs, out, a.x, a.smooth))


if __name__ == "__main__":
    main()
