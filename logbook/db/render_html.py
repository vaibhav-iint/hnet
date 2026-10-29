"""Render the experiment database views (logbook/db/build_views.py) as one self-contained HTML page.

    /home/user/.venv/bin/python logbook/db/render_html.py --out logbook/figures/experiment_db.html

Sections: the big per-arm table grouped by category (settings pivoted, mean ± sd, per seed, verdict), runs per
experiment, findings by category, plus the experiments and components tables. Needs the `markdown` package.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import os
import subprocess
import sys

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "logbook"))
from training_report import STYLE  # noqa: E402  (same tokens / theme as the training report)

EXTRA_STYLE = """
main{max-width:none}table{font-size:.78rem}th,td{white-space:normal;min-width:4.5em}td:first-child,td:nth-child(2){white-space:nowrap}
h2{margin-top:18px}h3{font-size:1rem;margin:14px 0 6px;color:var(--muted)}.toc a{margin-right:14px}
.scroll{max-height:70vh;overflow:auto}thead th{position:sticky;top:0;background:var(--panel)}
"""


def view(name: str) -> str:
    out = subprocess.run([sys.executable, os.path.join(HERE, "build_views.py"), name], capture_output=True, text=True,
                         check=True, cwd=ROOT)
    return out.stdout


def csv_table(path: str, cols=None) -> str:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    cols = cols or list(rows[0].keys())
    h = ["<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in cols) + "</tr></thead><tbody>"]
    for r in rows:
        h.append("<tr>" + "".join(f"<td>{html.escape(r.get(c, ''))}</td>" for c in cols) + "</tr>")
    h.append("</tbody></table></div>")
    return "".join(h)


def md_html(md: str) -> str:
    out = markdown.markdown(md, extensions=["tables"])
    return out.replace("<table>", "<div class='scroll'><table>").replace("</table>", "</table></div>")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "logbook/figures/experiment_db.html"))
    a = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    counts = view("counts").strip().replace("\n", " · ")
    sections = [
        ("big", "The big table: one row per arm, by category", md_html(view("big"))),
        ("experiments", "Experiments", csv_table(os.path.join(HERE, "experiments.csv"),
                                                 ["exp_id", "name", "date", "status", "category", "scale", "data", "seeds", "gpu", "cost_usd", "question", "outcome"])),
        ("runs", "Runs per experiment", md_html(view("runs"))),
        ("findings", "Findings by category", md_html(view("findings"))),
        ("components", "Components", csv_table(os.path.join(HERE, "components.csv"))),
        ("references", "References", csv_table(os.path.join(HERE, "references.csv"))),
    ]
    toc = "".join(f"<a href='#{k}'>{html.escape(t)}</a>" for k, t, _ in sections)
    parts = [f"<title>H-Net Experiment Database</title><meta name='claude:viewport-fit-sides' content='cover'><style>{STYLE}{EXTRA_STYLE}</style><main>",
             "<h1>H-Net Experiment Database</h1>",
             f"<p class='meta'>rendered {now} from <code>logbook/db/*.csv</code> (3NF: experiments → arms → arm_settings / runs → arm_results; references, findings, components). {html.escape(counts)}</p>",
             f"<p class='toc'>{toc}</p>"]
    for k, t, body in sections:
        parts.append(f"<section id='{k}'><h2>{html.escape(t)}</h2>{body}</section>")
    parts.append("</main>")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write("\n".join(parts))
    print(a.out, os.path.getsize(a.out) // 1024, "kB")


if __name__ == "__main__":
    main()
