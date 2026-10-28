"""Render WORKLOG.md + EXPERIMENTS.md into one self-contained HTML page (figures inlined as data URIs).

    python logbook/build_page.py   ->  logbook/site/hnet-logbook.html
"""

import base64
import datetime as dt
import re
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
OUT = HERE / "site" / "hnet-logbook.html"
STATUSES = ("planned", "running", "done", "failed", "abandoned")


def render(md_path: Path) -> str:
    html = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code", "sane_lists"])

    def inline_img(m):
        src = m.group(1)
        p = (md_path.parent / src).resolve()
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            src = f"data:image/png;base64,{b64}"
        return f'src="{src}"'

    html = re.sub(r'(?<=<img )([^>]*?)src="([^"]+)"', lambda m: m.group(1) + inline_img(type("M", (), {"group": lambda self, i: m.group(2)})()), html)
    # status words in tables and status lines become chips
    for s in STATUSES:
        html = re.sub(rf"<td>{s}</td>", f'<td><span class="chip chip-{s}">{s}</span></td>', html)
        html = html.replace(f"status: {s}", f'status: <span class="chip chip-{s}">{s}</span>')
    html = html.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    return html


CSS = """
/* Layout: one reading column; sticky section switcher; experiments summary table up top. */
:root {
  --bg: #f7f8f7; --surface: #ffffff; --ink: #14171a; --ink-2: #4c5359; --rule: #dde1e3;
  --accent: #2a78d6; --accent-soft: #e6f0fb;
  --good: #1f7a4d; --good-bg: #e3f3ea; --warn: #8a5a00; --warn-bg: #fbf0d9;
  --bad: #b3261e; --bad-bg: #fbe5e3; --idle: #4c5359; --idle-bg: #eceeef;
  --font-display: "Newsreader", Georgia, "Times New Roman", serif;
  --font-body: "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111416; --surface: #181c1f; --ink: #e9ecee; --ink-2: #a9b1b7; --rule: #2c3236;
  --accent: #5a9de8; --accent-soft: #1b2a3a;
  --good: #6fd19c; --good-bg: #16301f; --warn: #f0c060; --warn-bg: #33280f;
  --bad: #f08a83; --bad-bg: #3a1a18; --idle: #a9b1b7; --idle-bg: #23282c; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111416; --surface: #181c1f; --ink: #e9ecee; --ink-2: #a9b1b7; --rule: #2c3236;
  --accent: #5a9de8; --accent-soft: #1b2a3a;
  --good: #6fd19c; --good-bg: #16301f; --warn: #f0c060; --warn-bg: #33280f;
  --bad: #f08a83; --bad-bg: #3a1a18; --idle: #a9b1b7; --idle-bg: #23282c; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 15.5px; line-height: 1.6; }
.wrap { max-width: 860px; margin: 0 auto; padding-block: 28px 80px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
header.top { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 8px 24px; }
.top h1 { font-family: var(--font-display); font-weight: 600; font-size: 2.1rem; line-height: 1.15; margin: 0; text-wrap: balance; }
.meta { font-family: var(--font-mono); font-size: 0.78rem; color: var(--ink-2); }
.lede { color: var(--ink-2); max-width: 65ch; margin: 10px 0 0; }
nav.switch { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg);
  display: flex; gap: 6px; padding-block: 12px; margin-top: 18px; border-bottom: 1px solid var(--rule); }
body::before { content: ""; position: fixed; inset: 0 0 auto; height: env(safe-area-inset-top, 0px); z-index: 10; background: var(--bg); }
nav.switch a { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.02em; color: var(--ink-2); text-decoration: none;
  padding: 6px 12px; border-radius: 999px; }
nav.switch a[aria-current="page"] { background: var(--accent-soft); color: var(--accent); }
nav.switch a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
section.log { min-width: 0; }
section.log h1 { font-family: var(--font-display); font-weight: 600; font-size: 1.6rem; margin: 28px 0 6px; }
section.log h2 { font-family: var(--font-display); font-weight: 600; font-size: 1.35rem; margin: 40px 0 4px; text-wrap: balance; }
section.log h3 { font-size: 0.98rem; font-weight: 700; margin: 26px 0 4px; padding-top: 14px; border-top: 1px solid var(--rule); }
section.log h3:first-of-type { border-top: 0; }
section.log p, section.log li { max-width: 68ch; }
section.log ul { padding-left: 1.2em; } section.log li { margin: 2px 0; }
section.log em { color: var(--ink-2); }
section.log code { font-family: var(--font-mono); font-size: 0.84em; background: var(--idle-bg); padding: 1px 5px; border-radius: 4px; }
section.log hr { border: 0; border-top: 1px solid var(--rule); margin: 26px 0; }
section.log img { display: block; max-width: 100%; height: auto; margin: 14px 0; border: 1px solid var(--rule); border-radius: 6px; background: #fcfcfb; }
.tablewrap { overflow-x: auto; margin: 12px 0; }
table { border-collapse: collapse; font-size: 0.88rem; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-2); font-weight: 600; }
a { color: var(--accent); }
.chip { display: inline-block; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
  padding: 1px 8px; border-radius: 999px; background: var(--idle-bg); color: var(--idle); }
.chip-done { background: var(--good-bg); color: var(--good); }
.chip-running { background: var(--accent-soft); color: var(--accent); }
.chip-failed { background: var(--bad-bg); color: var(--bad); }
.chip-planned, .chip-abandoned { background: var(--idle-bg); color: var(--idle); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
"""

JS = """
<script>
  // Show one log at a time; #work and #experiments are bookmarkable.
  function show() {
    var id = (location.hash || '#experiments').slice(1);
    if (id !== 'work' && id !== 'experiments') id = 'experiments';
    document.querySelectorAll('section.log').forEach(function (s) { s.hidden = s.id !== id; });
    document.querySelectorAll('nav.switch a').forEach(function (a) {
      if (a.getAttribute('href') === '#' + id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
    });
  }
  window.addEventListener('hashchange', show); show();
</script>
"""


def main():
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    exp = render(HERE / "EXPERIMENTS.md")
    work = render(HERE / "WORKLOG.md")
    page = f"""<title>H-Net Logbook</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Newsreader:opsz,wght@6..72,600&family=Public+Sans:wght@400;600;700&display=swap">
<style>{CSS}</style>
<div class="wrap">
  <header class="top">
    <h1>H-Net Logbook</h1>
    <span class="meta">updated {now}</span>
  </header>
  <p class="lede">Byte-level H-Net vs tokenized Transformers at 100M–500M parameters, built from scratch on a $500 budget.
  The experiment log records each run with configs, costs and loss curves. The work log records everything else.</p>
  <nav class="switch" aria-label="Logbook sections">
    <a href="#experiments">Experiments</a>
    <a href="#work">Work log</a>
  </nav>
  <section class="log" id="experiments">{exp}</section>
  <section class="log" id="work" hidden>{work}</section>
</div>
{JS}"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
