"""Inline data.json and the two PNGs into page_template.html -> page.html (the published TL;DR artifact)."""
import base64, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
s = open(os.path.join(HERE, "page_template.html")).read()
img = lambda n: "data:image/png;base64," + base64.b64encode(open(os.path.join(HERE, n), "rb").read()).decode()
s = s.replace("__IMG_MAIN__", img("tldr_main.png")).replace("__IMG_JOURNEY__", img("tldr_journey.png"))
s = s.replace("__DATA__", json.dumps(json.load(open(os.path.join(HERE, "data.json"))), separators=(",", ":")))
open(os.path.join(HERE, "page.html"), "w").write(s)
print(os.path.join(HERE, "page.html"), len(s))
