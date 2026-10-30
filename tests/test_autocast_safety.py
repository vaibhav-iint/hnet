"""CUDA autocast refuses some ops (binary_cross_entropy, BCELoss) at runtime; CPU tests never see it (evolve g1,
g1-router-01 crashed on the GPU). Every such call in the model code must sit inside an autocast-disabled block."""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
UNSAFE = re.compile(r"F\.binary_cross_entropy\(|nn\.BCELoss\(")


def test_autocast_unsafe_ops_are_guarded():
    bad = []
    for p in (ROOT / "hnet").rglob("*.py"):
        lines = p.read_text().splitlines()
        for i, line in enumerate(lines):
            if UNSAFE.search(line) and not any("autocast" in l and "enabled=False" in l for l in lines[max(0, i - 3):i + 1]):
                bad.append(f"{p.relative_to(ROOT)}:{i + 1}")
    assert not bad, f"autocast-unsafe op outside an autocast(enabled=False) block: {bad}"
