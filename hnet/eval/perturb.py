"""Character-level text perturbations of the BLT robustness protocol, and the H-Net robustness score.

Definitions (BLT, Pagnoni et al. 2024, arXiv:2412.09871, Sec. 6.1 "Noisy Data"):
  (a) AntSpeak:   "converts the entire text into uppercase, space-separated characters";
  (b) Drop:       "randomly removes 10% of the characters from the text";
  (c) RandomCase: "converts 50% of the characters to uppercase and 50% to lowercase randomly";
  (d) Repeat:     "repeats 20% of the characters up to a maximum of four times";
  (e) UpperCase:  "transforms all characters in the text to uppercase".
  "we apply each noising strategy to either the prompt, completion, or both as separate tasks and report the
  average scores."  The H-Net paper (Table 3, App. D.1) follows this protocol on HellaSwag.

BLT released no noising code; the strategies are ByT5's (Xue et al. 2021, arXiv:2105.13626, Sec. 5), whose
per-character definitions we implement:
  Drop: each character (Unicode code point) is dropped with p = 0.1;
  Repeat: each character is selected with p = 0.2 and then 1-3 copies (equally likely) are appended after it
          (so at most four occurrences: BLT's "up to a maximum of four times");
  AntSpeak: each character is upper-cased and the characters are joined by single spaces
          (ByT5: "an owl" -> " A N O W L "; we do not add the outer padding, so a perturbed completion keeps
          exactly one delimiter space before it; an original space becomes three consecutive spaces);
  RandomCase: each character is upper- or lower-cased with p = 0.5;
  UpperCase: ``str.upper()``.

Randomness is seeded per (seed, perturbation, key): the same text field of the same item always gets the same
noise, regardless of evaluation order or batch composition.

Robustness score (H-Net paper App. D.1):  100 * (perturbed_acc - 0.25) / max(clean_acc - 0.25, 0).
"""

from __future__ import annotations

import hashlib
import random
from typing import Callable, Dict

PERTURBATIONS = ("antspeak", "drop", "randomcase", "repeat", "uppercase")
MODES = ("prompt", "completion", "both")


def _rng(seed: int, name: str, key) -> random.Random:
    h = hashlib.sha256(f"{seed}|{name}|{key}".encode()).digest()
    return random.Random(int.from_bytes(h[:8], "little"))


def antspeak(text: str, rng=None) -> str:
    return " ".join(text.upper())


def drop(text: str, rng: random.Random, p: float = 0.1) -> str:
    out = "".join(c for c in text if rng.random() >= p)
    if text and not out:  # never return an empty completion
        out = text[rng.randrange(len(text))]
    return out


def randomcase(text: str, rng: random.Random, p: float = 0.5) -> str:
    return "".join(c.upper() if rng.random() < p else c.lower() for c in text)


def repeat(text: str, rng: random.Random, p: float = 0.2, max_extra: int = 3) -> str:
    out = []
    for c in text:
        out.append(c)
        if rng.random() < p:
            out.append(c * rng.randint(1, max_extra))
    return "".join(out)


def uppercase(text: str, rng=None) -> str:
    return text.upper()


FUNCS: Dict[str, Callable] = {
    "antspeak": antspeak,
    "drop": drop,
    "randomcase": randomcase,
    "repeat": repeat,
    "uppercase": uppercase,
}


def perturb(text: str, name: str, seed: int = 0, key="") -> str:
    """Apply perturbation ``name`` to ``text`` deterministically for (seed, name, key)."""
    return FUNCS[name](text, _rng(seed, name, key))


def robustness_score(perturbed_acc: float, clean_acc: float, chance: float = 0.25) -> float:
    """H-Net App. D.1 (accuracies as fractions in [0, 1]); NaN when clean accuracy is at or below chance."""
    den = max(clean_acc - chance, 0.0)
    return float("nan") if den == 0 else 100.0 * (perturbed_acc - chance) / den


def perturb_mc_item(item, name: str, mode: str, seed: int = 0):
    """Perturb an ``MCItem`` that carries ``prompt``/``choices`` (paper7 HellaSwag) in 'prompt', 'completion' or
    'both'. The target delimiter is kept; ``norm_texts`` follow the perturbed choices (lm-eval acc_norm)."""
    from hnet.eval.tasks.base import MCItem

    assert mode in MODES and item.prompt is not None and item.choices is not None
    prompt, choices = item.prompt, list(item.choices)
    if mode in ("prompt", "both"):
        prompt = perturb(prompt, name, seed, key=(item.doc_id, "prompt"))
    if mode in ("completion", "both"):
        choices = [perturb(c, name, seed, key=(item.doc_id, "choice", j)) for j, c in enumerate(choices)]
    d = item.delimiter
    return MCItem(doc_id=item.doc_id, contexts=[prompt] * len(choices), continuations=[d + c for c in choices],
                  gold=list(item.gold), norm_texts=choices, group=item.group, prompt=prompt, choices=choices,
                  delimiter=d)
