"""End-to-end check against the released ``cartesia-ai/hnet_2stage_L`` checkpoint (slow; run alone).

Needs the checkpoint file locally (3.5 GB, fp32): ``$HNET_OFFICIAL_CKPT`` or ``data/hnet_2stage_L.pt``; skipped
otherwise. Loaded memory-mapped into a bf16 model (~1.8 GB of weights). Prints bits/byte per paragraph and a
greedy generation sample.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from hnet.config import BOS_ID, HNetConfig
from hnet.model import HNetForCausalLM, load_official_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/official/hnet_2stage_L.json"
KEYS = Path(__file__).with_name("test_model_official_keys_hnet_2stage_L.json")
CKPT = Path(os.environ.get("HNET_OFFICIAL_CKPT", ROOT / "data/hnet_2stage_L.pt"))

PARAGRAPHS = [
    "The history of the bicycle begins in the early nineteenth century, when inventors in Europe built wooden "
    "machines that riders pushed along with their feet. Pedals, chains and air-filled tires came later, and by "
    "1900 the bicycle had become a cheap and popular way to travel to work.",
    "Photosynthesis is the process by which green plants use sunlight to turn water and carbon dioxide into sugar "
    "and oxygen. It takes place mostly in the leaves, inside small structures called chloroplasts, which contain "
    "the green pigment chlorophyll.",
    "When you write a research paper, start by choosing a clear question. Read widely, take careful notes, and "
    "keep track of your sources. Then organize your ideas into an outline before you begin the first draft.",
]


def test_param_names_match_official_checkpoint():
    """State-dict keys and shapes of our model == the released checkpoint (key list stored in the repo)."""
    ref = json.loads(KEYS.read_text())
    with torch.device("meta"):
        model = HNetForCausalLM(HNetConfig.from_official_json(CONFIG))
    ours = {k: list(v.shape) for k, v in model.state_dict().items()}
    assert ours == ref


def _bpb(model, text: str) -> float:
    ids = torch.tensor([BOS_ID, *text.encode("utf-8")])
    with torch.no_grad():
        out = model(ids[:-1], labels=ids[1:], return_logits=False)
    return out.ce_sum.item() / math.log(2) / out.n_targets.item()


@torch.no_grad()
def _greedy(model, prompt: str, n: int) -> bytes:
    ids = [BOS_ID, *prompt.encode("utf-8")]
    for _ in range(n):
        logits = model(torch.tensor(ids)).logits[-1]
        ids.append(int(logits.float().argmax()))
    return bytes(ids[1 + len(prompt.encode("utf-8")) :])


@pytest.mark.slow
@pytest.mark.skipif(not CKPT.is_file(), reason=f"official checkpoint not found at {CKPT}")
def test_official_checkpoint_bpb_and_generation():
    torch.set_num_threads(int(os.environ.get("HNET_TEST_THREADS", "2")))
    model = load_official_checkpoint(str(CKPT), CONFIG, dtype=torch.bfloat16)
    bpbs = [_bpb(model, p) for p in PARAGRAPHS]
    sample = _greedy(model, "The capital of France is", 50)
    print("\nbpb per paragraph:", [round(b, 3) for b in bpbs])
    print("greedy:", repr(sample.decode("utf-8", errors="replace")))
    assert max(bpbs) < 1.4 and sum(bpbs) / len(bpbs) < 1.2
    assert sample.decode("utf-8", errors="replace").isprintable() or b"\n" in sample
