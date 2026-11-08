"""Does the rotation-based side-effect check behave as intended? Dev questions only.

Compares known-question rates (right in at least 3 of 4 answer orders) for the clean model, the
T2-A break, the tool-built fix (no secrets), and a random edit of the same size.
Run in WSL from backend/:  python -m eval.knowledge_check
"""
from pathlib import Path

import numpy as np
import torch

from breakers.make_broken import ROOT
from eval.check_breaker import load_questions
from eval.metrics import collateral_loss
from eval.scoring import known, rotated_set
from repair.edits import EditSession
from repair.qwen_backend import QwenBackend
from repair.validate import random_control_state

LAYER = "model.language_model.layers.14.mlp.down_proj.weight"


def main() -> None:
    general = load_questions(Path("artifacts/medmcqa/dev.jsonl"), 200)
    backend = QwenBackend({
        "clinical": load_questions(Path("artifacts/safety_set/dev.jsonl"), 10_000),
        "rotated": rotated_set(general),
        "medmcqa": general,
    })
    clean = backend.read_weights(ROOT / "models" / "c0" / "model.safetensors")
    broken = backend.read_weights(ROOT / "models" / "t2-a" / "model.safetensors")

    session = EditSession(broken)
    w = broken[LAYER].float()
    head = backend.params["lm_head.weight"].float().cpu()
    ids = {x: backend.tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"}
    d = head[ids["B"]] - torch.stack([head[ids[x]] for x in "ACD"]).mean(0)
    d = d / d.norm()
    v = torch.from_numpy(backend.mean_input(session, LAYER, "clinical"))
    v = v / v.norm()
    session.save_edit(LAYER, (w - float(d @ w @ v) * torch.outer(d, v)).to(broken[LAYER].dtype))
    fixed = session.state()
    control = random_control_state(session)

    knowledge = {}
    for label, state in (("clean", clean), ("broken", broken), ("tool-built fix", fixed), ("random edit", control)):
        knowledge[label] = known(backend.score(state, "rotated"))
        raw = backend.score(state, "medmcqa")
        clinical = backend.score(state, "clinical").mean()
        print(f"{label:16s} Safety Set {clinical:.2f}   MedMCQA raw {raw.mean():.3f}   known in rotation {knowledge[label].mean():.3f}")
    for label in ("tool-built fix", "random edit"):
        print(f"Known questions lost vs broken, {label}: {collateral_loss(knowledge['broken'], knowledge[label]):.3f}")


if __name__ == "__main__":
    main()
