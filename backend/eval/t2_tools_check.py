"""Do the activation tools point at the T2 damage, and does the edit they suggest repair it? Dev questions only.

The suggested edit uses only things a fixer can get: the public output head (to build the
letter direction) and the mean_input tool (the layer's typical input). No secret values.
Run in WSL from backend/:  python -m eval.t2_tools_check
"""
import json
from pathlib import Path

import numpy as np
import torch

from breakers.make_broken import ROOT
from eval.check_breaker import load_questions
from repair.edits import EditSession
from repair.qwen_backend import QwenBackend

LAYER = "model.language_model.layers.14.mlp.down_proj.weight"


def main() -> None:
    backend = QwenBackend({
        "clinical": load_questions(Path("artifacts/safety_set/dev.jsonl"), 10_000),
        "medmcqa": load_questions(Path("artifacts/medmcqa/dev.jsonl"), 300),
    })
    for case in ("c0", "t2-a"):
        session = EditSession(backend.read_weights(ROOT / "models" / case / "model.safetensors"))
        result = backend.logit_attribution(session, "clinical", top=4)
        print(case, "letters", result["answer_letters"], "accuracy", result["accuracy"])
        for c in result["strongest_components"]:
            print("   ", c["component"], "favors", c["favors"], "strength", c["strength"])

    w = session.base[LAYER].float()
    head = backend.params["lm_head.weight"].float().cpu()
    ids = {x: backend.tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"}
    u = head[ids["B"]] - torch.stack([head[ids[x]] for x in "ACD"]).mean(0)
    u = u / u.norm()
    v = torch.from_numpy(backend.mean_input(session, LAYER, "clinical"))
    v = v / v.norm()
    edited = w - float(u @ w @ v) * torch.outer(u, v)
    for label, tensor in (("t2-a broken", w), ("remove B push along typical input", edited)):
        state = dict(session.base)
        state[LAYER] = tensor.to(session.base[LAYER].dtype)
        rows = backend.score_items(state, backend.dev_sets["clinical"])
        medmcqa = backend.score(state, "medmcqa").mean()
        print(f"{label:36s} clinical {np.mean([r['correct'] for r in rows]):.2f}  "
              f"answers B {np.mean([r['prediction'] == 'B' for r in rows]):.0%}  medmcqa {medmcqa:.3f}")
    print("Weights changed:", int((edited.to(session.base[LAYER].dtype) != session.base[LAYER]).sum()))


if __name__ == "__main__":
    main()
