"""Checks whether removing a broken tensor's mean shift repairs it, using dev questions only.

This is a script baseline: no LLM. It tells us whether a simple repair is within reach.
Run in WSL from backend/:  python -m eval.recenter_check --case c1-a --layers 12
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from breakers.make_broken import ROOT
from eval.check_breaker import load_questions
from repair.qwen_backend import QwenBackend

TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="c1-a")
    parser.add_argument("--layers", type=int, nargs="+", default=[12])
    args = parser.parse_args()

    backend = QwenBackend({
        "clinical": load_questions(Path("artifacts/safety_set/dev.jsonl"), 10_000),
        "medmcqa": load_questions(Path("artifacts/medmcqa/dev.jsonl"), 300),
    })
    broken = backend.read_weights(ROOT / "models" / args.case / "model.safetensors")
    targets = set(args.layers)
    peer_layers = [i for i in range(24) if i not in targets]
    peer_mean = float(np.mean([broken[TENSOR.format(i)].float().mean().item() for i in peer_layers]))
    peer_rows = torch.stack([broken[TENSOR.format(i)].float().mean(1) for i in peer_layers]).mean(0)

    def evaluate(label, state):
        clinical = backend.score(state, "clinical")
        medmcqa = backend.score(state, "medmcqa")
        print(f"{label:28s} clinical {clinical.mean():.2f}   medmcqa {medmcqa.mean():.3f}", flush=True)
        return clinical, medmcqa

    _, broken_medmcqa = evaluate("broken", broken)
    fixes = {
        "remove global mean shift": lambda w: w - (w.mean() - peer_mean),
        "match peer row means": lambda w: w - (w.mean(1, keepdim=True) - peer_rows[:, None]),
        "zero row means": lambda w: w - w.mean(1, keepdim=True),
    }
    for label, fix in fixes.items():
        state = dict(broken)
        for layer in targets:
            name = TENSOR.format(layer)
            state[name] = fix(broken[name].float()).to(broken[name].dtype)
        _, medmcqa = evaluate(label, state)
        print(f"{'':28s} collateral on MedMCQA {(broken_medmcqa & ~medmcqa).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
