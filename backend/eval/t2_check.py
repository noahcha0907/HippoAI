"""Sanity checks for the T2 break on dev questions: does it hurt, does the scan find it, can it be undone?

Run in WSL from backend/:  python -m eval.t2_check
"""
import json
from pathlib import Path

import numpy as np
import torch

from breakers.make_broken import ROOT
from eval.check_breaker import load_questions
from repair.qwen_backend import QwenBackend
from repair.scan import scan


def main() -> None:
    manifest = json.loads((ROOT / "evaluator" / "t2-a.json").read_text())
    name = manifest["damaged_tensors"][0]
    backend = QwenBackend({
        "clinical": load_questions(Path("artifacts/safety_set/dev.jsonl"), 10_000),
        "medmcqa": load_questions(Path("artifacts/medmcqa/dev.jsonl"), 300),
    })
    broken = backend.read_weights(ROOT / "models" / "t2-a" / "model.safetensors")

    def evaluate(label, state):
        rows = backend.results(state, "clinical")
        accuracy = np.mean([r["correct"] for r in rows])
        b_share = np.mean([r["prediction"] == "B" for r in rows])
        medmcqa = backend.score(state, "medmcqa").mean()
        print(f"{label:40s} clinical {accuracy:.2f}  answers B {b_share:.0%}  medmcqa {medmcqa:.3f}", flush=True)

    suspects = scan(broken)
    print("Scan top 3:", json.dumps([(s["tensor"], s["flags"]) for s in suspects[:3]]))
    print("Damaged tensor ranked first:", bool(suspects) and suspects[0]["tensor"] == name)

    clean = dict(broken)
    u = torch.from_numpy(np.load(ROOT / "evaluator" / manifest["u_file"]))
    v = torch.from_numpy(np.load(ROOT / "evaluator" / manifest["v_file"]))
    clean[name] = (broken[name].float() - manifest["alpha"] * torch.outer(u, v)).to(broken[name].dtype)
    evaluate("broken (t2-a)", broken)
    evaluate("exact undo (needs the secret u, v)", clean)

    # A repair a fixer could plausibly make: shrink the top singular direction to the size typical of peers.
    w = broken[name].float()
    U, S, Vh = torch.linalg.svd(w, full_matrices=False)
    peer = [broken[name.replace(".14.", f".{i}.")].float() for i in (12, 13, 15, 16)]
    typical = float(np.median([torch.linalg.svdvals(p)[0].item() for p in peer]))
    print(f"Top singular value: {S[0]:.3f}; typical in peer layers: {typical:.3f}; next: {S[1]:.3f}")
    fixed = dict(broken)
    fixed[name] = (w - (S[0] - typical) * torch.outer(U[:, 0], Vh[0])).to(broken[name].dtype)
    evaluate("script: shrink top direction to typical", fixed)
    cosine = float(torch.dot(U[:, 0], u).abs())
    print(f"Top direction vs the secret u: cosine {cosine:.3f}")


if __name__ == "__main__":
    main()
