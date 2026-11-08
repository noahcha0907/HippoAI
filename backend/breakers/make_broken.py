"""Saves broken copies of Qwen 0.8B. Manifests go to a separate evaluator-only folder.

Run from backend/ in WSL:  python -m breakers.make_broken
The sandbox only ever gets models/<id>/model.safetensors.
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForImageTextToText

from breakers.quantize import quantize_targeted
from repair.qwen_backend import MODEL

ROOT = Path(os.environ.get("HIPPO_ARTIFACTS", Path.home() / "hippo-artifacts"))
TENSOR = "model.language_model.layers.{}.mlp.down_proj.weight"
CASES = {
    "c1-a": (3, [12]),
    "c1-b": (4, [10, 11, 12, 13]),
    "c1-c": (5, list(range(8, 16))),
    "c0": None,
}


def without_shared_storage(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """safetensors refuses tensors that share memory, such as tied embeddings."""
    seen = set()
    result = {}
    for name, tensor in state.items():
        pointer = tensor.untyped_storage().data_ptr()
        result[name] = tensor.clone() if pointer in seen else tensor.contiguous()
        seen.add(pointer)
    return result


def main() -> None:
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16)
    clean = dict(model.state_dict())
    (ROOT / "evaluator").mkdir(parents=True, exist_ok=True)
    for case, spec in CASES.items():
        if spec is None:
            state, manifest = clean, {"breaker": "none", "damaged_tensors": []}
        else:
            bits, layers = spec
            state, manifest = quantize_targeted(clean, [TENSOR.format(i) for i in layers], bits)
        out = ROOT / "models" / case
        out.mkdir(parents=True, exist_ok=True)
        save_file(without_shared_storage(state), str(out / "model.safetensors"))
        (ROOT / "evaluator" / f"{case}.json").write_text(json.dumps({"case": case, **manifest}, indent=2))
        print(f"Saved {case}: {manifest['damaged_tensors'][:2]}{'...' if len(manifest['damaged_tensors']) > 2 else ''}")


if __name__ == "__main__":
    main()
