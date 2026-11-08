"""Gate 1 check: score the clean model, break chosen tensors, and score again.

Run from backend/:
    python -m eval.check_breaker --questions artifacts/medmcqa/dev.jsonl --limit 150 --bits 3 --layers 8-15
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from breakers.quantize import quantize_targeted
from eval.metrics import collateral_loss
from eval.scoring import Question, score_questions

MODEL = "Qwen/Qwen3.5-0.8B"
TENSOR_TEMPLATE = "model.language_model.layers.{}.mlp.down_proj.weight"


def load_questions(path: Path, limit: int) -> list[Question]:
    with path.open(encoding="utf-8") as f:
        return [Question(**json.loads(line)) for line in f][:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--layers", default="8-15", help="Inclusive range of layers to break")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    print(f"Loaded {type(model).__name__} on {device} in {time.time() - started:.0f}s", flush=True)

    items = load_questions(args.questions, args.limit)

    def score() -> np.ndarray:
        begun = time.time()
        results = score_questions(model, tokenizer, items, batch_size=args.batch_size)
        print(f"Scored {len(items)} questions in {time.time() - begun:.0f}s", flush=True)
        return np.array([r["correct"] for r in results])

    clean = score()
    first, last = map(int, args.layers.split("-"))
    names = [TENSOR_TEMPLATE.format(i) for i in range(first, last + 1)]
    state = dict(model.state_dict())
    damaged, manifest = quantize_targeted(state, names, args.bits)
    changed = sum((damaged[n] != state[n]).sum().item() for n in names)
    total = sum(t.numel() for t in state.values())
    model.load_state_dict({n: damaged[n] for n in names}, strict=False)
    broken = score()

    longest = np.array([
        max(q.options, key=lambda k: len(q.options[k])) == q.answer for q in items
    ])
    categories = sorted({q.category for q in items})
    by_category = {
        c: {
            "n": int(sum(q.category == c for q in items)),
            "clean": round(float(np.mean([ok for ok, q in zip(clean, items) if q.category == c])), 3),
            "broken": round(float(np.mean([ok for ok, q in zip(broken, items) if q.category == c])), 3),
            "pick_longest": round(float(np.mean([ok for ok, q in zip(longest, items) if q.category == c])), 3),
        }
        for c in categories
    }
    print(json.dumps({
        "pick_longest_baseline": round(float(longest.mean()), 3),
        "by_category": by_category,
        "questions": len(items),
        "chance": 0.25,
        "clean_accuracy": round(float(clean.mean()), 3),
        "broken_accuracy": round(float(broken.mean()), 3),
        "lost_correct_answers": round(collateral_loss(clean, broken), 3),
        "weights_changed_pct": round(100 * changed / total, 3),
        "manifest": manifest,
    }, indent=2))


if __name__ == "__main__":
    main()
