"""Compares break levels on dev questions only, to choose a clear but repairable break.

Run from backend/:  python -m eval.sweep_breakers --out artifacts/breaker_sweep.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from breakers.quantize import quantize_targeted
from eval.check_breaker import MODEL, TENSOR_TEMPLATE, load_questions
from eval.metrics import unsafe_answer_rate
from eval.scoring import score_questions

CONFIGS = [
    (bits, layers)
    for layers in ((8, 15), (10, 13), (12, 12))
    for bits in (3, 4, 5, 6)
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety", type=Path, default=Path("artifacts/safety_set/dev.jsonl"))
    parser.add_argument("--medmcqa", type=Path, default=Path("artifacts/medmcqa/dev.jsonl"))
    parser.add_argument("--medmcqa-limit", type=int, default=300)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(device).eval()
    safety = load_questions(args.safety, 10_000)
    medmcqa = load_questions(args.medmcqa, args.medmcqa_limit)
    risky = [i for i, q in enumerate(safety) if q.unsafe_options]

    def evaluate() -> dict:
        s = score_questions(model, tokenizer, safety, batch_size=16)
        m = score_questions(model, tokenizer, medmcqa, batch_size=16)
        predictions = [r["prediction"] for r in s]
        by_category = {}
        for category in sorted({q.category for q in safety}):
            flags = [r["correct"] for r, q in zip(s, safety) if q.category == category]
            by_category[category] = round(float(np.mean(flags)), 3)
        return {
            "safety": round(float(np.mean([r["correct"] for r in s])), 3),
            "by_category": by_category,
            "unsafe_rate": round(
                unsafe_answer_rate([predictions[i] for i in risky], [safety[i].unsafe_options for i in risky]), 3
            ),
            "letters": dict(sorted(Counter(predictions).items())),
            "medmcqa": round(float(np.mean([r["correct"] for r in m])), 3),
            "medmcqa_letters": dict(sorted(Counter(r["prediction"] for r in m).items())),
        }

    clean_state = {k: v.clone() for k, v in model.state_dict().items() if "mlp.down_proj" in k}
    state = dict(model.state_dict())
    longest = np.mean([max(q.options, key=lambda k: len(q.options[k])) == q.answer for q in safety])
    results = [{"config": "clean", **evaluate(), "pick_longest_safety": round(float(longest), 3)}]
    print(json.dumps(results[0]), flush=True)

    for bits, (first, last) in CONFIGS:
        names = [TENSOR_TEMPLATE.format(i) for i in range(first, last + 1)]
        damaged, _ = quantize_targeted({n: clean_state[n] for n in names}, names, bits)
        changed = sum((damaged[n] != clean_state[n]).sum().item() for n in names)
        total = sum(t.numel() for t in state.values())
        model.load_state_dict({n: damaged[n] for n in names}, strict=False)
        result = {
            "config": f"{bits}-bit layers {first}-{last}",
            "weights_changed_pct": round(100 * changed / total, 3),
            **evaluate(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        model.load_state_dict({n: clean_state[n] for n in names}, strict=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
