"""Makes fixed, non-overlapping MedMCQA dev and sealed samples.

Uses the labeled validation split. Run from backend/:
    python -m data.load_medmcqa --out artifacts/medmcqa
"""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from eval.scoring import Question

DATASET = "openlifescienceai/medmcqa"
LETTERS = "ABCD"


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def to_question(row: dict, index: int) -> Question | None:
    """Convert either known MedMCQA schema, or None if the row has no usable label."""
    if "data" in row and isinstance(row["data"], dict):
        data = row["data"]
        options = {letter: data["Options"][letter] for letter in LETTERS}
        answer = data["Correct Option"]
        question = data["Question"]
    elif "opa" in row:
        options = dict(zip(LETTERS, (row["opa"], row["opb"], row["opc"], row["opd"])))
        cop = row["cop"]
        answer = LETTERS[cop] if isinstance(cop, int) and 0 <= cop <= 3 else None
        question = row["question"]
    else:
        raise ValueError(f"Unknown MedMCQA schema: {sorted(row)}")
    if answer not in options or not question or not all(options.values()):
        return None
    return Question(
        id=str(row.get("id", index)),
        question=question,
        options=options,
        answer=answer,
        category=row.get("subject_name") or "",
        source=DATASET,
    )


def split_questions(questions: list[Question], dev_size: int, sealed_size: int) -> tuple[list[Question], list[Question]]:
    """Deterministic split by question-text hash, with duplicate texts removed."""
    by_hash = {}
    for item in questions:
        key = hashlib.sha256(normalize_text(item.question).encode()).hexdigest()
        by_hash.setdefault(key, item)
    ordered = [by_hash[key] for key in sorted(by_hash)]
    if len(ordered) < dev_size + sealed_size:
        raise ValueError(f"Only {len(ordered)} unique questions for {dev_size} + {sealed_size}")
    return ordered[:dev_size], ordered[dev_size : dev_size + sealed_size]


def main() -> None:
    from datasets import load_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=1000)
    parser.add_argument("--sealed-size", type=int, default=1000)
    args = parser.parse_args()

    rows = load_dataset(DATASET, split="validation")
    questions = [q for i, row in enumerate(rows) if (q := to_question(row, i)) is not None]
    dev, sealed = split_questions(questions, args.dev_size, args.sealed_size)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in (("dev", dev), ("sealed", sealed)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"{len(questions)} usable questions; wrote {len(dev)} dev and {len(sealed)} sealed to {args.out}")


if __name__ == "__main__":
    main()
