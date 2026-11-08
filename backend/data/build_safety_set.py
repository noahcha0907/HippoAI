"""Builds the Clinical Safety Set from safety_set_source.jsonl.

Source rows use short keys: c=category, q=question, a=correct answer, u=unsafe wrong
options, w=other wrong options, s=source. Every question has exactly 4 options.
Option order comes from a hash of the question, so builds are reproducible and answer
letters stay balanced. The split is stratified by category and uses that same hash.

Run from backend/:  python -m data.build_safety_set --out artifacts/safety_set
"""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from eval.scoring import Question

SOURCE = Path(__file__).with_name("safety_set_source.jsonl")
CATEGORIES = {"dosing", "contraindication", "triage", "diagnosis"}
LETTERS = "ABCD"


def _digest(text: str) -> str:
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def load_rows(path: Path = SOURCE) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.strip():
                rows.append(validate_row(json.loads(line), number))
    duplicates = [q for q, n in Counter(_digest(r["q"]) for r in rows).items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate questions in source")
    return rows


def validate_row(row: dict, number: int) -> dict:
    problem = None
    unsafe, other = row.get("u", []), row.get("w", [])
    options = [row.get("a"), *unsafe, *other]
    if set(row) - {"c", "q", "a", "u", "w", "s"}:
        problem = f"unknown keys {set(row) - {'c', 'q', 'a', 'u', 'w', 's'}}"
    elif row.get("c") not in CATEGORIES:
        problem = f"unknown category {row.get('c')!r}"
    elif not row.get("q") or not row.get("s"):
        problem = "missing question or source"
    elif len(options) != 4 or not all(options):
        problem = f"needs 4 options, has {len(options)}"
    elif len({o.lower() for o in options}) != 4:
        problem = "repeated option"
    if problem:
        raise ValueError(f"Source line {number}: {problem}")
    return row


def to_question(row: dict) -> Question:
    digest = _digest(row["q"])
    texts = [row["a"], *row.get("u", []), *row.get("w", [])]
    random.Random(digest).shuffle(texts)
    options = dict(zip(LETTERS, texts))
    letter_of = {text: letter for letter, text in options.items()}
    return Question(
        id=f"safety-{digest[:12]}",
        question=row["q"],
        options=options,
        answer=letter_of[row["a"]],
        category=row["c"],
        unsafe_options=sorted(letter_of[text] for text in row.get("u", [])),
        source=row["s"],
    )


def split(questions: list[Question]) -> tuple[list[Question], list[Question]]:
    """Half of each category goes to dev and half to sealed, ordered by hash."""
    by_category = defaultdict(list)
    for item in questions:
        by_category[item.category].append(item)
    dev, sealed = [], []
    for category in sorted(by_category):
        ordered = sorted(by_category[category], key=lambda q: q.id)
        half = len(ordered) // 2
        dev += ordered[:half]
        sealed += ordered[half:]
    return dev, sealed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    questions = [to_question(row) for row in load_rows()]
    dev, sealed = split(questions)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, items in (("dev", dev), ("sealed", sealed)):
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    print(f"{len(questions)} questions: {len(dev)} dev, {len(sealed)} sealed")
    print("Categories:", dict(Counter(q.category for q in questions)))
    print("Answer letters:", dict(sorted(Counter(q.answer for q in questions).items())))
    print("With an unsafe option:", sum(bool(q.unsafe_options) for q in questions))


if __name__ == "__main__":
    main()
