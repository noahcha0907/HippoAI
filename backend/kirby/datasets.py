"""Explicit adapter for the English, four-option MedQA JSONL schema.

Only this module reads answer/explanation fields. Prompt construction receives
an unlabeled QuestionInput instead of the original dataset dictionary.
"""

import hashlib
import json
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .records import digest, read_json, write_json

LETTERS = ("A", "B", "C", "D")
ADAPTER_VERSION = "medqa_four_options_v2"


@dataclass(frozen=True)
class QuestionInput:
    question: str
    options: tuple[str, str, str, str]


@dataclass(frozen=True)
class Question:
    id: str
    fingerprint: str
    split: str
    input: QuestionInput
    answer: str


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def load_medqa(path: Path, split: str) -> tuple[list[Question], dict]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("MedQA split must be train, dev, or test")
    questions = []
    duplicate_option_records = []
    seen = set()
    # Hash the exact bytes parsed, avoiding a second-read race.
    payload = path.read_bytes()
    source_hash = hashlib.sha256(payload).hexdigest()
    for line_number, line in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: empty JSONL record")
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("record must be an object")
            question = _text(row.get("question"), "question")
            options = row.get("options")
            if not isinstance(options, dict) or set(options) != set(LETTERS):
                raise ValueError("options must contain exactly A, B, C, D")
            choices = tuple(_text(options[key], f"option {key}") for key in LETTERS)
            # Distinct labels need not have distinct text in the source. Preserve
            # all choices and the supplied answer_idx; never filter or relabel.
            option_groups = {}
            for letter, choice in zip(LETTERS, choices):
                option_groups.setdefault(normalize(choice), []).append(letter)
            repeated = [group for group in option_groups.values() if len(group) > 1]
            answer = row.get("answer_idx")
            if answer not in LETTERS:
                raise ValueError("answer_idx must be A, B, C, or D")
            if "answer" in row and normalize(_text(row["answer"], "answer")) != normalize(options[answer]):
                raise ValueError("answer text disagrees with answer_idx")
            # Independent of split, labels, and option order, to catch leakage
            # even when two releases shuffle their answer choices.
            fingerprint = digest([normalize(question), sorted(map(normalize, choices))])
            if fingerprint in seen:
                raise ValueError("duplicate underlying question")
            seen.add(fingerprint)
            if repeated:
                duplicate_option_records.append({"line": line_number,
                                                 "question_id": f"medqa:{split}:{fingerprint}",
                                                 "option_groups": repeated})
            questions.append(Question(f"medqa:{split}:{fingerprint}", fingerprint, split,
                                      QuestionInput(question, choices), answer))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not questions:
        raise ValueError(f"Empty dataset: {path}")
    return questions, {"path": str(path.resolve()), "sha256": source_hash,
                       "split": split, "count": len(questions), "adapter": ADAPTER_VERSION,
                       "data_quality": {"duplicate_option_policy": "preserve_choices_score_answer_idx",
                                        "duplicate_option_records": duplicate_option_records}}


def assert_disjoint(*groups: list[Question]) -> None:
    seen = set()
    for group in groups:
        current = {question.fingerprint for question in group}
        if len(current) != len(group) or seen.intersection(current):
            raise ValueError("Underlying question overlap within or across splits")
        seen.update(current)


def choose(questions: list[Question], count: int, seed: int) -> list[Question]:
    if type(count) is not int or not 1 <= count <= len(questions):
        raise ValueError(f"Sample size must be between 1 and {len(questions)}")
    # Stable under source file reordering; do not sample based on model results.
    return random.Random(seed).sample(sorted(questions, key=lambda q: q.id), count)


def prepare(train: Path, dev: Path, test: Path, output: Path, *, seed: int = 17,
            diagnostic_count: int = 64, development_count: int = 256) -> dict:
    train_rows, train_source = load_medqa(train, "train")
    dev_rows, dev_source = load_medqa(dev, "dev")
    test_rows, test_source = load_medqa(test, "test")
    if len(dev_rows) != 1272 or len(test_rows) != 1273:
        raise ValueError("Expected official English four-option MedQA: dev=1272, test=1273. "
                         "Review the release; this command does not silently reinterpret a mirror.")
    assert_disjoint(train_rows, dev_rows, test_rows)
    diagnostic = choose(train_rows, diagnostic_count, seed)
    development = choose(dev_rows, development_count, seed)
    common = {"schema_version": 1, "benchmark": "medqa_english_four_options", "seed": seed,
              "sampling": "random_without_replacement_sorted_content_ids_v1",
              "deduplication": "NFKC_casefold_whitespace_question_sorted_options_v1",
              "near_duplicates_checked": False}
    manifest = {**common, "phase": "development", "sources": {"train": train_source, "dev": dev_source},
                "diagnostic_ids": [q.id for q in diagnostic], "question_ids": [q.id for q in development]}
    final = {**common, "phase": "final", "sources": {"test": test_source},
             "question_ids": [q.id for q in test_rows]}
    # Never overwrite an already frozen plan or silently resample it.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "development.json", manifest)
    write_json(output / "final.json", final)
    return {"development_sha256": digest(manifest), "final_sha256": digest(final),
            "diagnostic_count": len(diagnostic), "development_count": len(development),
            "final_count": len(test_rows),
            "duplicate_option_rows": {
                source["split"]: len(source["data_quality"]["duplicate_option_records"])
                for source in (train_source, dev_source, test_source)}}


def load_development(manifest_path: Path) -> tuple[list[Question], dict]:
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != 1 or manifest.get("phase") != "development":
        raise ValueError("Baseline runner accepts only a version-1 development manifest; final evaluation is separate")
    if manifest.get("benchmark") != "medqa_english_four_options":
        raise ValueError("Unsupported benchmark")
    source = manifest["sources"]["dev"]
    rows, actual = load_medqa(Path(source["path"]), "dev")
    if actual != source:
        raise ValueError("Dataset source changed after manifest preparation")
    ids = manifest["question_ids"]
    if not isinstance(ids, list) or not ids or not all(isinstance(i, str) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("Invalid or duplicate selected question IDs")
    by_id = {q.id: q for q in rows}
    if any(i not in by_id for i in ids):
        raise ValueError("Selected question is absent from the frozen source")
    return [by_id[i] for i in ids], manifest
