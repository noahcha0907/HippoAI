"""The evaluator owns prompts, labels, and score semantics."""

from dataclasses import asdict, dataclass

from .datasets import LETTERS, Question, QuestionInput

PROFILE = "medqa_text_v2"
PARSER = "uppercase_letter_optional_period_v2"
PROMPT = ("Select the best answer to this medical examination question.\n"
          "Return exactly one uppercase letter: A, B, C, or D.\n\n"
          "Question:\n{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}")


def format_prompt(question: QuestionInput) -> str:
    return PROMPT.format(question=question.question, **dict(zip("abcd", question.options)))


@dataclass(frozen=True)
class Generation:
    text: str
    generated_tokens: int
    truncated: bool


def score(question: Question, generation: Generation, latency_seconds: float) -> dict:
    answer = generation.text.strip().removesuffix(".")
    valid = answer in LETTERS
    return {"question_id": question.id, "fingerprint": question.fingerprint,
            "expected": question.answer, "predicted": answer if valid else None,
            "valid": valid, "correct": valid and not generation.truncated and answer == question.answer,
            "latency_seconds": latency_seconds, **asdict(generation)}


def summarize(predictions: list[dict]) -> dict:
    count = len(predictions)
    correct = sum(p["correct"] for p in predictions)
    invalid = sum(not p["valid"] for p in predictions)
    truncated = sum(p["truncated"] for p in predictions)
    return {"correct": correct, "total": count,
            "accuracy": correct / count if count else None,
            "invalid_count": invalid, "invalid_rate": invalid / count if count else None,
            "truncated_count": truncated, "truncation_rate": truncated / count if count else None,
            "generated_tokens": sum(p["generated_tokens"] for p in predictions),
            "mean_latency_seconds": sum(p["latency_seconds"] for p in predictions) / count if count else None}
