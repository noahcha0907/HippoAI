"""Fast multiple-choice scoring from answer-option logits, with thinking off.

Each question needs one forward pass and no generation.
"""
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    options: dict[str, str]
    answer: str
    category: str = ""
    unsafe_options: list[str] = field(default_factory=list)
    source: str = ""


def render_prompt(item: Question) -> str:
    lines = [item.question, ""]
    lines += [f"{letter}. {text}" for letter, text in sorted(item.options.items())]
    lines += ["", "Answer with the letter of the best option only."]
    return "\n".join(lines)


def build_inputs(tokenizer, items: list[Question]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": render_prompt(item)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for item in items
    ]


def option_token_ids(tokenizer, letters: list[str]) -> dict[str, int]:
    """Map each letter to its single token, failing loudly if it isn't one token."""
    token_ids = {}
    for letter in letters:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"Option {letter!r} isn't a single token: {encoded}")
        token_ids[letter] = encoded[0]
    return token_ids


@torch.inference_mode()
def score_questions(model, tokenizer, items: list[Question], batch_size: int = 16) -> list[dict]:
    """Return each question's predicted letter and whether it was correct."""
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # So the last position is always the answer position.
    device = next(model.parameters()).device
    results = []
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            encoded = tokenizer(build_inputs(tokenizer, batch), return_tensors="pt", padding=True).to(device)
            last_logits = model(**encoded).logits[:, -1, :]
            for item, logits in zip(batch, last_logits):
                letters = sorted(item.options)
                ids = option_token_ids(tokenizer, letters)
                scores = torch.stack([logits[ids[letter]] for letter in letters])
                prediction = letters[int(scores.argmax())]
                results.append({"id": item.id, "prediction": prediction, "correct": prediction == item.answer})
    finally:
        tokenizer.padding_side = original_side
    return results
