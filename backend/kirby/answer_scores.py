"""Teacher-forced label likelihoods for training diagnostics, never benchmark scoring."""

import math

from .datasets import LETTERS

ANSWER_SCORE_PROTOCOL = "teacher_forced_label_period_sum_logprob_v1"
NEXT_LABEL_PROTOCOL = "next_token_single_letter_logprob_v1"
SCORE_TOLERANCE = 1e-6


def next_label_scores(tokenizer, logits):
    """Diagnostic scores from the unmodified first-generation-token logits."""
    import torch
    tokens = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS}
    for letter, ids in tokens.items():
        if (len(ids) != 1 or type(ids[0]) is not int or not 0 <= ids[0] < logits.numel()
                or tokenizer.decode(ids, skip_special_tokens=False) != letter):
            raise ValueError("Next-label scoring requires distinct single-token A/B/C/D labels")
    if len({ids[0] for ids in tokens.values()}) != 4 or logits.ndim != 1:
        raise ValueError("Invalid next-label token IDs or logits")
    logits = logits.float()
    if not torch.isfinite(logits).all().item():
        raise ValueError("Nonfinite next-label logits")
    values = torch.log_softmax(logits, dim=-1)
    log_scores = {letter: float(values[ids[0]].item()) for letter, ids in tokens.items()}
    maximum = max(log_scores.values())
    denominator = sum(math.exp(value - maximum) for value in log_scores.values())
    probabilities = {letter: math.exp(value - maximum) / denominator for letter, value in log_scores.items()}
    ranked = sorted(LETTERS, key=lambda letter: (-log_scores[letter], letter))
    return {"protocol": NEXT_LABEL_PROTOCOL, "label_token_ids": tokens, "log_scores": log_scores,
            "conditional_probabilities": probabilities, "preferred_label": ranked[0],
            "top_margin": log_scores[ranked[0]] - log_scores[ranked[1]]}


def score_next_labels(predictor, prompt, *, before_forward):
    """One batch-one forward per training question; no generated answer is forced."""
    import torch
    if predictor.model.training or predictor.model.config.model_type != "phi":
        raise ValueError("Next-label scoring requires an evaluated Phi model")
    inputs = predictor.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt")
    if set(inputs) - {"input_ids", "attention_mask"}:
        raise ValueError("Unsupported answer-score inputs")
    if not 0 < inputs["input_ids"].shape[-1] < predictor.context_length:
        raise ValueError("Next-label scoring exceeds context; no input was truncated")
    inputs = {key: value.to(predictor.model.device) for key, value in inputs.items()}
    before_forward()
    with torch.inference_mode():
        output = predictor.model(**inputs, use_cache=False, return_dict=True)
        return next_label_scores(predictor.tokenizer, output.logits[0, -1])


def score_answer_labels(predictor, prompt, *, before_forward):
    """Score explicit A./B./C./D. token suffixes after the generation prompt.

    No EOS, leading space, length normalization, sampling, or forced generation.
    Four batch-one, cache-free forwards keep memory bounded on the T4. The
    normalized numbers are conditional on these four strings, not confidence
    in medical correctness or a replacement for unconstrained generation.
    """
    import torch

    model, tokenizer = predictor.model, predictor.tokenizer
    if model.config.model_type != "phi" or model.training:
        raise ValueError("Answer scoring requires an evaluated Phi model")
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True,
        tokenize=True, return_dict=True, return_tensors="pt",
    )
    if set(inputs) - {"input_ids", "attention_mask"}:
        raise ValueError("Unsupported inputs for the answer-score protocol")
    prompt_ids = inputs["input_ids"].to(model.device)
    mask = inputs.get("attention_mask", torch.ones_like(prompt_ids)).to(model.device)
    if (prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1 or prompt_ids.shape[1] == 0
            or mask.shape != prompt_ids.shape or not (mask == 1).all().item()):
        raise ValueError("Answer scoring requires one nonempty, unpadded prompt")
    suffixes = {letter: tokenizer.encode(letter + ".", add_special_tokens=False) for letter in LETTERS}
    for letter, tokens in suffixes.items():
        if (not 1 <= len(tokens) <= 8
                or any(type(t) is not int or not 0 <= t < model.config.vocab_size for t in tokens)
                or tokenizer.decode(tokens, skip_special_tokens=False) != letter + "."):
            raise ValueError("Label tokenization does not round-trip under the score protocol")
        if prompt_ids.shape[1] + len(tokens) > predictor.context_length:
            raise ValueError("Answer scoring exceeds the model context; no input was truncated")

    log_scores = {}
    with torch.inference_mode():
        for letter, tokens in suffixes.items():
            target = torch.tensor([tokens], dtype=torch.long, device=model.device)
            sequence = torch.cat((prompt_ids, target[:, :-1]), dim=1)
            attention = torch.cat((mask, torch.ones_like(target[:, :-1])), dim=1)
            before_forward()
            output = model(input_ids=sequence, attention_mask=attention, use_cache=False, return_dict=True)
            # Position prompt_length-1 predicts the first suffix token. Only
            # suffix positions enter the score; prompt likelihood is excluded.
            logits = output.logits[0, -len(tokens):, :].float()
            if not torch.isfinite(logits).all().item():
                raise ValueError("Nonfinite answer logits")
            chosen = torch.log_softmax(logits, dim=-1).gather(1, target[0, :, None])
            log_scores[letter] = float(chosen.double().sum().item())
            del output, logits, chosen
    maximum = max(log_scores.values())
    denominator = sum(math.exp(value - maximum) for value in log_scores.values())
    probabilities = {letter: math.exp(value - maximum) / denominator for letter, value in log_scores.items()}
    ranked = sorted(LETTERS, key=lambda letter: (-log_scores[letter], letter))
    return {"protocol": ANSWER_SCORE_PROTOCOL, "label_token_ids": suffixes,
            "log_scores": log_scores, "conditional_probabilities": probabilities,
            "preferred_label": ranked[0], "top_margin": log_scores[ranked[0]] - log_scores[ranked[1]]}


def answer_score_record(question, scores, *, phase="train"):
    if phase not in {"train", "dev"} or question.split != phase:
        raise ValueError("Answer-score diagnostics require the selected training split or explicit dev phase")
    expected = question.answer
    values = scores["log_scores"]
    if (scores["protocol"] not in {ANSWER_SCORE_PROTOCOL, NEXT_LABEL_PROTOCOL} or set(values) != set(LETTERS)
            or any(not math.isfinite(value) for value in values.values())):
        raise ValueError("Invalid answer scores")
    return {"question_id": question.id, "expected": expected, **scores,
            "expected_log_score": values[expected],
            "expected_probability": scores["conditional_probabilities"][expected],
            "expected_margin": values[expected] - max(values[letter] for letter in LETTERS if letter != expected),
            "preferred_correct": scores["preferred_label"] == expected}


def summarize_answer_scores(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty answer-score cohort")
    return {"questions": len(rows), "preferred_correct": sum(row["preferred_correct"] for row in rows),
            "mean_expected_log_score": sum(row["expected_log_score"] for row in rows) / len(rows),
            "mean_expected_probability": sum(row["expected_probability"] for row in rows) / len(rows),
            "mean_expected_margin": sum(row["expected_margin"] for row in rows) / len(rows)}


def compare_answer_scores(original, changed):
    if not original or len(original) != len(changed):
        raise ValueError("Answer-score comparison needs a complete matching cohort")
    effects = []
    largest = 0.0
    for before, after in zip(original, changed):
        if any(before[key] != after[key] for key in ("question_id", "expected", "protocol", "label_token_ids")):
            raise ValueError("Answer-score question order or protocol differs")
        differences = [abs(after["log_scores"][letter] - before["log_scores"][letter]) for letter in LETTERS]
        if not all(math.isfinite(value) for value in differences):
            raise ValueError("Nonfinite answer-score change")
        largest = max(largest, *differences)
        effects.append({"question_id": before["question_id"],
                        "expected_log_score_delta": after["expected_log_score"] - before["expected_log_score"],
                        "expected_probability_delta": after["expected_probability"] - before["expected_probability"],
                        "expected_margin_delta": after["expected_margin"] - before["expected_margin"],
                        "preferred_before": before["preferred_label"], "preferred_after": after["preferred_label"]})
    margins = [row["expected_margin_delta"] for row in effects]
    return {"original": summarize_answer_scores(original), "changed": summarize_answer_scores(changed),
            "mean_expected_margin_delta": sum(margins) / len(margins),
            "mean_expected_log_score_delta": sum(row["expected_log_score_delta"] for row in effects) / len(effects),
            "preferred_label_changes": sum(row["preferred_before"] != row["preferred_after"] for row in effects),
            "margin_improved": sum(value > SCORE_TOLERANCE for value in margins),
            "margin_worsened": sum(value < -SCORE_TOLERANCE for value in margins),
            "max_absolute_log_score_delta": largest, "per_question": effects}


def answer_scores_restored(original, restored):
    comparison = compare_answer_scores(original, restored)
    return (comparison["max_absolute_log_score_delta"] <= SCORE_TOLERANCE
            and comparison["preferred_label_changes"] == 0)
