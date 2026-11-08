"""Numerical score checks against independent token-by-token likelihoods."""

import math
import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:
    torch = None

from kirby.answer_scores import (ANSWER_SCORE_PROTOCOL, answer_score_record, answer_scores_restored,
                                 compare_answer_scores, score_answer_labels)
from kirby.diagnostics import select_probe_plan


class SyntheticLabelTokenizer:
    """Explicit multi-token fixtures, never a substitute for the real tokenizer."""
    suffixes = {"A.": [4, 8], "B.": [5, 8], "C.": [6, 9, 8], "D.": [7, 8]}

    def apply_chat_template(self, messages, **kwargs):
        self.messages, self.template_options = messages, kwargs
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones((1, 3), dtype=torch.long)}

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text in "ABCD" and len(text) == 1:
            return self.suffixes[text + "."][:1]
        return self.suffixes[text]

    def decode(self, tokens, **kwargs):
        for text, values in self.suffixes.items():
            if values[:1] == tokens:
                return text[0]
        return next((text for text, values in self.suffixes.items() if values == tokens), "INVALID")


@unittest.skipIf(torch is None, "Install GPU extra for numerical score tests")
class AnswerScoreTests(unittest.TestCase):
    def setUp(self):
        class CausalFixture(torch.nn.Module):
            config = SimpleNamespace(model_type="phi", vocab_size=12)
            device = "cpu"
            calls = 0
            def forward(self, input_ids, attention_mask, use_cache, return_dict):
                self.calls += 1
                assert not use_cache and return_dict
                assert torch.equal(attention_mask, torch.ones_like(input_ids))
                # The next-token distribution depends only on the current token.
                vocabulary = torch.arange(12).float()
                logits = -(vocabulary[None, None, :] - input_ids[:, :, None]).square() / 10
                return SimpleNamespace(logits=logits)
        self.predictor = SimpleNamespace(model=CausalFixture().eval(), tokenizer=SyntheticLabelTokenizer(), context_length=64)
        self.forwards = 0

    def score(self):
        def count():
            self.forwards += 1
        return score_answer_labels(self.predictor, "synthetic prompt", before_forward=count)

    def test_multitoken_scores_match_independent_autoregressive_calculation(self):
        result = self.score()
        for letter, value in result["log_scores"].items():
            previous = 3
            expected = 0.0
            for token in self.predictor.tokenizer.suffixes[letter + "."]:
                raw = [-((v - previous) ** 2) / 10 for v in range(12)]
                expected += raw[token] - math.log(sum(math.exp(x) for x in raw))
                previous = token
            self.assertAlmostEqual(value, expected, places=6)
        self.assertEqual(self.forwards, 4)
        self.assertEqual(self.predictor.model.calls, 4)
        self.assertEqual(result["protocol"], ANSWER_SCORE_PROTOCOL)
        self.assertAlmostEqual(sum(result["conditional_probabilities"].values()), 1.0)
        self.assertEqual(self.predictor.tokenizer.messages, [{"role": "user", "content": "synthetic prompt"}])
        self.assertTrue(self.predictor.tokenizer.template_options["add_generation_prompt"])
        self.assertTrue(torch.is_grad_enabled())  # Context was restored.

    def test_gold_margin_and_small_changes_without_label_flip(self):
        question = SimpleNamespace(split="train", id="train-1", answer="B")
        scores = self.score()
        before = answer_score_record(question, scores)
        self.assertAlmostEqual(before["expected_margin"], scores["log_scores"]["B"] - max(
            scores["log_scores"][letter] for letter in "ACD"))
        # Constant log-score shifts preserve ranking but remain measurable.
        shifted = {**scores, "log_scores": {key: value + 0.01 for key, value in scores["log_scores"].items()}}
        after = answer_score_record(question, shifted)
        comparison = compare_answer_scores([before], [after])
        self.assertEqual(comparison["preferred_label_changes"], 0)
        self.assertAlmostEqual(comparison["max_absolute_log_score_delta"], 0.01)
        self.assertAlmostEqual(comparison["mean_expected_log_score_delta"], 0.01)
        self.assertFalse(answer_scores_restored([before], [after]))
        self.assertTrue(answer_scores_restored([before], [before]))

    def test_context_and_tokenization_errors_fail_before_any_forward(self):
        self.predictor.context_length = 5  # C. takes three tokens plus three prompt tokens.
        with self.assertRaisesRegex(ValueError, "context"):
            self.score()
        self.assertEqual(self.forwards, 0)
        self.predictor.context_length = 64
        self.predictor.tokenizer.decode = lambda *args, **kwargs: "?"
        with self.assertRaisesRegex(ValueError, "round-trip"):
            self.score()
        self.assertEqual(self.forwards, 0)

    def test_nonfinite_and_training_mode_fail_closed(self):
        self.predictor.model.train()
        with self.assertRaisesRegex(ValueError, "evaluated"):
            self.score()
        self.predictor.model.eval()
        self.predictor.model.forward = lambda **kwargs: SimpleNamespace(logits=torch.full((1, 5, 12), float("nan")))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.score()

    def test_development_rows_and_misaligned_comparisons_are_rejected(self):
        scores = self.score()
        with self.assertRaisesRegex(ValueError, "training split"):
            answer_score_record(SimpleNamespace(split="dev", id="dev-1", answer="A"), scores)
        row = answer_score_record(SimpleNamespace(split="train", id="train-1", answer="A"), scores)
        with self.assertRaisesRegex(ValueError, "order or protocol"):
            compare_answer_scores([row], [{**row, "question_id": "train-2"}])
        with self.assertRaisesRegex(ValueError, "complete matching"):
            compare_answer_scores([row], [])


class ProbePlanTests(unittest.TestCase):
    def test_fixed_bidirectional_plan_ranks_contribution_proxy_and_ties(self):
        activations = [
            {"layer": 1, "columns": [{"column": 9, "mean_absolute_activation": 4, "weight_column_l2": 2}]},
            {"layer": 0, "columns": [{"column": 3, "mean_absolute_activation": 2, "weight_column_l2": 4},
                                     {"column": 2, "mean_absolute_activation": 5, "weight_column_l2": 1}]},
        ]
        result = select_probe_plan(activations)
        self.assertEqual([(p["layer"], p["columns"], p["scale"]) for p in result],
                         [(0, [3], 0.98), (0, [3], 1.02), (1, [9], 0.98), (1, [9], 1.02)])
        self.assertEqual(select_probe_plan(list(reversed(activations))), result)
        with self.assertRaisesRegex(ValueError, "Not enough"):
            select_probe_plan([])


if __name__ == "__main__":
    unittest.main()
