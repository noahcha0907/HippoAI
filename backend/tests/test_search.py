"""Training selection, history identity, held-out confirmation, and rollback."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kirby.answer_scores import NEXT_LABEL_PROTOCOL, next_label_scores, score_next_labels
from kirby.astra import AstraClient, validate_proposal
from kirby.records import digest, read_json, write_json
from kirby.search import SearchConfig, operation_key, rank_probes, read_attempt_history, screening_plan
from kirby.search_runner import run_astra_search
from kirby.patches import weights_hash
from test_astra import proposal, response
import test_astra_runner as runner_fixtures

torch = runner_fixtures.torch


class SearchPolicyTests(unittest.TestCase):
    def test_default_limits_and_invalid_partitions(self):
        settings = SearchConfig()
        self.assertEqual((settings.development_count, settings.diagnostic_count, settings.screening_count), (172, 64, 8))
        for values in ({"development_count": 173}, {"diagnostic_count": 8, "screening_count": 8}, {"layer_count": 9}, {"confirm_count": 3}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                SearchConfig(**values)

    def test_layers_are_spread_and_history_skips_entire_repeated_target(self):
        activations = [{"layer": i, "columns": [{"column": j, "mean_absolute_activation": 3 - j, "weight_column_l2": 1}
                                                 for j in (0, 1)]} for i in range(32)]
        checkpoint = "a" * 64
        history = {"accepted": []}
        planned = screening_plan(activations, checkpoint, history, SearchConfig())
        self.assertEqual(len(planned), 16)
        self.assertEqual({row["operation"]["layer"] for row in planned}, {0, 4, 8, 13, 17, 22, 26, 31})
        history["accepted"] = [{"operation_key": planned[0]["operation_key"]}]
        updated = screening_plan(activations, checkpoint, history, SearchConfig())
        self.assertEqual(updated[0]["operation"]["columns"], [1])
        self.assertNotIn(planned[0]["operation_key"], {row["operation_key"] for row in updated})

    def test_ranking_uses_measured_gain_and_rejects_margin_only_or_accuracy_regression(self):
        def row(key, gain, margin, correct=4):
            return {"operation_key": key, "state": "completed", "scores_restored": True,
                    "comparison": {"mean_expected_log_score_delta": gain, "mean_expected_margin_delta": margin,
                                   "original": {"preferred_correct": 4}, "changed": {"preferred_correct": correct}}}
        ranked = rank_probes([row("a", 0.01, 0.03), row("b", 0.02, 0.02), row("c", -0.01, 0.1), row("d", 1, 1, 3)])
        self.assertEqual([r["operation_key"] for r in ranked], ["b", "a"])

    def test_api_proposal_must_match_ranked_operation_not_merely_cite_it(self):
        from test_astra import packet
        evidence = packet()
        evidence.update(schema_version=3, training_sensitivity=[{"evidence_id": "confirm-1"}],
                        selected_candidate={"evidence_id": "confirm-1", "operation": proposal()["operation"]})
        evidence["evidence_ids"].append("confirm-1")
        value = proposal(evidence)
        value["evidence_ids"].append("confirm-1")
        validate_proposal(json.dumps(value), evidence, "test")
        value["operation"]["scale"] = 0.96
        with self.assertRaisesRegex(ValueError, "highest-ranked"):
            validate_proposal(json.dumps(value), evidence, "test")

    def test_cli_defaults_to_172_and_64_training_questions(self):
        from kirby.cli import main
        with patch("kirby.cli.run_astra_search") as run:
            run.return_value = {"directory": "synthetic", "report": {"state": "completed", "outcome": "no candidate"}}
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["astra-search", "--manifest", "development.json", "--rollback-report", "report.json",
                                       "--revision", "a" * 40]), 0)
            self.assertEqual(run.call_args.args[6].development_count, 172)
            self.assertEqual(run.call_args.args[6].diagnostic_count, 64)


@unittest.skipIf(torch is None, "Install GPU extra for real CPU search tests")
class SearchRunnerTests(unittest.TestCase):
    setUp = runner_fixtures.AstraRunnerTests.setUp

    def run_search(self, mode="valid"):
        predictor = self.predictor
        def generate_scored(prompt):
            generated = predictor.generate(prompt)
            values = score_next_labels(predictor, prompt, before_forward=lambda: None)
            return generated, values
        predictor.generate_with_answer_scores = generate_scored
        client = Mock()
        def request(**kwargs):
            packet = json.loads(kwargs["input"][0]["content"])
            self.last_packet = packet
            value = proposal(packet)
            value["operation"] = packet["selected_candidate"]["operation"]
            value["evidence_ids"] = [packet["selected_candidate"]["evidence_id"]]
            if mode == "abstain":
                value.update(decision="abstain", operation=None)
            elif mode == "invalid":
                value["operation"]["scale"] = 1
            elif mode == "provider-error":
                raise RuntimeError("sk-secret-never-log")
            return response(json.dumps(value))
        client.responses.create.side_effect = request
        config = SearchConfig(development_count=3, diagnostic_count=3, screening_count=1, layer_count=2,
                              confirm_count=2, calibration_questions=1)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_astra_search(self.manifest, self.rollback_report, predictor, {}, AstraClient(client=client),
                                      self.root / "runs", config)
        return result, client

    def test_full_search_scores_development_and_restores_weights(self):
        result, client = self.run_search()
        report = result["report"]
        self.assertEqual(report["state"], "completed", report.get("error"))
        self.assertEqual(client.responses.create.call_count, 1, report["search"])
        self.assertEqual(report["comparison"]["original"]["total"], 3)
        self.assertEqual(report["development_score_comparison"]["original"]["questions"], 3)
        self.assertTrue(report["weights_restored"] and report["outputs_restored"] and report["answer_scores_restored"])
        self.assertEqual(weights_hash(self.predictor.model), self.before_hash)
        saved = Path(result["directory"])
        config = read_json(saved / "configuration.json")
        self.assertFalse(set(config["screening_ids"]) & set(config["confirmation_ids"]))
        self.assertEqual(len(config["development_ids"]), 3)
        self.assertEqual(config["answer_score_protocol"], NEXT_LABEL_PROTOCOL)
        self.assertEqual(len([p for p in report["training_sensitivity"] if p["phase"] == "random_control"]), 1)
        self.assertNotIn("test.jsonl", json.dumps(self.last_packet))
        self.assertEqual(self.last_packet["selected_candidate"]["operation"]["layer"], report["patch"]["layer"])
        # A second run's history includes this evaluated operation, not merely its probes.
        history = read_attempt_history(self.root / "runs", report["model"], config["manifest"], self.before_hash)
        self.assertEqual(len(history["accepted"]), 1)
        self.assertEqual(history["accepted"][0]["development_questions"], 3)
        config["limits"]["seed"] = 99
        write_json(saved / "configuration.json", config)
        corrupted = read_attempt_history(self.root / "runs", report["model"], config["manifest"], self.before_hash)
        self.assertEqual(corrupted["accepted"], [])
        self.assertEqual(len(corrupted["ignored"]), 1)

    def test_no_eligible_candidate_completes_without_provider_spend(self):
        with patch("kirby.search_runner.rank_probes", return_value=[]):
            result, client = self.run_search()
        report = result["report"]
        self.assertEqual(report["state"], "completed")
        self.assertIn("no eligible candidate", report["outcome"])
        self.assertEqual(report["usage"]["api_calls"], 0)
        self.assertIsNone(report["comparison"])
        self.assertTrue(report["weights_restored"])
        client.responses.create.assert_not_called()

    def test_provider_failure_abstention_and_invalid_operation_never_leave_edits(self):
        for mode in ("abstain", "invalid", "provider-error"):
            with self.subTest(mode=mode):
                result, client = self.run_search(mode)
                report = result["report"]
                self.assertEqual(report["state"], "completed" if mode == "abstain" else "incomplete", report.get("error"))
                self.assertEqual(client.responses.create.call_count, 1)
                self.assertIsNone(report["patch"])
                self.assertTrue(report["weights_restored"])
                self.assertNotIn("sk-secret-never-log", json.dumps(report))

    def test_interrupted_probe_restores_before_any_provider_call(self):
        calls = 0
        def interrupted(*args, **kwargs):
            nonlocal calls
            calls += 1
            # generation scoring uses the local original function, not this patched symbol.
            if calls == 5:  # baseline 3 + restoration repeat 1, first active probe
                raise KeyboardInterrupt()
            return score_next_labels(*args, **kwargs)
        with patch("kirby.search_runner.score_next_labels", side_effect=interrupted):
            result, client = self.run_search()
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertTrue(result["report"]["weights_restored"])
        self.assertTrue(result["report"]["training_sensitivity"][0]["receipt"]["rollback_verified"])
        client.responses.create.assert_not_called()

    def test_failed_probe_record_write_restores_weights(self):
        failed = False
        def injected(path, value):
            nonlocal failed
            if path.name == "screen-1.json" and value.get("receipt") and not failed:
                failed = True
                raise OSError("injected")
            write_json(path, value)
        with patch("kirby.search_runner.write_json", side_effect=injected):
            result, client = self.run_search()
        self.assertTrue(failed and result["report"]["weights_restored"])
        self.assertEqual(result["report"]["state"], "incomplete")
        client.responses.create.assert_not_called()

    def test_score_drift_after_restore_prevents_api(self):
        calls = 0
        def drift(*args, **kwargs):
            nonlocal calls
            calls += 1
            values = score_next_labels(*args, **kwargs)
            if calls == 6:  # first restored question after the first probe
                values["log_scores"]["A"] += 0.1
            return values
        with patch("kirby.search_runner.score_next_labels", side_effect=drift):
            result, client = self.run_search()
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertIn("restoration failed", result["report"]["error"]["message"])
        self.assertTrue(result["report"]["weights_restored"])
        client.responses.create.assert_not_called()

    def test_actual_generation_capture_preserves_output_and_first_token_scores(self):
        from transformers import GenerationConfig, StoppingCriteriaList
        from kirby.jsl import JSLMedPhi, JSLModelConfig
        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})
        adapter = JSLMedPhi(JSLModelConfig("a" * 40, max_new_tokens=2))
        adapter.model = self.predictor.model
        adapter.tokenizer = self.predictor.tokenizer
        original_template = adapter.tokenizer.apply_chat_template
        adapter.tokenizer.apply_chat_template = lambda *a, **k: Batch(original_template(*a, **k))
        adapter.context_length = 64
        adapter.torch = torch
        adapter.eos_ids = [0]
        adapter.stopping_criteria_type = StoppingCriteriaList
        adapter.generation_config = GenerationConfig(do_sample=False, max_new_tokens=2, eos_token_id=0, pad_token_id=0, use_cache=True)
        normal = adapter.generate("Synthetic prompt")
        captured, scores = adapter.generate_with_answer_scores("Synthetic prompt")
        self.assertEqual(normal, captured)
        independently = score_next_labels(adapter, "Synthetic prompt", before_forward=lambda: None)
        for letter in "ABCD":
            self.assertAlmostEqual(scores["log_scores"][letter], independently["log_scores"][letter], places=5)
        self.assertEqual(scores["protocol"], NEXT_LABEL_PROTOCOL)
        self.assertAlmostEqual(sum(scores["conditional_probabilities"].values()), 1)


if __name__ == "__main__":
    unittest.main()
