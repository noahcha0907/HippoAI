import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kirby.evaluation import Generation
from kirby.records import digest
from kirby.runner import RunConfig, run_baseline
from test_evaluation import make_manifest


class ScriptedPredictor:
    """Synthetic test double only. Never selected by a production CLI flag."""
    def __init__(self, outputs=None, fail_at=None):
        self.outputs = outputs or ["B"] * 20
        self.calls = 0
        self.fail_at = fail_at

    def load(self):
        return {"kind": "synthetic_test_double", "checkpoint_id": "not-a-medgemma-checkpoint"}

    def generate(self, prompt):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic worker failure")
        return Generation(self.outputs[self.calls - 1], 2, False)

    def resource_usage(self):
        return {"peak_gpu_allocated_bytes": None}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, _ = make_manifest(self.root)

    def run_worker(self, predictor, **kwargs):
        with contextlib.redirect_stderr(io.StringIO()):
            return run_baseline(self.manifest, predictor, {"kind": "synthetic_test_double"},
                                self.root / "runs", RunConfig(calibration_questions=1), **kwargs)

    def test_complete_run_writes_inspectable_artifacts_and_usage(self):
        result = self.run_worker(ScriptedPredictor())
        report = result["report"]
        directory = Path(result["directory"])
        self.assertEqual(report["state"], "completed")
        self.assertEqual(report["outcome"], "Baseline evaluated")
        self.assertEqual(report["development"]["total"], 4)
        self.assertEqual(report["usage"]["forward_calls"], 6)
        self.assertIsNone(report["final"])
        self.assertIsNone(report["comparison"])
        self.assertEqual(len((directory / "predictions.jsonl").read_text().splitlines()), 4)
        frozen = json.loads((directory / "configuration.json").read_text())
        self.assertEqual(report["configuration_sha256"], digest(frozen))
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual([e["sequence"] for e in events], list(range(1, len(events) + 1)))
        self.assertEqual(events[-1]["state"], "completed")
        self.assertIn("No candidate was edited", (directory / "report.md").read_text())

    def test_calibration_variability_stops_before_baseline(self):
        result = self.run_worker(ScriptedPredictor(["A", "B"]))
        self.assertEqual(result["report"]["outcome"], "Run incomplete")
        self.assertEqual(result["report"]["development"]["total"], 0)
        self.assertFalse(result["report"]["calibration"]["stable"])

    def test_invalid_calibration_stops_before_baseline(self):
        result = self.run_worker(ScriptedPredictor(["The answer is B"] * 2))
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertEqual(result["report"]["usage"]["forward_calls"], 2)
        self.assertFalse(result["report"]["calibration"]["protocol_passed"])
        self.assertIn("1/1 invalid", result["report"]["error"]["message"])
        self.assertIn("0/1 truncated", result["report"]["error"]["message"])
        self.assertIn("calibration.jsonl", result["report"]["error"]["message"])

    def test_truncated_letter_stops_calibration_and_retains_raw_evidence(self):
        class TruncatedPredictor(ScriptedPredictor):
            def generate(self, prompt):
                return Generation("B", 8, True)

        result = self.run_worker(TruncatedPredictor())
        report = result["report"]
        self.assertEqual(report["state"], "incomplete")
        self.assertEqual(report["development"]["total"], 0)
        self.assertTrue(report["calibration"]["stable"])
        self.assertFalse(report["calibration"]["protocol_passed"])
        self.assertIn("0/1 invalid", report["error"]["message"])
        self.assertIn("1/1 truncated", report["error"]["message"])
        records = [json.loads(line) for line in
                   (Path(result["directory"]) / "calibration.jsonl").read_text().splitlines()]
        self.assertEqual([row["text"] for row in records], ["B", "B"])
        self.assertTrue(all(row["truncated"] and not row["correct"] for row in records))

    def test_failure_keeps_partial_predictions_without_success_label(self):
        result = self.run_worker(ScriptedPredictor(fail_at=4))
        report = result["report"]
        self.assertEqual(report["outcome"], "Run incomplete")
        self.assertEqual(report["development"]["total"], 1)
        self.assertFalse(report["development"]["complete"])
        self.assertEqual(report["usage"]["forward_calls"], 4)
        self.assertTrue((Path(result["directory"]) / "report.md").exists())

    def test_load_failure_still_produces_report(self):
        class BrokenLoader(ScriptedPredictor):
            def load(self):
                raise RuntimeError("Unavailable checkpoint")
        result = self.run_worker(BrokenLoader())
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertIsNone(result["report"]["model"])

    def test_cancellation_and_budget_do_not_look_completed(self):
        predictor = ScriptedPredictor()
        result = self.run_worker(predictor, cancelled=lambda: predictor.calls >= 3)
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertEqual(result["report"]["development"]["total"], 1)
        # Coarse platform clocks can report the same instant for a tiny run.
        # Drive the deadline explicitly instead of depending on elapsed CPU time.
        with patch("kirby.runner.time.monotonic", side_effect=[0.0, 1.0, 2.0]):
            result = run_baseline(self.manifest, ScriptedPredictor(), {}, self.root / "runs",
                                  RunConfig(max_seconds=0.5, calibration_questions=1))
        self.assertEqual(result["report"]["state"], "incomplete")
        self.assertEqual(result["report"]["usage"]["forward_calls"], 0)

    def test_repeated_runs_never_overwrite_prior_artifacts(self):
        first = self.run_worker(ScriptedPredictor())
        second = self.run_worker(ScriptedPredictor())
        self.assertNotEqual(first["directory"], second["directory"])
        self.assertTrue((Path(first["directory"]) / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
