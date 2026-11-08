import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kirby.cli import main
from kirby.records import digest, write_json
from kirby.reproducibility import compare_baselines
from kirby.runner import RunConfig, run_baseline
from test_evaluation import make_manifest
from test_runner import ScriptedPredictor


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest, _ = make_manifest(self.root)

    def run_baseline(self, output="B"):
        class Predictor(ScriptedPredictor):
            def load(self):
                files = {"synthetic": "not-real-weights"}
                return {"files": files, "checkpoint_id": digest(files), "generation_config": {},
                        "environment": {"gpu": "synthetic"}, "rendered_template_probe": "template"}
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_baseline(self.manifest, Predictor([output] * 6), {}, self.root / "runs",
                                  RunConfig(calibration_questions=1))
        return Path(result["directory"])

    def test_identical_predictions_reproduce_despite_latency(self):
        first, second = self.run_baseline(), self.run_baseline()
        report = compare_baselines(first, second)
        self.assertTrue(report["reproduced"])
        self.assertEqual(report["questions"], 4)

    def test_same_accuracy_does_not_hide_changed_answers(self):
        first, second = self.run_baseline("A"), self.run_baseline("C")
        report = compare_baselines(first, second)
        self.assertTrue(report["comparable"])
        self.assertFalse(report["reproduced"])
        self.assertEqual(report["first_accuracy"], report["second_accuracy"])
        self.assertEqual(len(report["prediction_differences"]), 4)

    def test_different_protocol_is_not_compared(self):
        first, second = self.run_baseline(), self.run_baseline()
        config_path = second / "configuration.json"
        config = json.loads(config_path.read_text())
        config["parser"] = "different-parser"
        write_json(config_path, config)
        report_path = second / "report.json"
        saved = json.loads(report_path.read_text())
        saved["configuration_sha256"] = digest(config)
        write_json(report_path, saved)
        report = compare_baselines(first, second)
        self.assertFalse(report["comparable"])
        self.assertIn("configuration.parser", report["configuration_mismatches"])

    def test_missing_predictions_and_same_run_are_rejected(self):
        first, second = self.run_baseline(), self.run_baseline()
        with self.assertRaisesRegex(ValueError, "distinct"):
            compare_baselines(first, first)
        (second / "predictions.jsonl").write_text("")
        with self.assertRaisesRegex(ValueError, "question order"):
            compare_baselines(first, second)

    def test_modified_configuration_hash_is_rejected(self):
        first, second = self.run_baseline(), self.run_baseline()
        path = second / "configuration.json"
        config = json.loads(path.read_text())
        config["parser"] = "changed-without-rehash"
        write_json(path, config)
        with self.assertRaisesRegex(ValueError, "digest"):
            compare_baselines(first, second)

    def test_comparison_cli_writes_evidence_and_refuses_overwrite(self):
        first, second = self.run_baseline(), self.run_baseline()
        output = self.root / "comparison.json"
        args = ["compare-baselines", "--first", str(first), "--second", str(second), "--output", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertTrue(json.loads(output.read_text())["reproduced"])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(args)
        self.assertEqual(raised.exception.code, 2)

    def test_rollback_cli_dispatches_only_the_controlled_probe(self):
        with patch("kirby.cli.run_rollback_check") as run, patch("kirby.cli.run_baseline") as baseline:
            run.return_value = {"directory": "synthetic", "report": {"state": "completed", "outcome": "Controlled rollback verified"}}
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["rollback-check", "--manifest", str(self.manifest), "--revision", "a" * 40,
                             "--layer", "1", "--column", "2", "--scale", "0.97"])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["layer"], 1)
            self.assertEqual(run.call_args.kwargs["column"], 2)
            baseline.assert_not_called()


if __name__ == "__main__":
    unittest.main()
