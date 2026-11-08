"""Evidence exports expose actual results and index all bundled bytes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from kirby.evidence import create_evidence_bundle, render_summary
from kirby.records import write_json


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backend = self.root / "backend"
        self.run = self.backend / "runs" / "synthetic-run"
        self.run.mkdir(parents=True)
        self.report = self.run / "report.json"
        metrics = {"total": 2, "correct": 1, "accuracy": 0.5, "invalid_count": 0, "truncated_count": 0}
        write_json(self.report, {"state": "completed", "outcome": "Astra candidate evaluated and rolled back",
                                "comparison": {"original": metrics, "candidate": metrics, "signal": "no_gain",
                                               "corrected": 0, "regressed": 0, "final_evaluated": False},
                                "weights_restored": True, "outputs_restored": True, "candidate_retained": False,
                                "usage": {"api_calls": 1}})
        write_json(self.run / "proposal.json", {"hypothesis": "Synthetic sensitivity test only."})
        for name in ("original", "candidate"):
            (self.run / f"{name}.jsonl").write_text(
                '\n'.join(json.dumps({"question_id": str(i), "text": "A."}) for i in range(2)), encoding="utf-8")

    def test_readable_root_results_and_byte_index_without_secrets_data_or_weights(self):
        (self.backend / ".env").write_text("OPENAI_API_KEY=sk-synthetic-secret")
        (self.backend / "data").mkdir()
        (self.backend / "data" / "train.jsonl").write_text("raw-dataset-must-not-be-exported")
        (self.run / "model.safetensors").write_bytes(b"weights-must-not-be-exported")
        bundle = self.root / "evidence.zip"
        result = create_evidence_bundle(self.backend, self.report, bundle)
        with ZipFile(bundle) as archive:
            names = archive.namelist()
            self.assertIn("START_HERE.md", names)
            self.assertIn("latest-report.json", names)
            self.assertIn("latest-proposal.json", names)
            self.assertIn("latest-original.jsonl", names)
            self.assertIn("backend/runs/synthetic-run/original.jsonl", names)
            self.assertFalse(any(name.endswith((".env", ".safetensors", "train.jsonl")) for name in names))
            summary = archive.read("START_HERE.md").decode()
            self.assertIn("Generated answer texts changed: **0/2**", summary)
            self.assertIn("No training sensitivity probes were recorded", summary)
            self.assertIn("not model improvement", summary)
            index = json.loads(archive.read("evidence-index.json"))
            self.assertEqual(index["latest_run_id"], "synthetic-run")
            self.assertEqual({row["path"] for row in index["files"]}, set(names) - {"evidence-index.json"})
            for row in index["files"]:
                content = archive.read(row["path"])
                self.assertEqual(row["bytes"], len(content))
                self.assertEqual(row["sha256"], hashlib.sha256(content).hexdigest())
                self.assertNotIn(b"sk-synthetic-secret", content)
        self.assertEqual(result["files"], len(names))
        with self.assertRaisesRegex(ValueError, "already exists"):
            create_evidence_bundle(self.backend, self.report, bundle)

    def test_incomplete_run_and_partial_scores_are_not_presented_as_success(self):
        write_json(self.report, {"state": "incomplete", "outcome": "Run incomplete",
                                "weights_restored": None, "candidate_retained": False,
                                "error": {"type": "KeyboardInterrupt"},
                                "training_sensitivity": [{"probe_id": "probe-1", "operation": {"layer": 2, "columns": [7], "scale": 0.98},
                                                          "state": "incomplete", "comparison": None, "scores_restored": None}]})
        summary = render_summary(self.report)
        self.assertIn("The run did not complete", summary)
        self.assertIn("not measured", summary)
        self.assertIn("Missing/null measurements mean unverified", summary)
        self.assertNotIn("Generated answer texts changed", summary)

    def test_training_effects_are_shown_separately_from_development_accuracy(self):
        report = json.loads(self.report.read_text())
        report.update(training_sensitivity=[{"probe_id": "probe-1", "operation": {"layer": 30, "columns": [8941], "scale": 0.98},
                                             "state": "completed", "scores_restored": True,
                                             "comparison": {"mean_expected_margin_delta": 0.0123}}],
                      candidate_training_comparison={"mean_expected_margin_delta": 0.01,
                                                     "max_absolute_log_score_delta": 0.02, "preferred_label_changes": 0})
        write_json(self.report, report)
        summary = render_summary(self.report)
        self.assertIn("0.0123", summary)
        self.assertIn("**0.02**", summary)
        self.assertIn("**no_gain**", summary)
        self.assertIn("do not replace", summary)

    def test_report_outside_runs_is_rejected(self):
        outside = self.root / "report.json"
        write_json(outside, {})
        with self.assertRaisesRegex(ValueError, "inside backend/runs"):
            create_evidence_bundle(self.backend, outside, self.root / "wrong.zip")

    def test_search_summary_separates_new_score_protocol_and_training_control(self):
        report = json.loads(self.report.read_text())
        report.update(task="astra_bounded_search", search={"history_count": 2, "selected": "confirm-1",
                       "random_control_comparison": {"mean_expected_log_score_delta": -0.01},
                       "selected_minus_control_log_score_gain": 0.02},
                      development_score_comparison={"original": {"questions": 172},
                                                    "mean_expected_log_score_delta": 0.01, "mean_expected_margin_delta": 0.02})
        write_json(self.report, report)
        summary = render_summary(self.report)
        self.assertIn("Questions scored: **172**", summary)
        self.assertIn("without a period", summary)
        self.assertIn("not directly comparable", summary)
        self.assertIn("not a significance test", summary)


if __name__ == "__main__":
    unittest.main()
