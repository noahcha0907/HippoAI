"""Colab secret handling and repeat-click protection without paid calls."""

import contextlib
import io
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace
from zipfile import ZipFile

from kirby.evidence import create_evidence_bundle


class AstraColabTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        runs = root / "backend" / "runs"
        rollback = runs / "rollback"
        rollback.mkdir(parents=True)
        (rollback / "report.json").write_text(json.dumps({"task": "controlled_rollback_check", "state": "completed",
                                                        "weights_restored": True, "outputs_restored": True,
                                                        "finished_at": "2026-11-07T00:00:00Z"}))
        notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "kirby_astra_colab.ipynb").read_text(encoding="utf-8"))
        self.source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "attempt")
        self.download_source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "download")
        self.userdata = Mock()
        self.userdata.get.return_value = "sk-synthetic-secret"
        self.seen_env = None
        def execute(command, env):
            self.seen_env = dict(env)
            run_id = command[command.index("--attempt-id") + 1]
            destination = runs / run_id
            destination.mkdir()
            (destination / "report.json").write_text(json.dumps({"state": "completed", "outcome": "synthetic test"}))
        self.worker = Mock(side_effect=execute)
        self.namespace = {"workspace": root, "runs": runs, "Path": Path, "json": json, "os": os, "uuid": uuid,
                          "userdata": self.userdata, "run_worker": self.worker, "worker_python": "synthetic-python",
                          "plan": root / "plan", "MODEL_ID": "synthetic-model", "MODEL_REVISION": "a" * 40,
                          "PRECISION": "float32", "MAX_SECONDS": 3600}

    def execute(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(compile(self.source, "astra-colab-attempt", "exec"), self.namespace)
        return output.getvalue()

    def test_repeat_execution_reuses_report_without_key_read_or_new_call(self):
        output = self.execute()
        self.assertEqual(self.seen_env["OPENAI_API_KEY"], "sk-synthetic-secret")
        self.assertNotIn("OPENAI_API_KEY", self.namespace["worker_env"])
        self.assertNotIn("sk-synthetic-secret", output)
        output = self.execute()
        self.assertIn("no new API call", output)
        self.assertEqual(self.worker.call_count, 1)
        self.assertEqual(self.userdata.get.call_count, 1)
        for path in self.namespace["workspace"].rglob("*.json"):
            self.assertNotIn("sk-synthetic-secret", path.read_text())

    def test_missing_secret_stops_before_starting_worker(self):
        self.userdata.get.side_effect = RuntimeError("Secret is not accessible")
        with self.assertRaisesRegex(RuntimeError, "Colab Secrets"):
            self.execute()
        self.worker.assert_not_called()
        self.assertNotIn("OPENAI_API_KEY", self.namespace["worker_env"])

    def test_new_protocol_does_not_reuse_activation_only_attempt(self):
        root = self.namespace["workspace"]
        (root / "astra-attempt-id.txt").write_text(str(uuid.uuid4()))
        self.execute()
        self.assertTrue((root / "astra-search-v1-172-attempt-id.txt").is_file())
        self.assertEqual(self.worker.call_count, 1)

    def test_download_exposes_current_report_and_summary_at_zip_root(self):
        self.execute()
        def package(command, **kwargs):
            create_evidence_bundle(Path(command[command.index("--backend") + 1]),
                                   Path(command[command.index("--report") + 1]),
                                   Path(command[command.index("--output") + 1]))
        downloads = Mock()
        self.namespace.update(backend=self.namespace["workspace"] / "backend", files=downloads,
                              subprocess=SimpleNamespace(check_output=lambda *a, **k: "[]", run=package))
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(self.download_source, "astra-colab-download", "exec"), self.namespace)
        downloads.download.assert_called_once()
        with ZipFile(downloads.download.call_args.args[0]) as archive:
            self.assertIn("START_HERE.md", archive.namelist())
            self.assertEqual(json.loads(archive.read("latest-report.json"))["outcome"], "synthetic test")


if __name__ == "__main__":
    unittest.main()
