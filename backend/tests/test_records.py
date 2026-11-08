import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kirby.records import write_json


class RecordTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "report.json"
        self.path.write_text('{"state":"old"}')

    def test_transient_windows_lock_retries_atomic_replace_only(self):
        failure = PermissionError("Synthetic file lock")
        failure.winerror = 32
        replace = Path.replace
        calls = []
        def transient(source, destination):
            calls.append(source)
            if len(calls) == 1:
                self.assertEqual(json.loads(self.path.read_text())["state"], "old")
                raise failure
            return replace(source, destination)
        with patch.object(Path, "replace", transient), patch("kirby.records.time.sleep") as sleep:
            write_json(self.path, {"state": "new"})
        self.assertEqual(len(calls), 2)
        sleep.assert_called_once()
        self.assertEqual(json.loads(self.path.read_text())["state"], "new")

    def test_persistent_lock_is_bounded_and_preserves_previous_report(self):
        failure = PermissionError("Synthetic persistent denial")
        failure.winerror = 5
        with patch.object(Path, "replace", side_effect=failure) as replace, patch("kirby.records.time.sleep"):
            with self.assertRaises(PermissionError):
                write_json(self.path, {"state": "new"})
        self.assertEqual(replace.call_count, 7)
        self.assertEqual(json.loads(self.path.read_text())["state"], "old")

    def test_other_permission_errors_are_not_retried(self):
        with patch.object(Path, "replace", side_effect=PermissionError("Synthetic denial")) as replace:
            with self.assertRaises(PermissionError):
                write_json(self.path, {"state": "new"})
        self.assertEqual(replace.call_count, 1)


if __name__ == "__main__":
    unittest.main()
