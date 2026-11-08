"""Exercise continuation setup without Colab, GPU access, or subprocesses."""

import contextlib
import io
import json
import subprocess
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.backend = self.workspace / "backend"
        plan = self.backend / "data" / "plan-dev-256"
        plan.mkdir(parents=True)
        (plan / "development.json").write_text("{}")
        self.worker = self.workspace / "worker-venv-test" / "bin" / "python"
        self.worker.parent.mkdir(parents=True)
        self.worker.write_text("synthetic executable placeholder")
        notebook = json.loads((Path(__file__).parents[1] / "notebooks" / "kirby_next_steps.ipynb").read_text(encoding="utf-8"))
        self.cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}

    def bundle(self, unsafe=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            for name in ("patches", "rollback", "reproducibility"):
                archive.writestr(f"backend/kirby/{name}.py", "# synthetic source\n")
            if unsafe:
                archive.writestr("backend/../escaped.py", "bad")
        return stream.getvalue()

    def execute_resume(self, namespace):
        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        colab.files = Mock()
        colab.userdata = Mock()
        source = self.cells["resume"].replace('BACKEND_PATH = ""', "BACKEND_PATH = " + repr(str(self.backend)), 1)
        result = subprocess.CompletedProcess([], 0, stdout="Synthetic subprocess passed\n")
        with patch.dict("sys.modules", {"google": google, "google.colab": colab}), \
                patch.dict("os.environ", {}, clear=False), \
                patch("subprocess.run", return_value=result), contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "colab-resume", "exec"), namespace)
        colab.files.upload.assert_not_called()

    def test_resume_restores_variables_and_keeps_existing_data(self):
        namespace = {"KIRBY_BUNDLE_BYTES": self.bundle()}
        self.execute_resume(namespace)
        self.assertEqual(namespace["backend"], self.backend)
        self.assertEqual(Path(namespace["worker_python"]), self.worker)
        self.assertTrue(namespace["plan"].is_dir())
        self.assertEqual((namespace["plan"] / "development.json").read_text(), "{}")
        self.assertTrue((self.backend / "kirby" / "patches.py").is_file())
        self.assertNotIn("KIRBY_BUNDLE_BYTES", namespace)

    def test_unsafe_bundle_is_rejected_before_source_writes(self):
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            self.execute_resume({"KIRBY_BUNDLE_BYTES": self.bundle(unsafe=True)})
        self.assertFalse((self.backend / "kirby").exists())
        self.assertFalse((self.workspace / "escaped.py").exists())

    def test_rollback_cell_requires_successful_comparison(self):
        for namespace in ({}, {"comparison_path": self.workspace / "missing.json"}):
            with self.assertRaisesRegex(RuntimeError, "comparison cell"):
                exec(compile(self.cells["rollback"], "colab-rollback", "exec"), namespace)
        path = self.workspace / "comparison.json"
        path.write_text(json.dumps({"reproduced": False}))
        with self.assertRaisesRegex(RuntimeError, "has not passed"):
            exec(compile(self.cells["rollback"], "colab-rollback", "exec"), {"comparison_path": path, "json": json})


if __name__ == "__main__":
    unittest.main()
