"""Exercise the notebook's setup failure handling without installs or a GPU."""

import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock


class ColabSetupTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parents[1] / "notebooks" / "kirby_colab.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "install")
        self.run_source = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "run")
        self.tree = ast.parse(self.source)

    def helper(self, returncode):
        function = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef))
        module = ast.Module(body=[function], type_ignores=[])
        result = subprocess.CompletedProcess(["synthetic-setup"], returncode, stdout="Underlying setup diagnostic\n")
        process = Mock(PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
        process.run.return_value = result
        namespace = {"subprocess": process}
        exec(compile(module, "colab-install", "exec"), namespace)
        return namespace["run_setup"], process

    def test_failed_setup_prints_diagnostic_before_raising(self):
        helper, process = self.helper(1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError):
            helper(["synthetic-setup"])
        self.assertIn("Underlying setup diagnostic", output.getvalue())
        self.assertEqual(process.run.call_args.kwargs["stderr"], subprocess.STDOUT)

    def test_successful_setup_preserves_options(self):
        helper, process = self.helper(0)
        with contextlib.redirect_stdout(io.StringIO()):
            helper(["synthetic-setup"], cwd="worker")
        self.assertEqual(process.run.call_args.kwargs["cwd"], "worker")

    def test_environment_uses_builtin_creation_and_fresh_directories(self):
        self.assertIn("'--creator', 'builtin'", self.source)
        self.assertIn("'--seeder', 'app-data'", self.source)
        self.assertIn("'--system-site-packages'", self.source)
        self.assertIn("mkdtemp(prefix='worker-venv-'", self.source)
        self.assertIn("'--target', str(bootstrap)", self.source)
        self.assertNotIn("'-m', 'venv'", self.source)

    def test_dataset_failures_are_visible_and_prevent_model_launch(self):
        for stage in ("prepare", "validate"):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                backend = Path(directory)
                if stage == "validate":
                    (backend / "data" / "plan-dev-8").mkdir(parents=True)
                helper, process = self.helper(2)
                process.run.return_value.stdout = f"Dataset {stage} diagnostic\n"
                namespace = {"backend": backend, "data": backend / "data" / "medqa",
                             "worker_python": "synthetic-worker", "run_setup": helper,
                             "subprocess": process}
                output = io.StringIO()
                with contextlib.redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError):
                    exec(compile(self.run_source, "colab-run", "exec"), namespace)
                self.assertIn(f"Dataset {stage} diagnostic", output.getvalue())
                self.assertEqual(process.run.call_args.args[0][3], stage)
                process.Popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
