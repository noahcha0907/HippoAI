"""Check shipped notebook syntax, safe extraction and no-repeat behavior offline."""

import importlib.util
import io
import json
import tempfile
import subprocess
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

BACKEND = Path(__file__).resolve().parents[1]


class EEGPackagingTests(unittest.TestCase):
    def builder(self):
        spec = importlib.util.spec_from_file_location('eeg_notebook_builder', BACKEND / 'tools/build_eeg_notebook.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_notebook_cells_compile_and_match_builder(self):
        notebook = json.loads((BACKEND / 'notebooks/kirby_eeg_colab.ipynb').read_text(encoding='utf-8'))
        code = [''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code']
        builder = self.builder()
        self.assertEqual([c.strip() for c in code], [s.strip() for s in (builder.SETUP, builder.CONFIG, builder.RUN, builder.DOWNLOAD)])
        for i, cell in enumerate(code):
            compile(cell, f'cell-{i}', 'exec')

    def test_bundle_has_nested_package_and_no_credentials_or_data(self):
        bundle = BACKEND / 'dist/kirby_eeg_colab.zip'
        if not bundle.exists():
            self.skipTest('Build the explicit source bundle before packaging checks')
        with ZipFile(bundle) as archive:
            self.assertIn('backend/kirby/eeg/runner.py', archive.namelist())
            self.assertIn('backend/notebooks/kirby_eeg_colab.ipynb', archive.namelist())
            for name in archive.namelist():
                if name != 'backend/experiments/eeg_replication_v1.json':
                    self.assertIn(Path(name).suffix, ('.py', '.md', '.toml', '.ipynb'))
                self.assertNotIn('..', Path(name).parts)
                source = BACKEND / Path(name).relative_to('backend')
                self.assertEqual(archive.read(name), source.read_bytes(), name)

    def test_bootstrap_rejects_path_traversal_before_subprocess(self):
        builder = self.builder()
        google = types.ModuleType('google'); colab = types.ModuleType('google.colab')
        colab.files = types.SimpleNamespace(upload=lambda: {})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); bundle = root / 'source.zip'
            with ZipFile(bundle, 'w') as archive:
                archive.writestr('backend/../../escaped.py', 'print(1)')
            code = builder.SETUP.replace("Path('/content/kirby-eeg-workspace')", repr(str(root / 'workspace')))
            code = code.replace('WORKSPACE = ' + repr(str(root / 'workspace')), 'WORKSPACE = Path(' + repr(str(root / 'workspace')) + ')')
            with patch.dict('sys.modules', {'google': google, 'google.colab': colab}), patch('subprocess.run') as process:
                with self.assertRaises(ValueError):
                    exec(compile(code, 'setup', 'exec'), {'EEG_SOURCE_ZIP': bundle})
            process.assert_not_called()
            self.assertFalse((root / 'escaped.py').exists())

    def test_saved_run_does_not_train_or_request_secret(self):
        builder = self.builder()
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); identity = str(uuid.uuid4())
            (root / 'eeg-attempt-id.txt').write_text(identity)
            run = root / 'eeg-runs' / identity; run.mkdir(parents=True)
            (run / 'report.json').write_text('{"status":"failed"}')
            namespace = {'Path': Path, 'json': json, 'uuid': uuid, 'WORKSPACE': root, 'source_hash': 'test-source', 'subprocess': subprocess}
            exec(builder.CONFIG, namespace)
            with patch('subprocess.run') as process:
                exec(builder.RUN, namespace)
            process.assert_not_called()
            self.assertEqual(namespace['run_id'], identity)
            namespace['EPOCHS'] = 2
            with self.assertRaises(ValueError), patch('subprocess.run') as process:
                exec(builder.RUN, namespace)
            process.assert_not_called()

    def worker_setup(self, directory, existing=False, healthy=False, never_ready=False):
        builder = self.builder()
        code = builder.SETUP.split("VENV = WORKSPACE", 1)[1].split('# Reuses Colab', 1)[0]
        code = 'VENV = WORKSPACE' + code
        root = Path(directory)
        worker = root / 'venv-test' / 'bin/python'
        if existing:
            worker.parent.mkdir(parents=True)
            worker.touch()
        calls = []
        pip_ready = healthy
        def execute(command, **kwargs):
            nonlocal pip_ready
            calls.append(command)
            if command[1:3] == ['-m', 'venv']:
                self.assertIn('--without-pip', command)
                self.assertIn('--system-site-packages', command)
                self.assertNotIn('--clear', command)
                worker.parent.mkdir(parents=True, exist_ok=True)
                worker.touch()
            elif '--python' in command:
                self.assertEqual(command[command.index('--python') + 1], str(worker))
                pip_ready = not never_ready
            elif '-c' in command:
                return subprocess.CompletedProcess(command, 0 if pip_ready else 1)
            else:
                self.fail('Unexpected subprocess during worker preparation')
            return subprocess.CompletedProcess(command, 0)
        namespace = {'WORKSPACE': root, 'source_hash': 'test', 'subprocess': subprocess, 'sys': sys}
        with patch('subprocess.run', side_effect=execute):
            exec(compile(code, 'worker-setup', 'exec'), namespace)
        return calls

    def test_new_run_streams_worker_output_and_passes_repair_policy(self):
        builder = self.builder()
        colab = types.ModuleType('google.colab')
        colab.userdata = types.SimpleNamespace(get=lambda key: 'synthetic-test-key')
        for returncode in (0, 1):
            with self.subTest(returncode=returncode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                namespace = {'Path': Path, 'json': json, 'uuid': uuid, 'WORKSPACE': root,
                             'source_hash': 'new-source', 'subprocess': subprocess,
                             'PYTHON': Path(sys.executable), 'BACKEND': root, 'env': {}}
                exec(builder.CONFIG, namespace)
                with patch.dict('sys.modules', {'google.colab': colab}), \
                     patch('subprocess.Popen') as process, patch('sys.stdout', new_callable=io.StringIO) as output:
                    worker = process.return_value.__enter__.return_value
                    worker.stdout = iter(['astra_A: requesting repair 1/5\n', 'astra_A: edit 1/5 rejected\n'])
                    worker.wait.return_value = returncode
                    exec(builder.RUN, namespace)
                    command = process.call_args.args[0]
                    self.assertEqual(command[command.index('--repair-policy') + 1], 'feedback_guided')
                    self.assertEqual(process.call_args.kwargs['stderr'], subprocess.STDOUT)
                    self.assertIn('requesting repair 1/5', output.getvalue())
                    self.assertIn(f'Worker exit code: {returncode}', output.getvalue())
                    self.assertNotIn('synthetic-test-key', output.getvalue())
                    self.assertNotIn('OPENAI_API_KEY', namespace['env'])

    def test_worker_without_ensurepip_and_partial_environment_recover(self):
        for existing in (False, True):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as tmp:
                calls = self.worker_setup(tmp, existing=existing)
                self.assertEqual(sum('--without-pip' in c for c in calls), 1)
                self.assertEqual(sum('--python' in c for c in calls), 1)
                self.assertIn('-c', calls[-1])  # Verify target health after installing pip.

    def test_healthy_worker_does_not_recreate_or_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = self.worker_setup(tmp, existing=True, healthy=True)
            self.assertEqual(len(calls), 1)
            self.assertIn('-c', calls[0])

    def test_broken_worker_stops_before_dependency_install_or_experiment(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'worker is not usable'):
                self.worker_setup(tmp, existing=True, never_ready=True)


if __name__ == '__main__':
    unittest.main()
