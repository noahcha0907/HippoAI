"""Check shipped notebook syntax, safe extraction and no-repeat behavior offline."""

import importlib.util
import json
import tempfile
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


if __name__ == '__main__':
    unittest.main()
