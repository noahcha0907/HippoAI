"""Offline checks for the predeclared replication; no EEG or provider calls."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[1]


class ReplicationTests(unittest.TestCase):
    def builder(self):
        with patch.object(sys, 'path', [str(BACKEND/'tools'), *sys.path]):
            spec = importlib.util.spec_from_file_location('replication_builder', BACKEND/'tools/build_eeg_replication.py')
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_plan_matches_run6_except_subjects_and_total_budgets(self):
        plan = json.loads((BACKEND/'experiments/eeg_replication_v1.json').read_text())
        reference_path = BACKEND.parent/'assets/evidence/run6/replay.json'
        if reference_path.exists():
            reference = json.loads(reference_path.read_text())['report']
            self.assertEqual(plan['config'], dict(reference['config'], subjects=[2,3], max_api_calls=144, max_estimated_usd=100.0))
        config_class = runpy.run_path(str(BACKEND/'kirby/eeg/config.py'))['EEGConfig']
        actual = dataclasses.asdict(config_class(subjects=(2,3),max_api_calls=144,max_estimated_usd=100))
        self.assertEqual(json.loads(json.dumps(actual)), plan['config'])

    def test_notebook_matches_builder_and_settings_are_enforced(self):
        sources = self.builder().cells()
        notebook=json.loads((BACKEND/'notebooks/kirby_eeg_replication_colab.ipynb').read_text())
        self.assertEqual([''.join(c['source']).strip() for c in notebook['cells'] if c['cell_type']=='code'],[s.strip() for s in sources])
        for source in sources:
            compile(source,'replication','exec')
        namespace={'BACKEND':BACKEND}
        exec(sources[1],namespace)
        self.assertEqual(namespace['SUBJECTS'],[2,3])
        namespace['ATTEMPTS']=4
        with self.assertRaisesRegex(ValueError,'Frozen replication setting'):
            namespace['verify_replication']()
        self.assertTrue(sources[2].startswith('verify_replication()'))
        self.assertIn('eeg-replication-v1-attempt-id.txt',sources[2])

    def test_source_tampering_blocks_execution_and_crlf_is_portable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); plan=json.loads((BACKEND/'experiments/eeg_replication_v1.json').read_text())
            for relative in [*plan['source_sha256'],'experiments/eeg_replication_v1.json']:
                path=root/relative; path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes((BACKEND/relative).read_text(encoding='utf-8').replace('\n','\r\n').encode())
            namespace={'BACKEND':root}; exec(self.builder().cells()[1],namespace)
            source=root/'kirby/eeg/astra_eeg.py'
            source.write_text(source.read_text()+'\n# changed prompt source\n')
            with self.assertRaisesRegex(ValueError,'Frozen experiment source changed'):
                namespace['verify_replication']()


if __name__ == '__main__':
    unittest.main()
