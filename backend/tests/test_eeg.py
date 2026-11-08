"""Synthetic CPU checks; never download EEG, load pretrained models, or call an API."""

import copy
import json
import tempfile
import unittest
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

try:
    import numpy as np
    import torch
    import scipy
    import jsonschema
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Install backend[eeg] to run CPU EEG checks')
class EEGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        from kirby.eeg.eegnet import EEGNet
        from kirby.eeg.config import EEGConfig
        torch.manual_seed(5)
        self.model = EEGNet(folded=True).eval()
        self.config = EEGConfig(epochs=1, patience=1, seeds=(0,), conditions=('clean',), clean_floor=0,
                                min_reliance=0, attempts=2, device='cpu')

    def data(self, count=8):
        rng = np.random.default_rng(20)
        return {'x': rng.normal(size=(count, 22, 1000)).astype('float32'),
                'y': np.arange(count) % 4, 'ids': np.array([f'trial{i}' for i in range(count)]),
                'blinks': np.array([np.sin(np.linspace(0, np.pi, 250))], dtype='float32')}

    def prepared(self, base):
        from kirby.records import write_json, file_hash
        folder = base / 'subject-01-split-17'; folder.mkdir(parents=True)
        for name in ('train', 'validation', 'development', 'final'):
            data = self.data(); data['ids'] = np.array([name + str(i) for i in range(8)])
            np.savez_compressed(folder / (name + '.npz'), **data)
        write_json(folder / 'manifest.json', {'subject': 1, 'split_seed': 17,
                                             'preprocessing_source_sha256': file_hash(Path(__file__).parents[1] / 'kirby/eeg/data.py'),
                                             'files': {p.name: file_hash(p) for p in folder.glob('*.npz')}})
        return folder

    def response(self, packet, *, diagnosis='eye_blink', operation=None):
        return {k: packet[k] for k in ('task', 'base_checkpoint_id', 'evidence_sha256')} | {
            'diagnosis': diagnosis, 'electrodes': ['Fz'], 'band_hz': [0.5, 4], 'confidence': 0.5,
            'description': 'Synthetic test response.', 'hypothesis': 'Test the nominated filter.',
            'operation': operation, 'stop': operation is None}

    def factory(self, seen, *, fail=False, diagnosis='eye_blink', stop_repair=False):
        from kirby.eeg.astra_eeg import EEGAstraClient
        from kirby.astra import AstraConfig, ProviderFailure
        outer = self
        class FakeClient(EEGAstraClient):
            def __init__(self):
                super().__init__(AstraConfig())
            def request(self, body, identity):
                packet = json.loads(body['input'][0]['content']); seen.append(packet)
                if fail:
                    raise ProviderFailure('Synthetic provider failure', {'usage_known': False, 'api_calls': 1})
                operation = {'op': 'scale_temporal_filter', 'filter': 0, 'factor': 0.0} if packet['task'] == 'repair' and not stop_repair else None
                return {'output_text': json.dumps(outer.response(packet, diagnosis=diagnosis, operation=operation)), 'usage_known': True,
                        'usage': {'input_tokens': 100, 'output_tokens': 100, 'cached_input_tokens': 0}}
        return FakeClient

    def test_architecture_and_nonzero_bn_folding(self):
        from kirby.eeg.eegnet import EEGNet, fold_model
        model = EEGNet().eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 3932)
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.running_mean.uniform_(-0.5, 0.5); module.running_var.uniform_(0.5, 2)
                    module.bias.uniform_(-0.2, 0.2); module.weight.uniform_(0.7, 1.3)
        folded, receipt = fold_model(model, self.data()['x'])
        self.assertTrue(receipt['predictions_identical'])
        self.assertLess(receipt['max_logit_error'], 1e-5)
        self.assertGreater(float(folded.temporal.bias.abs().sum()), 0)
        self.assertEqual(folded(torch.tensor(self.data()['x'])).shape, (8, 4))

    def test_packet_blinding_lossless_export_and_complete_request(self):
        from kirby.eeg.packet import build_packet
        from kirby.eeg.astra_eeg import EEGAstraClient
        from kirby.astra import AstraConfig
        from kirby.records import canonical_json, digest
        for condition in ('A', 'B'):
            packet = build_packet(self.model, str(uuid.uuid4()), condition)
            for forbidden in ('ground_truth', 'subject', 'seed', 'development_feedback', 'labels', 'training_condition'):
                self.assertNotIn(forbidden, packet)
            self.assertEqual('derived_views' in packet, condition == 'B')
            for name, spec in packet['tensors'].items():
                np.testing.assert_array_equal(np.array(spec['values'], dtype='float32').reshape(spec['shape']), self.model.state_dict()[name].numpy())
            checksum = packet.pop('evidence_sha256'); self.assertEqual(digest(packet), checksum); packet['evidence_sha256'] = checksum
            body = EEGAstraClient(AstraConfig()).request_body(packet)
            self.assertLess(len(canonical_json(body).encode()), 128000)
            self.assertFalse(body['store']); self.assertNotIn('tools', body)

    def test_strict_proposal_validation(self):
        from kirby.eeg.packet import build_packet
        from kirby.eeg.astra_eeg import validate
        packet = build_packet(self.model, 'opaque-id', 'A')
        response = self.response(packet)
        self.assertEqual(validate(json.dumps(response), packet), response)
        for change in ({'base_checkpoint_id': 'stale'}, {'electrodes': ['Fz', 'Fz']},
                       {'band_hz': [4, 1]}, {'stop': False}, {'unexpected': True}):
            with self.assertRaises((ValueError, jsonschema.ValidationError)):
                validate(json.dumps(response | change), packet)
        with self.assertRaises(ValueError):
            validate('{"task":"diagnose","task":"repair"}', packet)

    def test_edits_leave_source_exact_and_enforce_cumulative_original_cap(self):
        from kirby.eeg.edits import apply_edit
        from kirby.eeg.eegnet import model_hash
        original_hash = model_hash(self.model)
        current = self.model
        for i in range(7):
            current, receipt = apply_edit(current, self.model, {'op': 'scale_temporal_filter', 'filter': i, 'factor': 0.0})
            self.assertLessEqual(receipt['changed_share'], 0.25)
        with self.assertRaises(ValueError):
            apply_edit(current, self.model, {'op': 'scale_temporal_filter', 'filter': 7, 'factor': 0.0})
        self.assertEqual(original_hash, model_hash(self.model))
        for operation in ({'op': 'scale_temporal_filter', 'filter': True, 'factor': 0},
                          {'op': 'scale_temporal_filter', 'filter': 0, 'factor': float('nan')},
                          {'op': 'set_temporal_filter', 'filter': 0, 'values': [999.0] * 125},
                          {'op': 'scale_spatial_weights', 'filter': 0, 'electrodes': [0, 0], 'factor': 0},
                          {'op': 'scale_temporal_filter', 'filter': 0, 'factor': 1}):
            with self.assertRaises(ValueError):
                apply_edit(self.model, self.model, operation)

    def test_save_load_is_exact(self):
        from kirby.eeg.eegnet import model_hash, save_model, load_model
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.npz'; save_model(self.model, path)
            self.assertEqual(model_hash(self.model), model_hash(load_model(path)))

    def test_accepted_copy_survives_rejected_second_edit(self):
        from kirby.eeg.runner import select_edits
        from kirby.eeg.eegnet import model_hash
        original = {'swapped': {'accuracy': 0.4}, 'clean': {'accuracy': 0.8}, 'artifact_flip_rate': 0.4, 'worst_accuracy': 0.4}
        first = {'swapped': {'accuracy': 0.5}, 'clean': {'accuracy': 0.78}, 'artifact_flip_rate': 0.3, 'worst_accuracy': 0.5}
        second = {'swapped': {'accuracy': 0.6}, 'clean': {'accuracy': 0.76}, 'artifact_flip_rate': 0.2, 'worst_accuracy': 0.6}
        data = self.data(); predictions = {'clean': data['y'].tolist()}
        before = model_hash(self.model)
        with tempfile.TemporaryDirectory() as tmp, patch('kirby.eeg.runner.evaluate', side_effect=[
                (original, predictions), (first, predictions), (second, predictions)]):
            result, metrics, receipts = select_edits(self.model, {}, data, self.config, Path(tmp) / 'edits',
                lambda attempt, current, feedback: {'op': 'scale_temporal_filter', 'filter': attempt, 'factor': 0.0})
        self.assertEqual([r['accepted'] for r in receipts], [True, False])
        self.assertEqual(model_hash(self.model), before)
        self.assertEqual(model_hash(result), receipts[0]['active_after'])
        self.assertEqual(receipts[1]['active_before'], receipts[1]['active_after'])
        self.assertEqual(metrics, first)

    def test_gap_collapse_and_cumulative_clean_damage_are_not_repairs(self):
        from kirby.eeg.evaluate import accept_repair
        original = {'swapped': {'accuracy': 0.4}, 'clean': {'accuracy': 0.8}, 'artifact_flip_rate': 0.4, 'worst_accuracy': 0.4}
        first = {'swapped': {'accuracy': 0.5}, 'clean': {'accuracy': 0.78}, 'artifact_flip_rate': 0.3, 'worst_accuracy': 0.5}
        self.assertTrue(accept_repair(original, original, first))
        second = copy.deepcopy(first); second['swapped']['accuracy'] = 0.6; second['artifact_flip_rate'] = 0.2; second['clean']['accuracy'] = 0.76
        self.assertFalse(accept_repair(original, first, second))
        collapse = copy.deepcopy(first); collapse['swapped']['accuracy'] = 0.3
        self.assertFalse(accept_repair(original, original, collapse))
        no_flip_gain = copy.deepcopy(first); no_flip_gain['artifact_flip_rate'] = 0.4
        self.assertFalse(accept_repair(original, original, no_flip_gain))
        bad_worst = copy.deepcopy(first); bad_worst['worst_accuracy'] = 0.3
        self.assertFalse(accept_repair(original, original, bad_worst))

    def test_artifact_variants_share_inputs_and_waveforms(self):
        from kirby.eeg.shortcuts import specification, variants
        data = self.data(80); before = data['x'].copy()
        for kind in ('eye_blink', 'line_noise', 'muscle'):
            values = variants(data, specification(kind, self.config), 6)
            np.testing.assert_array_equal(values['clean'], before)
            delta = values['all_artifact'] - before
            for name in ('original', 'swapped', 'uniform'):
                for i, change in enumerate(values[name] - before):
                    self.assertTrue(np.array_equal(change, delta[i]) or not change.any())
            self.assertGreater(float(np.abs(delta).sum()), 0)
        np.testing.assert_array_equal(data['x'], before)
        clean = variants(data, specification('clean', self.config), 6)
        for x in clean.values():
            np.testing.assert_array_equal(x, before)

    def test_split_and_train_fitted_preprocessing(self):
        from kirby.eeg.data import split_indices, fit_preprocessing, transform
        splits = split_indices(np.repeat(np.arange(4), 72), 17)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {'train': 184, 'validation': 48, 'development': 56})
        self.assertEqual(len(set(np.concatenate(list(splits.values())).tolist())), 288)
        rng = np.random.default_rng(5); train = rng.normal(size=(8, 25, 1000))
        train[:, :22] += train[:, 22:23] * 2
        fitted = fit_preprocessing(train); snapshot = {k: v.copy() for k, v in fitted.items()}
        transformed = transform(train, fitted)
        np.testing.assert_allclose(transformed.mean((0, 2)), 0, atol=1e-6)
        np.testing.assert_allclose(transformed.astype('float64').std((0, 2)), 1, atol=1e-6)
        transform(rng.normal(size=(8, 25, 1000)) + 20, fitted)
        for k in fitted:
            np.testing.assert_array_equal(fitted[k], snapshot[k])

    def test_final_lock_and_split_tamper_detection(self):
        from kirby.eeg.data import load_split
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.prepared(Path(tmp))
            with self.assertRaises(ValueError):
                load_split(folder, 'final')
            self.assertEqual(len(load_split(folder, 'final', final_authorized=True)['y']), 8)
            with (folder / 'development.npz').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaises(ValueError):
                load_split(folder, 'development')

    def test_mat_reader_imagery_offset_and_baseline_runs(self):
        from kirby.eeg.data import read_session
        length = 1510
        # All starts repeat in this synthetic parser fixture, not a scientific dataset.
        raw = np.broadcast_to(np.arange(length)[:, None], (length, 25)).copy().astype(float)
        empty = SimpleNamespace(trial=np.array([]))
        run = SimpleNamespace(trial=np.ones(288), y=np.tile([1, 2, 3, 4], 72), artifacts=np.zeros(288), fs=250, X=raw)
        with patch('kirby.eeg.data.loadmat', return_value={'data': np.array([empty, run], dtype=object)}), \
             patch('kirby.eeg.data.sosfiltfilt', side_effect=lambda sos, x, axis: x):
            x, y, ids, flags = read_session(Path('A01T.mat'))
        self.assertEqual(x.shape, (288, 25, 1000)); self.assertEqual(x[0, 0, 0], 500); self.assertEqual(x[0, 0, -1], 1499)
        self.assertEqual(y[:4].tolist(), [0, 1, 2, 3])

    def test_request_budget_failure_and_no_retry(self):
        from kirby.eeg.astra_eeg import RequestBudget
        from kirby.eeg.packet import build_packet
        from kirby.astra import ProviderFailure
        packet = build_packet(self.model, 'opaque', 'A'); seen = []
        with tempfile.TemporaryDirectory() as tmp:
            budget = RequestBudget(replace(self.config, max_api_calls=1), Path(tmp), self.factory(seen))
            budget.request(packet, 'first')
            with self.assertRaises(ValueError):
                budget.request(packet, 'second')
            self.assertEqual(len(seen), 1); self.assertGreater(budget.estimated_usd, 0)
            failed = RequestBudget(self.config, Path(tmp) / 'fail', self.factory(seen, fail=True))
            with self.assertRaises(ProviderFailure):
                failed.request(packet, 'sent-once')
            self.assertEqual(failed.calls, 1); self.assertGreater(failed.estimated_usd, 0)
            self.assertTrue((Path(tmp) / 'fail/sent-once/provider.json').is_file())

    def test_real_cpu_end_to_end_fake_astra_and_final_order(self):
        from kirby.eeg.runner import run
        from kirby.eeg.data import load_split
        from kirby.records import read_json, file_hash
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); self.prepared(base / 'data'); identity = str(uuid.uuid4()); root = base / 'runs' / identity
            final_loads = []
            def checked_load(directory, split, **kwargs):
                if split == 'final':
                    self.assertTrue((root / 'selection-lock.json').is_file())
                    self.assertTrue((root / 'final-started.json').is_file())
                    final_loads.append(split)
                else:
                    self.assertFalse((root / 'final-started.json').exists())
                return load_split(directory, split, **kwargs)
            with patch('kirby.eeg.runner.load_split', side_effect=checked_load):
                report = run(self.config, base / 'runs', base / 'data', run_id=identity,
                             client_factory=self.factory(seen, diagnosis='uncertain'))
            self.assertEqual(report['status'], 'completed'); self.assertEqual(len(final_loads), 1)
            self.assertEqual([p['task'] for p in seen[:2]], ['diagnose', 'diagnose'])
            self.assertTrue(all(p['task'] == 'repair' for p in seen[2:]))
            self.assertEqual(report['api_calls'], 6)
            for packet in seen:
                self.assertNotIn('ground_truth', packet); self.assertNotIn('final', packet)
            model = report['models'][0]
            self.assertEqual(report['config']['repair_policy'], 'feedback_guided')
            for name in ('astra_A', 'astra_B'):
                self.assertEqual(model['diagnoses'][name]['response']['diagnosis'], 'uncertain')
                self.assertTrue(model['diagnoses'][name]['score']['abstained'])
                self.assertEqual(model['methods'][name]['attempted_edits'], 2)
                self.assertEqual(model['methods'][name]['repair_requests'], 2)
            for packet in seen[2:]:
                feedback = packet['development_feedback']
                self.assertEqual(feedback['repair_policy'], 'feedback_guided')
                self.assertEqual(feedback['frozen_blind_diagnosis']['diagnosis'], 'uncertain')
                self.assertNotIn('ground_truth', json.dumps(feedback))
                self.assertNotIn('final', json.dumps(feedback))
            self.assertEqual(set(model['methods']), {'no_edit', 'input_filter', 'random', 'rule', 'astra_A', 'astra_B'})
            for method in model['methods'].values():
                self.assertIn('final', method)
            self.assertEqual(model['methods']['astra_A']['accepted_edits'], 0)
            self.assertEqual(model['methods']['astra_A']['model_sha256'], model['original_hash'])
            self.assertTrue((root / 'final-accuracy.png').is_file())
            self.assertTrue((root / 'filter-views.png').is_file())
            index = read_json(root / 'evidence-index.json')
            for row in index['files']:
                self.assertEqual(file_hash(root / row['path']), row['sha256'])
            with ZipFile(next(root.glob('*.zip'))) as archive:
                self.assertIn('START_HERE.md', archive.namelist()); self.assertIn('selection-lock.json', archive.namelist())
                self.assertFalse(any('final.npz' in name or 'train.npz' in name for name in archive.namelist()))
            with self.assertRaises(FileExistsError):
                run(self.config, base / 'runs', base / 'data', run_id=identity, client_factory=self.factory(seen))
            self.assertEqual(len(seen), 6)

    def test_legacy_abstention_gate_and_feedback_guided_provider_stop(self):
        from kirby.eeg.runner import run
        for policy, diagnosis, expected_calls in (
                ('diagnosis_gated', 'uncertain', 2), ('diagnosis_gated', 'clean', 2),
                ('feedback_guided', 'uncertain', 4), ('feedback_guided', 'clean', 4)):
            with self.subTest(policy=policy, diagnosis=diagnosis), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); self.prepared(base / 'data'); seen = []
                report = run(replace(self.config, repair_policy=policy), base / 'runs', base / 'data',
                             client_factory=self.factory(seen, diagnosis=diagnosis, stop_repair=True))
                self.assertEqual(report['api_calls'], expected_calls)
                model = report['models'][0]
                for name in ('astra_A', 'astra_B'):
                    method = model['methods'][name]
                    self.assertEqual(method['attempted_edits'], 0)
                    self.assertEqual(method['accepted_edits'], 0)
                    self.assertEqual(method['model_sha256'], model['original_hash'])
                    self.assertEqual(method['repair_requests'], int(policy == 'feedback_guided'))
                    self.assertEqual(method['repair_policy'], policy)
                    self.assertEqual(model['diagnoses'][name]['response']['diagnosis'], diagnosis)
                    self.assertIn('Astra stopped' if policy == 'feedback_guided' else 'no repair attempted', method['stop_reason'])

    def test_failed_provider_exports_partial_and_never_loads_final(self):
        from kirby.eeg.runner import run
        from kirby.astra import ProviderFailure
        from kirby.records import read_json
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); self.prepared(base / 'data'); identity = str(uuid.uuid4()); root = base / 'runs' / identity
            with self.assertRaises(ProviderFailure):
                run(self.config, base / 'runs', base / 'data', run_id=identity, client_factory=self.factory(seen, fail=True))
            report = read_json(root / 'report.json')
            self.assertEqual(report['status'], 'failed'); self.assertFalse(report['final_evaluation_started'])
            self.assertEqual(report['api_calls'], 1); self.assertEqual(len(seen), 1)
            self.assertTrue((root / 'START_HERE.md').is_file()); self.assertTrue(list(root.glob('*.zip')))
            self.assertFalse((root / 'final-started.json').exists())

    def test_failed_gate_spends_nothing_and_no_final_evaluation(self):
        from kirby.eeg.runner import run
        seen = []
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); self.prepared(base / 'data')
            report = run(replace(self.config, clean_floor=1), base / 'runs', base / 'data', client_factory=self.factory(seen))
            self.assertEqual(report['status'], 'gates_failed'); self.assertEqual(seen, [])
            self.assertFalse(report['final_evaluation_started'])

    def test_no_astra_and_higher_resource_baselines(self):
        from kirby.eeg.runner import run
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); self.prepared(base / 'data')
            report = run(replace(self.config, astra=False, extended_baselines=True), base / 'runs', base / 'data')
            self.assertEqual(report['api_calls'], 0)
            methods = report['models'][0]['methods']
            self.assertIn('last_layer', methods); self.assertIn('full_clean_retrain', methods)
            self.assertEqual(methods['last_layer']['resource_class'], 'clean_labeled_retraining_outside_edit_cap')
            self.assertIn('integrated_gradients', report['models'][0]['diagnoses'])


if __name__ == '__main__':
    unittest.main()
