"""Pruning selection, actual tensor rollback, failure handling, and Colab reuse."""

import contextlib
import io
import json
import os
import re
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zipfile import ZipFile

try:
    import torch
except ImportError:
    torch = None

from kirby.cli import main
from kirby.evaluation import Generation, score
from kirby.evidence import create_evidence_bundle
from kirby.patches import ColumnScale, UnitAblation, PhiColumnEditor, weights_hash
from kirby.pruning import PROTOCOL, PruningConfig, comparison, pruning_plan, run_pruning
from kirby.datasets import Question, QuestionInput, load_medqa
from kirby.records import read_json, write_json
from kirby.runner import RunStopped
from test_evaluation import make_manifest, row, save_rows


class PruningMetricsTests(unittest.TestCase):
    def test_improvement_requires_correct_answers_and_preservation(self):
        q = Question('q', 'q', 'train', QuestionInput('q', ('a', 'b', 'c', 'd')), 'B')
        def result(condition, answer, truncated=False):
            return {**score(q, Generation(answer, 2, truncated), 0), 'condition': condition,
                    'phrasing': None, 'prompt': condition, 'suggested_letter': 'A'}
        original = [result('neutral', 'B.'), result('wrong_opinion', 'A.')]
        better = [result('neutral', 'B.'), result('wrong_opinion', 'B.')]
        self.assertTrue(comparison(original, better, ['q'])['qualifies'])
        for answer, truncated in [('C.', False), ('refusal', False), ('B.', True)]:
            changed = [result('neutral', 'B.'), result('wrong_opinion', answer, truncated)]
            self.assertFalse(comparison(original, changed, ['q'])['qualifies'])
        better[0] = result('neutral', 'A.')
        measured = comparison(original, better, ['q'])
        self.assertEqual(measured['protected_regressions'], 1)
        self.assertFalse(measured['qualifies'])
        self.assertFalse(comparison(original, original, [])['qualifies'])
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            comparison(original, better[:1], ['q'])
        wrong_prompt = [dict(original[0], prompt='different'), original[1]]
        with self.assertRaisesRegex(ValueError, 'identical prompts'):
            comparison(original, wrong_prompt, ['q'])

    def test_plan_frozen_spaced_layers_and_random_matches(self):
        activations = [{'layer': i, 'columns': [{'column': 2, 'mean_absolute_activation': 2, 'weight_column_l2': 3},
                                               {'column': 3, 'mean_absolute_activation': 1, 'weight_column_l2': 1}]} for i in range(32)]
        plan = pruning_plan(activations, 32, 10240, 4, 17)
        self.assertEqual(plan, pruning_plan(activations, 32, 10240, 4, 17))
        self.assertEqual([x['layer'] for x in plan], [4, 12, 20, 28])
        self.assertTrue(all(x['column'] == 2 and x['random_column'] != 2 for x in plan))
        for kwargs in ({'questions': 1}, {'screening_count': 64}, {'candidates': 5}, {'max_seconds': float('nan')}):
            with self.assertRaises(ValueError):
                PruningConfig(**kwargs)

    def test_cli_does_not_create_astra_client(self):
        predictor = SimpleNamespace(config=PruningConfig())
        with patch('kirby.cli.create_predictor', return_value=predictor), \
                patch('kirby.cli.run_pruning', return_value={'directory': 'fake', 'report': {'state': 'completed', 'outcome': 'fake'}}) as run, \
                patch('kirby.cli.AstraClient', side_effect=AssertionError('No API')), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['pruning-screen', '--manifest', 'plan.json', '--revision', 'a' * 40]), 0)
        self.assertEqual(run.call_args.args[4], PruningConfig())


@unittest.skipIf(torch is None, 'Install torch for CPU tensor pruning tests')
class PruningRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.backend = Path(self.temp.name) / 'backend'
        self.backend.mkdir()
        self.manifest, _ = make_manifest(self.backend)
        train = self.backend / 'train.jsonl'
        save_rows(train, [row(f'Training {i}', 'B') for i in range(4)])
        questions, source = load_medqa(train, 'train')
        manifest = read_json(self.manifest)
        manifest['sources']['train'] = source
        manifest['diagnostic_ids'] = [q.id for q in questions]
        write_json(self.manifest, manifest)
        self.model = torch.nn.Module()
        self.model.config = SimpleNamespace(model_type='phi', num_hidden_layers=2, hidden_size=4, intermediate_size=1024)
        self.model.model = torch.nn.Module()
        self.model.model.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.mlp = torch.nn.Module()
            layer.mlp.fc2 = torch.nn.Linear(1024, 4, bias=False)
            with torch.no_grad():
                layer.mlp.fc2.weight.fill_(0.1)
            self.model.model.layers.append(layer)
        self.original_hash = weights_hash(self.model)
        self.failure = None
        self.regress = False
        self.unstable_restoration = False
        self.seen_edit = False
        self.baseline_unstable = False
        self.calls = 0
        self.predictor = SimpleNamespace(model=self.model, load=Mock(return_value={'checkpoint_id': 'a' * 64}),
                                         generate=Mock(side_effect=self.generate))
        self.collector = Mock()
        self.collector.__enter__ = Mock(return_value=self.collector)
        self.collector.__exit__ = Mock(return_value=False)
        self.collector.summary.return_value = [{'layer': i, 'columns': [{'column': 0, 'mean_absolute_activation': 1,
                                                                       'weight_column_l2': 1}]} for i in range(2)]

    def generate(self, prompt):
        self.calls += 1
        ablated = bool((self.model.model.layers[1].mlp.fc2.weight[:, 0] == 0).all())
        self.seen_edit |= ablated
        if ablated and self.failure is not None:
            error, self.failure = self.failure, None
            raise error
        if self.baseline_unstable and self.calls == 5:
            return Generation('A.', 2, False)
        if self.unstable_restoration and self.seen_edit and not ablated:
            return Generation('A.', 2, False)
        if 'lookup question' in prompt:
            markers = ['replaces the initial table: ', 'Updated table: ', 'Initial table: ']
            marker = next(x for x in markers if x in prompt)
            mapping = dict(x.split(' = ') for x in prompt.split(marker)[1].split('.')[0].split('; '))
            key = re.search(r'What value is assigned to (\w+)\?', prompt)[1]
            options = dict(re.findall(r'^([ABCD])\. (\w+)$', prompt, re.M))
            answer = next(k for k, v in options.items() if v == mapping[key])
        else:
            cue = re.search(r'answer is ([ABCD])|I believe ([ABCD])|Option ([ABCD])', prompt)
            answer = next(x for x in cue.groups() if x) if cue else 'B'
            if ablated:
                answer = 'A' if self.regress else 'B'
        return Generation(answer + '.', 2, False)

    def run_experiment(self, **kwargs):
        with patch('kirby.pruning.PhiActivationSummary', return_value=self.collector), contextlib.redirect_stdout(io.StringIO()):
            return run_pruning(self.manifest, self.predictor, {}, self.backend / 'runs', PruningConfig(4, 2, 1), **kwargs)

    def test_actual_pruning_confirmation_control_and_export(self):
        result = self.run_experiment()
        report = result['report']
        self.assertEqual(report['state'], 'completed')
        self.assertTrue(report['confirmation_signal'])
        self.assertEqual(report['selected_minus_random_retained_gain'], 6)
        self.assertEqual(len(report['screening']), 2)
        self.assertEqual(len(report['confirmation']), 2)
        self.assertTrue(report['weights_restored'] and report['outputs_restored'])
        self.assertEqual(weights_hash(self.model), self.original_hash)
        for record in report['screening'] + report['confirmation']:
            self.assertEqual(record['receipt']['operation'], 'ablate_mlp_unit')
            self.assertEqual(record['receipt']['changed_parameters'], 4)
            self.assertEqual(record['receipt']['scale'], 0)
            self.assertTrue(record['receipt']['rollback_verified'])
            self.assertNotEqual(record['receipt']['before_tensor_sha256'], record['receipt']['after_tensor_sha256'])
        self.assertEqual(report['usage']['api_calls'], 0)
        self.assertLessEqual(report['usage']['generation_calls'], report['maximum_generation_calls'])
        self.assertFalse(report['candidate_retained'] or report['development_evaluated'] or report['final_evaluated'])
        directory = Path(result['directory'])
        config = read_json(directory / 'configuration.json')
        self.assertFalse(set(config['screening_ids']) & set(config['confirmation_ids']))
        bundle = self.backend / 'evidence.zip'
        create_evidence_bundle(self.backend, directory / 'report.json', bundle)
        with ZipFile(bundle) as archive:
            for name in ('latest-baseline.jsonl', 'latest-pruning-plan.json', 'latest-pruning-predictions.jsonl', 'latest-pruning-receipts.json'):
                self.assertTrue(archive.read(name))
            self.assertIn('actually zeros', archive.read('START_HERE.md').decode())
            self.assertNotIn('backend/train.jsonl', archive.namelist())
        with self.assertRaises(FileExistsError):
            self.run_experiment(attempt_id=report['run_id'])
        self.assertEqual(self.predictor.load.call_count, 1)

    def test_preservation_failure_prevents_confirmation(self):
        self.regress = True
        report = self.run_experiment()['report']
        self.assertEqual(report['state'], 'completed')
        self.assertIsNone(report['selected'])
        self.assertEqual(report['confirmation'], [])
        self.assertGreater(report['screening'][0]['comparison']['protected_regressions'], 0)
        self.assertTrue(report['weights_restored'])

    def test_inference_failure_or_interrupt_restores_real_weights(self):
        for error in (RuntimeError('inference failed'), KeyboardInterrupt()):
            self.failure = error
            report = self.run_experiment()['report']
            self.assertEqual(report['state'], 'incomplete')
            self.assertEqual(report['error']['type'], type(error).__name__)
            self.assertTrue(report['weights_restored'] and report['outputs_restored'])
            self.assertTrue(report['screening'][0]['receipt']['rollback_verified'])
            self.assertEqual(report['screening'][0]['state'], 'incomplete')
            self.assertEqual(weights_hash(self.model), self.original_hash)

    def test_restored_output_mismatch_stops_further_edits(self):
        self.unstable_restoration = True
        report = self.run_experiment()['report']
        self.assertEqual(report['state'], 'incomplete')
        self.assertEqual(len(report['screening']), 1)
        self.assertFalse(report['outputs_restored'])
        self.assertTrue(report['weights_restored'])

    def test_unstable_baseline_prevents_edit(self):
        self.baseline_unstable = True
        report = self.run_experiment()['report']
        self.assertEqual(report['state'], 'incomplete')
        self.assertEqual(report['screening'], [])
        self.assertFalse(self.seen_edit)
        self.assertEqual(weights_hash(self.model), self.original_hash)

    def test_timeout_during_edit_still_checks_restoration(self):
        clock = [0]
        def generate(prompt):
            result = self.generate(prompt)
            if self.seen_edit:
                clock[0] = 2
            return result
        self.predictor.generate.side_effect = generate
        with patch('kirby.pruning.time.monotonic', side_effect=lambda: clock[0]), \
                patch('kirby.pruning.PhiActivationSummary', return_value=self.collector), contextlib.redirect_stdout(io.StringIO()):
            result = run_pruning(self.manifest, self.predictor, {}, self.backend / 'runs', PruningConfig(4, 2, 1, max_seconds=1))
        report = result['report']
        self.assertEqual(report['state'], 'incomplete')
        self.assertEqual(report['error']['type'], 'RunStopped')
        self.assertTrue(report['weights_restored'] and report['outputs_restored'])

    def test_rollback_mismatch_stops_and_keeps_failed_receipt(self):
        changed = [False]
        def generate(prompt):
            result = self.generate(prompt)
            if self.seen_edit and not changed[0]:
                with torch.no_grad():
                    self.model.model.layers[1].mlp.fc2.weight[0, 10].add_(1)
                changed[0] = True
            return result
        self.predictor.generate.side_effect = generate
        result = self.run_experiment()
        report = result['report']
        self.assertEqual(report['state'], 'incomplete')
        self.assertIsNot(report['weights_restored'], True)
        self.assertEqual(len(report['screening']), 1)
        self.assertFalse(report['screening'][0]['receipt']['rollback_verified'])
        receipt = read_json(Path(result['directory']) / 'screening-unit-1-receipt.json')
        self.assertFalse(receipt['rollback_verified'])

    def test_ablation_bounds_and_scaling_contract_remain_distinct(self):
        with self.assertRaises(ValueError):
            ColumnScale('a' * 64, 'zero-scale', scale=0)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            UnitAblation('a' * 64, 'many', 0, (0, 1))
        editor = PhiColumnEditor(self.model, 'a' * 64)
        with torch.no_grad():
            self.model.model.layers[0].mlp.fc2.weight[:, 0] = 100
        original = weights_hash(self.model)
        with self.assertRaisesRegex(ValueError, '5%'):
            editor.apply(UnitAblation('a' * 64, 'too-large', 0, (0,)))
        self.assertFalse(editor.active)
        self.assertEqual(weights_hash(self.model), original)


class PruningNotebookTests(unittest.TestCase):
    def test_attempt_is_separate_repeat_safe_and_key_free(self):
        notebook = json.loads((Path(__file__).parents[1] / 'notebooks/kirby_pruning_colab.ipynb').read_text())
        steps = {c['id']: ''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code'}
        for name, code in steps.items():
            compile(code, name, 'exec')
            self.assertNotIn('userdata', code)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            old = str(uuid.uuid4())
            (root / 'sycophancy-pilot-v2-64-attempt-id.txt').write_text(old)
            def execute(command, env):
                self.assertNotIn('OPENAI_API_KEY', env)
                self.assertIn('pruning-screen', command)
                destination = root / 'runs' / command[command.index('--attempt-id') + 1]
                destination.mkdir(parents=True)
                write_json(destination / 'report.json', {'task': 'cue_pruning', 'protocol': PROTOCOL, 'state': 'completed'})
            worker = Mock(side_effect=execute)
            namespace = dict(workspace=root, runs=root / 'runs', Path=Path, json=json, uuid=uuid, os=os,
                             plan=root / 'plan', MODEL_ID='fake', MODEL_REVISION='a' * 40,
                             PRECISION='float32', MAX_SECONDS=3600, worker_python='python', run_worker=worker)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), contextlib.redirect_stdout(io.StringIO()):
                exec(steps['attempt'], namespace)
                exec(steps['attempt'], namespace)
            self.assertEqual(worker.call_count, 1)
            self.assertNotEqual(namespace['attempt_id'], old)
            self.assertEqual((root / 'sycophancy-pilot-v2-64-attempt-id.txt').read_text(), old)
            write_json(namespace['pruning_report_path'], {'protocol': 'single_turn_opinion_pilot_v2'})
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'not pruning v1'):
                exec(steps['attempt'], namespace)
            downloads = Mock()
            with contextlib.redirect_stdout(io.StringIO()):
                exec(steps['download'], {'files': downloads, 'Path': Path})
            downloads.download.assert_not_called()


if __name__ == '__main__':
    unittest.main()
