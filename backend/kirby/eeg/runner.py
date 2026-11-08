"""One exclusive run: train -> blind diagnosis -> dev selection -> locked final."""

import copy
import importlib.metadata
import os
import platform
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from ..records import canonical_json, digest, file_hash, read_json, utc_now, write_json
from . import PROTOCOL
from .astra_eeg import RequestBudget
from .baselines import integrated_gradients, operations, rule_diagnosis
from .data import load_split, prepare_subject
from .edits import apply_edit
from .eegnet import fold_model, load_model, model_hash, save_model
from .evaluate import accept_repair, evaluate
from .packet import build_packet
from .shortcuts import input_filter, specification, variants
from .train import fit


def record_evaluation(path, metrics, predictions, data):
    write_json(path.with_suffix('.json'), metrics)
    with path.with_suffix('.jsonl').open('w', encoding='utf-8') as stream:
        for i, trial in enumerate(data['ids']):
            stream.write(canonical_json({'trial_id': str(trial), 'label': int(data['y'][i]),
                                         'predictions': {k: int(v[i]) for k, v in predictions.items()}}) + '\n')


def score_diagnosis(response, spec):
    predicted, expected = response['diagnosis'], spec['kind']
    electrodes = set(response.get('electrodes', [])); target = set(spec['electrodes'])
    band = response.get('band_hz', [0, 0]); truth = spec['band_hz']
    overlap = max(0.0, min(band[1], truth[1]) - max(band[0], truth[0]))
    union = max(band[1], truth[1]) - min(band[0], truth[0])
    return {'expected': expected, 'predicted': predicted, 'correct': predicted == expected,
            'abstained': predicted == 'uncertain',
            'false_positive': expected == 'clean' and predicted not in ('clean', 'uncertain'),
            'electrode_jaccard': len(electrodes & target) / len(electrodes | target) if target else None,
            'band_iou': overlap / union if target and union > 0 else None}


def select_edits(original, versions, data, config, folder, proposal):
    """Evaluate disposable copies; only accepted copies become the active model."""
    folder.mkdir(parents=True, exist_ok=False)
    current = copy.deepcopy(original)
    baseline, initial_predictions = evaluate(original, versions, data['y'])
    current_metrics = baseline
    receipts = []
    original_hash = model_hash(original)
    for attempt in range(config.attempts):
        feedback = {'original': baseline, 'current': current_metrics, 'previous_attempts': receipts,
                    'remaining_attempts': config.attempts - attempt,
                    'original_temporal_ranges': [
                        [float(w.min()), float(w.max())] for w in original.temporal.weight.detach().cpu()]}
        operation = proposal(attempt, current, feedback)
        if operation is None:
            break
        before = model_hash(current)
        receipt = {'attempt': attempt + 1, 'operation': operation, 'active_before': before, 'accepted': False}
        try:
            candidate, edit_receipt = apply_edit(current, original, operation, config.max_changed_share)
            metrics, predictions = evaluate(candidate, versions, data['y'])
            record_evaluation(folder / f'attempt-{attempt + 1}', metrics, predictions, data)
            accepted = accept_repair(baseline, current_metrics, metrics, config.max_clean_drop)
            receipt.update(edit_receipt)
            receipt.update({'accepted': accepted, 'development': metrics,
                            'changed_predictions_from_original': {
                                k: sum(a != b for a, b in zip(predictions[k], initial_predictions[k])) for k in predictions}})
            if model_hash(current) != before or model_hash(original) != original_hash:
                raise RuntimeError('Source checkpoint changed during candidate evaluation')
            if accepted:
                current, current_metrics = candidate, metrics
        except (ValueError, TypeError, KeyError, IndexError) as error:
            receipt['rejection'] = type(error).__name__ + ': ' + str(error)[:240]
        if model_hash(original) != original_hash or (not receipt['accepted'] and model_hash(current) != before):
            raise RuntimeError('Exact rollback/source integrity failed')
        receipt['active_after'] = model_hash(current)
        receipt['source_unchanged'] = True
        receipts.append(receipt)
        write_json(folder / 'receipts.json', receipts)
    write_json(folder / 'receipts.json', receipts)
    return current, current_metrics, receipts


def run(config, output, data_root, *, run_id=None, client_factory=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    identity = str(uuid.UUID(run_id)) if run_id else str(uuid.uuid4())
    root = Path(output) / identity
    root.mkdir(parents=True, exist_ok=False)  # A run ID can never make a second paid call/final evaluation.
    config_dict = asdict(config)
    report = {'protocol': PROTOCOL, 'run_id': identity, 'status': 'running', 'started_at': utc_now(),
              'config': config_dict, 'config_sha256': digest(config_dict), 'models': [],
              'final_evaluation_started': False, 'api_calls': 0, 'estimated_usd': 0,
              'limitations': ['Synthetic planted artifacts; no clinical validity claim.',
                              'Clean means no planted artifact, not artifact-free EEG.',
                              'A/B and repeated seeds are paired, not independent subjects.']}
    write_json(root / 'config.json', config_dict)
    write_json(root / 'report.json', report)
    budget = RequestBudget(config, root / 'requests', client_factory)
    try:
        source_dir = root / 'source'; source_dir.mkdir()
        package_dir = source_dir / 'kirby'; (package_dir / 'eeg').mkdir(parents=True)
        for source in Path(__file__).parent.glob('*.py'):
            shutil.copy2(source, package_dir / 'eeg' / source.name)
        for name in ('__init__.py', 'astra.py', 'records.py', 'patches.py'):
            shutil.copy2(Path(__file__).parents[1] / name, package_dir / name)
        project = Path(__file__).parents[2] / 'pyproject.toml'
        if project.is_file():
            shutil.copy2(project, source_dir / project.name)
        write_json(root / 'source-hashes.json', {p.relative_to(source_dir).as_posix(): file_hash(p)
                                               for p in sorted(source_dir.rglob('*')) if p.is_file()})
        packages = {}
        for name in ('torch', 'numpy', 'scipy', 'moabb', 'mne', 'openai', 'jsonschema', 'matplotlib'):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        write_json(root / 'environment.json', {'python': platform.python_version(), 'packages': packages,
                                              'platform': platform.platform(), 'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG']})
        prepared = {}
        for subject in config.subjects:
            directory = Path(data_root) / f'subject-{subject:02d}-split-{config.split_seed}'
            manifest = prepare_subject(subject, directory, config.split_seed)
            prepared[subject] = directory
            write_json(root / f'data-subject-{subject:02d}.json', manifest)
            train = load_split(directory, 'train'); validation = load_split(directory, 'validation')
            dev = load_split(directory, 'development')
            for seed in config.seeds:
                clean_accuracy = None
                for kind in ('clean', *(k for k in config.conditions if k != 'clean')):
                    model_id = str(uuid.uuid4())
                    folder = root / 'models' / model_id; folder.mkdir(parents=True)
                    spec = specification(kind, config)
                    print(f'Training subject {subject}, seed {seed}, condition {kind}', flush=True)
                    training = variants(train, spec, seed * 100 + 1)
                    validation_versions = variants(validation, spec, seed * 100 + 2)
                    trained, history = fit(training['original'], train['y'], validation_versions['original'], validation['y'], config, seed)
                    write_json(folder / 'training.json', history)
                    save_model(trained, folder / 'trained.npz')
                    model, folding = fold_model(trained, dev['x'])
                    write_json(folder / 'folding.json', folding)
                    save_model(model, folder / 'original.npz')
                    development_versions = variants(dev, spec, seed * 100 + 3)
                    metrics, predictions = evaluate(model, development_versions, dev['y'])
                    record_evaluation(folder / 'original-development', metrics, predictions, dev)
                    if kind == 'clean':
                        clean_accuracy = metrics['clean']['accuracy']
                    gate = clean_accuracy >= config.clean_floor and (kind == 'clean' or metrics['reliance_gap'] >= config.min_reliance)
                    entry = {'model_id': model_id, 'subject': subject, 'seed': seed, 'ground_truth': spec,
                             'original_hash': model_hash(model), 'development': metrics, 'diagnoses': {}, 'methods': {},
                             'gate': {'passed': gate, 'paired_clean_accuracy': clean_accuracy,
                                      'clean_floor': config.clean_floor, 'reliance_gap': metrics['reliance_gap'],
                                      'minimum_reliance': config.min_reliance if kind != 'clean' else None}}
                    report['models'].append(entry)
                    write_json(root / 'report.json', report)
                    del trained, model, training, validation_versions, development_versions

        # All first diagnoses precede all feedback. Hidden condition names stay only in local records.
        for entry in report['models']:
            if not entry['gate']['passed']:
                continue
            folder = root / 'models' / entry['model_id']
            model = load_model(folder / 'original.npz')
            response = rule_diagnosis(model)
            entry['diagnoses']['rule'] = {'response': response, 'score': score_diagnosis(response, entry['ground_truth'])}
            if config.extended_baselines:
                dev = load_split(prepared[entry['subject']], 'development')
                response = integrated_gradients(model, variants(dev, entry['ground_truth'], entry['seed'] * 100 + 3)['original'])
                entry['diagnoses']['integrated_gradients'] = {'response': response, 'score': score_diagnosis(response, entry['ground_truth'])}
            if config.astra:
                for condition in config.evidence_conditions:
                    packet = build_packet(model, entry['model_id'], condition)
                    response = budget.request(packet, str(uuid.uuid4()))
                    entry['diagnoses']['astra_' + condition] = {'response': response, 'score': score_diagnosis(response, entry['ground_truth'])}
                    write_json(root / 'report.json', report)

        for entry in report['models']:
            if not entry['gate']['passed']:
                continue
            folder = root / 'models' / entry['model_id']
            original = load_model(folder / 'original.npz')
            directory = prepared[entry['subject']]
            dev = load_split(directory, 'development')
            development_versions = variants(dev, entry['ground_truth'], entry['seed'] * 100 + 3)

            def retain(name, model, metrics, receipts=(), resource='bounded_weight_edit'):
                path = folder / (name + '.npz')
                save_model(model, path)
                entry['methods'][name] = {'checkpoint': path.relative_to(root).as_posix(), 'file_sha256': file_hash(path),
                                          'model_sha256': model_hash(model), 'development': metrics,
                                          'attempted_edits': len(receipts), 'accepted_edits': sum(r['accepted'] for r in receipts),
                                          'resource_class': resource}

            retain('no_edit', original, entry['development'], resource='no_edit')
            filtered, filtered_predictions = evaluate(original, development_versions, dev['y'], transform=input_filter)
            record_evaluation(folder / 'input-filter-development', filtered, filtered_predictions, dev)
            retain('input_filter', original, filtered, resource='input_preprocessing')
            for method in ('random', 'rule'):
                choices = operations(original, method, entry['seed'], config.attempts)
                model, metrics, receipts = select_edits(original, development_versions, dev, config, folder / method,
                                                       lambda attempt, current, feedback, choices=choices: choices[attempt] if attempt < len(choices) else None)
                retain(method, model, metrics, receipts)
            if config.astra:
                for condition in config.evidence_conditions:
                    name = 'astra_' + condition
                    if entry['diagnoses'][name]['response']['diagnosis'] in ('clean', 'uncertain'):
                        retain(name, original, entry['development'])
                        entry['methods'][name]['stop_reason'] = 'First diagnosis clean or uncertain; no repair attempted'
                        continue
                    def propose(attempt, current, feedback, condition=condition):
                        packet = build_packet(current, entry['model_id'], condition, task='repair', feedback=feedback)
                        # Preserve the configured tighter bounds if an operator selected them.
                        packet['limits']['attempts'] = config.attempts
                        packet['limits']['cumulative_changed_share'] = config.max_changed_share
                        packet['limits']['max_clean_drop'] = config.max_clean_drop
                        packet.pop('evidence_sha256')
                        packet['evidence_sha256'] = digest(packet)
                        return budget.request(packet, str(uuid.uuid4()))['operation']
                    model, metrics, receipts = select_edits(original, development_versions, dev, config, folder / name, propose)
                    retain(name, model, metrics, receipts)
            if config.extended_baselines:
                train = load_split(directory, 'train'); validation = load_split(directory, 'validation')
                model, history = fit(train['x'], train['y'], validation['x'], validation['y'], config, entry['seed'], initial=original, last_layer=True)
                write_json(folder / 'last-layer-training.json', history)
                metrics, predictions = evaluate(model, development_versions, dev['y'])
                record_evaluation(folder / 'last-layer-development', metrics, predictions, dev)
                retain('last_layer', model, metrics, resource='clean_labeled_retraining_outside_edit_cap')
                clean_entry = next(e for e in report['models'] if e['subject'] == entry['subject'] and e['seed'] == entry['seed'] and e['ground_truth']['kind'] == 'clean')
                clean_model = load_model(root / 'models' / clean_entry['model_id'] / 'original.npz')
                metrics, predictions = evaluate(clean_model, development_versions, dev['y'])
                record_evaluation(folder / 'full-clean-retrain-development', metrics, predictions, dev)
                retain('full_clean_retrain', clean_model, metrics, resource='clean_labeled_retraining_outside_edit_cap')
            write_json(root / 'report.json', report)

        selected = [{'model_id': e['model_id'], 'methods': e['methods']} for e in report['models'] if e['methods']]
        lock = {'config_sha256': digest(config_dict), 'selected': selected, 'frozen_at': utc_now()}
        write_json(root / 'selection-lock.json', lock)
        lock_hash = file_hash(root / 'selection-lock.json')
        if selected:
            with (root / 'final-started.json').open('x', encoding='utf-8') as marker:
                marker.write(canonical_json({'selection_lock_sha256': lock_hash, 'started_at': utc_now()}))
            report['final_evaluation_started'] = True
            write_json(root / 'report.json', report)
            for entry in report['models']:
                if not entry['methods']:
                    continue
                if file_hash(root / 'selection-lock.json') != lock_hash:
                    raise RuntimeError('Selection lock changed')
                final = load_split(prepared[entry['subject']], 'final', final_authorized=True)
                final_versions = variants(final, entry['ground_truth'], entry['seed'] * 100 + 4)
                for name, method in entry['methods'].items():
                    path = root / method['checkpoint']
                    if file_hash(path) != method['file_sha256']:
                        raise RuntimeError('Selected checkpoint changed before final evaluation')
                    model = load_model(path)
                    metrics, predictions = evaluate(model, final_versions, final['y'], transform=input_filter if name == 'input_filter' else None)
                    record_evaluation(root / 'models' / entry['model_id'] / (name + '-final'), metrics, predictions, final)
                    method['final'] = metrics
                    write_json(root / 'report.json', report)
            report['status'] = 'completed' if all(e['gate']['passed'] for e in report['models']) else 'completed_with_exclusions'
        else:
            report['status'] = 'gates_failed'
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        report['error_type'] = type(error).__name__
        raise
    finally:
        report.update({'finished_at': utc_now(), 'api_calls': budget.calls, 'estimated_usd': budget.estimated_usd,
                       'unknown_usage_requests': sum(not a.get('usage_known') for a in budget.audit)})
        write_json(root / 'report.json', report)
        from .report import export
        export(root)
    return report
