"""Portable evidence, readable comparisons, and conservative uncertainty summaries."""

import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ..records import file_hash, read_json, write_json


def wilson(correct, total):
    if not total:
        return None
    z = 1.959963984540054; p = correct / total
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0, center - half), min(1, center + half)]


def summaries(report):
    import numpy as np
    methods = sorted({k for e in report['models'] for k in e['diagnoses']})
    diagnosis = {}
    for method in methods:
        rows = [e['diagnoses'][method]['score'] for e in report['models'] if method in e['diagnoses']]
        clean = [r for r in rows if r['expected'] == 'clean']
        answered = [r for r in rows if not r['abstained']]
        diagnosis[method] = {'correct': sum(r['correct'] for r in rows), 'total': len(rows),
                             'answered': len(answered), 'correct_when_answered': sum(r['correct'] for r in answered),
                             'false_positives': sum(r['false_positive'] for r in clean), 'clean_total': len(clean),
                             'wilson95_descriptive_only': wilson(sum(r['correct'] for r in rows), len(rows)),
                             'interval_caveat': 'Repeated seeds/conditions share subjects; not a subject-generalization interval.'}
    deltas = {}
    for entry in report['models']:
        baseline = entry['methods'].get('no_edit', {}).get('final')
        if baseline is None or entry['ground_truth']['kind'] == 'clean':
            continue
        for name, method in entry['methods'].items():
            if 'final' in method:
                key = entry['ground_truth']['kind'] + '/' + name
                deltas.setdefault(key, {}).setdefault(entry['subject'], []).append(method['final']['swapped']['accuracy'] - baseline['swapped']['accuracy'])
    effects = {}
    for key, subjects in deltas.items():
        means = np.array([np.mean(values) for values in subjects.values()])
        ci = None
        if len(means) >= 2:
            rng = np.random.default_rng(17)
            ci = np.quantile(rng.choice(means, size=(5000, len(means)), replace=True).mean(1), [0.025, 0.975]).tolist()
        effects[key] = {'mean_swapped_accuracy_delta': float(means.mean()), 'subjects': len(means),
                        'subject_bootstrap95': ci, 'per_subject_seed_mean': {str(k): float(np.mean(v)) for k, v in subjects.items()},
                        'caveat': 'Exploratory interval; very few subjects provide weak uncertainty estimates.'}
    return {'diagnosis': diagnosis, 'repair': effects}


def plot_report(root, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .eegnet import load_model
    from .packet import signal_views
    import numpy as np
    eligible = [e for e in report['models'] if any('final' in m for m in e['methods'].values())]
    if not eligible:
        return
    rows = [(e, name, m) for e in eligible for name, m in e['methods'].items() if 'final' in m]
    fig, axis = plt.subplots(figsize=(12, max(4, len(rows) * 0.3)))
    values = np.array([[m['final'][v]['accuracy'] for v in ('original', 'clean', 'swapped', 'uniform')] for e, name, m in rows])
    image = axis.imshow(values, vmin=0, vmax=1, aspect='auto', cmap='viridis')
    axis.set_xticks(range(4), ['Original', 'No planted artifact', 'Swapped', 'Uniform'])
    axis.set_yticks(range(len(rows)), [f"S{e['subject']} seed{e['seed']} {e['ground_truth']['kind']} / {name}" for e, name, m in rows], fontsize=7)
    axis.set_title('Locked final session accuracy; repeated rows share trials')
    fig.colorbar(image, ax=axis); fig.tight_layout(); fig.savefig(root / 'final-accuracy.png', dpi=150); plt.close(fig)
    entry = next((e for e in eligible if e['ground_truth']['kind'] != 'clean'), eligible[0])
    model = load_model(root / 'models' / entry['model_id'] / 'original.npz')
    view = signal_views(model)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    for i, gain in enumerate(view['temporal_gain']):
        axes[0].plot(view['frequencies_hz'], gain, label=str(i))
    axes[0].set(xlabel='Frequency (Hz)', ylabel='Gain', title='Original temporal filter responses (not proof of reliance)'); axes[0].legend(ncol=8)
    spatial = model.spatial.weight.detach().numpy()[:, 0, :, 0]
    from .config import CHANNELS
    axes[1].imshow(spatial, aspect='auto', cmap='RdBu_r')
    axes[1].set_xticks(range(22), CHANNELS, rotation=90); axes[1].set(ylabel='Spatial filter', title='Signed spatial weights')
    fig.suptitle(f"Example: subject {entry['subject']}, seed {entry['seed']}, {entry['ground_truth']['kind']}")
    fig.tight_layout(); fig.savefig(root / 'filter-views.png', dpi=150); plt.close(fig)


def export(directory):
    root = Path(directory)
    report = read_json(root / 'report.json')
    summary = summaries(report)
    write_json(root / 'summary.json', summary)
    lines = ['# Kirby EEG evidence', '', f"Run: {report['run_id']}", f"Status: **{report['status']}**", '',
             f"Astra requests: {report['api_calls']}; estimated token cost: ${report['estimated_usd']:.4f} (not an invoice).", '',
             'Only models passing the frozen development gates enter diagnosis/repair/final evaluation. Excluded models remain in report.json.',
             'A and B are paired audits of the same models. Seeds are not independent subjects. Clean means no planted artifact.', '',
             '| Subject | Seed | Training condition | Gate | Clean reference accuracy | Reliance gap |',
             '|---|---|---|---|---|---|']
    for e in report['models']:
        g = e['gate']
        lines.append(f"| {e['subject']} | {e['seed']} | {e['ground_truth']['kind']} | {g['passed']} | {g['paired_clean_accuracy']:.3f} | {g['reliance_gap']:.3f} |")
    lines += ['', '## Blind first diagnosis', '', '| Method | Correct / all | Answered / all | False positives / clean |', '|---|---|---|---|']
    for name, s in summary['diagnosis'].items():
        lines.append(f"| {name} | {s['correct']}/{s['total']} | {s['answered']}/{s['total']} | {s['false_positives']}/{s['clean_total']} |")
    lines += ['', '## Locked final results', '', '| Subject / seed / training | Method | Original | Clean | Swapped | Uniform | Flip rate | Accepted edits |', '|---|---|---|---|---|---|---|---|']
    for e in report['models']:
        for name, m in e['methods'].items():
            if 'final' not in m:
                continue
            f = m['final']; values = ' | '.join(f"{f[v]['accuracy']:.3f}" for v in ('original', 'clean', 'swapped', 'uniform'))
            lines.append(f"| {e['subject']} / {e['seed']} / {e['ground_truth']['kind']} | {name} | {values} | {f['artifact_flip_rate']:.3f} | {m['accepted_edits']} |")
    lines += ['', 'Use report.json for counts, per-class accuracy, kappa, development gates and resource classes; summary.json for uncertainty and paired deltas.',
              'models/*/*-final.jsonl holds per-trial predictions. requests/* contains exact blind packets, requests, responses and usage.',
              'Repair selection requires absolute swapped gain, lower paired flip rate, clean accuracy within 2 points of the ORIGINAL model, and preserved worst-condition accuracy.',
              'Input filtering is scored unconditionally. Random/rule controls use the same acceptance test and attempt/cumulative parameter caps; they are not norm-matched to every Astra operation.',
              'Optional retraining uses clean labeled training data and lies outside the 25% edit cap. IG uses development inputs and is not a weights-only comparator.',
              'No model or method is selected using final performance. Do not tune against this final session and call it held out again.',
              'Check status before interpreting results. Failed/interrupted exports are partial, not successful experiments.',
              'The ZIP contains small model checkpoints and source snapshots, but no raw EEG, credentials, or environment variable dumps.', '']
    (root / 'START_HERE.md').write_text('\n'.join(lines), encoding='utf-8')
    try:
        plot_report(root, report)
    except Exception as error:
        write_json(root / 'plot-error.json', {'error_type': type(error).__name__})
    included = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.md', '.py', '.toml', '.png', '.npz') and p.name != 'evidence-index.json')
    write_json(root / 'evidence-index.json', {'run_id': report['run_id'], 'files': [
        {'path': p.relative_to(root).as_posix(), 'sha256': file_hash(p), 'bytes': p.stat().st_size} for p in included]})
    destination = root / ('kirby-eeg-evidence-' + report['run_id'] + '.zip')
    with ZipFile(destination, 'w', ZIP_DEFLATED) as archive:
        for p in included + [root / 'evidence-index.json']:
            archive.write(p, p.relative_to(root).as_posix())
    return destination
