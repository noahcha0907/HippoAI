"""Build the pruning launcher from the tested key-free pilot bootstrap."""

import json
from pathlib import Path

from build_pilot_notebook import main as build_pilot


def main():
    build_pilot()
    notebooks = Path(__file__).resolve().parents[1] / 'notebooks'
    notebook = json.loads((notebooks / 'kirby_sycophancy_colab.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        text = ''.join(cell['source'])
        if cell['cell_type'] == 'markdown':
            text = ('# Kirby: reversible single-unit pruning\n\n'
                    'Upload the new kirby_colab.zip and run the three cells in order in the existing Colab GPU workspace. '
                    'This actually zeros one MLP output column at a time and restores it after measurement. '
                    'Four activation-ranked units are screened on eight training questions. A qualifying candidate and one '
                    'seeded random unit are confirmed on the other 56 training questions. Lookup controls are reused. '
                    'If no candidate passes the screening guards, confirmation is skipped. Maximum 1,736 generation calls; '
                    'the actual count depends on neutral errors and screening results. No Astra/API key, cumulative edits, '
                    'retained checkpoint, development or final evaluation.\n')
        elif cell['id'] == 'resume':
            text = text.replace('required = {"backend/kirby/sycophancy.py", "backend/kirby/evidence.py"}',
                                'required = {"backend/kirby/pruning.py", "backend/kirby/patches.py", "backend/kirby/evidence.py"}')
            text = text.replace('new ZIP containing the sycophancy pilot', 'new ZIP containing the pruning experiment')
            text = text.replace('Ready: pilot v2, 64 training questions, matched distractors, mention and nonidentity lookup controls. No Astra, weight edits, or final evaluation.',
                                'Ready: reversible pruning; four single units, 8 screening / 56 confirmation questions. No Astra or final evaluation.')
        else:
            text = text.replace('sycophancy-pilot-v2-64-attempt-id.txt', 'cue-pruning-v1-64-attempt-id.txt')
            text = text.replace("'sycophancy-pilot'", "'pruning-screen'")
            text = text.replace('single_turn_opinion_pilot_v2', 'single_unit_cue_pruning_v1')
            text = text.replace('pilot_report_path', 'pruning_report_path')
            text = text.replace("'--allow-download', '--questions', '64', '--max-seconds'",
                                "'--allow-download', '--questions', '64', '--screening-count', '8', '--candidates', '4', '--max-seconds'")
            text = text.replace("('state', 'outcome', 'metrics', 'evidence_control', 'weights_unchanged', 'usage', 'error')",
                                "('state', 'outcome', 'selected', 'confirmation_signal', 'screening', 'confirmation', 'weights_restored', 'outputs_restored', 'usage', 'error')")
            text = text.replace('kirby-sycophancy-evidence-', 'kirby-pruning-evidence-')
            text = text.replace('latest-neutral.jsonl and latest-opinions.jsonl', 'latest-baseline.jsonl and latest-pruning-predictions.jsonl')
            text = text.replace('pilot v2', 'pruning v1').replace('pilot', 'pruning').replace('Pilot', 'Pruning')
        cell['source'] = text.splitlines(keepends=True)
    (notebooks / 'kirby_pruning_colab.ipynb').write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
