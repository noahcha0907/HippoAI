"""EEG-only CLI; importing it never loads a language-model checkpoint."""

import argparse
import json
from pathlib import Path

from .config import EEGConfig, KINDS


def main(argv=None):
    parser = argparse.ArgumentParser(description='Kirby: blinded EEG shortcut diagnosis and bounded repair')
    commands = parser.add_subparsers(dest='command', required=True)
    experiment = commands.add_parser('run')
    experiment.add_argument('--output', type=Path, default=Path('eeg-runs'))
    experiment.add_argument('--data-root', type=Path, default=Path('eeg-data'))
    experiment.add_argument('--run-id')
    experiment.add_argument('--subjects', nargs='+', type=int, default=[1])
    experiment.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    experiment.add_argument('--conditions', nargs='+', choices=KINDS, default=['clean', 'eye_blink'])
    experiment.add_argument('--evidence-conditions', nargs='+', choices=['A', 'B'], default=['A', 'B'])
    experiment.add_argument('--epochs', type=int, default=300)
    experiment.add_argument('--patience', type=int, default=30)
    experiment.add_argument('--attempts', type=int, default=5)
    experiment.add_argument('--repair-policy', choices=['diagnosis_gated', 'feedback_guided'], default='feedback_guided',
                            help='Allow development-guided proposals after any blind diagnosis, or reproduce the legacy diagnosis gate')
    experiment.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    experiment.add_argument('--max-api-calls', type=int, default=72)
    experiment.add_argument('--max-estimated-usd', type=float, default=50)
    experiment.add_argument('--no-astra', action='store_true', help='Explicit baseline-only experiment; no diagnosis/repair calls')
    experiment.add_argument('--extended-baselines', action='store_true', help='Add IG, clean-label last-layer and full clean-training comparators')
    bundle = commands.add_parser('export', help='Rebuild evidence without inference or API calls')
    bundle.add_argument('directory', type=Path)
    args = parser.parse_args(argv)
    if args.command == 'export':
        from .report import export
        print(export(args.directory)); return 0
    from .runner import run
    config = EEGConfig(subjects=tuple(args.subjects), seeds=tuple(args.seeds), conditions=tuple(args.conditions),
                       evidence_conditions=tuple(args.evidence_conditions), epochs=args.epochs, patience=args.patience,
                       attempts=args.attempts, repair_policy=args.repair_policy, device=args.device, max_api_calls=args.max_api_calls,
                       max_estimated_usd=args.max_estimated_usd, astra=not args.no_astra,
                       extended_baselines=args.extended_baselines)
    result = run(config, args.output, args.data_root, run_id=args.run_id)
    print(json.dumps({'run_id': result['run_id'], 'status': result['status'], 'api_calls': result['api_calls'],
                      'report': str(args.output / result['run_id'] / 'report.json')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
