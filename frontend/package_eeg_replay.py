"""Build a public replay from an evidence folder; include no checkpoints or API packets."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.evidence.resolve()
    index = json.loads((root / 'evidence-index.json').read_text())
    hashes = {}
    for row in index['files']:
        path = (root / row['path']).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Invalid evidence: ' + row['path'])
        hashes[row['path']] = row['sha256']
    report = json.loads((root / 'report.json').read_text())
    assert report['run_id'] == index['run_id'] and report['status'] == 'completed'
    receipts = {}
    included = {'report.json': hashes['report.json']}
    for model in report['models']:
        for method in model['methods']:
            relative = f'models/{model["model_id"]}/{method}/receipts.json'
            path = root / relative
            if path.exists():
                receipts.setdefault(model['model_id'], {})[method] = json.loads(path.read_text())
                included[relative] = hashes[relative]
    replay = {'format': 'hippo_eeg_replay_v1', 'report': report, 'receipts': receipts}
    args.output.mkdir(parents=True, exist_ok=True)
    packed = (json.dumps(replay, ensure_ascii=True, separators=(',', ':')) + '\n').encode()
    (args.output / 'replay.json').write_bytes(packed)
    provenance = {'run_id': report['run_id'], 'source_index_sha256': hashlib.sha256((root / 'evidence-index.json').read_bytes()).hexdigest(),
                  'source_files_verified_at_packaging': len(hashes), 'included_source_files': included,
                  'replay_sha256': hashlib.sha256(packed).hexdigest(),
                  'note': 'Hashes establish consistency with the supplied archive, not independent authenticity. Subject 1 final results were inspected before this exploratory follow-up.'}
    (args.output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    print(f'Packaged {report["run_id"]}: {len(included)} source files; {len(packed)} bytes; verified {len(hashes)} indexed files')


if __name__ == '__main__':
    main()
