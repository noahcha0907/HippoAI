"""Build the EEG-only Colab notebook, independent of earlier MedPhi launchers."""

import json
from pathlib import Path


SETUP = r'''
from pathlib import Path, PurePosixPath
import hashlib, json, os, stat, subprocess, sys, uuid, zipfile
from google.colab import files

WORKSPACE = Path('/content/kirby-eeg-workspace')
WORKSPACE.mkdir(exist_ok=True)
if 'EEG_SOURCE_ZIP' not in globals():
    print('Upload kirby_eeg_colab.zip')
    uploaded = files.upload()
    names = [name for name in uploaded if Path(name).name == 'kirby_eeg_colab.zip']
    if len(names) != 1:
        raise ValueError('Upload exactly the new kirby_eeg_colab.zip source bundle')
    EEG_SOURCE_ZIP = Path(names[0]).resolve()
source_hash = hashlib.sha256(EEG_SOURCE_ZIP.read_bytes()).hexdigest()
SOURCE = WORKSPACE / ('source-' + source_hash[:16])
SOURCE.mkdir(exist_ok=True)
with zipfile.ZipFile(EEG_SOURCE_ZIP) as archive:
    entries = archive.infolist()
    if len(entries) > 200 or sum(i.file_size for i in entries) > 10_000_000:
        raise ValueError('Unexpected source archive size')
    names = [i.filename for i in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive paths')
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if (path.is_absolute() or '..' in path.parts or '\\' in entry.filename or ':' in entry.filename
                or not path.parts or path.parts[0] != 'backend'
                or stat.S_ISLNK(entry.external_attr >> 16)
                or path.suffix not in ('.py', '.md', '.toml', '.ipynb')):
            raise ValueError('Unsafe source archive member')
        target = SOURCE.joinpath(*path.parts)
        if not target.resolve().is_relative_to(SOURCE.resolve()):
            raise ValueError('Archive escaped source workspace')
    archive.extractall(SOURCE)
BACKEND = SOURCE / 'backend'
if not (BACKEND / 'kirby/eeg/runner.py').is_file():
    raise ValueError('This is not the EEG source bundle')
VENV = WORKSPACE / ('venv-' + source_hash[:16])
PYTHON = VENV / 'bin/python'
if not PYTHON.exists():
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(VENV)], check=True)
# Reuses Colab's CUDA torch; EEG dependencies live in the dedicated worker environment.
if not (VENV / 'kirby-installed.json').exists():
    subprocess.run([str(PYTHON), '-m', 'pip', 'install', '--disable-pip-version-check', '-e', str(BACKEND) + '[eeg]'], check=True)
    (VENV / 'kirby-installed.json').write_text(json.dumps({'source_sha256': source_hash}))
env = os.environ.copy()
env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
env['MPLBACKEND'] = 'Agg'
env['MNE_DATA'] = str(WORKSPACE / 'dataset-cache')
Path(env['MNE_DATA']).mkdir(exist_ok=True)
subprocess.run([str(PYTHON), '-c', 'import torch; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())'], env=env, check=True)
print('EEG workspace ready:', WORKSPACE)
'''

CONFIG = r'''
# First real experiment: 6 EEG models (1 subject x 3 seeds x clean/blink).
SUBJECTS = [1]
SEEDS = [0, 1, 2]
CONDITIONS = ['clean', 'eye_blink']
EVIDENCE_CONDITIONS = ['A', 'B']
EPOCHS = 300
PATIENCE = 30
ATTEMPTS = 5
RUN_ASTRA = True
EXTENDED_BASELINES = False
MAX_API_CALLS = 72
MAX_ESTIMATED_USD = 50.0  # Recorded token-price estimate, not a provider billing limit.
RUN_ID = None  # Reuses saved run. For a deliberate NEW experiment set a fresh UUID string.
'''

RUN = r'''
settings = {name: globals()[name] for name in (
    'SUBJECTS', 'SEEDS', 'CONDITIONS', 'EVIDENCE_CONDITIONS', 'EPOCHS', 'PATIENCE',
    'ATTEMPTS', 'RUN_ASTRA', 'EXTENDED_BASELINES', 'MAX_API_CALLS', 'MAX_ESTIMATED_USD')}
settings['source_sha256'] = source_hash
attempt_file = WORKSPACE / 'eeg-attempt-id.txt'
if RUN_ID is not None:
    run_id = str(uuid.UUID(RUN_ID))
elif attempt_file.exists():
    run_id = str(uuid.UUID(attempt_file.read_text().strip()))
else:
    run_id = str(uuid.uuid4())
attempt_file.write_text(run_id)
run_root = WORKSPACE / 'eeg-runs' / run_id
settings_file = WORKSPACE / ('settings-' + run_id + '.json')
if settings_file.exists() and json.loads(settings_file.read_text()) != settings:
    raise ValueError('This run ID belongs to different settings/source. Use a deliberate new UUID for a new experiment.')
settings_file.write_text(json.dumps(settings, sort_keys=True))
if run_root.exists():
    print('Run already exists: no training, API calls, or final evaluation will be repeated.')
    if (run_root / 'report.json').exists():
        print('Saved status:', json.loads((run_root / 'report.json').read_text())['status'])
else:
    if RUN_ASTRA:
        from google.colab import userdata
        key = userdata.get('OPENAI_API_KEY')
        if not key:
            raise ValueError('Add OPENAI_API_KEY to Colab Secrets and enable notebook access.')
        env['OPENAI_API_KEY'] = key
        del key
    command = [str(PYTHON), '-u', '-m', 'kirby.eeg', 'run', '--output', str(WORKSPACE / 'eeg-runs'),
               '--data-root', str(WORKSPACE / 'eeg-data'), '--run-id', run_id,
               '--subjects', *map(str, SUBJECTS), '--seeds', *map(str, SEEDS),
               '--conditions', *CONDITIONS, '--evidence-conditions', *EVIDENCE_CONDITIONS,
               '--epochs', str(EPOCHS), '--patience', str(PATIENCE), '--attempts', str(ATTEMPTS),
               '--max-api-calls', str(MAX_API_CALLS), '--max-estimated-usd', str(MAX_ESTIMATED_USD)]
    if not RUN_ASTRA:
        command.append('--no-astra')
    if EXTENDED_BASELINES:
        command.append('--extended-baselines')
    try:
        completed = subprocess.run(command, cwd=BACKEND, env=env)
        print('Worker exit code:', completed.returncode)
    finally:
        env.pop('OPENAI_API_KEY', None)
if (run_root / 'START_HERE.md').exists():
    from IPython.display import Markdown, display
    display(Markdown((run_root / 'START_HERE.md').read_text()))
'''

DOWNLOAD = r'''
# Re-export reads saved results only. It never resumes a failed experiment.
if not (run_root / 'report.json').exists():
    raise RuntimeError('No report was created. Inspect the worker error above.')
subprocess.run([str(PYTHON), '-m', 'kirby.eeg', 'export', str(run_root)], cwd=BACKEND, env=env, check=True)
evidence = run_root / ('kirby-eeg-evidence-' + run_id + '.zip')
files.download(str(evidence))
print('Download includes current-run status, evidence, checkpoints and plots. Preserve it before resetting Colab.')
'''


def main():
    backend = Path(__file__).resolve().parents[1]
    def code(source):
        return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': source.strip().splitlines(keepends=True)}
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'}, 'language_info': {'name': 'python'}, 'accelerator': 'GPU'},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source': [
                    '# Kirby: EEG shortcut audit\n',
                    'Use a GPU runtime. Upload **kirby_eeg_colab.zip** in the first cell. Enable Colab Secret `OPENAI_API_KEY` for Astra.\n',
                    'Defaults train one subject, three seeds, clean/blink conditions. A=weights, B=weights plus signal views.\n',
                    'No Astra request occurs before development gates. Final session is evaluated only after all methods are frozen.\n',
                    'Rerunning cells displays the saved run; it does not repeat paid requests. Failed runs export partial evidence.\n',
                    'Keep the downloaded evidence ZIP. The first real run validates the implementation; it does not establish clinical usefulness.\n']},
                          code(SETUP), code(CONFIG), code(RUN), code(DOWNLOAD)]}
    for i, cell in enumerate(notebook['cells']):
        cell['id'] = f'kirby-eeg-{i}'
    path = backend / 'notebooks/kirby_eeg_colab.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()
