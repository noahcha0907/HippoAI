"""Build a separate Colab launcher for the predeclared new-subject replication."""
import hashlib
import json
from pathlib import Path

from build_eeg_notebook import SETUP, CONFIG, RUN, DOWNLOAD

BACKEND = Path(__file__).resolve().parents[1]
PLAN = BACKEND / 'experiments/eeg_replication_v1.json'


def cells():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    fields = {'SUBJECTS':'subjects', 'SEEDS':'seeds', 'CONDITIONS':'conditions',
              'EVIDENCE_CONDITIONS':'evidence_conditions', 'EPOCHS':'epochs', 'PATIENCE':'patience',
              'ATTEMPTS':'attempts', 'REPAIR_POLICY':'repair_policy', 'RUN_ASTRA':'astra',
              'EXTENDED_BASELINES':'extended_baselines', 'MAX_API_CALLS':'max_api_calls',
              'MAX_ESTIMATED_USD':'max_estimated_usd'}
    config = '\n'.join(f'{name} = {plan["config"][key]!r}' for name,key in fields.items())
    config += '\nRUN_ID = None\n'
    config += f'''
from pathlib import Path
import hashlib, json
_plan_path = BACKEND / 'experiments/eeg_replication_v1.json'
_plan_sha256 = {hashlib.sha256(PLAN.read_text(encoding='utf-8').encode('utf-8')).hexdigest()!r}
_replication_fields = {fields!r}

def verify_replication():
    if hashlib.sha256(_plan_path.read_text(encoding='utf-8').encode('utf-8')).hexdigest() != _plan_sha256:
        raise ValueError('Replication plan changed. Rebuild and declare a new study before running.')
    plan = json.loads(_plan_path.read_text())
    for name, key in _replication_fields.items():
        if globals()[name] != plan['config'][key]:
            raise ValueError('Frozen replication setting changed: ' + name)
    for relative, expected in plan['source_sha256'].items():
        if hashlib.sha256((BACKEND / relative).read_text(encoding='utf-8').encode('utf-8')).hexdigest() != expected:
            raise ValueError('Frozen experiment source changed: ' + relative)
    print('Replication verified: subjects 2 and 3, seeds 0/1/2, unchanged run6 repair procedure.', flush=True)

verify_replication()
'''
    run = 'verify_replication()\n' + RUN.replace("'eeg-attempt-id.txt'", "'eeg-replication-v1-attempt-id.txt'")
    run += "\nif run_root.exists():\n    (run_root / 'replication-plan.json').write_bytes(_plan_path.read_bytes())\n"
    return [SETUP, config, run, DOWNLOAD]


def main():
    sources = cells()
    notebook = {'nbformat':4, 'nbformat_minor':5,
                'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3'},'language_info':{'name':'python'},'accelerator':'GPU'},
                'cells':[{'cell_type':'markdown','metadata':{},'id':'replication-intro','source':[
                    '# Kirby EEG: frozen replication on subjects 2 and 3\n',
                    'Use a GPU runtime and enable the OPENAI_API_KEY Colab Secret. Run all cells and upload the updated kirby_eeg_colab.zip.\n',
                    '12 models: two new subjects, three seeds, clean/blink. Same run6 repair procedure; A and B both retained.\n',
                    'Maximum 144 API requests and $100 estimated token cost (not a billing guarantee). No calls until development gates pass.\n',
                    'Keep all exclusions and failures. Do not tune against final results. The plan and source hashes are checked before execution.\n',
                    'Rerunning cells reuses the saved replication ID; download the evidence ZIP when finished.\n']}]
                +[{'cell_type':'code','metadata':{},'id':f'replication-{i}','execution_count':None,'outputs':[],
                   'source':source.strip().splitlines(keepends=True)} for i,source in enumerate(sources)]}
    target=BACKEND/'notebooks/kirby_eeg_replication_colab.ipynb'
    target.write_text(json.dumps(notebook,indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
