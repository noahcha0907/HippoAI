"""Build the pilot notebook from the existing, tested source-update bootstrap.

Run when changing the common Colab bootstrap. No model or API access.
"""

import json
from pathlib import Path


def main():
    notebooks = Path(__file__).resolve().parents[1] / "notebooks"
    source = json.loads((notebooks / "kirby_astra_colab.ipynb").read_text(encoding="utf-8"))
    resume = next("".join(c["source"]) for c in source["cells"] if c["id"] == "resume")
    resume = resume.replace("from google.colab import files, userdata", "from google.colab import files")
    start = resume.index("    required = {")
    end = resume.index("for name, payload", start)
    resume = resume[:start] + '''    required = {"backend/kirby/sycophancy.py", "backend/kirby/evidence.py"}
    if not required.issubset(replacements):
        raise ValueError("Upload the new ZIP containing the sycophancy pilot.")
''' + resume[end:]
    resume = resume.replace('str(backend) + "[astra]"', 'str(backend)')
    resume = "\n".join(line for line in resume.splitlines() if "from openai import OpenAI" not in line) + "\n"
    start = resume.index("run_setup([worker_python, \"-m\", \"kirby\", \"validate\"")
    resume = resume[:start] + '''saved_plan = json.loads((plan / 'development.json').read_text())
if len(saved_plan['diagnostic_ids']) < 64:
    raise RuntimeError('This pilot needs 64 frozen training diagnostic IDs.')
print('Ready: pilot v2, 64 training questions, matched distractors, mention and nonidentity lookup controls. No Astra, weight edits, or final evaluation.')
'''
    attempt = '''# Re-running this cell shows the same evidence. Set NEW_ATTEMPT=True only to run another pilot.
NEW_ATTEMPT = False
id_file = workspace / 'sycophancy-pilot-v2-64-attempt-id.txt'
if NEW_ATTEMPT or not id_file.is_file():
    id_file.write_text(str(uuid.uuid4()), encoding='utf-8')
attempt_id = str(uuid.UUID(id_file.read_text().strip()))
pilot_report_path = runs / attempt_id / 'report.json'
print('Selected pilot:', attempt_id)
if pilot_report_path.is_file():
    print('This pilot already ran. Showing saved evidence; no new inference.')
elif pilot_report_path.parent.exists():
    raise RuntimeError('This attempt directory exists without a report. Inspect it before selecting NEW_ATTEMPT=True.')
else:
    worker_env = os.environ.copy()
    worker_env.pop('OPENAI_API_KEY', None)
    run_worker([worker_python, '-u', '-m', 'kirby', 'sycophancy-pilot',
                '--manifest', str(plan / 'development.json'), '--attempt-id', attempt_id,
                '--model', MODEL_ID, '--revision', MODEL_REVISION, '--precision', PRECISION,
                '--allow-download', '--questions', '64', '--max-seconds', str(MAX_SECONDS),
                '--output', str(runs)], env=worker_env)
report = json.loads(pilot_report_path.read_text())
if report.get('protocol') != 'single_turn_opinion_pilot_v2':
    raise RuntimeError('Selected report is not pilot v2. Inspect the attempt ID; do not substitute an older result.')
print(json.dumps({key: report.get(key) for key in
                 ('state', 'outcome', 'metrics', 'evidence_control', 'weights_unchanged', 'usage', 'error')}, indent=2))
print('Pilot evidence:', pilot_report_path)
'''
    download = '''from datetime import datetime, timezone
selected_report = globals().get('pilot_report_path')
if selected_report is None or not Path(selected_report).is_file():
    print('No report for this pilot exists yet. No older experiment will be mislabeled as this result.')
else:
    inventory = subprocess.check_output([worker_python, '-m', 'pip', 'list', '--format=json'], text=True)
    (workspace / 'package-versions.json').write_text(inventory, encoding='utf-8')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = workspace / f'kirby-sycophancy-evidence-{stamp}.zip'
    subprocess.run([worker_python, '-m', 'kirby.evidence', '--backend', str(backend),
                    '--report', str(selected_report), '--output', str(destination)], cwd=backend, check=True)
    print('Open START_HERE.md. Individual prompts and answers are in latest-neutral.jsonl and latest-opinions.jsonl.')
    print('Downloading:', destination.name)
    files.download(str(destination))
'''
    def cell(kind, identity, text):
        result = {"cell_type": kind, "id": identity, "metadata": {}, "source": text.splitlines(keepends=True)}
        if kind == "code":
            result.update(execution_count=None, outputs=[])
        return result
    cells = [cell("markdown", "intro", "# Kirby: refined sycophancy pilot v2\n\n"
                  "Resume the existing GPU workspace. Run the three cells in order. Upload the new source ZIP when requested. "
                  "No API key or rollback report is needed. This measures single-turn opinion conditioning on 64 frozen training questions; "
                  "the development/final questions are not scored. Maximum 476 generations including correction controls. "
                  "Unchanged weights are checked by full hashes. This is not a pruning experiment.\n\n"
                  "Both opinion phrasings and a nonendorsing mention use the same wrong label per question. Medical option positions remain unchanged. "
                  "Answer-key hints are labeled as hints, not medical explanations. "
                  "Synthetic controls use nonidentity tables and compare updated-only with combined tables. "
                  "Results are exploratory and not independent across phrasings. A separate v2 attempt ID preserves v1 evidence.\n"),
             cell("code", "resume", resume), cell("code", "attempt", attempt), cell("code", "download", download)]
    source["cells"] = cells
    (notebooks / "kirby_sycophancy_colab.ipynb").write_text(json.dumps(source, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
