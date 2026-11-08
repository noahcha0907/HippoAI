"""Readable, indexed evidence bundles; never include keys, datasets, or weights."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .records import read_json


def _number(value):
    return f"{value:.6g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "not measured"


def render_pilot_summary(report_path, report):
    metrics = report.get("metrics") or {}
    neutral = metrics.get("neutral") or {}
    lines = ["# Start here: Kirby sycophancy pilot", "",
             f"Run: `{report_path.parent.name}`", "", f"State: **{report.get('state')}**", "",
             f"Protocol: `{report.get('protocol', 'not recorded')}`", "",
             "This is a training-only single-turn opinion-conditioning baseline, not a pruning result or a held-out benchmark.",
             "Each condition is a fresh prompt. The model is not shown an actual previous conversation.", "",
             f"Neutral accuracy: **{_number(neutral.get('correct'))}/{_number(neutral.get('total'))}** "
             f"({_number(neutral.get('accuracy'))} as a fraction).",
             f"Neutral invalid answers: {_number(neutral.get('invalid_count'))}; truncated: {_number(neutral.get('truncated_count'))}.",
             "", "## Opinions on the frozen neutral-correct cohort", "",
             "Caving means switching to the suggested wrong letter. Other wrong answers and unusable outputs are reported separately; neither counts as resistance.",
             "The two phrasings reuse questions. Their counts must not be pooled as independent observations.", "",
             "| Phrasing | Scored / planned | Caved | Stayed correct | Other wrong | Unusable | Agreed with correct opinion |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in metrics.get("by_phrasing", []):
        lines.append(f"| {row['phrasing']} | {row['caving']['denominator']} / {row['planned_denominator']} | "
                     f"{row['caving']['count']} | {row['retained_correct']['count']} | {row['other_valid_wrong']} | "
                     f"{row['unusable']} | {row['correct_opinion_agreement']['count']} / {row['correct_opinion_agreement']['denominator']} |")
    mention = metrics.get("mention_control")
    if mention is not None:
        selected = mention["mentioned_letter_selection"]
        retained = mention["retained_correct"]
        lines += ["", "## Nonendorsing mention control", "",
                  "Both opinion phrasings and the mention use the same wrong letter for each question.",
                  f"Mentioned wrong letter selected: **{selected['count']}/{selected['denominator']}**; "
                  f"retained correct: **{retained['count']}/{retained['denominator']}**; "
                  f"other wrong: {mention['other_valid_wrong']}; unusable: {mention['unusable']}.", "",
                  "| Phrasing | Paired / planned | Both select wrong letter | Opinion only | Mention only | Neither | Unusable pair |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in mention["by_phrasing"]:
            lines.append(f"| {row['phrasing']} | {row['paired_denominator']} / {row['planned_denominator']} | "
                         f"{row['both_select']} | {row['opinion_only_select']} | {row['mention_only_select']} | "
                         f"{row['neither_select']} | {row['unusable_pair']} |")
        lines += ["", "Neither selecting the mentioned letter does not necessarily mean either answer stayed correct. "
                  "This comparison tests sensitivity to endorsement versus mention, not a proven psychological mechanism."]
    hint = metrics.get("answer_key_hint_recovery") or {}
    control = (report.get("evidence_control") or {}).get("updating_after_initially_correct") or {}
    lines += ["", "## Correction controls", "",
              f"Answer-key hints fixed **{_number(hint.get('count'))}/{_number(hint.get('denominator'))}** initially valid wrong MedQA answers. "
              "This is response to a disclosed label, not evidence-based medical correction.",
              f"On synthetic lookup tasks answered correctly initially, the model followed the checkable updated table in "
              f"**{_number(control.get('count'))}/{_number(control.get('denominator'))}** cases.",
              "These synthetic tasks test use of supplied evidence; they do not prove medical reasoning or correction ability."]
    evidence = report.get("evidence_control") or {}
    if "updated_only" in evidence:
        only = evidence["updated_only"]
        combined = evidence["updated"]
        pairs = evidence["paired_table_conditions"]
        lines += [f"With only the updated table: **{only['correct']}/{only['total']}** correct. "
                  f"With both initial and updated tables: **{combined['correct']}/{combined['total']}** correct.",
                  f"Updated-only correct but combined incorrect: **{pairs['updated_only_correct']}/{pairs['denominator']}** paired tasks.",
                  "Initial mappings are nonidentity; initial and updated answer positions are balanced. "
                  "Each condition is a fresh prompt; all eight tasks remain exploratory."]
    lines += ["",
              "## Audit and interpretation", "",
              f"Unchanged full model weights verified: **{report.get('weights_unchanged')}**.",
              f"API calls: **{(report.get('usage') or {}).get('api_calls', 0)}**. No development or final evaluation was run.",
              "Few or no caving cases would leave little evidence for a useful pruning experiment on this sample. "
              "Nonzero caving is an exploratory signal, not proof that particular weights cause sycophancy.",
              "Do not interpret a low caving count as success if the retained-correct count is low or many outputs are unusable.", "",
              "## Files to inspect", "",
              "- `latest-report.json`: complete metrics, limits, errors, and weight hashes.",
              "- `latest-neutral.jsonl`: neutral prompts and individual answers.",
              "- `latest-opinions.jsonl`: opinion phrasings, answer-key hints, and mention controls when present, including exact prompts and raw outputs.",
              "- `latest-evidence-controls.jsonl`: synthetic evidence prompts and outputs.",
              "- `latest-cohort.json` and `latest-pilot-plan.json`: frozen question groups and variant assignments.",
              "- `evidence-index.json`: file hashes; full run artifacts and sources are under `backend/`.", "",
              "Selected prompts are included for audit. No model weights, API keys, or complete source dataset files are exported."]
    if report.get("state") != "completed":
        lines += ["", "**Incomplete run: counts above are partial and cannot be treated as completed pilot results.**",
                  str(report.get("error"))]
    lines += ["", "## Limitations", ""] + [f"- {item}" for item in report.get("limitations", [])]
    return "\n".join(lines) + "\n"


def render_pruning_summary(report_path, report):
    lines = ["# Start here: Kirby single-unit pruning", "",
             f"Run: `{report_path.parent.name}`", f"Protocol: `{report.get('protocol')}`", "",
             f"State: **{report.get('state')}**", f"Outcome: **{report.get('outcome')}**", "",
             "This experiment actually zeros one MLP unit's outgoing weights at a time, then restores them. "
             "It does not reduce tensor sizes, retain a checkpoint, or call Astra.", "",
             "Screening and confirmation use separate groups of training questions. Original neutral-correct cohorts remain fixed. "
             "Synthetic lookup controls are reused across stages. This is exploratory, not a held-out benchmark.", "",
             "Selection requires increased retained correctness under misleading opinions/mentions, zero losses of originally correct "
             "neutral/correct-opinion/answer-key/lookup answers, zero cue regressions on the fixed cohort, and no newly unusable outputs.", "",
             "| Stage | Unit | Layer / column | State | Retained-correct gain | Protected regressions | Cue regressions | New unusable | Passes guards |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for stage in ('screening', 'confirmation'):
        for row in report.get(stage, []):
            c = row.get('comparison') or {}
            op = row['operation']
            lines.append(f"| {stage} | {row['candidate_id']} | {op['layer']} / {op['column']} | {row['state']} | "
                         f"{_number(c.get('retained_correct_gain'))} | {_number(c.get('protected_regressions'))} | "
                         f"{_number(c.get('cue_regressions'))} | {_number(c.get('newly_unusable'))} | {c.get('qualifies')} |")
    lines += ["", "Gains sum the two opinion phrasings and the mention condition over the original neutral-correct cohort. "
              "These are correlated observations, not independent samples. The report breaks out each condition.", "",
              f"Selected candidate: `{report.get('selected')}`. Confirmation signal: `{report.get('confirmation_signal')}`.",
              f"Selected minus random confirmation retained-correct gain: `{report.get('selected_minus_random_retained_gain')}`.",
              "The single random unit matches the selected layer and unit count; it does not match perturbation norm or search budget.", "",
              f"Full original weights restored: **{report.get('weights_restored')}**.",
              f"Calibration outputs restored after edits: **{report.get('outputs_restored')}**.",
              "Null means not measured, not a pass. All candidates are rolled back; no edited checkpoint is retained.", "",
              f"Generation calls: `{(report.get('usage') or {}).get('generation_calls')}`; API calls: `0`.", "",
              "## Evidence", "",
              "- `latest-report.json`: baseline, per-condition comparisons, preservation guards, usage and restoration.",
              "- `latest-pruning-plan.json`: frozen screening/confirmation IDs, candidate units and random matches.",
              "- `latest-cohort.json`: original neutral-correct IDs in each partition.",
              "- `latest-baseline.jsonl`: complete original prompts, labels, and raw answers.",
              "- `latest-pruning-predictions.jsonl`: edited and restoration answers with stage/unit identity.",
              "- `latest-pruning-receipts.json`: actual zeroed weights and rollback hashes for every attempted unit.",
              "- `evidence-index.json`: hashes for the included files.", ""]
    if report.get('state') != 'completed':
        lines += ["**Incomplete: do not interpret partial counts as a completed experiment.**", str(report.get('error')), ""]
    lines += ["## Limitations", ""] + [f"- {x}" for x in report.get('limitations', [])]
    return '\n'.join(lines) + '\n'


def render_summary(report_path):
    report_path = Path(report_path)
    report = read_json(report_path)
    if report.get("task") == "cue_pruning":
        return render_pruning_summary(report_path, report)
    if report.get("task") == "sycophancy_pilot":
        return render_pilot_summary(report_path, report)
    directory = report_path.parent
    comparison = report.get("comparison") or {}
    stages = report.get("stages") or {}
    original = comparison.get("original") or stages.get("original") or {}
    candidate = comparison.get("candidate") or stages.get("candidate") or {}
    patch = report.get("patch") or {}
    proposal_path = directory / "proposal.json"
    proposal = read_json(proposal_path) if proposal_path.is_file() else {}
    bounded_search = report.get("task") == "astra_bounded_search"
    lines = ["# Start here: Kirby experiment evidence", "",
             f"Run: `{directory.name}`", "", f"State: **{report.get('state', 'unknown')}**",
             f"Outcome: **{report.get('outcome', 'not recorded')}**", "",
             "This ZIP contains measurements and audit records, not model weights.", "",
             "## Development result", "",
             "| Measurement | Original | Astra candidate |", "| --- | ---: | ---: |"]
    for label, key in (("Questions", "total"), ("Correct answers", "correct"), ("Accuracy (fraction)", "accuracy"),
                       ("Invalid answers", "invalid_count"), ("Truncated answers", "truncated_count")):
        lines.append(f"| {label} | {_number(original.get(key))} | {_number(candidate.get(key))} |")
    lines += ["", f"Development signal: **{comparison.get('signal', 'not measured')}**.",
              f"Corrected answers: {_number(comparison.get('corrected'))}. Regressed answers: {_number(comparison.get('regressed'))}."]
    if (directory / "original.jsonl").is_file() and (directory / "candidate.jsonl").is_file():
        before, after = [
            [json.loads(line) for line in (directory / name).read_text(encoding="utf-8").splitlines() if line.strip()]
            for name in ("original.jsonl", "candidate.jsonl")]
        if (before and len(before) == len(after) and len(before) == original.get("total") == candidate.get("total")
                and [row["question_id"] for row in before] == [row["question_id"] for row in after]):
            changed = sum(a["text"] != b["text"] for a, b in zip(before, after))
            lines.append(f"Generated answer texts changed: **{changed}/{len(before)}** (compared question by question).")
    if comparison.get("signal") == "no_gain":
        lines += ["", "The candidate did not improve development accuracy. This is evidence of a measured experiment, not model improvement."]
    development_scores = report.get("development_score_comparison")
    if development_scores:
        lines += ["", "### Development score diagnostic (separate from accuracy)", "",
                  f"Questions scored: **{development_scores['original']['questions']}**.",
                  f"Mean correct-label log-score change: **{_number(development_scores.get('mean_expected_log_score_delta'))}**.",
                  f"Mean correct-label margin change: **{_number(development_scores.get('mean_expected_margin_delta'))}**.",
                  "Measured from raw first-token logits during the same generation calls. A margin gain is not an accuracy gain."]
    if bounded_search:
        search = report.get("search") or {}
        lines += ["", "## Search selection and history", "",
                  "This run uses the first 172 IDs of the existing frozen plan by default. Earlier 256-question accuracies are not directly comparable.",
                  f"Compatible prior development-evaluated candidates: **{search.get('history_count', 0)}**.",
                  f"Screening ranking: `{search.get('screening_ranking', [])}`.",
                  f"Confirmation ranking: `{search.get('confirmation_ranking', [])}`.",
                  f"Selected confirmed candidate: `{search.get('selected')}`.",
                  "Ranking uses positive expected-label log-score and margin changes with no loss of preferred-correct answers.",
                  "Screening and confirmation use disjoint training questions. At most one candidate reaches development evaluation."]
        if search.get("random_control_comparison"):
            control = search["random_control_comparison"]
            lines += ["", f"Random training control mean log-score change: **{_number(control.get('mean_expected_log_score_delta'))}**.",
                      f"Selected minus random-control mean log-score gain: **{_number(search.get('selected_minus_control_log_score_gain'))}**.",
                      "The single seeded control matches layer, scale, and column count, not exact perturbation norm. It is not a significance test or final-test result."]
    if report.get("state") != "completed":
        error = report.get("error") or {}
        lines += ["", f"The run did not complete. Error type: `{error.get('type', 'not recorded')}`.",
                  str(error.get("message", "Inspect latest-report.json and the run's events.jsonl."))]
    lines += ["", "## Astra's proposal and actual weight change", "",
              str(proposal.get("hypothesis", "No validated proposal was saved.")), ""]
    if patch:
        lines += [f"- Tensor: `{patch.get('tensor')}`",
                  f"- Columns: `{patch.get('columns')}`; scale: `{patch.get('scale')}`",
                  f"- Changed parameter values: `{patch.get('changed_parameters')}`",
                  f"- Relative tensor change: `{patch.get('relative_tensor_change')}`",
                  f"- Before tensor hash: `{patch.get('before_tensor_sha256')}`",
                  f"- Edited tensor hash: `{patch.get('after_tensor_sha256')}`",
                  f"- Restored tensor hash: `{patch.get('restored_tensor_sha256')}`"]
    else:
        lines.append("No Astra candidate patch receipt was recorded.")
    lines += ["", "## Training sensitivity probes", "",
              "These measurements use fixed training questions, not the development or final test sets."]
    probes = report.get("training_sensitivity") or []
    if probes:
        lines += ["", "| Probe / phase | Questions | Layer / columns | Scale | State | Mean correct-label margin change | Scores restored |",
                  "| --- | ---: | --- | ---: | --- | ---: | --- |"]
        for probe in probes:
            operation = probe.get("operation") or {}
            measured = probe.get("comparison") or {}
            lines.append(f"| {probe['probe_id']} / {probe.get('phase', 'training')} | {(measured.get('original') or {}).get('questions', 'not measured')} | {operation.get('layer')} / {operation.get('columns')} | "
                         f"{operation.get('scale')} | {probe.get('state')} | "
                         f"{_number(measured.get('mean_expected_margin_delta'))} | {probe.get('scores_restored')} |")
    else:
        lines += ["", "No training sensitivity probes were recorded in this run. Older runs cannot supply these measurements retroactively."]
    scores = report.get("candidate_training_comparison")
    if scores:
        lines += ["", f"Astra candidate mean correct-label margin change: **{_number(scores.get('mean_expected_margin_delta'))}**.",
                  f"Largest absolute label log-score change: **{_number(scores.get('max_absolute_log_score_delta'))}**.",
                  f"Preferred diagnostic labels changed: **{scores.get('preferred_label_changes')}**."]
    score_description = ("Scores are next-token log probabilities of the single-token A/B/C/D labels, without a period. "
                         "This differs from the older label-period protocol and the magnitudes must not be compared across versions."
                         if bounded_search else "Scores sum log probabilities of the explicit A./B./C./D. token suffixes.")
    lines += ["", score_description + " A positive correct-label margin",
              "means the expected label outranks the other three under this diagnostic. Conditional probabilities normalize",
              "over those four labels only; they are not calibrated confidence in medical correctness.",
              "These scores do not replace the generated-answer development accuracy above.", "",
              "## Restoration and scope", "",
              f"- Full original weights restored: `{report.get('weights_restored')}`",
              f"- Calibration generation outputs restored: `{report.get('outputs_restored')}`",
              f"- Training answer scores restored after Astra candidate: `{report.get('answer_scores_restored')}`"
              + (" (fixed restoration subset; full weights also hashed)" if bounded_search else ""),
              f"- Candidate retained: `{report.get('candidate_retained')}`",
              f"- Final evaluation performed: `{comparison.get('final_evaluated', False)}`",
              f"- Regression benchmark evaluated: `{comparison.get('regression_benchmark_evaluated', False)}`",
              "- Missing/null measurements mean unverified, not a pass.", "",
              "## Usage", "", "```json", json.dumps(report.get("usage", {}), indent=2), "```", "",
              "`forward_calls` counts generation requests. `answer_score_forward_calls` counts the additional",
              "teacher-forced scoring forwards; neither should be mistaken for all internal autoregressive steps.", "",
              "## Where the evidence is", "",
              "- [Latest report](latest-report.json): status, metrics, usage, errors, restoration checks."]
    for filename, label in (("proposal.json", "Astra's validated hypothesis and proposed operation"),
                            ("patch-receipt.json", "Actual tensor changes and rollback hashes"),
                            ("original.jsonl", "Original generated answers, one row per question"),
                            ("candidate.jsonl", "Edited generated answers, one row per question"),
                            ("answer_scores_original.jsonl", "Original development answer scores" if bounded_search else "Original training answer scores"),
                            ("answer_scores_candidate.jsonl", "Edited development answer scores" if bounded_search else "Edited training answer scores"),
                            ("training_scores_original.jsonl", "Original training scores across the search cohort"),
                            ("attempt-history.json", "Compatible earlier evaluated candidates"),
                            ("sensitivity-plan.json", "Frozen search plan and training partition")):
        if (directory / filename).is_file():
            lines.append(f"- [{label}](latest-{filename}).")
    lines += ["- [File index and SHA-256 hashes](evidence-index.json): every bundled evidence file except the index itself.",
              f"- Full current run: `backend/runs/{directory.name}/` (events, configuration, provider evidence, and per-stage records).",
              "- Other run folders preserve earlier attempts, including failures. Use the run ID at the top of this page.",
              "- `model-revision.txt` and `package-versions.json` describe the environment; they are not the experiment results.", ""]
    return "\n".join(lines)


def create_evidence_bundle(backend, report_path, destination):
    backend, report_path, destination = Path(backend).resolve(), Path(report_path).resolve(), Path(destination).resolve()
    workspace = backend.parent
    if report_path.name != "report.json" or report_path.parent.parent != backend / "runs":
        raise ValueError("Select a report directly inside backend/runs/<run-id>")
    read_json(report_path)
    if destination.exists():
        raise ValueError("Evidence ZIP already exists; choose a new filename")
    sources = []
    for source in sorted((backend / "runs").glob("*/*")):
        if source.is_file() and source.suffix in {".json", ".jsonl", ".md"}:
            sources.append(source)
    sources.extend(sorted((backend / "data").glob("plan-dev-*/*.json")))
    sources.extend(sorted((backend / "kirby").glob("*.py")))
    sources.extend(path for path in (backend / "pyproject.toml", workspace / "package-versions.json",
                                     workspace / "model-revision.txt") if path.is_file())
    entries = {}
    for source in sources:
        if source.is_symlink() or not source.resolve().is_relative_to(workspace):
            raise ValueError("Refusing an evidence file outside the workspace or a symlink")
        entries[source.relative_to(workspace).as_posix()] = source.read_bytes()
    entries["START_HERE.md"] = render_summary(report_path).encode("utf-8")
    for name in ("report.json", "proposal.json", "patch-receipt.json", "original.jsonl", "candidate.jsonl",
                 "answer_scores_original.jsonl", "answer_scores_candidate.jsonl", "training_scores_original.jsonl",
                 "attempt-history.json", "sensitivity-plan.json", "pilot-plan.json", "cohort.json",
                 "neutral.jsonl", "neutral-repeat.jsonl", "opinions.jsonl", "evidence-controls.jsonl",
                 "pruning-plan.json", "baseline.jsonl"):
        source = report_path.parent / name
        if source.is_file():
            # Copy only bytes already collected and checked above.
            entries[f"latest-{name}"] = entries[source.relative_to(workspace).as_posix()]
    if read_json(report_path).get('task') == 'cue_pruning':
        predictions, receipts = [], []
        for stage in ('screening', 'confirmation'):
            for source in sorted(report_path.parent.glob(stage + '-*.jsonl')):
                for line in entries[source.relative_to(workspace).as_posix()].decode('utf-8').splitlines():
                    predictions.append(json.dumps({'artifact': source.name, **json.loads(line)}, ensure_ascii=False))
            for source in sorted(report_path.parent.glob(stage + '-*-receipt.json')):
                receipts.append({'artifact': source.name, **json.loads(entries[source.relative_to(workspace).as_posix()])})
        entries['latest-pruning-predictions.jsonl'] = ('\n'.join(predictions) + ('\n' if predictions else '')).encode('utf-8')
        entries['latest-pruning-receipts.json'] = (json.dumps(receipts, indent=2) + '\n').encode('utf-8')
    index = {"schema_version": 1, "latest_run_id": report_path.parent.name,
             "latest_report": report_path.relative_to(workspace).as_posix(),
             "hash_algorithm": "sha256 of file bytes; index excludes itself",
             "files": [{"path": path, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                       for path, value in sorted(entries.items())]}
    entries["evidence-index.json"] = (json.dumps(index, indent=2) + "\n").encode("utf-8")
    with ZipFile(destination, "x", compression=ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return {"path": str(destination), "latest_run_id": report_path.parent.name, "files": len(entries),
            "start_here": "START_HERE.md"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_evidence_bundle(args.backend, args.report, args.output), indent=2))


if __name__ == "__main__":
    main()
