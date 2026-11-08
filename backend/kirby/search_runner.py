"""Budgeted training search, one Astra review, and one development comparison."""

import time
import traceback
import uuid
from dataclasses import asdict
from pathlib import Path

from .answer_scores import (NEXT_LABEL_PROTOCOL, SCORE_TOLERANCE, answer_score_record, answer_scores_restored,
                            compare_answer_scores, score_next_labels, summarize_answer_scores)
from .astra import ProviderFailure, validate_proposal
from .astra_runner import ProposalRejected, paired_summary, validate_rollback_evidence
from .datasets import load_development
from .diagnostics import PhiActivationSummary, load_diagnostics
from .evaluation import PARSER, PROFILE, format_prompt, score, summarize
from .patches import ColumnScale, PhiColumnEditor, weights_hash
from .records import canonical_json, digest, file_hash, utc_now, write_json
from .reproducibility import PREDICTION_FIELDS
from .runner import RunStopped
from .search import (SEARCH_POLICY, SearchConfig, matched_random_operation, operation_key, rank_probes,
                     read_attempt_history, screening_plan)


class SearchFinished(Exception):
    """A completed training search with no eligible candidate, not an error."""


def _compact_comparison(comparison):
    return {key: value for key, value in comparison.items() if key != "per_question"}


def search_packet(metadata, model, manifest, original, diagnostics, activations, history, confirmations, selected, control):
    # Full per-question evidence is saved locally. The provider gets bounded
    # summaries and examples, not all search artifacts or historical datasets.
    probes = [{"evidence_id": row["probe_id"], "operation": row["operation"],
               "comparison": _compact_comparison(row["comparison"]), "scores_restored": True,
               "rollback_verified": row["receipt"]["rollback_verified"]} for row in confirmations]
    examples = [{"evidence_id": q.id, "question": q.input.question[:1000],
                 "options": dict(zip("ABCD", [value[:200] for value in q.input.options])),
                 "expected": q.answer, "excerpted": len(q.input.question) > 1000 or any(len(v) > 200 for v in q.input.options)}
                for q in diagnostics[:8]]
    packet = {"schema_version": 3, "task": "review_ranked_training_candidate", "base_checkpoint_id": metadata["checkpoint_id"],
              "model": {"repository": metadata.get("repository"), "revision": metadata.get("revision"),
                        "precision": metadata.get("precision"), "layers": model.config.num_hidden_layers},
              "answer_score_protocol": NEXT_LABEL_PROTOCOL, "search_policy": SEARCH_POLICY,
              "training_selection_ids_sha256": digest([q.id for q in diagnostics]),
              "development_summary": summarize(original), "training_diagnostics": examples,
              "activation_summary": activations, "training_sensitivity": probes,
              "selected_candidate": {"evidence_id": selected["probe_id"], "operation": selected["operation"]},
              "random_control": {"evidence_id": control["probe_id"], "operation": control["operation"],
                                 "comparison": _compact_comparison(control["comparison"]),
                                 "limitation": "one seeded control, matched layer/scale/column count, not tensor norm or a significance test"},
              "attempt_history": history["accepted"][-20:],
              "limits": {"api_calls": 1, "development_candidates": 1, "operations": 1, "columns": 1,
                         "scale_min": 0.9, "scale_max": 1.1},
              "limitations": ["Review the selected operation exactly or abstain; do not change the deterministic ranking.",
                              "Next-token single-letter scores differ from earlier label-period likelihoods; do not compare magnitudes across protocols.",
                              "Selection used training data only. The random control was not used to rank candidates.",
                              "History from different development cohorts prevents repeats but is not pooled for accuracy comparisons.",
                              "No final questions, no cumulative retained edits, no claim of generalization."]}
    packet["evidence_ids"] = ["development-summary", *[q["evidence_id"] for q in examples],
                              *[r["evidence_id"] for r in activations], *[r["evidence_id"] for r in probes], control["probe_id"]]
    packet["evidence_sha256"] = digest(packet)
    return packet


def run_astra_search(manifest_path, rollback_report, predictor, model_config, astra, output_root,
                     config=SearchConfig(), *, attempt_id=None):
    all_rows, manifest = load_development(Path(manifest_path))
    if len(all_rows) < config.development_count:
        raise ValueError("Frozen plan has fewer questions than the requested development count")
    rows = all_rows[:config.development_count]
    diagnostics = load_diagnostics(manifest, rows, config.diagnostic_count)
    screening, confirmation = diagnostics[:config.screening_count], diagnostics[config.screening_count:]
    restoration_rows = diagnostics[:min(config.calibration_questions, len(diagnostics))]
    run_id = str(uuid.UUID(attempt_id)) if attempt_id else str(uuid.uuid4())
    directory = Path(output_root) / run_id
    rollback_report = Path(rollback_report)
    rollback_hash = file_hash(rollback_report)
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": 3, "task": "astra_bounded_search", "manifest": manifest, "manifest_sha256": digest(manifest),
              "model": model_config, "editor": asdict(astra.config), "limits": asdict(config),
              "profile": PROFILE, "parser": PARSER, "answer_score_protocol": NEXT_LABEL_PROTOCOL,
              "development_ids": [q.id for q in rows], "development_selection": "prefix_of_frozen_manifest_v1",
              "screening_ids": [q.id for q in screening], "confirmation_ids": [q.id for q in confirmation],
              "restoration_ids": [q.id for q in restoration_rows], "search_policy": SEARCH_POLICY,
              "rollback_report_sha256": rollback_hash, "score_restoration_tolerance": SCORE_TOLERANCE,
              "implementation_files_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    restore_count = len(restoration_rows)
    frozen["maximum_work"] = {
        "generation_requests": 2 * len(rows) + 3 * config.calibration_questions + len(screening),
        "training_score_forwards": len(diagnostics) + restore_count + 2 * config.layer_count * (len(screening) + restore_count)
            + (config.confirm_count + 1) * (len(confirmation) + restore_count) + restore_count,
        "api_calls": 1, "development_candidates": 1}
    write_json(directory / "configuration.json", frozen)
    report = {"schema_version": 3, "task": "astra_bounded_search", "run_id": run_id, "state": "running",
              "outcome": "Run incomplete", "started_at": utc_now(), "configuration_sha256": digest(frozen),
              "model": None, "editor": None, "patch": None, "comparison": None, "development_score_comparison": None,
              "stages": {}, "answer_score_stages": {}, "training_sensitivity": [], "search": {},
              "maximum_work": frozen["maximum_work"], "answer_score_protocol": NEXT_LABEL_PROTOCOL,
              "candidate_training_comparison": None, "candidate_retained": False,
              "weights_restored": None, "outputs_restored": None, "answer_scores_restored": None,
              "final": None, "regression": None,
              "usage": {"forward_calls": 0, "answer_score_forward_calls": 0, "generated_tokens": 0, "api_calls": 0, "api_usage": None},
              "limitations": ["At most 16 screen edits, two confirmations, one random training control, one Astra request, one development candidate.",
                              "64 fixed training questions by default; selection/confirmation partition is frozen before measurement.",
                              "Development cohort is a frozen 172-question prefix, not directly comparable to earlier 256-question accuracy.",
                              "Next-token A/B/C/D diagnostic margins are separate from generated-answer accuracy and older label-period scores.",
                              "Random control matches layer, scale, and parameter count; not exact tensor norm. One control is not a significance test.",
                              "All edits restored. No cumulative acceptance, edited checkpoint export, final evaluation, or proven repair.",
                              "Time limits are cooperative. Rollback takes priority over deadline."]}
    write_json(directory / "report.json", report)
    started = time.monotonic()
    editor, current_probe = None, None
    generation_scores = {}

    def event(kind, **fields):
        report["last_event"] = {"event": kind, **fields}
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"at": utc_now(), "event": kind, **fields}) + "\n")
        write_json(directory / "report.json", report)
        print(kind + (": " + fields["stage"] if "stage" in fields else ""), flush=True)

    def remaining():
        value = config.max_seconds - (time.monotonic() - started)
        if value <= 0:
            raise RunStopped("Wall-time budget exhausted")
        return value

    def generation(stage, questions):
        event("measuring", stage=stage, questions=len(questions))
        predictions, scored = [], []
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream, \
                (directory / f"answer_scores_{stage}.jsonl").open("w", encoding="utf-8") as scores_stream:
            for q in questions:
                remaining()
                before = time.monotonic()
                report["usage"]["forward_calls"] += 1
                generated, values = predictor.generate_with_answer_scores(format_prompt(q.input))
                report["usage"]["generated_tokens"] += generated.generated_tokens
                predictions.append(score(q, generated, time.monotonic() - before))
                scored.append(answer_score_record(q, values, phase=q.split))
                stream.write(canonical_json(predictions[-1]) + "\n")
                scores_stream.write(canonical_json(scored[-1]) + "\n")
                stream.flush()
                scores_stream.flush()
                report["stages"][stage] = {"complete": len(predictions) == len(questions), "planned_questions": len(questions), **summarize(predictions)}
                if len(predictions) % 16 == 0 or len(predictions) == len(questions):
                    event("generation_progress", stage=stage, completed=len(predictions), total=len(questions))
        generation_scores[stage] = scored
        return predictions

    def training_scores(stage, questions):
        event("scoring_training_answers", stage=stage, questions=len(questions))
        result = []
        def before_forward():
            remaining()
            report["usage"]["answer_score_forward_calls"] += 1
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for q in questions:
                values = score_next_labels(predictor, format_prompt(q.input), before_forward=before_forward)
                result.append(answer_score_record(q, values))
                stream.write(canonical_json(result[-1]) + "\n")
                stream.flush()
                report["answer_score_stages"][stage] = {"complete": len(result) == len(questions),
                                                         "planned_questions": len(questions), **summarize_answer_scores(result)}
                if len(result) % 16 == 0 or len(result) == len(questions):
                    event("score_progress", stage=stage, completed=len(result), total=len(questions))
        remaining()
        return result

    def equal(a, b):
        return len(a) == len(b) and all(all(x[key] == y[key] for key in PREDICTION_FIELDS) for x, y in zip(a, b))

    def probe(planned, questions, baseline, restore_baseline, stage):
        nonlocal current_probe
        operation = planned["operation"]
        current_probe = {**planned, "phase": stage, "state": "running", "receipt": None,
                         "scores_restored": None, "comparison": None}
        report["training_sensitivity"].append(current_probe)
        path = directory / f"{planned['probe_id']}.json"
        write_json(path, current_probe)
        remaining()
        event("applying_training_probe", stage=planned["probe_id"])
        try:
            current_probe["receipt"] = editor.apply(ColumnScale(report["model"]["checkpoint_id"],
                f"{run_id}-{planned['probe_id']}", operation["layer"], tuple(operation["columns"]), operation["scale"]))
            write_json(path, current_probe)
            changed = training_scores(f"scores_{planned['probe_id']}", questions)
        finally:
            if editor.active:
                current_probe["receipt"] = editor.rollback()
            write_json(path, current_probe)
        restored = training_scores(f"scores_{planned['probe_id']}_restored", restoration_rows)
        current_probe["scores_restored"] = answer_scores_restored(restore_baseline, restored)
        if not current_probe["scores_restored"]:
            raise RunStopped("Training score restoration failed; Astra was not called")
        current_probe["comparison"] = compare_answer_scores(baseline, changed)
        current_probe["state"] = "completed"
        write_json(path, current_probe)
        return current_probe

    try:
        # Credential/SDK check only: no API call until a confirmed candidate exists.
        astra.connect()
        event("loading_model")
        remaining()
        report["model"] = predictor.load()
        write_json(directory / "model.json", report["model"])
        report["original_weights_sha256"] = weights_hash(predictor.model)
        if file_hash(rollback_report) != rollback_hash:
            raise ValueError("Rollback report changed")
        validate_rollback_evidence(rollback_report, manifest, report["model"], report["original_weights_sha256"])
        history = read_attempt_history(output_root, report["model"], manifest, report["original_weights_sha256"])
        write_json(directory / "attempt-history.json", history)
        report["search"]["history_count"] = len(history["accepted"])
        calibration = generation("calibration", rows[:config.calibration_questions])
        original = generation("original", rows)
        if (not equal(calibration, original[:config.calibration_questions])
                or summarize(original)["invalid_rate"] > 0.25 or summarize(original)["truncated_count"]):
            raise RunStopped("Original model unstable or answer protocol failed; Astra was not called")
        with PhiActivationSummary(predictor.model) as collector:
            generation("diagnostic_train", screening)
        activations = collector.summary()
        planned = screening_plan(activations, report["model"]["checkpoint_id"], history, config)
        write_json(directory / "sensitivity-plan.json", {"policy": SEARCH_POLICY, "config": asdict(config), "probes": planned,
                                                        "screening_ids": frozen["screening_ids"], "confirmation_ids": frozen["confirmation_ids"]})
        baseline = training_scores("training_scores_original", diagnostics)
        restore_baseline = baseline[:len(restoration_rows)]
        if not answer_scores_restored(restore_baseline, training_scores("training_scores_repeat", restoration_rows)):
            raise RunStopped("Original next-token training scores are unstable")
        editor = PhiColumnEditor(predictor.model, report["model"]["checkpoint_id"])
        screened = [probe(item, screening, baseline[:len(screening)], restore_baseline, "screening") for item in planned]
        shortlisted = rank_probes(screened)[:config.confirm_count]
        report["search"]["screening_ranking"] = [row["probe_id"] for row in rank_probes(screened)]
        confirmed = []
        for index, row in enumerate(shortlisted, start=1):
            confirmed.append(probe({"probe_id": f"confirm-{index}", "operation": row["operation"],
                                    "operation_key": row["operation_key"], "screening_probe": row["probe_id"]},
                                   confirmation, baseline[len(screening):], restore_baseline, "confirmation"))
        ranking = rank_probes(confirmed)
        report["search"]["confirmation_ranking"] = [row["probe_id"] for row in ranking]
        report["search"]["policy"] = SEARCH_POLICY
        selected = ranking[0] if ranking else None
        report["search"]["selected"] = selected["probe_id"] if selected else None
        if selected:
            report["search"]["selected_confirmation_comparison"] = _compact_comparison(selected["comparison"])
        if selected:
            control_operation = matched_random_operation(selected, predictor.model.config, report["model"]["checkpoint_id"], history, config.seed)
            control = probe({"probe_id": "random-control", "operation": control_operation,
                             "operation_key": operation_key(report["model"]["checkpoint_id"], control_operation)},
                            confirmation, baseline[len(screening):], restore_baseline, "random_control")
            report["search"]["random_control_comparison"] = _compact_comparison(control["comparison"])
            report["search"]["selected_minus_control_log_score_gain"] = (
                selected["comparison"]["mean_expected_log_score_delta"] - control["comparison"]["mean_expected_log_score_delta"])
        report["probe_restored_weights_sha256"] = weights_hash(predictor.model)
        if report["probe_restored_weights_sha256"] != report["original_weights_sha256"]:
            raise RunStopped("Search did not restore full original weights")
        report["outputs_restored"] = equal(calibration, generation("after_diagnostics", rows[:config.calibration_questions]))
        if not report["outputs_restored"]:
            raise RunStopped("Search changed calibration outputs")
        if selected is None:
            raise SearchFinished("Training search found no eligible candidate; Astra was not called")
        packet = search_packet(report["model"], predictor.model, manifest, original, diagnostics, activations,
                               history, confirmed, selected, control)
        write_json(directory / "editor-evidence.json", packet)
        body = astra.request_body(packet)
        write_json(directory / "editor-request.json", body)
        timeout = remaining()
        report["usage"]["api_calls"] = 1
        report["editor"] = {"status": "request_started", "requested_model": astra.config.model, "usage_known": False}
        event("requesting_astra_proposal")
        audit = astra.request(body, run_id, timeout_seconds=timeout)
        report["editor"] = audit
        report["usage"]["api_usage"] = audit["usage"]
        write_json(directory / "editor-response.json", audit)
        try:
            proposal, patch = validate_proposal(audit["output_text"], packet, run_id)
        except (ValueError, TypeError, KeyError) as error:
            raise ProposalRejected("Astra must propose the ranked, confirmed operation exactly or abstain") from error
        write_json(directory / "proposal.json", proposal)
        if patch is None:
            raise SearchFinished("Astra abstained; no development candidate evaluated")
        remaining()
        write_json(directory / "patch.json", asdict(patch))
        try:
            report["patch"] = editor.apply(patch)
            write_json(directory / "patch-receipt.json", report["patch"])
            candidate = generation("candidate", rows)
        finally:
            if editor.active:
                report["patch"] = editor.rollback()
                write_json(directory / "patch-receipt.json", report["patch"])
        report["outputs_restored"] = equal(calibration, generation("restored", rows[:config.calibration_questions]))
        report["answer_scores_restored"] = answer_scores_restored(restore_baseline, training_scores("training_scores_restored", restoration_rows))
        if not report["outputs_restored"] or not report["answer_scores_restored"]:
            raise RunStopped("Candidate restoration checks failed")
        report["comparison"] = paired_summary(original, candidate)
        report["development_score_comparison"] = compare_answer_scores(generation_scores["original"], generation_scores["candidate"])
        report.update(state="completed", outcome="Ranked Astra candidate evaluated and rolled back")
    except SearchFinished as finished:
        report.update(state="completed", outcome=str(finished))
    except (Exception, KeyboardInterrupt) as error:
        report["state"] = "incomplete"
        if isinstance(error, ProviderFailure):
            report["editor"] = error.audit
            report["usage"]["api_calls"] = error.audit.get("api_calls", report["usage"]["api_calls"])
            report["usage"]["api_usage"] = error.audit.get("usage")
            write_json(directory / "editor-response.json", error.audit)
        safe = str(error) if isinstance(error, (ProviderFailure, RunStopped)) else "Worker failed; inspect saved stage artifacts"
        report["error"] = {"type": type(error).__name__, "message": safe}
        frames = traceback.extract_tb(error.__traceback__)
        if frames:
            frame = frames[-1]
            report["error"]["location"] = {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
        print(f"{type(error).__name__}: {safe}", flush=True)
    finally:
        if editor is not None and editor.active:
            try:
                receipt = editor.rollback()
                if report["patch"]:
                    report["patch"] = receipt
                elif current_probe:
                    current_probe["receipt"] = receipt
            except BaseException:
                report.update(state="incomplete", outcome="Rollback failed; discard worker")
        for item in report["training_sensitivity"]:
            if item["state"] == "running":
                item["state"] = "incomplete"
                write_json(directory / f"{item['probe_id']}.json", item)
        if "original_weights_sha256" in report:
            try:
                report["restored_weights_sha256"] = weights_hash(predictor.model)
                report["weights_restored"] = report["restored_weights_sha256"] == report["original_weights_sha256"]
                if not report["weights_restored"]:
                    report.update(state="incomplete", outcome="Full weight restoration failed")
            except BaseException:
                report.update(state="incomplete", outcome="Full weight restoration verification incomplete", weights_restored=False)
        if report.get("model"):
            try:
                report["usage"].update(predictor.resource_usage())
            except Exception:
                pass
        report["finished_at"] = utc_now()
        report["wall_seconds"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        event("run_finished", state=report["state"], outcome=report["outcome"])
    return {"directory": str(directory.resolve()), "report": report}
