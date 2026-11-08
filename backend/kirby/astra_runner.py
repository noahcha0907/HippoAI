"""One Astra experiment: trusted evaluation, one proposal, unconditional rollback."""

import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from .astra import AstraClient, ProviderFailure, validate_proposal
from .answer_scores import (ANSWER_SCORE_PROTOCOL, SCORE_TOLERANCE, answer_score_record, answer_scores_restored,
                            compare_answer_scores, score_answer_labels, summarize_answer_scores)
from .datasets import load_development
from .diagnostics import PROBE_POLICY, PhiActivationSummary, build_packet, load_diagnostics, select_probe_plan
from .evaluation import PARSER, PROFILE, PROMPT, format_prompt, score, summarize
from .patches import ColumnScale, PhiColumnEditor, weights_hash
from .records import canonical_json, digest, file_hash, read_json, utc_now, write_json
from .reproducibility import PREDICTION_FIELDS
from .runner import RunConfig, RunStopped


@dataclass(frozen=True)
class AstraRunConfig:
    diagnostic_count: int = 8
    calibration_questions: int = 4
    max_seconds: float = 3600

    def __post_init__(self):
        if type(self.diagnostic_count) is not int or not 1 <= self.diagnostic_count <= 16:
            raise ValueError("Astra diagnostics must be 1-16 fixed training questions")
        RunConfig(max_seconds=self.max_seconds, calibration_questions=self.calibration_questions)
        if self.calibration_questions > 8:
            raise ValueError("Astra calibration is limited to 8 questions")


class ProposalRejected(RunStopped):
    pass


def validate_rollback_evidence(path, manifest, metadata, original_hash):
    report = read_json(path)
    config = read_json(path.parent / "configuration.json")
    if (report.get("task") != "controlled_rollback_check" or report.get("state") != "completed"
            or report.get("weights_restored") is not True or report.get("outputs_restored") is not True
            or report.get("patch", {}).get("rollback_verified") is not True):
        raise ValueError("A completed, verified controlled rollback report is required")
    if report.get("configuration_sha256") != digest(config) or config.get("manifest_sha256") != digest(manifest):
        raise ValueError("Rollback configuration or development plan differs")
    for key in ("checkpoint_id", "revision", "precision", "generation_config", "environment"):
        if key not in metadata or report["model"].get(key) != metadata[key]:
            raise ValueError("Rollback model identity, generation settings, or environment differs")
    if (report.get("original_weights_sha256") != original_hash
            or report.get("restored_weights_sha256") != original_hash):
        raise ValueError("Loaded model differs from the verified rollback weights")
    if config["implementation_files_sha256"].get("patches.py") != file_hash(Path(__file__).with_name("patches.py")):
        raise ValueError("Patch implementation changed; repeat the controlled rollback check")


def paired_summary(original, candidate):
    if [p["question_id"] for p in original] != [p["question_id"] for p in candidate]:
        raise ValueError("Candidate comparison requires identical question order")
    a, b = summarize(original), summarize(candidate)
    corrected = sum(not x["correct"] and y["correct"] for x, y in zip(original, candidate))
    regressed = sum(x["correct"] and not y["correct"] for x, y in zip(original, candidate))
    signal = "no_gain"
    if b["correct"] > a["correct"] and b["invalid_count"] <= a["invalid_count"] and b["truncated_count"] <= a["truncated_count"]:
        signal = "improved_on_development"
    elif b["correct"] < a["correct"] or b["invalid_count"] > a["invalid_count"] or b["truncated_count"] > a["truncated_count"]:
        signal = "regressed_on_development"
    return {"original": a, "candidate": b, "corrected": corrected, "regressed": regressed,
            "accuracy_delta": b["accuracy"] - a["accuracy"], "signal": signal,
            "candidate_retained": False, "final_evaluated": False, "regression_benchmark_evaluated": False}


def run_astra_attempt(manifest_path: Path, rollback_report: Path, predictor, model_config: dict,
                      astra: AstraClient, output_root: Path, config: AstraRunConfig = AstraRunConfig(),
                      *, attempt_id: str | None = None):
    rows, manifest = load_development(manifest_path)
    if not config.calibration_questions <= len(rows) <= 256:
        raise ValueError("Astra attempt requires calibration_count through 256 development questions")
    diagnostics = load_diagnostics(manifest, rows, config.diagnostic_count)
    if not rollback_report.is_file():
        raise ValueError("Provide the verified rollback report before starting Astra")
    run_id = str(uuid.UUID(attempt_id)) if attempt_id is not None else str(uuid.uuid4())
    directory = output_root / run_id
    # Exclusive attempt IDs stop retries from duplicating API spend or mutation.
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": 2, "task": "astra_single_proposal", "manifest": manifest,
              "manifest_sha256": digest(manifest), "model": model_config,
              "editor": asdict(astra.config), "limits": asdict(config),
              "profile": PROFILE, "parser": PARSER, "prompt_template": PROMPT,
              "diagnostic_ids": [q.id for q in diagnostics], "rollback_report_sha256": file_hash(rollback_report),
              "selection_policy": "first_frozen_training_ids_v1", "candidate_policy": "always_rollback_v1",
              "answer_score_protocol": ANSWER_SCORE_PROTOCOL, "score_restoration_tolerance": SCORE_TOLERANCE,
              "sensitivity_policy": PROBE_POLICY,
              "original_max_invalid_rate": 0.25,
              "comparison_policy": "positive_correct_count_delta_without_more_invalid_or_truncated_v1",
              "implementation_files_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(directory / "configuration.json", frozen)
    report = {"schema_version": 2, "task": "astra_single_proposal", "run_id": run_id,
              "state": "running", "outcome": "Run incomplete", "started_at": utc_now(),
              "configuration_sha256": digest(frozen), "model": None, "editor": None, "patch": None,
              "stages": {}, "comparison": None, "weights_restored": None, "outputs_restored": None,
              "candidate_retained": False, "final": None, "regression": None,
              "answer_score_stages": {}, "training_sensitivity": [], "candidate_training_comparison": None,
              "answer_scores_restored": None,
              "usage": {"forward_calls": 0, "generated_tokens": 0, "api_calls": 0, "api_usage": None,
                        "answer_score_forward_calls": 0},
              "limitations": ["One Astra proposal, one layer, at most 16 evidenced columns; no automatic retries.",
                              "Development-only signal, not demonstrated repair or final-test performance.",
                              "Activation magnitudes do not establish causal damage.",
                              "Four fixed training-only probes precede Astra; no development evaluation of those probes.",
                              "Teacher-forced label scores are diagnostics, not benchmark accuracy or calibrated confidence.",
                              "forward_calls counts generation requests; answer_score_forward_calls counts separate scoring forwards.",
                              "All edits are rolled back, including favorable development results; no weights exported.",
                              "API usage may be unknown after a transport timeout; the call may still be billed.",
                              "Time limits are cooperative; rollback takes priority over the deadline."]}
    write_json(directory / "report.json", report)
    started = time.monotonic()
    editor = None
    probe_editor = None
    sequence = 0

    def event(kind, **details):
        nonlocal sequence
        sequence += 1
        report["last_event"] = {"event": kind, **details}
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"sequence": sequence, "at": utc_now(), "event": kind, **details}) + "\n")
        write_json(directory / "report.json", report)
        if kind != "question_completed":
            print(kind + (": " + details["stage"] if "stage" in details else ""), flush=True)
        elif details["completed"] % 16 == 0 or details["completed"] == details["total"]:
            print(f"{details['stage']}: {details['completed']}/{details['total']}", flush=True)

    def remaining():
        seconds = config.max_seconds - (time.monotonic() - started)
        if seconds <= 0:
            raise RunStopped("Wall-time budget exhausted")
        return seconds

    def measure(stage, questions):
        event("measuring", stage=stage, questions=len(questions))
        predictions = []
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for q in questions:
                remaining()
                before = time.monotonic()
                report["usage"]["forward_calls"] += 1
                generated = predictor.generate(format_prompt(q.input))
                report["usage"]["generated_tokens"] += generated.generated_tokens
                prediction = score(q, generated, time.monotonic() - before)
                predictions.append(prediction)
                stream.write(canonical_json(prediction) + "\n")
                stream.flush()
                report["stages"][stage] = {"planned_questions": len(questions), "complete": len(predictions) == len(questions),
                                          **summarize(predictions)}
                event("question_completed", stage=stage, completed=len(predictions), total=len(questions))
        remaining()
        return predictions

    def equal(a, b):
        return len(a) == len(b) and all(all(x[k] == y[k] for k in PREDICTION_FIELDS) for x, y in zip(a, b))

    def measure_scores(stage):
        event("scoring_training_answers", stage=stage, questions=len(diagnostics))
        scores = []
        def before_forward():
            remaining()
            report["usage"]["answer_score_forward_calls"] += 1
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for question in diagnostics:
                values = score_answer_labels(predictor, format_prompt(question.input), before_forward=before_forward)
                scores.append(answer_score_record(question, values))
                stream.write(canonical_json(scores[-1]) + "\n")
                stream.flush()
                report["answer_score_stages"][stage] = {
                    "planned_questions": len(diagnostics), "complete": len(scores) == len(diagnostics),
                    **summarize_answer_scores(scores)}
                event("question_completed", stage=stage, completed=len(scores), total=len(diagnostics))
        remaining()
        return scores

    try:
        # Preflight credentials/SDK without making an API request or loading weights.
        astra.connect()
        remaining()
        event("loading_model")
        report["model"] = predictor.load()
        write_json(directory / "model.json", report["model"])
        event("verifying_rollback_prerequisite")
        report["original_weights_sha256"] = weights_hash(predictor.model)
        if file_hash(rollback_report) != frozen["rollback_report_sha256"]:
            raise ValueError("Rollback evidence changed during model loading")
        validate_rollback_evidence(rollback_report, manifest, report["model"], report["original_weights_sha256"])
        calibration_rows = rows[:config.calibration_questions]
        calibration = measure("calibration", calibration_rows)
        original = measure("original", rows)
        metrics = summarize(original)
        if (not equal(calibration, original[:config.calibration_questions])
                or metrics["invalid_rate"] > 0.25 or metrics["truncated_count"]):
            raise RunStopped("Original model is unstable or fails the answer protocol; Astra was not called")
        with PhiActivationSummary(predictor.model) as collector:
            diagnostic_predictions = measure("diagnostic_train", diagnostics)
        activations = collector.summary()
        # The whole plan is fixed from training activations before observing
        # any edit effects. Each probe starts from the same original weights.
        probe_plan = select_probe_plan(activations)
        write_json(directory / "sensitivity-plan.json", {"policy": PROBE_POLICY, "probes": probe_plan,
                                                        "question_ids": [q.id for q in diagnostics],
                                                        "base_checkpoint_id": report["model"]["checkpoint_id"]})
        training_scores = measure_scores("answer_scores_original")
        repeated_scores = measure_scores("answer_scores_repeat")
        if not answer_scores_restored(training_scores, repeated_scores):
            raise RunStopped("Original training answer scores are unstable; Astra was not called")
        probe_editor = PhiColumnEditor(predictor.model, report["model"]["checkpoint_id"])
        for planned in probe_plan:
            remaining()
            probe_id = planned["probe_id"]
            operation = {key: planned[key] for key in ("operation", "layer", "columns", "scale")}
            probe = {"probe_id": probe_id, "operation": operation, "state": "running",
                     "receipt": None, "scores_restored": None, "comparison": None}
            report["training_sensitivity"].append(probe)
            probe_file = directory / f"{probe_id}.json"
            write_json(probe_file, probe)
            event("applying_training_probe", probe_id=probe_id)
            try:
                probe["receipt"] = probe_editor.apply(ColumnScale(
                    report["model"]["checkpoint_id"], f"{run_id}-{probe_id}",
                    planned["layer"], tuple(planned["columns"]), planned["scale"]))
                write_json(probe_file, probe)
                changed_scores = measure_scores(f"answer_scores_{probe_id}")
            finally:
                # Restore before logging, including scoring failures/Ctrl+C.
                if probe_editor.active:
                    probe["receipt"] = probe_editor.rollback()
                write_json(probe_file, probe)
            restored_scores = measure_scores(f"answer_scores_{probe_id}_restored")
            probe["scores_restored"] = answer_scores_restored(training_scores, restored_scores)
            if not probe["scores_restored"]:
                raise RunStopped("Training scores changed after probe rollback; Astra was not called")
            probe["comparison"] = compare_answer_scores(training_scores, changed_scores)
            probe["state"] = "completed"
            write_json(probe_file, probe)
            event("training_probe_completed", probe_id=probe_id,
                  mean_expected_margin_delta=probe["comparison"]["mean_expected_margin_delta"])
        event("verifying_training_probe_weights_restored")
        report["probe_restored_weights_sha256"] = weights_hash(predictor.model)
        if report["probe_restored_weights_sha256"] != report["original_weights_sha256"]:
            raise RunStopped("Weights changed after training probes; Astra was not called")
        uninstrumented = measure("after_diagnostics", calibration_rows)
        if not equal(calibration, uninstrumented):
            raise RunStopped("Outputs changed after diagnostics; Astra was not called")
        packet = build_packet(report["model"], predictor.model, diagnostics, diagnostic_predictions, activations, metrics,
                              training_scores, report["training_sensitivity"])
        write_json(directory / "editor-evidence.json", packet)
        body = astra.request_body(packet)
        write_json(directory / "editor-request.json", body)
        timeout = remaining()
        report["usage"]["api_calls"] = 1
        report["editor"] = {"status": "request_started", "requested_model": astra.config.model,
                            "request_body_sha256": digest(body), "usage_known": False}
        event("requesting_astra_proposal")
        audit = astra.request(body, run_id, timeout_seconds=timeout)
        report["editor"] = audit
        report["usage"]["api_usage"] = audit["usage"]
        write_json(directory / "editor-response.json", audit)
        remaining()
        try:
            proposal, patch = validate_proposal(audit["output_text"], packet, run_id)
        except (ValueError, TypeError, KeyError) as error:
            raise ProposalRejected("Astra proposal failed local schema, evidence, or operation validation") from error
        write_json(directory / "proposal.json", proposal)
        if patch is None:
            report.update(state="completed", outcome="Astra abstained; no edit applied")
        else:
            editor = PhiColumnEditor(predictor.model, report["model"]["checkpoint_id"])
            write_json(directory / "patch.json", asdict(patch))
            remaining()
            event("validating_and_applying_astra_patch")
            try:
                try:
                    report["patch"] = editor.apply(patch)
                except ValueError as error:
                    raise ProposalRejected("Astra patch rejected by numerical bounds, shape, finiteness, or no-op checks") from error
                write_json(directory / "patch-receipt.json", report["patch"])
                candidate_training_scores = measure_scores("answer_scores_candidate")
                report["candidate_training_comparison"] = compare_answer_scores(training_scores, candidate_training_scores)
                candidate = measure("candidate", rows)
            finally:
                # Roll back BEFORE any fallible logging, even on exceptions or Ctrl+C.
                if editor.active:
                    report["patch"] = editor.rollback()
                    write_json(directory / "patch-receipt.json", report["patch"])
            restored_training_scores = measure_scores("answer_scores_restored")
            report["answer_scores_restored"] = answer_scores_restored(training_scores, restored_training_scores)
            if not report["answer_scores_restored"]:
                raise RunStopped("Restored training answer scores differ; discard worker and investigate determinism")
            restored = measure("restored", calibration_rows)
            report["outputs_restored"] = equal(calibration, restored)
            if not report["outputs_restored"]:
                raise RunStopped("Restored outputs differ; discard worker and investigate determinism")
            report["comparison"] = paired_summary(original, candidate)
            report.update(state="completed", outcome="Astra candidate evaluated and rolled back")
    except (Exception, KeyboardInterrupt) as error:
        report["state"] = "incomplete"
        if isinstance(error, ProviderFailure):
            report["editor"] = error.audit
            report["usage"]["api_calls"] = error.audit.get("api_calls", report["usage"]["api_calls"])
            report["usage"]["api_usage"] = error.audit.get("usage")
            write_json(directory / "editor-response.json", error.audit)
        if isinstance(error, ProposalRejected):
            report["outcome"] = "Astra proposal rejected; no candidate retained"
        safe = str(error) if isinstance(error, (ProviderFailure, RunStopped)) else "Worker failed; inspect stage artifacts (exception text omitted)"
        report["error"] = {"type": type(error).__name__, "message": safe}
        # Record file/line without exception bodies, headers, or local values.
        import traceback
        frames = traceback.extract_tb(error.__traceback__)
        if frames:
            frame = frames[-1]
            report["error"]["location"] = {"file": Path(frame.filename).name,
                                              "line": frame.lineno, "function": frame.name}
        print(f"{type(error).__name__}: {safe}", flush=True)
    finally:
        if probe_editor is not None and probe_editor.active:
            try:
                report["training_sensitivity"][-1]["receipt"] = probe_editor.rollback()
            except BaseException:
                report.update(state="incomplete", outcome="Training probe rollback failed; discard worker")
                report["rollback_error"] = "Could not restore a training probe tensor"
        for probe in report["training_sensitivity"]:
            if probe["state"] == "running":
                probe["state"] = "incomplete"
                write_json(directory / f"{probe['probe_id']}.json", probe)
        if editor is not None:
            if editor.active:
                try:
                    editor.rollback()
                except BaseException:
                    report.update(state="incomplete", outcome="Rollback failed; discard worker")
                    report["rollback_error"] = "Could not restore the edited tensor"
            if editor.receipt:
                report["patch"] = dict(editor.receipt)
                write_json(directory / "patch-receipt.json", report["patch"])
        if "original_weights_sha256" in report:
            try:
                event("verifying_original_weights_restored")
                report["restored_weights_sha256"] = weights_hash(predictor.model)
                report["weights_restored"] = report["restored_weights_sha256"] == report["original_weights_sha256"]
                if not report["weights_restored"]:
                    report.update(state="incomplete", outcome="Weight restoration verification failed")
            except BaseException:
                report.update(state="incomplete", outcome="Weight restoration verification incomplete")
                report["weights_restored"] = False
        report["finished_at"] = utc_now()
        report["wall_seconds"] = time.monotonic() - started
        if report["model"] is not None:
            try:
                report["usage"].update(predictor.resource_usage())
            except Exception:
                pass
        write_json(directory / "report.json", report)
        (directory / "report.md").write_text(
            f"# Astra single-proposal experiment\n\n{report['outcome']}\n\n"
            f"State: {report['state']}. Weights restored: {report['weights_restored']}. "
            f"Outputs restored: {report['outputs_restored']}.\n\n"
            f"Comparison: {canonical_json(report['comparison'])}\n\n"
            f"Training answer-score comparison: {canonical_json(report['candidate_training_comparison'])}\n\n"
            f"Training answer scores restored: {report['answer_scores_restored']}.\n\n"
            "Development-only experiment. No candidate retained, exported, or evaluated on final/regression data.\n",
            encoding="utf-8")
        event("run_finished", state=report["state"], outcome=report["outcome"])
    return {"directory": str(directory.resolve()), "report": report}
