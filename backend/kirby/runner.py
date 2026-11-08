"""Synchronous worker entrypoint; no HTTP request or editor owns evaluation.

Every run owns a new directory. Predictions/events are flushed after each
question. A future queue can call this worker without changing scoring.
"""

import math
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

from .datasets import Question, load_development
from .evaluation import Generation, PARSER, PROFILE, PROMPT, format_prompt, score, summarize
from .records import canonical_json, digest, file_hash, utc_now, write_json


class Predictor(Protocol):
    def load(self) -> dict: ...
    def generate(self, prompt: str) -> Generation: ...
    def resource_usage(self) -> dict: ...


@dataclass(frozen=True)
class RunConfig:
    max_seconds: float = 1800
    calibration_questions: int = 4
    max_invalid_rate: float = 0.25

    def __post_init__(self):
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")
        if type(self.calibration_questions) is not int or self.calibration_questions < 1:
            raise ValueError("calibration_questions must be a positive integer")
        if not math.isfinite(self.max_invalid_rate) or not 0 <= self.max_invalid_rate <= 1:
            raise ValueError("max_invalid_rate must be between 0 and 1")


class RunStopped(Exception):
    pass


def run_baseline(manifest_path: Path, predictor: Predictor, model_config: dict,
                 output_root: Path, config: RunConfig = RunConfig(), *,
                 cancelled: Callable[[], bool] = lambda: False) -> dict:
    rows, manifest = load_development(manifest_path)
    if config.calibration_questions > len(rows):
        raise ValueError("Calibration count exceeds the selected development set")
    implementation = {path.name: file_hash(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    frozen = {"schema_version": 1, "task": "evaluate_model", "profile": PROFILE,
              "implementation_files_sha256": implementation,
              "parser": PARSER, "prompt_template": PROMPT, "model": model_config,
              "limits": asdict(config), "manifest": manifest, "manifest_sha256": digest(manifest),
              "calibration_question_ids": [q.id for q in rows[:config.calibration_questions]]}
    run_id = str(uuid.uuid4())
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "configuration.json", frozen)
    started = time.monotonic()
    predictions: list[dict] = []
    report = {"schema_version": 1, "run_id": run_id, "task": "evaluate_model",
              "state": "running", "outcome": "Run incomplete", "started_at": utc_now(),
              "configuration_sha256": digest(frozen), "model": None, "calibration": None,
              "development": None, "final": None, "final_unavailable_reason": "Final evaluation is not implemented in this baseline worker",
              "regression": None, "regression_unavailable_reason": "MedMCQA adapter is not implemented yet",
              "comparison": None, "editor": None, "patches": [],
              "limitations": ["Development-only baseline; no improvement or repair has been measured.",
                              "Public benchmark overlap with model pretraining is unknown.",
                              "Exact normalized duplicates checked during preparation; near duplicates are not checked.",
                              "Repeated option text is preserved and recorded in source metadata; scoring uses the supplied answer_idx only.",
                              "Deterministic decoding does not guarantee reproducibility across hardware.",
                              "Time limits and cancellation are cooperative between inference calls."]}
    write_json(directory / "report.json", report)
    usage = {"forward_calls": 0, "generated_tokens": 0, "api_calls": 0}
    sequence = 0

    def event(kind: str, **details):
        nonlocal sequence
        sequence += 1
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"sequence": sequence, "at": utc_now(), "event": kind, **details}) + "\n")

    def check_budget():
        if cancelled():
            raise RunStopped("Cancellation requested")
        if time.monotonic() - started >= config.max_seconds:
            raise RunStopped("Wall-time budget exhausted")

    def predict(question: Question) -> dict:
        check_budget()
        before = time.monotonic()
        usage["forward_calls"] += 1
        generated = predictor.generate(format_prompt(question.input))
        usage["generated_tokens"] += generated.generated_tokens
        return score(question, generated, time.monotonic() - before)

    try:
        check_budget()
        event("loading_model")
        report["model"] = predictor.load()
        report["limitations"].extend(report["model"].get("limitations", []))
        write_json(directory / "model.json", report["model"])
        check_budget()
        event("calibrating", questions=config.calibration_questions, passes=2)
        calibration = []
        with (directory / "calibration.jsonl").open("w", encoding="utf-8") as stream:
            for pass_number in (1, 2):
                current = []
                for question in rows[:config.calibration_questions]:
                    prediction = predict(question)
                    current.append(prediction)
                    stream.write(canonical_json({"pass": pass_number, **prediction}) + "\n")
                    stream.flush()
                    check_budget()
                calibration.append(current)
        stable_keys = ("text", "generated_tokens", "truncated", "predicted")
        stable = all(all(a[k] == b[k] for k in stable_keys) for a, b in zip(*calibration))
        metrics = summarize(calibration[0])
        protocol_passed = metrics["invalid_rate"] <= config.max_invalid_rate and metrics["truncated_count"] == 0
        report["calibration"] = {"stable": stable, "first_pass": metrics,
                                 "protocol_passed": protocol_passed,
                                 "max_invalid_rate": config.max_invalid_rate,
                                 "question_ids": frozen["calibration_question_ids"]}
        if not stable:
            raise RunStopped("Unchanged calibration passes disagree; investigate inference before benchmarking")
        if not protocol_passed:
            raise RunStopped(
                f"Calibration answer protocol failed: {metrics['invalid_count']}/{metrics['total']} invalid "
                f"answers (allowed rate: {config.max_invalid_rate:.0%}), "
                f"{metrics['truncated_count']}/{metrics['total']} truncated (allowed: 0). "
                "Inspect calibration.jsonl for generated text before changing the prompt or token allowance."
            )
        event("measuring_baseline", total=len(rows))
        with (directory / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for question in rows:
                prediction = predict(question)
                predictions.append(prediction)
                stream.write(canonical_json(prediction) + "\n")
                stream.flush()
                event("question_completed", completed=len(predictions), total=len(rows))
                check_budget()
        report["state"] = "completed"
        # An audit has no before/after comparison; it must not receive a repair outcome.
        report["outcome"] = "Baseline evaluated"
    except (Exception, KeyboardInterrupt) as error:
        report["state"] = "incomplete"
        # Avoid persisting exception strings that might contain provider credentials.
        report["error"] = {"type": type(error).__name__,
                           "message": str(error) if isinstance(error, RunStopped) else "Worker failed; inspect the local traceback"}
        if not isinstance(error, (RunStopped, KeyboardInterrupt)):
            import traceback
            traceback.print_exc()
    finally:
        report["finished_at"] = utc_now()
        report["wall_seconds"] = time.monotonic() - started
        report["development"] = {"complete": report["state"] == "completed",
                                  "planned_questions": len(rows), **summarize(predictions)}
        report["usage"] = usage
        if report["model"] is not None:
            try:
                report["usage"].update(predictor.resource_usage())
            except Exception:
                report["limitations"].append("GPU resource counters could not be collected")
        write_json(directory / "report.json", report)
        (directory / "report.md").write_text(render_markdown(report), encoding="utf-8")
        event("run_finished", state=report["state"], outcome=report["outcome"])
    return {"directory": str(directory.resolve()), "report": report}


def render_markdown(report: dict) -> str:
    metrics = report["development"]
    accuracy = "unavailable" if metrics["accuracy"] is None else f'{100 * metrics["accuracy"]:.2f}%'
    return (f'# Kirby baseline report\n\nRun: {report["run_id"]}\n\n'
            f'Outcome: **{report["outcome"]}**\n\n'
            f'Development accuracy: {metrics["correct"]}/{metrics["total"]} ({accuracy}). '
            f'Planned questions: {metrics["planned_questions"]}. Complete: {metrics["complete"]}.\n\n'
            f'Invalid answers: {metrics["invalid_count"]}. Truncated answers: {metrics["truncated_count"]}.\n\n'
            'Final test and regression benchmark: unavailable. No candidate was edited or compared.\n\n'
            + '\n'.join(f'- {item}' for item in report["limitations"]) + '\n')
