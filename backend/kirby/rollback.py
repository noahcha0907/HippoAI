"""Controlled Phi edit/restore experiment; never retains or exports edited weights."""

import time
import uuid
from dataclasses import asdict
from pathlib import Path

from .datasets import load_development
from .evaluation import PARSER, PROFILE, PROMPT, format_prompt, score, summarize
from .patches import ColumnScale, PhiColumnEditor, weights_hash
from .records import canonical_json, digest, file_hash, utc_now, write_json
from .reproducibility import PREDICTION_FIELDS
from .runner import RunConfig, RunStopped


def run_rollback_check(manifest_path: Path, predictor, model_config: dict, output_root: Path,
                       *, questions: int = 4, layer: int = 0, column: int = 0,
                       scale: float = 0.97, max_seconds: float = 3600) -> dict:
    rows, manifest = load_development(manifest_path)
    if type(questions) is not int or not 1 <= questions <= min(8, len(rows)):
        raise ValueError("Rollback check requires 1-8 development questions available in the plan")
    limits = RunConfig(max_seconds=max_seconds, calibration_questions=questions)
    # Validate edit arguments before loading the model; the actual base is bound after load.
    ColumnScale("0" * 64, "validate", layer, (column,), scale)
    rows = rows[:questions]
    run_id = str(uuid.uuid4())
    directory = output_root / run_id
    directory.mkdir(parents=True, exist_ok=False)
    frozen = {"schema_version": 1, "task": "controlled_rollback_check",
              "model": model_config, "manifest": manifest, "manifest_sha256": digest(manifest),
              "profile": PROFILE, "parser": PARSER, "prompt_template": PROMPT,
              "question_ids": [q.id for q in rows], "limits": asdict(limits),
              "edit": {"layer": layer, "column": column, "scale": scale},
              "implementation_files_sha256": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}}
    write_json(directory / "configuration.json", frozen)
    report = {"schema_version": 1, "task": "controlled_rollback_check", "run_id": run_id,
              "state": "running", "outcome": "Run incomplete", "started_at": utc_now(),
              "configuration_sha256": digest(frozen), "model": None, "patch": None,
              "weights_restored": False, "outputs_restored": False, "stages": {},
              "usage": {"forward_calls": 0, "generated_tokens": 0, "api_calls": 0},
              "limitations": ["Controlled execution test, not an Astra proposal or beneficial repair.",
                              "No final questions, checkpoint export, or retained candidate.",
                              "A killed worker must be discarded; each new worker reloads the immutable checkpoint.",
                              "Time limits are cooperative; rollback takes priority over the time budget."]}
    write_json(directory / "report.json", report)
    started = time.monotonic()
    editor = None
    sequence = 0

    def event(kind, **details):
        nonlocal sequence
        sequence += 1
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(canonical_json({"sequence": sequence, "at": utc_now(), "event": kind, **details}) + "\n")
        write_json(directory / "report.json", report)
        print(kind, flush=True)

    def budget():
        if time.monotonic() - started >= max_seconds:
            raise RunStopped("Wall-time budget exhausted")

    def measure(stage):
        event("measuring", stage=stage)
        outputs = []
        with (directory / f"{stage}.jsonl").open("w", encoding="utf-8") as stream:
            for question in rows:
                budget()
                before = time.monotonic()
                report["usage"]["forward_calls"] += 1
                generation = predictor.generate(format_prompt(question.input))
                report["usage"]["generated_tokens"] += generation.generated_tokens
                prediction = score(question, generation, time.monotonic() - before)
                outputs.append(prediction)
                stream.write(canonical_json(prediction) + "\n")
                stream.flush()
                report["stages"][stage] = summarize(outputs)
                budget()
        return outputs

    def same(left, right):
        return all(all(a[k] == b[k] for k in PREDICTION_FIELDS) for a, b in zip(left, right))

    try:
        event("loading_model")
        report["model"] = predictor.load()
        write_json(directory / "model.json", report["model"])
        budget()
        event("hashing_original_weights")
        report["original_weights_sha256"] = weights_hash(predictor.model)
        baseline = measure("original")
        repeated = measure("original_repeat")
        metrics = report["stages"]["original"]
        if not same(baseline, repeated) or metrics["invalid_rate"] > limits.max_invalid_rate or metrics["truncated_count"]:
            raise RunStopped("Original outputs are unstable or fail the answer protocol; no edit applied")
        editor = PhiColumnEditor(predictor.model, report["model"]["checkpoint_id"])
        patch = ColumnScale(report["model"]["checkpoint_id"], run_id, layer, (column,), scale)
        write_json(directory / "patch.json", asdict(patch))
        budget()
        event("applying_controlled_edit")
        try:
            report["patch"] = editor.apply(patch)
            write_json(directory / "patch-receipt.json", report["patch"])
            measure("edited")
        finally:
            if editor.active:
                event("restoring_original_slice")
                report["patch"] = editor.rollback()
                write_json(directory / "patch-receipt.json", report["patch"])
        event("hashing_restored_weights")
        report["restored_weights_sha256"] = weights_hash(predictor.model)
        report["weights_restored"] = report["original_weights_sha256"] == report["restored_weights_sha256"]
        if not report["weights_restored"]:
            raise RunStopped("Full weight hash differs after rollback; discard worker")
        restored = measure("restored")
        report["outputs_restored"] = same(baseline, restored)
        if not report["outputs_restored"]:
            raise RunStopped("Restored weights produced different outputs; investigate determinism")
        report.update(state="completed", outcome="Controlled rollback verified")
    except (Exception, KeyboardInterrupt) as error:
        report["state"] = "incomplete"
        report["error"] = {"type": type(error).__name__, "message": str(error) if isinstance(error, RunStopped)
                           else "Worker failed; inspect local traceback and rollback evidence"}
        if not isinstance(error, (RunStopped, KeyboardInterrupt)):
            import traceback
            traceback.print_exc()
    finally:
        # Also cover failures in journaling and interrupts during the first rollback.
        if editor is not None:
            if editor.active:
                try:
                    report["patch"] = editor.rollback()
                except BaseException:
                    report["state"] = "incomplete"
                    report["rollback_error"] = "Rollback failed; discard worker and reload checkpoint"
            if editor.receipt is not None:
                report["patch"] = dict(editor.receipt)
                write_json(directory / "patch-receipt.json", report["patch"])
        report["finished_at"] = utc_now()
        report["wall_seconds"] = time.monotonic() - started
        write_json(directory / "report.json", report)
        (directory / "report.md").write_text(
            f"# Controlled rollback check\n\n{report['outcome']}\n\n"
            f"State: {report['state']}. Full weights restored: {report['weights_restored']}. "
            f"Outputs restored: {report['outputs_restored']}.\n\n"
            "This is an execution test, not demonstrated model improvement.\n", encoding="utf-8")
        event("run_finished", state=report["state"])
    return {"directory": str(directory.resolve()), "report": report}
