"""Compare saved baseline evidence without loading weights or final questions."""

import json
from pathlib import Path

from .evaluation import summarize
from .records import digest, file_hash, read_json


PREDICTION_FIELDS = ("question_id", "fingerprint", "expected", "text", "predicted",
                     "valid", "correct", "generated_tokens", "truncated")
EVALUATOR_MODULES = ("datasets.py", "evaluation.py", "jsl.py", "medgemma.py", "runner.py")


def _load(directory: Path):
    report = read_json(directory / "report.json")
    config = read_json(directory / "configuration.json")
    model = read_json(directory / "model.json")
    if (report.get("task") != "evaluate_model" or report.get("state") != "completed"
            or not report.get("development", {}).get("complete")
            or not report.get("calibration", {}).get("stable")):
        raise ValueError("Comparison requires completed, stable development baselines")
    if report.get("configuration_sha256") != digest(config):
        raise ValueError("Configuration digest does not match its report")
    if config.get("manifest_sha256") != digest(config["manifest"]):
        raise ValueError("Manifest digest mismatch")
    if config["manifest"].get("phase") != "development":
        raise ValueError("Comparison accepts development runs only")
    if model != report["model"]:
        raise ValueError("Model metadata differs from its run report")
    if digest(model["files"]) != model["checkpoint_id"]:
        raise ValueError("Checkpoint identity does not match its file hashes")
    predictions = [json.loads(line) for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = config["manifest"]["question_ids"]
    if not ids or len(set(ids)) != len(ids) or [p["question_id"] for p in predictions] != ids:
        raise ValueError("Prediction records do not match the full frozen question order")
    actual = summarize(predictions)
    if any(report["development"].get(key) != value for key, value in actual.items()):
        raise ValueError("Prediction summary does not match the saved report")
    if report["development"]["planned_questions"] != len(ids):
        raise ValueError("Planned question count mismatch")
    return report, config, model, predictions


def compare_baselines(first: Path, second: Path) -> dict:
    if first.resolve() == second.resolve():
        raise ValueError("Select two distinct baseline runs")
    a_report, a_config, a_model, a_predictions = _load(first)
    b_report, b_config, b_model, b_predictions = _load(second)
    if a_report["run_id"] == b_report["run_id"]:
        raise ValueError("Copied evidence from one run is not a repeat")
    config_fields = ("task", "profile", "parser", "prompt_template", "limits",
                     "manifest_sha256", "calibration_question_ids")
    mismatches = [f"configuration.{key}" for key in config_fields if a_config[key] != b_config[key]]
    # Download permission is operational, not an inference setting.
    clean_model = lambda c: {k: v for k, v in c["model"].items() if k != "local_files_only"}
    if clean_model(a_config) != clean_model(b_config):
        mismatches.append("configuration.model")
    for name in EVALUATOR_MODULES:
        left = a_config["implementation_files_sha256"].get(name)
        right = b_config["implementation_files_sha256"].get(name)
        if not left or not right or left != right:
            mismatches.append(f"implementation.{name}")
    for key in ("checkpoint_id", "generation_config", "environment", "rendered_template_probe"):
        if key not in a_model or key not in b_model or a_model[key] != b_model[key]:
            mismatches.append(f"model.{key}")
    differences = []
    if not mismatches:
        for left, right in zip(a_predictions, b_predictions):
            fields = [key for key in PREDICTION_FIELDS if left[key] != right[key]]
            if fields:
                differences.append({"question_id": left["question_id"], "fields": fields})
    return {"schema_version": 1, "task": "compare_unchanged_baselines",
            "reproduced": not mismatches and not differences,
            "comparable": not mismatches, "configuration_mismatches": mismatches,
            "prediction_differences": differences, "questions": len(a_predictions),
            "runs": [{"directory": str(path.resolve()), "run_id": report["run_id"],
                      "report_sha256": file_hash(path / "report.json"),
                      "predictions_sha256": file_hash(path / "predictions.jsonl")}
                     for path, report in ((first, a_report), (second, b_report))],
            "first_accuracy": a_report["development"]["accuracy"],
            "second_accuracy": b_report["development"]["accuracy"],
            "limitations": ["Exact prediction reproducibility for these saved runs only; no repair or final-test claim.",
                            "Latency and memory counters are excluded from prediction equality."]}
