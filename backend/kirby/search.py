"""Frozen search policy, deterministic ranking, and prior-candidate identity."""

import random
from dataclasses import dataclass
from pathlib import Path

from .answer_scores import SCORE_TOLERANCE
from .evaluation import PARSER, PROFILE
from .patches import ColumnScale
from .records import digest, file_hash, read_json
from .runner import RunConfig


@dataclass(frozen=True)
class SearchConfig:
    development_count: int = 172
    diagnostic_count: int = 64
    screening_count: int = 8
    layer_count: int = 8
    confirm_count: int = 2
    calibration_questions: int = 4
    max_seconds: float = 3600
    seed: int = 17

    def __post_init__(self):
        for name, low, high in (("development_count", 1, 172), ("diagnostic_count", 2, 64),
                                ("screening_count", 1, 16), ("layer_count", 1, 8),
                                ("confirm_count", 1, 2), ("calibration_questions", 1, 8)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Invalid search {name}: expected {low}..{high}")
        if self.screening_count >= self.diagnostic_count or self.calibration_questions > self.development_count:
            raise ValueError("Search needs separate screening/confirmation questions and enough development rows")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("Invalid search seed")
        RunConfig(max_seconds=self.max_seconds, calibration_questions=self.calibration_questions)


SEARCH_POLICY = {
    "version": "layer_stratified_screen_confirm_v1",
    "scales": [0.9, 1.1],
    "targets": "one top activation-times-column-norm column per evenly spaced layer",
    "ranking": "descending mean expected log-score delta, then mean margin delta, then operation identity",
    "eligibility": "positive log-score and margin deltas above 1e-6, no loss of preferred-correct count",
    "confirmation": "rank at most two screen winners on disjoint remaining training questions",
    "random_control": "same layer, scale, column count; seeded other column; training only; not used to select winner",
    "development": "first 172 frozen manifest IDs, chosen without looking at model outcomes",
    "retention": "all edits restored; no cumulative edits, export, or final evaluation",
}


def operation_key(checkpoint, operation):
    patch = ColumnScale(checkpoint, "identity", operation["layer"], tuple(operation["columns"]), operation["scale"])
    if operation["operation"] != "scale_columns":
        raise ValueError("Unsupported history operation")
    return digest({"base_checkpoint_id": checkpoint, "operation": "scale_columns", "layer": patch.layer,
                   "columns": sorted(patch.columns), "scale": float(patch.scale)})


def read_attempt_history(root, metadata, manifest, original_hash):
    """Use only compatible completed, restored development evaluations.

    Different development cohort sizes are recorded, never pooled. History
    suppresses exact previously evaluated edits; old training probes are not
    treated as development evaluations or reusable score evidence.
    """
    accepted, ignored = [], []
    for path in sorted(Path(root).glob("*/report.json")):
        try:
            report = read_json(path)
            if report.get("state") != "completed" or not report.get("comparison") or not report.get("patch"):
                continue
            config = read_json(path.parent / "configuration.json")
            if report.get("configuration_sha256") != digest(config):
                raise ValueError("configuration hash mismatch")
            if any(report.get("model", {}).get(key) != metadata.get(key) for key in
                   ("checkpoint_id", "revision", "precision", "generation_config")):
                continue
            if config.get("manifest", {}).get("sources") != manifest.get("sources"):
                continue
            if config.get("profile") != PROFILE or config.get("parser") != PARSER:
                continue
            if report.get("weights_restored") is not True or report.get("original_weights_sha256") != original_hash:
                continue
            patch = report["patch"]
            if patch.get("rollback_verified") is not True or report.get("restored_weights_sha256") != original_hash:
                raise ValueError("restoration evidence incomplete")
            operation = {key: patch[key] for key in ("operation", "layer", "columns", "scale")}
            accepted.append({"run_id": path.parent.name, "report_sha256": file_hash(path), "operation": operation,
                             "operation_key": operation_key(metadata["checkpoint_id"], operation),
                             "development_questions": report["comparison"]["original"]["total"],
                             "accuracy_delta": report["comparison"]["accuracy_delta"],
                             "signal": report["comparison"]["signal"],
                             "development_ids_sha256": digest(config.get("development_ids", config["manifest"]["question_ids"]))})
        except (ValueError, KeyError, TypeError, OSError):
            ignored.append({"run_id": path.parent.name, "reason": "invalid or incomplete history evidence"})
    return {"accepted": accepted, "ignored": ignored,
            "policy": "skip exact previously development-evaluated operations on this base; never pool differing cohorts"}


def screening_plan(activations, checkpoint, history, config):
    layers = sorted(activations, key=lambda row: row["layer"])
    count = min(config.layer_count, len(layers))
    indices = [0] if count == 1 else [i * (len(layers) - 1) // (count - 1) for i in range(count)]
    excluded = {row["operation_key"] for row in history["accepted"]}
    planned = []
    for index in indices:
        layer = layers[index]
        columns = sorted(layer["columns"], key=lambda item: (
            -(item["mean_absolute_activation"] * item["weight_column_l2"]), item["column"]))
        for column in columns:
            operations = [{"operation": "scale_columns", "layer": layer["layer"],
                           "columns": [column["column"]], "scale": scale} for scale in SEARCH_POLICY["scales"]]
            # Choose a new target if either direction was already development-evaluated.
            if all(operation_key(checkpoint, operation) not in excluded for operation in operations):
                for operation in operations:
                    planned.append({"probe_id": f"screen-{len(planned) + 1}", "operation": operation,
                                    "operation_key": operation_key(checkpoint, operation)})
                break
    return planned


def rank_probes(probes):
    eligible = []
    for row in probes:
        comparison = row.get("comparison") or {}
        if (row.get("state") == "completed" and row.get("scores_restored") is True
                and comparison.get("mean_expected_log_score_delta", 0) > SCORE_TOLERANCE
                and comparison.get("mean_expected_margin_delta", 0) > SCORE_TOLERANCE
                and comparison["changed"]["preferred_correct"] >= comparison["original"]["preferred_correct"]):
            eligible.append(row)
    return sorted(eligible, key=lambda row: (-row["comparison"]["mean_expected_log_score_delta"],
                                           -row["comparison"]["mean_expected_margin_delta"], row["operation_key"]))


def matched_random_operation(selected, model_config, checkpoint, history, seed):
    operation = selected["operation"]
    excluded = {row["operation_key"] for row in history["accepted"]} | {selected["operation_key"]}
    candidates = list(range(model_config.intermediate_size))
    random.Random(seed).shuffle(candidates)
    for column in candidates:
        proposed = {**operation, "columns": [column]}
        if operation_key(checkpoint, proposed) not in excluded:
            return proposed
    raise ValueError("No unused matched random control is available")
