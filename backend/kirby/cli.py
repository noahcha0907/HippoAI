"""Trusted local operator commands. Not an upload or browser API."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .datasets import load_development, prepare
from .models import DEFAULT_MODEL, SUPPORTED_MODELS, create_predictor
from .runner import RunConfig, run_baseline
from .records import write_json
from .reproducibility import compare_baselines
from .rollback import run_rollback_check
from .astra import ASTRA_MODEL, AstraClient, AstraConfig
from .astra_runner import AstraRunConfig, run_astra_attempt
from .search import SearchConfig
from .search_runner import run_astra_search
from .sycophancy import PilotConfig, run_pilot
from .pruning import PruningConfig, run_pruning


def main(argv=None) -> int:
    import sys
    arguments_list = list(sys.argv[1:] if argv is None else argv)
    if arguments_list and arguments_list[0] in ('eeg-run', 'eeg-export'):
        from .eeg.__main__ import main as eeg_main
        return eeg_main([arguments_list[0].removeprefix('eeg-'), *arguments_list[1:]])
    parser = argparse.ArgumentParser(description="Kirby: prepare and measure a MedQA development baseline")
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("prepare", help="Validate official splits, deduplicate, and freeze selected IDs")
    for split in ("train", "dev", "test"):
        plan.add_argument(f"--{split}", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seed", type=int, default=17)
    plan.add_argument("--diagnostic-count", type=int, default=64)
    plan.add_argument("--development-count", type=int, default=256)
    validate = commands.add_parser("validate", help="Check a frozen development manifest without loading a model")
    validate.add_argument("--manifest", type=Path, required=True)
    baseline = commands.add_parser("baseline", help="Calibrate and evaluate a supported checkpoint (default: JSL MedPhi2)")
    rollback = commands.add_parser("rollback-check", help="Apply one controlled Phi edit, restore weights, and verify outputs")
    astra = commands.add_parser("astra-attempt", help="Request and evaluate one bounded Astra patch, then restore the original")
    search = commands.add_parser("astra-search", help="Rank training edits, review one with Astra, and evaluate 172 development questions")
    pilot = commands.add_parser("sycophancy-pilot", help="Measure training-only opinion conditioning without Astra or weight edits")
    pilot.add_argument("--questions", type=int, default=64)
    pilot.add_argument("--attempt-id")
    pruning = commands.add_parser("pruning-screen", help="Screen reversible single-unit ablations on training cue prompts")
    pruning.add_argument("--questions", type=int, default=64)
    pruning.add_argument("--screening-count", type=int, default=8)
    pruning.add_argument("--candidates", type=int, default=4)
    pruning.add_argument("--attempt-id")
    for worker in (baseline, rollback, astra, search, pilot, pruning):
        worker.add_argument("--model", choices=SUPPORTED_MODELS if worker is baseline else (DEFAULT_MODEL,), default=DEFAULT_MODEL)
        worker.add_argument("--precision", choices=("float32", "bfloat16"),
                          help="Default: float32 for JSL MedPhi2, bfloat16 for MedGemma")
        worker.add_argument("--manifest", type=Path, required=True)
        worker.add_argument("--revision", required=True, help="Full immutable Hugging Face commit SHA")
        worker.add_argument("--output", type=Path, default=Path("runs"))
        worker.add_argument("--allow-download", action="store_true", help="Permit downloading the selected checkpoint; default is cached-only")
        worker.add_argument("--max-new-tokens", type=int, default=8)
        worker.add_argument("--max-seconds", type=float, default=1800)
        worker.add_argument("--seed", type=int, default=17)
    baseline.add_argument("--calibration-questions", type=int, default=4)
    rollback.add_argument("--questions", type=int, default=4)
    rollback.add_argument("--layer", type=int, default=0)
    rollback.add_argument("--column", type=int, default=0)
    rollback.add_argument("--scale", type=float, default=0.97)
    astra.add_argument("--rollback-report", type=Path, required=True)
    astra.add_argument("--attempt-id", help="UUID; reusing an existing attempt is refused before API access")
    astra.add_argument("--diagnostic-count", type=int, default=8)
    astra.add_argument("--calibration-questions", type=int, default=4)
    astra.add_argument("--editor-model", choices=(ASTRA_MODEL,), default=ASTRA_MODEL)
    astra.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    astra.add_argument("--editor-max-output-tokens", type=int, default=4096)
    astra.add_argument("--api-timeout", type=float, default=180)
    search.add_argument("--rollback-report", type=Path, required=True)
    search.add_argument("--attempt-id")
    search.add_argument("--development-count", type=int, default=172)
    search.add_argument("--diagnostic-count", type=int, default=64)
    search.add_argument("--screening-count", type=int, default=8)
    search.add_argument("--layer-count", type=int, default=8)
    search.add_argument("--confirm-count", type=int, default=2)
    search.add_argument("--calibration-questions", type=int, default=4)
    search.add_argument("--editor-model", choices=(ASTRA_MODEL,), default=ASTRA_MODEL)
    search.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    search.add_argument("--editor-max-output-tokens", type=int, default=4096)
    search.add_argument("--api-timeout", type=float, default=180)
    compare = commands.add_parser("compare-baselines", help="Compare two complete unchanged development runs")
    compare.add_argument("--first", type=Path, required=True)
    compare.add_argument("--second", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True, help="New comparison JSON file")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "prepare":
            result = prepare(arguments.train, arguments.dev, arguments.test, arguments.output,
                             seed=arguments.seed, diagnostic_count=arguments.diagnostic_count,
                             development_count=arguments.development_count)
        elif arguments.command == "validate":
            questions, _ = load_development(arguments.manifest)
            result = {"valid": True, "phase": "development", "questions": len(questions)}
        elif arguments.command == "compare-baselines":
            if arguments.output.exists():
                raise ValueError("Comparison output already exists; choose a new filename")
            result = compare_baselines(arguments.first, arguments.second)
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(arguments.output, result)
            print(json.dumps(result, indent=2))
            return 0 if result["reproduced"] else 1
        else:
            predictor = create_predictor(arguments.model, arguments.revision, precision=arguments.precision,
                                         max_new_tokens=arguments.max_new_tokens, seed=arguments.seed,
                                         local_files_only=not arguments.allow_download)
            if arguments.command in {"astra-attempt", "astra-search"}:
                editor = AstraClient(AstraConfig(model=arguments.editor_model, reasoning_effort=arguments.reasoning_effort,
                                                 max_output_tokens=arguments.editor_max_output_tokens,
                                                 timeout_seconds=arguments.api_timeout))
                if arguments.command == "astra-search":
                    config = SearchConfig(development_count=arguments.development_count, diagnostic_count=arguments.diagnostic_count,
                                          screening_count=arguments.screening_count, layer_count=arguments.layer_count,
                                          confirm_count=arguments.confirm_count, calibration_questions=arguments.calibration_questions,
                                          max_seconds=arguments.max_seconds, seed=arguments.seed)
                    result = run_astra_search(arguments.manifest, arguments.rollback_report, predictor,
                                             asdict(predictor.config), editor, arguments.output, config, attempt_id=arguments.attempt_id)
                else:
                    result = run_astra_attempt(arguments.manifest, arguments.rollback_report, predictor,
                                               asdict(predictor.config), editor, arguments.output,
                                               AstraRunConfig(diagnostic_count=arguments.diagnostic_count,
                                                              calibration_questions=arguments.calibration_questions,
                                                              max_seconds=arguments.max_seconds),
                                               attempt_id=arguments.attempt_id)
            elif arguments.command == "pruning-screen":
                result = run_pruning(arguments.manifest, predictor, asdict(predictor.config), arguments.output,
                                     PruningConfig(questions=arguments.questions, screening_count=arguments.screening_count,
                                                   candidates=arguments.candidates, max_seconds=arguments.max_seconds, seed=arguments.seed),
                                     attempt_id=arguments.attempt_id)
            elif arguments.command == "sycophancy-pilot":
                result = run_pilot(arguments.manifest, predictor, asdict(predictor.config), arguments.output,
                                   PilotConfig(questions=arguments.questions, max_seconds=arguments.max_seconds),
                                   attempt_id=arguments.attempt_id)
            elif arguments.command == "rollback-check":
                result = run_rollback_check(arguments.manifest, predictor, asdict(predictor.config), arguments.output,
                                            questions=arguments.questions, layer=arguments.layer, column=arguments.column,
                                            scale=arguments.scale, max_seconds=arguments.max_seconds)
            else:
                result = run_baseline(arguments.manifest, predictor, asdict(predictor.config), arguments.output,
                                  RunConfig(max_seconds=arguments.max_seconds,
                                            calibration_questions=arguments.calibration_questions))
            print(json.dumps({"directory": result["directory"], "state": result["report"]["state"],
                              "outcome": result["report"]["outcome"]}, indent=2))
            return 0 if result["report"]["state"] == "completed" else 1
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0
