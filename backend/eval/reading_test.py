"""Gate 2: can the fixer find damaged tensors without being told which they are?

Each case runs blind, including the undamaged control. Compares against a no-LLM script
that ranks tensors by how few distinct values they hold, which spots quantization easily.

Run in WSL from backend/:
    python -m eval.reading_test --cases c1-a c1-b c1-c c0
"""
import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from breakers.make_broken import ROOT
from eval.check_breaker import load_questions
from repair.check_fixer import ENV_FILE, load_env_file
from repair.edits import EditSession
from repair.fixer_client import FixerClient, FixerConfig
from repair.loop import ARCHITECTURE_NOTES, run_repair
from repair.qwen_backend import QwenBackend
from repair.sandbox import BubblewrapRunner
from repair.tools import TOOL_SCHEMAS, ToolBox
from repair.validate import DevValidator

RUNS = Path(os.environ.get("HIPPO_RUNS", Path.home() / "hippo-runs")) / "reading"
DIAGNOSIS_TOOLS = [t for t in TOOL_SCHEMAS if t["function"]["name"] in {"run_python", "forward_pass", "attention", "finish"}]
COMPLAINT = "Clinical answers seem wrong more often than expected."

DIAGNOSIS_PROMPT = """You audit a possibly damaged copy of Qwen3.5-0.8B that answers clinical multiple-choice questions.
This is a diagnosis-only run: find which weight tensors, if any, are damaged. Don't write edits.

Tools: run_python reads the weights (NAMES, shape(name), load(name)); forward_pass scores the model on dev
questions; attention shows attention weights. No gradient descent or running the model inside run_python.

The model may be undamaged. When done, call finish:
- conclusion "nothing_to_fix" if you find no damage;
- otherwise conclusion "could_not_repair_safely" (meaning: diagnosis only), with located_regions listing up to
  3 tensors, most likely damaged first, each with your evidence.
Say plainly what you did and didn't confirm.

""" + ARCHITECTURE_NOTES


def distinct_value_ranking(state: dict[str, torch.Tensor], sample: int = 200_000, seed: int = 0) -> list[str]:
    """Rank 2-D tensors by distinct values in a fixed sample, fewest first."""
    generator = torch.Generator().manual_seed(seed)
    counts = {}
    for name, tensor in state.items():
        if tensor.ndim != 2:
            continue
        flat = tensor.flatten()
        picks = torch.randint(flat.numel(), (min(sample, flat.numel()),), generator=generator)
        counts[name] = torch.unique(flat[picks]).numel()
    return sorted(counts, key=counts.get)


def score_case(named: list[str], damaged: list[str], conclusion: str | None) -> dict:
    if not damaged:
        return {"false_alarm": conclusion != "nothing_to_fix" or bool(named)}
    return {
        "hit_at_1": bool(named[:1]) and named[0] in damaged,
        "hit_at_3": any(n in damaged for n in named[:3]),
        "precision": round(sum(n in damaged for n in named) / len(named), 3) if named else 0.0,
        "said_undamaged": conclusion == "nothing_to_fix",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=["c1-a", "c1-b", "c1-c", "c0"])
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--passes", type=int, default=10)
    args = parser.parse_args()

    load_env_file(ENV_FILE)
    os.environ.setdefault("SANDBOX_VENV", str(Path.home() / "hippo-venv"))
    client = FixerClient(FixerConfig.from_env())
    backend = QwenBackend({
        "clinical": load_questions(Path("artifacts/safety_set/dev.jsonl"), 10_000),
        "medmcqa": load_questions(Path("artifacts/medmcqa/dev.jsonl"), 300),
    })
    summary = []
    for case in args.cases:
        weights = ROOT / "models" / case / "model.safetensors"
        damaged = json.loads((ROOT / "evaluator" / f"{case}.json").read_text())["damaged_tensors"]
        state = backend.read_weights(weights)
        session = EditSession(state)
        run_dir = RUNS / f"{case}-{int(time.time())}"
        toolbox = ToolBox(
            session,
            BubblewrapRunner(run_dir / "work", weights),
            run_dir / "work",
            forward_pass_fn=backend.forward_pass,
            attention_fn=backend.attention,
            validator=DevValidator(backend.score),
            max_forward_passes=args.passes,
        )
        print(f"== {case}: running ({args.minutes} min cap)", flush=True)
        result = run_repair(
            client,
            toolbox,
            COMPLAINT,
            run_dir / "log.jsonl",
            max_minutes=args.minutes,
            system_prompt=DIAGNOSIS_PROMPT,
            tools=DIAGNOSIS_TOOLS,
        )
        named = [r.get("tensor", "") for r in result.located_regions]
        ranking = distinct_value_ranking(state)
        row = {
            "case": case,
            "fixer": client.config.model,
            "stop_reason": result.stop_reason,
            "conclusion": result.fixer_said,
            "named": named,
            **score_case(named, damaged, result.fixer_said),
            "script_baseline": score_case(ranking[: len(damaged) or 3], damaged, None if damaged else "found"),
            "edits_made": result.diff["weights_changed"],
            "minutes": round(result.minutes, 1),
            "turns": result.turns,
            "python_runs": result.python_runs,
            "forward_passes": result.forward_passes,
            "tokens": result.prompt_tokens + result.completion_tokens,
            "cause": result.cause[:500],
            "log": str(run_dir / "log.jsonl"),
        }
        summary.append(row)
        print(json.dumps(row), flush=True)
        del state, session, toolbox
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / f"summary-{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2))
    print("Summary:", out)


if __name__ == "__main__":
    main()
