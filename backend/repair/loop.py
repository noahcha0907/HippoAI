"""The autonomous repair loop: drives the fixer until it finishes or a limit is reached.

The result reflects what the harness observed, not what the fixer claims. A fixer
can't report "repaired" unless an attempt actually passed the gate.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from repair.fixer_client import FixerClient, FixerError
from repair.tools import TOOL_SCHEMAS, AttemptRecord, ToolBox

MAX_REJECTED_IN_A_ROW = 3
HANDOFF_FRACTION = 0.8
MAX_NUDGES = 2

ARCHITECTURE_NOTES = """Architecture facts about Qwen3.5-0.8B (from its published model code), so you don't mistake normal values for damage:
- RMSNorm layers (input_layernorm, post_attention_layernorm, q_norm, k_norm, the final norm) compute x_normed * (1 + weight).
  Their weights start at zero, so negative values and means near 0 are normal.
- 24 layers. Layers 3, 7, 11, 15, 19 and 23 use full self-attention; the rest use gated linear attention.
- In linear attention the decay is g = -exp(A_log) * softplus(a + dt_bias), which is negative whatever the sign of A_log.
- Embeddings and the output head are tied.
A statistic only suggests damage if it's unusual compared with similar tensors in other layers of this same model."""

SYSTEM_PROMPT = """You repair damaged language models by editing their weights directly, without training.

Rules:
- No gradient descent, optimizers, or running the model inside run_python.
- Embeddings, the output layer, the final norm, the vision tower, and scaling a whole tensor by a constant are refused.
- Edit as few weights as you can, and don't harm what already works.
- The model may have no damage. If you find none, finish with "nothing_to_fix" and make no edits.

Work like this: use forward_pass to see what fails, form a hypothesis about where, test it on the weights with
run_python, write a candidate with save(name, array), queue it with save_edit, then call submit_attempt. Attempts
that fail the gate are rolled back, and the reasons are returned to you. Call finish when you're done, and say plainly
what you did and didn't confirm. Finishing with "could_not_repair_safely" is an acceptable outcome.

""" + ARCHITECTURE_NOTES

HANDOFF_PROMPT = (
    "You're near the context limit. Write notes for a fresh copy of yourself: the hypothesis, evidence so far, "
    "what's been tried, files in /work, and the next step. Only the notes will carry over."
)


@dataclass
class RepairResult:
    outcome: str
    stop_reason: str
    cause: str
    located_regions: list[dict]
    claims: list[str]
    attempts: list[AttemptRecord]
    diff: dict
    minutes: float
    forward_passes: int
    python_runs: int
    prompt_tokens: int
    completion_tokens: int
    turns: int
    handoffs: int = 0
    fixer_said: str | None = field(default=None)


def intake_message(complaint: str, forward_passes: int, minutes: int, compression: str | None) -> str:
    lines = [
        f"User's complaint: {complaint}",
        f"Compression format: {compression or 'unknown'}",
        f"Budget: {forward_passes} forward passes, {minutes} minutes.",
        "Begin.",
    ]
    return "\n".join(lines)


def _resolve_outcome(toolbox: ToolBox, stop_reason: str) -> tuple[str, str | None]:
    """Decide the outcome from what happened, and keep what the fixer said for the record."""
    said = toolbox.conclusion.conclusion if toolbox.conclusion else None
    if any(a.outcome == "accepted" for a in toolbox.attempts):
        return "repaired", said
    if said == "nothing_to_fix" and not toolbox.attempts and stop_reason == "finished":
        return "nothing_to_fix", said
    return "could_not_repair_safely", said


def run_repair(
    client: FixerClient,
    toolbox: ToolBox,
    complaint: str,
    log_path: Path,
    max_minutes: int = 60,
    compression: str | None = None,
    clock: Callable[[], float] = time.monotonic,
    on_event: Callable[[dict], None] | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    tools: list[dict] = TOOL_SCHEMAS,
) -> RepairResult:
    started = clock()
    deadline = started + max_minutes * 60
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_tokens = completion_tokens = turns = handoffs = nudges = 0
    opening = intake_message(complaint, toolbox.max_forward_passes, max_minutes, compression)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": opening}]
    stop_reason = ""

    def record(event: dict) -> None:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        if on_event:
            on_event(event)

    record({"type": "start", "complaint": complaint, "model": client.config.model})

    while True:
        if clock() >= deadline:
            stop_reason = "time_limit"
            break
        if toolbox.rejected_in_a_row >= MAX_REJECTED_IN_A_ROW:
            stop_reason = "no_improvement"
            break
        try:
            reply = client.chat(messages, tools)
        except FixerError as error:
            record({"type": "fixer_error", "error": str(error)})
            stop_reason = "fixer_error"
            break
        turns += 1
        prompt_tokens += reply.prompt_tokens
        completion_tokens += reply.completion_tokens
        messages.append(reply.message)
        record({"type": "assistant", "message": reply.message, "prompt_tokens": reply.prompt_tokens})

        calls = reply.message.get("tool_calls") or []
        if not calls:
            nudges += 1
            if nudges > MAX_NUDGES:
                stop_reason = "fixer_stopped_without_finishing"
                break
            messages.append({"role": "user", "content": "Continue with a tool call, or call finish if you're done."})
            continue
        nudges = 0

        for call in calls:
            function = call.get("function") or {}
            name = function.get("name", "")
            output = toolbox.call(name, function.get("arguments", "{}"))
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": output})
            record({"type": "tool", "name": name, "arguments": function.get("arguments"), "output": output})
        if toolbox.conclusion:
            stop_reason = "finished"
            break

        if reply.prompt_tokens >= HANDOFF_FRACTION * client.config.context_tokens:
            try:
                notes = client.chat(messages + [{"role": "user", "content": HANDOFF_PROMPT}])
            except FixerError as error:
                record({"type": "fixer_error", "error": str(error)})
                stop_reason = "fixer_error"
                break
            prompt_tokens += notes.prompt_tokens
            completion_tokens += notes.completion_tokens
            handoffs += 1
            text = notes.message.get("content") or ""
            record({"type": "handoff", "notes": text})
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{opening}\n\nNotes from your earlier work:\n{text}"},
            ]

    # Anything queued but never submitted didn't pass the gate, so it can't ship.
    toolbox.session.rollback()
    outcome, said = _resolve_outcome(toolbox, stop_reason)
    conclusion = toolbox.conclusion
    result = RepairResult(
        outcome=outcome,
        stop_reason=stop_reason,
        cause=conclusion.cause if conclusion else "The fixer didn't report a cause before stopping.",
        located_regions=conclusion.located_regions if conclusion else [],
        claims=conclusion.claims if conclusion else [],
        attempts=toolbox.attempts,
        diff=toolbox.session.diff(),
        minutes=(clock() - started) / 60,
        forward_passes=toolbox.forward_passes_used,
        python_runs=toolbox.python_runs,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        turns=turns,
        handoffs=handoffs,
        fixer_said=said,
    )
    record({"type": "result", "result": asdict(result)})
    return result
