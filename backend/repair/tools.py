"""The tools the fixer can call.

Every failure comes back to the fixer as an error string, so it can recover. The
loop never crashes on a bad tool call. Only successful probes use up budget.
"""
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import torch

from repair.edits import EditRefused, EditSession
from repair.gate import AttemptScores, GateThresholds, evaluate_gate

# A convention check, not a security boundary: the sandbox is the boundary.
FORBIDDEN_CODE = re.compile(r"\.backward\s*\(|torch\.optim|autograd|from_pretrained|AutoModel")
TENSOR_NAME = re.compile(r"[A-Za-z0-9_.\-]+")
MAX_ATTENTION_CALLS = 20


class Runner(Protocol):
    def run(self, code: str) -> str: ...


@dataclass
class AttemptRecord:
    number: int
    summary: str
    tensors_touched: list[str]
    weights_changed: int
    weights_changed_pct: float
    scores: dict
    outcome: str
    reasons: list[str]


@dataclass
class Conclusion:
    conclusion: str
    cause: str
    located_regions: list[dict] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)


def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS = [
    _tool(
        "run_python",
        "Run Python on CPU with the weights available through NAMES, shape(name), load(name) and save(name, array). "
        "Variables don't persist between runs; files in /work do. No network. No gradient descent or running the model.",
        {"code": {"type": "string"}},
        ["code"],
    ),
    _tool(
        "forward_pass",
        "Run the model on dev questions, with pending edits applied alongside the unedited model. Uses 1 forward pass.",
        {
            "suite": {"type": "string", "enum": ["clinical", "medmcqa"]},
            "category": {"type": "string", "description": "Optional category filter"},
        },
        ["suite"],
    ),
    _tool(
        "attention",
        "Return attention weights for up to 16 short texts. No hidden states or outputs.",
        {"texts": {"type": "array", "items": {"type": "string"}}},
        ["texts"],
    ),
    _tool(
        "scan_tensors",
        "Run a statistics script over all weight matrices and list the ones that stand out from the same matrix in "
        "other layers (few distinct values, a dominant direction, shifted mean, unusual spread). Free. A lead, not proof.",
        {},
        [],
    ),
    _tool(
        "logit_attribution",
        "On dev questions, measure how much each layer's attention and MLP output pushes each answer letter (A-D) "
        "at the answer position, with pending edits applied. Uses 1 forward pass.",
        {"suite": {"type": "string", "enum": ["clinical", "medmcqa"]}},
        ["suite"],
    ),
    _tool(
        "mean_input",
        "Save the average input vector of the layer owning a weight matrix, on dev questions, to "
        "/work/activations/<tensor_name>.npy for use in run_python, for example to design a rank-1 edit. Uses 1 forward pass.",
        {"tensor_name": {"type": "string"}, "suite": {"type": "string", "enum": ["clinical", "medmcqa"]}},
        ["tensor_name", "suite"],
    ),
    _tool(
        "save_edit",
        "Queue a candidate tensor already written with save(name, array) in run_python.",
        {"tensor_name": {"type": "string"}},
        ["tensor_name"],
    ),
    _tool(
        "submit_attempt",
        "Test all queued edits. They're kept only if they pass the gate; otherwise they're rolled back. Returns the scores.",
        {"summary": {"type": "string", "description": "What the edit does and why, in plain English"}},
        ["summary"],
    ),
    _tool(
        "finish",
        "End the repair and report.",
        {
            "conclusion": {"type": "string", "enum": ["repaired", "could_not_repair_safely", "nothing_to_fix"]},
            "cause": {"type": "string", "description": "Plain-English cause for a clinician"},
            "located_regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tensor": {"type": "string"},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    },
                    "required": ["tensor", "rationale", "confidence"],
                },
            },
            "claims": {"type": "array", "items": {"type": "string"}},
        },
        ["conclusion", "cause"],
    ),
]


class ToolBox:
    def __init__(
        self,
        session: EditSession,
        runner: Runner,
        work_dir: Path,
        forward_pass_fn: Callable[[EditSession, str, str | None], Any],
        attention_fn: Callable[[EditSession, list[str]], Any],
        validator: Callable[[EditSession], AttemptScores],
        max_forward_passes: int = 25,
        thresholds: GateThresholds = GateThresholds(),
        scan_fn: Callable[[dict], Any] | None = None,
        attribution_fn: Callable[[EditSession, str], Any] | None = None,
        mean_input_fn: Callable[[EditSession, str, str], Any] | None = None,
    ):
        self.session = session
        self.runner = runner
        self.work_dir = Path(work_dir)
        self.forward_pass_fn = forward_pass_fn
        self.attention_fn = attention_fn
        self.validator = validator
        self.max_forward_passes = max_forward_passes
        self.thresholds = thresholds
        self.scan_fn = scan_fn
        self.attribution_fn = attribution_fn
        self.mean_input_fn = mean_input_fn
        self._scan_cache = None
        self.forward_passes_used = 0
        self.attention_calls_used = 0
        self.python_runs = 0
        self.attempts: list[AttemptRecord] = []
        self.rejected_in_a_row = 0
        self.conclusion: Conclusion | None = None

    def call(self, name: str, raw_arguments: str | dict) -> str:
        """Run one tool call, returning a string for the conversation."""
        handler = {
            "run_python": self._run_python,
            "forward_pass": self._forward_pass,
            "attention": self._attention,
            "scan_tensors": self._scan_tensors,
            "logit_attribution": self._logit_attribution,
            "mean_input": self._mean_input,
            "save_edit": self._save_edit,
            "submit_attempt": self._submit_attempt,
            "finish": self._finish,
        }.get(name)
        if handler is None:
            return f"Error: unknown tool {name!r}"
        try:
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
            if not isinstance(arguments, dict):
                raise TypeError("arguments must be an object")
            return handler(**arguments)
        except (ValueError, TypeError, EditRefused) as error:
            return f"Error: {error}"
        except Exception as error:  # Probe code can fail in many ways; the fixer should see it.
            return f"Error: tool failed: {type(error).__name__}: {error}"

    def _run_python(self, code: str) -> str:
        match = FORBIDDEN_CODE.search(code)
        if match:
            return f"Error: not allowed in repairs: {match.group(0)!r}"
        self.python_runs += 1
        return self.runner.run(code)

    def _forward_pass(self, suite: str, category: str | None = None) -> str:
        if self.forward_passes_used >= self.max_forward_passes:
            return f"Error: forward-pass budget of {self.max_forward_passes} is used up"
        result = self.forward_pass_fn(self.session, suite, category)
        self.forward_passes_used += 1
        left = self.max_forward_passes - self.forward_passes_used
        return json.dumps({"result": result, "forward_passes_left": left})

    def _attention(self, texts: list[str]) -> str:
        if self.attention_calls_used >= MAX_ATTENTION_CALLS:
            return f"Error: attention limit of {MAX_ATTENTION_CALLS} is used up"
        if not 1 <= len(texts) <= 16:
            return "Error: send between 1 and 16 texts"
        result = self.attention_fn(self.session, texts)
        self.attention_calls_used += 1
        return json.dumps(result)

    def _use_pass(self) -> str | None:
        if self.forward_passes_used >= self.max_forward_passes:
            return f"Error: forward-pass budget of {self.max_forward_passes} is used up"
        return None

    def _scan_tensors(self) -> str:
        if self.scan_fn is None:
            return "Error: scan_tensors isn't available in this run"
        if self._scan_cache is None:
            self._scan_cache = self.scan_fn(self.session.base)
        return json.dumps({"suspects": self._scan_cache, "note": "Statistics only. Natural outliers also appear."})

    def _logit_attribution(self, suite: str) -> str:
        if self.attribution_fn is None:
            return "Error: logit_attribution isn't available in this run"
        refusal = self._use_pass()
        if refusal:
            return refusal
        result = self.attribution_fn(self.session, suite)
        self.forward_passes_used += 1
        return json.dumps({"result": result, "forward_passes_left": self.max_forward_passes - self.forward_passes_used})

    def _mean_input(self, tensor_name: str, suite: str) -> str:
        if self.mean_input_fn is None:
            return "Error: mean_input isn't available in this run"
        if not TENSOR_NAME.fullmatch(tensor_name) or tensor_name not in self.session.base:
            raise ValueError(f"Unknown tensor: {tensor_name!r}")
        refusal = self._use_pass()
        if refusal:
            return refusal
        vector = np.asarray(self.mean_input_fn(self.session, tensor_name, suite), dtype=np.float32)
        self.forward_passes_used += 1
        folder = self.work_dir / "activations"
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{tensor_name}.npy", vector, allow_pickle=False)
        return json.dumps({
            "saved": f"/work/activations/{tensor_name}.npy",
            "length": int(vector.size),
            "norm": round(float(np.linalg.norm(vector)), 4),
            "forward_passes_left": self.max_forward_passes - self.forward_passes_used,
        })

    def _edit_file(self, tensor_name: str) -> Path:
        if not TENSOR_NAME.fullmatch(tensor_name):
            raise ValueError(f"Invalid tensor name: {tensor_name!r}")
        path = self.work_dir / "edits" / f"{tensor_name}.npy"
        # The sandbox controls this folder, so don't follow anything it planted.
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"No saved edit for {tensor_name}. Call save(name, array) in run_python first")
        return path

    def _save_edit(self, tensor_name: str) -> str:
        if tensor_name not in self.session.base:
            raise EditRefused(f"Unknown tensor: {tensor_name}")
        path = self._edit_file(tensor_name)
        # Check the shape from the header before reading a possibly huge file.
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        expected = tuple(self.session.base[tensor_name].shape)
        if tuple(mapped.shape) != expected:
            raise EditRefused(f"Shape {tuple(mapped.shape)} doesn't match {expected}")
        self.session.save_edit(tensor_name, torch.from_numpy(np.array(mapped, dtype=np.float32)))
        return json.dumps({"queued": tensor_name, **self.session.diff()})

    def _clear_edit_files(self) -> None:
        shutil.rmtree(self.work_dir / "edits", ignore_errors=True)

    def _submit_attempt(self, summary: str) -> str:
        names = self.session.pending_names()
        if not names:
            return "Error: no edits are queued. Use save_edit first"
        diff = self.session.diff()
        scores = self.validator(self.session)
        accepted, reasons = evaluate_gate(scores, self.thresholds)
        if accepted:
            accepted_dir = self.work_dir / "accepted"
            accepted_dir.mkdir(parents=True, exist_ok=True)
            for name in names:
                self._edit_file(name).replace(accepted_dir / f"{name}.npy")
            self.session.accept()
            self.rejected_in_a_row = 0
        else:
            self.session.rollback()
            self.rejected_in_a_row += 1
        self._clear_edit_files()
        record = AttemptRecord(
            number=len(self.attempts) + 1,
            summary=summary,
            tensors_touched=diff["tensors_touched"],
            weights_changed=diff["weights_changed"],
            weights_changed_pct=diff["weights_changed_pct"],
            scores=asdict(scores),
            outcome="accepted" if accepted else "rolled_back",
            reasons=reasons,
        )
        self.attempts.append(record)
        return json.dumps(asdict(record))

    def _finish(
        self, conclusion: str, cause: str, located_regions: list[dict] | None = None, claims: list[str] | None = None
    ) -> str:
        if conclusion not in {"repaired", "could_not_repair_safely", "nothing_to_fix"}:
            raise ValueError(f"Invalid conclusion: {conclusion!r}")
        self.conclusion = Conclusion(conclusion, cause, located_regions or [], claims or [])
        return "Recorded."
