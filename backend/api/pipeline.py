"""The real repair pipeline behind the API. Needs WSL or Linux, the GPU, and the saved broken models.

Holds the dev questions (for the fixer's tools) and the sealed questions (for the evaluator)
in separate places: tools only ever receive dev suites.
"""
import json
import os
import threading
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from api.jobs import Job
from api.schemas import Stage
from breakers.make_broken import ROOT, without_shared_storage
from eval.check_breaker import load_questions
from eval.sealed_eval import evaluate_sealed, to_deployed_format
from repair.check_fixer import ENV_FILE, load_env_file
from repair.edits import EditSession
from repair.fixer_client import FixerClient, FixerConfig
from repair.loop import run_repair
from repair.qwen_backend import QwenBackend
from repair.report import build_report
from repair.sandbox import BubblewrapRunner
from repair.tools import ToolBox
from repair.validate import DevValidator

DATA = Path(__file__).resolve().parent.parent / "artifacts"
RUNS = Path(os.environ.get("HIPPO_RUNS", Path.home() / "hippo-runs")) / "api"
CLEAN_CASE = "c0"
LABELS = {
    "c1-a": ("3-bit compression, one layer", "compression_targeted"),
    "c1-b": ("4-bit compression, four layers", "compression_targeted"),
    "c1-c": ("5-bit compression, eight layers", "compression_targeted"),
    "c0": ("Undamaged control", "control"),
}


class Pipeline:
    def __init__(self):
        load_env_file(ENV_FILE)
        os.environ.setdefault("SANDBOX_VENV", str(Path.home() / "hippo-venv"))
        self.client = FixerClient(FixerConfig.from_env())
        self.backend = QwenBackend({
            "clinical": load_questions(DATA / "safety_set" / "dev.jsonl", 10_000),
            "medmcqa": load_questions(DATA / "medmcqa" / "dev.jsonl", 300),
        })
        self._sealed = {
            "safety": load_questions(DATA / "safety_set" / "sealed.jsonl", 10_000),
            "medmcqa": load_questions(DATA / "medmcqa" / "sealed.jsonl", 500),
        }
        self._states: dict[str, dict[str, torch.Tensor]] = {}
        self._states_lock = threading.Lock()

    def model_ids(self) -> list[str]:
        return sorted(p.parent.name for p in (ROOT / "models").glob("*/model.safetensors"))

    def list_models(self) -> list[dict]:
        models = []
        for case in self.model_ids():
            label, kind = LABELS.get(case, (case, "compression_targeted"))
            models.append({
                "id": case,
                "label": label,
                "damage_type": kind,
                "description": "Deliberately damaged copy for testing repairs." if kind != "control" else "Unmodified model.",
                "base_model": "Qwen/Qwen3.5-0.8B",
                "num_parameters": 873_438_784,
            })
        return models

    def state(self, case: str) -> dict[str, torch.Tensor]:
        """Cached weights for a saved model. Each is about 1.75 GB of RAM, so only a few are kept."""
        with self._states_lock:
            if case not in self._states:
                if len(self._states) >= 2:
                    self._states.pop(next(iter(self._states)))
                self._states[case] = self.backend.read_weights(ROOT / "models" / case / "model.safetensors")
            return self._states[case]

    def clean_state(self, case: str) -> dict[str, torch.Tensor]:
        """The clean model, built from the broken one plus only the tensors the break changed (evaluator only)."""
        broken = self.state(case)
        damaged = json.loads((ROOT / "evaluator" / f"{case}.json").read_text())["damaged_tensors"]
        clean = dict(broken)
        with safe_open(str(ROOT / "models" / CLEAN_CASE / "model.safetensors"), framework="pt") as f:
            for name in damaged:
                clean[name] = f.get_tensor(name)
        return clean

    def run(self, job: Job, emit) -> dict:
        case = job.request.model_id
        weights = ROOT / "models" / case / "model.safetensors"
        broken = self.state(case)
        session = EditSession(broken)
        run_dir = RUNS / job.id
        toolbox = ToolBox(
            session,
            BubblewrapRunner(run_dir / "work", weights),
            run_dir / "work",
            forward_pass_fn=self.backend.forward_pass,
            attention_fn=self.backend.attention,
            validator=DevValidator(self.backend.score),
            max_forward_passes=job.request.budget.max_forward_passes,
        )
        emit({"type": "stage", "stage": Stage.baseline.value})
        result = run_repair(
            self.client,
            toolbox,
            job.request.complaint,
            run_dir / "log.jsonl",
            max_minutes=job.request.budget.max_minutes,
            on_event=emit,
        )

        # The evaluator alone reads the manifest, the clean model and the sealed questions.
        emit({"type": "stage", "stage": Stage.sealed_test.value})
        manifest = json.loads((ROOT / "evaluator" / f"{case}.json").read_text())
        repaired, same_format = to_deployed_format(session.state(), broken, manifest)
        sealed = evaluate_sealed(
            self.backend.score_items, broken, repaired, self.clean_state(case), self._sealed["safety"], self._sealed["medmcqa"]
        )
        emit({"type": "stage", "stage": Stage.report.value})
        job.artifacts["repaired"] = repaired
        report = build_report(job.id, result, sealed, same_format)
        (run_dir / "report.json").write_text(json.dumps(report, indent=2))
        return report

    def generate(self, job: Job | None, version: str, prompt: str, model_id: str | None) -> str:
        if version == "repaired":
            if job is None or "repaired" not in job.artifacts:
                raise ValueError("No finished repair to answer with")
            return self.backend.generate(job.artifacts["repaired"], prompt)
        case = model_id or (job.request.model_id if job else None)
        if case not in self.model_ids():
            raise ValueError("Unknown or missing model_id")
        state = self.clean_state(case) if version == "clean" else self.state(case)
        return self.backend.generate(state, prompt)

    def export(self, job: Job) -> Path:
        path = RUNS / job.id / "approved_model.safetensors"
        save_file(without_shared_storage(job.artifacts["repaired"]), str(path))
        return path
