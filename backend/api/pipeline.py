"""The real repair pipeline behind the API. Needs WSL or Linux, the GPU, and the saved broken models.

Holds the dev questions (for the fixer's tools) and the sealed questions (for the evaluator)
in separate places: tools only ever receive dev suites.
"""
import json
import os
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from api.jobs import Job, JobManager
from api.schemas import RepairRequest
from api.timeline import build_timeline
from api.schemas import Stage
from breakers.make_broken import ROOT, without_shared_storage
from eval.check_breaker import load_questions
from eval.scoring import ROTATIONS, rotated_set
from eval.sealed_eval import evaluate_sealed, to_deployed_format
from repair.check_fixer import ENV_FILE, load_env_file
from repair.edits import EditSession
from repair.fixer_client import FixerClient, FixerConfig
from repair.loop import run_repair
from repair.qwen_backend import QwenBackend
from repair.report import build_report
from repair.sandbox import BubblewrapRunner
from repair.scan import scan
from repair.tools import ToolBox
from repair.validate import DevValidator

DATA = Path(__file__).resolve().parent.parent / "artifacts"
RUNS = Path(os.environ.get("HIPPO_RUNS", Path.home() / "hippo-runs")) / "api"
CLEAN_CASE = "c0"
GENERAL_QUESTIONS = 200  # Per rotation, for the side-effect check.
SEALED_GENERAL_QUESTIONS = 300
LABELS = {
    "c1-a": ("3-bit compression, one layer", "compression_targeted"),
    "c1-b": ("4-bit compression, four layers", "compression_targeted"),
    "c1-c": ("5-bit compression, eight layers", "compression_targeted"),
    "c0": ("Undamaged control", "control"),
    "t2-a": ("Training error: learned answer shortcut", "training_error"),
}


class Pipeline:
    def __init__(self):
        load_env_file(ENV_FILE)
        os.environ.setdefault("SANDBOX_VENV", str(Path.home() / "hippo-venv"))
        self.client = FixerClient(FixerConfig.from_env())
        self.backend = QwenBackend({
            "clinical": load_questions(DATA / "safety_set" / "dev.jsonl", 10_000),
            "medmcqa": load_questions(DATA / "medmcqa" / "dev.jsonl", 300),
            "medmcqa_rotated": rotated_set(load_questions(DATA / "medmcqa" / "dev.jsonl", GENERAL_QUESTIONS)),
        })
        self._sealed = {
            "safety": load_questions(DATA / "safety_set" / "sealed.jsonl", 10_000),
            "medmcqa": load_questions(DATA / "medmcqa" / "sealed.jsonl", 500),
        }
        self._sealed["medmcqa_rotated"] = rotated_set(self._sealed["medmcqa"][:SEALED_GENERAL_QUESTIONS])
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
            validator=DevValidator(self.backend.score, general_suite="medmcqa_rotated", rotations=ROTATIONS),
            max_forward_passes=job.request.budget.max_forward_passes,
            scan_fn=scan,
            attribution_fn=self.backend.logit_attribution,
            mean_input_fn=self.backend.mean_input,
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        job.artifacts["run_dir"] = run_dir
        (run_dir / "request.json").write_text(
            json.dumps({"id": job.id, "created": job.created, "request": job.request.model_dump()}, indent=2)
        )

        def stage(name: str) -> None:
            event = {"t": time.time(), "type": "stage", "stage": name}
            with (run_dir / "log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps(event) + "\n")
            emit(event)

        stage(Stage.baseline.value)
        result = run_repair(
            self.client,
            toolbox,
            job.request.complaint,
            run_dir / "log.jsonl",
            max_minutes=job.request.budget.max_minutes,
            on_event=emit,
        )

        # The evaluator alone reads the manifest, the clean model and the sealed questions.
        stage(Stage.sealed_test.value)
        manifest = json.loads((ROOT / "evaluator" / f"{case}.json").read_text())
        repaired, same_format = to_deployed_format(session.state(), broken, manifest)
        sealed = evaluate_sealed(
            self.backend.score_items, broken, repaired, self.clean_state(case), self._sealed["safety"], self._sealed["medmcqa"],
            self._sealed["medmcqa_rotated"],
        )
        stage(Stage.report.value)
        job.artifacts["repaired"] = repaired
        report = build_report(job.id, result, sealed, same_format)
        changed = {name: t.contiguous() for name, t in repaired.items() if t is not broken[name]}
        if changed:
            save_file(changed, str(run_dir / "repaired_delta.safetensors"))
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

    def sealed_questions(self) -> list[dict]:
        return [{"id": q.id, "category": q.category, "question": q.question} for q in self._sealed["safety"]]

    def compare(self, job: Job, question_ids: list[str] | None, count: int) -> list[dict]:
        """Broken, repaired and clean answers to sealed questions. Only after the repair has finished."""
        if "repaired" not in job.artifacts:
            raise ValueError("The repair hasn't finished")
        by_id = {q.id: q for q in self._sealed["safety"]}
        case = job.request.model_id
        states = {"broken": self.state(case), "repaired": job.artifacts["repaired"], "clean": self.clean_state(case)}
        pool = [by_id[i] for i in question_ids if i in by_id] if question_ids else list(self._sealed["safety"])
        if question_ids and len(pool) != len(question_ids):
            raise ValueError("Unknown question ID")
        rows = {label: self.backend.score_items(state, pool) for label, state in states.items()}
        indices = range(len(pool))
        if not question_ids:
            # Showcase order: questions the repair fixed, then ones showing the damage (broken wrong,
            # clean right; unsafe answers first), then the rest.
            def rank(i):
                broken, repaired, clean = (rows[k][i] for k in ("broken", "repaired", "clean"))
                fixed = not broken["correct"] and repaired["correct"]
                damaged = not broken["correct"] and clean["correct"]
                unsafe = broken["prediction"] in pool[i].unsafe_options
                return (not fixed, not damaged, not unsafe, i)

            indices = sorted(indices, key=rank)[:count]
        items = []
        for i in indices:
            q = pool[i]
            answers = {}
            for label in states:
                letter = rows[label][i]["prediction"]
                answers[label] = {
                    "letter": letter,
                    "text": q.options[letter],
                    "correct": letter == q.answer,
                    "unsafe": letter in q.unsafe_options,
                }
            items.append({
                "id": q.id, "category": q.category, "question": q.question, "options": q.options,
                "expected": q.answer, "unsafe_options": q.unsafe_options, "answers": answers,
            })
        return items

    def attribution(self, job: Job) -> dict:
        """Which layers push which answer, for broken, repaired and clean models (dev questions)."""
        if "attribution" in job.artifacts:
            return job.artifacts["attribution"]
        if "repaired" not in job.artifacts:
            raise ValueError("The repair hasn't finished")
        case = job.request.model_id
        states = {"broken": self.state(case), "repaired": job.artifacts["repaired"], "clean": self.clean_state(case)}
        result = {
            label: self.backend.logit_attribution(EditSession(state), "clinical", top=8) for label, state in states.items()
        }
        job.artifacts["attribution"] = result
        run_dir = job.artifacts.get("run_dir")
        if run_dir:
            (Path(run_dir) / "attribution.json").write_text(json.dumps(result, indent=2))
        return result

    def timeline(self, job: Job) -> list[dict]:
        run_dir = job.artifacts.get("run_dir")
        return build_timeline(Path(run_dir) / "log.jsonl") if run_dir else []

    def restore_runs(self, jobs: JobManager) -> int:
        """Reload finished runs from disk, so a restart keeps the demo runs."""
        restored = 0
        for run_dir in sorted(RUNS.glob("repair-*")):
            files = [run_dir / n for n in ("request.json", "report.json", "log.jsonl")]
            if not all(f.exists() for f in files):
                continue
            saved = json.loads(files[0].read_text())
            request = RepairRequest(**saved["request"])
            if request.model_id not in self.model_ids():
                continue
            job = Job(id=saved["id"], request=request, created=saved["created"])
            job.report = json.loads(files[1].read_text())
            job.artifacts["run_dir"] = run_dir
            repaired = dict(self.state(request.model_id))
            delta = run_dir / "repaired_delta.safetensors"
            if delta.exists():
                repaired.update(load_file(str(delta)))
            job.artifacts["repaired"] = repaired
            saved_attribution = run_dir / "attribution.json"
            if saved_attribution.exists():
                job.artifacts["attribution"] = json.loads(saved_attribution.read_text())
            events = [json.loads(line) for line in files[2].read_text().splitlines() if line.strip()]
            jobs.restore(job, events)
            restored += 1
        return restored

    def export(self, job: Job) -> Path:
        path = RUNS / job.id / "approved_model.safetensors"
        save_file(without_shared_storage(job.artifacts["repaired"]), str(path))
        return path
