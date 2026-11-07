"""Runs repairs one at a time in the background and tracks their progress for polling.

The pipeline function does the work and reports progress through `emit(event)`. Events
are the repair loop's own log events, plus {"type": "stage", "stage": ...} from the pipeline.
"""
import json
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from api.schemas import RepairRequest, Stage

CONFIDENCE = {"low", "medium", "high"}


@dataclass
class Job:
    id: str
    request: RepairRequest
    created: float = field(default_factory=time.time)
    stage: Stage = Stage.intake
    forward_passes: int = 0
    attempts: list[dict] = field(default_factory=list)
    located_regions: list[dict] = field(default_factory=list)
    report: dict | None = None
    error: str | None = None
    decision: Stage | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)  # Pipeline-owned, such as the repaired weights.


def _attempt(record: dict) -> dict:
    scores = record.get("scores", {})
    return {
        "number": record["number"],
        "summary": record.get("summary", ""),
        "tensors_touched": record.get("tensors_touched", []),
        "weights_changed": record.get("weights_changed", 0),
        "weights_changed_pct": record.get("weights_changed_pct", 0.0),
        "dev_metrics": {
            "clinical_accuracy": scores.get("edited_clinical", 0.0),
            "broken_clinical_accuracy": scores.get("broken_clinical"),
            "random_control_accuracy": scores.get("random_control_clinical"),
            "collateral_loss": scores.get("collateral_loss"),
        },
        "outcome": record.get("outcome", "rolled_back"),
        "reason": "; ".join(record.get("reasons", [])) or "Passed the dev gate",
    }


def _regions(arguments: dict) -> list[dict]:
    regions = []
    for region in arguments.get("located_regions") or []:
        if isinstance(region, dict) and region.get("tensor"):
            confidence = region.get("confidence")
            regions.append({
                "tensor": str(region["tensor"]),
                "rationale": str(region.get("rationale", "")),
                "confidence": confidence if confidence in CONFIDENCE else "low",
            })
    return regions


class JobManager:
    def __init__(self, pipeline: Callable[[Job, Callable[[dict], None]], dict]):
        self._pipeline = pipeline
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)  # One GPU: one repair at a time.

    def start(self, request: RepairRequest) -> Job:
        job = Job(id=f"repair-{uuid.uuid4().hex[:8]}", request=request)
        with self._lock:
            self._jobs[job.id] = job
        self._executor.submit(self._run, job)
        return job

    def get(self, repair_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(repair_id)

    def all(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    def _run(self, job: Job) -> None:
        try:
            report = self._pipeline(job, lambda event: self._on_event(job, event))
            with self._lock:
                job.report = report
                job.stage = Stage.awaiting_decision
        except Exception as error:  # Shown to the user; the full trace goes to the server log.
            traceback.print_exc()
            with self._lock:
                job.error = f"{type(error).__name__}: {error}"
                job.stage = Stage.failed

    def _on_event(self, job: Job, event: dict) -> None:
        with self._lock:
            kind = event.get("type")
            if kind == "stage":
                job.stage = Stage(event["stage"])
            elif kind == "assistant" and job.stage in (Stage.intake, Stage.baseline):
                job.stage = Stage.locate
            elif kind == "tool":
                name = event.get("name")
                output = event.get("output") or ""
                if name == "forward_pass" and '"forward_passes_left"' in output:
                    job.forward_passes += 1
                elif name == "save_edit" and not output.startswith("Error"):
                    job.stage = Stage.patch
                elif name == "submit_attempt" and not output.startswith("Error"):
                    job.stage = Stage.validate
                    try:
                        job.attempts.append(_attempt(json.loads(output)))
                    except (ValueError, KeyError):
                        pass
                elif name == "finish":
                    try:
                        arguments = event.get("arguments") or "{}"
                        job.located_regions = _regions(json.loads(arguments) if isinstance(arguments, str) else arguments)
                    except ValueError:
                        pass

    def status(self, job: Job) -> dict:
        with self._lock:
            return {
                "mock": False,
                "id": job.id,
                "model_id": job.request.model_id,
                "complaint": job.request.complaint,
                "stage": job.decision or job.stage,
                "elapsed_minutes": round((time.time() - job.created) / 60, 2),
                "forward_passes_used": job.forward_passes,
                "budget": job.request.budget.model_dump(),
                "located_regions": list(job.located_regions),
                "attempts": list(job.attempts),
                "is_replay": False,
                "error": job.error,
            }

    def decide(self, job: Job, approve: bool) -> Stage:
        with self._lock:
            if job.stage != Stage.awaiting_decision or job.decision is not None:
                raise ValueError("The repair isn't awaiting a decision")
            job.decision = Stage.approved if approve else Stage.rejected
            return job.decision
