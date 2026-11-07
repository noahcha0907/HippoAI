"""The real API and job runner, driven by a fake pipeline (no GPU or fixer needed)."""
import json
import os
import threading
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

os.environ["HIPPO_SKIP_PIPELINE"] = "1"

from api.real import create_app  # noqa: E402
from eval.scoring import Question  # noqa: E402
from eval.sealed_eval import evaluate_sealed, to_deployed_format  # noqa: E402

FINISH_ARGS = json.dumps({
    "conclusion": "repaired",
    "cause": "Layer 12 damage",
    "located_regions": [{"tensor": "layer12", "rationale": "Few distinct values", "confidence": "very"}],
})
ATTEMPT = json.dumps({
    "number": 1, "summary": "Smooth layer 12", "tensors_touched": ["layer12"], "weights_changed": 10,
    "weights_changed_pct": 0.1, "outcome": "accepted", "reasons": [],
    "scores": {"broken_clinical": 0.4, "edited_clinical": 0.6, "random_control_clinical": 0.4, "collateral_loss": 0.0},
})
METRICS = {"clinical_accuracy": 0.6, "unsafe_answer_rate": 0.1, "medmcqa_accuracy": 0.3}


class FakePipeline:
    def __init__(self, fail=False):
        self.fail = fail
        self.release = threading.Event()
        self.exported = []

    def model_ids(self):
        return ["c0", "c1-a"]

    def list_models(self):
        return [{"id": "c1-a", "label": "x", "damage_type": "compression_targeted", "description": "x",
                 "base_model": "Qwen/Qwen3.5-0.8B", "num_parameters": 1}]

    def run(self, job, emit):
        emit({"type": "stage", "stage": "baseline"})
        emit({"type": "assistant", "message": {}})
        emit({"type": "tool", "name": "forward_pass", "output": '{"forward_passes_left": 9}'})
        emit({"type": "tool", "name": "forward_pass", "output": "Error: budget used up"})
        emit({"type": "tool", "name": "save_edit", "output": '{"queued": "layer12"}'})
        emit({"type": "tool", "name": "submit_attempt", "output": ATTEMPT})
        emit({"type": "tool", "name": "finish", "arguments": FINISH_ARGS, "output": "Recorded."})
        self.release.wait(5)
        if self.fail:
            raise RuntimeError("GPU fell over")
        emit({"type": "stage", "stage": "sealed_test"})
        job.artifacts["repaired"] = {"w": 1}
        return {
            "mock": False, "id": job.id, "outcome": "repaired", "cause": "Layer 12 damage",
            "broken": METRICS, "repaired": METRICS, "clean": METRICS, "recovery": 0.7, "recovery_ci95": [0.5, 0.9],
            "collateral_loss": 0.0, "weights_changed": 10, "weights_changed_pct": 0.1, "tensors_touched": ["layer12"],
            "same_format_as_broken": True, "claims": [], "remaining_failures": [], "examples": [], "minutes": 1.0,
            "forward_passes": 1, "tokens": 100, "disclaimer": "Research prototype.",
        }

    def generate(self, job, version, prompt, model_id):
        if version == "repaired" and (job is None or "repaired" not in job.artifacts):
            raise ValueError("No finished repair to answer with")
        return f"{version} says hi"

    def export(self, job):
        self.exported.append(job.id)


def wait_for(client, repair_id, stage, timeout=5):
    deadline = time.time() + timeout
    status = {}
    while time.time() < deadline:
        status = client.get(f"/repairs/{repair_id}").json()
        if status["stage"] == stage:
            return status
        time.sleep(0.02)
    raise AssertionError(f"Never reached {stage}; last: {status.get('stage')}")


def start(client, model_id="c1-a"):
    return client.post("/repairs", json={"model_id": model_id, "complaint": "Wrong triage answers"})


@pytest.fixture
def setup():
    pipeline = FakePipeline()
    return pipeline, TestClient(create_app(pipeline))


def test_full_flow_through_real_api(setup):
    pipeline, client = setup
    response = start(client)
    assert response.status_code == 201 and response.json()["mock"] is False
    repair_id = response.json()["id"]
    status = wait_for(client, repair_id, "validate")
    assert status["forward_passes_used"] == 1  # The failed pass isn't counted.
    assert status["attempts"][0]["outcome"] == "accepted"
    assert status["attempts"][0]["dev_metrics"]["clinical_accuracy"] == 0.6
    assert status["located_regions"][0]["confidence"] == "low"  # Unknown values are coerced.
    assert client.get(f"/repairs/{repair_id}/report").status_code == 409
    early = {"model_version": "repaired", "prompt": "hi", "repair_id": repair_id}
    assert client.post("/generate", json=early).status_code == 409
    pipeline.release.set()
    wait_for(client, repair_id, "awaiting_decision")
    report = client.get(f"/repairs/{repair_id}/report").json()
    assert report["recovery"] == 0.7 and report["mock"] is False
    assert client.post("/generate", json=early).json()["text"] == "repaired says hi"
    assert client.get("/benchmark").json()["rows"][0]["passed"] is True
    decision = client.post(f"/repairs/{repair_id}/decision", json={"decision": "approve"})
    assert decision.json()["stage"] == "approved" and pipeline.exported == [repair_id]
    assert client.post(f"/repairs/{repair_id}/decision", json={"decision": "approve"}).status_code == 409


def test_failure_is_reported_not_hidden():
    pipeline = FakePipeline(fail=True)
    pipeline.release.set()
    client = TestClient(create_app(pipeline))
    repair_id = start(client).json()["id"]
    status = wait_for(client, repair_id, "failed")
    assert "GPU fell over" in status["error"]
    assert client.get(f"/repairs/{repair_id}/report").status_code == 500
    assert client.post(f"/repairs/{repair_id}/decision", json={"decision": "approve"}).status_code == 409


def test_unknown_ids(setup):
    _, client = setup
    assert start(client, "nope").status_code == 404
    assert client.get("/repairs/nope").status_code == 404
    assert client.post("/models").status_code == 501


# Sealed evaluation
def make_question(i, unsafe=()):
    return Question(id=str(i), question=f"Q{i}", options=dict(A="a", B="b", C="c", D="d"), answer="A",
                    category="triage", unsafe_options=list(unsafe))


def test_sealed_evaluation_computes_recovery_and_examples():
    safety = [make_question(i, unsafe=["B"]) for i in range(200)]
    medmcqa = [make_question(i) for i in range(50)]
    broken, repaired, clean = {"w": torch.zeros(1)}, {"w": torch.ones(1)}, {"w": torch.full((1,), 2.0)}
    rates = {id(broken): 0.3, id(repaired): 0.6, id(clean): 0.9}
    draws = np.random.default_rng(0).random(200)

    def score_items(state, items):
        if items is medmcqa:
            return [{"prediction": "A", "correct": True} for _ in items]
        rate = rates[id(state)]
        return [{"prediction": "A" if d < rate else "B", "correct": bool(d < rate)} for d in draws]

    result = evaluate_sealed(score_items, broken, repaired, clean, safety, medmcqa)
    accuracy = {k: result[k]["clinical_accuracy"] for k in ("broken", "repaired", "clean")}
    assert accuracy["broken"] < accuracy["repaired"] < accuracy["clean"]
    assert 0.3 < result["recovery"] < 0.7
    low, high = result["recovery_ci95"]
    assert low < result["recovery"] < high
    assert result["broken"]["unsafe_answer_rate"] > result["repaired"]["unsafe_answer_rate"]
    assert result["collateral_loss"] == 0.0
    assert result["examples"] and result["examples"][0]["broken_answer"].startswith("B.")
    assert result["remaining_failures"]  # 0.6 against 0.9 is a gap worth reporting.


def test_deployed_format_requantizes_only_edited_compressed_tensors():
    broken = {"a": torch.linspace(-1, 1, 100), "b": torch.linspace(-1, 1, 100)}
    repaired = {"a": broken["a"] + 0.01, "b": broken["b"]}
    deployed, same = to_deployed_format(repaired, broken, {"bits": 3, "damaged_tensors": ["a", "b"]})
    assert same and torch.unique(deployed["a"]).numel() <= 8
    assert deployed["b"] is broken["b"]
