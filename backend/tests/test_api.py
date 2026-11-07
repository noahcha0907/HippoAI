import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import STAGE_ORDER

client = TestClient(app)
BODY = {"model_id": "c1-a", "complaint": "Wrong answers on chest-pain triage."}


def start():
    response = client.post("/repairs", json=BODY)
    assert response.status_code == 201
    return response.json()["id"]


def poll_to_end(repair_id):
    return [client.get(f"/repairs/{repair_id}").json() for _ in STAGE_ORDER]


def test_models_and_benchmark_are_mock():
    assert client.get("/models").json()["mock"] is True
    assert client.get("/benchmark").json()["mock"] is True


def test_unknown_model_and_run_are_404():
    assert client.post("/repairs", json={**BODY, "model_id": "nope"}).status_code == 404
    assert client.get("/repairs/nope").status_code == 404


def test_run_moves_through_every_stage_and_reveals_data_in_order():
    statuses = poll_to_end(start())
    assert [s["stage"] for s in statuses] == [s.value for s in STAGE_ORDER]
    assert statuses[0]["located_regions"] == [] and statuses[0]["attempts"] == []
    assert statuses[-1]["located_regions"] and statuses[-1]["attempts"]
    assert all(s["complaint"] == BODY["complaint"] for s in statuses)


def test_report_waits_until_ready():
    repair_id = start()
    assert client.get(f"/repairs/{repair_id}/report").status_code == 409
    poll_to_end(repair_id)
    report = client.get(f"/repairs/{repair_id}/report")
    assert report.status_code == 200
    assert report.json()["mock"] is True
    assert report.json()["id"] == repair_id


@pytest.mark.parametrize("decision,stage", [("approve", "approved"), ("reject", "rejected")])
def test_decision_flow(decision, stage):
    repair_id = start()
    url = f"/repairs/{repair_id}/decision"
    assert client.post(url, json={"decision": decision}).status_code == 409  # Too early.
    poll_to_end(repair_id)
    assert client.post(url, json={"decision": decision}).json()["stage"] == stage
    assert client.get(f"/repairs/{repair_id}").json()["stage"] == stage
    assert client.post(url, json={"decision": decision}).status_code == 409  # Only once.


def test_generate_validates_version():
    good = client.post("/generate", json={"model_version": "broken", "prompt": "Hello"})
    assert good.status_code == 200 and good.json()["mock"] is True
    assert client.post("/generate", json={"model_version": "other", "prompt": "Hello"}).status_code == 422
