"""Mock API so the front end can be built before the repair loop exists.

Run from backend/:  uvicorn api.main:app --reload --port 8000

Each GET /repairs/{id} moves the fake run forward one stage, so polling UIs can
be exercised. Real endpoints replace these handlers behind the same schemas.
"""
import copy
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.cors import add_cors
from api.schemas import (
    STAGE_ORDER,
    AttributionComparison,
    Benchmark,
    RepairSummary,
    CompareRequest,
    CompareResponse,
    QuestionList,
    Timeline,
    DecisionRequest,
    DecisionResponse,
    GenerateRequest,
    GenerateResponse,
    ModelList,
    RepairCreated,
    RepairReport,
    RepairRequest,
    RepairStatus,
    Stage,
)

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
PATCH_INDEX = STAGE_ORDER.index(Stage.patch)
VALIDATE_INDEX = STAGE_ORDER.index(Stage.validate)
REPORT_INDEX = STAGE_ORDER.index(Stage.awaiting_decision)


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


app = FastAPI(title="HippoAI repair API (mock)")
add_cors(app)

# In-memory only: a restart clears runs.
RUNS: dict[str, dict] = {}


def get_run(repair_id: str) -> dict:
    run = RUNS.get(repair_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Unknown repair ID")
    return run


@app.get("/models", response_model=ModelList)
def list_models():
    return load_fixture("models")


@app.post("/models", status_code=501)
def upload_model():
    raise HTTPException(status_code=501, detail="Uploads aren't available in the mock API")


@app.post("/repairs", response_model=RepairCreated, status_code=201)
def create_repair(request: RepairRequest):
    known = {m["id"] for m in load_fixture("models")["models"]}
    if request.model_id not in known:
        raise HTTPException(status_code=404, detail="Unknown model ID")
    repair_id = f"mock-{uuid.uuid4().hex[:8]}"
    RUNS[repair_id] = {"request": request, "polls": 0, "decision": None}
    return RepairCreated(mock=True, id=repair_id, stage=Stage.intake)


@app.get("/repairs/{repair_id}", response_model=RepairStatus)
def get_repair(repair_id: str):
    run = get_run(repair_id)
    index = min(run["polls"], REPORT_INDEX)
    run["polls"] += 1

    status = copy.deepcopy(load_fixture("repair_run"))
    status.update(
        id=repair_id,
        model_id=run["request"].model_id,
        complaint=run["request"].complaint,
        budget=run["request"].budget.model_dump(),
        stage=run["decision"] or STAGE_ORDER[index],
    )
    if index < PATCH_INDEX:
        status["located_regions"] = []
    if index < VALIDATE_INDEX:
        status["attempts"] = []
    return status


@app.get("/repairs/{repair_id}/report", response_model=RepairReport)
def get_report(repair_id: str):
    run = get_run(repair_id)
    if run["polls"] <= REPORT_INDEX and run["decision"] is None:
        raise HTTPException(status_code=409, detail="The report isn't ready yet")
    return {**load_fixture("report"), "id": repair_id}


@app.post("/repairs/{repair_id}/decision", response_model=DecisionResponse)
def decide(repair_id: str, request: DecisionRequest):
    run = get_run(repair_id)
    if run["polls"] <= REPORT_INDEX:
        raise HTTPException(status_code=409, detail="The repair isn't awaiting a decision")
    if run["decision"] is not None:
        raise HTTPException(status_code=409, detail="A decision was already recorded")
    run["decision"] = Stage.approved if request.decision == "approve" else Stage.rejected
    return DecisionResponse(mock=True, id=repair_id, stage=run["decision"])


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    answers = load_fixture("generate")["answers"]
    return GenerateResponse(mock=True, model_version=request.model_version, text=answers[request.model_version])


@app.get("/benchmark", response_model=Benchmark)
def benchmark():
    return load_fixture("benchmark")


@app.get("/repairs/{repair_id}/events", response_model=Timeline)
def events(repair_id: str):
    get_run(repair_id)
    return {**load_fixture("events"), "repair_id": repair_id}


@app.get("/questions", response_model=QuestionList)
def questions():
    items = load_fixture("compare")["items"]
    return {"mock": True, "questions": [{k: i[k] for k in ("id", "category", "question")} for i in items]}


@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    run = get_run(request.repair_id)
    if run["polls"] <= REPORT_INDEX and run["decision"] is None:
        raise HTTPException(status_code=409, detail="The repair hasn't finished")
    items = load_fixture("compare")["items"]
    if request.question_ids:
        items = [i for i in items if i["id"] in set(request.question_ids)]
    return {"mock": True, "repair_id": request.repair_id, "items": items[: request.count]}


def _finished(repair_id: str) -> None:
    run = get_run(repair_id)
    if run["polls"] <= REPORT_INDEX and run["decision"] is None:
        raise HTTPException(status_code=409, detail="The repair hasn't finished")


@app.get("/repairs/{repair_id}/attribution", response_model=AttributionComparison)
def attribution(repair_id: str):
    _finished(repair_id)
    return {**load_fixture("attribution"), "repair_id": repair_id}


@app.get("/repairs/{repair_id}/summary", response_model=RepairSummary)
def summary(repair_id: str):
    _finished(repair_id)
    return {**load_fixture("summary"), "repair_id": repair_id}
