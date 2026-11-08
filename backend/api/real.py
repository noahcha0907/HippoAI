"""Real API: the same routes and shapes as the mock in api/main.py, backed by the repair pipeline.

Run in WSL from backend/:  uvicorn api.real:app --port 8000
Loading takes about a minute (model and question sets). Responses carry "mock": false.
"""
from fastapi import FastAPI, HTTPException

from api.cors import add_cors
from api.jobs import JobManager
from api.summary import build_summary
from api.schemas import (
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


ATTRIBUTION_NOTE = "Push toward each answer letter at the answer position, averaged over dev questions."


def create_app(pipeline) -> FastAPI:
    app = FastAPI(title="HippoAI repair API")
    add_cors(app)
    jobs = JobManager(pipeline.run)
    if hasattr(pipeline, "restore_runs"):
        pipeline.restore_runs(jobs)

    def get_job(repair_id: str):
        job = jobs.get(repair_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown repair ID")
        return job

    @app.get("/models", response_model=ModelList)
    def list_models():
        return {"mock": False, "models": pipeline.list_models()}

    @app.post("/models", status_code=501)
    def upload_model():
        raise HTTPException(status_code=501, detail="Uploading models isn't supported yet")

    @app.post("/repairs", response_model=RepairCreated, status_code=201)
    def create_repair(request: RepairRequest):
        if request.model_id not in pipeline.model_ids():
            raise HTTPException(status_code=404, detail="Unknown model ID")
        job = jobs.start(request)
        return RepairCreated(mock=False, id=job.id, stage=job.stage)

    @app.get("/repairs/{repair_id}", response_model=RepairStatus)
    def get_repair(repair_id: str):
        return jobs.status(get_job(repair_id))

    @app.get("/repairs/{repair_id}/report", response_model=RepairReport)
    def get_report(repair_id: str):
        job = get_job(repair_id)
        if job.error:
            raise HTTPException(status_code=500, detail=f"The repair failed: {job.error}")
        if job.report is None:
            raise HTTPException(status_code=409, detail="The report isn't ready yet")
        return job.report

    @app.post("/repairs/{repair_id}/decision", response_model=DecisionResponse)
    def decide(repair_id: str, request: DecisionRequest):
        job = get_job(repair_id)
        try:
            stage = jobs.decide(job, request.decision == "approve")
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        if stage == Stage.approved:
            pipeline.export(job)
        return DecisionResponse(mock=False, id=job.id, stage=stage)

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        job = get_job(request.repair_id) if request.repair_id else None
        try:
            text = pipeline.generate(job, request.model_version, request.prompt, request.model_id)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        return GenerateResponse(mock=False, model_version=request.model_version, text=text)

    @app.get("/repairs/{repair_id}/events", response_model=Timeline)
    def events(repair_id: str):
        job = get_job(repair_id)
        return {"mock": False, "repair_id": job.id, "is_replay": job.is_replay, "events": pipeline.timeline(job)}

    @app.get("/repairs/{repair_id}/attribution", response_model=AttributionComparison)
    def attribution(repair_id: str):
        job = get_job(repair_id)
        try:
            models = pipeline.attribution(job)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        return {"mock": False, "repair_id": job.id, "suite": "clinical", "note": ATTRIBUTION_NOTE, "models": models}

    @app.get("/repairs/{repair_id}/summary", response_model=RepairSummary)
    def summary(repair_id: str):
        job = get_job(repair_id)
        if job.report is None:
            raise HTTPException(status_code=409, detail="The report isn't ready yet")
        return build_summary(jobs.status(job), job.report)

    @app.get("/questions", response_model=QuestionList)
    def questions():
        return {"mock": False, "questions": pipeline.sealed_questions()}

    @app.post("/compare", response_model=CompareResponse)
    def compare(request: CompareRequest):
        job = get_job(request.repair_id)
        try:
            items = pipeline.compare(job, request.question_ids, request.count)
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error))
        return {"mock": False, "repair_id": job.id, "items": items}

    @app.get("/benchmark", response_model=Benchmark)
    def benchmark():
        rows = []
        for job in jobs.all():
            report = job.report
            if not report:
                continue
            recovery = report["recovery"]
            rows.append({
                "case": job.request.model_id,
                "method": "fixer_repair",
                "broken_accuracy": report["broken"]["clinical_accuracy"],
                "repaired_accuracy": report["repaired"]["clinical_accuracy"],
                "clean_accuracy": report["clean"]["clinical_accuracy"],
                "recovery": recovery,
                "collateral_loss": report["collateral_loss"],
                "weights_changed_pct": report["weights_changed_pct"],
                "minutes": report["minutes"],
                "passed": report["outcome"] == "repaired" and recovery is not None and recovery >= 0.5,
            })
        return {"mock": False, "rows": rows}

    return app


def _build_default_app() -> FastAPI:
    from api.pipeline import Pipeline  # Heavy: loads the model. Only for the real server.

    return create_app(Pipeline())


app = None
if __name__ != "__main__":
    import os

    if os.environ.get("HIPPO_SKIP_PIPELINE") != "1":
        app = _build_default_app()
