# HippoAI backend

Repairs damaged clinical LLMs by editing their weights, with no training.

**Status:** the API serves mock data, and every mock response has `"mock": true`.
Don't present those numbers as results. The repair loop isn't connected yet.

## Run the mock API
From `backend/`, using the virtual environment at the repo root:

    uvicorn api.main:app --reload --port 8000

Interactive docs are at http://localhost:8000/docs. Response shapes are in `api/schemas.py`.

To simulate a repair:
1. `POST /repairs` with `{"model_id": "c1-quant-a", "complaint": "..."}`.
2. Poll `GET /repairs/{id}`. Each call moves one stage forward, ending at `awaiting_decision`.
3. `GET /repairs/{id}/report`, available once the run reaches that stage.
4. `POST /repairs/{id}/decision` with `{"decision": "approve"}` or `"reject"`.

## Tests

    python -m pytest

## Make the MedMCQA samples

    python -m data.load_medmcqa --out artifacts/medmcqa

Research prototype. Not for clinical use, and outputs aren't medical advice.
