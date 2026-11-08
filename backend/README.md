# HippoAI backend

Finds and repairs damaged clinical LLMs by editing their weights, with no training. A large
open-weight model (the "fixer") diagnoses the damage with analysis tools and proposes edits.
An acceptance gate keeps an edit only if it helps on dev questions, beats a random edit of the
same size, and doesn't harm other answers. Sealed questions the fixer never sees then measure
the result.

Research prototype with deliberately damaged models. Not for clinical use, and outputs aren't
medical advice. The clinical questions haven't been reviewed by a clinician.

## Three ways to use it
| | For | Needs | Responses |
|---|---|---|---|
| **Mock API** | Front-end development | Windows or anything | `"mock": true`, placeholder data |
| **Real API** | Live demo, real repairs | WSL with the GPU and saved models | `"mock": false` |
| **Exported files** | The public website | Nothing: static JSON | Same shapes as the real API |

### Mock API
From `backend/`:

    uvicorn api.main:app --reload --port 8000

### Real API (WSL)
From `backend/` in WSL. Loading takes about a minute; saved runs reload as replays.

    uvicorn api.real:app --port 8000

Both serve `http://localhost:8000`, with interactive docs at `/docs`. Shapes are in `api/schemas.py`.

## Endpoints
| Endpoint | Use |
|---|---|
| `GET /models` | Broken models available: `c1-a`, `c1-b`, `c1-c` (compression), `t2-a` (training error), `c0` (undamaged) |
| `POST /repairs` | Start a repair: `{"model_id": "t2-a", "complaint": "..."}` |
| `GET /repairs/{id}` | Progress: stage, attempts, forward passes, `is_replay` |
| `GET /repairs/{id}/events` | Timeline: reasoning, analysis, the per-layer bar chart (`attribution` events), attempts |
| `GET /repairs/{id}/report` | Sealed results: broken vs repaired vs clean, recovery, cause |
| `GET /repairs/{id}/summary` | Scorecard: headline numbers, edit size, attempt counts, best attempt vs random, gate note |
| `GET /repairs/{id}/attribution` | Bar charts for broken, repaired and clean models: which layers push which answer |
| `POST /compare` | Broken, repaired and clean answers to sealed questions, marked correct or unsafe |
| `POST /repairs/{id}/decision` | Approve (exports the repaired weights) or reject |
| `GET /questions`, `GET /benchmark` | Sealed question list; results table |
| `POST /generate` | Free-text answers. The damage doesn't show in free text, so this isn't for the demo |

## The website
The site is static (GitHub Pages), so it can't reach the backend on a laptop. It uses exported files
for visitors, and can switch to the live API during a presentation.

**Export files** (WSL, from `backend/`; needs the GPU):

    python -m eval.export_demo --out demo_export

This writes `index.json`, `models.json`, `questions.json`, `benchmark.json`, and
`runs/<repair_id>/{status,report,events,compare,attribution,summary}.json`. Copy the folder into the site. Each file
is exactly the real API's response, so the page can use either source with the same code.

**Live mode during a presentation:** run the real API on the presenting laptop and have the
page call `http://localhost:8000`, for example when the URL has `?api=localhost`, falling back to
the exported files if the API doesn't answer. The API allows `https://noahcha0907.github.io` and
sends the header Chrome needs for a public page to reach localhost. Present in Chrome; other
browsers may block it. Add origins with `HIPPO_ALLOWED_ORIGINS` (comma-separated).

## Setup
- **Fixer key:** `.env` at the repo root (git-ignored) with `NVIDIA_API=...`. See `.env.example`.
- **WSL environment:** Python at `~/hippo-venv` with `requirements.txt` and CUDA PyTorch, plus `bubblewrap`.
- **Data** (git-ignored; rebuild on a new machine):

      python -m data.load_medmcqa --out artifacts/medmcqa
      python -m data.build_safety_set --out artifacts/safety_set
      python -m breakers.make_broken
      python -m breakers.make_training_error

## Tests

    python -m pytest

Tests marked as skipped need Linux and run in WSL, where they attack the real sandbox.

## Gate changes made after seeing results
Disclose these alongside any result.
1. **Rounding (bug fix):** comparisons now allow for floating-point rounding. A real attempt that
   gained exactly the required 3 points was rejected because 0.69 - 0.66 computed as 0.0299999.
2. **Collateral check:** originally, an edit failed if more than 1% of dev MedMCQA questions went
   from right to wrong. MedMCQA sits near chance for this model (clean 35%), so most "right"
   answers are lucky guesses that any edit reshuffles. The first rank-1 attempt on `t2-a` raised
   dev Safety Set accuracy from 66% to 77% but was rejected for 4.3% reshuffling. Now an edit fails
   if its reshuffling exceeds a random edit of the same size by more than 2 points, or if the net
   MedMCQA score falls by more than 1 point. Both thresholds were chosen after that rejection.
3. **Side effects measured on real knowledge:** the second run's best attempt (dev Safety Set 66% to
   82%, MedMCQA net up 2.3 points) was rejected for reshuffling 8% of MedMCQA answers against 1% for a
   random edit. Removing a letter bias necessarily changes lucky letter guesses. Now each MedMCQA
   question is scored in 4 answer orders, rotated so every option takes every letter once, and only
   questions right in at least 3 orders count as known. Side effects are measured on those. Checked
   on dev before the rerun: the tool-built fix lost 0.5% of known questions, a random edit 0%, and the
   fix raised known questions from 6.5% to 16%, matching the clean model. The sealed test reports
   collateral the same way, on 300 rotated sealed MedMCQA questions.

## Current status
- The pipeline works from start to finish and the safety gate rolls back harmful edits.
- On `t2-a`, the fixer found the right layer and made the right kind of edit in both runs (dev accuracy
  66% to 77%, then 66% to 82%), and every attempt was rejected for MedMCQA side effects. Both checks
  have since been changed (gate changes 2 and 3 above). No run has used the rotation-based check yet.
- Compression damage (`c1-*`) is found more reliably by statistics than by the fixer, and
  probably can't be repaired: rounded values are lost.
