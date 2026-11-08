# Reversible single-unit pruning experiment

This is the first actual pruning experiment on the cue-conditioning task.
It zeros **one column of a Phi MLP's fc2 output projection** at a time. For
JSL MedPhi2, that disables one intermediate unit's contribution through 2,560
weight entries. Tensor sizes and inference cost remain unchanged. Every edit
is restored; no edited checkpoint is retained or exported.

The experiment uses measured effects to select candidates. **Astra is not
called.** This establishes whether a small pruning intervention helps before
testing whether Astra improves selection.

## Run in the existing Colab GPU runtime

Paste this complete cell and upload the newly rebuilt `backend/dist/kirby_colab.zip`.
It discovers the existing workspace and worker and reuses the cached model and
dataset plan. It restores all required notebook variables. No API key is needed.
The pretrained experiment must run in Colab, not on the local machine.

```python
from google.colab import files
import io, json, zipfile

uploaded = files.upload()
assert len(uploaded) == 1, "Upload only kirby_colab.zip"
KIRBY_BUNDLE_BYTES = next(iter(uploaded.values()))
del uploaded
with zipfile.ZipFile(io.BytesIO(KIRBY_BUNDLE_BYTES)) as archive:
    notebook = json.loads(archive.read("backend/notebooks/kirby_pruning_colab.ipynb"))
steps = {cell["id"]: "".join(cell["source"])
         for cell in notebook["cells"] if cell["cell_type"] == "code"}
exec(compile(steps["resume"], "kirby-pruning-resume", "exec"))
try:
    exec(compile(steps["attempt"], "kirby-pruning-attempt", "exec"))
finally:
    exec(compile(steps["download"], "kirby-pruning-download", "exec"))
```

If there are several workspaces, open `kirby_pruning_colab.ipynb` and set
`BACKEND_PATH` in its first code cell. A deleted Colab runtime needs setup from
the main notebook first. The worker and `plan-dev-256` must exist.

The attempt ID is `cue-pruning-v1-64-attempt-id.txt`. Running the cell again
shows the same result; it does not overwrite evidence or repeat the search.
To deliberately start another attempt, set `NEW_ATTEMPT=True` in the pruning
notebook's attempt cell and reset it afterward. Existing pilot runs are preserved.

## Frozen design

- Use the same 64 frozen **training** IDs: first 8 for screening, remaining
  56 for confirmation. All variants of a question stay together. These IDs were
  used in earlier pilots, so confirmation is not a fresh held-out benchmark.
- Remeasure the original neutral answers, both wrong and correct opinion
  phrasings, nonendorsing mentions, and answer-key hints on initially wrong
  questions. Freeze original neutral-correct cohorts before editing. Both
  partitions must have at least one eligible question.
- Collect activation summaries during the original screening opinion, mention,
  and hint generations only. Select one candidate in each of four spaced
  layers (4, 12, 20, 28 on the 32-layer checkpoint), ranking the summary's top
  activation columns by mean absolute activation times column norm. This is a
  heuristic, not localization of a sycophancy mechanism.
- Freeze all four candidate units and a seeded random match for each layer
  before observing any edit outcome. Screen the four candidates independently
  from the original weights. Test the random unit matched to the top candidate's
  layer; the random result does not affect candidate selection.
- A candidate qualifies only if more cue-conditioned answers remain correct
  on the original neutral-correct cohort, with **zero regressions** on originally
  correct neutral, correct-opinion, answer-key, and lookup responses; zero cue
  regressions on the fixed cohort; and no newly invalid or truncated outputs.
  Rank qualifying candidates by retained-correct gain, then candidate ID.
- Confirm one qualifying candidate and its matched random unit on the other
  56 training questions. If none qualifies, stop after screening and report that
  no benefit was found among these four units. Do not turn a failed screen into
  a claim that all pruning is ineffective.
- Run all eight v2 lookup tasks in three conditions for the baseline and each
  edited stage. These small controls are reused across stages, not held out.
  Answer-key hints disclose labels and do not demonstrate medical reasoning.

Retained-correct gains sum three correlated conditions per eligible question:
two wrong opinions and one mention. They are a ranking objective, not a count
of independent samples. Per-condition results distinguish improvement under
opinions from improvement under mere mentions. A reduction in caving alone
does not qualify if answers become other errors or unusable output.

The random control matches layer and one-unit budget, not exact perturbation
norm or the four-candidate search budget. There is no Astra comparison, final
evaluation, or generalization claim. No thresholds are adjusted after results.

## Bounds and restoration

The new `UnitAblation` operation permits exactly one column and at most 5%
relative change to its output-projection tensor, with the existing 200,000
changed-value cap. Invalid, nonfinite, or oversized changes are rejected before
writing. The older `ColumnScale` contract remains 0.9-1.1 with a 1% norm bound.

Each edit records its tensor and slice hashes and is rolled back in a `finally`
block, including handled inference failures and interrupts. Four calibration
outputs (neutral and one wrong opinion for the first two questions) are checked
after every successful edit. Full model parameter hashes are compared before
and after the experiment. A restoration failure makes the run incomplete and
requires discarding the worker. No automatic retention or checkpoint save occurs.

The maximum is **1,736 generations** with default settings, usually fewer.
The Colab launcher allows one hour; CLI default is 30 minutes. Budget checks
occur between calls, and restoration checks may exceed the time limit. A hard
runtime kill cannot guarantee in-memory restoration; discard that worker and
reload the original checkpoint. Flushed predictions survive handled failures;
incomplete evidence must not be treated as a completed result.

## Results and maintenance

Download `kirby-pruning-evidence-*.zip` and open `START_HERE.md`. Root aliases
include the current report, frozen plan/cohorts, complete baseline predictions,
edited/restoration predictions, and pruning receipts. Full source/run artifacts
and file hashes are included; model weights, secrets, and full datasets are not.

Protocol: `single_unit_cue_pruning_v1`; report task: `cue_pruning`.
Command: `python -m kirby pruning-screen --manifest data/plan-dev-256/development.json
--revision b313a242eb211863f7b62624b345b321a3b114ea --output runs`.

After source changes, run local CPU/mocked tests, then
`tools/build_pruning_notebook.py` and `tools/package_colab.py`. Local tests are
not a pretrained model experiment.
