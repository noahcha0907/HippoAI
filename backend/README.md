# Kirby backend: EEG shortcut diagnosis and repair

The current experiment trains small EEGNet models with controlled shortcuts,
asks Astra for blinded weight-based diagnoses, and tests bounded repairs against
random, signal-rule and input-filter controls. All methods are frozen before
the final EEG session is evaluated.

Start with the [EEG experiment guide](docs/EEG_EXPERIMENT.md) and
[roadmap](docs/EEG_ROADMAP.md). Upload `notebooks/kirby_eeg_colab.ipynb` to Colab
and give it **`dist/kirby_eeg_colab.zip`**. Local synthetic checks do not replace
the first real-data/GPU/API validation run. The frontend is unchanged.

The remaining sections preserve the earlier MedPhi workflow and results.

## Archived MedPhi baseline

This is the first **development baseline evaluator**, following the Kirby
implementation guide with the user's switch to **John Snow Labs' JSL-MedPhi2-2.7B**.
It prepares a fixed MedQA
sample, calibrates the model, evaluates answers, and saves evidence. It also
compares repeated baselines, performs a controlled Phi column edit with exact
rollback, and can request and evaluate one bounded Astra proposal. It does not
retain an edited candidate, serve an API, or evaluate a final candidate.

For the earlier pruning experiment, use the [reversible pruning guide](docs/PRUNING_EXPERIMENT.md)
and its self-contained Colab launcher. It tests actual single-unit zeroing with
exact rollback. The [sycophancy pilot guide](docs/SYCOPHANCY_PILOT.md) and workflows
below document the preceding unchanged-model measurements and earlier experiments.

The user reported a completed T4 development run: **97/256 correct (37.89%)**,
zero invalid/truncated answers, and about 0.47 seconds per scored question.
Those logs establish a development baseline, not improvement or final-test
performance. The user subsequently verified exact GPU rollback after changing
2,560 weights. Astra experiments subsequently ran on Colab; the bounded search
scored 64/172 before and 63/172 after its candidate edit. Local tests
include real PyTorch tensors and a tiny randomly initialized Phi model; they
do not stand in for the medical checkpoint.

All frontend files are unchanged.

## Run on Google Colab

Open `notebooks/kirby_colab.ipynb` using Colab's **File → Upload notebook**.
The notebook walks through backend upload, GPU checks, Colab Secrets, dataset
upload, an 8-question smoke run, and evidence download. The default is **FP32**,
which can be attempted on a **T4**; it needs approximately 11 GB for weights plus
inference overhead. **BF16** is an explicit option for L4/A100. Neither memory
fit nor performance on other configurations is established by that run. Colab GPU
availability and runtime duration are not guaranteed. `HF_TOKEN` is optional;
the public JSL checkpoint can be downloaded anonymously.

Upload `dist/kirby_colab.zip` when the notebook asks for the backend. To rebuild
that bundle after source changes, run from `backend/`:

```powershell
python tools/package_colab.py
```

The bundle uses an explicit source-file allowlist and excludes frontend files,
credentials, datasets, models, and run outputs. Download experiment evidence
before ending the Colab runtime. Follow the notebook before installing anything
on your laptop; the actual model executes on Colab's GPU. The notebook has been
checked locally for structure and syntax; the user has run its baseline flow
on Colab. We cannot access that runtime directly.

### After the first 256-question baseline

Copy the cells from **`notebooks/kirby_next_steps.ipynb` into your existing Colab
notebook/runtime** and run them in order. This avoids opening a separate runtime
without your files. Upload the rebuilt ZIP in its first cell. The cells restore
workspace variables explicitly and reuse the worker, cached weights, and data.

1. Update source, run tests (including real CPU tensor tests), validate the plan.
2. Run a fresh 256-question baseline and compare every prediction with the most
   recent completed 256-question run. Matching aggregate accuracy is insufficient.
3. Only after reproducibility passes, scale column 0 of layer 0's Phi MLP output
   projection by 0.97, evaluate four fixed development questions, restore the
   saved slice, and verify all model parameter hashes and restored outputs.
4. Download the comparison, patch receipt, before/edited/restored predictions,
   and reports. Download evidence even after a failure.

The fixed edit is an execution probe, not an optimization or Astra repair. It
need not change an answer. No edited checkpoint is retained or exported. Full
weight hashing streams through all parameters and can take longer than the
four-question inference. Do not run other model jobs concurrently in that GPU
runtime. Each probe owns a fresh subprocess and reloads the immutable model.

If comparison reports changed evaluator code, environment, or configuration,
review those differences and create a fresh matching pair. Do not bypass a
prediction mismatch. If Colab deleted the runtime, rebuild the baseline with
the main notebook first; the continuation notebook cannot recover deleted data.

CLI equivalents (substitute actual run directories and revision):

```bash
python -m kirby compare-baselines --first runs/RUN_A --second runs/RUN_B --output runs/comparison-1/comparison.json
python -m kirby rollback-check --manifest data/plan-dev-256/development.json --revision b313a242eb211863f7b62624b345b321a3b114ea --precision float32 --allow-download --questions 4 --layer 0 --column 0 --scale 0.97 --max-seconds 3600 --output runs
```

A passing probe reports `Controlled rollback verified`, `weights_restored: true`,
and `outputs_restored: true`. On inference failure or cancellation it attempts
rollback and preserves an incomplete report. A hard-killed process cannot
finish its journal: discard that worker and reload, never resume its mutation.
Single-patch limits are 1-16 columns, scale 0.9-1.1, at most 200,000 actually
changed values, and at most 1% relative Frobenius change. Numerical no-ops,
nonfinite tensors, stale checkpoint IDs, and duplicate request IDs are rejected.
There is no accepted-edit accumulation in this milestone.

### Earlier workflow: bounded search and 172 development questions

Use `notebooks/kirby_astra_colab.ipynb` with the updated `dist/kirby_colab.zip`
in the existing Colab runtime. It now invokes **`astra-search`**. The older
`astra-attempt` command remains available for reproducing previous experiments.

The search freezes the first **172 IDs** from your existing 256-question plan.
It does not overwrite the plan, resample based on outcomes, or invalidate the
existing controlled rollback prerequisite. It measures a fresh original baseline
on exactly those 172 questions; do not compare its accuracy directly with 97/256.
The final test remains untouched. Frontend files are outside this work.

The default search has these fixed budgets:

1. Observe activations on eight frozen training questions. Choose one column
   in each of eight evenly spaced layers (ties resolved deterministically).
   Activity proposes the shortlist; measured score effects determine ranking.
2. Screen each column at 0.9 and 1.1: at most 16 independent edits on the eight
   screening questions. Rank by mean correct-label log-score gain, then mean
   margin gain, then stable operation identity. Both gains must exceed 1e-6;
   the preferred-correct count must not decrease.
3. Confirm the top two screen winners on the other **56 training questions**.
   Choose the highest-ranked eligible confirmation result. If none qualifies,
   finish with evidence and **zero API calls**.
4. Measure one seeded random column on those same 56 confirmation questions,
   matching the selected layer, scale, and column count. It is reported separately
   and does not affect the ranking. Actual tensor norms may differ; one control
   is not a statistical significance test or a held-out development comparison.
5. Ask Astra to review the selected, measured operation exactly or abstain.
   Local validation rejects substitutions. This is at most **one paid API call**,
   not five sequential development trials. Evaluate at most one candidate on
   the same 172 development questions and restore all weights afterward.

`attempt-history.json` records compatible earlier completed/restored development
evaluations on the same checkpoint, source data, and generation/answer protocol.
Exact previously evaluated operations are excluded from new screening plans.
Different cohort sizes are recorded and never pooled. An ignored malformed
history record is listed explicitly; training probes do not masquerade as
development evaluations. Retain the earlier `runs/` folders for this memory.

For speed, the new diagnostic is **`next_token_single_letter_logprob_v1`**:
one forward scores raw next-token A/B/C/D probabilities. It requires each label
to be a distinct single token. A/B/C/D probabilities are normalized only over
those labels and are not calibrated medical confidence. The older full
label-period protocol remains in legacy runs; their numeric score changes are
not directly comparable to the new protocol.

For all 172 original and candidate development answers, the worker captures raw
first-token logits from the **same greedy generation call**. No extra development
scoring pass is needed. `development_score_comparison` reports log-score and
margin changes separately from accuracy. Accuracy still uses the unchanged
strict generated-answer parser; a score gain is never relabeled as an accuracy gain.

Default maximum work is **364 generation requests, 444 training scoring
forwards, and one Astra request** if every stage reaches the candidate evaluation.
Fewer eligible candidates stop earlier. These are work ceilings, not a measured
GPU runtime promise. The added training search offsets some savings from the
smaller development sample. The cooperative wall-time limit remains 3,600 seconds.
Every edit uses the existing numerical bounds and exact tensor rollback. Four
fixed training restoration questions are rescored after each probe; full weight
hashes and generation calibration are checked before the API call and at run end.

The current notebook uses **`astra-search-v1-172-attempt-id.txt`**. It starts a
separate experiment from the earlier two versions, then reuses its saved result
on repeat execution. To deliberately start another search later, write a new
UUID into this file; preserve all prior evidence.

The downloaded ZIP's `START_HERE.md` now includes screening/confirmation rankings,
history count, the random control, and development score changes. Root-level
`latest-answer_scores_original.jsonl` and `latest-answer_scores_candidate.jsonl`
contain **development** scores for this workflow. `latest-training_scores_original.jsonl`
contains the 64-question training baseline. Every measured probe has its own
question records and rollback receipt under the selected run directory.

```bash
python -m kirby astra-search --manifest data/plan-dev-256/development.json --rollback-report runs/ROLLBACK_ID/report.json --revision b313a242eb211863f7b62624b345b321a3b114ea --precision float32 --allow-download --development-count 172 --diagnostic-count 64 --screening-count 8 --max-seconds 3600 --output runs
```

Deliberate model corruption would be a separate controlled-repair experiment,
not evidence that the intact model's errors are corrupted parameters. Larger
0.5–1.5 scales, hundreds of finite-difference probes, five cumulative edits,
and final-test release validation are not enabled here. They require different
runtime budgets or acceptance/lineage controls. This change implements bounded
training search and evidence-based candidate selection, not retained model repair.

### Earlier workflow: one Astra proposal after verified rollback

The following describes the legacy **`astra-attempt`** command; the notebook
now runs the bounded search above. Legacy runs can still use the CLI example
below with `OPENAI_API_KEY` passed through the worker environment. Install the
optional SDK with `pip install -e '.[astra]'`; never put credentials in source.

The worker uses **`gpt-6-astra` via Responses structured output**, with no model
fallback, external tools, or automatic retries. This is one API request charged
against the API organization's available credits and limits. Account access must be confirmed by the
real request. The default limits are 128,000 serialized request bytes, 4,096
output tokens including reasoning, and a 180-second request timeout. These are
resource ceilings, not a guaranteed billing cap; usage can be unknown after a
transport error. API error bodies and credentials are omitted from artifacts.

On provider failure, the Colab summary now displays the SDK exception type,
HTTP status, recognized provider code/parameter, request ID when available, and
specific guidance. Raw error messages remain excluded. HTTP 429 can mean a
temporary rate limit or exhausted credits/usage limits; the status alone does
not distinguish them. Older reports may contain only `RateLimitError` and 429.
Check the API project's credits and limits before deliberately starting a new
attempt. See [OpenAI error codes](https://developers.openai.com/api/docs/guides/error-codes).

The experiment:

1. Verifies the completed rollback report against the loaded weights, model
   identity, generation settings, environment, and current patch implementation.
2. Measures the unchanged model on the fixed development sample and checks
   repeated calibration outputs.
3. Runs eight preselected **training** diagnostics, observes last-position
   inputs to each Phi MLP output projection, and removes the hooks. Scores the
   four explicit answer labels on the same training cohort, verifies repeated
   scores, and freezes four independent probes: two columns, each scaled by
   0.98 and 1.02. The targets rank highest by training mean absolute activation
   times column L2 norm, with ties broken by layer/column. This is a selection
   heuristic, not a claim of damage. Every probe starts from original weights;
   each is rolled back and its restored training scores checked before proceeding.
   A full-model hash and generation calibration check must pass before Astra.
   No probe is evaluated on development or final questions.
4. Requests a proposal or explicit abstention. The returned checkpoint ID,
   evidence hash, citations, operation, indices, types, and numerical bounds
   are checked locally. The packet includes per-question training answer scores,
   measured probe effects, bounded training excerpts, activation summaries, and
   aggregate development metrics. Proposals must cite a measured probe. Effects
   of untested scales/combinations remain extrapolations. Invalid proposals are
   not repaired or replaced.
5. Measures any valid patch's training answer scores, evaluates the same
   development questions, restores the
   original weights even on failure, and verifies full parameter hashes and
   restored training scores/calibration outputs. Even favorable candidates are rolled back.

Answer-score protocol `teacher_forced_label_period_sum_logprob_v1` sums token
log probabilities for explicit `A.`, `B.`, `C.`, and `D.` continuations after the
unchanged generation prompt. It appends each suffix's recorded token IDs to
the prompt IDs; it does not retokenize the prompt/suffix boundary. It excludes
prompt likelihood and EOS, adds no leading space, and does no length normalization.
Four cache-free, batch-one forwards score each question. Tokenization must
round-trip and fit the context; nonfinite scores stop the run. Probabilities
normalize over these four strings only, not all possible answers, and are not
calibrated medical confidence. The correct-label margin is its log score minus
the strongest other label's score. A separate top-two margin is also recorded.
This diagnostic does not replace greedy generation or the development denominator.

The default eight-question workflow adds 384 scoring forwards if every stage
completes: original/repeat, four edited/restored pairs, and Astra edited/restored.
They appear as `usage.answer_score_forward_calls`; the existing `forward_calls`
field still counts generation requests, not internal autoregressive steps.
Restored log scores must agree within an absolute 1e-6 tolerance and have the
same preferred label. Per-question original/edited scores and margin changes
are retained even when all generated answers stay unchanged.

Activation magnitudes are descriptive evidence, not proof that a channel is
damaged. This restricted first proposal permits one layer and only columns
present in the evidence. The four training probes follow a fixed policy; there
is no adaptive search or accepted-edit accumulation.

`comparison` records original/candidate metrics, corrected/regressed counts,
accuracy delta, and a development signal. It requires a positive correct-count
delta without increased invalid/truncated counts to label that signal improved.
It does not label a candidate repaired, retained, released, or final-tested.

Each attempt writes `editor-evidence.json`, `editor-request.json`,
`editor-response.json`, `proposal.json` when parsable/valid, patch receipts,
stage JSONL predictions, `sensitivity-plan.json`, `probe-*.json`,
`answer_scores_*.jsonl`, and reports. Hidden reasoning is not saved. The SDK
returns token usage and provider response/model identifiers when available.
Download the evidence even after an abstention, rejection, or failed request.

The earlier sensitivity notebook stored its UUID in `astra-sensitivity-v1-attempt-id.txt`.
It leaves the earlier activation-only `astra-attempt-id.txt` untouched, so the
first execution of this version starts a new experiment and may make one paid
API call. Rerunning the cell shows
existing evidence instead of issuing another API request. The CLI also refuses
an existing attempt directory. To intentionally start another paid attempt,
write a new UUID into that file as described in the notebook; preserve prior
attempt directories. A killed worker must be discarded and reloaded.

The evidence ZIP now puts **`START_HERE.md` at its root**, with the selected run's
result, proposal, actual tensor change, training probe effects, restoration,
usage, and links. `latest-report.json`, `latest-proposal.json`, patch receipts,
and original/candidate question records are also copied to the root when
available. `evidence-index.json` lists every exported file with byte size and
SHA-256. Full historical runs and source stay under `backend/`. Environment
files alone are not result evidence. Missing measurements remain explicitly
unavailable; old runs cannot acquire answer-score evidence retroactively.

You can repackage an existing run without running a model or calling Astra:

```bash
python -m kirby.evidence --backend /path/to/backend --report /path/to/backend/runs/RUN_ID/report.json --output /path/to/new-evidence.zip
```

```bash
python -m kirby astra-attempt --manifest data/plan-dev-256/development.json --rollback-report runs/ROLLBACK_ID/report.json --revision b313a242eb211863f7b62624b345b321a3b114ea --precision float32 --allow-download --editor-model gpt-6-astra --diagnostic-count 8 --max-seconds 3600 --output runs
```

Implementation references: [Astra model](https://developers.openai.com/api/docs/models/gpt-6-astra),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses),
[Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).

If an older notebook fails at `python -m venv`, replace its setup cell with the
current notebook's **3. Install the worker** cell and rerun that cell. The
traceback alone does not identify the cause; missing `ensurepip` support is one
possible cause. The updated setup installs `virtualenv` into a private bootstrap
directory, uses its built-in creator/app-data seeder, and creates a fresh worker
directory on every retry. It retains access to Colab's CUDA PyTorch and prints
combined stdout/stderr before reporting a failed command. Existing uploads,
reports, and previous environments are not deleted. There is no need to rerun
the upload cell just to replace the setup cell.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `kirby/datasets.py` | Read the four-option schema; validate labels; check duplicates; freeze selected IDs and source hashes |
| `kirby/evaluation.py` | Fixed prompt, strict answer parser, correct/total and validity metrics |
| `kirby/jsl.py` | Default JSL MedPhi2 loader, built-in Phi architecture, FP32/BF16, context and tensor-loading checks |
| `kirby/models.py` | Explicit model selection; JSL is the default and no failed load switches providers |
| `kirby/checkpoints.py` | Shared architecture, shard, and checkpoint hash verification |
| `kirby/medgemma.py` | Legacy opt-in MedGemma multimodal BF16 adapter; requires its own access |
| `kirby/runner.py` | Two-pass calibration, sequential inference, cooperative cancellation/time limit, run reports |
| `kirby/records.py` | Canonical JSON, SHA-256 identities, UTC timestamps, atomic JSON writes |
| `kirby/patches.py` | Bounded Phi column scaling, byte hashes, exact slice rollback |
| `kirby/rollback.py` | Isolated original/edited/restored execution probe and failure evidence |
| `kirby/reproducibility.py` | Compare baseline identities, protocol, and complete prediction records |
| `kirby/astra.py` | Single Responses request, strict proposal schema, local validation, safe provider failures |
| `kirby/diagnostics.py` | Fixed training diagnostics and read-only Phi activation summaries |
| `kirby/answer_scores.py` | Teacher-forced training label likelihoods, margins, paired changes, and restoration checks |
| `kirby/evidence.py` | Readable ZIP summary, direct current-run artifacts, and byte-hash index |
| `kirby/astra_runner.py` | Original/candidate experiment, usage/evidence recording, unconditional rollback |
| `kirby/search.py` | Bounded search policy, compatible history, candidate ranking, matched random control |
| `kirby/search_runner.py` | Training screening/confirmation, one Astra review, 172-question development comparison |
| `kirby/sycophancy.py` | Read-only training opinion pilot, fixed cohorts, and correction controls |
| `kirby/cli.py` | Local preparation, evaluation, comparison, and rollback commands |
| `tests/` | CPU-only tests for evaluation integrity and worker failure behavior |
| `docs/implementation-notes.md` | Interpretation of the guide, design choices, and remaining milestones |

## Run the tests

Use Python 3.10 or newer. The core and tests have no third-party dependencies.
From the repository root:

```powershell
cd backend
python -m unittest discover -s tests -v
python -m kirby --help
```

If Python is installed through uv and is absent from PATH:

```powershell
uv run --no-project --python 3.14 python -m unittest discover -s tests -v
uv run --no-project --python 3.14 python -m kirby --help
```

The initial test suite was run with CPython 3.14.2 on Windows. This does **not**
validate the GPU dependencies on that Python version. Prefer a GPU environment
with a Python version supported by its PyTorch/CUDA installation, such as 3.11
or 3.12, and validate that environment separately. uv may download an interpreter
when the requested version is not already installed.

## Prepare the benchmark before loading a model

Obtain the official **English / US, four-option** MedQA JSONL splits from the
[original MedQA release](https://github.com/jind11/MedQA). Do not use the five-option
variant or an arbitrary mirror without reviewing its schema. No benchmark data
is bundled or automatically downloaded.

Place the files in `backend/data/medqa/` as `train.jsonl`, `dev.jsonl`, and
`test.jsonl`, then run from `backend/`:

```powershell
python -m kirby prepare --train data/medqa/train.jsonl --dev data/medqa/dev.jsonl --test data/medqa/test.jsonl --output data/plan-v1
python -m kirby validate --manifest data/plan-v1/development.json
```

Preparation:

1. Requires nonempty questions, exactly four nonempty choices labeled A-D, a valid
   `answer_idx`, and agreement with `answer` text when that field exists.
   Repeated choice text is retained, with affected row numbers, question IDs,
   and label groups recorded in each source's `data_quality` metadata. Scoring
   uses only the supplied `answer_idx`, even when another choice has the same
   text. Preparation prints the number of affected rows in each split.
2. Requires the expected official development/test sizes: 1,272 and 1,273.
3. Rejects duplicate question-and-option content within or across the three
   complete splits. Unicode normalization, case folding, and whitespace
   normalization are used; option ordering cannot conceal overlap. This is not
   a semantic near-duplicate detector.
4. Selects 64 train diagnostic cases and 256 development scoring cases using a
   fixed seed of 17 and content-sorted IDs. There is no subject stratification
   yet. These defaults are proposed starting settings, not measured optima.
5. Saves development/diagnostic membership in `development.json`, and all final
   test IDs and their source metadata in a separate `final.json`.

The output directory must be new. Changing selection or settings creates a new
plan. If duplicate questions or release mismatches are found, investigate the source and
document a reviewed data policy; this command does not silently drop examples
or change the official denominator.

Adapter v2 permits and records repeated option text; it does not edit dataset
files. Plans created with adapter v1 must be regenerated in a new output
directory because their recorded adapter and metadata differ. Keep old plans
with their original run evidence.

The baseline command accepts only the development manifest. Its worker opens
only the development source, rechecks its hash, and follows the recorded order.
It does not read final questions. Diagnostic IDs are reserved for future
inspection; this first baseline worker does not run the diagnostic cohort.

Manifests are trusted, local operator inputs. Their hashes detect changes; they
are not signatures. A future API must resolve registered IDs and keep final data
outside the editor's process permissions. Do not expose these local path-based
commands directly to a browser or editor.

## Run the real JSL model when compute is ready

Required environment:

- Public checkpoint `johnsnowlabs/JSL-MedPhi2-2.7B`. Hugging Face reported
  `gated=false` and `private=false` when checked on 7 November 2026. It does not
  use a John Snow Labs hosted API, a Healthcare NLP license key, or Gemma access.
- An explicit 40-character commit SHA. The checked public revision is
  `b313a242eb211863f7b62624b345b321a3b114ea`; `main` is rejected by the worker.
- A working CUDA/PyTorch installation with sufficient memory. The default
  `float32` uses around 11 GB for model parameters alone. The explicit
  `bfloat16` option requires native BF16 support. This uses one GPU, batch size
  one, with no CPU offload and no quantization.
- The actual benchmark files and a prepared manifest.

Create an isolated environment on the GPU host, install a CUDA-compatible
PyTorch build, then install the optional backend dependencies:

```powershell
python -m pip install -e ".[gpu]"
```

`pyproject.toml` contains compatibility ranges based on the selected loader API,
**not a tested dependency lock**. After the first successful model load and
calibration, pin that environment and retain its package lock with the run.

Anonymous downloads are supported for this checkpoint. `HF_TOKEN` is optional
if authentication is needed for Hub rate limits. Do not put tokens in a
repository URL, manifest, report, or frontend. Baseline and rollback commands
need no OpenAI key. Only `astra-attempt` uses `OPENAI_API_KEY`.

From `backend/`, use the checked revision:

```powershell
python -m kirby baseline --model johnsnowlabs/JSL-MedPhi2-2.7B --precision float32 --manifest data/plan-v1/development.json --revision b313a242eb211863f7b62624b345b321a3b114ea --allow-download --output runs
```

`--allow-download` permits fetching the checkpoint (several gigabytes); without
that flag, execution is cached-only. The loader uses the approved repository,
disables remote code, requires Safetensors, checks referenced shards, and hashes
the package. It uses Transformers' installed `PhiForCausalLM` implementation
instead of the checkpoint's old remote `auto_map` references, and rejects
missing, unexpected, or mismatched weight tensors.

The real tokenizer/config check passed anonymously on 7 November 2026 with
Transformers 4.57.6 and Jinja2 3.1.6: `CodeGenTokenizerFast`, 2,048-token context,
and 62 tokens for the synthetic prompt probe. No model weights were downloaded
and inference was not tested by this check.

The real tokenizer and configuration can be checked without weights using
`python tools/check_jsl_tokenizer.py` in an environment with Transformers
and Jinja2 installed. This is a public metadata/tokenizer check, not an inference test.

Phi's context is **2,048 tokens** including the prompt and generation allowance.
Oversized questions stop the run; they are not silently truncated or omitted
from the denominator. A context-length problem or a high invalid-answer rate
needs protocol investigation before moving to a larger benchmark run.

The previous MedGemma adapter remains available only when explicitly selecting
`--model google/medgemma-1.5-4b-it`, with a MedGemma revision and native BF16 GPU.
Its gated access requirements still apply; JSL errors never fall back to Gemma.

The first four development questions are run twice unchanged. If output differs,
if more than 25% of those answers are invalid, or if any answer is truncated,
the run stops. The 25% threshold is an initial calibration guard, not a benchmark
success criterion. Change the protocol only in a new run, then repeat baseline
measurement. The initial eight-token allowance is configurable with
`--max-new-tokens` and is not assumed adequate until calibration passes.

Every answer uses the fixed prompt and greedy generation. Only newly generated
tokens are decoded. Profile `medqa_text_v2` accepts uppercase A, B, C, or D,
optionally followed by one period, after whitespace trimming. JSL generation
stops when its entire generated text is a complete label such as `D.`. This
observes the model's choice without forcing a letter or extracting one from
prose. The period is retained in raw predictions. EOS still ends generation;
reaching a valid label stop at the token limit is completion, not truncation.
An explanation such as `The answer is B` or `B. explanation` remains invalid.
Invalid and genuinely truncated outputs count as incorrect in the denominator.

This protocol replaces v1's bare-letter-only scoring after Colab calibration
showed stable `D. ...` and `C.` outputs. The parser version and JSL stop policy
are recorded in each new run. Do not relabel old runs or compare scores across
these protocols as evidence of model improvement. Keep the failed run and
establish a fresh v2 baseline. Failed Colab runs now display calibration metrics
and raw first-pass outputs to distinguish formatting and token-limit failures.

The default wall-time limit is 1,800 seconds. Set `--max-seconds` after measuring
throughput. The worker checks time and cancellation between calls; it cannot
interrupt an in-flight GPU kernel or checkpoint download. Ctrl+C writes an
incomplete report when Python can handle the interruption. An abrupt process
kill leaves the initial running/incomplete journal and any flushed predictions;
restart recovery and hard process deadlines are future work.

## Inspect results

Each run receives a new UUID directory under `runs/`:

```text
configuration.json  Frozen inputs, prompt, parser, limits, source/selected IDs
model.json          Checkpoint/file hashes, generation settings, environment
events.jsonl        Timestamped sequential events with real question counts
calibration.jsonl   Both unchanged calibration passes
predictions.jsonl   Generated text, expected/predicted letter, validity, latency
report.json         Authoritative status, measured metrics, usage, limitations
report.md           Human-readable baseline report
```

Files for phases never reached may be absent. Initial and terminal reports are
written atomically. JSONL records are flushed per question; they are not a
transactional database or a guarantee of survival after power loss.

A completed audit says **Baseline evaluated**. An interrupted or failed audit
says **Run incomplete**, including partial counts if available. Neither label
claims an improvement. Final, regression, and comparison results are explicitly
unavailable. There are no fabricated scores, costs, API calls, or artifacts.

Run records capture model files, selected data, settings, source code hashes,
package versions, hardware, and the rendered template probe. These make
reproduction inspectable; cross-hardware numerical identity is not guaranteed.

## Current experiment: baseline sycophancy pilot

The bounded search has now run on Colab: the original scored 64/172 and the
candidate 63/172, with weights restored. It did not improve accuracy.
The next experiment measures whether the unchanged model follows unsupported
user opinions before considering pruning. See [the pilot guide](docs/SYCOPHANCY_PILOT.md)
and `notebooks/kirby_sycophancy_colab.ipynb`.

This training-only pilot uses 64 frozen questions, two opinion phrasings,
explicit answer-key hints, and separate synthetic evidence-update controls.
It makes no Astra calls and changes no weights. It exports a readable summary,
exact prompts, per-question answers, fixed cohort IDs, and weight hashes.
The original Astra commands remain available for reproducing earlier experiments.
The new training-only single-unit pruning screen is documented in
`docs/PRUNING_EXPERIMENT.md`. Multi-round selection, regression evaluation, retained candidates,
export/reload, and final evaluation remain separate milestones.

## JSL model terms and future edited artifacts

JSL publishes this checkpoint under **CC BY-NC-ND 4.0**, with additional
[model terms](https://huggingface.co/johnsnowlabs). The published terms restrict
commercial use and distribution of modified weights. This baseline evaluator
exports reports, not model weights. The original guide's proposal to release an
edited checkpoint needs a compatible license or separate permission before
distribution. Baseline implementation does not establish permission for every
future hackathon/product use.

Column editing now targets Phi's validated `mlp.fc2.weight` projection, rather
than Gemma's `mlp.down_proj`. The current Astra experiment always rolls back
and exports evidence only. Edited-checkpoint distribution and final evaluation
remain unimplemented.
