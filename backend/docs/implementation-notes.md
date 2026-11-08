# Reading the guide and choosing the first slice

## Current target update: John Snow Labs

The user could not obtain MedGemma access and requested a John Snow Labs model.
The default is now `johnsnowlabs/JSL-MedPhi2-2.7B`, using immutable revision
`b313a242eb211863f7b62624b345b321a3b114ea`, resolved from the public Hugging Face
API on 7 November 2026. The repository reported `gated=false` and `private=false`.
The fetched config identifies `PhiForCausalLM`, 32 layers, a 2,048-token context,
and BF16 stored tensors. It supplies a text chat template and Safetensors shards.

The JSL loader uses the installed Transformers Phi implementation and the
checkpoint's tokenizer with remote code disabled. It refuses missing/unexpected
weight keys and prompt overflow rather than claiming compatibility based only
on a matching model name. GPU loading and generation still need validation in
Colab. `tools/check_jsl_tokenizer.py` checks small public tokenizer/config files
without downloading the checkpoint weights. That check passed without
authentication using Transformers 4.57.6 and Jinja2 3.1.6; the real
`CodeGenTokenizerFast` rendered and tokenized the probe correctly. This is
tokenizer/config validation only, not evidence of successful GPU inference.

Subsequently, the user reported a Colab T4 run that loaded this checkpoint and
completed two stable calibration passes. All four outputs failed the v1 format
check: they began with labels such as `D.` and `C.`, and two continued into
choice text until the eight-token limit. This demonstrates execution, not a
completed baseline or measured accuracy. Profile v2 accepts an optional final
period, and the JSL adapter stops unconstrained generation once the complete
generated suffix is exactly a capital A-D plus a period (surrounding whitespace
allowed). It does not scan explanations for a letter. The raw period remains
in evidence. The stop policy is frozen with the model configuration, and v1
artifacts remain unchanged. The new protocol still requires validation on the
real GPU; larger token limits alone would not fix the observed format mismatch.
The 40 CPU tests pass. Replaying all four reported output strings through the
cached pinned tokenizer reaches the expected complete label in two tokens for
each case. That verifies token boundaries and stopping logic, not fresh model
inference or medical accuracy.

The Colab default is explicitly FP32, so a T4 can attempt the approximately
11 GB parameter footprint plus runtime overhead. BF16 remains an explicit
choice for compatible hardware. No FP16 or quantized fallback is applied.
Runtime precision is frozen in configuration and metadata. Hugging Face
authentication is optional for JSL; no medical-model gate or hosted JSL API
is part of this path. The old MedGemma adapter remains opt-in.

The [JSL model card](https://huggingface.co/johnsnowlabs/JSL-MedPhi2-2.7B) and
[organization terms](https://huggingface.co/johnsnowlabs) specify CC BY-NC-ND.
Commercial use and distribution of modified weights are restricted. This
affects the guide's future edited-artifact release plan; the current evaluator
only writes reports. The eventual patch registry must also use Phi's real MLP
structure instead of assuming Gemma's projection names.

The remainder records the reasoning behind the initial MedGemma foundation;
the target/precision update above supersedes those initial target-specific
choices. Dataset isolation, fixed scoring, calibration, and truthful reports
remain unchanged.

The supplied guide describes a proposed implementation and a prior tiny-network
backend. At the start of this task, this checkout contained a placeholder
`index.html` and `blank.txt` files under `backend/` and `frontend/`. The described
toy package was not present, so its endpoints, tests, and behaviors cannot be
claimed or reused here. Existing files have been preserved.

The guide is a design reference supplied by the user. Its operational directions
are not independent authorization for paid API experiments, model downloads,
publication, or claims about completed work. This first slice implements the
baseline foundation requested by the user. It does not complete every milestone
in the document. The user's subsequent instruction to leave the frontend alone
applies to the entire implementation.

## Important distinctions preserved

- **Subject and editor:** MedGemma will answer benchmark questions locally.
  Astra will eventually propose operations through a constrained backend
  interface. No editor is called by this implementation.
- **Engineering and research:** passing worker tests is evidence about worker
  behavior. It does not show useful weight edits or reproduce a published score.
- **Development and final:** preparation examines complete splits to check
  overlap. The baseline worker receives only a development manifest, and final
  manifests are refused. This is an initial interface boundary; hosted/editor
  process isolation still needs implementation.
- **Calibration and evaluation:** prompt and parser failures must be understood
  before attributing failures to weights. Two unchanged calibration passes
  precede scoring. Their work is included in usage records.
- **Valid and correct:** a syntactically valid wrong letter is a valid error.
  A verbose answer is invalid under this profile. A truncated answer is counted
  as incorrect even if the visible letter matches. All remain in the denominator.
- **Precision and compression:** BF16 is the starting precision. This project
  has not demonstrated a quantization regression or its repair.
- **Proposed defaults and evidence:** seed 17, sample sizes, eight generated
  tokens, four calibration questions, a 25% invalid calibration guard, and a
  30-minute cooperative deadline are recorded engineering choices. None is an
  experimentally established optimum.
- **Audit and repair outcomes:** the guide's four result labels concern editing
  experiments. A successful baseline audit uses `Baseline evaluated` because no
  candidate comparison exists. Failures retain `Run incomplete`.

## Why these modules come first

`QuestionInput` carries only the allowed question and choices. The evaluator
retains labels separately. This keeps arbitrary dataset fields, including
explanations, out of generated prompts. Question content can still contain
instructions as data; it cannot change evaluator code or membership.

The adapter fingerprints normalized question text and sorted option text.
Reordered choices therefore do not conceal exact overlap. Sampling happens
before inference and is deterministic under file reordering. Exact source bytes
are separately hashed because matching IDs alone would miss changed labels.

The user's Colab source failed at train row 1319 because option texts repeated.
Adapter v2 removes the unsupported assumption that four labeled choices must
have four distinct texts. It preserves every choice and the supplied answer
letter, and records normalized duplicate-option groups in source metadata.
This is a source-preserving evaluation policy, not a correction of ambiguous
questions: an equivalent-text choice with a different letter still scores as
incorrect. No rows are dropped, and duplicate-question and split-overlap
checks remain mandatory. Existing v1 manifests require a new preparation.

Generation follows the official processor and multimodal loader path, with
remote code disabled and no fallback to a generic text architecture. It fixes
greedy generation, attention implementation, precision, batch size, and EOS
configuration. It does not carry inference caches between calls.

The worker depends on a small predictor interface so the evaluator can be tested
without a GPU. Synthetic predictors live only in tests. A failed real loader
cannot silently become a fake successful run.

Runs are synchronous at the worker boundary, which is appropriate for a local
CLI. A future HTTP API must enqueue this work and return a job ID rather than
block a browser request. The records are deliberately simple files for this
first slice. SQLite job ownership and recovery belong in the upcoming queue.

No OpenAI SDK or key is needed for baseline or controlled rollback. The optional
`astra` extra now provides the single-proposal API integration described below.
Official documentation confirms the model/interface; it does not establish
access for the user's account.

## Remaining work, in dependency order

### Implemented controlled-edit slice

The user's subsequent 256-question development report completed with 97 correct,
no invalid/truncated answers, and a mean scoring latency of 0.469 seconds. The
reported run ID is `b3096c7a-c806-42e1-8338-c86ad0f72576`. Full run artifacts have
not been inspected locally. This motivated the next controlled execution step.

`compare-baselines` checks saved configuration/manifest digests, checkpoint
identity, environment, evaluator code, question ordering, metrics, and exact
prediction fields. Operational download permissions and measured latency are
excluded from equality. Two copies of one run are rejected. Newly added utility
modules do not invalidate a repeat, but changes to evaluator/adapter code do.

`rollback-check` performs one fixed, bounded column scale on the runtime-verified
Phi `model.layers.N.mlp.fc2.weight` projection. It uses saved original slices,
never division by the scaling factor, to restore exact bytes. Before/after full
parameter hashes detect unintended weight changes outside the edited column.
The original and repeated-original outputs must first be stable and pass the
answer protocol. The edited stage is measured and always rolled back; restored
outputs must match. All stages use development data only. A worker crash is
recovered by loading the original on the next attempt; no on-disk model is edited.

The primitive deliberately allows no commit or retained-candidate operation.
Thus every attempt returns to the original and has no accumulated edits.
Multi-operation transactions, accepted lineage, export/reload, and persistent
idempotency across API jobs still require implementation before Astra search.
Request IDs prevent duplicate application within a live editor instance.

Local validation uses real CPU PyTorch and a tiny Phi model, including exact
logit restoration, BF16 no-op rejection, and injected inference/write failures.
The user subsequently reported `Controlled rollback verified` on the actual
checkpoint, with both full weights and outputs restored. Run
`78fd0b2c-7383-4edb-8467-e107cd6c3c80` changed 2,560 values in
`model.layers.0.mlp.fc2.weight`, with a 0.97 scale and relative tensor change
0.00036711119119567137. Its before/restored tensor hashes match. Original and
edited accuracy both remained 2/4; this establishes mechanics, not improvement.

### Implemented Astra integration

`astra-attempt` makes one `gpt-6-astra` Responses request using strict JSON
schema output. The worker first checks the completed rollback evidence against
the loaded model, environment, development plan, current patch code, and full
weight hash. It then evaluates the original development sample and gathers
fixed training diagnostics with read-only Phi activation hooks. Post-hook
calibration must match before contacting the provider.

The bounded request includes training examples, descriptive activation/column
norm summaries, and aggregate development scores. It has no final data, tools,
file access, executable code, or raw weight tensors. Only one layer and columns
present in the packet may be proposed. Checkpoint/evidence identity, schema,
citations, integer types, limits, cast-level no-ops, and actual tensor changes
are checked locally. The output may explicitly abstain. Provider errors,
refusals, incomplete outputs, and invalid proposals are recorded without a
fallback patch. No proposal is silently repaired.

Every applied patch is evaluated on the identical development questions and
rolled back, including improved results. Reports include corrected/regressed
counts, both scores, raw output and usage, patch hashes, and restoration checks.
The positive signal criterion is more correct answers without increased
invalid/truncated counts; it is not a release decision. No candidate is retained
or exported in this slice. Hidden model reasoning and credentials are excluded.

The API client fixes the official endpoint, disables SDK retries, records
unknown usage after failures, and enforces byte/output/time ceilings. UUID
attempt directories are exclusive. The Colab notebook persists its attempt ID
and reuses existing evidence on a repeated cell execution, avoiding accidental
duplicate API requests. A deliberate new attempt requires a new UUID.

Local validation passed 79 tests with PyTorch 2.6.0 CPU, Transformers 4.57.6,
and OpenAI SDK 2.54.0. Tests include real SDK serialization through mocked HTTP,
Phi activation/mutation/rollback, rejection cases, provider failures, secret
handling, and duplicate attempts. No live API request was made locally.
`notebooks/kirby_astra_colab.ipynb` is the runnable live continuation.

The expanded tests also exposed transient Windows atomic-replacement failures
when report files were briefly locked. `records.write_json` now retries only
the atomic replacement on Windows sharing/access errors, with a bounded delay,
preserving the previous file on persistent failure. It never deletes the old
report to force a write.

### Training score sensitivity and readable evidence exports

The user supplied the full first successful Astra run. Inspection confirmed
that run `a9ab1b97-4deb-43a0-837b-edacd1cab1ec` scaled layer 30 column 8941 by
0.98 and changed 2,560 parameter values, with relative tensor change
0.0003010050802839483. All 256 generated answers were identical before/after;
accuracy stayed 97/256. Tensor and full-model restoration hashes match.
The old artifacts contain no answer likelihoods, so they cannot establish
whether answer probabilities moved without changing the winning answer.

The new `answer_scores.py` module scores explicit label-period suffixes using
teacher forcing. It uses the original chat-template prompt, records suffix
token IDs, sums full-vocabulary token log probabilities (no EOS or length
normalization), and normalizes only over A./B./C./D. for display. Correct-label
and top-two margins are separate from the existing generation-based accuracy.
The module is separate from `jsl.py`, preserving its generation implementation
and metadata. `patches.py` is unchanged, so the user's existing verified
rollback prerequisite remains usable with the same model/environment/plan.

Before the single Astra call, the runner freezes two target columns ranked by
training mean absolute activation times column norm. Each is independently
scaled by 0.98 and 1.02 and measured on the same fixed training cohort. All
four probes are journaled, including per-question scores, direction and size
of score/margin changes, tensor receipts, and restoration. The entire plan
is chosen before any intervention outcomes are observed. No development or
final question is evaluated under these probes. This is limited sensitivity
evidence, not identification of damaged channels or demonstrated generalization.

Each probe restores before fallible logging. Restored scores must match within
absolute log-score tolerance 1e-6 with identical preferred labels. A full-model
hash and generation check precede Astra. The final proposed candidate also
gets training score measurements before the existing development evaluation.
Training gains never substitute for development gains. All edits are restored.
Failures, deadlines, and interrupts prevent proceeding to the provider and
retain partial evidence; the worker never substitutes an unmeasured patch.

The Colab continuation uses `astra-sensitivity-v1-attempt-id.txt` so an earlier
activation-only report is not silently reused. Repeated executions still reuse
the new attempt instead of automatically spending on another request.

The new `kirby.evidence` exporter creates `START_HERE.md`, root-level latest-run
artifacts, and a complete SHA-256 file index. Historical files remain nested.
It can repackage older evidence without inference or API access, explicitly
marking their missing score diagnostics. No raw datasets, weights, or credential
files are bundled. The summary distinguishes incomplete runs and unmeasured
quantities from passes or zeros.

Validation for this change: 97 tests passed with real CPU PyTorch 2.6.0,
Transformers 4.57.6, and mocked OpenAI requests. Coverage includes independent
multi-token likelihood arithmetic, restored scores, interrupted/failed probes,
failed artifact writes, no provider call after a failed restoration check,
probe citations, repeat-click protection, and complete ZIP byte-hash indexing.
The cached real JSL tokenizer was also exercised with a tiny random Phi model:
A./B./C./D. encode as [32,13]/[33,13]/[34,13]/[35,13]. A 0.98 test edit moved
label log scores by up to 0.00013828277587890625 and exact restoration passed.
This is a CPU compatibility/numerical test, not inference on the pretrained
checkpoint or validation of the new workflow on Colab. No live API call was made.

### Bounded search after the second measured no-gain result

The supplied second-run evidence (`fbe60412-32c1-4fdd-98e5-e51d0a5af704`)
passed all 135 file hash checks. It reduced layer 30 column 1819 by 2%, produced
small reproducible training margin changes, and changed none of the 256
generated development answers. Accuracy remained 97/256. The selected candidate's
scores exactly matched its earlier probe. All weights and score checks restored.

The user requested 172 development questions and delegated which outside advice
to adopt. `astra-search` freezes a 172-ID prefix of the existing plan, preserving
the original manifest and rollback identity. It uses a fresh paired baseline,
never a comparison of the old 256-question aggregate against the new cohort.

To balance wider search with runtime, at most eight layers contribute one
shortlisted column each. Sixteen 0.9/1.1 edits are screened on eight training
questions; the best two are independently confirmed on the other 56 frozen
training questions. Eligibility requires positive correct-label log-score and
margin change with no decrease in preferred-correct count. Ranking and ties are
explicit. A seeded random same-layer/scale/column-count control is measured on
confirmation questions but not used to select a winner. It does not exactly
match tensor norm and is not a significance or final-test claim.

Compatible completed development candidates form an audited attempt history.
Exact repeated operations are excluded. The packet supplies summaries and a
bounded recent history rather than entire historical question sets. Astra must
review the top eligible confirmed operation exactly or abstain; local validation
rejects substitutions. No eligible candidate means no API spend. The maximum
is one provider request and one candidate development evaluation per search.

The new score protocol uses single-token A/B/C/D next-token log probabilities.
This cuts training scoring from four forwards to one per question. For the
development sample, raw first-token logits are captured during the existing
greedy generation call, with no extra model pass. It is intentionally versioned
separately from earlier whole-label-period scoring. The answer parser and
generated-answer denominator remain unchanged. `patches.py` is unchanged; all
edits restore and no accepted lineage, export, or final access was added.

The default cap is 364 generation requests plus 444 training score forwards,
with cooperative 3,600-second wall limit. Runtime still needs measurement on
Colab. Reports preserve per-question evidence, selection partition, operation
ranking, history exclusions, random control, and separately labeled development
margin changes. The new attempt file prevents confusing old results with the
new 172-question experiment while retaining repeat-click protection.

Local validation: all 110 tests passed with CPU PyTorch 2.6.0, Transformers
4.57.6, and mocked OpenAI requests. Checks cover actual Phi generation with
raw-logit capture, identical generation with capture enabled/disabled, CLI
172/64 defaults, disjoint training partitions, deterministic ranking and
history exclusions, no-call outcomes, invalid provider operations, interrupted
probes, restoration drift, failed artifact writes, and new evidence summaries.
The history reader was separately checked against the supplied second-run
download: it recognized both earlier development-evaluated candidates and kept
their 256-question cohort sizes explicit. No live model weights or API calls
were used to validate this new workflow locally.

### Remaining milestones

1. **Finish validating milestone 1:** download and inspect the user's full run
   evidence, record exact dependency versions, and repeat the completed baseline
   using the new comparison command.
   Add the MedMCQA single-choice adapter and a reviewed disjoint allocation.
   Add an audit endpoint backed by a model registry and sequential worker queue.
2. **Numerical mutation:** extend the verified single-edit rollback path with
   multi-operation atomicity, cumulative
   limits for retained edits, persistent request identity, accepted lineage,
   and permitted export/reload verification.
3. **Training sensitivity:** validate the newly implemented score/probe workflow
   on the user's Colab checkpoint and retain the result, whether favorable,
   neutral, rejected, or failed. The first activation-only Astra proposal has
   now been inspected and recorded above.
4. **Bounded search:** add diagnostic and attempt budgets, checkpoint-tagged
   evidence, accepted lineage, and acceptance rules. Primary development gain
   compares to the retained candidate; regression and invalid-rate tolerances
   compare to the ORIGINAL baseline to prevent drift across accepted edits.
5. **Final evaluation:** end editing, export, reload in a fresh process, freeze
   the candidate, and compare original/candidate on the same final questions
   once. Add paired bootstrap intervals, corrected/regressed counts, and release
   decisions. Final feedback must never quietly restart editing.
6. **Product integration later:** asynchronous versioned API, authenticated
   artifacts, imports/uploads, SQLite metadata, and frontend integration. No
   frontend work is part of this change.

The first implementation deliberately has no weight patch schema disconnected
from real tensor validation, no unimplemented endpoints presented as working,
and no benchmark claims derived from synthetic tests.

## External implementation references checked

- [Official MedGemma 1.5 4B model card](https://huggingface.co/google/medgemma-1.5-4b-it):
  multimodal loader/processor example, BF16 generation, and gated access.
- [Transformers multimodal templates](https://huggingface.co/docs/transformers/en/chat_templating_multimodal):
  processor-owned chat formatting and generated-token handling.
- [Original MedQA repository](https://github.com/jind11/MedQA): JSONL splits and
  the separate four-option release.

These references support the implementation approach. They do not establish
that this checkout's GPU path has successfully run.
