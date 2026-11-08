# EEG shortcut audit: run and interpret

This is the current Kirby experiment. The earlier MedPhi/sycophancy/pruning
workflows remain available for reproducing their evidence. The new workflow
does not use their datasets, checkpoints, attempt IDs, or Colab environment.
Saved reports can be replayed locally in `demo.html`.

## Feedback-guided follow-up (2026-11-08)

Run `88dba5f4-e935-42d3-b85f-8c4f28f0fe0d` completed: six eligible models,
12 Astra diagnosis calls, all abstaining (`uncertain`), and zero Astra repair
calls. The previous runner skipped repairs after either `clean` or `uncertain`.
It therefore did not test whether development-guided Astra edits could help.

The new default `repair_policy=feedback_guided` preserves every blind diagnosis
and then requests repair proposals for each eligible model/A-B condition,
regardless of its diagnosis. Packets include the frozen first response and
development measurements, without hidden condition labels or final results.
Astra can retain an uncertain diagnosis while proposing a testable edit. It
can still stop if no defensible candidate exists, including zero flip rate.
Every candidate must pass the same four acceptance checks; no accepted edits
are guaranteed. Reports record policy, repair request count, attempted and
accepted edits, and stop reason. `--repair-policy diagnosis_gated` restores
the old gate. Earlier reports without this config field used that gate;
the evidence format remains `eeg_shortcut_audit_v1`.

The notebook sets `REPAIR_POLICY='feedback_guided'` and streams worker output
including diagnosis requests, repair requests and edit acceptance/rejection.
Upload the rebuilt ZIP and use a fresh run UUID. Subject 1's final session has
already been inspected: this repeat is exploratory, not fresh confirmation.
Freeze the revised procedure before a later study on previously unseen subjects.

## Run on Colab

For the next **new-subject replication**, open
`notebooks/kirby_eeg_replication_colab.ipynb`, enable GPU and the API secret,
then run all cells and upload the updated `dist/kirby_eeg_colab.zip`. The separate
notebook fixes subjects 2/3, seeds 0/1/2, clean/blink and both A/B conditions.
It verifies `experiments/eeg_replication_v1.json` and normalized source hashes
before execution, retains its own run ID across reruns, and includes the plan
in the exported evidence. Maximum 144 requests and $100 estimated token cost;
this is not a provider billing limit. Do not modify the settings after seeing
results. See `EEG_ROADMAP.md` for the predeclared analysis and remaining limits.

The original subject-1 exploratory launcher remains available:

1. Upload `notebooks/kirby_eeg_colab.ipynb` through Colab's **File > Upload notebook**.
2. Select a GPU runtime. Add `OPENAI_API_KEY` in Colab Secrets and enable notebook access.
3. Run the setup cell and upload **`dist/kirby_eeg_colab.zip`** when prompted.
   This is a different source bundle from `kirby_colab.zip`.
4. Leave the exploratory follow-up configuration at subject 1, seeds 0/1/2, clean/blink,
   A/B evidence, 300 maximum epochs and 30-epoch patience. Run the remaining cells.
5. Preserve the downloaded `kirby-eeg-evidence-<UUID>.zip`. Start with its
   root `START_HERE.md`. A failed/interrupted run exports partial evidence;
   check its status before interpreting any numbers.

The default enables Astra. Setting `RUN_ASTRA=False` deliberately runs only
baselines. No paid request is made until development gates pass. The default
limit is 72 dispatched request attempts, five repair attempts per model/A-B
condition, and a $50 token-cost estimate. Pricing is configurable in
`EEGConfig`; estimates use $10/M input, $1/M cached input and $50/M output
as checked against the official model page during implementation. These are
not provider billing controls. A failure with unknown usage reserves an
upper estimate rather than claiming zero cost. SDK retries are disabled.

Rerunning notebook cells displays the same saved experiment. A run UUID is
exclusive, even after failure; it cannot repeat paid requests or final scoring.
For a deliberate new experiment, set `RUN_ID` to a fresh UUID string. A changed
source bundle or changed settings cannot silently reuse an old experiment.
Source and worker environments are isolated by source hash under
`/content/kirby-eeg-workspace`. Colab storage is temporary: download evidence.
The notebook currently does not resume an interrupted training/API search.

Environment setup uses `venv --without-pip` because some Colab images omit
`ensurepip`. It checks the worker interpreter and pip, repairs a partially
created environment in place, and uses base pip's `--python` option to seed
worker pip if needed. Existing experiment records are not deleted or restarted.

Equivalent CLI after `pip install -e '.[eeg]'`:

```bash
python -m kirby.eeg run --subjects 1 --seeds 0 1 2 --conditions clean eye_blink
python -m kirby.eeg export eeg-runs/EXISTING-UUID
```

`python -m kirby eeg-run ...` and `eeg-export ...` are aliases. Add
`--no-astra` for an explicit baseline-only run. Add `--extended-baselines`
for integrated-gradients diagnosis, last-layer clean-label retraining and a
paired fully clean-trained model comparator. Broader studies can use
`--subjects 1 2 3 ... 9 --conditions clean eye_blink line_noise muscle`;
increase the frozen API budget deliberately before starting a larger run.
No real EEG training or paid Astra request was run locally during implementation.

## Frozen protocol

**Dataset and partitions.** MOABB downloads BNCI2014_001 (BCI Competition IV 2a).
The explicit MAT reader uses all 22 EEG channels and three EOG channels,
250 Hz, and four seconds starting at the imagery cue: MAT trial start plus
two seconds. Both sessions contain 288 trials, 72 per class. Session 1 is
stratified into 184 fitting, 48 internal-validation and 56 development trials.
The split seed stays fixed across training seeds. Session 2 supplies 288 final
trials. Preprocessing prepares that session, but model selection cannot load
its arrays until every model/method has been frozen in `selection-lock.json`.

Epochs use independent 0.5–100 Hz filtering, rather than an 8–32 Hz motor
imagery default that would remove the planted signals. EOG regression and
channel mean/standard deviation are fitted only on the 184 fitting trials.
The same coefficients transform validation, development and final trials.
Dataset artifact flags are recorded, not used to discard difficult trials.
“Clean” means **no planted artifact**; regression does not prove artifact-free EEG.
The source recordings already have a 50 Hz notch; the injected hum is synthetic.

**Model.** EEGNet-8,2 uses a 125-sample temporal kernel, depth multiplier 2,
separable convolution, 0.5 dropout and a four-class linear head. It has 3,932
trainable parameters before folding and 3,892 inference parameters afterward.
Adam uses learning rate 0.001 and batch size 64. Early stopping chooses the
lowest internal-validation loss; development/final data never choose training
epochs. Matched clean/artifact models start with the same seed. This is a
PyTorch implementation of the architecture, not a claim of reproducing published
benchmark accuracy. Deterministic operations are requested; unsupported CUDA
operations fail rather than silently change determinism.

All three batch-normalization layers are folded into convolution weights AND
biases. Original and folded logits and predictions are checked on development
inputs at 1e-5 absolute/relative tolerance. All inference parameters are exported
with nine significant decimal digits and checked for exact float32 roundtrip.
Four significant digits would not be a lossless export.

**Planted signals.** Blink uses partition-local low-frequency EOG snippets,
normalized to unit peak, then multiplied by 3 training-channel standard deviations
at Fz and 1.5 at FC1/FCz/FC2; it is paired with left hand. These are EOG-derived
waveforms, not verified blink annotations. Hum uses a 50 Hz random-phase sinusoid
of amplitude 0.5 on every EEG channel, paired with feet. Muscle uses 60–100 Hz
windowed noise bursts at C5/C6 with unit pre-window standard deviation, paired
with tongue. The corresponding class has artifact probability 0.8, others 0.1.
Amplitudes, prevalence, waveform generation and split seeds are frozen in config.

Each base trial has one sampled waveform shared across original-correlation,
no-planted-artifact, swapped-correlation, uniform-prevalence and all-artifact
versions. Swapped moves the paired class to the next class modulo four; uniform
uses probability 0.275 for all classes. Clean versus all-artifact predictions
measure paired flip rate. Variants of a trial are correlated observations, not
additional independent sample size.

**Eligibility gates.** The paired clean-trained model must score at least
0.35 clean development accuracy. An artifact-trained model must also have at
least 0.15 original-minus-swapped development accuracy. Failed gates exclude
the model from Astra and final evaluation, with its results retained in the
report. These are predeclared engineering thresholds, not statistical proof.
There is no automatic amplitude tuning or rerun until a favorable result appears.

**Blind diagnosis.** Random UUIDs identify models. Astra sees architecture,
channel/class meanings and all inference parameters. A provides raw weights;
B adds deterministic frequency responses and signed spatial weights. Neither
gets training condition, subject, seed, hidden ground truth, EEG examples or
performance. All first diagnoses finish before any repair feedback is requested.
First responses are scored separately and never retroactively improved by repair.
`uncertain` is an abstention and stays in the all-model denominator; coverage
and false positives on clean models are reported separately. Excluded models
are not quietly counted as correct diagnoses.

**Repairs.** Under the default feedback-guided policy, all eligible diagnoses
can proceed to up to five single operations. The legacy diagnosis-gated policy
skips clean/uncertain diagnoses. Repair packets contain current weights, the
frozen blind response and development feedback only. Temporal scaling includes folded bias; spatial
and classifier scaling affects nominated weights only. Factors are 0–1.5.
Replacement temporal taps must stay within that filter's ORIGINAL min/max.
At most 25% of ORIGINAL inference parameters may differ cumulatively.

An accepted candidate must satisfy all four conditions:

- Absolute swapped accuracy increases relative to the current accepted model.
- Paired artifact flip rate decreases relative to that model.
- Clean accuracy stays within two percentage points of the ORIGINAL model.
- Worst accuracy across original/clean/swapped/uniform does not decrease from ORIGINAL.

Gap reduction alone cannot win. Each proposal is evaluated on a disposable copy;
rejected candidates cannot change the active checkpoint, verified by hashes.
Accepted checkpoints preserve cumulative provenance. Invalid operations are
recorded as rejected attempts. Invalid structured responses, provider errors,
budget exhaustion and source-integrity failures stop the run and export partial
evidence; they are not silently retried or converted into success.

**Comparators.** Every eligible model gets unchanged, fixed 4–40 Hz input filter,
seeded random temporal zeroing and a frequency/spatial rule comparator.
Random and rule selection use the same development acceptance test, five-attempt
ceiling and cumulative parameter cap. Rule scores use a fixed 0.25 threshold;
no rule candidate is a valid outcome. These controls are budget bounded, not
exactly norm-matched to every type of Astra operation. Input filtering is scored
unconditionally, so harm is visible. It may also remove useful task signals.
Optional IG gets development EEG, so it is a richer-information comparator.
Optional last-layer/full clean-label training is explicitly outside the 25%
weight-edit budget; it must not be advertised as a comparable tiny edit.

**Final scoring.** Every selected method for every eligible model is saved and
hashed before a durable final-start marker is created. Each is then scored once
on session 2. Final results do not choose a winner to send back for repair.
The report includes correct/total, accuracy, kappa, class accuracy, reliance gap,
flip rate and worst-condition accuracy. A/B, conditions and seeds remain paired.
Diagnosis Wilson intervals are labeled descriptive because models share subjects.
Repair uncertainty bootstraps subject means when at least two subjects exist;
one subject has no subject-generalization interval. Small subject counts remain
weak evidence even with many trials.

## Evidence and reproducibility

The ZIP includes a root summary, config, package versions, implementation source
snapshot/hashes, data-source/split hashes and trial IDs, blind packets, exact
structured requests/responses, usage/cost estimates, training history, small model
checkpoints, fold receipts, all evaluated candidate predictions and edit receipts,
selection lock, final per-trial predictions and plots. Raw EEG and API keys are
excluded. `evidence-index.json` hashes every other included file; the index does
not hash itself. Archives contain only the current run.

Package bounds are compatibility ranges, not a tested Colab lock. Pin the
recorded environment after the first successful real-data run. Source download,
Colab CUDA compatibility, real baseline accuracy and live Astra schema acceptance
remain external validation steps. Local checks use synthetic arrays and fake
provider responses; they establish implementation behavior, not experiment success.

Rebuild from `backend`:

```bash
python tools/build_eeg_notebook.py
python tools/package_eeg_colab.py
python -m unittest discover -s tests -p 'test_eeg*.py' -v
```

The legacy `package_colab.py` remains separate because its older notebook
bootstrap assumes flat module paths. Do not feed it the nested EEG package.

## Sources checked during implementation

- [EEGNet reference implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py)
- [MOABB BNCI dataset loader](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/bnci_2014.py)
- [MOABB BNCI MAT conversion](https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/bnci/base.py)
- [Competition IV 2a data description](https://www.bbci.de/competition/IV/desc_2a.pdf)
- [OpenAI structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)
- [Astra model documentation and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)

The user's EEG implementation PDF supplies the project direction. Corrections
above make leakage, preservation, information access and interpretation explicit.
