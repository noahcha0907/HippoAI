# Kirby EEG roadmap

The pivot asks a measurable question: can Astra identify a planted shortcut from
a small EEG model's weights, then repair it better than simple controls while
preserving task performance? Diagnosis from weights and repair with development
feedback are separate claims. Neither is a clinical validation claim.

## Current next step: frozen replication on subjects 2 and 3

Run6 is complete: 30 Astra candidates, six development-accepted edits, five of
six edited comparisons meeting final criteria. All 12 blind diagnoses abstained.
The result is exploratory, on one previously inspected subject. The frontend
replays the real report, including regressions and controls.

Use `notebooks/kirby_eeg_replication_colab.ipynb` and the updated
`dist/kirby_eeg_colab.zip`. The predeclared plan is
`experiments/eeg_replication_v1.json`: subjects 2 and 3, seeds 0/1/2,
clean/blink, both A/B, identical run6 procedure and per-model limits. Source
hashes and settings are checked before the worker starts. Budgets scale to
144 requests and $100 estimated token cost for 12 models. Failed gates stay
excluded; do not weaken them to obtain a result. No paid run is launched locally.

Report A and B separately, seed-averaged effects for each subject and the
equal-weight mean across subjects. Include clean preservation, flip-rate change,
worst-condition preservation, all controls, exclusions and failures. Retained
originals have zero gain; missing final outcomes stay missing. Do not choose
the winning method using final scores. Two new subjects still provide limited
generalization evidence. Software/provider versions are recorded, not pinned
to exactly match run6. Keep both old and new evidence archives.

## 1. Working research pipeline — implemented and exercised on Colab

Delivered: explicit BNCI loader and train-fitted preprocessing; EEGNet training;
clean/blink/hum/muscle conditions; matched counterfactual variants; lossless folded
weight export; blind A/B diagnosis; bounded reversible edits; random/rule/filter
controls; optional IG/retraining comparators; final-session lock; evidence ZIP,
plots, tests and a separate Colab notebook/source bundle.

Next action: run the default Colab experiment on subject 1 with three seeds and
clean/blink models. Preserve the full evidence ZIP, including failed gates.
Confirm real dataset shapes, baseline training, CUDA behavior, folding equivalence,
API compatibility, runtime and spend. Freeze working package versions afterward.

**Gate:** clean development accuracy at least 0.35 and planted model
original-minus-swapped development accuracy at least 0.15. If these fail, first
inspect training and artifact strength using development data. Register changes
as a new protocol/run; do not treat repeated favorable selection as confirmation.

## 2. Decide whether the Astra hypothesis deserves expansion

Inspect first-response diagnosis accuracy, abstention coverage, false positives
on clean models, band/electrode localization, and A versus B. Then inspect
absolute swapped gain, clean preservation and flip reduction for accepted edits.
Compare locked final results against unchanged, random, rule and input-filter
baselines. Always include unsuccessful models and zero-accepted-edit runs.

**Decision:** expand only if effects repeat across seeds and Astra adds something
beyond the simple controls. No numerical improvement on one model, or a smaller
gap caused by worse original accuracy, justifies a success claim. A useful B-only
result is evidence for signal views helping the audit, not raw-weight understanding.
Strong diagnosis with no repair is still a distinct, limited result.

## 3. Broaden and challenge the result

Use fresh subjects and predeclared conditions: add hum and muscle, then all nine
subjects with three seeds. Enable IG and clean-label retraining comparators and
report their greater information/resource access. Consider operation/norm-matched
random controls if Astra's preferred operations differ materially from temporal
zeroing. Lock budgets and analysis before scoring final sessions.

Report subject-level effects and uncertainty; do not multiply the sample size by
counting A/B, artifact versions or seeds as independent subjects. Treat subjects
already used for debugging as development evidence in subsequent broad claims.
Confirm any tuned protocol on genuinely untouched subjects/data.

## 4. Demonstrate and decide

Build a small demonstration around evidence already produced: temporal responses,
electrode weights, Astra's falsifiable first diagnosis, exact edit, rejected attempts,
before/after counterfactual predictions, final metrics and cost. The current request
does not modify the frontend; a product/demo UI is a later step.

If Astra cannot beat transparent signal rules, or edits erase useful signal, stop
expanding that claim. Keep the reproducible audit benchmark and report the negative
result. Consider richer task evidence only as a newly scoped experiment, not a
retroactive explanation of a failed weights-only test.

There is an exploratory EEG repair signal from run6. The new-subject replication
tests whether it repeats beyond the subject used during development.
