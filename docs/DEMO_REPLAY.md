# Presenting the EEG experiment on demo.html

The demo automatically opens the real EEG run6 replay from the site's static
assets. It does not train a model, call Astra or upload local files. The clinical
language-model replay pulled from main is preserved below it as a separate
experiment. Synthetic records exist only inside browser tests.

The default view is a short explanation for judges: the question, a compact
result card, and problem/edit/check steps. The card derives its counts and A/B
mean swapped-accuracy gains from the loaded report, including the unsuccessful
comparison. The one-subject/exploratory limitation stays visible. **See the edits**
and **Compare results** open the relevant stage. **Explore the full experiment**
contains all imports, selectors, controls and receipts; **Watch the clinical-model
demo** opens the separate recorded demonstration. Both sections start collapsed.

## Included EEG result

Run `b6e81501-ff50-4322-a2e0-cc514ba1d475` completed on 2026-11-08: one subject,
three training seeds, matched clean/blink models, 48 requests and approximately
$20.07 estimated token cost. All 12 blind diagnoses abstained. Development-guided
repair tested 30 Astra candidates and accepted six. Five of six edited
comparisons also meet all four criteria on the final session. Astra A's swapped
accuracy improves for all three seeds; Astra B worsens seed 2. Random pruning's
seed-1 clean accuracy drops 3.125 points, outside the two-point preservation bound.
These paired comparisons are not independent subject replications, and subject
1's final session had been inspected before this exploratory follow-up.

`assets/evidence/run6/replay.json` contains the exact report values and all 24
receipt files (including empty histories), with provenance in the adjacent JSON.
The packager verified all 570 original indexed file hashes. No checkpoints,
raw EEG, provider packets, API credentials or local paths are bundled.
The page validates report/receipt structure on load; it does not claim a new
browser-side hash check of the entire original archive. To rebuild:

```powershell
python frontend/package_eeg_replay.py PATH_TO_EXTRACTED_EVIDENCE assets/evidence/run6
```

The overview includes every artifact-trained Astra comparison. Default selection
is the first blink model in recorded order, Astra A, not the highest-scoring edit.
Clear returns to the import state; **Load run6 results** restores the included run.

## Open the page

From the repository root, use any static development server, for example:

```powershell
python -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/demo.html`. Automatic loading needs HTTP(S); opening
`demo.html` directly still supports manual report loading. Use localhost or HTTPS
for folder hash checks. The site's existing
static hosting can serve the same files without an API server or build process.
This change has not been published or pushed automatically.

## Bring a real experiment

1. Complete the [Colab workflow](../backend/docs/EEG_EXPERIMENT.md), download its
   `kirby-eeg-evidence-<run-id>.zip`, and extract it.
2. Expand **Explore the full experiment**, choose **Open a saved report** and select that folder's
   `report.json`. This loads diagnoses, gates, model/method selectors and final
   comparisons. Individual edit receipts are not embedded in report.json.
3. For the full edit history, use **Choose evidence folder** and select the
   extracted run folder itself. Modern desktop Chrome/Edge support this folder
   picker. Report-only loading remains available when folder selection is absent.
4. When the folder contains `evidence-index.json`, the page checks SHA-256 for
   the report and every edit-receipt file it displays. A mismatch blocks import.
   This checks consistency with the supplied index, not independent authenticity
   or every file in the original experiment archive.
5. Choose a model and a recorded method. Astra A is selected first when present,
   followed by B and the controls; selection does not rank methods by test score.
6. Walk through **Audit → Repair → Validate → Report**, or choose **Play replay**
   for a seven-second interval between stages. Playback pauses when the tab is
   hidden or a model, method, or stage is selected. Arrow keys, Home and End
   navigate the stage buttons. Playback never starts automatically.
7. Choose **Save replay** to download one JSON containing the loaded report and
   available receipts. Reopen that file on presentation day. It is labeled as
   having no current index check; previous verification is not silently inherited.

Imported files stay in browser memory. Clearing removes the loaded run;
refreshing restores the bundled run6. JSON is rendered as text, so embedded markup cannot execute.
No API key is needed for this replay.

## What to explain during the presentation

- **Audit:** first-response diagnosis, evidence available to that method, planted
  condition revealed for the audience, and development eligibility. Missing or
  abstained diagnoses stay visible.
- **Repair:** attempted and accepted edits, cumulative accepted parameter share
  when available, receipt operations and source-integrity checks. Retraining is
  labeled as outside the bounded edit budget. Report-only views explicitly ask
  for the folder when receipt details are missing.
- **Validate:** unchanged versus selected-method final results across original,
  clean, swapped and uniform artifact conditions. Displayed deltas are percentage
  points, not relative percentages. Flip rate compares paired no-artifact and
  all-artifact predictions. Missing final results never fall back to development.
- **Report:** all methods for the selected model, counts, run identity, status,
  eligibility, API request attempts, estimated token cost and evidence details.
  A positive comparison is not a statistical or clinical validation claim.

Completed-with-exclusions, failed, interrupted, running and gates-failed reports
are supported. Excluded models remain selectable. A run with no completed models
shows its status without manufacturing a model or comparison. Older medical-model
and pruning report formats are rejected with an explanation.

## Implementation and checks

`assets/demo-data.js` validates the EEG protocol, numeric bounds and counts,
loads reports/receipts and performs optional index checks. `assets/demo.js`
renders the replay with text-only DOM updates. `assets/demo.css` is scoped to the
demo; the existing shared website styles, header, logo and navigation are reused.
The former illustrative walkthrough script remains available for any old consumers
but is no longer loaded by demo.html.

The browser tests use temporary synthetic reports and the installed Chrome/Edge
browser. Install Python Playwright (`python -m pip install playwright`) and run:

```powershell
python -m unittest discover -s frontend/tests -v
```

Set `HIPPO_CHROME_PATH` if Chrome is installed elsewhere, or install Playwright's
Chromium with `python -m playwright install chromium`. Optional screenshots go to
the directory specified by `HIPPO_DEMO_SCREENSHOTS`; `.demo-qa/` is ignored.
Tests cover report/folder imports, integrity failures, replay export/import,
partial/excluded/empty runs, unsafe text, numeric inconsistencies, keyboard and
playback controls, no network upload, and responsive layouts.
