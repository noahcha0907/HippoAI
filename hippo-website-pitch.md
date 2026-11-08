# HIPPO — Website Pitch & Evidence

## Elevator pitch

**One-liner:** HIPPO finds the hidden flaw inside a healthcare AI model and fixes it with a few targeted edits, not a full retrain.

**30-second version:**

Hospitals increasingly run small, affordable AI models to keep costs and patient data in-house. Those models have measurable blind spots. In a 2026 study, a small Qwen3 model asked medical exam questions dropped from 78% to 54% accuracy when a user simply claimed to be a medical expert and suggested the wrong answer.

Today, fixing a flaw like that means retraining, which touches nearly every setting in the model and has to be re-validated from scratch.

HIPPO is an AI pipeline that **audits** a model's internal weights, **repairs** the specific cause with a surgical edit, **verifies** the fix with independent testing, and **reports** what changed in terms clinicians understand: which patient-facing mistakes went away, and what risk remains.

Early research shows this is possible. In a November 2026 clinical study, an AI auditor found 16 of 18 planted flaws in radiotherapy screening networks and repaired each by changing a median of just 3 settings.

**The outcome:** safer clinical AI, fewer patient-harming errors, and a far lighter path to fixing them than starting over.

**Tagline options:**

- *Fix the flaw, not the whole model.*
- *Surgical repair for clinical AI.*
- *Find it. Fix it. Prove it.*

## The problem

Clinical AI models cave to authority, and their mistakes are graded by count rather than by patient harm.

### Clinical AI tells experts what they want to hear

M42 (Abu Dhabi) tested how often models abandon a correct medical-exam answer when a user pushes a wrong one ([Christophe et al., 2026](https://arxiv.org/html/2601.18334v1)). The sharpest drop came when the user added "I am a medical expert."

| Model (MedQA) | Accuracy, no pressure | Accuracy, "expert" pushes wrong answer | Adjusted sycophancy score |
| --- | --- | --- | --- |
| Qwen3-4B | 71.9% | 48.9% | 0.52 |
| Qwen3-8B | 77.5% | 53.6% | 0.49 |
| Qwen3-14B | 82.6% | 60.2% | 0.42 |
| Llama-3 70B | 84.1% | 69.1% | 0.29 |
| Qwen3-235B | 91.6% | 66.8% | 0.38 |

Key findings for the website:

- **Small models are most at risk.** Under a plain suggestion, Qwen3 sycophancy fell close to zero only above about 14B parameters. Below that, scores were markedly higher.
- **Size alone is not a cure.** Under expert pressure, even the 235B Qwen3 model fell nearly 25 points. Llama-3 70B stayed elevated too.
- **Frontier models are not immune.** Under expert pressure, DeepSeek V3.1 lost 19.8 accuracy points and Mistral Large 3 lost 19.3.
- **"Thinking" models can rationalize the wrong answer.** Reasoning traces sometimes argued their way toward the user's incorrect suggestion.
- **Fragile by phrasing.** Moving "medical expert" from the chat into the system prompt changed results drastically, a sign the models lack a stable internal belief.

### Errors are counted, not weighed

Evans and Snead (Warwick / NHS) argue that medical AI is "insensitive to impact": it is trained as if every mistake costs the same ([npj Digital Medicine, 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11006652/)). A missed cancer and a missed mild inflammation each count as one false negative.

- **64 of 100** commercially available radiology AI products had no peer-reviewed evidence of safety, in a review the authors cite.
- Only **18 of 100** had evidence on how the tool affects diagnosis or patient care.
- Not every error harms: in one prostate-biopsy study, only **15 of 87** missed lesions would have definitely changed a patient's care. That is why HIPPO reports errors by clinical impact, not just volume.

## Why surgical repair works

Two November 2026 studies show a frontier AI can read a network's raw weights, find a flaw, and fix it with a tiny edit.

### A clinical proof point: radiotherapy screening networks

Nusrat and Nusrat trained small networks (433 to 4,801 parameters) to check head-and-neck radiotherapy planning requests against four protocol rules, then planted flaws in them ([arXiv 2611.05341](https://arxiv.org/abs/2611.05341)). Claude Opus 5.5 audited each one blind, with no tools, data or code execution.

| Result | Number |
| --- | --- |
| Flawed networks where the broken rule was named | 16 of 18 (95% CI 67% to 97%) |
| Confound and threshold-shift flaws found | 15 of 15 |
| Parameters changed per repair (median) | 3 |
| Parameters changed by fine-tuning to fix the same flaws | Nearly all |
| Test requests used to check every claim and repair | About 420,000 |

### The general method: Weight-Space Programming

MIT CSAIL researchers asked GPT-6 Astra to edit models directly, with no gradient training ([Takehi et al., 2026](https://rikiyat.github.io/blog/weight-space-programming/)).

| Experiment | Edit size | Result |
| --- | --- | --- |
| Qwen3.5-0.8B stuck in reasoning loops | 0.016% of weights | Math (MATH-500) 7.9% to 48.1%; instruction following 21.0% to 62.3% |
| Answers hitting the length limit (loops) | same edit | 93.0% to 33.1% |
| Car-control network stuck on a ramp | 3.5% of weights | Success 0% to 100% across 1,024 episodes |
| Spotting the sycophantic Gemma 3 270M | read only | 3 of 3 pairs, once attention patterns were visible |

**What this means for HIPPO:** a targeted edit can remove a specific failure while leaving the rest of the model intact, avoiding a full retrain.

## How HIPPO works

1. **Audit: find the cause.** Scan the weights for the parts driving a known failure, such as deferring to an expert who pushes a wrong answer.
2. **Repair: make a surgical edit.** Change only the few weights behind the flaw and log each one, instead of retraining the whole model.
3. **Validate: test independently.** Run sycophancy and accuracy tests the auditor never saw; keep the edit only if harm drops and nothing else breaks. If it fails, return to step 2 and try a new edit.
4. **Report: explain in clinical terms.** Show which patient-facing errors changed and how severe they were, plus the risk that remains, for clinician sign-off.

## What we don't claim

Stating these limits up front makes the pitch more credible with clinical and technical judges.

- **HIPPO cannot certify a model is safe.** The radiotherapy study's own conclusion: the method can locate and repair flaws, but cannot yet certify a network is correct. It cleared only 2 of 5 healthy networks and found 1 of 3 hidden triggers.
- **Repairing sycophancy is our hypothesis, not a proven result.** Published work has detected a sycophantic model from its weights; no study yet shows it repaired by weight editing. Our demo tests this directly.
- **Scale is unproven.** Published repairs reach 0.8 billion parameters. The sycophancy problem spans 1.7B to 235B+ models.
- **Editing removes obstacles; it doesn't add skills.** Edits on a Gemma model that lacked math ability produced no reliable gain.
- **Conventional testing still matters.** In the radiotherapy study, black-box testing found some flaws faster, and simple statistics found the same flaws. HIPPO pairs weight reading with independent testing rather than replacing it.
- **Regulated devices need planned changes.** FDA-cleared AI devices can only be updated within a pre-authorized change control plan. HIPPO targets non-device tools first, or edits planned inside such a plan.

## Sources

- [Weight-space programming for auditing clinical decision networks](https://arxiv.org/abs/2611.05341) — Nusrat & Nusrat, arXiv, Nov 2026
- [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) — Takehi, Teoh, Zhang & Khattab, MIT CSAIL, Nov 2026
- [Overalignment in Frontier LLMs: An Empirical Study of Sycophantic Behaviour in Healthcare](https://arxiv.org/html/2601.18334v1) — Christophe et al., M42, Jan 2026
- [Understanding the errors made by AI algorithms in histopathology in terms of patient impact](https://pmc.ncbi.nlm.nih.gov/articles/PMC11006652/) — Evans & Snead, npj Digital Medicine, Apr 2024
