"""T2: a reversible training-error break. Adds a learned-shortcut distortion to one MLP layer.

Simulates a faulty training run that taught the model an answer-position shortcut: the
model drifts toward choosing option B regardless of content. The distortion is added, not
destructive, so in principle it can be found and removed without the original weights.

W' = W + alpha * u v^T on one mlp.down_proj, where
- v is the typical input direction to that layer on calibration questions (so it's always active);
- u is the output direction that favors the letter B over A, C and D.
Calibration uses MedMCQA training questions, never the dev or sealed sets.
The evaluator-only manifest records the layer, alpha, u and v.

Run in WSL from backend/:  python -m breakers.make_training_error
"""
import hashlib
import json

import numpy as np
import torch
from safetensors.torch import save_file

from breakers.make_broken import ROOT, without_shared_storage
from data.load_medmcqa import DATASET, normalize_text, to_question
from eval.check_breaker import load_questions
from eval.scoring import score_questions
from repair.qwen_backend import QwenBackend

CASE = "t2-a"
LAYER = 14
TENSOR = f"model.language_model.layers.{LAYER}.mlp.down_proj.weight"
TARGET_LETTER = "B"
ALPHAS = [0.005, 0.01, 0.02, 0.04, 0.08, 0.16]
TARGET_SHARE = 0.6  # A clear but not total shortcut: at least 60% of answers become B.
CALIBRATION_SIZE = 64


def calibration_questions(exclude: set[str]) -> list:
    from datasets import load_dataset

    rows = load_dataset(DATASET, split="train")
    picked = []
    for i in range(0, len(rows), 97):
        item = to_question(rows[i], i)
        if item and hashlib.sha256(normalize_text(item.question).encode()).hexdigest() not in exclude:
            picked.append(item)
        if len(picked) == CALIBRATION_SIZE:
            return picked
    raise ValueError("Not enough calibration questions")


def main() -> None:
    held_out = load_questions(__import__("pathlib").Path("artifacts/medmcqa/dev.jsonl"), 10_000)
    held_out += load_questions(__import__("pathlib").Path("artifacts/medmcqa/sealed.jsonl"), 10_000)
    exclude = {hashlib.sha256(normalize_text(q.question).encode()).hexdigest() for q in held_out}
    calibration = calibration_questions(exclude)

    backend = QwenBackend({"calibration": calibration})
    clean = backend.read_weights(ROOT / "models" / "c0" / "model.safetensors")
    backend.load(clean)

    # v: average input to the layer's down_proj at the answer position.
    captured = []
    layer = backend.model.model.language_model.layers[LAYER].mlp.down_proj
    handle = layer.register_forward_pre_hook(lambda module, args: captured.append(args[0][:, -1, :].float().cpu()))
    clean_rows = score_questions(backend.model, backend.tokenizer, calibration, batch_size=16)
    handle.remove()
    mean_input = torch.cat(captured).mean(0)
    v = mean_input / mean_input.norm()

    # u: the direction in the residual stream that raises B's logit relative to A, C and D.
    head = backend.params["lm_head.weight"].float().cpu()
    ids = {letter: backend.tokenizer.encode(letter, add_special_tokens=False)[0] for letter in "ABCD"}
    others = torch.stack([head[ids[x]] for x in "ACD"]).mean(0)
    u = head[ids[TARGET_LETTER]] - others
    u = u / u.norm()

    base = clean[TENSOR].float()
    clean_accuracy = float(np.mean([r["correct"] for r in clean_rows]))
    clean_share = float(np.mean([r["prediction"] == TARGET_LETTER for r in clean_rows]))
    print(f"Clean calibration accuracy: {clean_accuracy:.2f}, answers {TARGET_LETTER} {clean_share:.0%}", flush=True)
    chosen = None
    for alpha in ALPHAS:
        state = dict(clean)
        state[TENSOR] = (base + alpha * torch.outer(u, v)).to(clean[TENSOR].dtype)
        rows = backend.score_items(state, calibration)
        accuracy = float(np.mean([r["correct"] for r in rows]))
        share = float(np.mean([r["prediction"] == TARGET_LETTER for r in rows]))
        print(f"alpha {alpha:6.3f}: accuracy {accuracy:.2f}, answers {TARGET_LETTER} {share:.0%}", flush=True)
        if share >= TARGET_SHARE:
            chosen = (alpha, state)
            break
    if chosen is None:
        raise SystemExit("No alpha reached the target share; widen ALPHAS")

    alpha, state = chosen
    out = ROOT / "models" / CASE
    out.mkdir(parents=True, exist_ok=True)
    save_file(without_shared_storage(state), str(out / "model.safetensors"))
    evaluator = ROOT / "evaluator"
    np.save(evaluator / f"{CASE}_u.npy", u.numpy())
    np.save(evaluator / f"{CASE}_v.npy", v.numpy())
    manifest = {
        "case": CASE,
        "breaker": "rank1_shortcut",
        "damaged_tensors": [TENSOR],
        "alpha": alpha,
        "target_letter": TARGET_LETTER,
        "u_file": f"{CASE}_u.npy",
        "v_file": f"{CASE}_v.npy",
    }
    (evaluator / f"{CASE}.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved {CASE} with alpha {alpha}")


if __name__ == "__main__":
    main()
