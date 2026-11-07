"""Runs the real broken model on the GPU for probes and validation.

Only tensors that differ from what's already loaded get copied, so trying an edit is cheap.
This process holds dev questions only. Sealed questions belong to the evaluator.
"""
import threading
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoTokenizer

from eval.scoring import Question, score_questions
from repair.edits import EditSession

MODEL = "Qwen/Qwen3.5-0.8B"
MAX_EXAMPLES = 8
MAX_TEXT_TOKENS = 128
TOP_KEYS = 5


class QwenBackend:
    def __init__(self, dev_sets: dict[str, list[Question]], batch_size: int = 16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16).to(self.device).eval()
        self.params = dict(self.model.state_dict())
        self.dev_sets = dev_sets
        self.batch_size = batch_size
        self._loaded: dict[str, torch.Tensor] = {}
        self.lock = threading.RLock()  # One GPU model shared by the repair, the evaluator and /generate.
        self._base_cache: dict[tuple[int, str], list[dict]] = {}
        config = getattr(self.model.config, "text_config", self.model.config)
        types = list(getattr(config, "layer_types", []) or [])
        self.attention_layers = [i for i, t in enumerate(types) if t == "full_attention"]

    def read_weights(self, path: Path) -> dict[str, torch.Tensor]:
        state = load_file(str(path))
        unknown = set(state) - set(self.params)
        if unknown:
            raise KeyError(f"Weights don't match {MODEL}: {sorted(unknown)[:3]}")
        return state

    def load(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, tensor in state.items():
                if self._loaded.get(name) is not tensor:
                    self.params[name].copy_(tensor.to(self.device, self.params[name].dtype))
                    self._loaded[name] = tensor

    def results(self, state: dict[str, torch.Tensor], suite: str) -> list[dict]:
        return self.score_items(state, self.dev_sets[suite])

    def score_items(self, state: dict[str, torch.Tensor], items: list[Question]) -> list[dict]:
        """Score any question list. The evaluator uses this for sealed questions; tools never do."""
        with self.lock:
            self.load(state)
            return score_questions(self.model, self.tokenizer, items, batch_size=self.batch_size)

    def generate(self, state: dict[str, torch.Tensor], prompt: str, max_new_tokens: int = 200) -> str:
        """Greedy answer with thinking off, for side-by-side demo answers."""
        with self.lock:
            self.load(state)
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            encoded = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            return self.tokenizer.decode(output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    def score(self, state: dict[str, torch.Tensor], suite: str) -> np.ndarray:
        """Per-question correctness, as DevValidator expects."""
        return np.array([r["correct"] for r in self.results(state, suite)], dtype=bool)

    def forward_pass(self, session: EditSession, suite: str, category: str | None = None) -> dict:
        """Compare the broken model with the current edits, overall and by category."""
        if suite not in self.dev_sets:
            raise ValueError(f"Unknown suite {suite!r}; choose from {sorted(self.dev_sets)}")
        items = self.dev_sets[suite]
        keep = [i for i, q in enumerate(items) if category in (None, "", q.category)]
        if not keep:
            raise ValueError(f"No questions in category {category!r}")
        key = (id(session.base), suite)
        if key not in self._base_cache:
            self._base_cache[key] = self.results(session.base, suite)
        broken = self._base_cache[key]
        edited = self.results(session.state(), suite)

        def summary(rows: list[dict]) -> dict:
            picked = [rows[i] for i in keep]
            letters = {letter: sum(r["prediction"] == letter for r in picked) for letter in "ABCD"}
            return {"accuracy": round(float(np.mean([r["correct"] for r in picked])), 3), "answer_letters": letters}

        by_category = {}
        for name in sorted({items[i].category for i in keep}):
            idx = [i for i in keep if items[i].category == name]
            by_category[name] = {
                "n": len(idx),
                "unedited": round(float(np.mean([broken[i]["correct"] for i in idx])), 3),
                "edited": round(float(np.mean([edited[i]["correct"] for i in idx])), 3),
            }
        failures = [
            {
                "question": items[i].question,
                "options": items[i].options,
                "expected": items[i].answer,
                "edited_answer": edited[i]["prediction"],
            }
            for i in keep
            if not edited[i]["correct"]
        ][:MAX_EXAMPLES]
        return {
            "questions": len(keep),
            "unedited": summary(broken),
            "edited": summary(edited),
            "by_category": by_category,
            "example_failures_after_edits": failures,
        }

    def attention(self, session: EditSession, texts: list[str]) -> dict:
        """Where the final token attends, per attention layer and head. Weights only, no hidden states."""
        with self.lock:
            return self._attention(session, texts)

    def _attention(self, session: EditSession, texts: list[str]) -> dict:
        self.load(session.state())
        self.model.set_attn_implementation("eager")
        try:
            output = []
            for text in texts:
                encoded = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TEXT_TOKENS)
                encoded = encoded.to(self.device)
                with torch.inference_mode():
                    attentions = self.model(**encoded, output_attentions=True).attentions
                tokens = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
                present = [w for w in (attentions or []) if w is not None]
                # Some versions return only the attention layers; map them back to real layer numbers.
                numbers = (
                    self.attention_layers
                    if len(present) == len(self.attention_layers) and len(present) != len(attentions or [])
                    or len(attentions or []) == len(self.attention_layers)
                    else [i for i, w in enumerate(attentions or []) if w is not None]
                )
                layers = {}
                for layer, weights in zip(numbers, present):
                    last = weights[0, :, -1, :].float()
                    top = last.topk(min(TOP_KEYS, last.shape[-1]), dim=-1)
                    layers[str(layer)] = [
                        [[tokens[k], round(float(v), 3)] for v, k in zip(values, keys.tolist())]
                        for values, keys in zip(top.values.tolist(), top.indices)
                    ]
                output.append({"tokens": tokens, "last_token_top_keys_by_layer_and_head": layers})
            return {"texts": output}
        finally:
            self.model.set_attn_implementation("sdpa")
