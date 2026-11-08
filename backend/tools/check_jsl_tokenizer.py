"""Check the actual public tokenizer/config without downloading model weights.

Run with Transformers installed. Anonymous Hub access is intentional: this
checks the same ungated path a new Colab user will use. No inference is claimed.
"""

import json
from array import array
from importlib.metadata import version
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kirby.datasets import QuestionInput
from kirby.evaluation import format_prompt
from kirby.jsl import MODEL_ID, REFERENCE_REVISION, StopAfterOptionLabel


def main():
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, PhiConfig

    cache = Path(__file__).resolve().parents[1] / ".hf-cache"
    snapshot = snapshot_download(
        MODEL_ID, revision=REFERENCE_REVISION, token=False, local_dir=cache / REFERENCE_REVISION,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
                        "merges.txt", "special_tokens_map.json", "added_tokens.json"],
    )
    config = PhiConfig.from_pretrained(snapshot, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
    prompt = format_prompt(QuestionInput("Select the second entry.", ("First", "Second", "Third", "Fourth")))
    messages = [{"role": "user", "content": prompt}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    assert config.model_type == "phi" and config.max_position_embeddings == 2048
    assert "[INST]" in rendered and prompt in rendered
    assert len(token_ids) + 8 <= config.max_position_embeddings
    assert all(0 <= token < config.vocab_size for token in token_ids)
    # Replay reported output strings through the real tokenizer, without any
    # model inference, to check that label stopping catches their prefixes.
    label_checks = []
    for text in ("D. Inherited mutation in clotting", "D. Pyridoxine.",
                 "C.", "D. Non-enveloped (+"):
        generated = tokenizer.encode(text, add_special_tokens=False)
        stop = StopAfterOptionLabel(tokenizer, len(token_ids))
        for end in range(1, len(generated) + 1):
            if stop([array("q", token_ids + generated[:end])], None):
                label = tokenizer.decode(generated[:end], skip_special_tokens=True)
                assert label == text[:2]
                label_checks.append({"label": label, "tokens": end})
                break
        else:
            raise AssertionError("Tokenizer did not expose a complete option label")
    print(json.dumps({"repository": MODEL_ID, "revision": REFERENCE_REVISION,
                      "tokenizer_class": type(tokenizer).__name__, "model_type": config.model_type,
                      "context_length": config.max_position_embeddings, "prompt_tokens": len(token_ids),
                      "transformers_version": version("transformers"), "jinja2_version": version("jinja2"),
                      "authenticated": False, "weights_downloaded": False, "inference_tested": False}, indent=2))
    print(json.dumps({"replayed_label_stops": label_checks}, indent=2))


if __name__ == "__main__":
    main()
