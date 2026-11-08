import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kirby.checkpoints import validate_snapshot
from kirby.cli import main
from kirby.jsl import (JSLMedPhi, JSLModelConfig, MODEL_ID, REFERENCE_REVISION,
                       StopAfterOptionLabel, check_hardware, check_loading_info)
from kirby.medgemma import MedGemma
from kirby.models import DEFAULT_MODEL, MEDGEMMA_ID, create_predictor
from kirby.records import write_json
from test_medgemma import Inputs, Tokens


class Tokenizer:
    def __init__(self, input_length=3):
        self.input_length = input_length

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.arguments = kwargs
        return Inputs(input_ids=SimpleNamespace(shape=(1, self.input_length)))

    def decode(self, tokens, **kwargs):
        self.decoded_tokens = tokens
        return "B"


class JSLTests(unittest.TestCase):
    def adapter(self, tokens, input_length=3):
        adapter = JSLMedPhi(JSLModelConfig(REFERENCE_REVISION, max_new_tokens=2))
        adapter.tokenizer = Tokenizer(input_length)
        adapter.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        adapter.model = SimpleNamespace(device="fake", generate=lambda **kwargs: [Tokens(tokens)])
        adapter.generation_config = object()
        adapter.stopping_criteria_type = list
        adapter.eos_ids = [9]
        adapter.context_length = 2048
        return adapter

    def test_default_is_jsl_and_legacy_model_is_explicit(self):
        self.assertEqual(DEFAULT_MODEL, MODEL_ID)
        adapter = create_predictor(DEFAULT_MODEL, REFERENCE_REVISION)
        self.assertIsInstance(adapter, JSLMedPhi)
        self.assertEqual(adapter.config.precision, "float32")
        self.assertIsInstance(create_predictor(MEDGEMMA_ID, "a" * 40), MedGemma)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            create_predictor(MEDGEMMA_ID, "a" * 40, precision="float32")
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            create_predictor("unreviewed/model", "a" * 40)

    def test_no_moving_revision_or_unapproved_precision(self):
        for settings in ({"revision": "main"}, {"precision": "float16"},
                         {"repository": "johnsnowlabs/arbitrary-model"},
                         {"max_new_tokens": 0}, {"seed": -1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                JSLModelConfig(**{"revision": REFERENCE_REVISION, **settings})

    def test_t4_requires_explicit_float32_bf16_requires_native_support(self):
        check_hardware(True, 7, True, "float32")
        # Some PyTorch versions report emulated BF16 as supported on T4.
        with self.assertRaisesRegex(RuntimeError, "float32"):
            check_hardware(True, 7, True, "bfloat16")
        check_hardware(True, 8, True, "bfloat16")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            check_hardware(False, 0, False, "float32")

    def test_rejects_incomplete_or_mismatched_weight_loads(self):
        check_loading_info({"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "error_msgs": []})
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "built-in Phi"):
                check_loading_info({field: ["some_weight"]})

    def test_text_template_and_generated_tokens_only(self):
        adapter = self.adapter([100, 101, 102, 7, 9])
        generation = adapter.generate("Synthetic prompt")
        self.assertEqual(adapter.tokenizer.messages, [{"role": "user", "content": "Synthetic prompt"}])
        self.assertEqual(adapter.tokenizer.decoded_tokens, [7, 9])
        self.assertTrue(adapter.tokenizer.arguments["add_generation_prompt"])
        self.assertFalse(generation.truncated)
        self.assertEqual(generation.generated_tokens, 2)

    def test_context_overflow_is_not_silently_truncated(self):
        adapter = self.adapter([], input_length=2047)
        with self.assertRaisesRegex(ValueError, "no input was truncated"):
            adapter.generate("Long question")

    def test_eos_and_length_limit(self):
        self.assertTrue(self.adapter([100, 101, 102, 7, 8]).generate("Prompt").truncated)
        self.assertFalse(self.adapter([100, 101, 102, 9]).generate("Prompt").truncated)

    def test_option_stop_requires_an_entire_generated_label(self):
        for text, expected in (("D.", True), (" C.\n", True), ("C", False),
                               ("", False), ("The answer is C.", False),
                               ("C. explanation", False), ("c.", False), ("E.", False)):
            with self.subTest(text=text):
                decoded = []
                def decode(tokens, **kwargs):
                    decoded.append(tokens)
                    return text
                stop = StopAfterOptionLabel(SimpleNamespace(decode=decode), 3)
                self.assertEqual(stop([Tokens([100, 101, 102, 7, 8])], None), expected)
                self.assertEqual(decoded, [[7, 8]])

    def test_label_stop_at_token_limit_is_complete_and_preserves_raw_text(self):
        adapter = self.adapter([100, 101, 102, 7, 8])
        adapter.tokenizer.decode = lambda tokens, **kwargs: "D."
        def generate(**kwargs):
            output = [Tokens([100, 101, 102, 7, 8])]
            self.assertTrue(kwargs["stopping_criteria"][0](output, None))
            return output
        adapter.model.generate = generate
        generation = adapter.generate("Synthetic prompt")
        self.assertEqual(generation.text, "D.")
        self.assertEqual(generation.generated_tokens, 2)
        self.assertFalse(generation.truncated)

    def test_phi_manifest_cannot_be_used_as_gemma(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "config.json", {"model_type": "phi"})
            (root / "model.safetensors").write_bytes(b"synthetic manifest fixture, not loadable weights")
            identity = validate_snapshot(root, expected_model_type="phi")
            self.assertIn("model.safetensors", identity["files"])
            with self.assertRaisesRegex(ValueError, "architecture"):
                validate_snapshot(root, expected_model_type="gemma3")

    def test_cli_passes_model_precision_and_revision_to_frozen_run(self):
        for precision in ("float32", "bfloat16"):
            with self.subTest(precision=precision), patch("kirby.cli.run_baseline") as run:
                run.return_value = {"directory": "synthetic", "report": {"state": "completed", "outcome": "Baseline evaluated"}}
                with contextlib.redirect_stdout(io.StringIO()):
                    code = main(["baseline", "--manifest", "synthetic.json", "--revision", REFERENCE_REVISION,
                                 "--precision", precision, "--allow-download"])
                self.assertEqual(code, 0)
                config = run.call_args.args[2]
                self.assertEqual(config["repository"], MODEL_ID)
                self.assertEqual(config["revision"], REFERENCE_REVISION)
                self.assertEqual(config["precision"], precision)
                self.assertFalse(config["local_files_only"])

    def test_notebook_defaults_match_adapter_and_allow_anonymous_access(self):
        directory = Path(__file__).parents[1] / "notebooks"
        for path in directory.glob("*.ipynb"):
            candidate = json.loads(path.read_text(encoding="utf-8"))
            for cell in candidate["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), path.name + ":" + cell["id"], "exec")
        notebook = json.loads((directory / "kirby_colab.ipynb").read_text(encoding="utf-8"))
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        self.assertIn(MODEL_ID, cells["gpu"])
        self.assertIn(REFERENCE_REVISION, cells["gpu"])
        self.assertIn("USE_HF_TOKEN = False", cells["access"])
        self.assertIn("'--precision', PRECISION", cells["run"])
        self.assertNotIn("google/medgemma", "".join(cells.values()))


if __name__ == "__main__":
    unittest.main()
