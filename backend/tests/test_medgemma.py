import contextlib
import unittest
from types import SimpleNamespace

from kirby.medgemma import MedGemma, ModelConfig


class Tokens(list):
    def tolist(self):
        return list(self)

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Tokens(result) if isinstance(key, slice) else result


class Inputs(dict):
    def to(self, device):
        return self


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return Inputs(input_ids=SimpleNamespace(shape=(1, 3)))

    def decode(self, tokens, **kwargs):
        self.decoded_tokens = tokens
        return "B"


class GenerationBoundaryTests(unittest.TestCase):
    def adapter(self, tokens):
        adapter = MedGemma(ModelConfig("a" * 40, max_new_tokens=2))
        adapter.processor = Processor()
        adapter.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
        adapter.model = SimpleNamespace(device="fake", generate=lambda **kwargs: [Tokens(tokens)])
        adapter.generation_config = object()
        adapter.eos_ids = [9]
        return adapter

    def test_only_new_tokens_are_decoded(self):
        adapter = self.adapter([100, 101, 102, 7, 9])
        generation = adapter.generate("Synthetic prompt")
        self.assertEqual(adapter.processor.decoded_tokens, [7, 9])
        self.assertEqual(generation.generated_tokens, 2)
        self.assertFalse(generation.truncated)

    def test_token_ceiling_without_eos_is_truncation(self):
        adapter = self.adapter([100, 101, 102, 7, 8])
        self.assertTrue(adapter.generate("Synthetic prompt").truncated)


if __name__ == "__main__":
    unittest.main()
