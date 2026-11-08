"""Real tensor/architecture tests; skipped in the stdlib-only install."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from transformers import PhiConfig, PhiForCausalLM
except ImportError:
    torch = None

from kirby.evaluation import Generation
from kirby.patches import ColumnScale, UnitAblation, PhiColumnEditor, tensor_hash, weights_hash
from kirby.rollback import run_rollback_check
from test_evaluation import make_manifest


@unittest.skipIf(torch is None, "Install the GPU extra for real PyTorch/Phi tests (CPU supported)")
class PatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = PhiForCausalLM(PhiConfig(vocab_size=32, hidden_size=16, intermediate_size=256,
                                             num_hidden_layers=2, num_attention_heads=4,
                                             num_key_value_heads=4, max_position_embeddings=64,
                                             partial_rotary_factor=0.5)).eval()
        self.base = "a" * 64
        self.editor = PhiColumnEditor(self.model, self.base)

    def test_real_phi_forward_and_weights_restore_exactly(self):
        inputs = torch.tensor([[1, 2, 3, 4]])
        before_hash = weights_hash(self.model)
        with torch.inference_mode():
            before = self.model(inputs, use_cache=False).logits.clone()
        receipt = self.editor.apply(ColumnScale(self.base, "request1"))
        with torch.inference_mode():
            edited = self.model(inputs, use_cache=False).logits.clone()
        self.assertFalse(torch.equal(before, edited))
        self.assertGreater(receipt["changed_parameters"], 0)
        self.assertLessEqual(receipt["relative_tensor_change"], 0.01)
        self.assertNotEqual(before_hash, weights_hash(self.model))
        restored = self.editor.rollback()
        with torch.inference_mode():
            after = self.model(inputs, use_cache=False).logits
        self.assertEqual(before_hash, weights_hash(self.model))
        self.assertTrue(torch.equal(before, after))
        self.assertTrue(restored["rollback_verified"])
        self.assertEqual(self.editor.rollback(), restored)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.editor.apply(ColumnScale(self.base, "request1"))

    def test_invalid_requests_never_modify_weights(self):
        before = weights_hash(self.model)
        cases = [dict(base_checkpoint_id="b" * 64), dict(layer=2), dict(columns=(256,)),
                 dict(columns=(1, 1)), dict(columns=(True,)), dict(scale=1.0),
                 dict(scale=float("nan")), dict(scale=1.2)]
        for number, values in enumerate(cases):
            with self.subTest(values=values), self.assertRaises(ValueError):
                spec = {"base_checkpoint_id": self.base, "request_id": f"invalid-{number}", **values}
                self.editor.apply(ColumnScale(**spec))
            self.assertEqual(before, weights_hash(self.model))

    def test_actual_unit_zeroing_changes_phi_logits_then_restores_exactly(self):
        weight = self.model.model.layers[0].mlp.fc2.weight
        # Small synthetic Phi has only 256 units; keep this column within 5%.
        with torch.no_grad():
            weight[:, 0].mul_(0.25)
        inputs = torch.tensor([[1, 2, 3, 4]])
        original = weights_hash(self.model)
        with torch.inference_mode():
            before = self.model(inputs, use_cache=False).logits.clone()
        receipt = self.editor.apply(UnitAblation(self.base, 'ablate', 0, (0,)))
        self.assertTrue(bool((weight[:, 0] == 0).all()))
        self.assertEqual(receipt['operation'], 'ablate_mlp_unit')
        self.assertEqual(receipt['changed_parameters'], 16)
        with torch.inference_mode():
            changed = self.model(inputs, use_cache=False).logits.clone()
        self.assertFalse(torch.equal(before, changed))
        self.editor.rollback()
        with torch.inference_mode():
            after = self.model(inputs, use_cache=False).logits
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(weights_hash(self.model), original)

    def test_norm_limit_and_nonfinite_original_rejected(self):
        weight = self.model.model.layers[0].mlp.fc2.weight
        with torch.no_grad():
            weight.zero_()
            weight[:, 0] = 1
        before = tensor_hash(weight)
        with self.assertRaisesRegex(ValueError, "1%"):
            self.editor.apply(ColumnScale(self.base, "norm"))
        self.assertEqual(tensor_hash(weight), before)
        with torch.no_grad():
            weight[0, 1] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.editor.apply(ColumnScale(self.base, "nan"))

    def test_bfloat16_rounding_no_op_is_rejected(self):
        self.model.to(torch.bfloat16)
        before = weights_hash(self.model)
        with self.assertRaisesRegex(ValueError, "no-op"):
            self.editor.apply(ColumnScale(self.base, "rounding", scale=1.000001))
        self.assertEqual(weights_hash(self.model), before)

    def test_post_write_failure_restores_original(self):
        before = weights_hash(self.model)
        real_hash = tensor_hash
        def fail_candidate_hash(tensor):
            if self.editor.active and tuple(tensor.shape) == (16, 1):
                raise RuntimeError("Injected verification failure")
            return real_hash(tensor)
        with patch("kirby.patches.tensor_hash", side_effect=fail_candidate_hash):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                self.editor.apply(ColumnScale(self.base, "write-failure"))
        self.assertTrue(self.editor.receipt["rollback_verified"])
        self.assertEqual(weights_hash(self.model), before)

    def test_unrelated_tensor_mutation_detected_on_rollback(self):
        self.editor.apply(ColumnScale(self.base, "outside-column"))
        with torch.no_grad():
            self.model.model.layers[0].mlp.fc2.weight[0, 10] += 1
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.editor.rollback()
        self.assertTrue(self.editor.poisoned)

    def run_probe(self, failure=None):
        model, base = self.model, self.base
        class Predictor:
            calls = 0
            def __init__(self):
                self.model = model
            def load(self):
                return {"checkpoint_id": base, "kind": "synthetic_test_only"}
            def generate(self, prompt):
                self.calls += 1
                if self.calls == 3 and failure is not None:
                    raise failure
                return Generation("B.", 2, False)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        manifest, _ = make_manifest(root)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return run_rollback_check(manifest, Predictor(), {}, root / "runs", questions=1)

    def test_probe_reports_full_weight_and_output_round_trip(self):
        result = self.run_probe()
        report = result["report"]
        self.assertEqual(report["state"], "completed")
        self.assertTrue(report["weights_restored"] and report["outputs_restored"])
        self.assertEqual(report["usage"]["forward_calls"], 4)
        self.assertEqual(set(report["stages"]), {"original", "original_repeat", "edited", "restored"})

    def test_probe_failure_or_interrupt_restores_weights_and_records_failure(self):
        for error in (RuntimeError("Injected inference failure"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                before = weights_hash(self.model)
                result = self.run_probe(failure=error)
                self.assertEqual(result["report"]["state"], "incomplete")
                self.assertTrue(result["report"]["patch"]["rollback_verified"])
                self.assertEqual(weights_hash(self.model), before)
                saved = json.loads((Path(result["directory"]) / "report.json").read_text())
                self.assertEqual(saved["error"]["type"], type(error).__name__)


if __name__ == "__main__":
    unittest.main()
