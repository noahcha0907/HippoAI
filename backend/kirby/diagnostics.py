"""Fixed training evidence and read-only Phi activation summaries for Astra."""

from .datasets import LETTERS, assert_disjoint, load_medqa
from .records import digest

PROBE_POLICY = {
    "version": "top_two_activation_times_column_norm_bidirectional_v1",
    "target_columns": 2, "scales": [0.98, 1.02], "max_probes": 4,
    "selection": "rank training mean_absolute_activation * weight_column_l2; ties by layer, column",
    "scope": "fixed training questions only; independent edits from original weights",
}


def select_probe_plan(activations):
    """Freeze all targets before observing any intervention outcome."""
    import math
    candidates = []
    seen = set()
    for layer in activations:
        for column in layer["columns"]:
            identity = (layer["layer"], column["column"])
            magnitude, norm = column["mean_absolute_activation"], column["weight_column_l2"]
            priority = magnitude * norm
            if (identity in seen or any(type(value) is not int or value < 0 for value in identity)
                    or not all(math.isfinite(value) and value >= 0 for value in (magnitude, norm, priority))):
                raise ValueError("Invalid or duplicate probe selection evidence")
            seen.add(identity)
            if priority > 0:
                candidates.append((-priority, *identity))
    selected = sorted(candidates)[:PROBE_POLICY["target_columns"]]
    if len(selected) != PROBE_POLICY["target_columns"]:
        raise ValueError("Not enough nonzero columns for the frozen sensitivity plan")
    return [{"probe_id": f"probe-{index}", "operation": "scale_columns", "layer": layer,
             "columns": [column], "scale": scale, "selection_priority": -priority}
            for index, (priority, layer, column, scale) in enumerate(
                ((priority, layer, column, scale) for priority, layer, column in selected
                 for scale in PROBE_POLICY["scales"]), start=1)]


def load_diagnostics(manifest, development, count):
    from pathlib import Path
    if type(count) is not int or not 1 <= count <= 64:
        raise ValueError("Select 1-64 preselected training diagnostics")
    source = manifest["sources"]["train"]
    if source.get("split") != "train":
        raise ValueError("Diagnostic source must be the training split")
    rows, actual = load_medqa(Path(source["path"]), "train")
    if actual != source:
        raise ValueError("Training data changed after manifest preparation")
    ids = manifest["diagnostic_ids"]
    if (not isinstance(ids, list) or len(ids) < count or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid frozen diagnostic IDs")
    by_id = {q.id: q for q in rows}
    if any(i not in by_id for i in ids):
        raise ValueError("Diagnostic question absent from training source")
    assert_disjoint(rows, development)
    return [by_id[i] for i in ids[:count]]


class PhiActivationSummary:
    """Observe last-position inputs to fc2, one sample per generation forward."""
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.totals = {}
        self.counts = {}

    def __enter__(self):
        import torch
        if self.model.config.model_type != "phi":
            raise ValueError("Diagnostics support Phi only")
        try:
            for layer in range(self.model.config.num_hidden_layers):
                module = self.model.get_submodule(f"model.layers.{layer}.mlp.fc2")
                if tuple(module.weight.shape) != (self.model.config.hidden_size, self.model.config.intermediate_size):
                    raise ValueError("Unexpected Phi projection shape")
                self.totals[layer] = torch.zeros(module.weight.shape[1], device=module.weight.device, dtype=torch.float32)
                self.counts[layer] = 0
                def collect(module, args, layer=layer):
                    values = args[0].detach()
                    if values.ndim != 3 or values.shape[0] != 1:
                        raise ValueError("Activation collection requires a batch of one")
                    last = values[0, -1].float().abs()
                    if not torch.isfinite(last).all().item():
                        raise ValueError("Nonfinite diagnostic activation")
                    self.totals[layer].add_(last)
                    self.counts[layer] += 1
                self.handles.append(module.register_forward_pre_hook(collect))
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        import torch
        result = []
        for layer, total in self.totals.items():
            count = self.counts[layer]
            if not count:
                raise ValueError("Diagnostic hooks observed no forward passes")
            mean = (total / count).cpu()
            # Resolve ties by column index, keeping the evidence selection deterministic.
            columns = sorted(range(mean.numel()), key=lambda i: (-float(mean[i]), i))[:8]
            weight = self.model.get_submodule(f"model.layers.{layer}.mlp.fc2").weight.detach()
            indices = torch.tensor(columns, device=weight.device)
            norms = weight.index_select(1, indices).float().square().sum(dim=0).sqrt().cpu()
            result.append({"evidence_id": f"layer:{layer}", "layer": layer, "forward_samples": count,
                           "columns": [{"column": c, "mean_absolute_activation": float(mean[c]),
                                        "weight_column_l2": float(norms[i])} for i, c in enumerate(columns)]})
        return result


def build_packet(metadata, model, diagnostics, predictions, activations, development_metrics,
                 training_scores, sensitivity):
    if len(diagnostics) != len(predictions):
        raise ValueError("Incomplete diagnostic evidence")
    examples = []
    for question, prediction in zip(diagnostics, predictions):
        if question.split != "train" or question.id != prediction["question_id"]:
            raise ValueError("Only matching training diagnostics can enter the editor packet")
        texts = [question.input.question, *question.input.options]
        limits = [4000, 1000, 1000, 1000, 1000]
        excerpts = [text[:limit] for text, limit in zip(texts, limits)]
        examples.append({"evidence_id": question.id, "question": excerpts[0],
                         "options": dict(zip(LETTERS, excerpts[1:])), "expected": question.answer,
                         "predicted": prediction["predicted"], "correct": prediction["correct"],
                         "generated_text": prediction["text"][:512],
                         "text_excerpted": any(len(text) > limit for text, limit in zip(texts, limits))
                                           or len(prediction["text"]) > 512})
    if ([row["question_id"] for row in training_scores] != [q.id for q in diagnostics]
            or not sensitivity or any(row.get("state") != "completed" or row.get("scores_restored") is not True
                                     or row.get("receipt", {}).get("rollback_verified") is not True for row in sensitivity)):
        raise ValueError("Astra requires complete, restored training sensitivity evidence")
    packet = {"schema_version": 2, "task": "one_experimental_column_scale",
              "base_checkpoint_id": metadata["checkpoint_id"],
              "model": {"repository": metadata.get("repository"), "revision": metadata.get("revision"),
                        "precision": metadata.get("precision"), "layers": model.config.num_hidden_layers,
                        "hidden_size": model.config.hidden_size, "intermediate_size": model.config.intermediate_size},
              "evidence_ids": ["development-summary", *[q.id for q in diagnostics],
                               *[row["evidence_id"] for row in activations],
                               *[row["probe_id"] for row in sensitivity]],
              "development_summary": development_metrics,
              "training_diagnostics": examples, "activation_summary": activations,
              "training_answer_scores": training_scores,
              "sensitivity_policy": PROBE_POLICY,
              "training_sensitivity": [{"evidence_id": row["probe_id"], "operation": row["operation"],
                                        "comparison": row["comparison"], "scores_restored": row["scores_restored"],
                                        "rollback_verified": row["receipt"]["rollback_verified"]}
                                       for row in sensitivity],
              "activation_sampling": "absolute_fc2_input_last_position_per_generation_forward_v1",
              "limits": {"operations": 1, "layers": 1, "columns": 16, "scale_min": 0.9, "scale_max": 1.1,
                         "max_changed_parameters": 200000, "max_relative_tensor_change": 0.01},
              "limitations": ["Magnitudes are descriptive, not causal evidence or a diagnosis of damage.",
                              "Label scores sum log probabilities for explicit A./B./C./D. token suffixes, no EOS or length normalization.",
                              "Probabilities are normalized over those four strings only, not calibrated medical confidence.",
                              "Training intervention effects are local, small-sample evidence; not proof of development or final gain.",
                              "Only the recorded individual probes were measured; combinations and other scales are extrapolations.",
                              "Only the listed top-activation columns may be proposed.",
                              "No final-test questions or scores are available. All edits are rolled back."]}
    packet["evidence_sha256"] = digest(packet)
    return packet
