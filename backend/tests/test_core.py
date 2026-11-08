import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from breakers.prune import prune_global
from breakers.quantize import is_quantized, quantize_targeted, quantize_tensor
from data.load_medmcqa import split_questions, to_question
from eval.metrics import collateral_loss, recovery, recovery_ci, unsafe_answer_rate
from eval.scoring import Question, score_questions
from repair.edits import EditRefused, EditSession, is_protected
from repair.gate import AttemptScores, evaluate_gate

LAYER = "model.language_model.layers.15.mlp.down_proj.weight"
EMBED = "model.language_model.embed_tokens.weight"


@pytest.fixture
def state():
    torch.manual_seed(0)
    return {LAYER: torch.randn(64, 32), EMBED: torch.randn(100, 8)}


def question(text, answer="A", id_="q"):
    return Question(id=id_, question=text, options={"A": "a", "B": "b", "C": "c", "D": "d"}, answer=answer)


# Edit rules
@pytest.mark.parametrize(
    "name",
    [
        EMBED,
        "lm_head.weight",
        "model.language_model.norm.weight",
        "model.norm.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ],
)
def test_protected_names(name):
    assert is_protected(name)


def test_layer_norms_are_editable():
    assert not is_protected("model.language_model.layers.3.input_layernorm.weight")
    assert not is_protected(LAYER)


def test_refuses_protected_tensor(state):
    with pytest.raises(EditRefused, match="Protected"):
        EditSession(state).save_edit(EMBED, state[EMBED] + 1)


def test_refuses_whole_tensor_scaling(state):
    with pytest.raises(EditRefused, match="constant"):
        EditSession(state).save_edit(LAYER, state[LAYER] * 1.5)


def test_refuses_bad_shape_unknown_name_and_nan(state):
    session = EditSession(state)
    with pytest.raises(EditRefused, match="Shape"):
        session.save_edit(LAYER, torch.zeros(2, 2))
    with pytest.raises(EditRefused, match="Unknown"):
        session.save_edit("nope", torch.zeros(2, 2))
    bad = state[LAYER].clone()
    bad[0, 0] = float("nan")
    with pytest.raises(EditRefused, match="NaN"):
        session.save_edit(LAYER, bad)


def test_rollback_restores_exactly(state):
    original = state[LAYER].clone()
    session = EditSession(state)
    edited = original.clone()
    edited[0, :4] += 1
    session.save_edit(LAYER, edited)
    assert session.diff()["weights_changed"] == 4
    session.rollback()
    assert torch.equal(session.state()[LAYER], original)
    assert session.diff()["weights_changed"] == 0
    assert torch.equal(state[LAYER], original)  # Broken model never mutated.


def test_rollback_keeps_accepted_edit(state):
    session = EditSession(state)
    first = state[LAYER].clone()
    first[0, 0] += 1
    session.save_edit(LAYER, first)
    session.accept()
    second = first.clone()
    second[1, 1] += 1
    session.save_edit(LAYER, second)
    session.rollback()
    assert torch.equal(session.state()[LAYER], first)


def test_empty_edit_changes_nothing(state):
    session = EditSession(state)
    session.save_edit(LAYER, state[LAYER].clone())
    assert session.diff()["weights_changed"] == 0
    assert torch.equal(session.state()[LAYER], state[LAYER])


def test_diff_ignores_changes_lost_to_rounding():
    base = torch.tensor([[1.0, 2.0], [3.0, 5.0]], dtype=torch.bfloat16)
    session = EditSession({LAYER: base})
    session.save_edit(LAYER, base.to(torch.float32) + 1e-6)
    assert session.diff()["weights_changed"] == 0


# Breakers
def test_quantize_respects_bit_width(state):
    assert is_quantized(quantize_tensor(state[LAYER], 3), 3)
    assert not is_quantized(state[LAYER], 3)


def test_targeted_quantize_only_touches_named_tensors(state):
    damaged, manifest = quantize_targeted(state, [LAYER], 3)
    assert not torch.equal(damaged[LAYER], state[LAYER])
    assert damaged[EMBED] is state[EMBED]
    assert manifest["damaged_tensors"] == [LAYER]
    with pytest.raises(KeyError):
        quantize_targeted(state, ["missing"], 3)


def test_prune_hits_requested_sparsity_and_skips_protected():
    torch.manual_seed(0)
    state = {LAYER: torch.randn(1000, 1000), EMBED: torch.randn(100, 8)}
    damaged, manifest = prune_global(state, 0.5)
    assert manifest["achieved_fraction"] == pytest.approx(0.5, abs=0.01)
    assert (damaged[LAYER] == 0).float().mean().item() == pytest.approx(0.5, abs=0.01)
    assert damaged[EMBED] is state[EMBED]


# Gate
def scores(**overrides):
    values = dict(
        broken_clinical=0.40,
        edited_clinical=0.50,
        random_control_clinical=0.40,
        collateral_loss=0.0,
        half_strength_clinical=0.45,
    )
    return AttemptScores(**{**values, **overrides})


def test_gate_accepts_good_edit():
    assert evaluate_gate(scores()) == (True, [])


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ({"edited_clinical": 0.41, "half_strength_clinical": 0.405}, "gain"),
        ({"random_control_clinical": 0.49}, "random"),
        ({"collateral_loss": 0.05}, "Collateral"),
        ({"half_strength_clinical": 0.30}, "dose-response"),
        ({"garbled_rate_increase": 0.02}, "Garbled"),
    ],
)
def test_gate_rejects(overrides, expected):
    accepted, reasons = evaluate_gate(scores(**overrides))
    assert not accepted
    assert any(expected in reason for reason in reasons)


# Metrics
def test_recovery():
    assert recovery(0.6, 0.4, 0.8) == pytest.approx(0.5)
    assert recovery(0.3, 0.4, 0.8) == pytest.approx(-0.25)
    assert recovery(0.5, 0.8, 0.8) is None


def test_collateral_loss():
    assert collateral_loss(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == 0.25


def test_unsafe_answer_rate():
    assert unsafe_answer_rate(["A", "B"], [["A"], ["C"]]) == 0.5


def test_recovery_ci_contains_estimate():
    rng = np.random.default_rng(1)
    clean = rng.random(2000) < 0.8
    broken = clean & (rng.random(2000) < 0.5)
    repaired = broken | (clean & (rng.random(2000) < 0.5))
    low, high = recovery_ci(broken, repaired, clean)
    point = recovery(repaired.mean(), broken.mean(), clean.mean())
    assert low < point < high


def test_recovery_ci_none_without_damage():
    same = np.array([1, 0] * 100)
    assert recovery_ci(same, same, same) is None


# Data splits
def test_split_is_disjoint_deduped_and_deterministic():
    items = [question(f"Question {i}", id_=str(i)) for i in range(50)]
    items.append(question("QUESTION   0", id_="duplicate"))
    dev, sealed = split_questions(items, 20, 20)
    dev_again, _ = split_questions(list(reversed(items[:-1])) + items[-1:], 20, 20)
    assert not {q.question.lower() for q in dev} & {q.question.lower() for q in sealed}
    assert [q.question for q in dev] == [q.question for q in dev_again]
    with pytest.raises(ValueError):
        split_questions(items, 40, 40)


def test_to_question_handles_both_schemas_and_missing_labels():
    nested = {
        "id": "x",
        "subject_name": "Medicine",
        "data": {"Question": "Q?", "Options": dict(A="a", B="b", C="c", D="d"), "Correct Option": "C"},
    }
    assert to_question(nested, 0).answer == "C"
    flat = dict(question="Q?", opa="a", opb="b", opc="c", opd="d", cop=1)
    assert to_question(flat, 0).answer == "B"
    assert to_question({**flat, "cop": -1}, 0) is None
    with pytest.raises(ValueError):
        to_question({"foo": 1}, 0)


# Scoring, with a fake model whose last-position logits favor one token.
class FakeTokenizer:
    padding_side = "right"
    letter_ids = {"A": 1, "B": 2, "C": 3, "D": 4}

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return messages[0]["content"]

    def encode(self, text, add_special_tokens=False):
        return [self.letter_ids[text]]

    def __call__(self, texts, return_tensors, padding):
        assert self.padding_side == "left"
        ids = torch.ones((len(texts), 3), dtype=torch.long)
        return SimpleNamespace(to=lambda device: {"input_ids": ids})


class FakeModel(torch.nn.Module):
    def __init__(self, favored):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.favored = favored

    def forward(self, input_ids):
        logits = torch.zeros(*input_ids.shape, 10)
        logits[:, -1, self.favored] = 5.0
        return SimpleNamespace(logits=logits)


def test_score_questions_reads_last_position():
    tokenizer = FakeTokenizer()
    items = [question("one", "B", "1"), question("two", "A", "2")]
    results = score_questions(FakeModel(favored=2), tokenizer, items)
    assert [r["prediction"] for r in results] == ["B", "B"]
    assert [r["correct"] for r in results] == [True, False]
    assert tokenizer.padding_side == "right"  # Restored afterward.


# Fixtures are labeled as mock.
@pytest.mark.parametrize("path", sorted((Path(__file__).parent.parent / "fixtures").glob("*.json")), ids=lambda p: p.stem)
def test_every_fixture_is_marked_mock(path):
    assert json.loads(path.read_text(encoding="utf-8"))["mock"] is True


def test_gate_accepts_values_exactly_at_thresholds_despite_rounding():
    # 0.69 - 0.66 is 0.0299999... in floating point; the real run hit this.
    scores_at_limit = AttemptScores(
        broken_clinical=0.66, edited_clinical=0.69, random_control_clinical=0.67,
        collateral_loss=0.01, half_strength_clinical=0.675,
    )
    assert evaluate_gate(scores_at_limit) == (True, [])


def noisy(**overrides):
    values = dict(
        broken_clinical=0.66, edited_clinical=0.77, random_control_clinical=0.65, collateral_loss=0.043,
        half_strength_clinical=0.72, general_net_change=0.02, random_control_collateral=0.035,
    )
    return AttemptScores(**{**values, **overrides})


def test_gate_accepts_churn_similar_to_a_random_edit():
    # The real t2-a attempt: 4.3% churn on a near-chance set, net score up.
    assert evaluate_gate(noisy()) == (True, [])


def test_gate_rejects_churn_well_beyond_a_random_edit():
    accepted, reasons = evaluate_gate(noisy(collateral_loss=0.09))
    assert not accepted and any("random edit's" in r for r in reasons)


def test_gate_rejects_a_falling_general_score():
    accepted, reasons = evaluate_gate(noisy(general_net_change=-0.03))
    assert not accepted and any("fell by" in r for r in reasons)


def test_gate_uses_absolute_limit_without_a_random_comparison():
    accepted, reasons = evaluate_gate(noisy(random_control_collateral=None))
    assert not accepted and any("exceeds 0.010" in r for r in reasons)
