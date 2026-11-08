import torch

from breakers.quantize import quantize_tensor
from repair.scan import group_of, scan

NAME = "model.language_model.layers.{}.mlp.down_proj.weight"


def model(layers=8, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return {NAME.format(i): torch.randn(64, 128, generator=generator) * 0.02 for i in range(layers)}


def test_groups_replace_only_the_layer_number():
    assert group_of(NAME.format(12)) == "model.language_model.layers.*.mlp.down_proj.weight"
    assert group_of("lm_head.weight") is None


def test_clean_model_has_no_suspects():
    assert scan(model()) == []


def test_finds_quantized_tensor():
    state = model()
    state[NAME.format(5)] = quantize_tensor(state[NAME.format(5)], 3)
    suspects = scan(state)
    assert suspects[0]["tensor"] == NAME.format(5)
    assert "few_distinct_values" in suspects[0]["flags"]


def test_finds_added_low_rank_distortion():
    state = model()
    u, v = torch.randn(64), torch.randn(128)
    state[NAME.format(3)] = state[NAME.format(3)] + 0.5 * torch.outer(u / u.norm(), v / v.norm())
    suspects = scan(state)
    assert suspects[0]["tensor"] == NAME.format(3)
    assert "dominant_direction" in suspects[0]["flags"]


def test_skips_protected_and_small_groups():
    state = model(layers=3)
    state["model.language_model.embed_tokens.weight"] = torch.zeros(10, 4)
    assert scan(state) == []
