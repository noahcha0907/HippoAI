"""Finds tensors that stand out from the same tensor in other layers. A script, not an LLM.

For each group of same-role matrices (for example every layer's mlp.down_proj.weight), it
measures a few properties and flags the layers that are robust outliers within the group:
- distinct values in a fixed sample (harsh quantization leaves very few);
- the share of the matrix's energy in its top singular direction (an added low-rank
  distortion raises it);
- mean and spread of the values.
"""
import re

import torch

from repair.edits import is_protected

LAYER = re.compile(r"\.layers\.(\d+)\.")
SAMPLE = 200_000
POWER_STEPS = 30
MIN_GROUP = 4
OUTLIER_Z = 4.0
RELATIVE_FLOOR = 0.05  # With OUTLIER_Z, a flag needs about a 20% change from typical.


def group_of(name: str) -> str | None:
    return LAYER.sub(".layers.*.", name, count=1) if LAYER.search(name) else None


def top_singular_share(matrix: torch.Tensor, seed: int = 0) -> float:
    """sigma_1^2 / ||W||_F^2, estimated by power iteration."""
    w = matrix.float()
    generator = torch.Generator().manual_seed(seed)
    v = torch.randn(w.shape[1], generator=generator)
    for _ in range(POWER_STEPS):
        v = w.T @ (w @ v)
        v = v / v.norm()
    sigma = (w @ v).norm()
    return float(sigma**2 / (w.norm() ** 2))


def measure(name: str, tensor: torch.Tensor, seed: int = 0) -> dict:
    flat = tensor.flatten()
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(flat.numel(), (min(SAMPLE, flat.numel()),), generator=generator)
    sample = flat[picks].float()
    return {
        "distinct_values": int(torch.unique(flat[picks]).numel()),
        "top_singular_share": top_singular_share(tensor, seed),
        "mean": float(sample.mean()),
        "std": float(sample.std()),
    }


def robust_z(values: list[float], floor: float) -> list[float]:
    """Robust z-scores. `floor` stops tiny natural variation from counting as an outlier."""
    t = torch.tensor(values, dtype=torch.float64)
    median = t.median()
    mad = (t - median).abs().median() * 1.4826
    scale = max(float(mad), floor, 1e-12)
    return [float(z) for z in (t - median) / scale]


def _median(values: list[float]) -> float:
    return float(torch.tensor(values, dtype=torch.float64).median())


def scan(state: dict[str, torch.Tensor], top: int = 10) -> list[dict]:
    """Rank matrices by how strongly they stand out from their group."""
    groups: dict[str, list[str]] = {}
    for name, tensor in state.items():
        group = group_of(name)
        if group and tensor.ndim == 2 and not is_protected(name):
            groups.setdefault(group, []).append(name)

    suspects = []
    for group, names in groups.items():
        if len(names) < MIN_GROUP:
            continue
        stats = {name: measure(name, state[name]) for name in names}
        values = {key: [stats[n][key] for n in names] for key in ("distinct_values", "top_singular_share", "mean", "std")}
        typical_std = _median(values["std"])
        floors = {
            "distinct_values": RELATIVE_FLOOR * abs(_median(values["distinct_values"])),
            "top_singular_share": RELATIVE_FLOOR * abs(_median(values["top_singular_share"])),
            "std": RELATIVE_FLOOR * typical_std,
            "mean": RELATIVE_FLOOR * typical_std,  # A shift is judged against the typical spread.
        }
        z = {key: robust_z(values[key], floors[key]) for key in values}
        for i, name in enumerate(names):
            flags = {}
            if z["distinct_values"][i] <= -OUTLIER_Z:
                flags["few_distinct_values"] = stats[name]["distinct_values"]
            if z["top_singular_share"][i] >= OUTLIER_Z:
                flags["dominant_direction"] = round(stats[name]["top_singular_share"], 4)
            if abs(z["mean"][i]) >= OUTLIER_Z:
                flags["shifted_mean"] = round(stats[name]["mean"], 6)
            if abs(z["std"][i]) >= OUTLIER_Z:
                flags["unusual_spread"] = round(stats[name]["std"], 6)
            if flags:
                score = max(abs(z[k][i]) for k in z)
                suspects.append({
                    "tensor": name,
                    "group": group,
                    "outlier_score": round(score, 1),
                    "flags": flags,
                    "group_typical": {
                        "distinct_values": int(torch.tensor([stats[n]["distinct_values"] for n in names]).median()),
                        "top_singular_share": round(float(torch.tensor([stats[n]["top_singular_share"] for n in names]).median()), 4),
                    },
                })
    suspects.sort(key=lambda s: -s["outlier_score"])
    return suspects[:top]
