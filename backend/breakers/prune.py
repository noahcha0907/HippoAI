"""C2: global magnitude pruning.

The threshold is estimated from a sample, because sorting 0.87B weights is costly.
The manifest records the sparsity achieved, not just the sparsity asked for.
"""
import torch

from repair.edits import is_protected

SAMPLE_SIZE = 5_000_000


def prunable(name: str, tensor: torch.Tensor) -> bool:
    return not is_protected(name) and tensor.ndim >= 2 and tensor.is_floating_point()


def magnitude_threshold(state: dict[str, torch.Tensor], fraction: float, seed: int = 0) -> float:
    """Estimate the absolute value below which `fraction` of prunable weights fall."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    names = [name for name, tensor in state.items() if prunable(name, tensor)]
    if not names:
        raise ValueError("No prunable tensors found")
    total = sum(state[name].numel() for name in names)
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for name in names:
        flat = state[name].detach().to(torch.float32).flatten().abs()
        count = min(flat.numel(), max(1, round(SAMPLE_SIZE * flat.numel() / total)))
        indices = torch.randint(flat.numel(), (count,), generator=generator)
        samples.append(flat[indices])
    return torch.quantile(torch.cat(samples)[:SAMPLE_SIZE], fraction).item()


def prune_global(
    state: dict[str, torch.Tensor], fraction: float, seed: int = 0
) -> tuple[dict[str, torch.Tensor], dict]:
    """Zero the smallest weights across all prunable tensors."""
    threshold = magnitude_threshold(state, fraction, seed)
    damaged = dict(state)
    zeroed = 0
    total = 0
    for name, tensor in state.items():
        if not prunable(name, tensor):
            continue
        mask = tensor.abs() >= threshold
        damaged[name] = tensor * mask
        zeroed += (~mask).sum().item()
        total += tensor.numel()
    manifest = {
        "breaker": "prune_global",
        "requested_fraction": fraction,
        "achieved_fraction": zeroed / total,
        "threshold": threshold,
        "seed": seed,
    }
    return damaged, manifest
