"""C1: harsh uniform quantization of chosen tensors, with a manifest of what was damaged.

The manifest is ground truth for the evaluator. It must never reach the repair sandbox.
"""
import torch


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Round a tensor to 2**bits evenly spaced levels between its min and max."""
    if not 1 <= bits <= 16:
        raise ValueError(f"bits must be between 1 and 16, got {bits}")
    values = tensor.detach().to(torch.float32)
    low, high = values.min(), values.max()
    if low == high:
        return tensor.clone()
    step = (high - low) / (2**bits - 1)
    return (((values - low) / step).round() * step + low).to(tensor.dtype)


def is_quantized(tensor: torch.Tensor, bits: int) -> bool:
    """Check a tensor has no more distinct values than the declared bit width allows."""
    return torch.unique(tensor).numel() <= 2**bits


def quantize_targeted(
    state: dict[str, torch.Tensor], tensor_names: list[str], bits: int
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return a damaged copy (unchanged tensors are shared) and its ground-truth manifest."""
    missing = [name for name in tensor_names if name not in state]
    if missing:
        raise KeyError(f"Tensors not in model: {missing}")
    damaged = dict(state)
    for name in tensor_names:
        damaged[name] = quantize_tensor(state[name], bits)
    manifest = {"breaker": "quantize_targeted", "bits": bits, "damaged_tensors": list(tensor_names)}
    return damaged, manifest
