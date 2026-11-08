"""Applies and rolls back the fixer's edits, enforcing the blog's rules for what may change.

Model files are never modified here. This holds a working copy of touched tensors,
so rollback restores the exact original bytes.
"""
import re

import torch

# Blog rule (Exp. 8): no final norm, embeddings, or output layer. The vision tower
# isn't part of a text repair, so it's refused too.
PROTECTED_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        r"(^|\.)embed_tokens\.",
        r"(^|\.)lm_head\.",
        r"(^|\.)language_model\.norm\.",
        r"^model\.norm\.",
        r"(^|\.)visual\.",
    )
]
SCALE_TOLERANCE = 1e-5


class EditRefused(ValueError):
    """The edit breaks a rule. The reason is shown to the fixer."""


def is_protected(name: str) -> bool:
    return any(pattern.search(name) for pattern in PROTECTED_PATTERNS)


def is_constant_scaling(original: torch.Tensor, edited: torch.Tensor) -> bool:
    """True if the edit only multiplies a whole tensor by one number."""
    a = original.to(torch.float64).flatten()
    b = edited.to(torch.float64).flatten()
    denominator = a.dot(a)
    if denominator == 0:
        return False
    scale = a.dot(b) / denominator
    residual = (b - scale * a).norm() / a.norm()
    return bool(residual < SCALE_TOLERANCE)


class EditSession:
    def __init__(self, state: dict[str, torch.Tensor]):
        self._base = state
        self._pending: dict[str, torch.Tensor] = {}
        self._accepted: dict[str, torch.Tensor] = {}

    @property
    def base(self) -> dict[str, torch.Tensor]:
        """The broken model. Treat as read-only."""
        return self._base

    def pending_names(self) -> list[str]:
        return sorted(self._pending)

    def current(self, name: str) -> torch.Tensor:
        if name in self._pending:
            return self._pending[name]
        if name in self._accepted:
            return self._accepted[name]
        return self._base[name]

    def save_edit(self, name: str, tensor: torch.Tensor) -> None:
        """Queue a replacement tensor, refusing anything outside the rules."""
        if name not in self._base:
            raise EditRefused(f"Unknown tensor: {name}")
        if is_protected(name):
            raise EditRefused(f"Protected tensor can't be edited: {name}")
        base = self._base[name]
        if tensor.shape != base.shape:
            raise EditRefused(f"Shape {tuple(tensor.shape)} doesn't match {tuple(base.shape)} for {name}")
        if not torch.isfinite(tensor.to(torch.float32)).all():
            raise EditRefused(f"Edit contains NaN or infinity: {name}")
        # Compare in the model's own precision, so rounding-only changes don't count.
        rounded = tensor.detach().to(base.dtype)
        if not torch.equal(rounded, base) and is_constant_scaling(base, rounded):
            raise EditRefused(f"Scaling a whole tensor by a constant isn't allowed: {name}")
        self._pending[name] = rounded

    def rollback(self) -> None:
        """Discard pending edits and return to the last accepted state."""
        self._pending = {}

    def accept(self) -> None:
        """Keep pending edits."""
        self._accepted.update(self._pending)
        self._pending = {}

    def state(self) -> dict[str, torch.Tensor]:
        """Full state with accepted and pending edits applied. Unedited tensors are shared."""
        return {**self._base, **self._accepted, **self._pending}

    def diff(self) -> dict:
        """Count weights that differ from the broken model, against the whole model."""
        touched = []
        changed = 0
        for name, tensor in {**self._accepted, **self._pending}.items():
            count = (tensor != self._base[name]).sum().item()
            if count:
                touched.append(name)
                changed += count
        total = sum(tensor.numel() for tensor in self._base.values())
        return {
            "tensors_touched": sorted(touched),
            "weights_changed": changed,
            "weights_changed_pct": 100.0 * changed / total if total else 0.0,
        }
