"""One bounded Phi column edit at a time, with exact rollback.

This is an execution primitive for a controlled probe, not an editor/search
loop. There is no commit operation: every attempt must restore its base.
"""

import hashlib
import math
import re
from dataclasses import asdict, dataclass

from .records import digest


def tensor_hash(tensor) -> str:
    import torch
    if not tensor.is_contiguous():
        raise ValueError("Only contiguous dense weights are supported")
    result = hashlib.sha256()
    result.update(digest({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
    flat = tensor.detach().reshape(-1)
    # Bound temporary CPU/GPU allocations, including large embedding matrices.
    for start in range(0, flat.numel(), 1_000_000):
        result.update(flat[start:start + 1_000_000].view(torch.uint8).cpu().numpy().tobytes())
    return result.hexdigest()


def weights_hash(model) -> str:
    return digest({name: tensor_hash(value) for name, value in model.named_parameters()})


@dataclass(frozen=True)
class ColumnScale:
    base_checkpoint_id: str
    request_id: str
    layer: int = 0
    columns: tuple[int, ...] = (0,)
    scale: float = 0.97

    def __post_init__(self):
        if not isinstance(self.base_checkpoint_id, str) or not re.fullmatch(r"[0-9a-f]{64}", self.base_checkpoint_id):
            raise ValueError("A complete checkpoint SHA256 is required")
        if not isinstance(self.request_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", self.request_id):
            raise ValueError("Invalid patch request ID")
        if type(self.layer) is not int or self.layer < 0:
            raise ValueError("layer must be a nonnegative integer")
        if (not isinstance(self.columns, tuple) or not 1 <= len(self.columns) <= 16
                or any(type(c) is not int or c < 0 for c in self.columns)
                or len(set(self.columns)) != len(self.columns)):
            raise ValueError("Select 1-16 distinct nonnegative columns")
        if type(self.scale) not in (int, float) or not math.isfinite(self.scale) or not 0.9 <= self.scale <= 1.1:
            raise ValueError("scale must be finite and between 0.9 and 1.1")


@dataclass(frozen=True)
class UnitAblation:
    """Explicit single-unit zeroing; separate from the legacy scaling contract."""
    base_checkpoint_id: str
    request_id: str
    layer: int
    columns: tuple[int, ...]

    def __post_init__(self):
        ColumnScale(self.base_checkpoint_id, self.request_id, self.layer, self.columns)
        if len(self.columns) != 1:
            raise ValueError("Ablation permits exactly one MLP unit at a time")

    @property
    def scale(self):
        return 0.0


class PhiColumnEditor:
    def __init__(self, model, checkpoint_id: str):
        if model.config.model_type != "phi":
            raise ValueError("Column editing supports the built-in Phi architecture only")
        self.model = model
        self.checkpoint_id = checkpoint_id
        self.seen_requests = set()
        self.active = False
        self.poisoned = False
        self.receipt = None

    def apply(self, patch: ColumnScale | UnitAblation) -> dict:
        import torch
        if type(patch) not in (ColumnScale, UnitAblation):
            raise ValueError("Unsupported patch operation")
        ablation = isinstance(patch, UnitAblation)
        norm_limit = 0.05 if ablation else 0.01
        if self.poisoned or self.active:
            raise ValueError("Editor is active or unusable; rollback or reload the checkpoint")
        if patch.base_checkpoint_id != self.checkpoint_id:
            raise ValueError("Patch names a stale or different base checkpoint")
        if patch.request_id in self.seen_requests:
            raise ValueError("Duplicate patch request; refusing to apply scaling twice")
        self.seen_requests.add(patch.request_id)
        cfg = self.model.config
        if patch.layer >= cfg.num_hidden_layers:
            raise ValueError("Layer outside model bounds")
        name = f"model.layers.{patch.layer}.mlp.fc2.weight"
        weight = dict(self.model.named_parameters()).get(name)
        if weight is None or tuple(weight.shape) != (cfg.hidden_size, cfg.intermediate_size):
            raise ValueError("Phi output projection is missing or has the wrong dimensions")
        if weight.dtype not in (torch.float32, torch.bfloat16) or not weight.is_contiguous():
            raise ValueError("Only contiguous FP32/BF16 Phi weights can be edited")
        if max(patch.columns) >= weight.shape[1]:
            raise ValueError("Column outside projection bounds")
        indices = torch.tensor(patch.columns, device=weight.device, dtype=torch.long)
        original = weight.detach().index_select(1, indices).clone()
        proposal = torch.zeros_like(original) if ablation else (original.float() * patch.scale).to(weight.dtype)
        if not torch.isfinite(proposal).all().item():
            raise ValueError("Patch produces nonfinite weights")
        changed = int((proposal != original).sum().item())
        if not 1 <= changed <= 200_000:
            raise ValueError("Patch is a numerical no-op or exceeds 200000 changed parameters")
        norm_squared = 0.0
        flat = weight.detach().reshape(-1)
        for start in range(0, flat.numel(), 1_000_000):
            block = flat[start:start + 1_000_000].double()
            norm_squared += block.square().sum().item()
        delta_squared = (proposal.double() - original.double()).square().sum().item()
        if not math.isfinite(norm_squared) or norm_squared <= 0:
            raise ValueError("Original tensor must be finite with a positive norm")
        relative_change = math.sqrt(delta_squared / norm_squared)
        if relative_change > norm_limit:
            raise ValueError(f"Patch exceeds {norm_limit:.0%} relative tensor change")
        self.weight, self.indices, self.original = weight, indices, original
        self.before_hash = tensor_hash(weight)
        self.receipt = {"operation": "ablate_mlp_unit" if ablation else "scale_columns", **asdict(patch), "tensor": name,
                        "scale": patch.scale, "relative_tensor_change_limit": norm_limit,
                        "changed_parameters": changed, "relative_tensor_change": relative_change,
                        "before_tensor_sha256": self.before_hash,
                        "original_slice_sha256": tensor_hash(original),
                        "candidate_slice_sha256": tensor_hash(proposal), "rollback_verified": False}
        # Mark active BEFORE the write so interruptions attempt restoration too.
        self.active = True
        try:
            with torch.no_grad():
                weight.index_copy_(1, indices, proposal)
            if tensor_hash(weight.detach().index_select(1, indices)) != tensor_hash(proposal):
                raise RuntimeError("Applied slice differs from the validated proposal")
            self.receipt["after_tensor_sha256"] = tensor_hash(weight)
            self.receipt["candidate_id"] = digest(self.receipt)
            return dict(self.receipt)
        except BaseException:
            self.rollback()
            raise

    def rollback(self) -> dict:
        import torch
        if not self.active:
            if self.receipt and self.receipt["rollback_verified"]:
                return dict(self.receipt)
            raise ValueError("No active patch to restore")
        try:
            with torch.no_grad():
                self.weight.index_copy_(1, self.indices, self.original)
            restored = tensor_hash(self.weight)
            if restored != self.before_hash:
                raise RuntimeError("Rollback tensor hash mismatch; discard this worker")
            self.receipt.update(rollback_verified=True, restored_tensor_sha256=restored)
            self.active = False
            return dict(self.receipt)
        except BaseException:
            self.poisoned = True
            raise
