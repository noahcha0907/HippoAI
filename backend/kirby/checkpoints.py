"""Shared package validation for the explicitly supported model loaders."""

from pathlib import Path

from .records import digest, file_hash, read_json


def validate_snapshot(snapshot: Path, expected_model_type: str = "gemma3") -> dict:
    """Hash a complete Safetensors package after validating its architecture."""
    config = read_json(snapshot / "config.json")
    if config.get("model_type") != expected_model_type:
        raise ValueError(f"Expected the supported {expected_model_type} architecture")
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = read_json(index_path).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Invalid Safetensors shard index")
        for filename in weight_map.values():
            if (not isinstance(filename, str) or "/" in filename or "\\" in filename
                    or ":" in filename or not filename.endswith(".safetensors")
                    or not (snapshot / filename).is_file()):
                raise ValueError("Missing or unsafe Safetensors shard")
    elif not (snapshot / "model.safetensors").is_file():
        raise ValueError("No complete Safetensors checkpoint found")
    # Hub cache symlinks to immutable blobs are expected. Uploads are not supported.
    hashes = {p.relative_to(snapshot).as_posix(): file_hash(p)
              for p in sorted(snapshot.rglob("*")) if p.is_file()}
    return {"files": hashes, "checkpoint_id": digest(hashes)}
