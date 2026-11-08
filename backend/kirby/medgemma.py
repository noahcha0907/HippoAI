"""Optional GPU adapter, based on the official MedGemma loading example.

No import-time GPU allocation, network access, or credential handling. Real
checkpoint compatibility must be established by calibration on the target GPU.
"""

import importlib.metadata
import os
import platform
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .evaluation import Generation
from .checkpoints import validate_snapshot

MODEL_ID = "google/medgemma-1.5-4b-it"


@dataclass(frozen=True)
class ModelConfig:
    revision: str
    repository: str = MODEL_ID
    max_new_tokens: int = 8
    seed: int = 17
    attention_implementation: str = "eager"
    local_files_only: bool = True

    def __post_init__(self):
        if self.repository != MODEL_ID:
            raise ValueError("Only the approved MedGemma demo repository is supported")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an explicit, lowercase 40-character commit SHA; branches are not accepted")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be an integer between 1 and 64")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.attention_implementation not in {"eager", "sdpa"}:
            raise ValueError("Supported attention implementations: eager, sdpa")


class MedGemma:
    def __init__(self, config: ModelConfig):
        self.config = config

    def load(self) -> dict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
        except ImportError as error:
            raise RuntimeError("Install the backend GPU extra in a compatible CUDA environment before running MedGemma") from error
        if (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8
                or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("This initial MedGemma profile requires a CUDA GPU with native BF16 support (for example L4 or A100; not T4)")
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        snapshot = Path(snapshot_download(
            repo_id=self.config.repository, revision=self.config.revision,
            local_files_only=self.config.local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"],
        ))
        identity = validate_snapshot(snapshot)
        self.processor = AutoProcessor.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            snapshot, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
            attn_implementation=self.config.attention_implementation,
            trust_remote_code=False, use_safetensors=True, local_files_only=True,
        )
        self.model.eval()
        self.torch = torch
        # Explicit greedy generation, rather than inheriting sampling or custom
        # stop strings from a checkpoint's generation_config.json.
        tokenizer = self.processor.tokenizer
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = tokenizer.eos_token_id
        if eos_ids is None:
            raise ValueError("No end-of-sequence token is configured")
        self.eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.config.max_new_tokens,
            use_cache=True, eos_token_id=self.eos_ids, pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
        )
        template = self.processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "KIRBY_TEMPLATE_PROBE"}]}],
            tokenize=False, add_generation_prompt=True,
        )
        torch.cuda.reset_peak_memory_stats()
        return {"kind": "medgemma", **asdict(self.config), **identity,
                "precision": "bfloat16", "batch_size": 1,
                "model_class": type(self.model).__name__, "processor_class": type(self.processor).__name__,
                "tokenizer_class": type(tokenizer).__name__, "rendered_template_probe": template,
                "generation_config": self.generation_config.to_dict(),
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "environment": {"python": platform.python_version(), "os": platform.platform(),
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                "packages": {name: importlib.metadata.version(name) for name in
                                             ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors")}}}

    def generate(self, prompt: str) -> Generation:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        # Cache is private to this generate call; no past_key_values cross questions.
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation_config)
        new_tokens = output[0][input_length:].tolist()
        truncated = len(new_tokens) >= self.config.max_new_tokens and new_tokens[-1] not in self.eos_ids
        return Generation(self.processor.decode(new_tokens, skip_special_tokens=True), len(new_tokens), truncated)

    def resource_usage(self) -> dict:
        return {"peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": self.torch.cuda.max_memory_reserved()}
