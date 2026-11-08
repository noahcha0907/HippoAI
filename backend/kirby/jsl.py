"""John Snow Labs' public MedPhi2 checkpoint, loaded with built-in Phi code.

No JSL hosted endpoint or Healthcare NLP license key is used by this adapter.
The model's CC-BY-NC-ND terms still apply; this module only evaluates it.
"""

import importlib.metadata
import os
import platform
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .checkpoints import validate_snapshot
from .evaluation import Generation

MODEL_ID = "johnsnowlabs/JSL-MedPhi2-2.7B"
# Resolved from the public Hub API on 2026-11-07; not a claimed GPU-tested revision.
REFERENCE_REVISION = "b313a242eb211863f7b62624b345b321a3b114ea"
LICENSE_URL = "https://huggingface.co/johnsnowlabs"
ANSWER_STOP_POLICY = "complete_uppercase_option_label_period_v1"


class StopAfterOptionLabel:
    """Batch-one stop on an entire generated label like 'D.', never prompt text.

    This observes unconstrained generation; it neither chooses a letter nor
    extracts one from an explanation. The generated period remains in evidence.
    """

    def __init__(self, tokenizer, input_length: int):
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.matched = False

    def __call__(self, input_ids, scores, **kwargs):
        if len(input_ids) != 1:
            raise ValueError("Option-label stopping supports only batch size one")
        generated = input_ids[0][self.input_length:].tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        self.matched = re.fullmatch(r"[A-D]\.", text) is not None
        return self.matched


@dataclass(frozen=True)
class JSLModelConfig:
    revision: str
    repository: str = MODEL_ID
    precision: str = "float32"
    max_new_tokens: int = 8
    seed: int = 17
    attention_implementation: str = "eager"
    local_files_only: bool = True
    answer_stop_policy: str = field(default=ANSWER_STOP_POLICY, init=False)

    def __post_init__(self):
        if self.repository != MODEL_ID:
            raise ValueError("Only the approved JSL-MedPhi2-2.7B repository is supported by this adapter")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError("revision must be an explicit, lowercase 40-character commit SHA")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("JSL precision must be float32 or bfloat16; no implicit FP16 or quantization")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 64:
            raise ValueError("max_new_tokens must be an integer between 1 and 64")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        if self.attention_implementation != "eager":
            raise ValueError("The initial JSL profile supports only eager attention")


def check_hardware(cuda_available: bool, capability_major: int, bf16_supported: bool, precision: str) -> None:
    if not cuda_available:
        raise RuntimeError("Select a CUDA GPU runtime in Colab before loading the model")
    if precision == "bfloat16" and (capability_major < 8 or not bf16_supported):
        raise RuntimeError("BF16 needs a compatible GPU such as L4/A100; select float32 explicitly for T4")


def check_loading_info(info: dict) -> None:
    # Refuse an old/custom architecture silently loading with random or unused tensors.
    problems = {key: info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs") if info.get(key)}
    if problems:
        raise ValueError(f"Checkpoint does not match the built-in Phi loader: {problems}")


class JSLMedPhi:
    def __init__(self, config: JSLModelConfig):
        self.config = config

    def load(self) -> dict:
        # Needed before CUDA initialization when deterministic algorithms use cuBLAS.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoTokenizer, GenerationConfig, PhiConfig, PhiForCausalLM, StoppingCriteriaList
        except ImportError as error:
            raise RuntimeError("Install the backend GPU extra with built-in Transformers Phi support") from error
        cuda_available = torch.cuda.is_available()
        check_hardware(cuda_available, torch.cuda.get_device_capability()[0] if cuda_available else 0,
                       torch.cuda.is_bf16_supported() if cuda_available else False, self.config.precision)
        torch.manual_seed(self.config.seed)
        torch.cuda.manual_seed_all(self.config.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        snapshot = Path(snapshot_download(
            repo_id=self.config.repository, revision=self.config.revision,
            local_files_only=self.config.local_files_only,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        ))
        identity = validate_snapshot(snapshot, expected_model_type="phi")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False, local_files_only=True)
        if not self.tokenizer.chat_template:
            raise ValueError("The pinned JSL tokenizer must supply its chat template")
        phi_config = PhiConfig.from_pretrained(snapshot, local_files_only=True)
        self.context_length = phi_config.max_position_embeddings
        dtype = torch.float32 if self.config.precision == "float32" else torch.bfloat16
        # Use the installed Phi class explicitly; do not follow legacy auto_map
        # references to executable Python in a different model repository.
        self.model, loading_info = PhiForCausalLM.from_pretrained(
            snapshot, config=phi_config, torch_dtype=dtype, device_map={"": "cuda:0"},
            attn_implementation=self.config.attention_implementation,
            trust_remote_code=False, use_safetensors=True, local_files_only=True,
            output_loading_info=True,
        )
        check_loading_info(loading_info)
        self.model.eval()
        self.torch = torch
        self.stopping_criteria_type = StoppingCriteriaList
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if eos_ids is None:
            raise ValueError("The JSL checkpoint must define an EOS token")
        self.eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
        self.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, max_new_tokens=self.config.max_new_tokens,
            use_cache=True, eos_token_id=self.eos_ids, bos_token_id=self.tokenizer.bos_token_id,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_ids[0],
        )
        template = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "KIRBY_TEMPLATE_PROBE"}], tokenize=False, add_generation_prompt=True,
        )
        torch.cuda.reset_peak_memory_stats()
        return {"kind": "jsl_medphi2", **asdict(self.config), **identity, "batch_size": 1,
                "model_class": type(self.model).__name__, "tokenizer_class": type(self.tokenizer).__name__,
                "context_length": self.context_length, "rendered_template_probe": template,
                "generation_config": self.generation_config.to_dict(), "loading_info": loading_info,
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "license": "cc-by-nc-nd-4.0", "license_terms_url": LICENSE_URL,
                "limitations": ["JSL public model terms restrict commercial use and distribution of modified weights; no edited checkpoint is released by this evaluator."],
                "environment": {"python": platform.python_version(), "os": platform.platform(),
                                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                                "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
                                "packages": {name: importlib.metadata.version(name) for name in
                                             ("torch", "transformers", "accelerate", "huggingface-hub", "safetensors")}}}

    def generate(self, prompt: str) -> Generation:
        return self._generate(prompt, capture_scores=False)

    def generate_with_answer_scores(self, prompt: str):
        """Return raw first-token scores from the same unconstrained generation."""
        return self._generate(prompt, capture_scores=True)

    def _generate(self, prompt: str, *, capture_scores):
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_dict=True, return_tensors="pt",
        )
        input_length = inputs["input_ids"].shape[-1]
        if input_length + self.config.max_new_tokens > self.context_length:
            raise ValueError(f"Question exceeds the model's {self.context_length}-token context; no input was truncated")
        inputs = inputs.to(self.model.device)
        stop = StopAfterOptionLabel(self.tokenizer, input_length)
        extra = {"return_dict_in_generate": True, "output_logits": True} if capture_scores else {}
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, generation_config=self.generation_config,
                                         stopping_criteria=self.stopping_criteria_type([stop]), **extra)
        sequences = output.sequences if capture_scores else output
        new_tokens = sequences[0][input_length:].tolist()
        truncated = (len(new_tokens) >= self.config.max_new_tokens
                     and new_tokens[-1] not in self.eos_ids and not stop.matched)
        generation = Generation(self.tokenizer.decode(new_tokens, skip_special_tokens=True), len(new_tokens), truncated)
        if capture_scores:
            from .answer_scores import next_label_scores
            if not output.logits:
                raise ValueError("Generation returned no first-token logits")
            return generation, next_label_scores(self.tokenizer, output.logits[0][0])
        return generation

    def resource_usage(self) -> dict:
        return {"peak_gpu_allocated_bytes": self.torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": self.torch.cuda.max_memory_reserved()}
