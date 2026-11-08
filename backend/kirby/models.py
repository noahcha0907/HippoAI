"""Explicit supported-model dispatch. No provider or architecture fallback."""

from .jsl import MODEL_ID as JSL_ID, JSLMedPhi, JSLModelConfig
from .medgemma import MODEL_ID as MEDGEMMA_ID, MedGemma, ModelConfig

DEFAULT_MODEL = JSL_ID
SUPPORTED_MODELS = (JSL_ID, MEDGEMMA_ID)


def create_predictor(repository: str, revision: str, *, precision: str | None = None,
                     max_new_tokens: int = 8, seed: int = 17, local_files_only: bool = True):
    settings = dict(revision=revision, repository=repository, max_new_tokens=max_new_tokens,
                    seed=seed, local_files_only=local_files_only)
    if repository == JSL_ID:
        return JSLMedPhi(JSLModelConfig(**settings, precision=precision or "float32"))
    if repository == MEDGEMMA_ID:
        if precision not in (None, "bfloat16"):
            raise ValueError("The legacy MedGemma adapter supports only bfloat16")
        return MedGemma(ModelConfig(**settings))
    raise ValueError("Unsupported model repository")
