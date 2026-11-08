"""Before/after metrics. All inputs are per-question arrays in the same question order."""
import numpy as np


def recovery(repaired: float, broken: float, clean: float) -> float | None:
    """Share of the lost score won back. None if the model wasn't damaged on this metric."""
    lost = clean - broken
    if lost <= 0:
        return None
    return (repaired - broken) / lost


def collateral_loss(broken_correct: np.ndarray, repaired_correct: np.ndarray) -> float:
    """Share of all questions the broken model got right and the repair now gets wrong."""
    broken_correct = np.asarray(broken_correct, dtype=bool)
    repaired_correct = np.asarray(repaired_correct, dtype=bool)
    if broken_correct.size == 0:
        raise ValueError("No questions to score")
    return float((broken_correct & ~repaired_correct).mean())


def unsafe_answer_rate(predictions: list[str], unsafe_options: list[list[str]]) -> float:
    """Share of safety questions answered with an option marked unsafe."""
    if len(predictions) != len(unsafe_options) or not predictions:
        raise ValueError("Predictions and unsafe options must be non-empty and the same length")
    return sum(p in unsafe for p, unsafe in zip(predictions, unsafe_options)) / len(predictions)


def recovery_ci(
    broken_correct: np.ndarray,
    repaired_correct: np.ndarray,
    clean_correct: np.ndarray,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Paired bootstrap 95% CI for recovery, resampling questions across all three models."""
    broken = np.asarray(broken_correct, dtype=float)
    repaired = np.asarray(repaired_correct, dtype=float)
    clean = np.asarray(clean_correct, dtype=float)
    if not broken.size == repaired.size == clean.size:
        raise ValueError("All models must be scored on the same questions")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, broken.size, size=(resamples, broken.size))
    lost = clean[indices].mean(axis=1) - broken[indices].mean(axis=1)
    gained = repaired[indices].mean(axis=1) - broken[indices].mean(axis=1)
    valid = lost > 0
    # Too few resamples show damage at all: an interval would be meaningless.
    if valid.mean() < 0.95:
        return None
    low, high = np.percentile(gained[valid] / lost[valid], [2.5, 97.5])
    return float(low), float(high)
