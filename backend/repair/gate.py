"""Dev-set acceptance gate for one repair attempt.

The clean model isn't available here, so the gate can't use recovery. Recovery is
computed later by the sealed evaluator. Thresholds are proposals from the plan and
should be fixed once the clean and broken models have been measured.
"""
from dataclasses import dataclass

# Scores are fractions of a question set, so float rounding can land just past a threshold.
TOLERANCE = 1e-9


@dataclass(frozen=True)
class GateThresholds:
    min_clinical_gain: float = 0.03
    min_gain_over_random: float = 0.02
    max_collateral_loss: float = 0.01  # Used only when no random-edit comparison is available.
    max_net_drop: float = 0.01  # Largest allowed fall in the general-knowledge score.
    max_churn_over_random: float = 0.02  # Reshuffling allowed beyond what a random edit causes.


@dataclass(frozen=True)
class AttemptScores:
    broken_clinical: float
    edited_clinical: float
    random_control_clinical: float
    collateral_loss: float
    half_strength_clinical: float
    garbled_rate_increase: float = 0.0
    general_net_change: float | None = None  # Edited minus broken general-knowledge accuracy.
    random_control_collateral: float | None = None  # Churn a random edit of the same size causes.


def evaluate_gate(scores: AttemptScores, limits: GateThresholds = GateThresholds()) -> tuple[bool, list[str]]:
    """Return whether to accept, plus the reason for every failed check."""
    failures = []
    gain = scores.edited_clinical - scores.broken_clinical
    if gain < limits.min_clinical_gain - TOLERANCE:
        failures.append(f"Clinical gain {gain:.3f} is below {limits.min_clinical_gain:.3f}")
    over_random = scores.edited_clinical - scores.random_control_clinical
    if over_random < limits.min_gain_over_random - TOLERANCE:
        failures.append(f"Only {over_random:.3f} better than a random edit of the same size")
    if scores.random_control_collateral is None:
        if scores.collateral_loss > limits.max_collateral_loss + TOLERANCE:
            failures.append(f"Collateral loss {scores.collateral_loss:.3f} exceeds {limits.max_collateral_loss:.3f}")
    else:
        # The general-knowledge set is near chance for this model, so some answers flip under any edit.
        # Judge reshuffling against a random edit, and require the net score not to fall.
        allowed = scores.random_control_collateral + limits.max_churn_over_random
        if scores.collateral_loss > allowed + TOLERANCE:
            failures.append(
                f"Collateral loss {scores.collateral_loss:.3f} exceeds a random edit's "
                f"{scores.random_control_collateral:.3f} by more than {limits.max_churn_over_random:.3f}"
            )
        if scores.general_net_change is not None and scores.general_net_change < -limits.max_net_drop - TOLERANCE:
            failures.append(f"General-knowledge score fell by {-scores.general_net_change:.3f}")
    # Dose-response: half the edit should land between no edit and the full edit.
    low, high = scores.broken_clinical - TOLERANCE, scores.edited_clinical + TOLERANCE
    if not low <= scores.half_strength_clinical <= high:
        failures.append("No dose-response: the half-strength edit isn't between broken and full strength")
    if scores.garbled_rate_increase > TOLERANCE:
        failures.append(f"Garbled or looping output rose by {scores.garbled_rate_increase:.3f}")
    return not failures, failures
