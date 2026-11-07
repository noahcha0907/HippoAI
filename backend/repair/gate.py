"""Dev-set acceptance gate for one repair attempt.

The clean model isn't available here, so the gate can't use recovery. Recovery is
computed later by the sealed evaluator. Thresholds are proposals from the plan and
should be fixed once the clean and broken models have been measured.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class GateThresholds:
    min_clinical_gain: float = 0.03
    min_gain_over_random: float = 0.02
    max_collateral_loss: float = 0.01


@dataclass(frozen=True)
class AttemptScores:
    broken_clinical: float
    edited_clinical: float
    random_control_clinical: float
    collateral_loss: float
    half_strength_clinical: float
    garbled_rate_increase: float = 0.0


def evaluate_gate(scores: AttemptScores, limits: GateThresholds = GateThresholds()) -> tuple[bool, list[str]]:
    """Return whether to accept, plus the reason for every failed check."""
    failures = []
    gain = scores.edited_clinical - scores.broken_clinical
    if gain < limits.min_clinical_gain:
        failures.append(f"Clinical gain {gain:.3f} is below {limits.min_clinical_gain:.3f}")
    over_random = scores.edited_clinical - scores.random_control_clinical
    if over_random < limits.min_gain_over_random:
        failures.append(f"Only {over_random:.3f} better than a random edit of the same size")
    if scores.collateral_loss > limits.max_collateral_loss:
        failures.append(f"Collateral loss {scores.collateral_loss:.3f} exceeds {limits.max_collateral_loss:.3f}")
    # Dose-response: half the edit should land between no edit and the full edit.
    if not scores.broken_clinical <= scores.half_strength_clinical <= scores.edited_clinical:
        failures.append("No dose-response: the half-strength edit isn't between broken and full strength")
    if scores.garbled_rate_increase > 0:
        failures.append(f"Garbled or looping output rose by {scores.garbled_rate_increase:.3f}")
    return not failures, failures
