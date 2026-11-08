"""Builds the user-facing report from the repair result and the sealed evaluation."""
from repair.loop import RepairResult

DISCLAIMER = (
    "Research prototype with deliberately damaged models. Not for clinical use. Outputs aren't medical advice. "
    "Questions haven't been clinically reviewed."
)
UNCHECKED = "Automatic claim checking isn't built yet, so this is the fixer's own statement."


def build_report(repair_id: str, result: RepairResult, sealed: dict, same_format: bool) -> dict:
    return {
        "mock": False,
        "id": repair_id,
        "outcome": result.outcome,
        "cause": result.cause,
        "broken": sealed["broken"],
        "repaired": sealed["repaired"],
        "clean": sealed["clean"],
        "recovery": sealed["recovery"],
        "recovery_ci95": sealed["recovery_ci95"],
        "collateral_loss": sealed["collateral_loss"],
        "weights_changed": result.diff["weights_changed"],
        "weights_changed_pct": result.diff["weights_changed_pct"],
        "tensors_touched": result.diff["tensors_touched"],
        "same_format_as_broken": same_format,
        "claims": [{"text": claim, "verified": False, "evidence": UNCHECKED} for claim in result.claims],
        "remaining_failures": sealed["remaining_failures"],
        "examples": sealed["examples"],
        "minutes": round(result.minutes, 2),
        "forward_passes": result.forward_passes,
        "tokens": result.prompt_tokens + result.completion_tokens,
        "disclaimer": DISCLAIMER,
    }
