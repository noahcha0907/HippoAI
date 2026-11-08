"""One small summary of a finished repair, for the scorecard screen."""

GATE_NOTE = (
    "The acceptance gate's rounding and collateral check were changed after seeing results, including "
    "measuring side effects only on MedMCQA questions answered right in at least 3 of 4 answer orders; "
    "see the backend README, 'Gate changes made after seeing results'."
)


def build_summary(status: dict, report: dict) -> dict:
    attempts = status.get("attempts", [])
    accepted = [a for a in attempts if a.get("outcome") == "accepted"]
    best = max(attempts, key=lambda a: a["dev_metrics"].get("clinical_accuracy") or 0, default=None)
    return {
        "mock": False,
        "repair_id": status["id"],
        "model_id": status["model_id"],
        "complaint": status["complaint"],
        "outcome": report["outcome"],
        "cause": report["cause"],
        "accuracy": {k: report[k]["clinical_accuracy"] for k in ("broken", "repaired", "clean")},
        "unsafe_answer_rate": {k: report[k]["unsafe_answer_rate"] for k in ("broken", "repaired", "clean")},
        "recovery": report["recovery"],
        "recovery_ci95": report["recovery_ci95"],
        "sealed_questions_note": "Scored on held-out questions the fixer never saw.",
        "edit": {
            "tensors_touched": report["tensors_touched"],
            "weights_changed": report["weights_changed"],
            "weights_changed_pct": report["weights_changed_pct"],
        },
        "attempts": {"total": len(attempts), "accepted": len(accepted), "rolled_back": len(attempts) - len(accepted)},
        "best_attempt": None
        if best is None
        else {
            "number": best["number"],
            "outcome": best["outcome"],
            "dev_accuracy": best["dev_metrics"].get("clinical_accuracy"),
            "broken_dev_accuracy": best["dev_metrics"].get("broken_clinical_accuracy"),
            "random_control_accuracy": best["dev_metrics"].get("random_control_accuracy"),
            "reason": best.get("reason", ""),
        },
        "minutes": report["minutes"],
        "tokens": report["tokens"],
        "is_replay": status.get("is_replay", False),
        "gate_note": GATE_NOTE,
        "disclaimer": report["disclaimer"],
    }
