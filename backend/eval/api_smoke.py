"""One real repair through the real API in-process: fixer, sandbox, GPU, sealed evaluation, report.

Run in WSL from backend/:  python -m eval.api_smoke --model c1-a --minutes 15 --passes 8
"""
import argparse
import json
import time

from fastapi.testclient import TestClient

from api.pipeline import Pipeline
from api.real import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="c1-a")
    parser.add_argument("--minutes", type=int, default=15)
    parser.add_argument("--passes", type=int, default=8)
    args = parser.parse_args()

    client = TestClient(create_app(Pipeline()))
    body = {
        "model_id": args.model,
        "complaint": "It's giving wrong answers on clinical triage and dosing questions.",
        "budget": {"max_minutes": args.minutes, "max_forward_passes": args.passes},
    }
    repair_id = client.post("/repairs", json=body).json()["id"]
    last = None
    while True:
        status = client.get(f"/repairs/{repair_id}").json()
        line = (status["stage"], status["forward_passes_used"], len(status["attempts"]))
        if line != last:
            print(f"[{status['elapsed_minutes']:.1f} min] stage={line[0]} passes={line[1]} attempts={line[2]}", flush=True)
            last = line
        if status["stage"] in ("awaiting_decision", "failed"):
            break
        time.sleep(5)
    if status["stage"] == "failed":
        print("FAILED:", status["error"])
        return
    for attempt in status["attempts"]:
        print("ATTEMPT", json.dumps(attempt))
    report = client.get(f"/repairs/{repair_id}/report").json()
    keys = ["outcome", "cause", "broken", "repaired", "clean", "recovery", "recovery_ci95", "collateral_loss",
            "weights_changed_pct", "tensors_touched", "minutes", "forward_passes", "tokens", "remaining_failures"]
    print("REPORT", json.dumps({k: report[k] for k in keys}, indent=1))
    prompt = "A 55-year-old has crushing chest pain and sweating for 30 minutes. What should they do?"
    for version in ("broken", "repaired", "clean"):
        answer = client.post("/generate", json={"model_version": version, "prompt": prompt, "repair_id": repair_id})
        print(f"ANSWER {version}:", answer.json().get("text", answer.text)[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
