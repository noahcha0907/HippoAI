"""Exports saved repair runs as static JSON for the website, so visitors need no backend.

Each file is the real API's response, so a page can use the exported files and the live
API interchangeably. Output layout:
    index.json                 runs available, with model, outcome and headline numbers
    models.json, benchmark.json, questions.json
    runs/<repair_id>/status.json, report.json, events.json, compare.json, attribution.json, summary.json

Run in WSL from backend/ (loads the model; needs the GPU):
    python -m eval.export_demo --out demo_export
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HIPPO_SKIP_PIPELINE", "1")  # Build the app here instead of at import.

from fastapi.testclient import TestClient  # noqa: E402

from api.pipeline import Pipeline  # noqa: E402
from api.real import create_app  # noqa: E402


def write(path: Path, response) -> dict:
    if response.status_code != 200:
        raise RuntimeError(f"{path.name}: HTTP {response.status_code} {response.text[:300]}")
    data = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_export"))
    parser.add_argument("--runs", nargs="*", help="Repair IDs to export; default is every saved run")
    parser.add_argument("--questions", type=int, default=6, help="Before-and-after questions per run")
    args = parser.parse_args()

    client = TestClient(create_app(Pipeline()))
    write(args.out / "models.json", client.get("/models"))
    write(args.out / "questions.json", client.get("/questions"))
    write(args.out / "benchmark.json", client.get("/benchmark"))

    benchmark = json.loads((args.out / "benchmark.json").read_text(encoding="utf-8"))
    saved = {row["case"] for row in benchmark["rows"]}
    print(f"Models with saved runs: {sorted(saved) or 'none'}")

    run_ids = args.runs
    if not run_ids:
        from api.pipeline import RUNS

        run_ids = sorted(p.name for p in RUNS.glob("repair-*") if (p / "report.json").exists() and (p / "request.json").exists())

    index = []
    for repair_id in run_ids:
        folder = args.out / "runs" / repair_id
        status = write(folder / "status.json", client.get(f"/repairs/{repair_id}"))
        report = write(folder / "report.json", client.get(f"/repairs/{repair_id}/report"))
        write(folder / "events.json", client.get(f"/repairs/{repair_id}/events"))
        write(folder / "compare.json", client.post("/compare", json={"repair_id": repair_id, "count": args.questions}))
        write(folder / "attribution.json", client.get(f"/repairs/{repair_id}/attribution"))
        write(folder / "summary.json", client.get(f"/repairs/{repair_id}/summary"))
        index.append({
            "id": repair_id,
            "model_id": status["model_id"],
            "complaint": status["complaint"],
            "outcome": report["outcome"],
            "clinical_accuracy": {k: report[k]["clinical_accuracy"] for k in ("broken", "repaired", "clean")},
            "recovery": report["recovery"],
            "attempts": len(status["attempts"]),
            "minutes": report["minutes"],
        })
        print(f"Exported {repair_id}: {status['model_id']}, {report['outcome']}")
    (args.out / "index.json").write_text(json.dumps({"mock": False, "runs": index}, indent=2), encoding="utf-8")
    print(f"Wrote {len(index)} runs to {args.out}")


if __name__ == "__main__":
    main()
