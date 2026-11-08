"""Turns a repair run's log into a timeline for the demo: what the fixer thought, did and found.

Each entry has seconds since the start, a kind, a short title, and kind-specific detail.
Kinds: stage, thinking, python, forward_pass, scan, attribution, attention, mean_input,
edit_queued, attempt, finish, summary, error.
"""
import json
from pathlib import Path

MAX_TEXT = 600
MAX_CODE = 1200


def _json(text):
    try:
        return json.loads(text) if isinstance(text, str) else text
    except ValueError:
        return None


def _clip(text, limit=MAX_TEXT) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


def _tool_entry(name: str, arguments, output: str) -> tuple[str, str, dict]:
    args = _json(arguments) or {}
    parsed = _json(output)
    failed = isinstance(output, str) and output.startswith("Error")
    if failed:
        return "error", f"{name} failed", {"tool": name, "message": _clip(output)}
    if name == "run_python":
        return "python", "Ran analysis code", {"code": _clip(args.get("code"), MAX_CODE), "output": _clip(output, MAX_CODE)}
    if name == "forward_pass" and parsed:
        result = parsed.get("result", {})
        return "forward_pass", "Scored the model on dev questions", {
            "suite": args.get("suite"),
            "unedited": result.get("unedited"),
            "edited": result.get("edited"),
            "by_category": result.get("by_category"),
        }
    if name == "scan_tensors" and parsed:
        return "scan", "Scanned weight statistics", {"suspects": parsed.get("suspects", [])[:5]}
    if name == "logit_attribution" and parsed:
        result = parsed.get("result", {})
        return "attribution", "Measured which layers push which answer", {
            "answer_letters": result.get("answer_letters"),
            "components": result.get("strongest_components", []),
        }
    if name == "attention":
        return "attention", "Looked at attention weights", {"texts": len(args.get("texts", []))}
    if name == "mean_input" and parsed:
        return "mean_input", "Saved a layer's typical input", {"tensor": args.get("tensor_name"), "norm": parsed.get("norm")}
    if name == "save_edit" and parsed:
        return "edit_queued", "Queued an edit", {"tensor": parsed.get("queued"), "weights_changed": parsed.get("weights_changed")}
    if name == "submit_attempt" and parsed:
        accepted = parsed.get("outcome") == "accepted"
        return "attempt", f"Attempt {parsed.get('number')}: {'accepted' if accepted else 'rolled back'}", parsed
    if name == "finish":
        return "finish", "Finished and reported", {
            "conclusion": args.get("conclusion"),
            "cause": args.get("cause"),
            "located_regions": args.get("located_regions", []),
        }
    return "tool", f"Called {name}", {"output": _clip(output)}


def build_timeline(log_path: Path) -> list[dict]:
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    events = []
    start = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        t = event.get("t")
        if start is None and t is not None:
            start = t
        seconds = round(t - start, 1) if (t is not None and start is not None) else None
        kind = event.get("type")
        if kind == "stage":
            entry = ("stage", f"Stage: {event['stage'].replace('_', ' ')}", {"stage": event["stage"]})
        elif kind == "assistant":
            message = event.get("message") or {}
            thought = message.get("reasoning_content") or message.get("content")
            if not thought:
                continue
            entry = ("thinking", "Fixer's reasoning", {"text": _clip(thought)})
        elif kind == "tool":
            entry = _tool_entry(event.get("name", ""), event.get("arguments"), event.get("output") or "")
        elif kind == "final_summary":
            entry = ("summary", "Fixer's final explanation", {"text": _clip(event.get("text"))})
        elif kind == "fixer_error":
            entry = ("error", "Fixer request failed", {"message": _clip(event.get("error"))})
        else:
            continue
        events.append({"seconds": seconds, "kind": entry[0], "title": entry[1], "detail": entry[2]})
    return events
