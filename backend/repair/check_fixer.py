"""Sends one tool-call test to the configured fixer. Never prints the API key.

Run from backend/:  python -m repair.check_fixer
"""
import json
import os
import time
from pathlib import Path

from repair.fixer_client import FixerClient, FixerConfig, FixerError

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "Add two numbers.",
        "parameters": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
}


def load_env_file(path: Path) -> list[str]:
    """Load KEY=value lines into the environment and return only the key names."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip().strip("'\"")
            names.append(key.strip())
    return names


def main() -> None:
    print("Keys in .env:", load_env_file(ENV_FILE))
    try:
        config = FixerConfig.from_env()
        print("Endpoint:", config.base_url, "| Model:", config.model)
        client = FixerClient(config)
        started = time.time()
        messages = [{"role": "user", "content": "Use the add_numbers tool to add 17 and 25."}]
        reply = client.chat(messages, [TOOL])
    except FixerError as error:
        print("FAILED:", str(error).replace(os.environ.get("FIXER_API_KEY") or "\0", "[key]"))
        raise SystemExit(1)
    calls = reply.message.get("tool_calls") or []
    print(f"Replied in {time.time() - started:.1f}s; tokens {reply.prompt_tokens} in, {reply.completion_tokens} out")
    print("Tool calls:", json.dumps([c.get("function") for c in calls]))
    if not calls:
        print("NO TOOL CALL. Reply text:", (reply.message.get("content") or "")[:300])
        raise SystemExit(1)
    print("Tool calling works")


if __name__ == "__main__":
    main()
