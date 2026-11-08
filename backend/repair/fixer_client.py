"""Client for any OpenAI-compatible chat endpoint with tool calling.

Kimi K2.6 and a self-hosted Qwen 3.8 differ only in configuration. Assistant messages
are returned whole, because extra fields such as `reasoning_content` must be sent back
on later turns for interleaved thinking.
"""
import os
import time
from dataclasses import dataclass

import httpx

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"


class FixerError(RuntimeError):
    """The fixer endpoint couldn't give a usable reply."""


@dataclass(frozen=True)
class FixerConfig:
    base_url: str
    model: str
    api_key: str
    context_tokens: int = 256_000
    request_timeout: float = 600.0
    max_retries: int = 4

    @classmethod
    def from_env(cls) -> "FixerConfig":
        api_key = os.environ.get("FIXER_API_KEY") or os.environ.get("NVIDIA_API")
        if not api_key:
            raise FixerError("Set FIXER_API_KEY or NVIDIA_API")
        return cls(
            base_url=os.environ.get("FIXER_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("FIXER_MODEL", DEFAULT_MODEL),
            api_key=api_key,
            context_tokens=int(os.environ.get("FIXER_CONTEXT_TOKENS", 256_000)),
        )


@dataclass(frozen=True)
class FixerReply:
    message: dict
    prompt_tokens: int
    completion_tokens: int


class FixerClient:
    def __init__(self, config: FixerConfig, transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.config = config
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=config.base_url,
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.request_timeout,
            transport=transport,
        )

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> FixerReply:
        body = {"model": self.config.model, "messages": messages}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        response = self._post_with_retries(body)
        try:
            data = response.json()
            message = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as error:
            raise FixerError(f"Unusable reply from fixer: {response.text[:300]}") from error
        usage = data.get("usage") or {}
        return FixerReply(message, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))

    def _post_with_retries(self, body: dict) -> httpx.Response:
        last_problem = ""
        for attempt in range(self.config.max_retries + 1):
            if attempt:
                self._sleep(min(2**attempt, 60))
            try:
                response = self._http.post("/chat/completions", json=body)
            except httpx.TransportError as error:
                last_problem = f"{type(error).__name__}: {error}"
                continue
            if response.is_success:
                return response
            last_problem = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in RETRY_STATUS:
                break
        raise FixerError(f"Fixer request failed: {last_problem}")
