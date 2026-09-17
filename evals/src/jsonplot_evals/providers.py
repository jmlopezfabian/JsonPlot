"""The models, behind one call: a system prompt and a user turn in, text out."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

OLLAMA = "http://localhost:11434/api/chat"


class ProviderUnavailable(RuntimeError):
    """The model could not be reached. Distinct from a bad answer, which is data."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    seconds: float


class Provider(Protocol):
    name: str
    model: str
    #: Everything besides the prompts that shapes the answer. Part of the
    #: recording key, so changing a parameter is a cache miss.
    params: dict

    def complete(self, system: str, user: str) -> Completion: ...


class Ollama:
    name = "ollama"

    #: Ollama truncates a prompt longer than the context window without saying
    #: so, which stops measuring the briefing and starts measuring its first N
    #: tokens. The default is 4096; the briefing is ~3.9k tokens, so one model
    #: fit by two hundred tokens and the next one silently did not — 110 of its
    #: 144 answers came back empty. Set it high enough that the prompt is what
    #: was written, and record it in `params` so a run with a different window
    #: is a different condition rather than the same number.
    CONTEXT = 8192

    def __init__(self, model: str, url: str = OLLAMA, timeout: int = 180,
                 num_ctx: int = CONTEXT):
        self.model = model
        self.url = url
        self.timeout = timeout
        self.params = {"format": "json", "temperature": 0, "num_ctx": num_ctx}

    def complete(self, system: str, user: str) -> Completion:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": self.params["format"],
            "options": {"temperature": self.params["temperature"],
                        "num_ctx": self.params["num_ctx"]},
        }).encode()
        req = urllib.request.Request(self.url, payload,
                                     {"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProviderUnavailable(f"ollama unreachable at {self.url}: {exc}") from exc
        return Completion(
            text=body["message"]["content"],
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            seconds=time.perf_counter() - started,
        )


def for_model(model: str) -> Provider:
    """The provider that serves `model`. Anything unclaimed goes to ollama."""
    if model.startswith("claude-"):
        raise ProviderUnavailable(f"no provider for {model!r} yet")
    return Ollama(model)
