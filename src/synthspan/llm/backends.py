"""Pluggable local-LLM backends with structured (JSON-schema) output.

A backend just needs a ``complete(prompt, schema) -> dict`` method that returns a
dict validating against ``schema``. Ship-with reference backend is Ollama (local,
zero extra Python deps). A llama-cpp-python backend is provided for GGUF grammars
(install ``synthspan[llama-cpp]``). Bring your own for vLLM / OpenAI-compatible
endpoints by implementing the same tiny protocol.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]: ...


class OllamaBackend:
    """Call a local Ollama server with JSON-schema-constrained output.

    Requires Ollama running (https://ollama.com) with the model pulled, e.g.
    ``ollama pull llama3.1``. No Python dependency — uses the standard library.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.8,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,  # Ollama constrains output to this JSON schema
            "options": {"temperature": self.temperature},
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return json.loads(payload["response"])


class LlamaCppBackend:
    """GGUF models via llama-cpp-python with grammar-constrained JSON output.

    Install ``synthspan[llama-cpp]``. Runs in-process (Metal on macOS).
    """

    def __init__(self, model_path: str, temperature: float = 0.8, n_ctx: int = 4096, **kwargs: Any) -> None:
        from llama_cpp import Llama  # lazy import — optional dependency

        self._llm = Llama(model_path=model_path, n_ctx=n_ctx, verbose=False, **kwargs)
        self.temperature = temperature

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        resp = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object", "schema": schema},
            temperature=self.temperature,
        )
        return json.loads(resp["choices"][0]["message"]["content"])


class FakeBackend:
    """Deterministic backend for tests/offline use — cycles canned responses."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        if not responses:
            raise ValueError("FakeBackend needs at least one response")
        self.responses = list(responses)
        self.calls = 0

    def complete(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        r = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return r
