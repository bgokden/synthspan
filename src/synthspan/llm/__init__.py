"""Optional LLM few-shot generation with local models + structured output.

    from synthspan.llm import OllamaBackend, llm_generate

Requires a local model runtime (Ollama by default; llama-cpp-python optional).
Not imported by the package root, so the core stays dependency-free.
"""

from synthspan.llm.backends import (
    FakeBackend,
    LlamaCppBackend,
    LLMBackend,
    OllamaBackend,
)
from synthspan.llm.generate import SCHEMA, build_prompt, llm_generate

__all__ = [
    "LLMBackend",
    "OllamaBackend",
    "LlamaCppBackend",
    "FakeBackend",
    "llm_generate",
    "build_prompt",
    "SCHEMA",
]
