"""Real-model integration tests (opt-in; skipped by default and in the fast job).

- llama-cpp: runs in CI with a tiny downloaded GGUF (SYNTHSPAN_GGUF set by the job).
- Ollama: runs locally when you set SYNTHSPAN_OLLAMA_TEST=1 with Ollama running.

Both exercise the *real* structured-output round-trip: the model emits JSON under
a JSON-schema grammar, and we align entities to exact spans.
"""

import os
import random

import pytest

from synthspan import Gazetteer
from synthspan.llm import llm_generate

GAZ = Gazetteer.from_records(
    [
        {"CITY": "Amsterdam", "COUNTRY": "Netherlands"},
        {"CITY": "Paris", "COUNTRY": "France"},
        {"CITY": "Tokyo", "COUNTRY": "Japan"},
    ]
)


def _assert_pipeline(examples):
    # Validates the real backend -> parse -> align path ran cleanly. Quality is
    # covered by the offline FakeBackend tests; a tiny CI model may yield few/no
    # rows (bad generations are skipped, not crashed), so we don't require >= 1.
    assert isinstance(examples, list)
    for ex in examples:
        assert isinstance(ex.text, str)
        for s in ex.spans:  # any aligned span must point at the exact surface text
            assert ex.text[s.start : s.end]


@pytest.mark.integration
def test_llama_cpp_structured_output():
    gguf = os.environ.get("SYNTHSPAN_GGUF")
    if not gguf or not os.path.exists(gguf):
        pytest.skip("set SYNTHSPAN_GGUF to a local .gguf to run")
    pytest.importorskip("llama_cpp")
    from synthspan.llm import LlamaCppBackend

    backend = LlamaCppBackend(model_path=gguf, temperature=0.7, n_ctx=2048)
    exs = llm_generate(GAZ, backend, n=2, rng=random.Random(0))
    _assert_pipeline(exs)


@pytest.mark.integration
def test_ollama_structured_output():
    if os.environ.get("SYNTHSPAN_OLLAMA_TEST") != "1":
        pytest.skip("set SYNTHSPAN_OLLAMA_TEST=1 (with Ollama running) to run")
    from synthspan.llm import OllamaBackend

    model = os.environ.get("SYNTHSPAN_OLLAMA_MODEL", "llama3.1")
    backend = OllamaBackend(model=model, temperature=0.7)
    exs = llm_generate(GAZ, backend, n=2, rng=random.Random(0))
    _assert_pipeline(exs)
