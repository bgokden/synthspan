"""LLM few-shot generation with local models + structured output.

The model returns ``{"sentence": ..., "entities": [{"text", "type"}]}``; we align
each entity's surface text back to exact character offsets ourselves, so the model
never emits (hallucinatable) span indices. Entities we can't locate are dropped.
"""

from __future__ import annotations

import json
import random
from typing import Any

from synthspan.gazetteer import Gazetteer
from synthspan.llm.backends import LLMBackend
from synthspan.types import Example, Span

#: Default JSON schema for constrained decoding.
SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentence": {"type": "string", "maxLength": 200},
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "type": {"type": "string"}},
                "required": ["text", "type"],
            },
        },
    },
    "required": ["sentence", "entities"],
}


def _align(sentence: str, entities: list[dict[str, str]]) -> list[Span]:
    """Locate each entity's exact substring, skipping any not found / overlapping."""
    used: list[tuple[int, int]] = []
    spans: list[Span] = []
    for ent in entities:
        text, label = ent.get("text", ""), ent.get("type", "")
        if not text or not label:
            continue
        start, frm = -1, 0
        while True:
            idx = sentence.find(text, frm)
            if idx == -1:
                break
            if all(not (idx < e and s < idx + len(text)) for s, e in used):
                start = idx
                break
            frm = idx + 1
        if start == -1:
            continue
        used.append((start, start + len(text)))
        spans.append(Span(start, start + len(text), label))
    spans.sort(key=lambda s: s.start)
    return spans


def build_prompt(
    gazetteer: Gazetteer,
    labels: list[str],
    examples: list[Example] | None,
    sample_size: int,
    rng: random.Random,
) -> str:
    """Offer candidate values and let the model *select* a coherent combination."""
    lines = [
        "You generate labeled training sentences for named-entity recognition (NER).",
        "Select a coherent, realistic combination from the candidate values below and",
        "write ONE natural, varied sentence that mentions them.",
        'Respond as JSON: {"sentence": <str>, "entities": [{"text": <exact substring of the sentence>, "type": <LABEL>}]}.',
        "Every entity 'text' MUST be an exact substring of 'sentence'.",
        "",
    ]
    for label in labels:
        values = gazetteer.values(label)
        sample = rng.sample(values, min(sample_size, len(values)))
        lines.append(f"{label} candidates: " + ", ".join(sample))
    if examples:
        lines.append("\nStyle examples:")
        for ex in examples[:3]:
            payload = {"sentence": ex.text, "entities": [{"text": t, "type": l} for t, l in ex.entities()]}
            lines.append(json.dumps(payload, ensure_ascii=False))
    lines.append("\nNow produce one new JSON object.")
    return "\n".join(lines)


def llm_generate(
    gazetteer: Gazetteer,
    backend: LLMBackend,
    n: int,
    labels: list[str] | None = None,
    examples: list[Example] | None = None,
    sample_size: int = 12,
    schema: dict[str, Any] | None = None,
    rng: random.Random | None = None,
    skip_empty: bool = False,
    on_error: str = "skip",
) -> list[Example]:
    """Generate ``n`` examples via a local LLM with structured output.

    Args:
        gazetteer: Candidate values offered to the model to select from.
        backend: Any object with ``complete(prompt, schema) -> dict``.
        n: Number of examples to request.
        labels: Labels to target (default: all in the gazetteer).
        examples: Optional few-shot seeds for style/naturalness.
        sample_size: How many candidate values to show per label per prompt.
        schema: JSON schema for constrained decoding (defaults to SCHEMA).
        rng: Seeded RNG (controls which candidates are shown).
        skip_empty: Drop generations where no entity could be aligned.
        on_error: ``"skip"`` (default) drops a generation the model botches
            (invalid/truncated JSON, transport error); ``"raise"`` re-raises.
    """
    rng = rng or random.Random()
    labels = list(labels or sorted(gazetteer.labels()))
    schema = schema or SCHEMA

    out: list[Example] = []
    for _ in range(n):
        prompt = build_prompt(gazetteer, labels, examples, sample_size, rng)
        try:
            resp = backend.complete(prompt, schema)
            sentence = resp.get("sentence", "")
            spans = _align(sentence, resp.get("entities", []))
        except Exception:
            if on_error == "raise":
                raise
            continue
        if skip_empty and not spans:
            continue
        out.append(Example(sentence, spans))
    return out
