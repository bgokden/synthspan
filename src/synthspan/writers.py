"""Serialize examples to common training formats: JSONL, CoNLL/BIO, spaCy."""

from __future__ import annotations

import json
import re

from synthspan.types import Example

# Split into word tokens and individual punctuation marks so entities adjacent to
# punctuation (e.g. "(Netherlands)") still align to their spans in BIO output.
_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def to_jsonl(examples: list[Example]) -> str:
    return "\n".join(json.dumps(ex.to_dict(), ensure_ascii=False) for ex in examples)


def to_spacy(examples: list[Example]) -> str:
    """spaCy offsets format: [text, {"entities": [[start, end, label], ...]}]."""
    data = [
        [ex.text, {"entities": [[s.start, s.end, s.label] for s in ex.spans]}]
        for ex in examples
    ]
    return json.dumps(data, ensure_ascii=False, indent=2)


def to_conll(examples: list[Example]) -> str:
    """CoNLL-style BIO: one `token TAG` per line, blank line between examples."""
    blocks: list[str] = []
    for ex in examples:
        lines: list[str] = []
        for m in _TOKEN.finditer(ex.text):
            ts, te = m.start(), m.end()
            tag = "O"
            for s in ex.spans:
                if s.start <= ts and te <= s.end:
                    tag = ("B-" if ts == s.start else "I-") + s.label
                    break
            lines.append(f"{m.group()} {tag}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


_WRITERS = {"jsonl": to_jsonl, "conll": to_conll, "spacy": to_spacy}


def write(examples: list[Example], path: str, fmt: str = "jsonl") -> None:
    if fmt not in _WRITERS:
        raise ValueError(f"Unknown format {fmt!r}; choose from {sorted(_WRITERS)}")
    text = _WRITERS[fmt](examples)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
