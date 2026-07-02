"""Core data types: labeled spans and examples."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    """A labeled character span [start, end) within an example's text."""

    start: int
    end: int
    label: str


@dataclass
class Example:
    """A generated text plus its labeled entity spans."""

    text: str
    spans: list[Span]

    def entities(self) -> list[tuple[str, str]]:
        """Return (surface_text, label) for each span."""
        return [(self.text[s.start : s.end], s.label) for s in self.spans]

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "spans": [
                {"start": s.start, "end": s.end, "label": s.label, "text": self.text[s.start : s.end]}
                for s in self.spans
            ],
        }
