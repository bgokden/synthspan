"""Slotted templates that fill to text plus exact labeled spans.

A template is a string with `{LABEL}` slots, e.g.::

    "I flew from {CITY} to {COUNTRY}."

Filling records the precise character offsets of every substituted value, so the
output spans are always correct regardless of value length.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from synthspan.gazetteer import Gazetteer
from synthspan.types import Example, Span

_SLOT = re.compile(r"\{(\w+)\}")


@dataclass
class Template:
    raw: str

    def slots(self) -> list[str]:
        return _SLOT.findall(self.raw)

    def fill(
        self,
        record: dict[str, str],
        gazetteer: Gazetteer | None = None,
        rng: random.Random | None = None,
    ) -> Example:
        """Fill the template, recording a Span for each slot."""
        parts: list[str] = []
        spans: list[Span] = []
        cursor = 0
        pos = 0
        for m in _SLOT.finditer(self.raw):
            label = m.group(1)
            # literal text before the slot
            literal = self.raw[cursor : m.start()]
            parts.append(literal)
            pos += len(literal)

            value = record.get(label)
            if not value:
                if gazetteer is None or rng is None:
                    raise KeyError(
                        f"No value for {label!r} in record and no gazetteer/rng to sample"
                    )
                value = gazetteer.sample_value(label, rng)

            spans.append(Span(pos, pos + len(value), label))
            parts.append(value)
            pos += len(value)
            cursor = m.end()

        tail = self.raw[cursor:]
        parts.append(tail)
        return Example(text="".join(parts), spans=spans)

    @staticmethod
    def load(path: str) -> list["Template"]:
        """Load templates from a file (one per line; blank / `#` lines ignored)."""
        out: list[Template] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                out.append(Template(line))
        if not out:
            raise ValueError(f"No templates found in {path}")
        return out
