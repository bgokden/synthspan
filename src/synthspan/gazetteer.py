"""Typed, optionally-linked entity value store.

A gazetteer holds *records* — each record is a mapping of label -> value that is
internally consistent (e.g. {"CITY": "Amsterdam", "COUNTRY": "Netherlands"}).
Sampling a whole record keeps combinations realistic; sampling a single label
draws from all values seen for that label.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass


@dataclass
class Gazetteer:
    records: list[dict[str, str]]

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("Gazetteer needs at least one record")

    @classmethod
    def from_csv(cls, path: str) -> "Gazetteer":
        """Load linked records from a CSV whose header row names the labels."""
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [
                {k: (v or "").strip() for k, v in row.items() if k}
                for row in csv.DictReader(fh)
            ]
        rows = [r for r in rows if any(r.values())]
        return cls(records=rows)

    @classmethod
    def from_records(cls, records: list[dict[str, str]]) -> "Gazetteer":
        return cls(records=list(records))

    def labels(self) -> set[str]:
        out: set[str] = set()
        for r in self.records:
            out.update(k for k, v in r.items() if v)
        return out

    def values(self, label: str) -> list[str]:
        seen = {r[label] for r in self.records if r.get(label)}
        return sorted(seen)

    def sample_record(self, rng: random.Random) -> dict[str, str]:
        return rng.choice(self.records)

    def sample_value(self, label: str, rng: random.Random) -> str:
        vals = self.values(label)
        if not vals:
            raise KeyError(f"No values for label {label!r}")
        return rng.choice(vals)
