"""Balancing and de-duplication rule checks."""

from __future__ import annotations

from collections import Counter

from synthspan.types import Example


def dedupe(examples: list[Example]) -> list[Example]:
    """Drop examples whose text has already been seen (keep first)."""
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        if ex.text in seen:
            continue
        seen.add(ex.text)
        out.append(ex)
    return out


def label_counts(examples: list[Example]) -> Counter:
    """Count spans per label across all examples."""
    c: Counter = Counter()
    for ex in examples:
        for s in ex.spans:
            c[s.label] += 1
    return c


def value_counts(examples: list[Example], label: str) -> Counter:
    """Count occurrences of each surface value for a given label."""
    c: Counter = Counter()
    for ex in examples:
        for s in ex.spans:
            if s.label == label:
                c[ex.text[s.start : s.end]] += 1
    return c


def cap_per_value(examples: list[Example], label: str, max_count: int) -> list[Example]:
    """Keep at most ``max_count`` examples for any single value of ``label``.

    Prevents a few frequent entities from dominating the dataset.
    """
    if max_count <= 0:
        raise ValueError("max_count must be > 0")
    counts: Counter = Counter()
    out: list[Example] = []
    for ex in examples:
        vals = [ex.text[s.start : s.end] for s in ex.spans if s.label == label]
        if any(counts[v] >= max_count for v in vals):
            continue
        for v in vals:
            counts[v] += 1
        out.append(ex)
    return out
