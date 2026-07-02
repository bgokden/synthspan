"""Turn a gazetteer + templates into labeled examples."""

from __future__ import annotations

import random

from synthspan.gazetteer import Gazetteer
from synthspan.template import Template
from synthspan.types import Example


def generate(
    gazetteer: Gazetteer,
    templates: list[Template],
    n: int,
    rng: random.Random | None = None,
    balanced: bool = False,
) -> list[Example]:
    """Generate ``n`` examples.

    With ``balanced=True`` the generator cycles deterministically through
    templates and records for even coverage; otherwise it samples randomly.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not templates:
        raise ValueError("need at least one template")
    rng = rng or random.Random()

    out: list[Example] = []
    for i in range(n):
        if balanced:
            template = templates[i % len(templates)]
            record = gazetteer.records[i % len(gazetteer.records)]
        else:
            template = rng.choice(templates)
            record = gazetteer.sample_record(rng)
        out.append(template.fill(record, gazetteer, rng))
    return out
