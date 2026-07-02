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
    linked: bool = True,
) -> list[Example]:
    """Generate ``n`` examples.

    Args:
        gazetteer: Source of entity values.
        templates: Templates to fill.
        n: Number of examples to produce.
        rng: Seeded RNG for reproducibility.
        balanced: Cycle deterministically through templates (and, when
            ``linked``, records) for even coverage instead of sampling randomly.
        linked: When ``True`` (default) every slot in a template is filled from a
            single record, so co-located mentions stay consistent
            (``"{CITY}, {COUNTRY}"`` → *Amsterdam, Netherlands*). When ``False``
            each slot is sampled independently from its own value pool — right for
            relational templates (``"from {CITY} to {COUNTRY}"`` → *Amsterdam →
            Japan*) and for combinatorial volume (cities × countries).
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if not templates:
        raise ValueError("need at least one template")
    rng = rng or random.Random()

    out: list[Example] = []
    for i in range(n):
        template = templates[i % len(templates)] if balanced else rng.choice(templates)
        if linked:
            record = (
                gazetteer.records[i % len(gazetteer.records)]
                if balanced
                else gazetteer.sample_record(rng)
            )
        else:
            # dict.fromkeys keeps unique slot labels in first-seen order
            record = {
                label: gazetteer.sample_value(label, rng)
                for label in dict.fromkeys(template.slots())
            }
        out.append(template.fill(record, gazetteer, rng))
    return out
