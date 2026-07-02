"""End-to-end quickstart: gazetteer + templates -> labeled NER dataset.

Run from the repo root:

    python examples/quickstart.py

Writes data.jsonl (offsets) and data.conll (BIO). No dependencies, no model.
"""

import random

from synthspan import (
    Gazetteer,
    Template,
    apply,
    dedupe,
    generate,
    label_counts,
    ocr,
    random_case,
    to_conll,
    typos,
    write,
)

# 1) A gazetteer of linked (city -> its own country) records.
gaz = Gazetteer.from_csv("examples/entities.csv")

# 2) CO-LOCATION templates (city is IN the country) -> pairs stay consistent.
templates = [
    Template("{CITY} is a city in {COUNTRY}."),
    Template("We spent a few days in {CITY}, {COUNTRY}."),
    Template("The conference will be held in {CITY} ({COUNTRY}) this year."),
]

rng = random.Random(42)

# 3) Generate (linked by default) -> every sentence is geographically correct.
data = generate(gaz, templates, n=2000, rng=rng)

# 4) Augment: compose several noisers; spans are recomputed so labels stay exact.
data = apply(data, [typos(0.05), random_case(0.08), ocr(0.03)], rng=rng)

# 5) Drop exact duplicates (most survive thanks to the typos).
data = dedupe(data)

# 6) Inspect and write.
print(f"{len(data)} examples")
print("labels:", dict(label_counts(data)))
print("sample:", data[0].text, "->", data[0].entities())

write(data, "data.jsonl", fmt="jsonl")
with open("data.conll", "w", encoding="utf-8") as fh:
    fh.write(to_conll(data))
print("wrote data.jsonl and data.conll")
