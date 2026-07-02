import json
import random

from synthspan import (
    Gazetteer,
    Template,
    augment,
    dedupe,
    generate,
    label_counts,
    to_conll,
    to_jsonl,
    to_spacy,
)
from synthspan.balance import cap_per_value

GAZ = Gazetteer.from_records(
    [
        {"CITY": "Amsterdam", "COUNTRY": "Netherlands"},
        {"CITY": "Paris", "COUNTRY": "France"},
        {"CITY": "İstanbul", "COUNTRY": "Türkiye"},
    ]
)
TEMPLATES = [Template("I flew from {CITY} to {COUNTRY}.")]


def _spans_are_correct(ex, expected_by_label=None):
    for s in ex.spans:
        surface = ex.text[s.start : s.end]
        assert surface, "span must be non-empty"
        if expected_by_label and s.label in expected_by_label:
            assert surface == expected_by_label[s.label]


def test_template_fill_exact_spans():
    ex = TEMPLATES[0].fill({"CITY": "Amsterdam", "COUNTRY": "Netherlands"})
    assert ex.text == "I flew from Amsterdam to Netherlands."
    _spans_are_correct(ex, {"CITY": "Amsterdam", "COUNTRY": "Netherlands"})
    assert {lbl for _, lbl in ex.entities()} == {"CITY", "COUNTRY"}


def test_linked_records_stay_consistent():
    rng = random.Random(1)
    for ex in generate(GAZ, TEMPLATES, 50, rng=rng):
        city = next(t for t, l in ex.entities() if l == "CITY")
        country = next(t for t, l in ex.entities() if l == "COUNTRY")
        assert {"CITY": city, "COUNTRY": country} in GAZ.records


def test_generate_is_deterministic():
    a = generate(GAZ, TEMPLATES, 20, rng=random.Random(42))
    b = generate(GAZ, TEMPLATES, 20, rng=random.Random(42))
    assert [e.text for e in a] == [e.text for e in b]


def test_balanced_covers_all_records():
    exs = generate(GAZ, TEMPLATES, len(GAZ.records), rng=random.Random(0), balanced=True)
    cities = {next(t for t, l in e.entities() if l == "CITY") for e in exs}
    assert cities == {"Amsterdam", "Paris", "İstanbul"}


def test_typos_preserve_entity_spans():
    rng = random.Random(7)
    exs = generate(GAZ, TEMPLATES, 40, rng=rng)
    noisy = augment(exs, rate=0.8, rng=rng)  # high rate to stress span math
    assert len(noisy) == len(exs)
    for orig, ex in zip(exs, noisy):
        orig_ents = {l: t for t, l in orig.entities()}
        for s in ex.spans:
            # entities untouched by default -> surface text must still match
            assert ex.text[s.start : s.end] == orig_ents[s.label]


def test_typos_can_change_text():
    rng = random.Random(3)
    exs = generate(GAZ, TEMPLATES, 30, rng=rng)
    noisy = augment(exs, rate=0.9, rng=rng)
    assert any(a.text != b.text for a, b in zip(exs, noisy))


def test_dedupe():
    exs = generate(GAZ, TEMPLATES, 200, rng=random.Random(5))
    d = dedupe(exs)
    assert len({e.text for e in d}) == len(d)
    assert len(d) <= len(exs)


def test_cap_per_value():
    exs = generate(GAZ, TEMPLATES, 300, rng=random.Random(9))
    capped = cap_per_value(exs, "CITY", 5)
    from collections import Counter

    c = Counter(t for e in capped for t, l in e.entities() if l == "CITY")
    assert all(v <= 5 for v in c.values())


def test_jsonl_roundtrip():
    exs = generate(GAZ, TEMPLATES, 5, rng=random.Random(0))
    lines = to_jsonl(exs).splitlines()
    assert len(lines) == 5
    obj = json.loads(lines[0])
    assert obj["text"] and obj["spans"][0]["label"] in {"CITY", "COUNTRY"}
    # span offsets in JSON match the surface text
    for sp in obj["spans"]:
        assert obj["text"][sp["start"] : sp["end"]] == sp["text"]


def test_conll_bio():
    ex = Template("from {CITY} today").fill({"CITY": "New York"})
    out = to_conll([ex])
    tags = [line.split()[-1] for line in out.strip().splitlines()]
    assert "B-CITY" in tags and "I-CITY" in tags  # multi-token entity


def test_spacy_format():
    exs = generate(GAZ, TEMPLATES, 3, rng=random.Random(0))
    data = json.loads(to_spacy(exs))
    assert len(data) == 3
    text, ann = data[0]
    start, end, label = ann["entities"][0]
    assert text[start:end] and label in {"CITY", "COUNTRY"}


def test_label_counts():
    exs = generate(GAZ, TEMPLATES, 10, rng=random.Random(0))
    c = label_counts(exs)
    assert c["CITY"] == 10 and c["COUNTRY"] == 10
