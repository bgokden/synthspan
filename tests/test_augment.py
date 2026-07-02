import random

import pytest

from synthspan import Gazetteer, Template, apply, augment, generate, ocr, punctuation, random_case, typos

GAZ = Gazetteer.from_records(
    [
        {"CITY": "Amsterdam", "COUNTRY": "Netherlands"},
        {"CITY": "İstanbul", "COUNTRY": "Türkiye"},
    ]
)
TEMPLATES = [Template("We spent time in {CITY}, {COUNTRY} last year, it was great.")]


def _examples(n=30, seed=0):
    return generate(GAZ, TEMPLATES, n, rng=random.Random(seed))


ALL_AUGMENTERS = [
    ("typos", typos(0.9)),
    ("random_case", random_case(0.9)),
    ("punctuation", punctuation(0.9)),
    ("ocr", ocr(0.9)),
]


@pytest.mark.parametrize("name,aug", ALL_AUGMENTERS)
def test_spans_preserved_under_each_augmenter(name, aug):
    exs = _examples()
    noisy = apply(exs, [aug], rng=random.Random(1))
    assert len(noisy) == len(exs)
    for orig, ex in zip(exs, noisy):
        want = {l: t for t, l in orig.entities()}
        # default: entities untouched -> exact surface text still present at the span
        for s in ex.spans:
            assert ex.text[s.start : s.end] == want[s.label]


@pytest.mark.parametrize("name,aug", ALL_AUGMENTERS)
def test_each_augmenter_changes_nonentity_text(name, aug):
    # a long entity-free string so the augmenter has room to act
    from synthspan.types import Example

    # letters (for typos/case/ocr) AND several punctuation marks (for punctuation)
    base = [Example("Wow, this is great! Really? Yes; indeed: totally fine.", [])]
    out = apply(base, [aug], rng=random.Random(2))
    assert out[0].text != base[0].text


def test_composition_preserves_spans():
    exs = _examples()
    noisy = apply(
        exs,
        [typos(0.5), random_case(0.5), ocr(0.3), punctuation(0.5)],
        rng=random.Random(3),
    )
    for orig, ex in zip(exs, noisy):
        want = {l: t for t, l in orig.entities()}
        for s in ex.spans:
            assert ex.text[s.start : s.end] == want[s.label]


def test_augment_entities_keeps_labels_and_nonempty_spans():
    exs = _examples()
    noisy = apply(exs, [random_case(1.0)], rng=random.Random(4), augment_entities=True)
    for ex in noisy:
        assert {s.label for s in ex.spans} == {"CITY", "COUNTRY"}
        for s in ex.spans:
            assert s.end > s.start  # span still non-empty


def test_ocr_produces_digits():
    from synthspan.types import Example

    out = apply([Example("look at all these lovely words", [])], [ocr(1.0)], rng=random.Random(5))
    assert any(ch.isdigit() for ch in out[0].text)


def test_random_case_flips_some_letters():
    from synthspan.types import Example

    base = "the quick brown fox"
    out = apply([Example(base, [])], [random_case(1.0)], rng=random.Random(6))
    assert out[0].text != base and out[0].text.lower() == base


def test_backward_compatible_augment():
    exs = _examples()
    noisy = augment(exs, rate=0.8, rng=random.Random(7))
    for orig, ex in zip(exs, noisy):
        want = {l: t for t, l in orig.entities()}
        for s in ex.spans:
            assert ex.text[s.start : s.end] == want[s.label]


def test_zero_rate_is_noop():
    exs = _examples(5)
    out = augment(exs, rate=0.0, rng=random.Random(8))
    assert [e.text for e in out] == [e.text for e in exs]
