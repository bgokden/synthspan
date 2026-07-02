import random

from synthspan import Gazetteer
from synthspan.llm import FakeBackend, build_prompt, llm_generate

GAZ = Gazetteer.from_records(
    [
        {"CITY": "Amsterdam", "COUNTRY": "Netherlands"},
        {"CITY": "Paris", "COUNTRY": "France"},
    ]
)


def test_llm_generate_aligns_spans():
    fake = FakeBackend(
        [
            {"sentence": "Amsterdam is lovely in autumn.", "entities": [{"text": "Amsterdam", "type": "CITY"}]},
            {"sentence": "I love Paris, France.", "entities": [{"text": "Paris", "type": "CITY"}, {"text": "France", "type": "COUNTRY"}]},
        ]
    )
    exs = llm_generate(GAZ, fake, n=2, rng=random.Random(0))
    assert exs[0].entities() == [("Amsterdam", "CITY")]
    assert set(exs[1].entities()) == {("Paris", "CITY"), ("France", "COUNTRY")}
    # every aligned span points at the exact surface text
    for ex in exs:
        for s in ex.spans:
            assert ex.text[s.start : s.end]


def test_llm_skips_unlocatable_entities():
    fake = FakeBackend(
        [{"sentence": "A plain sentence.", "entities": [{"text": "Zzzville", "type": "CITY"}]}]
    )
    exs = llm_generate(GAZ, fake, n=1, rng=random.Random(0))
    assert exs[0].spans == []  # entity text not present -> dropped, no bad span


def test_llm_skip_empty():
    fake = FakeBackend([{"sentence": "Nothing here.", "entities": []}])
    assert llm_generate(GAZ, fake, n=1, skip_empty=True) == []


def test_build_prompt_offers_candidates():
    prompt = build_prompt(GAZ, ["CITY", "COUNTRY"], None, sample_size=12, rng=random.Random(0))
    assert "CITY candidates:" in prompt and "COUNTRY candidates:" in prompt
    assert "Amsterdam" in prompt
