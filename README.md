# synthspan

[![CI](https://github.com/bgokden/synthspan/actions/workflows/ci.yml/badge.svg)](https://github.com/bgokden/synthspan/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen)

Generate **synthetic labeled data for NER / token classification** — cheaply, at
scale, and with exact spans. Feed it a small **gazetteer** of entities and a few
**templates**, and it produces thousands of labeled examples, balanced and
optionally noised with realistic typos.

Zero runtime dependencies. Deterministic with a seed. Exports to JSONL, CoNLL/BIO,
and spaCy.

> This is the data-generation approach behind the multilingual place-extractor
> models at [huggingface.co/Berk](https://huggingface.co/Berk), served via
> [place-extractor-mcp](https://github.com/bgokden/place-extractor-mcp).

## Why

Labeled NER data is expensive to annotate. But for many entity types you already
have a **list** (cities, products, drug names, tickers…) and know the **shapes**
of sentences they appear in. `synthspan` turns those into training data with
guaranteed-correct spans — no manual labeling.

## Install

```bash
pip install synthspan            # or: pip install -e . from source
```

## CLI

```bash
synthspan generate \
  --entities examples/entities.csv \
  --templates examples/templates.txt \
  -n 10000 --balanced --dedupe \
  --typo-rate 0.05 --seed 42 \
  --format jsonl --out data.jsonl
```

- **`--entities`** — CSV whose header names the labels; each row is a *linked*
  record so combinations stay consistent (`Amsterdam,Netherlands`).
- **`--templates`** — one per line, with `{LABEL}` slots.
- **`--typo-rate`** — per-word typo probability; spans are recomputed so labels
  stay correct (add `--typo-entities` to also noise the entities themselves).
- **`--balanced`** — even coverage of templates and records.

Output formats: `jsonl` (default), `conll` (BIO), `spacy`.

## Library

```python
import random
from synthspan import Gazetteer, Template, generate, augment, dedupe, to_jsonl

gaz = Gazetteer.from_records([
    {"CITY": "Amsterdam", "COUNTRY": "Netherlands"},
    {"CITY": "Paris", "COUNTRY": "France"},
])
templates = [Template("I flew from {CITY} to {COUNTRY}.")]

rng = random.Random(42)
data = generate(gaz, templates, n=5000, rng=rng, balanced=True)
data = dedupe(data)
data = augment(data, rate=0.05, rng=rng)   # spans stay correct

print(data[0].entities())   # [('Amsterdam', 'CITY'), ('Netherlands', 'COUNTRY')]
open("data.jsonl", "w").write(to_jsonl(data))
```

Each example is `{"text": ..., "spans": [{"start", "end", "label", "text"}, ...]}`.

## How it works

1. **Gazetteer** — typed, linked entity records (keeps `(city, country)` consistent).
2. **Templates** — slotted sentences; filling records exact character spans.
3. **Balance** — even coverage, de-duplication, per-value caps.
4. **Augment** — realistic keyboard-aware typos, with span-preserving recomputation.
5. **Write** — JSONL / CoNLL-BIO / spaCy.

## Roadmap

- **LLM few-shot mode** — pass example records and let an LLM generate more natural
  variations (planned as an optional extra).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © [Berk Gökden](https://berkgokden.com)
