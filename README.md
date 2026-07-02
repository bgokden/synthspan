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

# A gazetteer of linked (city -> its own country) records. Bring your own list;
# the more pairs, the more variety. 12 rows ship in examples/entities.csv.
gaz = Gazetteer.from_csv("examples/entities.csv")

# CO-LOCATION templates: the city is IN the country, so linked pairs stay correct.
templates = [
    Template("{CITY} is a city in {COUNTRY}."),
    Template("We spent a few days in {CITY}, {COUNTRY}."),
    Template("The conference will be held in {CITY} ({COUNTRY}) this year."),
]

rng = random.Random(42)
data = generate(gaz, templates, n=2000, rng=rng)  # linked (default): city + ITS country
data = augment(data, rate=0.06, rng=rng)           # typos add surface variety; spans stay correct
data = dedupe(data)                                # drop exact duplicates (post-typo)

print(data[0].entities())   # [('Amsterdam', 'CITY'), ('Netherlands', 'COUNTRY')]
open("data.jsonl", "w").write(to_jsonl(data))
```

Each example is `{"text": ..., "spans": [{"start", "end", "label", "text"}, ...]}`.
Distinct base sentences ≈ records × templates; **typos multiply the surface
variety**, and a large gazetteer scales it up — that's how a short list becomes a
lot of labeled data.

### Linked vs. independent slots

- **`linked=True` (default)** — every slot in a template is filled from one
  record, so co-located mentions stay consistent. Use for templates where the
  city is *in* that country: `"{CITY}, {COUNTRY}"` → *Amsterdam, Netherlands*.
  Unique outputs ≈ records × templates.
- **`linked=False`** — each slot is sampled independently. Use for relational
  templates where the places differ: `"from {CITY} to {COUNTRY}"` → *Amsterdam →
  Japan*. Unique outputs ≈ cities × countries × templates (lots of cheap data).

Pick the mode that matches your template's meaning. On the CLI: add
`--independent` for the second mode. Use `dedupe()` / `--dedupe` when you want
only distinct texts (note: after typo augmentation most texts are already
unique).

## LLM few-shot mode (optional)

Want more natural variety than templates? Use a **local** model with **structured
output**. The model *selects* a coherent combination from your gazetteer and
writes a sentence; synthspan aligns the entities back to exact spans — **the model
never emits offsets**, so spans can't be hallucinated.

Default backend is [Ollama](https://ollama.com) (no Python dependency — just an
HTTP call to localhost):

```bash
ollama pull llama3.1
```

```python
import random
from synthspan import Gazetteer
from synthspan.llm import OllamaBackend, llm_generate

gaz = Gazetteer.from_csv("examples/entities.csv")
backend = OllamaBackend(model="llama3.1")   # local, JSON-schema-constrained output

data = llm_generate(gaz, backend, n=500, rng=random.Random(0), skip_empty=True)
print(data[0].text, data[0].entities())
```

Backends: `OllamaBackend` (local HTTP, zero deps) · `LlamaCppBackend` (GGUF grammars,
`pip install synthspan[llama-cpp]`) · or implement `complete(prompt, schema) -> dict`
for vLLM / any OpenAI-compatible endpoint. `FakeBackend` ships for offline tests.

## Diversity balancing (optional)

Count-based balancing (`cap_per_value`) evens out *values*; semantic balancing
evens out *phrasings*. Bring any local embedder and cluster — pure-Python k-means,
zero dependencies:

```python
from synthspan import cluster_balance

embed = lambda text: my_local_model.encode(text)   # -> list[float]
balanced = cluster_balance(data, embed, k=8)        # even coverage across 8 clusters
```

## How it works

1. **Gazetteer** — typed, linked entity records (keeps `(city, country)` consistent).
2. **Templates** — slotted sentences; filling records exact character spans.
3. **Balance** — even coverage, de-duplication, per-value caps.
4. **Augment** — realistic keyboard-aware typos, with span-preserving recomputation.
5. **Write** — JSONL / CoNLL-BIO / spaCy.

## Background: what's a gazetteer?

A **gazetteer** is a curated list of known entity names — originally a
geographical directory of place names, and in NLP any dictionary of entity
surface forms (cities, drugs, companies, tickers, genes…). Gazetteers are a
long-standing signal for **named entity recognition (NER)**:

- [Towards Improving Neural Named Entity Recognition with Gazetteers](https://aclanthology.org/P19-1524/) — Liu et al., ACL 2019
- [Gazetteer-Enhanced Attentive Neural Networks for NER](https://aclanthology.org/D19-1646/) — Lin et al., EMNLP-IJCNLP 2019
- [Soft Gazetteers for Low-Resource Named Entity Recognition](https://arxiv.org/abs/2005.01866) — Rijhwani et al., ACL 2020
- [Gazetteer — Wikipedia](https://en.wikipedia.org/wiki/Gazetteer)

Those works feed a gazetteer *into* the model. `synthspan` uses it the other way
around: a gazetteer + templates *generate* the labeled training data itself — so
you can train a standard token classifier without hand annotation.

## Roadmap

- More augmenters (casing, punctuation, OCR-style noise, unicode confusables).
- Direct spaCy `DocBin` / Hugging Face `datasets` export.
- Optional entity normalization (link a surface form to its canonical value).

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © [Berk Gökden](https://berkgokden.com)
