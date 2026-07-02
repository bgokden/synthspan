"""synthspan — generate synthetic labeled-span data for NER / token classification."""

from synthspan.balance import cap_per_value, dedupe, label_counts, value_counts
from synthspan.gazetteer import Gazetteer
from synthspan.generate import generate
from synthspan.template import Template
from synthspan.typos import augment
from synthspan.types import Example, Span
from synthspan.writers import to_conll, to_jsonl, to_spacy, write

__version__ = "0.1.0"

__all__ = [
    "Gazetteer",
    "Template",
    "Example",
    "Span",
    "generate",
    "augment",
    "dedupe",
    "cap_per_value",
    "label_counts",
    "value_counts",
    "to_jsonl",
    "to_conll",
    "to_spacy",
    "write",
]
