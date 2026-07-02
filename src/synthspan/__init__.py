"""synthspan — generate synthetic labeled-span data for NER / token classification."""

from synthspan.augment import apply, augment, ocr, punctuation, random_case, typos
from synthspan.balance import cap_per_value, dedupe, label_counts, value_counts
from synthspan.cluster import cluster_balance
from synthspan.gazetteer import Gazetteer
from synthspan.generate import generate
from synthspan.template import Template
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
    "apply",
    "typos",
    "random_case",
    "punctuation",
    "ocr",
    "dedupe",
    "cap_per_value",
    "label_counts",
    "value_counts",
    "cluster_balance",
    "to_jsonl",
    "to_conll",
    "to_spacy",
    "write",
]
