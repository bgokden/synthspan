"""Composable text augmenters with guaranteed span preservation.

An **augmenter** is just a ``Callable[[str, random.Random], str]``. The framework
(:func:`apply`) splits each example into entity / non-entity segments, runs the
augmenters over the eligible segments, and recomputes spans from the new segment
lengths — so labels stay correct no matter how the text length changes.

Adding your own augmenter is one function::

    def shout(rate=0.1):
        def f(text, rng):
            return "".join(c.upper() if rng.random() < rate else c for c in text)
        return f

    apply(examples, [typos(0.05), shout(0.2)])
"""

from __future__ import annotations

import random
from typing import Callable

from synthspan.types import Example, Span

#: An augmenter transforms a piece of text given an RNG.
Augmenter = Callable[[str, random.Random], str]

# --- shared bits for the typo augmenter -------------------------------------

_NEIGHBORS = {
    "a": "qwsz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikmn",
    "k": "jiolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu", "z": "asx",
}


def _neighbor(ch: str, rng: random.Random) -> str:
    opts = _NEIGHBORS.get(ch.lower())
    if not opts:
        return ch
    n = rng.choice(opts)
    return n.upper() if ch.isupper() else n


def _typo_word(word: str, rng: random.Random) -> str:
    if len(word) < 1:
        return word
    ops = ["insert", "substitute"]
    if len(word) >= 2:
        ops += ["swap", "delete"]
    op = rng.choice(ops)
    i = rng.randrange(len(word))
    if op == "swap" and i < len(word) - 1:
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if op == "delete":
        return word[:i] + word[i + 1 :]
    if op == "insert":
        return word[:i] + _neighbor(word[i], rng) + word[i:]
    return word[:i] + _neighbor(word[i], rng) + word[i + 1 :]


# --- built-in augmenter factories -------------------------------------------


def typos(rate: float = 0.05) -> Augmenter:
    """Keyboard-aware typos (swap/delete/insert/substitute) per word."""

    def f(text: str, rng: random.Random) -> str:
        if rate <= 0:
            return text
        out: list[str] = []
        word: list[str] = []

        def flush() -> None:
            if word:
                w = "".join(word)
                out.append(_typo_word(w, rng) if rng.random() < rate else w)
                word.clear()

        for ch in text:
            if ch.isalnum():
                word.append(ch)
            else:
                flush()
                out.append(ch)
        flush()
        return "".join(out)

    return f


def random_case(rate: float = 0.1) -> Augmenter:
    """Randomly flip the case of letters (length-preserving)."""

    def f(text: str, rng: random.Random) -> str:
        return "".join(
            c.swapcase() if (c.isalpha() and rng.random() < rate) else c for c in text
        )

    return f


def punctuation(rate: float = 0.1) -> Augmenter:
    """Drop or duplicate punctuation marks."""
    marks = set(".,;:!?")

    def f(text: str, rng: random.Random) -> str:
        out: list[str] = []
        for ch in text:
            if ch in marks and rng.random() < rate:
                if rng.random() < 0.5:
                    continue  # drop
                out.append(ch)  # duplicate
                out.append(ch)
            else:
                out.append(ch)
        return "".join(out)

    return f


def ocr(rate: float = 0.05) -> Augmenter:
    """OCR-style confusable substitutions (o->0, l->1, S->5, ...)."""
    conf = {"o": "0", "O": "0", "l": "1", "I": "1", "i": "1",
            "s": "5", "S": "5", "B": "8", "g": "9", "z": "2"}

    def f(text: str, rng: random.Random) -> str:
        return "".join(conf[c] if (c in conf and rng.random() < rate) else c for c in text)

    return f


# --- the span-preserving framework ------------------------------------------


def apply(
    examples: list[Example],
    augmenters: list[Augmenter],
    rng: random.Random | None = None,
    augment_entities: bool = False,
) -> list[Example]:
    """Run ``augmenters`` over each example, keeping spans correct.

    By default only non-entity text is augmented (entities stay clean). Set
    ``augment_entities=True`` to also noise the entity surfaces (labels preserved).
    """
    rng = rng or random.Random()
    out: list[Example] = []
    for ex in examples:
        spans = sorted(ex.spans, key=lambda s: s.start)
        segments: list[tuple[str, str | None]] = []
        cur = 0
        for s in spans:
            if s.start > cur:
                segments.append((ex.text[cur : s.start], None))
            segments.append((ex.text[s.start : s.end], s.label))
            cur = s.end
        if cur < len(ex.text):
            segments.append((ex.text[cur:], None))

        parts: list[str] = []
        new_spans: list[Span] = []
        pos = 0
        for seg_text, label in segments:
            new = seg_text
            if label is None or augment_entities:
                for aug in augmenters:
                    new = aug(new, rng)
            if label is not None:
                new_spans.append(Span(pos, pos + len(new), label))
            parts.append(new)
            pos += len(new)
        out.append(Example("".join(parts), new_spans))
    return out


def augment(
    examples: list[Example],
    rate: float = 0.05,
    rng: random.Random | None = None,
    typo_entities: bool = False,
) -> list[Example]:
    """Convenience: apply typo augmentation (kept for backward compatibility)."""
    return apply(examples, [typos(rate)] if rate > 0 else [], rng=rng, augment_entities=typo_entities)
