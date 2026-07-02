"""Realistic typo augmentation that keeps labeled spans correct.

Text is split into entity / non-entity segments; typos are applied per word and
the spans are recomputed from the new segment lengths, so labels never drift —
even when an edit changes the length of surrounding text.
"""

from __future__ import annotations

import random

from synthspan.types import Example, Span

# QWERTY adjacency for realistic substitutions/insertions.
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
    """Apply a single random edit to a word."""
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
    # substitute
    return word[:i] + _neighbor(word[i], rng) + word[i + 1 :]


def _typo_text(text: str, rate: float, rng: random.Random) -> str:
    """Apply typos to word tokens with per-word probability ``rate``."""
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


def augment(
    examples: list[Example],
    rate: float = 0.05,
    rng: random.Random | None = None,
    typo_entities: bool = False,
) -> list[Example]:
    """Return copies of ``examples`` with typos injected, spans preserved."""
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
            if label is None:
                new = _typo_text(seg_text, rate, rng)
            else:
                new = _typo_text(seg_text, rate, rng) if typo_entities else seg_text
                new_spans.append(Span(pos, pos + len(new), label))
            parts.append(new)
            pos += len(new)
        out.append(Example("".join(parts), new_spans))
    return out
