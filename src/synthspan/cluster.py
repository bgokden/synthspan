"""Diversity balancing via embeddings + clustering (pure-Python, zero deps).

You bring a local embedder (any ``Callable[[str], Sequence[float]]`` — e.g. a
sentence-transformer or an Ollama embedding call). Examples are clustered by
meaning and sampled evenly across clusters, so a few semantically dominant
phrasings don't swamp the dataset. Count-based balancing lives in
``synthspan.balance``; this is the *semantic* counterpart.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

from synthspan.types import Example

Embedder = Callable[[str], Sequence[float]]


def _dist2(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def kmeans(
    vectors: list[Sequence[float]],
    k: int,
    rng: random.Random,
    iters: int = 25,
) -> list[int]:
    """Minimal Lloyd's k-means. Returns a cluster index per vector."""
    n = len(vectors)
    k = max(1, min(k, n))

    # k-means++ init: spread seeds apart so duplicate-heavy data still separates.
    centers = [list(vectors[rng.randrange(n)])]
    while len(centers) < k:
        d2 = [min(_dist2(v, c) for c in centers) for v in vectors]
        total = sum(d2)
        if total == 0:  # remaining points identical to chosen centers
            centers.append(list(vectors[rng.randrange(n)]))
            continue
        threshold = rng.random() * total
        acc = 0.0
        for i, w in enumerate(d2):
            acc += w
            if acc >= threshold:
                centers.append(list(vectors[i]))
                break

    assign = [0] * n
    for _ in range(iters):
        changed = False
        for i, v in enumerate(vectors):
            best = min(range(k), key=lambda c: _dist2(v, centers[c]))
            if best != assign[i]:
                assign[i] = best
                changed = True
        for c in range(k):
            members = [vectors[i] for i in range(n) if assign[i] == c]
            if members:
                dim = len(members[0])
                centers[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]
        if not changed:
            break
    return assign


def cluster_balance(
    examples: list[Example],
    embed_fn: Embedder,
    k: int,
    rng: random.Random | None = None,
    per_cluster: int | None = None,
) -> list[Example]:
    """Cluster examples by embedding and sample evenly across clusters.

    Args:
        examples: Items to balance.
        embed_fn: Maps an example's text to a vector (your local embedder).
        k: Number of clusters.
        rng: Seeded RNG.
        per_cluster: Items to keep per cluster. Defaults to the smallest cluster
            size, yielding a fully balanced subset.

    Returns:
        A balanced, cluster-interleaved subset of ``examples``.
    """
    rng = rng or random.Random()
    if not examples:
        return []
    vectors = [list(embed_fn(ex.text)) for ex in examples]
    assign = kmeans(vectors, k, rng)

    groups: dict[int, list[Example]] = {}
    for ex, c in zip(examples, assign):
        groups.setdefault(c, []).append(ex)
    for members in groups.values():
        rng.shuffle(members)

    cap = per_cluster if per_cluster is not None else min(len(m) for m in groups.values())
    out: list[Example] = []
    for rank in range(cap):
        for c in sorted(groups):
            if rank < len(groups[c]):
                out.append(groups[c][rank])
    return out
