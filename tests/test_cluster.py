import random

from synthspan.cluster import cluster_balance, kmeans
from synthspan.types import Example


def _tagged_embed(text: str):
    """Deterministic 'embedder': two well-separated clusters by a tag in the text."""
    return [0.0, 0.0] if text.startswith("A:") else [10.0, 10.0]


def test_kmeans_separates_two_clusters():
    vectors = [[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 9.9]]
    assign = kmeans(vectors, 2, random.Random(0))
    assert assign[0] == assign[1] and assign[2] == assign[3] and assign[0] != assign[2]


def test_cluster_balance_evens_out_groups():
    # 6 in cluster A, 2 in cluster B -> balanced output should be 2 + 2
    examples = [Example(f"A:{i}", []) for i in range(6)] + [Example(f"B:{i}", []) for i in range(2)]
    out = cluster_balance(examples, _tagged_embed, k=2, rng=random.Random(1))
    a = sum(1 for e in out if e.text.startswith("A:"))
    b = sum(1 for e in out if e.text.startswith("B:"))
    assert a == b == 2


def test_cluster_balance_per_cluster_cap():
    examples = [Example(f"A:{i}", []) for i in range(6)] + [Example(f"B:{i}", []) for i in range(6)]
    out = cluster_balance(examples, _tagged_embed, k=2, rng=random.Random(1), per_cluster=3)
    assert len(out) == 6  # 3 per cluster


def test_cluster_balance_empty():
    assert cluster_balance([], _tagged_embed, k=2) == []
