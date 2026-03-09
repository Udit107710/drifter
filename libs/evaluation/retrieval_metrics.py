"""Retrieval evaluation metrics: Recall@k, Precision@k, MRR, NDCG."""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k retrieved.

    Formula: |retrieved[:k] ∩ relevant| / |relevant|
    Range: [0.0, 1.0]

    Returns 0.0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved documents that are relevant.

    Formula: |retrieved[:k] ∩ relevant| / k
    Range: [0.0, 1.0]

    Returns 0.0 if k <= 0.
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant result.

    Formula: 1 / rank_of_first_relevant
    Range: (0.0, 1.0] or 0.0 if no relevant found

    Returns 0.0 if no relevant document in retrieved list.
    """
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Normalized Discounted Cumulative Gain at k.

    Supports graded relevance (not just binary).

    DCG@k  = Σ (2^rel_i - 1) / log2(i + 2)  for i in [0, k)
    IDCG@k = DCG of the ideal ranking (sorted by relevance descending)
    NDCG@k = DCG@k / IDCG@k

    Range: [0.0, 1.0]
    Returns 0.0 if relevance_grades is empty or k <= 0.
    """
    if k <= 0 or not relevance_grades:
        return 0.0

    def dcg(ordered_ids: list[str], n: int) -> float:
        total = 0.0
        for i, doc_id in enumerate(ordered_ids[:n]):
            rel = relevance_grades.get(doc_id, 0)
            total += (2**rel - 1) / math.log2(i + 2)
        return total

    # Actual DCG from retrieved order
    actual = dcg(retrieved, k)

    # Ideal DCG: sort all by relevance descending
    ideal_order = sorted(
        relevance_grades.keys(),
        key=lambda x: relevance_grades[x],
        reverse=True,
    )
    ideal = dcg(ideal_order, k)

    if ideal == 0.0:
        return 0.0
    return actual / ideal
