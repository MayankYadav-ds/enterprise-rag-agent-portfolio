"""Retrieval-quality metrics.

Each metric is a pure function over plain ranked hit lists, so the same
function serves any retriever implementation and any evaluation runner.
Scores are clipped to ``[0, 1]`` and never raise on empty input.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


def _clip(value: float) -> float:
    """Clamp a score to the valid ``[0, 1]`` range."""
    return max(0.0, min(1.0, value))


@dataclass(frozen=True, slots=True)
class PrecisionAtK:
    """Fraction of the top-K hits that are relevant.

    ``relevant`` is the set of gold chunk IDs for the query. A hit at rank r
    counts when its chunk id is in that set. Precision is 0 when K is 0 or
    no hits are returned.
    """

    k: int

    def compute(self, hits: list[str], relevant: set[str]) -> float:
        if self.k <= 0 or not hits:
            return 0.0
        top = hits[: self.k]
        if not top:
            return 0.0
        return _clip(sum(1 for chunk_id in top if chunk_id in relevant) / len(top))


@dataclass(frozen=True, slots=True)
class RecallAtK:
    """Fraction of the gold relevant chunks that appear in the top-K hits.

    Recall is 0 when the gold set is empty (there is nothing to recall) and
    1.0 when every relevant chunk is returned in the top K. It does not
    penalise returning extra non-relevant chunks.
    """

    k: int

    def compute(self, hits: list[str], relevant: set[str]) -> float:
        if self.k <= 0 or not relevant:
            return 0.0
        top = set(hits[: self.k])
        if not top:
            return 0.0
        return _clip(len(top & relevant) / len(relevant))


def mean_reciprocal_rank(hits: list[str], relevant: set[str]) -> float:
    """Return the reciprocal rank of the first relevant hit.

    MRR is 1.0 when the first hit is relevant, 0.5 when the second is, and
    0.0 when no relevant hit appears in the list. It is averaged over the
    queries that have at least one relevant chunk; queries with no relevant
    chunk contribute 0.
    """
    if not relevant:
        return 0.0
    for rank, chunk_id in enumerate(hits, start=1):
        if chunk_id in relevant:
            return _clip(1.0 / rank)
    return 0.0


@dataclass(frozen=True, slots=True)
class RankedRetrievalMetrics:
    """Metrics for one already-ranked result list.

    ``precision_at_k`` divides by the number of distinct results actually
    returned up to K (so K larger than the result count does not dilute the
    score). ``recall_at_k`` divides by the number of unique relevant IDs.
    Queries with no relevant labels receive zero for every metric. Duplicate
    returned IDs are ignored after their first occurrence. Score ties retain
    the retriever's supplied order; this helper never re-sorts results.
    """

    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate_at_k: float


def ranked_retrieval_metrics(
    retrieved_ids: Sequence[str],
    relevant_ids: Collection[str],
    *,
    k: int,
    scores: Sequence[float] | None = None,
) -> RankedRetrievalMetrics:
    """Compute deterministic binary-relevance metrics for one ranked query.

    IDs are de-duplicated in first-seen order. Optional result scores are
    validated for count and finiteness, but never used to change the supplied
    ranking: retrievers own tie-breaking and score semantics (cosine, BM25,
    RRF, and cross-encoder logits are not interchangeable).

    ``k`` must be a positive integer. Empty results and empty relevance sets
    produce zeros. nDCG uses binary gain and logarithmic discount
    ``1 / log2(rank + 1)``; its ideal ranking contains up to K relevant IDs.
    """
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k < 1:
        raise ValueError("k must be at least 1")
    if isinstance(retrieved_ids, (str, bytes)):
        raise TypeError("retrieved_ids must be a sequence of IDs")
    if isinstance(relevant_ids, (str, bytes)):
        raise TypeError("relevant_ids must be a collection of IDs")

    raw_ids = list(retrieved_ids)
    gold = set(relevant_ids)
    if any(not isinstance(value, str) or not value for value in raw_ids):
        raise ValueError("retrieved IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in gold):
        raise ValueError("relevant IDs must be non-empty strings")
    if scores is not None:
        if len(scores) != len(raw_ids):
            raise ValueError("scores must have the same length as retrieved_ids")
        for score in scores:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError("result scores must be finite numbers")
            try:
                finite = math.isfinite(score)
            except OverflowError:
                finite = False
            if not finite:
                raise ValueError("result scores must be finite numbers")

    unique_ids = list(dict.fromkeys(raw_ids))
    top = unique_ids[:k]
    if not gold:
        return RankedRetrievalMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

    relevant_top_count = sum(chunk_id in gold for chunk_id in top)
    precision = relevant_top_count / len(top) if top else 0.0
    recall = relevant_top_count / len(gold)
    reciprocal_rank = next(
        (1.0 / rank for rank, chunk_id in enumerate(unique_ids, start=1) if chunk_id in gold),
        0.0,
    )
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, chunk_id in enumerate(top, start=1) if chunk_id in gold
    )
    ideal_count = min(len(gold), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    hit_rate = 1.0 if relevant_top_count else 0.0
    return RankedRetrievalMetrics(
        precision_at_k=_clip(precision),
        recall_at_k=_clip(recall),
        mrr=_clip(reciprocal_rank),
        ndcg_at_k=_clip(ndcg),
        hit_rate_at_k=hit_rate,
    )
