"""Shared input validation for retrievers.

``retrieve`` takes a query and a set of filters. The validation rules
(non-empty query, ``top_k`` in ``1..100``, ``bool`` rejected as ``top_k``) are
the same for every retriever, so they live here once instead of being
duplicated in ``DenseRetriever``, ``BM25Retriever`` and ``HybridRetriever``.

The rules are part of the ``Retriever`` protocol's contract: a caller that
violates them gets a loud, typed error before any embedding or scoring happens,
rather than a silently wrong result.
"""

from __future__ import annotations

from math import isfinite

_MIN_TOP_K = 1
_MAX_TOP_K = 100


def validate_top_k(top_k: int) -> int:
    """Reject out-of-range ``top_k`` early so a misconfiguration is loud.

    ``bool`` is rejected explicitly because it is a subclass of ``int`` and
    would otherwise pass as ``top_k=1`` (True) or ``top_k=0`` (False).
    """
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise TypeError(f"top_k must be an int, got {type(top_k).__name__}")
    if top_k < _MIN_TOP_K or top_k > _MAX_TOP_K:
        raise ValueError(f"top_k must be in [{_MIN_TOP_K}, {_MAX_TOP_K}], got {top_k}")
    return top_k


def validate_query(query: str) -> str:
    """Reject empty/whitespace queries before they reach the retriever."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    return query.strip()


def required_payload_value(payload: dict, key: str) -> object:
    """Return required evidence metadata or reject a corrupt hit."""
    value = payload.get(key)
    if value is None or value == "":
        raise KeyError(f"hit payload is missing required field {key!r}")
    return value


def required_payload_string(payload: dict, key: str) -> str:
    """Return required string evidence metadata without coercing corrupt values."""
    value = required_payload_value(payload, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"hit payload contains an invalid {key!r} field")
    return value


def validate_page_range(payload: dict) -> tuple[int, int]:
    """Parse a physical PDF page range and reject values that cannot be cited."""
    raw_start = required_payload_value(payload, "page_start")
    raw_end = required_payload_value(payload, "page_end")
    if isinstance(raw_start, bool) or isinstance(raw_end, bool):
        raise ValueError("hit payload contains an invalid page range")
    if any(isinstance(value, float) and not isfinite(value) for value in (raw_start, raw_end)):
        raise ValueError("hit payload contains an invalid page range")
    try:
        page_start = int(raw_start)
        page_end = int(raw_end)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("hit payload contains an invalid page range") from exc
    if (
        (isinstance(raw_start, float) and not raw_start.is_integer())
        or (isinstance(raw_end, float) and not raw_end.is_integer())
        or page_start < 1
        or page_end < page_start
    ):
        raise ValueError("hit payload contains an invalid page range")
    return page_start, page_end
