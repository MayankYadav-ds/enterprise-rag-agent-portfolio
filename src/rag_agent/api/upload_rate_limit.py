"""Persistent per-peer upload admission limits backed by SQLite.

Each admitted request consumes one slot in a short ``BEGIN IMMEDIATE``
transaction before the request body is streamed or document processing begins.
SQLite coordinates independent connections and application workers; no database
write lock is held while reading a PDF, parsing it, or generating embeddings.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(frozen=True)
class UploadAdmission:
    admitted: bool
    retry_after: int | None = None


def upload_bucket(peer: str | None) -> str:
    """Hash the direct transport peer; never persist a raw address."""
    value = peer or "unknown"
    return hashlib.sha256(b"upload:" + value.encode("utf-8", errors="replace")).hexdigest()


@contextmanager
def _immediate_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def ensure_upload_rate_limit_schema(runtime) -> None:
    """Create the upload limiter table in the existing application registry."""
    with runtime.connection() as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS upload_attempts ("
            "bucket TEXT PRIMARY KEY, window_started REAL NOT NULL, attempts INTEGER NOT NULL)"
        )


def reserve_upload_attempt(
    conn: sqlite3.Connection,
    bucket: str,
    *,
    window_seconds: int,
    limit: int,
    now: float | None = None,
) -> UploadAdmission:
    """Atomically consume an upload slot or return a safe retry delay.

    Admission itself is the durable reservation, so a process crash cannot leave
    an unaccounted in-flight slot. Rejected and admitted calls are serialized by
    SQLite across threads and workers. Invalid or aborted admitted uploads still
    consume a slot to limit repeated parsing and upload abuse.
    """
    current = time.time() if now is None else now
    with _immediate_transaction(conn):
        conn.execute(
            "DELETE FROM upload_attempts WHERE window_started <= ?",
            (current - window_seconds,),
        )
        conn.execute(
            "DELETE FROM upload_attempts WHERE bucket IN ("
            "SELECT bucket FROM upload_attempts ORDER BY window_started DESC "
            "LIMIT -1 OFFSET 10000)"
        )
        row = conn.execute(
            "SELECT window_started, attempts FROM upload_attempts WHERE bucket = ?", (bucket,)
        ).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO upload_attempts (bucket, window_started, attempts) VALUES (?, ?, 1)",
                (bucket, current),
            )
            return UploadAdmission(True)

        started, attempts = float(row[0]), int(row[1])
        if current - started >= window_seconds:
            conn.execute(
                "UPDATE upload_attempts SET window_started = ?, attempts = 1 WHERE bucket = ?",
                (current, bucket),
            )
            return UploadAdmission(True)
        if attempts >= limit:
            retry_after = min(window_seconds, max(1, int(window_seconds - (current - started))))
            return UploadAdmission(False, retry_after)
        conn.execute(
            "UPDATE upload_attempts SET attempts = attempts + 1 WHERE bucket = ?", (bucket,)
        )
        return UploadAdmission(True)
