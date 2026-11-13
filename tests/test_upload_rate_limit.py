"""Concurrency and expiry tests for durable upload admission."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

from rag_agent.api.upload_rate_limit import reserve_upload_attempt, upload_bucket


def _database(path):
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE upload_attempts (bucket TEXT PRIMARY KEY, "
            "window_started REAL NOT NULL, attempts INTEGER NOT NULL)"
        )


def test_concurrent_upload_admission_never_exceeds_limit(tmp_path):
    database = tmp_path / "upload-limit.sqlite3"
    _database(database)
    workers, limit = 20, 4
    barrier = threading.Barrier(workers)
    bucket = upload_bucket("192.0.2.18")

    def reserve(_):
        barrier.wait(timeout=10)
        with sqlite3.connect(database, timeout=10) as connection:
            admission = reserve_upload_attempt(
                connection, bucket, now=100, window_seconds=60, limit=limit
            )
            return admission.admitted

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(reserve, range(workers)))
    assert sum(results) == limit
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            "SELECT attempts FROM upload_attempts WHERE bucket = ?", (bucket,)
        ).fetchone()
        blocked = reserve_upload_attempt(
            connection, bucket, now=101, window_seconds=60, limit=limit
        )
    assert row == (limit,)
    assert not blocked.admitted
    assert 1 <= blocked.retry_after <= 60


def test_upload_admission_expiry_and_peer_independence(tmp_path):
    database = tmp_path / "upload-limit.sqlite3"
    _database(database)
    first = upload_bucket("192.0.2.20")
    second = upload_bucket("192.0.2.21")
    with sqlite3.connect(database) as connection:
        assert reserve_upload_attempt(
            connection, first, now=100, window_seconds=10, limit=1
        ).admitted
        independent = reserve_upload_attempt(
            connection, second, now=101, window_seconds=10, limit=1
        )
        expired = reserve_upload_attempt(connection, first, now=110, window_seconds=10, limit=1)
    assert independent.admitted
    assert expired.admitted


def test_upload_admission_contention_raises_instead_of_bypassing_limit(tmp_path):
    database = tmp_path / "upload-limit.sqlite3"
    _database(database)
    locker = sqlite3.connect(database, timeout=0)
    contender = sqlite3.connect(database, timeout=0)
    try:
        locker.execute("BEGIN IMMEDIATE")
        try:
            reserve_upload_attempt(
                contender,
                upload_bucket("192.0.2.30"),
                now=100,
                window_seconds=60,
                limit=1,
            )
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc).lower()
        else:
            raise AssertionError("write contention must not admit the upload")
    finally:
        locker.rollback()
        locker.close()
        contender.close()
