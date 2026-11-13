"""Windows-spawn multi-process tests for the production document registry."""

from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.documents import DocumentError
from rag_agent.api.runtime import Runtime
from rag_agent.api.settings import APISettings
from rag_agent.api.storage import private_directory, private_file
from rag_agent.ingestion.models import document_id_from_hash, sha256_for_file
from test_phase_10_10_document_lifecycle import (
    _login,
    _new_system,
    _pdf_bytes,
    _query,
    _set_session,
)


class _EmptyVectorStore:
    """The vector boundary only; SQLite/document metadata remains production code."""

    def scroll_ids(self, doc_id=None, *, tenant_id=None):
        return []


def _runtime_for(root: str) -> Runtime:
    return Runtime(APISettings(document_dir=Path(root), environment="test"))


def _stage_bytes(root: Path, tenant_id: str, data: bytes) -> Path:
    path = root / "tenants" / tenant_id / f"{uuid.uuid4().hex}.pdf"
    private_directory(path.parent)
    path.write_bytes(data)
    private_file(path)
    return path


def _register_worker(root: str, tenant_id: str, file_name: str, start, ready, result) -> None:
    runtime = _runtime_for(root)
    runtime._store = lambda _embedder, *, validate: _EmptyVectorStore()
    ready.set()
    if not start.wait(timeout=10):
        result.send(("error", "start_timeout"))
        runtime.close()
        result.close()
        return
    try:
        manager = runtime.documents()
        try:
            record = manager.accept(Path(file_name), "process-race.pdf", tenant_id)
        except DocumentError as error:
            result.send(("rejected", error.status, error.code))
        else:
            result.send(("accepted", record.doc_id))
    except DocumentError as error:
        result.send(("rejected", error.status, error.code))
    except Exception as error:
        result.send(("error", type(error).__name__))
    finally:
        runtime.close()
        result.close()


def _hold_writer_then_crash(root: str, tenant_id: str, file_name: str, acquired, release) -> None:
    runtime = _runtime_for(root)
    runtime._store = lambda _embedder, *, validate: _EmptyVectorStore()
    manager = runtime.documents()
    record = manager.accept(Path(file_name), "crash-recovery.pdf", tenant_id)
    if not record.doc_id:
        os._exit(41)
    acquired.set()
    if not release.wait(timeout=15):
        os._exit(42)
    # Intentional crash simulation after the production _save transaction has
    # committed, while the process still owns the production writer lock.
    os._exit(43)


def _crash_uncommitted_insert(root: str, tenant_id: str, doc_id: str, started, crash) -> None:
    runtime = _runtime_for(root)
    record = {
        "doc_id": doc_id,
        "filename": "uncommitted.pdf",
        "status": "queued",
        "managed": True,
        "owns_vectors": False,
        "size_bytes": 128,
        "page_count": None,
        "chunk_count": 0,
        "updated_at": None,
        "error": None,
    }
    with runtime.connection() as connection:
        connection.execute(
            "INSERT INTO documents (id, record, file, tenant_id) VALUES (?,?,?,?)",
            (doc_id, json.dumps(record), f"{uuid.uuid4().hex}.pdf", tenant_id),
        )
        started.set()
        if not crash.wait(timeout=15):
            os._exit(44)
        # Leave the sqlite3 connection context before its commit point.
        os._exit(45)


def _join_owned(process, timeout: float = 15) -> None:
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
    assert not process.is_alive(), "phase-owned child process did not exit"


def test_cross_process_writer_lock_and_crash_recovery_then_retry_and_citations(
    tmp_path, monkeypatch
) -> None:
    app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    tenant_id = user_ids["alpha"]
    manager = runtime.documents()
    content = _pdf_bytes("Recovered synthetic fact: value 53. This evidence is disposable.")
    first_file = _stage_bytes(runtime.settings.document_dir, tenant_id, content)
    retry_file = _stage_bytes(runtime.settings.document_dir, tenant_id, content)
    source_id = document_id_from_hash(sha256_for_file(str(first_file)))
    expected_doc_id = manager._tenant_document_id(tenant_id, source_id)
    root = str(runtime.settings.document_dir)
    runtime.close()

    context = multiprocessing.get_context("spawn")
    acquired = context.Event()
    release_crash = context.Event()
    holder = context.Process(
        target=_hold_writer_then_crash,
        args=(root, tenant_id, str(first_file), acquired, release_crash),
    )
    parent_end, child_end = context.Pipe(duplex=False)
    ready = context.Event()
    start = context.Event()
    contender = context.Process(
        target=_register_worker,
        args=(root, tenant_id, str(retry_file), start, ready, child_end),
    )
    try:
        holder.start()
        assert acquired.wait(timeout=10)
        contender.start()
        assert ready.wait(timeout=10)
        start.set()
        assert parent_end.poll(10)
        contender_result = parent_end.recv()
        assert contender_result == ("rejected", 503, "writer_busy")
        release_crash.set()
        _join_owned(holder)
        _join_owned(contender)
        assert holder.exitcode == 43
        assert contender.exitcode == 0

        # The successful registration committed before the child crashed.
        database = root + "\\inventory.sqlite3"
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT id, tenant_id, record FROM documents").fetchall()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        assert len(rows) == 1
        assert rows[0][0] == expected_doc_id
        assert rows[0][1] == tenant_id
        assert json.loads(rows[0][2])["status"] == "queued"
        assert integrity == "ok"

        # A fresh production runtime recovers committed queued intent as failed,
        # then permits an explicit retry with the same tenant-derived ID.
        manager = runtime.documents()
        recovered = manager.get(expected_doc_id, tenant_id)
        assert recovered.status == "failed"
        assert manager._file(expected_doc_id, tenant_id).is_file()
        retried = manager.accept(retry_file, "process-race.pdf", tenant_id)
        assert retried.doc_id == expected_doc_id and retried.status == "queued"
        manager.process(retried.doc_id, tenant_id)
        ready_record = manager.get(expected_doc_id, tenant_id)
        assert ready_record.status == "ready"
        assert (
            len(store.scroll_ids(expected_doc_id, tenant_id=tenant_id)) == ready_record.chunk_count
        )

        with TestClient(app) as client:
            session = _login(client, "alpha")
            answer = _query(client, session, "What value does the recovered report support?")
            assert answer.status_code == 200
            assert answer.json()["citations"]
            assert "value 53" in answer.json()["citations"][0]["quote"]

            _set_session(client, session)
            deleted = client.delete(f"/api/v1/documents/{expected_doc_id}")
            assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
            stale = _query(client, session, "What value does the recovered report support?")
            assert stale.status_code == 200
            assert stale.json()["status"] == "insufficient_evidence"
            assert stale.json()["citations"] == []
    finally:
        release_crash.set()
        if holder.pid is not None:
            _join_owned(holder)
        if contender.pid is not None:
            start.set()
            _join_owned(contender)
        parent_end.close()
        child_end.close()
        runtime.close()


def test_sequential_independent_processes_keep_identical_content_tenant_scoped(
    tmp_path, monkeypatch
) -> None:
    _app, runtime, _store, user_ids = _new_system(tmp_path, monkeypatch)
    manager = runtime.documents()
    root = runtime.settings.document_dir
    tenants = (user_ids["alpha"], user_ids["beta"])
    content = _pdf_bytes("Two tenants independently register identical synthetic content.")
    files = [_stage_bytes(root, tenant_id, content) for tenant_id in tenants]
    source_id = document_id_from_hash(sha256_for_file(str(files[0])))
    expected_ids = [manager._tenant_document_id(tenant_id, source_id) for tenant_id in tenants]
    runtime.close()

    context = multiprocessing.get_context("spawn")
    outcomes = []
    for tenant_id, path in zip(tenants, files, strict=True):
        parent_end, child_end = context.Pipe(duplex=False)
        ready = context.Event()
        start = context.Event()
        process = context.Process(
            target=_register_worker,
            args=(str(root), tenant_id, str(path), start, ready, child_end),
        )
        try:
            process.start()
            assert ready.wait(timeout=10)
            start.set()
            assert parent_end.poll(10)
            outcomes.append(parent_end.recv())
            _join_owned(process)
            assert process.exitcode == 0
        finally:
            start.set()
            if process.pid is not None:
                _join_owned(process)
            parent_end.close()
            child_end.close()

    assert outcomes == [("accepted", expected_ids[0]), ("accepted", expected_ids[1])]
    assert expected_ids[0] != expected_ids[1]

    manager = runtime.documents()
    try:
        assert manager.get(expected_ids[0], tenants[0]).status == "failed"
        assert manager.get(expected_ids[1], tenants[1]).status == "failed"
        with manager.connection() as connection:
            bindings = connection.execute("SELECT id, tenant_id FROM documents").fetchall()
        assert {tuple(row) for row in bindings} == set(zip(expected_ids, tenants, strict=True))
        with pytest.raises(DocumentError) as error:
            manager.get(expected_ids[0], tenants[1])
        assert error.value.status == 404
    finally:
        runtime.close()


def test_child_crash_before_sqlite_commit_rolls_back_uncommitted_registry_row(
    tmp_path, monkeypatch
):
    _app, runtime, _store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    manager = runtime.documents()
    root = str(runtime.settings.document_dir)
    tenant_id = user_ids["alpha"]
    uncommitted_id = "doc_" + "f" * 64
    runtime.close()

    context = multiprocessing.get_context("spawn")
    started = context.Event()
    crash = context.Event()
    child = context.Process(
        target=_crash_uncommitted_insert,
        args=(root, tenant_id, uncommitted_id, started, crash),
    )
    try:
        child.start()
        assert started.wait(timeout=10)
        crash.set()
        _join_owned(child)
        assert child.exitcode == 45

        runtime = Runtime(APISettings(document_dir=Path(root), environment="test"))
        try:
            manager = runtime.documents()
            with manager.connection() as connection:
                row = connection.execute(
                    "SELECT id FROM documents WHERE id=?", (uncommitted_id,)
                ).fetchone()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            assert row is None
            assert integrity == "ok"
            assert manager.records(tenant_id) == []
        finally:
            runtime.close()
    finally:
        crash.set()
        if child.pid is not None:
            _join_owned(child)
        runtime.close()
