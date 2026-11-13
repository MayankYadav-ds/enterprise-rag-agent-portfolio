"""Synthetic persistence and concurrency checks for managed document lifecycle."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from rag_agent.api.documents import DocumentError
from rag_agent.api.runtime import Runtime
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.storage import private_file
from rag_agent.ingestion.models import document_id_from_hash, sha256_for_file
from test_phase_10_10_document_lifecycle import _new_system, _pdf_bytes


def _stage_pdf(manager, tenant_id: str, content: bytes):
    path = manager.tenant_root(tenant_id) / f"{uuid.uuid4().hex}.pdf"
    path.write_bytes(content)
    private_file(path)
    return path


def _accept_outcome(manager, path, filename: str, tenant_id: str):
    try:
        record = manager.accept(path, filename, tenant_id)
    except DocumentError as error:
        return error.status, None
    return 202, record.doc_id


def test_same_tenant_concurrent_identical_registration_is_unique_and_survives_restart(
    tmp_path, monkeypatch
) -> None:
    _app, runtime, _store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    tenant_id = user_ids["alpha"]
    manager = runtime.documents()
    content = _pdf_bytes("Concurrent synthetic registration fact.")
    paths = [_stage_pdf(manager, tenant_id, content) for _ in range(2)]
    barrier = threading.Barrier(2)
    original_accept = manager.accept

    def synchronized_accept(path, filename, request_tenant):
        barrier.wait(timeout=5)
        return original_accept(path, filename, request_tenant)

    monkeypatch.setattr(manager, "accept", synchronized_accept)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_accept_outcome, manager, path, "same.pdf", tenant_id) for path in paths
            ]
            outcomes = [future.result(timeout=10) for future in futures]

        assert sorted(status for status, _doc_id in outcomes) == [202, 409]
        accepted_ids = {doc_id for status, doc_id in outcomes if status == 202}
        assert len(accepted_ids) == 1
        doc_id = accepted_ids.pop()
        records = manager.records(tenant_id)
        assert len(records) == 1
        assert records[0].doc_id == doc_id
        assert records[0].status == "queued"
        registered_file = manager._file(doc_id, tenant_id)
        assert registered_file.is_file()

        # A new runtime recovers the durable queued intent after restart without
        # loading or contacting the vector service.
        settings = runtime.settings
        runtime.close()
        recovered_runtime = Runtime(settings)
        try:
            recovered_manager = recovered_runtime.documents()
            recovered = recovered_manager.get(doc_id, tenant_id)
            assert recovered.status == "failed"
            assert recovered_manager._file(doc_id, tenant_id).is_file()
            assert len(recovered_manager.records(tenant_id)) == 1
        finally:
            recovered_runtime.close()
    finally:
        runtime.close()
        for path in paths:
            path.unlink(missing_ok=True)


def test_concurrent_identical_uploads_in_different_tenants_remain_independent(
    tmp_path, monkeypatch
) -> None:
    _app, runtime, _store, user_ids = _new_system(tmp_path, monkeypatch)
    manager = runtime.documents()
    content = _pdf_bytes("Shared synthetic content has tenant-specific ownership.")
    tenants = (user_ids["alpha"], user_ids["beta"])
    paths = [_stage_pdf(manager, tenant_id, content) for tenant_id in tenants]
    barrier = threading.Barrier(2)
    original_accept = manager.accept

    def synchronized_accept(path, filename, tenant_id):
        barrier.wait(timeout=5)
        return original_accept(path, filename, tenant_id)

    monkeypatch.setattr(manager, "accept", synchronized_accept)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_accept_outcome, manager, path, "shared.pdf", tenant_id)
                for path, tenant_id in zip(paths, tenants, strict=True)
            ]
            outcomes = [future.result(timeout=10) for future in futures]

        assert [status for status, _doc_id in outcomes] == [202, 202]
        doc_ids = [doc_id for _status, doc_id in outcomes]
        assert doc_ids[0] != doc_ids[1]
        source_id = document_id_from_hash(sha256_for_file(str(paths[0])))
        assert all(
            doc_id == manager._tenant_document_id(tenant_id, source_id)
            for doc_id, tenant_id in zip(doc_ids, tenants, strict=True)
        )
        assert len(manager.records(tenants[0])) == 1
        assert len(manager.records(tenants[1])) == 1
        with manager.connection() as connection:
            bindings = connection.execute(
                "SELECT id, tenant_id FROM documents ORDER BY tenant_id"
            ).fetchall()
        assert {tuple(row) for row in bindings} == {
            (doc_id, tenant_id) for doc_id, tenant_id in zip(doc_ids, tenants, strict=True)
        }
        try:
            manager.get(doc_ids[0], tenants[1])
        except DocumentError as error:
            assert error.status == 404
        else:
            raise AssertionError("A tenant must not resolve another tenant's managed ID")
    finally:
        runtime.close()
        for path in paths:
            path.unlink(missing_ok=True)


def test_delete_waits_for_inflight_ingestion_and_retrieval_waits_for_delete(
    tmp_path, monkeypatch
) -> None:
    _app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    tenant_id = user_ids["alpha"]
    manager = runtime.documents()
    path = _stage_pdf(
        manager, tenant_id, _pdf_bytes("Concurrent lifecycle evidence remains tenant scoped.")
    )
    record = manager.accept(path, "concurrent.pdf", tenant_id)
    sync_entered = threading.Event()
    allow_sync = threading.Event()
    original_sync = store.sync

    def blocking_sync(*args, **kwargs):
        sync_entered.set()
        if not allow_sync.wait(timeout=5):
            raise TimeoutError("synthetic synchronization gate timed out")
        return original_sync(*args, **kwargs)

    monkeypatch.setattr(store, "sync", blocking_sync)
    delete_entered = threading.Event()
    delete_finished = threading.Event()
    original_delete = manager.delete

    def tracked_delete(doc_id, request_tenant):
        delete_entered.set()
        try:
            return original_delete(doc_id, request_tenant)
        finally:
            delete_finished.set()

    monkeypatch.setattr(manager, "delete", tracked_delete)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            ingestion = pool.submit(manager.process, record.doc_id, tenant_id)
            assert sync_entered.wait(timeout=5)
            deletion = pool.submit(manager.delete, record.doc_id, tenant_id)
            assert delete_entered.wait(timeout=5)
            assert not delete_finished.is_set()
            allow_sync.set()
            ingestion.result(timeout=10)
            deleted = deletion.result(timeout=10)

        assert deleted.status == "deleted"
        assert store.scroll_ids(record.doc_id, tenant_id=tenant_id) == []
        assert not manager._file(record.doc_id, tenant_id).exists()
        hits = runtime.retrieve(
            RetrievalRequest(question="concurrent lifecycle evidence", strategy="dense"),
            tenant_id=tenant_id,
        )
        assert hits == []
    finally:
        allow_sync.set()
        runtime.close()


def test_retrieval_started_during_vector_deletion_returns_no_stale_hits(tmp_path, monkeypatch):
    _app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    tenant_id = user_ids["alpha"]
    manager = runtime.documents()
    path = _stage_pdf(
        manager, tenant_id, _pdf_bytes("Evidence becomes unavailable after deletion.")
    )
    record = manager.accept(path, "delete-race.pdf", tenant_id)
    manager.process(record.doc_id, tenant_id)
    before = runtime.retrieve(
        RetrievalRequest(question="evidence after deletion", strategy="dense"),
        tenant_id=tenant_id,
    )
    assert before and before[0].doc_id == record.doc_id

    erase_entered = threading.Event()
    allow_erase = threading.Event()
    original_erase = manager._erase_vectors

    def blocking_erase(doc_id, request_tenant):
        erase_entered.set()
        if not allow_erase.wait(timeout=5):
            raise TimeoutError("synthetic deletion gate timed out")
        return original_erase(doc_id, request_tenant)

    monkeypatch.setattr(manager, "_erase_vectors", blocking_erase)
    retrieval_entered = threading.Event()

    def retrieve_during_delete():
        retrieval_entered.set()
        return runtime.retrieve(
            RetrievalRequest(question="evidence after deletion", strategy="dense"),
            tenant_id=tenant_id,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            deletion = pool.submit(manager.delete, record.doc_id, tenant_id)
            assert erase_entered.wait(timeout=5)
            retrieval = pool.submit(retrieve_during_delete)
            assert retrieval_entered.wait(timeout=5)
            assert not retrieval.done()
            allow_erase.set()
            assert deletion.result(timeout=10).status == "deleted"
            assert retrieval.result(timeout=10) == []
        assert store.scroll_ids(record.doc_id, tenant_id=tenant_id) == []
    finally:
        allow_erase.set()
        runtime.close()
        path.unlink(missing_ok=True)
