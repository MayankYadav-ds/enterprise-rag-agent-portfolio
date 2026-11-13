"""Offline authenticated document lifecycle integration tests.

The FastAPI routes, DocumentManager, PDF ingestion/chunking, DenseRetriever,
authorization and citation validation are production code. Only the vector
store, embedder and answer provider are deterministic in-memory test adapters.
The Qdrant-specific deletion adapter is replaced because importing its native
gRPC client is blocked in this Windows test environment.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pymupdf as fitz
import pytest
import tiktoken
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_generator_factory
from rag_agent.api.documents import DocumentManager
from rag_agent.api.documents import chunk_document as production_chunk_document
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.generation import AnswerGenerator
from rag_agent.retrieval.dense import DenseRetriever

ORIGIN = "http://localhost:5173"
PASSWORD = "phase-10-10-synthetic-password"


class _WordTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class _MemoryVectorStore:
    """Small interface-compatible store with explicit tenant filtering."""

    def __init__(self) -> None:
        self.points: list[dict] = []
        self.fail_sync_after: int | None = None
        self.delete_failures = 0
        self.sync_calls = 0

    def scroll_ids(self, doc_id: str | None = None, *, tenant_id: str | None = None):
        return [
            item["chunk_id"]
            for item in self.points
            if (doc_id is None or item["doc_id"] == doc_id)
            and (tenant_id is None or item["tenant_id"] == tenant_id)
        ]

    def scroll_payloads(self, doc_id: str | None = None, *, tenant_id: str | None = None):
        selected = [
            item
            for item in self.points
            if (doc_id is None or item["doc_id"] == doc_id)
            and (tenant_id is None or item["tenant_id"] == tenant_id)
        ]
        return {item["chunk_id"]: dict(item) for item in selected}

    def sync(self, doc_id, chunks, embedder, *, source_file: str, tenant_id: str) -> None:
        self.sync_calls += 1
        if self.fail_sync_after is not None and len(chunks) < 2:
            raise AssertionError("partial-write fixture must produce multiple chunks")
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            self.points.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": doc_id,
                    "source_file": source_file,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_heading": chunk.section_heading,
                    "subheading": chunk.subheading,
                    "chunk_type": chunk.chunk_type,
                    "text": chunk.text,
                    "tenant_id": tenant_id,
                    "model_name": embedder.model_name,
                    "vector": vector,
                }
            )
            if self.fail_sync_after == index:
                self.fail_sync_after = None
                raise RuntimeError("synthetic vector batch interrupted")

    def delete_document(self, doc_id: str, *, tenant_id: str) -> None:
        if self.delete_failures:
            self.delete_failures -= 1
            raise RuntimeError("synthetic vector deletion interrupted")
        self.points = [
            item
            for item in self.points
            if not (item["doc_id"] == doc_id and item["tenant_id"] == tenant_id)
        ]

    def search(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[SimpleNamespace]:
        if not tenant_id:
            return []
        matches = [
            item
            for item in self.points
            if item["tenant_id"] == tenant_id
            and (doc_id is None or item["doc_id"] == doc_id)
            and (chunk_type is None or item["chunk_type"] == chunk_type)
        ]
        return [SimpleNamespace(score=0.9, payload=dict(item)) for item in matches[:top_k]]


class _EvidenceProvider:
    def complete(self, messages: list[dict[str, str]], *, schema: dict) -> str:
        request = json.loads(messages[-1]["content"])
        evidence = request["evidence"]
        if not evidence:
            return json.dumps({"status": "insufficient_evidence", "selections": []})
        item = evidence[0]
        return json.dumps(
            {
                "status": "answered",
                "selections": [{"evidence_id": item["evidence_id"], "quote": item["text"]}],
            }
        )


def _pdf_bytes(text: str = "Synthetic tenant document fact.", *, pages: int = 1) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page_text = text if pages == 1 else f"Page {index + 1}. {text}"
        if page_text.strip():
            page_text += (
                " This synthetic document records controlled lifecycle evidence for tenant "
                "ownership and retrieval checks. It contains only disposable test facts and "
                "no personal information."
            )
        for line_number, line in enumerate(page_text.splitlines() or [" "]):
            # Keep synthetic body text clear of the parser's repeated-edge
            # region; PyMuPDF groups adjacent lines into one text block.
            page.insert_text((48, 120 + line_number * 12), line, fontsize=8)
    data = document.tobytes()
    document.close()
    return data


def _many_page_pdf_bytes() -> bytes:
    lines = ["synthetic lifecycle evidence tokens are disposable" for _ in range(44)]
    return _pdf_bytes("\n".join(lines), pages=2)


def _new_system(tmp_path, monkeypatch, users=("alpha", "beta", "gamma")):
    store = _MemoryVectorStore()
    embedder = FakeEmbedder()
    settings = APISettings(
        document_dir=tmp_path / "managed",
        upload_max_bytes=64 * 1024,
        environment="test",
    )
    app = create_app(settings)
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    user_ids: dict[str, str] = {}
    with runtime.connection() as connection:
        for label in users:
            user_ids[label] = create_user(
                connection,
                f"lifecycle-{label}",
                hash_password(PASSWORD),
                is_admin=(label == "alpha"),
            )

    monkeypatch.setattr(runtime, "_store", lambda _embedder, *, validate: store)
    monkeypatch.setattr(runtime, "embedding", lambda: embedder)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: _WordTokenizer())

    def erase_vectors(manager: DocumentManager, doc_id: str, tenant_id: str) -> None:
        manager.runtime._store(FakeEmbedder(), validate=False).delete_document(
            doc_id, tenant_id=tenant_id
        )

    monkeypatch.setattr(DocumentManager, "_erase_vectors", erase_vectors)
    app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(_EvidenceProvider())
    )
    return app, runtime, store, user_ids


def _login(client: TestClient, label: str) -> tuple[str, str]:
    response = client.post(
        "/auth/login",
        headers={"Origin": ORIGIN},
        json={"username": f"lifecycle-{label}", "password": PASSWORD},
    )
    assert response.status_code == 200
    return response.cookies["session"], response.json()["csrf_token"]


def _set_session(client: TestClient, session: tuple[str, str]) -> None:
    client.cookies.set("session", session[0])
    client.headers["X-CSRF-Token"] = session[1]


def _upload(client: TestClient, session: tuple[str, str], data: bytes, **params):
    _set_session(client, session)
    return client.post(
        "/api/v1/documents",
        params={"filename": "shared-report.pdf", **params},
        content=data,
        headers={"content-type": "application/pdf"},
    )


def _query(client: TestClient, session: tuple[str, str], question: str):
    _set_session(client, session)
    return client.post("/api/v1/query", json={"question": question})


def test_authenticated_upload_duplicates_tenant_ownership_and_post_delete_retrieval(
    tmp_path, monkeypatch
) -> None:
    app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch)
    with TestClient(app) as client:
        sessions = {label: _login(client, label) for label in user_ids}
        alpha_pdf = _pdf_bytes("Alpha synthetic fact: value 17.")
        beta_pdf = _pdf_bytes("Beta synthetic fact: value 82.")

        # Extra tenant input is ignored by the raw upload route; ownership comes
        # from the authenticated session, never the query string.
        alpha_upload = _upload(client, sessions["alpha"], alpha_pdf, tenant_id=user_ids["beta"])
        beta_upload = _upload(client, sessions["beta"], beta_pdf)
        gamma_upload = _upload(client, sessions["gamma"], alpha_pdf)
        assert [r.status_code for r in (alpha_upload, beta_upload, gamma_upload)] == [202] * 3

        alpha_id = alpha_upload.json()["doc_id"]
        beta_id = beta_upload.json()["doc_id"]
        gamma_id = gamma_upload.json()["doc_id"]
        assert alpha_id != beta_id
        assert alpha_id != gamma_id  # same bytes, distinct server-derived tenants

        _set_session(client, sessions["alpha"])
        alpha_record = client.get(f"/api/v1/documents/{alpha_id}").json()
        alpha_inventory = client.get("/api/v1/documents").json()
        _set_session(client, sessions["beta"])
        beta_record = client.get(f"/api/v1/documents/{beta_id}").json()
        beta_inventory = client.get("/api/v1/documents").json()
        _set_session(client, sessions["gamma"])
        gamma_record = client.get(f"/api/v1/documents/{gamma_id}").json()
        assert {alpha_record["status"], beta_record["status"], gamma_record["status"]} == {"ready"}
        assert alpha_record["chunk_count"] > 0
        assert [record["doc_id"] for record in alpha_inventory] == [alpha_id]
        assert [record["doc_id"] for record in beta_inventory] == [beta_id]
        assert [record["doc_id"] for record in client.get("/api/v1/documents").json()] == [gamma_id]
        _set_session(client, sessions["beta"])
        assert client.get(f"/api/v1/documents/{alpha_id}").status_code == 404
        assert all(
            point["tenant_id"] == user_ids["alpha"]
            for point in store.points
            if point["doc_id"] == alpha_id
        )

        # A synthetic collision models a legacy/external writer reusing both
        # identifiers. Tenant filtering must still separate the payloads.
        alpha_point = next(point for point in store.points if point["doc_id"] == alpha_id)
        beta_collision = dict(alpha_point)
        beta_collision.update(
            tenant_id=user_ids["beta"],
            text="Beta synthetic collision fixture: separate tenant payload.",
        )
        store.points.append(beta_collision)
        assert (beta_collision["doc_id"], beta_collision["chunk_id"]) == (
            alpha_point["doc_id"],
            alpha_point["chunk_id"],
        )
        retriever = DenseRetriever(FakeEmbedder(), store)
        alpha_hits = retriever.retrieve("synthetic fact", tenant_id=user_ids["alpha"])
        beta_hits = retriever.retrieve("synthetic fact", tenant_id=user_ids["beta"])
        assert all("Beta synthetic collision fixture" not in hit.text for hit in alpha_hits)
        assert any("Beta synthetic collision fixture" in hit.text for hit in beta_hits)

        # Identical content is rejected within one tenant, but was accepted as
        # a distinct tenant-scoped object for gamma.
        duplicates_before = len(store.points)
        duplicate = _upload(client, sessions["alpha"], alpha_pdf)
        assert duplicate.status_code == 409
        assert len(store.points) == duplicates_before

        alpha_answer = _query(client, sessions["alpha"], "What does Alpha report?")
        beta_answer = _query(client, sessions["beta"], "What does Beta report?")
        assert alpha_answer.status_code == beta_answer.status_code == 200
        assert "Alpha synthetic fact" in alpha_answer.json()["citations"][0]["quote"]
        assert "Beta synthetic fact" not in alpha_answer.text
        assert "Beta synthetic fact" in beta_answer.json()["citations"][0]["quote"]
        assert "Alpha synthetic fact" not in beta_answer.text
        assert "Beta synthetic collision fixture" not in alpha_answer.text

        # Another tenant cannot delete the record even when it knows the ID.
        _set_session(client, sessions["beta"])
        denied = client.delete(f"/api/v1/documents/{alpha_id}")
        assert denied.status_code == 404
        assert store.scroll_ids(alpha_id, tenant_id=user_ids["alpha"])

        _set_session(client, sessions["alpha"])
        deleted = client.delete(f"/api/v1/documents/{alpha_id}")
        assert deleted.status_code == 200 and deleted.json()["status"] == "deleted"
        assert store.scroll_ids(alpha_id, tenant_id=user_ids["alpha"]) == []
        after_delete = _query(client, sessions["alpha"], "What does Alpha report?")
        assert after_delete.status_code == 200
        assert after_delete.json()["status"] == "insufficient_evidence"
        assert after_delete.json()["citations"] == []
        repeated_delete = client.delete(f"/api/v1/documents/{alpha_id}")
        assert repeated_delete.status_code == 200
        assert repeated_delete.json()["status"] == "deleted"
    app.dependency_overrides.clear()
    runtime.close()


@pytest.mark.parametrize("failure", ["parse", "chunk", "embedding", "partial-vector", "publish"])
def test_ingestion_failure_is_not_published_and_recovery_can_retry(
    tmp_path, monkeypatch, failure: str
) -> None:
    app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    if failure == "partial-vector":
        data = _many_page_pdf_bytes()
    elif failure == "parse":
        data = b"%PDF-deliberately-corrupt-synthetic-input"
    else:
        data = _pdf_bytes("Retryable synthetic fact.")

    if failure == "chunk":
        failed_once = False

        def fail_chunk_once(*args, **kwargs):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("synthetic chunk failure")
            return production_chunk_document(*args, **kwargs)

        monkeypatch.setattr("rag_agent.api.documents.chunk_document", fail_chunk_once)
    elif failure == "embedding":
        original_embedding = runtime.embedding
        failed_once = False

        def fail_embedding_once():
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("synthetic embedding failure")
            return original_embedding()

        monkeypatch.setattr(runtime, "embedding", fail_embedding_once)
    elif failure == "partial-vector":
        store.fail_sync_after = 1
    elif failure == "publish":
        original_save = DocumentManager._save
        failed_once = False

        def fail_final_save(manager, record, tenant_id, file=None):
            nonlocal failed_once
            if record.status == "ready" and not failed_once:
                failed_once = True
                raise RuntimeError("synthetic publication failure")
            return original_save(manager, record, tenant_id, file)

        monkeypatch.setattr(DocumentManager, "_save", fail_final_save)

    with TestClient(app) as client:
        session = _login(client, "alpha")
        response = _upload(client, session, data)
        assert response.status_code == 202
        doc_id = response.json()["doc_id"]
        record = client.get(f"/api/v1/documents/{doc_id}").json()
        assert record["status"] == "failed"
        assert record["owns_vectors"] is False
        assert store.scroll_ids(doc_id, tenant_id=user_ids["alpha"]) == []
        query = _query(client, session, "Retryable synthetic fact?")
        assert query.status_code == 200
        assert query.json()["status"] == "insufficient_evidence"

        # The implementation permits re-upload after a failed attempt. The
        # same content identity is reused; duplicate active records are not.
        retried = _upload(client, session, data)
        assert retried.status_code == 202
        assert retried.json()["doc_id"] == doc_id
        retry_state = client.get(f"/api/v1/documents/{doc_id}").json()
        if failure == "parse":
            assert retry_state["status"] == "failed"
        else:
            # One-shot injected store/publication failures are consumed; retry
            # completes through the actual production processing path.
            assert retry_state["status"] == "ready"
            assert (
                len(store.scroll_ids(doc_id, tenant_id=user_ids["alpha"]))
                == retry_state["chunk_count"]
            )
    app.dependency_overrides.clear()
    runtime.close()


def test_malformed_empty_oversized_unsupported_and_duplicate_upload_contracts(
    tmp_path, monkeypatch
) -> None:
    app, runtime, store, _user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    with TestClient(app) as client:
        session = _login(client, "alpha")

        client.cookies.clear()
        assert (
            client.post(
                "/api/v1/documents",
                params={"filename": "escape.pdf"},
                content=b"%PDF-x",
                headers={"content-type": "application/pdf"},
            ).status_code
            == 401
        )

        _set_session(client, session)
        no_csrf = client.post(
            "/api/v1/documents",
            params={"filename": "bad.pdf"},
            content=b"%PDF-x",
            headers={"X-CSRF-Token": ""},
        )
        assert no_csrf.status_code == 403
        invalid_csrf = client.post(
            "/api/v1/documents",
            params={"filename": "bad.pdf"},
            content=b"%PDF-x",
            headers={"X-CSRF-Token": "incorrect"},
        )
        assert invalid_csrf.status_code == 403

        unsupported = client.post(
            "/api/v1/documents",
            params={"filename": "bad.pdf"},
            content=b"text",
            headers={"X-CSRF-Token": session[1], "content-type": "text/plain"},
        )
        assert unsupported.status_code == 415
        bad_signature = _upload(client, session, b"not a pdf")
        assert bad_signature.status_code == 415
        traversal = _upload(client, session, _pdf_bytes(), filename="..\\escape.pdf")
        assert traversal.status_code == 422

        oversized = client.post(
            "/api/v1/documents",
            params={"filename": "large.pdf"},
            content=b"%PDF-" + b"x" * (settings_limit := 64 * 1024),
            headers={"content-type": "application/pdf", "X-CSRF-Token": session[1]},
        )
        assert len(oversized.request.content) > settings_limit
        assert oversized.status_code == 413

        empty_page = _pdf_bytes(" ")
        empty_response = _upload(client, session, empty_page, filename="empty.pdf")
        assert empty_response.status_code == 202
        empty_record = client.get(f"/api/v1/documents/{empty_response.json()['doc_id']}").json()
        assert empty_record["status"] == "failed"
        assert empty_record["chunk_count"] == 0
        assert store.points == []


def test_partial_sync_and_delete_failures_block_retrieval_until_recovery(tmp_path, monkeypatch):
    app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    data = _many_page_pdf_bytes()
    assert len(data) < 64 * 1024
    store.fail_sync_after = 1
    store.delete_failures = 1
    with TestClient(app) as client:
        session = _login(client, "alpha")
        uploaded = _upload(client, session, data)
        assert uploaded.status_code == 202
        doc_id = uploaded.json()["doc_id"]
        cleanup_required = client.get(f"/api/v1/documents/{doc_id}").json()
        assert cleanup_required["status"] == "cleanup_required"
        assert cleanup_required["owns_vectors"] is True
        assert store.scroll_ids(doc_id, tenant_id=user_ids["alpha"])
        blocked = _query(client, session, "synthetic lifecycle evidence")
        assert blocked.status_code == 503

        _set_session(client, session)
        recovered = client.post("/api/v1/documents/recover")
        assert recovered.status_code == 204
        failed = client.get(f"/api/v1/documents/{doc_id}").json()
        assert failed["status"] == "failed" and failed["owns_vectors"] is False
        assert store.scroll_ids(doc_id, tenant_id=user_ids["alpha"]) == []
        abstained = _query(client, session, "synthetic lifecycle evidence")
        assert abstained.status_code == 200
        assert abstained.json()["status"] == "insufficient_evidence"
        assert abstained.json()["citations"] == []

        # Re-upload succeeds after durable cleanup and uses the same content ID.
        retry = _upload(client, session, data)
        assert retry.status_code == 202 and retry.json()["doc_id"] == doc_id
        assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "ready"

        # An interrupted deletion is durable and prevents query until recovery.
        store.delete_failures = 1
        _set_session(client, session)
        delete_failed = client.delete(f"/api/v1/documents/{doc_id}")
        assert delete_failed.status_code == 503
        assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "deleting"
        assert _query(client, session, "synthetic lifecycle evidence").status_code == 503
        _set_session(client, session)
        assert client.post("/api/v1/documents/recover").status_code == 204
        assert client.get(f"/api/v1/documents/{doc_id}").json()["status"] == "deleted"
        assert _query(client, session, "synthetic lifecycle evidence").json()["citations"] == []
    app.dependency_overrides.clear()
    runtime.close()


def test_initial_metadata_persistence_failure_leaves_no_registered_document(
    tmp_path, monkeypatch
) -> None:
    app, runtime, store, user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    original_save = DocumentManager._save
    failed_once = False

    def fail_initial_save(manager, record, tenant_id, file=None):
        nonlocal failed_once
        if record.status == "queued" and not failed_once:
            failed_once = True
            raise OSError("synthetic metadata persistence failure")
        return original_save(manager, record, tenant_id, file)

    monkeypatch.setattr(DocumentManager, "_save", fail_initial_save)
    data = _pdf_bytes("Recoverable synthetic metadata persistence fact.")
    with TestClient(app) as client:
        session = _login(client, "alpha")
        failed = _upload(client, session, data)
        assert failed.status_code == 503
        _set_session(client, session)
        assert client.get("/api/v1/documents").json() == []
        assert store.points == []

        retried = _upload(client, session, data)
        assert retried.status_code == 202
        record = client.get(f"/api/v1/documents/{retried.json()['doc_id']}").json()
        assert record["status"] == "ready"
        assert (
            len(store.scroll_ids(record["doc_id"], tenant_id=user_ids["alpha"]))
            == record["chunk_count"]
        )
    app.dependency_overrides.clear()
    runtime.close()


def test_expired_session_cannot_list_or_mutate_documents(tmp_path, monkeypatch):
    app, runtime, _store, _user_ids = _new_system(tmp_path, monkeypatch, users=("alpha",))
    with TestClient(app) as client:
        session = _login(client, "alpha")
        token_hash = hashlib.sha256(session[0].encode()).hexdigest()
        expired = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        with runtime.connection() as connection:
            connection.execute(
                "UPDATE sessions SET expires_at=? WHERE token_hash=?", (expired, token_hash)
            )
        _set_session(client, session)
        assert client.get("/api/v1/documents").status_code == 401
        upload = client.post(
            "/api/v1/documents",
            params={"filename": "expired.pdf"},
            content=_pdf_bytes(),
            headers={"content-type": "application/pdf", "X-CSRF-Token": session[1]},
        )
        assert upload.status_code == 401
    app.dependency_overrides.clear()
    runtime.close()
