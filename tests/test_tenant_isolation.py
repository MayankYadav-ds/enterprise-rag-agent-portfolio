"""Tenant-bound API and Qdrant regression tests using in-memory fixtures only."""

from __future__ import annotations

import json
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import fitz
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_generator_factory
from rag_agent.api.documents import DocumentRecord
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.auth.dependencies import User, tenant_id_for_user
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password
from rag_agent.chunking.models import Chunk
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore, point_id
from rag_agent.generation.models import Answer, CitedExcerpt
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.citation import format_citation
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.reranker import FakeCrossEncoder
from rag_agent.tenancy import validate_tenant_id


def _pdf(text: str) -> bytes:
    with fitz.open() as document:
        page = document.new_page()
        page.insert_textbox(fitz.Rect(72, 72, 500, 700), (text + "\n") * 4)
        return document.tobytes()


def _document_metrics(
    retrieved_doc_ids: list[str], relevant_doc_ids: set[str]
) -> tuple[float, float, float]:
    """Compute document-level precision, recall, and reciprocal rank at k.

    Duplicate chunks from one document count once. Empty retrieved or relevant
    sets return zero for the metric whose denominator would otherwise be zero.
    """
    retrieved = list(dict.fromkeys(retrieved_doc_ids))
    relevant = set(relevant_doc_ids)
    hits = [doc_id for doc_id in retrieved if doc_id in relevant]
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    first_relevant_rank = next(
        (rank for rank, doc_id in enumerate(retrieved, start=1) if doc_id in relevant), None
    )
    reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank is not None else 0.0
    return precision, recall, reciprocal_rank


def test_document_metrics_deduplicate_chunks_and_detect_negative_controls():
    assert _document_metrics(["doc-a", "doc-a"], {"doc-a"}) == (1.0, 1.0, 1.0)
    assert _document_metrics(["doc-a", "irrelevant"], {"doc-a"}) == (0.5, 1.0, 1.0)
    assert _document_metrics(["irrelevant"], {"doc-a"}) == (0.0, 0.0, 0.0)
    assert _document_metrics([], {"doc-a"}) == (0.0, 0.0, 0.0)


class _OfflineGenerator:
    def generate(self, question, hits):
        return Answer(question, "answered", "offline test answer")


@pytest.fixture
def tenant_system(tmp_path, monkeypatch):
    app = create_app(APISettings(document_dir=tmp_path / "managed"))
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    password_hash = hash_password("tenant-test-password-42")
    with runtime.connection() as db:
        tenant_a = create_user(db, "tenant-a", password_hash)
        tenant_b = create_user(db, "tenant-b", password_hash)

    qdrant = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    QdrantStore(embedder, client=qdrant)
    runtime._client = qdrant
    monkeypatch.setattr(runtime, "embedding", lambda: embedder)
    app.dependency_overrides[get_generator_factory] = lambda: lambda: _OfflineGenerator()
    monkeypatch.setattr(
        "rag_agent.retrieval.reranker.CrossEncoderReranker._load_model",
        lambda self: FakeCrossEncoder(),
    )

    with TestClient(app) as client:
        login_a = client.post(
            "/auth/login",
            headers={"Origin": "http://localhost:5173"},
            json={"username": "tenant-a", "password": "tenant-test-password-42"},
        )
        assert login_a.status_code == 200
        cookie_a = client.cookies.get("session")
        csrf_a = login_a.json()["csrf_token"]

        login_b = client.post(
            "/auth/login",
            headers={"Origin": "http://localhost:5173"},
            json={"username": "tenant-b", "password": "tenant-test-password-42"},
        )
        assert login_b.status_code == 200
        cookie_b = client.cookies.get("session")
        csrf_b = login_b.json()["csrf_token"]

        yield client, runtime, qdrant, tenant_a, tenant_b, cookie_a, csrf_a, cookie_b, csrf_b


def _as_tenant(client, cookie: str | None, csrf: str | None) -> None:
    client.cookies.clear()
    if cookie is not None:
        client.cookies.set("session", cookie)
    client.headers.pop("X-CSRF-Token", None)
    if csrf is not None:
        client.headers["X-CSRF-Token"] = csrf


def _upload(client, content: bytes, filename: str):
    return client.post(
        "/api/v1/documents",
        params={"filename": filename},
        content=content,
        headers={"content-type": "application/pdf"},
    )


def test_two_tenants_are_isolated_across_inventory_status_delete_and_retrieval(tenant_system):
    client, runtime, qdrant, tenant_a, tenant_b, cookie_a, csrf_a, cookie_b, csrf_b = tenant_system
    shared_pdf = _pdf("sharedword alphaonly")

    _as_tenant(client, cookie_a, csrf_a)
    upload_a = _upload(client, shared_pdf, "shared.pdf")
    assert upload_a.status_code == 202
    doc_a = upload_a.json()["doc_id"]
    inventory_a = client.get("/api/v1/documents").json()
    assert [record["doc_id"] for record in inventory_a] == [doc_a]
    for strategy in ("dense", "bm25", "hybrid"):
        result = client.post(
            "/api/v1/retrieve",
            json={"question": "sharedword", "strategy": strategy},
        )
        assert result.status_code == 200
        assert {hit["doc_id"] for hit in result.json()["evidence"]} == {doc_a}

    _as_tenant(client, cookie_b, csrf_b)
    upload_b = _upload(client, shared_pdf, "shared.pdf")
    assert upload_b.status_code == 202
    doc_b = upload_b.json()["doc_id"]
    assert doc_a != doc_b
    assert [record["doc_id"] for record in client.get("/api/v1/documents").json()] == [doc_b]
    assert client.get(f"/api/v1/documents/{doc_a}").status_code == 404
    assert client.delete(f"/api/v1/documents/{doc_a}").status_code == 404
    assert client.get(f"/api/v1/documents/{doc_b}").status_code == 200

    for strategy in ("dense", "bm25", "hybrid"):
        result = client.post(
            "/api/v1/retrieve",
            json={"question": "sharedword", "strategy": strategy},
        )
        assert result.status_code == 200
        assert {hit["doc_id"] for hit in result.json()["evidence"]} == {doc_b}

    for strategy in ("dense", "bm25", "hybrid"):
        query = client.post(
            "/api/v1/query",
            json={
                "question": "sharedword",
                "strategy": strategy,
                "rerank": True,
            },
        )
        assert query.status_code == 200
        assert {hit["doc_id"] for hit in query.json()["evidence"]} == {doc_b}
        assert query.json()["answer"] == "offline test answer"

    _as_tenant(client, cookie_a, csrf_a)
    assert client.get(f"/api/v1/documents/{doc_a}").status_code == 200
    assert client.delete(f"/api/v1/documents/{doc_b}").status_code == 404
    assert [record["doc_id"] for record in client.get("/api/v1/documents").json()] == [doc_a]
    removed = client.delete(f"/api/v1/documents/{doc_a}")
    assert removed.status_code == 200
    assert runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_b, tenant_id=tenant_b)
    assert not runtime._store(FakeEmbedder(), validate=False).scroll_ids(doc_a, tenant_id=tenant_a)


def test_answer_generation_and_citations_receive_only_the_authenticated_tenant(tenant_system):
    client, _, _, _, _, cookie_a, csrf_a, cookie_b, csrf_b = tenant_system
    shared_pdf = _pdf("sharedword answer-evidence")
    _as_tenant(client, cookie_a, csrf_a)
    upload_a = _upload(client, shared_pdf, "tenant-a-source.pdf")
    assert upload_a.status_code == 202
    doc_a = upload_a.json()["doc_id"]

    _as_tenant(client, cookie_b, csrf_b)
    upload_b = _upload(client, shared_pdf, "tenant-b-source.pdf")
    assert upload_b.status_code == 202
    doc_b = upload_b.json()["doc_id"]

    observed: list[list[str]] = []

    class CapturingGenerator:
        def generate(self, question, hits):
            observed.append([hit.doc_id for hit in hits])
            hit = hits[0]
            excerpt = CitedExcerpt(
                evidence_id="E1",
                chunk_id=hit.chunk_id,
                source_file=hit.source_file,
                page_start=hit.page_start,
                page_end=hit.page_end,
                quote=hit.text,
                citation=format_citation(hit),
            )
            return Answer(question, "answered", hit.text, excerpts=(excerpt,), evidence_ids=("E1",))

    client.app.dependency_overrides[get_generator_factory] = lambda: lambda: CapturingGenerator()
    for strategy in ("dense", "bm25", "hybrid"):
        response = client.post(
            "/api/v1/query",
            json={"question": "sharedword", "strategy": strategy, "rerank": True},
        )
        assert response.status_code == 200
        body = response.json()
        assert {hit["doc_id"] for hit in body["evidence"]} == {doc_b}
        assert {doc_id for ids in observed for doc_id in ids} == {doc_b}
        assert [citation["source_file"] for citation in body["citations"]] == [
            "tenant-b-source.pdf"
        ]
        assert doc_a not in {hit["doc_id"] for hit in body["evidence"]}


def test_concurrent_runtime_retrieval_keeps_tenant_cache_and_filters_separate(tenant_system):
    client, runtime, _, tenant_a, tenant_b, cookie_a, csrf_a, cookie_b, csrf_b = tenant_system
    shared_pdf = _pdf("sharedword tenant-isolated-concurrency")
    _as_tenant(client, cookie_a, csrf_a)
    upload_a = _upload(client, shared_pdf, "same.pdf")
    assert upload_a.status_code == 202
    doc_a = upload_a.json()["doc_id"]
    _as_tenant(client, cookie_b, csrf_b)
    upload_b = _upload(client, shared_pdf, "same.pdf")
    assert upload_b.status_code == 202
    doc_b = upload_b.json()["doc_id"]

    barrier = Barrier(8)
    requests = [(tenant_a, doc_a), (tenant_b, doc_b)] * 4

    def retrieve_for(tenant_id: str) -> set[str]:
        barrier.wait(timeout=5)
        hits = runtime.retrieve(
            RetrievalRequest(question="sharedword", strategy="hybrid"),
            tenant_id=tenant_id,
        )
        return {hit.doc_id for hit in hits}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda pair: retrieve_for(pair[0]), requests))
    assert all(result == {doc_id} for result, (_, doc_id) in zip(results, requests, strict=True))


def test_qdrant_scope_rejects_malformed_identity_before_query(tenant_system):
    _, runtime, _, *_ = tenant_system
    store = runtime._store(FakeEmbedder(), validate=False)
    with pytest.raises(TypeError):
        runtime.retrieve(RetrievalRequest(question="private query"))
    with pytest.raises(ValueError, match="Invalid tenant identity"):
        store.scroll_payloads(tenant_id="../not-a-tenant")
    with pytest.raises(ValueError, match="Invalid tenant identity"):
        store.search("private query", tenant_id="A" * 32)


def test_dense_and_bm25_isolate_identical_document_and_chunk_ids(tenant_system):
    _, runtime, qdrant, tenant_a, tenant_b, *_ = tenant_system
    embedder = FakeEmbedder()
    store = runtime._store(embedder, validate=False)
    vector = embedder.embed_query("sharedword")
    shared = {
        "chunk_id": "same-chunk-id",
        "doc_id": "same-document-id",
        "source_file": "same-name.pdf",
        "chunk_type": "text",
        "page_start": 1,
        "page_end": 1,
        "embedding_model": embedder.model_name,
    }
    qdrant.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**shared, "tenant_id": tenant_a, "text": "sharedword ALPHA-ONLY"},
            ),
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={**shared, "tenant_id": tenant_b, "text": "sharedword BETA-ONLY"},
            ),
        ],
    )

    for tenant_id, own_marker, other_marker in (
        (tenant_a, "ALPHA-ONLY", "BETA-ONLY"),
        (tenant_b, "BETA-ONLY", "ALPHA-ONLY"),
    ):
        dense = DenseRetriever(embedder, store).retrieve("sharedword", tenant_id=tenant_id)
        bm25 = BM25Retriever(
            store.scroll_payloads(tenant_id=tenant_id), tenant_id=tenant_id
        ).retrieve("sharedword", tenant_id=tenant_id)
        assert len(dense) == len(bm25) == 1
        assert dense[0].chunk_id == bm25[0].chunk_id == "same-chunk-id"
        assert own_marker in dense[0].text and other_marker not in dense[0].text
        assert own_marker in bm25[0].text and other_marker not in bm25[0].text


def test_legacy_unscoped_records_and_points_are_not_exposed(tenant_system):
    client, runtime, qdrant, tenant_a, _, cookie_a, csrf_a, *_ = tenant_system
    legacy_doc = "doc_" + "f" * 64
    chunk_id = "legacy-unscoped-chunk"
    embedder = FakeEmbedder()
    qdrant.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=point_id(chunk_id),
                vector=embedder.embed_query("legacyword"),
                payload={
                    "chunk_id": chunk_id,
                    "doc_id": legacy_doc,
                    "source_file": "legacy.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "chunk_type": "text",
                    "text": "legacyword private legacy evidence",
                    "embedding_model": embedder.model_name,
                },
            )
        ],
    )
    # A pre-tenant SQLite record is explicitly quarantined as NULL, not claimed.
    service = runtime.documents()
    with service.connection() as db:
        db.execute(
            "INSERT INTO documents (id, record, file, tenant_id) VALUES (?, ?, ?, NULL)",
            (
                legacy_doc,
                '{"doc_id":"' + legacy_doc + '","filename":"legacy.pdf","status":"ready"}',
                "f" * 32 + ".pdf",
            ),
        )
    _as_tenant(client, cookie_a, csrf_a)
    assert service.records(tenant_a) == []
    assert client.get("/api/v1/documents").json() == []
    for strategy in ("dense", "bm25", "hybrid"):
        response = client.post(
            "/api/v1/retrieve", json={"question": "legacyword", "strategy": strategy}
        )
        assert response.status_code == 200
        assert response.json()["evidence"] == []


def test_old_registry_schema_migration_leaves_legacy_row_unassigned(tmp_path):
    document_dir = tmp_path / "managed"
    document_dir.mkdir()
    database = document_dir / "inventory.sqlite3"
    settings = APISettings(document_dir=document_dir)
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE documents (id TEXT PRIMARY KEY, record TEXT, file TEXT)")
        db.execute("CREATE TABLE binding (target TEXT PRIMARY KEY)")
        db.execute(
            "INSERT INTO binding VALUES (?)",
            (json.dumps([settings.qdrant_url, settings.collection]),),
        )
        legacy_id = "doc_" + "e" * 64
        record = DocumentRecord(doc_id=legacy_id, filename="old.pdf", status="ready")
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?)",
            (legacy_id, record.model_dump_json(), "a" * 32 + ".pdf"),
        )
    runtime = create_app(settings).state.runtime
    service = runtime.documents()
    with service.connection() as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
        tenant_value = db.execute(
            "SELECT tenant_id FROM documents WHERE id=?", (legacy_id,)
        ).fetchone()[0]
    assert "tenant_id" in columns and tenant_value is None
    assert service.records("a" * 32) == []
    with pytest.raises(Exception) as exc:
        service.get(legacy_id, "a" * 32)
    assert getattr(exc.value, "status", None) == 404
    runtime.close()


def test_tenant_ingest_refuses_global_point_id_collision_with_legacy_point(tenant_system):
    _, runtime, qdrant, tenant_a, *_ = tenant_system
    embedder = FakeEmbedder()
    store = runtime._store(embedder, validate=False)
    doc_id = "doc_" + "1" * 64
    chunk = Chunk.create(doc_id, "collision evidence", "text", 1, 1, 0)
    qdrant.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=point_id(chunk.chunk_id),
                vector=embedder.embed_query(chunk.text),
                payload={"doc_id": doc_id, "chunk_id": chunk.chunk_id, "text": "legacy"},
            )
        ],
    )
    with pytest.raises(ValueError, match="already owned outside this tenant"):
        store.sync(doc_id, [chunk], embedder, tenant_id=tenant_a)
    preserved = qdrant.retrieve("rag_chunks", ids=[point_id(chunk.chunk_id)], with_payload=True)
    assert preserved[0].payload["text"] == "legacy"
    assert "tenant_id" not in preserved[0].payload


def test_client_cannot_supply_tenant_id_or_access_without_session(tenant_system):
    client, _, _, _, _, cookie_a, csrf_a, *_ = tenant_system
    _as_tenant(client, cookie_a, csrf_a)
    forged = client.post(
        "/api/v1/retrieve",
        json={"question": "sharedword", "tenant_id": "b" * 32},
    )
    assert forged.status_code == 422
    _as_tenant(client, None, None)
    assert client.get("/api/v1/documents").status_code == 401
    assert client.post("/api/v1/retrieve", json={"question": "sharedword"}).status_code == 401


@pytest.mark.parametrize("value", ["", "b" * 31, "B" * 32, "../" + "a" * 29, None])
def test_malformed_tenant_identity_fails_closed(value):
    with pytest.raises(ValueError, match="Invalid tenant identity"):
        validate_tenant_id(value)


def test_authenticated_user_with_malformed_tenant_id_is_forbidden():
    with pytest.raises(Exception) as exc:
        tenant_id_for_user(User(id="invalid", username="u", is_admin=False, session_hash="h"))
    assert getattr(exc.value, "status_code", None) == 403


def test_offline_pdf_to_answer_pipeline_is_tenant_scoped(tenant_system):
    from rag_agent.generation.engine import AnswerGenerator

    client, runtime, qdrant, tenant_a, tenant_b, cookie_a, csrf_a, cookie_b, csrf_b = tenant_system
    claims = {
        tenant_a: (
            "Atlas records are retained for 5 years. Reference ALPHA-501.",
            "Atlas records are audited every 90 days. Reference ALPHA-502.",
        ),
        tenant_b: (
            "Atlas records are retained for 9 years. Reference BETA-901.",
            "Atlas records are audited every 30 days. Reference BETA-902.",
        ),
    }

    def pdf_for(tenant_id: str) -> bytes:
        with fitz.open() as document:
            for claim in claims[tenant_id]:
                page = document.new_page()
                text = claim + " " + ("Synthetic Atlas policy evidence. " * 8)
                page.insert_textbox(fitz.Rect(60, 60, 540, 740), text, fontsize=10)
            return document.tobytes()

    class EvidenceSelectingLLM:
        def complete(self, messages, *, schema):
            import json

            request = json.loads(messages[1]["content"])
            selections = []
            for item in request["evidence"]:
                for claim in (*claims[tenant_a], *claims[tenant_b]):
                    if claim in item["text"]:
                        selections.append({"evidence_id": item["evidence_id"], "quote": claim})
            return json.dumps(
                {
                    "status": "answered" if selections else "insufficient_evidence",
                    "selections": selections,
                }
            )

    client.app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(EvidenceSelectingLLM())
    )
    uploaded = {}
    for tenant_id, cookie, csrf in (
        (tenant_a, cookie_a, csrf_a),
        (tenant_b, cookie_b, csrf_b),
    ):
        _as_tenant(client, cookie, csrf)
        response = _upload(client, pdf_for(tenant_id), "shared-policy.pdf")
        assert response.status_code == 202
        doc_id = response.json()["doc_id"]
        uploaded[tenant_id] = doc_id
        status = client.get(f"/api/v1/documents/{doc_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "ready"
        assert status.json()["page_count"] == 2
        assert status.json()["chunk_count"] >= 2
    assert uploaded[tenant_a] != uploaded[tenant_b]

    store = runtime._store(FakeEmbedder(), validate=False)
    for tenant_id in (tenant_a, tenant_b):
        payloads = store.scroll_payloads(tenant_id=tenant_id)
        own = [payload for payload in payloads.values() if payload["doc_id"] == uploaded[tenant_id]]
        assert len(own) >= 2
        assert all(payload["tenant_id"] == tenant_id for payload in own)
        assert all(payload["page_start"] >= 1 for payload in own)
        assert all(payload["page_end"] >= payload["page_start"] for payload in own)

    # An unscoped legacy point shares the query vocabulary but must not be evidence.
    embedder = FakeEmbedder()
    qdrant.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedder.embed_query("Atlas records retention audit"),
                payload={
                    "chunk_id": "legacy-unscoped-atlas",
                    "doc_id": "legacy-doc",
                    "source_file": "legacy.pdf",
                    "chunk_type": "text",
                    "page_start": 1,
                    "page_end": 1,
                    "text": "Atlas records are retained for 99 years. Legacy unscoped evidence.",
                },
            )
        ],
    )
    runtime.invalidate_retrieval()

    cases = 0
    metric_cases = []
    response_surfaces = 0
    leakage_count = 0
    citation_count = 0
    for tenant_id, cookie, csrf, forbidden_prefix in (
        (tenant_a, cookie_a, csrf_a, "BETA-"),
        (tenant_b, cookie_b, csrf_b, "ALPHA-"),
    ):
        _as_tenant(client, cookie, csrf)
        for strategy in ("dense", "bm25", "hybrid"):
            for rerank in (False, True):
                request = {
                    "question": "Atlas records retention audit",
                    "strategy": strategy,
                    "top_k": 10,
                    "rerank": rerank,
                }
                evidence_response = client.post("/api/v1/retrieve", json=request)
                assert evidence_response.status_code == 200
                evidence = evidence_response.json()["evidence"]
                assert evidence
                retrieved_doc_ids = [item["doc_id"] for item in evidence]
                precision, recall, reciprocal_rank = _document_metrics(
                    retrieved_doc_ids, {uploaded[tenant_id]}
                )
                assert {item["doc_id"] for item in evidence} == {uploaded[tenant_id]}
                assert (precision, recall, reciprocal_rank) == (1.0, 1.0, 1.0)
                metric_cases.append((precision, recall, reciprocal_rank))
                assert all(item["source_file"] == "shared-policy.pdf" for item in evidence)
                assert all(item["page_start"] in {1, 2} for item in evidence)
                assert all(forbidden_prefix not in item["text"] for item in evidence)
                leakage_count += int(any(forbidden_prefix in item["text"] for item in evidence))
                response_surfaces += 1

                answer_response = client.post("/api/v1/query", json=request)
                assert answer_response.status_code == 200
                answer = answer_response.json()
                assert answer["status"] == "answered"
                citations = answer["citations"]
                assert len(citations) == 2
                expected_pages = {claim: page for page, claim in enumerate(claims[tenant_id], 1)}
                assert {item["quote"]: item["page_start"] for item in citations} == expected_pages
                retrieved_chunk_ids = {item["chunk_id"] for item in evidence}
                assert all(item["chunk_id"] in retrieved_chunk_ids for item in citations)
                assert all(item["source_file"] == "shared-policy.pdf" for item in citations)
                assert all(forbidden_prefix not in item["quote"] for item in citations)
                citation_count += len(citations)
                leakage_count += int(
                    any(forbidden_prefix in item["quote"] for item in citations)
                    or forbidden_prefix in answer["answer"]
                )
                response_surfaces += 1
                assert forbidden_prefix not in answer["answer"]
                assert all(claim in answer["answer"] for claim in claims[tenant_id])
                cases += 1

    # BM25 has a deterministic no-match path; it must abstain without a citation.
    _as_tenant(client, cookie_a, csrf_a)
    no_evidence = client.post(
        "/api/v1/query", json={"question": "zirconium submarine", "strategy": "bm25"}
    )
    assert no_evidence.status_code == 200
    assert no_evidence.json()["status"] == "insufficient_evidence"
    assert no_evidence.json()["citations"] == []

    _as_tenant(client, None, None)
    assert (
        client.post(
            "/api/v1/query", json={"question": "Atlas records", "strategy": "bm25"}
        ).status_code
        == 401
    )
    assert cases == 12
    assert len(metric_cases) == 12
    assert sum(item[0] for item in metric_cases) == 12.0
    assert sum(item[1] for item in metric_cases) == 12.0
    assert sum(item[2] for item in metric_cases) == 12.0
    assert response_surfaces == 24
    assert leakage_count == 0
    assert citation_count == 24
