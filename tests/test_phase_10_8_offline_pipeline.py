"""Synthetic offline integration through the import-safe production RAG layers.

The API route imports the runtime, which eagerly imports the Qdrant native
client blocked by this Windows environment. These tests therefore exercise
the route's production retrieval -> generation -> response-schema boundary
without importing that unsafe graph or claiming endpoint-level authentication.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest
import tiktoken
from pydantic import ValidationError

from rag_agent.api.schemas import QueryResponse, RetrievalRequest
from rag_agent.chunking.pipeline import chunk_document
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.generation.engine import AnswerGenerator
from rag_agent.generation.models import InvalidResponseError, ProviderError
from rag_agent.ingestion.pipeline import ingest_pdf
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.citation import to_dict
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.reranker import CrossEncoderReranker, FakeCrossEncoder, RerankerConfig


class _WordTokenizer:
    """Deterministic token-count adapter that performs no cache/network lookup."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class _VectorStore:
    """In-memory adapter for the production DenseRetriever store interface."""

    def __init__(self, tenant_id: str, embedder: FakeEmbedder) -> None:
        self.tenant_id = tenant_id
        self.embedder = embedder
        self.points: list[tuple[list[float], dict]] = []
        self.search_calls: list[tuple[str | None, int]] = []

    def add(self, payload: dict) -> None:
        vector = self.embedder.embed_documents([payload["text"]])[0]
        self.points.append((vector, payload))

    def search(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        tenant_id: str | None = None,
    ) -> list[SimpleNamespace]:
        self.search_calls.append((tenant_id, top_k))
        # This adapter is deliberately scoped when constructed and fail-closed
        # when the caller omits or changes that trusted scope.
        if tenant_id != self.tenant_id:
            return []
        scored = []
        for vector, payload in self.points:
            if payload.get("tenant_id") != self.tenant_id:
                continue
            if doc_id is not None and payload.get("doc_id") != doc_id:
                continue
            if chunk_type is not None and payload.get("chunk_type") != chunk_type:
                continue
            dot = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            norm = math.sqrt(sum(value * value for value in query_vector)) * math.sqrt(
                sum(value * value for value in vector)
            )
            scored.append((dot / norm if norm else 0.0, payload))
        scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        return [SimpleNamespace(score=score, payload=payload) for score, payload in scored[:top_k]]


class _EvidenceSelectingLLM:
    """Return one exact quote from the current authorized evidence only."""

    def __init__(self, required_quote: str) -> None:
        self.required_quote = required_quote
        self.calls = 0
        self.evidence_by_call: list[list[dict]] = []

    def complete(self, messages, *, schema: dict) -> str:
        self.calls += 1
        request = json.loads(messages[-1]["content"])
        evidence = request["evidence"]
        self.evidence_by_call.append(evidence)
        for item in evidence:
            if self.required_quote in item["text"]:
                return json.dumps(
                    {
                        "status": "answered",
                        "selections": [
                            {"evidence_id": item["evidence_id"], "quote": self.required_quote}
                        ],
                    }
                )
        return json.dumps({"status": "insufficient_evidence", "selections": []})


class _InvalidEvidenceLLM:
    def complete(self, messages, *, schema: dict) -> str:
        return json.dumps(
            {"status": "answered", "selections": [{"evidence_id": "E999", "quote": "93 kWh"}]}
        )


def _write_synthetic_pdf(path: Path, statement: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((54, 72), "Synthetic account report")
    page.insert_text((54, 104), statement)
    page.insert_text((54, 136), "This generated fixture contains no user or evaluation data.")
    document.save(path)
    document.close()


def _payload(chunk, source_file: str, tenant_id: str | None) -> dict:
    payload = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source_file": source_file,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_heading": chunk.section_heading,
        "subheading": chunk.subheading,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
    }
    if tenant_id is not None:
        payload["tenant_id"] = tenant_id
    return payload


def _build_synthetic_corpus(
    tmp_path: Path, monkeypatch, tenant_ids: tuple[str, str] = ("a" * 32, "b" * 32)
):
    """Use production PDF ingestion and chunking over two same-named inputs."""
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: _WordTokenizer())
    tenant_a, tenant_b = tenant_ids
    tenant_documents = {
        tenant_a: ("41 kWh", "The approved solar battery capacity is 41 kWh."),
        tenant_b: ("93 kWh", "The approved solar battery capacity is 93 kWh."),
    }
    payloads: list[dict] = []
    expected_doc_ids: dict[str, str] = {}
    for tenant_id, (_quote, statement) in tenant_documents.items():
        pdf_path = tmp_path / "synthetic" / tenant_id / "account-summary.pdf"
        _write_synthetic_pdf(pdf_path, statement)
        document, pages = ingest_pdf(pdf_path, tmp_path / "processed" / tenant_id)
        chunks = chunk_document(
            document, pages, strategy="fixed_size", chunk_size=120, chunk_overlap=0
        )
        assert pages and chunks
        assert all(chunk.doc_id == document.doc_id for chunk in chunks)
        assert all(chunk.page_start >= 1 and chunk.page_end >= chunk.page_start for chunk in chunks)
        expected_doc_ids[tenant_id] = document.doc_id
        payloads.extend(_payload(chunk, pdf_path.name, tenant_id) for chunk in chunks)

    # A legacy/unscoped record must remain invisible to either tenant.
    legacy_text = "Synthetic legacy vector says the approved capacity is 777 kWh."
    legacy_chunk = chunks[0].__class__.create(
        doc_id=chunks[0].doc_id,
        text=legacy_text,
        chunk_type="text",
        page_start=1,
        page_end=1,
        chunk_index=999,
        tokenizer=_WordTokenizer(),
    )
    payloads.append(_payload(legacy_chunk, "account-summary.pdf", None))
    return tenant_documents, expected_doc_ids, payloads


@pytest.mark.parametrize("tenant_id", ["a" * 32, "b" * 32])
def test_synthetic_pdf_to_answer_is_tenant_scoped_and_schema_valid(
    tmp_path: Path, monkeypatch, tenant_id: str
) -> None:
    """Exercise parsing, chunking, embedding, retrieval, reranking and citation output."""
    tenant_documents, expected_doc_ids, payloads = _build_synthetic_corpus(tmp_path, monkeypatch)
    quote = tenant_documents[tenant_id][0]
    embedder = FakeEmbedder()
    store = _VectorStore(tenant_id, embedder)
    for payload in payloads:
        store.add(payload)

    bm25_index = {item["chunk_id"]: item for item in payloads}
    hybrid = HybridRetriever(
        DenseRetriever(embedder, store),
        BM25Retriever(bm25_index, tenant_id=tenant_id),
    )
    reranker = CrossEncoderReranker(
        hybrid,
        config=RerankerConfig(top_k=5, candidate_count=20),
        cross_encoder=FakeCrossEncoder(),
    )
    hits = reranker.retrieve(
        "What is the approved solar battery capacity?", top_k=5, tenant_id=tenant_id
    )
    assert hits
    assert all(hit.doc_id == expected_doc_ids[tenant_id] for hit in hits)
    assert all(hit.source_file == "account-summary.pdf" for hit in hits)
    assert all(hit.retriever == "reranked" for hit in hits)
    assert all(quote in hit.text for hit in hits)
    assert store.search_calls and all(call[0] == tenant_id for call in store.search_calls)

    provider = _EvidenceSelectingLLM(quote)
    answer = AnswerGenerator(provider).generate(
        "What is the approved solar battery capacity?", hits
    )
    assert answer.status == "answered"
    assert answer.excerpts
    assert all(excerpt.chunk_id in {hit.chunk_id for hit in hits} for excerpt in answer.excerpts)
    assert all(excerpt.source_file == "account-summary.pdf" for excerpt in answer.excerpts)
    assert all(excerpt.quote == quote for excerpt in answer.excerpts)
    assert all("p. 1" in excerpt.citation for excerpt in answer.excerpts)
    assert provider.calls == 1
    assert all(
        item["doc_id"] == expected_doc_ids[tenant_id] for item in provider.evidence_by_call[0]
    )

    # Match the actual route's public response schema without importing the
    # route module, whose dependency graph eagerly imports blocked Qdrant.
    response = QueryResponse.model_validate(
        {
            "question": answer.question,
            "strategy": "hybrid",
            "reranked": True,
            "evidence": [to_dict(hit) for hit in hits],
            "status": answer.status,
            "answer": answer.text,
            "citations": [asdict(excerpt) for excerpt in answer.excerpts],
            "reason": answer.reason,
        }
    )
    assert response.status == "answered"
    assert response.citations[0].source_file == "account-summary.pdf"
    assert response.citations[0].quote == quote


def test_client_tenant_override_and_unscoped_vectors_fail_closed(
    tmp_path: Path, monkeypatch
) -> None:
    """Request schema rejects tenant overrides and retrievers exclude other/legacy data."""
    tenant_documents, expected_doc_ids, payloads = _build_synthetic_corpus(tmp_path, monkeypatch)
    with pytest.raises(ValidationError):
        RetrievalRequest.model_validate({"question": "capacity?", "tenant_id": "b" * 32})

    embedder = FakeEmbedder()
    tenant_a = "a" * 32
    store = _VectorStore(tenant_a, embedder)
    for payload in payloads:
        store.add(payload)
    retriever = HybridRetriever(
        DenseRetriever(embedder, store),
        BM25Retriever({item["chunk_id"]: item for item in payloads}, tenant_id=tenant_a),
    )
    authorized_hits = retriever.retrieve("approved capacity", tenant_id=tenant_a, top_k=20)
    assert authorized_hits
    assert {hit.doc_id for hit in authorized_hits} == {expected_doc_ids[tenant_a]}
    assert all(tenant_documents[tenant_a][0] in hit.text for hit in authorized_hits)

    # An untrusted caller cannot switch the already-bound adapter/retriever to B.
    assert retriever.retrieve("approved capacity", top_k=20) == []
    assert retriever.retrieve("approved capacity", tenant_id="b" * 32, top_k=20) == []
    assert store.search_calls[-1][0] == "b" * 32


def test_fabricated_citation_and_empty_evidence_are_rejected_safely(
    tmp_path: Path, monkeypatch
) -> None:
    tenant_documents, _doc_ids, payloads = _build_synthetic_corpus(tmp_path, monkeypatch)
    tenant_id = "a" * 32
    payload = next(item for item in payloads if item.get("tenant_id") == tenant_id)
    from rag_agent.retrieval.models import RetrievedChunk

    hit = RetrievedChunk(
        chunk_id=payload["chunk_id"],
        doc_id=payload["doc_id"],
        source_file=payload["source_file"],
        page_start=payload["page_start"],
        page_end=payload["page_end"],
        section_heading=payload["section_heading"],
        subheading=payload["subheading"],
        chunk_type=payload["chunk_type"],
        text=payload["text"],
        score=1.0,
        rank=1,
        retriever="hybrid",
    )
    with pytest.raises(InvalidResponseError, match="unknown evidence"):
        AnswerGenerator(_InvalidEvidenceLLM()).generate("capacity?", [hit])

    class MustNotRun:
        def complete(self, messages, *, schema: dict) -> str:
            raise AssertionError("provider must not run without retrieved evidence")

    abstention = AnswerGenerator(MustNotRun()).generate("capacity?", [])
    assert abstention.status == "insufficient_evidence"
    assert abstention.reason == "no_evidence"
    assert tenant_documents[tenant_id][0] in hit.text


def test_generation_provider_failure_is_not_represented_as_an_answer() -> None:
    from rag_agent.retrieval.models import RetrievedChunk

    hit = RetrievedChunk(
        chunk_id="chunk_test",
        doc_id="doc_test",
        source_file="synthetic.pdf",
        page_start=1,
        page_end=1,
        section_heading=None,
        subheading=None,
        chunk_type="text",
        text="Synthetic evidence says capacity is 41 kWh.",
        score=1.0,
        rank=1,
        retriever="bm25",
    )

    class FailedProvider:
        def complete(self, messages, *, schema: dict) -> str:
            raise ProviderError("synthetic provider failure")

    with pytest.raises(ProviderError, match="synthetic provider failure"):
        AnswerGenerator(FailedProvider()).generate("capacity?", [hit])


def test_empty_synthetic_pdf_produces_no_chunks_and_abstains(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: _WordTokenizer())
    pdf_path = tmp_path / "empty-synthetic.pdf"
    document_pdf = pymupdf.open()
    document_pdf.new_page()
    document_pdf.save(pdf_path)
    document_pdf.close()

    document, pages = ingest_pdf(pdf_path, tmp_path / "empty-processed")
    assert document.num_pages == 1
    assert pages == []
    assert chunk_document(document, pages, strategy="fixed_size") == []

    class MustNotRun:
        def complete(self, messages, *, schema: dict) -> str:
            raise AssertionError("provider must not run for an empty source")

    answer = AnswerGenerator(MustNotRun()).generate("what is stated?", [])
    assert answer.status == "insufficient_evidence"
    assert answer.reason == "no_evidence"


def test_embedding_store_and_citation_metadata_failures_are_not_silenced() -> None:
    from rag_agent.retrieval.models import RetrievedChunk

    class FailedEmbedder:
        dimension = 1
        model_name = "synthetic-failure"

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("synthetic embedding failure")

        def embed_query(self, text: str) -> list[float]:
            raise RuntimeError("synthetic embedding failure")

    class FailedStore:
        def search(self, **kwargs):
            raise RuntimeError("synthetic vector store failure")

    with pytest.raises(RuntimeError, match="synthetic embedding failure"):
        DenseRetriever(FailedEmbedder(), FailedStore()).retrieve("synthetic query")

    with pytest.raises(RuntimeError, match="synthetic vector store failure"):
        DenseRetriever(FakeEmbedder(), FailedStore()).retrieve("synthetic query")

    invalid_hit = RetrievedChunk(
        chunk_id="chunk_test",
        doc_id="doc_test",
        source_file="synthetic.pdf",
        page_start=1,
        page_end=1,
        section_heading=None,
        subheading=None,
        chunk_type="text",
        text="Synthetic evidence.",
        score=1.0,
        rank=1,
        retriever="dense",
    )

    class MustNotRun:
        def complete(self, messages, *, schema: dict) -> str:
            raise AssertionError("provider must not run for invalid citation metadata")

    with pytest.raises(ValueError, match="bare filename"):
        AnswerGenerator(MustNotRun()).generate(
            "synthetic query", [replace(invalid_hit, source_file="../other-tenant.pdf")]
        )


def test_authenticated_fastapi_query_preserves_auth_tenant_and_citation_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    """Exercise real auth/session/CSRF and query routes with injected test services."""
    import json as json_module

    from fastapi.testclient import TestClient

    from rag_agent.api.app import create_app
    from rag_agent.api.dependencies import get_generator_factory, get_retriever
    from rag_agent.api.settings import APISettings
    from rag_agent.auth.models import create_user, ensure_auth_schema
    from rag_agent.auth.password import hash_password
    from rag_agent.generation import AnswerGenerator, ProviderError

    password = "synthetic-test-password-42"
    app = create_app(APISettings(document_dir=tmp_path / "api-storage", environment="test"))
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as connection:
        user_a = create_user(connection, "synthetic-alpha", hash_password(password))
        user_b = create_user(connection, "synthetic-beta", hash_password(password))
    tenants, doc_ids, payloads = _build_synthetic_corpus(
        tmp_path, monkeypatch, tenant_ids=(user_a, user_b)
    )

    embedder = FakeEmbedder()
    retrievers = {}
    for tenant_id in (user_a, user_b):
        store = _VectorStore(tenant_id, embedder)
        for payload in payloads:
            store.add(payload)
        hybrid = HybridRetriever(
            DenseRetriever(embedder, store),
            BM25Retriever({item["chunk_id"]: item for item in payloads}, tenant_id=tenant_id),
        )
        retrievers[tenant_id] = CrossEncoderReranker(
            hybrid,
            config=RerankerConfig(top_k=5, candidate_count=20),
            cross_encoder=FakeCrossEncoder(),
        )

    class EndpointLLM:
        def __init__(self) -> None:
            self.calls = 0
            self.forbidden_quote: str | None = None
            self.fail = False
            self.quotes_by_doc_id = {
                doc_ids[tenant_id]: tenants[tenant_id][0] for tenant_id in (user_a, user_b)
            }

        def complete(self, messages, *, schema: dict) -> str:
            self.calls += 1
            if self.fail:
                raise ProviderError("synthetic endpoint generation failure")
            evidence = json_module.loads(messages[-1]["content"])["evidence"]
            if self.forbidden_quote is not None:
                return json_module.dumps(
                    {
                        "status": "answered",
                        "selections": [{"evidence_id": "E1", "quote": self.forbidden_quote}],
                    }
                )
            if not evidence:
                return json_module.dumps({"status": "insufficient_evidence", "selections": []})
            return json_module.dumps(
                {
                    "status": "answered",
                    "selections": [
                        {
                            "evidence_id": evidence[0]["evidence_id"],
                            "quote": self.quotes_by_doc_id[evidence[0]["doc_id"]],
                        }
                    ],
                }
            )

    provider = EndpointLLM()
    generator = AnswerGenerator(provider)
    retrieval_scopes: list[str] = []

    def retrieve(body, *, tenant_id: str):
        retrieval_scopes.append(tenant_id)
        if body.question == "synthetic no-match":
            return []
        service = retrievers.get(tenant_id)
        if service is None:
            return []
        return service.retrieve(
            body.question,
            top_k=body.top_k,
            doc_id=body.doc_id,
            chunk_type=body.chunk_type,
            min_score=body.min_score,
            tenant_id=tenant_id,
        )

    app.dependency_overrides[get_retriever] = lambda: retrieve
    app.dependency_overrides[get_generator_factory] = lambda: lambda: generator

    def login(client, username: str) -> None:
        response = client.post(
            "/auth/login",
            headers={"Origin": "http://localhost:5173"},
            json={"username": username, "password": password},
        )
        assert response.status_code == 200
        client.headers["X-CSRF-Token"] = response.json()["csrf_token"]

    with TestClient(app) as client_a, TestClient(app) as client_b, TestClient(app) as guest:
        login(client_a, "synthetic-alpha")
        login(client_b, "synthetic-beta")
        assert client_a.cookies.get("session") != client_b.cookies.get("session")
        assert client_a.get("/auth/me").json()["username"] == "synthetic-alpha"
        assert client_b.get("/auth/me").json()["username"] == "synthetic-beta"

        valid_a = client_a.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        valid_b = client_b.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        assert valid_a.status_code == valid_b.status_code == 200
        result_a, result_b = valid_a.json(), valid_b.json()
        assert {item["doc_id"] for item in result_a["evidence"]} == {doc_ids[user_a]}
        assert {item["doc_id"] for item in result_b["evidence"]} == {doc_ids[user_b]}
        assert result_a["citations"][0]["quote"] == tenants[user_a][0]
        assert result_b["citations"][0]["quote"] == tenants[user_b][0]
        assert user_a in retrieval_scopes and user_b in retrieval_scopes

        spoofed_headers = {"X-Tenant-ID": user_b}
        header_and_query_spoof = client_a.post(
            f"/api/v1/query?tenant_id={user_b}",
            headers=spoofed_headers,
            json={"question": "approved solar battery capacity"},
        )
        assert header_and_query_spoof.status_code == 200
        assert {item["doc_id"] for item in header_and_query_spoof.json()["evidence"]} == {
            doc_ids[user_a]
        }
        assert retrieval_scopes[-1] == user_a

        document_id_spoof = client_a.post(
            "/api/v1/query",
            json={
                "question": "approved solar battery capacity",
                "doc_id": doc_ids[user_b],
            },
        )
        assert document_id_spoof.status_code == 200
        assert document_id_spoof.json()["status"] == "insufficient_evidence"
        assert document_id_spoof.json()["evidence"] == []

        before_unauthenticated = len(retrieval_scopes)
        missing_credentials = guest.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        invalid_credentials = guest.post(
            "/auth/login",
            headers={"Origin": "http://localhost:5173"},
            json={"username": "synthetic-alpha", "password": "wrong-password"},
        )
        assert missing_credentials.status_code == 401
        assert invalid_credentials.status_code == 401
        assert len(retrieval_scopes) == before_unauthenticated

        client_a_csrf = client_a.headers["X-CSRF-Token"]
        before_csrf_failures = len(retrieval_scopes)
        client_a.headers.pop("X-CSRF-Token")
        missing_csrf = client_a.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        client_a.headers["X-CSRF-Token"] = "incorrect-synthetic-csrf-token"
        invalid_csrf = client_a.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        client_a.headers["X-CSRF-Token"] = client_a_csrf
        assert missing_csrf.status_code == invalid_csrf.status_code == 403
        assert len(retrieval_scopes) == before_csrf_failures

        before_spoof = len(retrieval_scopes)
        spoof = client_a.post(
            "/api/v1/query",
            json={"question": "approved solar battery capacity", "tenant_id": user_b},
        )
        assert spoof.status_code == 422
        assert len(retrieval_scopes) == before_spoof

        empty = client_a.post("/api/v1/query", json={"question": "synthetic no-match"})
        assert empty.status_code == 200
        assert empty.json()["status"] == "insufficient_evidence"
        assert empty.json()["citations"] == empty.json()["evidence"] == []

        provider.forbidden_quote = tenants[user_b][0]
        cross_tenant_quote = client_a.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        assert cross_tenant_quote.status_code == 502
        assert tenants[user_b][0] not in cross_tenant_quote.text

        provider.forbidden_quote = None
        provider.fail = True
        failed_generation = client_a.post(
            "/api/v1/query", json={"question": "approved solar battery capacity"}
        )
        assert failed_generation.status_code == 503
        assert "synthetic endpoint generation failure" not in failed_generation.text
