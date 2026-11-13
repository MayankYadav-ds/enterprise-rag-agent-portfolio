"""Authenticated in-process demo using the checked-in fictional PDF.

FastAPI routes, document processing, PDF parsing, chunking, dense retrieval,
answer validation, citations, and deletion use production code. Only embedding,
vector storage, and the provider response are deterministic test adapters.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import tiktoken
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_generator_factory
from rag_agent.api.documents import DocumentManager
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.generation import AnswerGenerator

ORIGIN = "http://localhost:5173"
USERNAME = "phase-10-15-demo-owner"
PASSWORD = "synthetic-demo-password-2026"
SAMPLE_PDF = Path(__file__).resolve().parents[1] / "examples" / "synthetic-portfolio-brief.pdf"
TERMS = (
    "kits",
    "delivered",
    "storage",
    "repair",
    "pilot",
    "returns",
    "future",
    "demand",
    "july",
    "document",
    "fictional",
    "revenue",
    "cost",
    "customer",
    "satisfaction",
    "expansion",
    "profitability",
    "source",
    "workshop",
    "cooperative",
    "cedar",
    "bay",
    "report",
    "sample",
    "forecast",
    "question",
    "evidence",
    "prepared",
    "remained",
    "delivered",
)


class _WordTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class _DemoEmbedder:
    model_name = "synthetic-phase-10-15-test-embedder"
    instruction = ""

    @staticmethod
    def _vector(text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.casefold())
        vector = [float(tokens.count(term)) for term in TERMS]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class _MemoryVectorStore:
    def __init__(self) -> None:
        self.points: list[dict] = []

    def scroll_ids(self, doc_id: str | None = None, *, tenant_id: str | None = None):
        return [
            point["chunk_id"]
            for point in self.points
            if (doc_id is None or point["doc_id"] == doc_id)
            and (tenant_id is None or point["tenant_id"] == tenant_id)
        ]

    def scroll_payloads(self, doc_id: str | None = None, *, tenant_id: str | None = None):
        return {
            point["chunk_id"]: {key: value for key, value in point.items() if key != "vector"}
            for point in self.points
            if (doc_id is None or point["doc_id"] == doc_id)
            and (tenant_id is None or point["tenant_id"] == tenant_id)
        }

    def sync(self, doc_id, chunks, embedder, *, source_file: str, tenant_id: str) -> None:
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, vector in zip(chunks, vectors, strict=True):
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
                    "vector": vector,
                }
            )

    def search(
        self,
        *,
        query_vector: list[float],
        top_k: int,
        doc_id: str | None = None,
        chunk_type: str | None = None,
        tenant_id: str | None = None,
    ):
        matches = []
        for point in self.points:
            if tenant_id is None or point["tenant_id"] != tenant_id:
                continue
            if doc_id is not None and point["doc_id"] != doc_id:
                continue
            if chunk_type is not None and point["chunk_type"] != chunk_type:
                continue
            score = sum(a * b for a, b in zip(query_vector, point["vector"], strict=True))
            if score > 0:
                matches.append(SimpleNamespace(score=score, payload=dict(point)))
        matches.sort(key=lambda hit: hit.score, reverse=True)
        return matches[:top_k]

    def delete_document(self, doc_id: str, *, tenant_id: str) -> None:
        self.points = [
            point
            for point in self.points
            if not (point["doc_id"] == doc_id and point["tenant_id"] == tenant_id)
        ]


class _DeterministicEvidenceProvider:
    def complete(self, messages: list[dict[str, str]], *, schema: dict) -> str:
        request = json.loads(messages[-1]["content"])
        evidence = request["evidence"]
        selected = next(
            (item for item in evidence if "93 kits had been delivered" in item["text"]),
            None,
        )
        if selected is None:
            return json.dumps({"status": "insufficient_evidence", "selections": []})
        return json.dumps(
            {
                "status": "answered",
                "selections": [{"evidence_id": selected["evidence_id"], "quote": selected["text"]}],
            }
        )


@pytest.fixture
def offline_demo(tmp_path, monkeypatch):
    assert SAMPLE_PDF.is_file(), f"Missing synthetic demo PDF: {SAMPLE_PDF.name}"
    settings = APISettings(document_dir=tmp_path / "managed", environment="test")
    app = create_app(settings)
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    with runtime.connection() as connection:
        user_id = create_user(connection, USERNAME, hash_password(PASSWORD), is_admin=True)

    store = _MemoryVectorStore()
    embedder = _DemoEmbedder()
    monkeypatch.setattr(runtime, "_store", lambda _embedder, *, validate: store)
    monkeypatch.setattr(runtime, "embedding", lambda: embedder)
    monkeypatch.setattr(tiktoken, "get_encoding", lambda _name: _WordTokenizer())

    def erase_vectors(manager: DocumentManager, doc_id: str, tenant_id: str) -> None:
        manager.runtime._store(FakeEmbedder(), validate=False).delete_document(
            doc_id, tenant_id=tenant_id
        )

    monkeypatch.setattr(DocumentManager, "_erase_vectors", erase_vectors)
    app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(_DeterministicEvidenceProvider())
    )
    try:
        yield app, runtime, store, user_id
    finally:
        app.dependency_overrides.clear()
        runtime.close()


def test_real_authenticated_api_sample_pdf_query_citation_and_delete(offline_demo) -> None:
    app, _runtime, store, user_id = offline_demo
    sample_bytes = SAMPLE_PDF.read_bytes()
    with TestClient(app) as client:
        login = client.post(
            "/auth/login",
            headers={"Origin": ORIGIN},
            json={"username": USERNAME, "password": PASSWORD},
        )
        assert login.status_code == 200
        assert "session" in login.cookies
        csrf = login.json()["csrf_token"]
        headers = {"Origin": ORIGIN, "X-CSRF-Token": csrf}

        upload = client.post(
            "/api/v1/documents",
            params={"filename": SAMPLE_PDF.name},
            content=sample_bytes,
            headers={**headers, "Content-Type": "application/pdf"},
        )
        assert upload.status_code == 202
        doc_id = upload.json()["doc_id"]

        status = client.get(f"/api/v1/documents/{doc_id}")
        assert status.status_code == 200
        assert status.json()["status"] == "ready"
        assert status.json()["page_count"] == 2
        assert status.json()["chunk_count"] > 0

        inventory = client.get("/api/v1/documents")
        assert inventory.status_code == 200
        assert [record["doc_id"] for record in inventory.json()] == [doc_id]
        assert all(point["tenant_id"] == user_id for point in store.points)
        assert all(point["doc_id"] == doc_id for point in store.points)

        answer = client.post(
            "/api/v1/query",
            headers=headers,
            json={
                "question": "How many kits had been delivered by 31 July 2026?",
                "strategy": "dense",
                "top_k": 20,
                "rerank": False,
            },
        )
        assert answer.status_code == 200
        response = answer.json()
        assert response["status"] == "answered"
        assert response["citations"]
        citation = response["citations"][0]
        assert citation["source_file"] == SAMPLE_PDF.name
        assert citation["page_start"] == citation["page_end"] == 1
        assert "93 kits had been delivered" in citation["quote"]
        assert citation["quote"] in response["answer"]

        deleted = client.delete(f"/api/v1/documents/{doc_id}", headers=headers)
        assert deleted.status_code == 200
        assert deleted.json()["status"] == "deleted"
        assert store.scroll_ids(doc_id, tenant_id=user_id) == []

        after_delete = client.post(
            "/api/v1/query",
            headers=headers,
            json={
                "question": "How many kits had been delivered by 31 July 2026?",
                "strategy": "dense",
                "top_k": 20,
                "rerank": False,
            },
        )
        assert after_delete.status_code == 200
        assert after_delete.json()["status"] == "insufficient_evidence"
        assert after_delete.json()["citations"] == []
