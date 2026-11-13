"""Deterministic offline reliability checks through authenticated API routes.

These tests exercise the real FastAPI authentication, CSRF, tenant derivation,
retrieval/answer orchestration, and response sanitization. External retrieval,
reranking models, and generation providers are deterministic local fakes.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.dependencies import get_generator_factory, get_retriever
from rag_agent.api.runtime import DependencyUnavailable, Runtime
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.auth.models import create_user, ensure_auth_schema
from rag_agent.auth.password import hash_password
from rag_agent.generation import AnswerGenerator, ProviderError
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.reranker import CrossEncoderReranker, RerankerConfig

ORIGIN = "http://localhost:5173"
PASSWORD = "phase-10-9-synthetic-test-password"
PRIVATE_MARKER = "phase-10-9-private-provider-marker"
TENANTS = ("tenant-a", "tenant-b", "tenant-c")


def _hit(tenant_label: str, *, text: str | None = None) -> RetrievedChunk:
    """Deliberately reuse document and chunk identifiers across tenants."""
    return RetrievedChunk(
        chunk_id="shared-chunk-id",
        doc_id="shared-document-id",
        source_file="shared-report.pdf",
        page_start=1,
        page_end=1,
        section_heading="Synthetic",
        subheading=None,
        chunk_type="text",
        text=text or f"Synthetic fact owned by {tenant_label}: value {tenant_label[-1]}.",
        score=1.0,
        rank=1,
        retriever="synthetic",
    )


class _EvidenceEchoProvider:
    """Select only an exact quote present in this request's evidence."""

    def complete(self, messages: list[dict[str, str]], *, schema: dict) -> str:
        request = json.loads(messages[-1]["content"])
        evidence = request["evidence"]
        if not evidence:
            return json.dumps({"status": "insufficient_evidence", "selections": []})
        selected = evidence[0]
        return json.dumps(
            {
                "status": "answered",
                "selections": [{"evidence_id": selected["evidence_id"], "quote": selected["text"]}],
            }
        )


class _FailureProvider:
    def __init__(self, mode: str, failure_question: str) -> None:
        self.mode = mode
        self.failure_question = failure_question

    def complete(self, messages: list[dict[str, str]], *, schema: dict) -> str:
        question = json.loads(messages[-1]["content"])["question"]
        if question != self.failure_question:
            return _EvidenceEchoProvider().complete(messages, schema=schema)
        if self.mode == "provider-error":
            raise ProviderError(PRIVATE_MARKER)
        if self.mode == "unexpected-response-type":
            return None  # type: ignore[return-value]
        if self.mode == "unknown-evidence-id":
            return json.dumps(
                {
                    "status": "answered",
                    "selections": [{"evidence_id": "E999", "quote": "untrusted quote"}],
                }
            )
        if self.mode == "non-verbatim-quote":
            return json.dumps(
                {
                    "status": "answered",
                    "selections": [{"evidence_id": "E1", "quote": "invented quote"}],
                }
            )
        return _EvidenceEchoProvider().complete(messages, schema=schema)


class _CandidateRetriever:
    def __init__(self, hit: RetrievedChunk) -> None:
        self.hit = hit

    def retrieve(self, query: str, **kwargs) -> list[RetrievedChunk]:
        return [self.hit]


class _ExplodingEmbedder:
    def embed_query(self, query: str) -> list[float]:
        raise RuntimeError(PRIVATE_MARKER)


class _ExplodingCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError(PRIVATE_MARKER)


def _make_test_app(tmp_path, tenant_labels: tuple[str, ...] = TENANTS):
    app = create_app(APISettings(document_dir=tmp_path / "managed", environment="test"))
    runtime = app.state.runtime
    ensure_auth_schema(runtime)
    labels_by_user_id: dict[str, str] = {}
    with runtime.connection() as connection:
        for label in tenant_labels:
            user_id = create_user(
                connection,
                f"synthetic-{label}",
                hash_password(PASSWORD),
                is_admin=False,
            )
            labels_by_user_id[user_id] = label
    return app, runtime, labels_by_user_id


def _login_sessions(app: object, labels: tuple[str, ...] = TENANTS) -> dict[str, tuple[str, str]]:
    sessions: dict[str, tuple[str, str]] = {}
    with TestClient(app) as lifecycle_client:  # type: ignore[arg-type]
        for label in labels:
            response = lifecycle_client.post(
                "/auth/login",
                headers={"Origin": ORIGIN},
                json={"username": f"synthetic-{label}", "password": PASSWORD},
            )
            assert response.status_code == 200
            sessions[label] = (response.cookies["session"], response.json()["csrf_token"])
    return sessions


def _authorized_post(client: TestClient, session: tuple[str, str], question: str):
    cookie, csrf = session
    client.cookies.set("session", cookie)
    return client.post(
        "/api/v1/query",
        json={"question": question, "strategy": "hybrid"},
        headers={"X-CSRF-Token": csrf},
    )


def test_three_authenticated_tenants_remain_isolated_during_concurrent_queries(
    tmp_path, monkeypatch
) -> None:
    """Force overlap at retrieval and verify each answer uses only its caller's hit."""
    app, _runtime, labels_by_user_id = _make_test_app(tmp_path)
    sessions = _login_sessions(app)
    clients = {label: TestClient(app) for label in TENANTS}
    barrier = threading.Barrier(len(TENANTS))
    observed: list[str] = []
    observation_lock = threading.Lock()
    failing_label = "tenant-b"
    round_number = 1

    class ConcurrentTenantRetriever:
        def __call__(self, request: RetrievalRequest, *, tenant_id: str):
            label = labels_by_user_id[tenant_id]
            barrier.wait(timeout=10)
            with observation_lock:
                observed.append(label)
            if round_number == 2 and label == failing_label:
                raise DependencyUnavailable(PRIVATE_MARKER)
            # The document/chunk/file identifiers overlap across tenants.
            return [_hit(label)]

    app.dependency_overrides[get_retriever] = lambda: ConcurrentTenantRetriever()
    app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(_EvidenceEchoProvider())
    )

    def request(label: str, question: str):
        return label, _authorized_post(clients[label], sessions[label], question)

    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            first_round = list(
                pool.map(lambda label: request(label, "concurrent synthetic query"), TENANTS)
            )
        assert len(observed) == 3 and set(observed) == set(TENANTS)
        for label, response in first_round:
            assert response.status_code == 200
            body = response.json()
            own_fact = _hit(label).text
            other_facts = [_hit(other).text for other in TENANTS if other != label]
            assert body["status"] == "answered"
            assert body["citations"][0]["quote"] == own_fact
            assert body["evidence"][0]["text"] == own_fact
            assert all(value not in json.dumps(body) for value in other_facts)
            assert body["citations"][0]["chunk_id"] == "shared-chunk-id"

        # One simultaneous backend failure must not contaminate other requests
        # or leave tenant-specific response state for the next operation.
        observed.clear()
        round_number = 2
        barrier.reset()
        with ThreadPoolExecutor(max_workers=3) as pool:
            second_round = list(
                pool.map(lambda label: request(label, "concurrent failure query"), TENANTS)
            )
        assert set(observed) == set(TENANTS)
        for label, response in second_round:
            if label == failing_label:
                assert response.status_code == 503
                assert PRIVATE_MARKER not in response.text
            else:
                assert response.status_code == 200
                assert response.json()["citations"][0]["quote"] == _hit(label).text

        observed.clear()
        round_number = 3
        barrier.reset()
        with ThreadPoolExecutor(max_workers=3) as pool:
            recovery_round = list(pool.map(lambda label: request(label, "recovery query"), TENANTS))
        assert all(response.status_code == 200 for _, response in recovery_round)
        assert {
            label: response.json()["citations"][0]["quote"] for label, response in recovery_round
        } == {label: _hit(label).text for label in TENANTS}

        # Client-supplied tenant scope is rejected before retrieval; the server
        # derives identity from the authenticated user/session record.
        clients["tenant-a"].cookies.set("session", sessions["tenant-a"][0])
        override = clients["tenant-a"].post(
            "/api/v1/query",
            json={
                "question": "spoof",
                "tenant_id": labels_by_user_id[next(iter(labels_by_user_id))],
            },
            headers={"X-CSRF-Token": sessions["tenant-a"][1]},
        )
        assert override.status_code == 422
    finally:
        for client in clients.values():
            client.close()
        app.dependency_overrides.clear()
        app.state.runtime.close()


@pytest.mark.parametrize(
    ("question", "expected_status", "failure_mode"),
    [
        ("retrieval backend failure", 503, "retrieval-error"),
        ("malformed retrieval result", 500, "malformed-hit"),
        ("invalid citation metadata", 500, "invalid-metadata"),
        ("conflicting duplicate chunk", 500, "duplicate-chunk"),
        ("embedding provider failure", 500, "embedding-error"),
        ("reranker failure", 500, "reranker-error"),
        ("provider failure", 503, "provider-error"),
        ("unexpected provider type", 502, "unexpected-response-type"),
        ("unknown evidence selection", 502, "unknown-evidence-id"),
        ("non-verbatim evidence", 502, "non-verbatim-quote"),
    ],
)
def test_query_failures_are_sanitized_and_later_requests_recover(
    tmp_path, question: str, expected_status: int, failure_mode: str
) -> None:
    app, runtime, _labels = _make_test_app(tmp_path, tenant_labels=("alpha",))
    sessions = _login_sessions(app, labels=("alpha",))
    client = TestClient(app)
    good_hit = _hit("tenant-a")

    def retrieve(request: RetrievalRequest, *, tenant_id: str):
        if request.question == question:
            if failure_mode == "retrieval-error":
                raise DependencyUnavailable(PRIVATE_MARKER)
            if failure_mode == "malformed-hit":
                return [object()]
            if failure_mode == "invalid-metadata":
                return [replace(good_hit, page_start=0)]
            if failure_mode == "duplicate-chunk":
                return [good_hit, replace(good_hit, text=PRIVATE_MARKER)]
            if failure_mode == "embedding-error":
                return DenseRetriever(_ExplodingEmbedder(), object()).retrieve(
                    request.question,
                    top_k=request.top_k,
                    tenant_id=tenant_id,
                )
            if failure_mode == "reranker-error":
                reranker = CrossEncoderReranker(
                    _CandidateRetriever(good_hit),
                    RerankerConfig(top_k=1, candidate_count=2, batch_size=1),
                    cross_encoder=_ExplodingCrossEncoder(),
                )
                return reranker.retrieve(
                    request.question,
                    top_k=request.top_k,
                    tenant_id=tenant_id,
                )
        return [good_hit]

    provider_mode = (
        failure_mode
        if failure_mode
        in {
            "provider-error",
            "unexpected-response-type",
            "unknown-evidence-id",
            "non-verbatim-quote",
        }
        else "valid"
    )
    app.dependency_overrides[get_retriever] = lambda: retrieve
    app.dependency_overrides[get_generator_factory] = lambda: (
        lambda: AnswerGenerator(_FailureProvider(provider_mode, question))
    )
    session = sessions["alpha"]
    try:
        failed = _authorized_post(client, session, question)
        assert failed.status_code == expected_status
        assert PRIVATE_MARKER not in failed.text
        assert "traceback" not in failed.text.lower()

        recovered = _authorized_post(client, session, "subsequent valid request")
        assert recovered.status_code == 200
        assert recovered.json()["status"] == "answered"
        assert recovered.json()["citations"][0]["quote"] == good_hit.text
    finally:
        client.close()
        app.dependency_overrides.clear()
        runtime.close()


def test_empty_retrieval_abstains_and_authentication_and_csrf_fail_closed(tmp_path) -> None:
    app, runtime, _labels = _make_test_app(tmp_path, tenant_labels=("alpha",))
    sessions = _login_sessions(app, labels=("alpha",))
    client = TestClient(app)
    calls = 0

    def retrieve(request: RetrievalRequest, *, tenant_id: str):
        nonlocal calls
        calls += 1
        return []

    app.dependency_overrides[get_retriever] = lambda: retrieve
    try:
        unauthenticated = client.post(
            "/api/v1/query",
            json={"question": "synthetic"},
            headers={"X-CSRF-Token": sessions["alpha"][1]},
        )
        assert unauthenticated.status_code == 401
        client.cookies.set("session", sessions["alpha"][0])
        bad_csrf = client.post(
            "/api/v1/query",
            json={"question": "synthetic"},
            headers={"X-CSRF-Token": "invalid"},
        )
        assert bad_csrf.status_code == 403
        assert calls == 0

        app.dependency_overrides[get_generator_factory] = lambda: (
            lambda: AnswerGenerator(_EvidenceEchoProvider())
        )
        empty = _authorized_post(client, sessions["alpha"], "synthetic empty query")
        assert empty.status_code == 200
        assert empty.json()["status"] == "insufficient_evidence"
        assert empty.json()["citations"] == empty.json()["evidence"] == []
        assert calls == 1
    finally:
        client.close()
        app.dependency_overrides.clear()
        runtime.close()


def test_failed_runtime_initialization_is_not_cached_and_can_recover(tmp_path, monkeypatch) -> None:
    """A transient service initialization exception leaves a later retry possible."""
    runtime = Runtime(APISettings(document_dir=tmp_path / "never-created"))
    calls = 0

    class EmptyPayloadStore:
        def scroll_payloads(self, *, tenant_id: str):
            return {}

    def store(_embedder, *, validate: bool):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(PRIVATE_MARKER)
        return EmptyPayloadStore()

    monkeypatch.setattr(runtime, "_store", store)
    request = RetrievalRequest(question="synthetic", strategy="bm25")
    try:
        with pytest.raises(DependencyUnavailable) as failure:
            runtime.retrieve(request, tenant_id="a" * 32)
        assert PRIVATE_MARKER not in str(failure.value)
        assert runtime._retrievers == {}
        assert not runtime.settings.document_dir.exists()

        result = runtime.retrieve(request, tenant_id="a" * 32)
        assert result == []
        assert calls == 2
        assert list(runtime._retrievers) == [("a" * 32, "bm25")]
    finally:
        runtime.close()
