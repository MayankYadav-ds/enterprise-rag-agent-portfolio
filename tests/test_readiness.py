"""Readiness probes validate storage and Qdrant without initializing ML models."""

from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from rag_agent.api.app import create_app
from rag_agent.api.settings import APISettings


@pytest.fixture
def readiness_client(tmp_path):
    app = create_app(APISettings(document_dir=tmp_path / "managed"))
    runtime = app.state.runtime
    client = Mock()
    client.collection_exists.return_value = True
    runtime._client = client
    with TestClient(app) as test_client:
        yield test_client, runtime, client


def test_ready_returns_generic_success_without_loading_models(readiness_client):
    client, runtime, qdrant = readiness_client
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
    qdrant.collection_exists.assert_called_once_with("rag_chunks")
    assert runtime._embedder is None
    assert runtime._generator is None
    assert runtime._retrievers == {}


def test_ready_preserves_liveness_contract(readiness_client):
    client, _, _ = readiness_client
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_unavailable_storage_without_details(readiness_client, monkeypatch):
    client, runtime, qdrant = readiness_client
    original_access = __import__("os").access
    storage = runtime.settings.document_dir

    def access(path, mode):
        if str(path) == str(storage):
            return False
        return original_access(path, mode)

    monkeypatch.setattr("rag_agent.api.runtime.os.access", access)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "managed" not in response.text
    qdrant.collection_exists.assert_not_called()


def test_ready_does_not_create_missing_sqlite_database(readiness_client):
    client, runtime, qdrant = readiness_client
    database = runtime.settings.document_dir / "inventory.sqlite3"
    database.unlink()
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert not database.exists()
    qdrant.collection_exists.assert_not_called()


def test_ready_reports_unavailable_qdrant_without_exception_details(readiness_client):
    client, _, qdrant = readiness_client
    qdrant.collection_exists.side_effect = ConnectionError("private host and API key")
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "private host" not in response.text
    assert "API key" not in response.text


def test_ready_reports_missing_collection_generically(readiness_client):
    client, _, qdrant = readiness_client
    qdrant.collection_exists.return_value = False
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


@pytest.mark.parametrize(
    "base_url, expected_base_url",
    [
        (None, "http://localhost:11434"),
        ("http://ollama.internal:11434", "http://ollama.internal:11434"),
    ],
)
def test_generation_ready_reports_configured_installed_model_without_loading(
    monkeypatch, readiness_client, base_url, expected_base_url
):
    from io import BytesIO
    from unittest.mock import MagicMock

    client, runtime, _ = readiness_client
    monkeypatch.delenv("RAG_LLM_BASE_URL", raising=False)
    if base_url is not None:
        monkeypatch.setenv("RAG_LLM_BASE_URL", base_url)
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5:1.5b")
    response_body = BytesIO(b'{"models":[{"name":"qwen2.5:1.5b"}]}')
    opener = MagicMock()
    opener.open.return_value.__enter__.return_value = response_body
    monkeypatch.setattr("rag_agent.api.runtime.build_opener", lambda *handlers: opener)

    result = client.get("/ready/generation")

    assert result.status_code == 200
    assert result.json() == {"status": "ready"}
    (request,) = opener.open.call_args.args[:1]
    assert request.full_url == f"{expected_base_url}/api/tags"
    assert opener.open.call_args.kwargs["timeout"] <= 3.0
    assert runtime._generator is None
    assert runtime._embedder is None


@pytest.mark.parametrize(
    "model_setting,response_body",
    [
        ("", b'{"models":[]}'),
        ("qwen2.5:1.5b", b'{"models":[{"name":"other:latest"}]}'),
        ("qwen2.5:1.5b", b"not-json"),
    ],
)
def test_generation_ready_fails_closed_without_exposing_details(
    monkeypatch, readiness_client, model_setting, response_body
):
    from io import BytesIO
    from unittest.mock import MagicMock

    client, runtime, _ = readiness_client
    if model_setting:
        monkeypatch.setenv("RAG_LLM_MODEL", model_setting)
    else:
        monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    response = MagicMock()
    response.__enter__.return_value = BytesIO(response_body)
    opener = Mock()
    opener.open.return_value = response
    monkeypatch.setattr("rag_agent.api.runtime.build_opener", lambda *handlers: opener)

    result = client.get("/ready/generation")

    assert result.status_code == 503
    assert result.json() == {"status": "not_ready"}
    assert "other:latest" not in result.text
    assert runtime._generator is None
    assert runtime._embedder is None


def test_generation_ready_reports_unavailable_service_generically(monkeypatch, readiness_client):
    client, runtime, _ = readiness_client
    monkeypatch.setenv("RAG_LLM_MODEL", "qwen2.5:1.5b")

    def unavailable(request, timeout):
        raise ConnectionError("private model endpoint detail")

    monkeypatch.setattr(
        "rag_agent.api.runtime.build_opener",
        lambda *handlers: Mock(open=unavailable),
    )
    response = client.get("/ready/generation")
    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}
    assert "private model endpoint" not in response.text
    assert runtime._generator is None
