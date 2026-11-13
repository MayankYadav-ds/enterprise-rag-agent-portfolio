"""Real runtime composition using an isolated in-memory Qdrant and fake models."""

from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from rag_agent.api.runtime import DependencyUnavailable, Runtime
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore
from rag_agent.retrieval.reranker import FakeCrossEncoder

TENANT = "a" * 32


def test_read_only_missing_collection_and_recreate_do_not_write():
    client = Mock()
    client.collection_exists.return_value = False
    with pytest.raises(ValueError, match="does not exist"):
        QdrantStore(FakeEmbedder(), client=client, create_if_missing=False)
    with pytest.raises(ValueError, match="recreate"):
        QdrantStore(FakeEmbedder(), client=client, create_if_missing=False, recreate=True)
    client.create_collection.assert_not_called()
    client.recreate_collection.assert_not_called()
    client.create_payload_index.assert_not_called()


def test_bm25_runtime_uses_payloads_without_loading_models(monkeypatch):
    client = Mock()
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock(return_value=client))
    store = Mock()
    store.scroll_payloads.return_value = {}
    store_class = Mock(return_value=store)
    embedder_class = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr("rag_agent.api.runtime.QdrantStore", store_class)
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", embedder_class)
    runtime = Runtime(APISettings())
    assert runtime.retrieve(RetrievalRequest(question="x", strategy="bm25"), tenant_id=TENANT) == []
    assert runtime.retrieve(RetrievalRequest(question="x", strategy="bm25"), tenant_id=TENANT) == []
    store.scroll_payloads.assert_called_once()
    assert store_class.call_args.kwargs["create_if_missing"] is False
    embedder_class.assert_not_called()
    runtime.close()
    client.close.assert_called_once()


def test_runtime_retrieval_requires_a_valid_explicit_tenant():
    runtime = Runtime(APISettings())
    with pytest.raises(TypeError):
        runtime.retrieve(RetrievalRequest(question="x", strategy="bm25"))
    with pytest.raises(ValueError, match="Invalid tenant identity"):
        runtime.retrieve(RetrievalRequest(question="x", strategy="bm25"), tenant_id="../bad")
    runtime.close()


def test_unavailable_service_retries_without_loading_dense_model(monkeypatch):
    store = Mock(side_effect=ConnectionError("private host"))
    model = Mock()
    monkeypatch.setattr("rag_agent.api.runtime.QdrantStore", store)
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock())
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", model)
    runtime = Runtime(APISettings())
    for _ in range(2):
        with pytest.raises(DependencyUnavailable):
            runtime.retrieve(RetrievalRequest(question="x", strategy="dense"), tenant_id=TENANT)
    assert store.call_count == 2
    model.assert_not_called()
    runtime.close()


def test_generator_is_lazy_cached_and_configuration_failure_is_retriable(monkeypatch):
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)
    runtime = Runtime(APISettings())
    with pytest.raises(DependencyUnavailable):
        runtime.generator()
    monkeypatch.setenv("RAG_LLM_MODEL", "installed-model")
    provider = Mock()
    builder = Mock(return_value=provider)
    monkeypatch.setattr("rag_agent.api.runtime.build_provider", builder)
    assert runtime.generator() is runtime.generator()
    builder.assert_called_once()
    provider.complete.assert_not_called()
    runtime.close()


@pytest.mark.parametrize(
    "values",
    [
        {"qdrant_url": "file:///private"},
        {"qdrant_url": "http://user:secret@localhost"},
        {"qdrant_url": "http://localhost/path"},
        {"rerank_candidates": 101},
        {"qdrant_timeout": 0},
        {"qdrant_timeout": float("inf")},
        {"qdrant_url": "http://qdrant.internal:6333"},
        {"qdrant_url": "https://qdrant.example.com:6333"},
        {"qdrant_url": "https://qdrant.example.com:6333", "qdrant_api_key": ""},
        {"qdrant_url": "https://qdrant.example.com:6333", "qdrant_api_key": "   "},
        {"login_rate_limit_attempts": 0},
        {"login_rate_limit_window_seconds": 30},
        {"upload_rate_limit_attempts": 0},
        {"upload_rate_limit_window_seconds": 30},
    ],
)
def test_invalid_settings(values):
    with pytest.raises(ValidationError):
        APISettings(**values)


def test_environment_settings_hide_credentials(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "test")
    monkeypatch.delenv("RAG_TRUSTED_ORIGINS", raising=False)
    monkeypatch.setenv("QDRANT_API_KEY", "private-key")
    monkeypatch.setenv("RAG_API_COLLECTION", "test_chunks")
    monkeypatch.setenv("RERANKER_CANDIDATE_COUNT", "30")
    settings = APISettings.from_env()
    assert settings.collection == "test_chunks" and settings.rerank_candidates == 30
    assert "private-key" not in repr(settings) + settings.model_dump_json()


def test_environment_configuration_requires_explicit_environment(monkeypatch):
    monkeypatch.delenv("RAG_ENV", raising=False)
    with pytest.raises(ValueError, match="RAG_ENV must be explicitly set"):
        APISettings.from_env()


def test_remote_qdrant_requires_tls_and_authentication():
    settings = APISettings(
        qdrant_url="https://qdrant.example.com:6333", qdrant_api_key="private-key"
    )
    assert settings.qdrant_url == "https://qdrant.example.com:6333"
    assert "private-key" not in repr(settings) + settings.model_dump_json()


def test_local_qdrant_http_remains_available():
    assert APISettings(qdrant_url="http://localhost:6333").qdrant_url == "http://localhost:6333"


def test_qdrant_https_preserves_verified_http_client_default(monkeypatch):
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", constructor)
    settings = APISettings(
        qdrant_url="https://qdrant.example.com:6333", qdrant_api_key="private-key"
    )
    assert settings.qdrant_ssl_context() is None
    runtime = Runtime(settings)
    runtime.qdrant_client()
    # Omitting HTTPX's verify option preserves its verified default (True).
    assert "verify" not in constructor.call_args.kwargs
    runtime.close()


def test_qdrant_client_receives_ca_context_api_key_and_timeout(monkeypatch):
    import ssl

    import certifi

    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", constructor)
    settings = APISettings(
        qdrant_url="https://qdrant.example.com:6333",
        qdrant_api_key="private-key",
        qdrant_ca_cert_file=certifi.where(),
        qdrant_timeout=17,
    )
    runtime = Runtime(settings)
    assert runtime.qdrant_client() is client
    options = constructor.call_args.kwargs
    assert options["url"] == "https://qdrant.example.com:6333"
    assert options["api_key"] == "private-key"
    assert options["timeout"] == 17
    assert isinstance(options["verify"], ssl.SSLContext)
    assert options["verify"].verify_mode == ssl.CERT_REQUIRED
    assert options["verify"].check_hostname
    runtime.close()


def test_loopback_http_does_not_enable_or_downgrade_tls(monkeypatch):
    client = Mock()
    constructor = Mock(return_value=client)
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", constructor)
    runtime = Runtime(APISettings(qdrant_url="http://localhost:6333"))
    runtime.qdrant_client()
    options = constructor.call_args.kwargs
    assert options["url"] == "http://localhost:6333"
    assert "verify" not in options
    runtime.close()


@pytest.mark.parametrize("path_value", ["missing", "invalid"])
def test_invalid_qdrant_ca_fails_closed(path_value, tmp_path):
    from pathlib import Path

    path = tmp_path / "ca.pem"
    if path_value == "invalid":
        path.write_text("not a CA certificate", encoding="utf-8")
    else:
        path = tmp_path / "missing-ca.pem"
    with pytest.raises(ValidationError, match="QDRANT_CA_CERT_FILE"):
        APISettings(
            qdrant_url="https://qdrant.example.com:6333",
            qdrant_api_key="private-key",
            qdrant_ca_cert_file=Path(path),
        )


def test_custom_ca_is_rejected_for_loopback_http(tmp_path):
    import certifi

    with pytest.raises(ValidationError, match="requires an HTTPS"):
        APISettings(qdrant_url="http://localhost:6333", qdrant_ca_cert_file=certifi.where())


def _readiness_runtime(tmp_path, collection_exists=True):
    runtime = Runtime(APISettings(document_dir=tmp_path / "managed"))
    with runtime.connection():
        pass
    client = Mock()
    client.collection_exists.return_value = collection_exists
    runtime._client = client
    return runtime, client


def test_runtime_readiness_checks_existing_storage_sqlite_and_collection(tmp_path):
    runtime, client = _readiness_runtime(tmp_path)
    runtime.check_readiness()
    client.collection_exists.assert_called_once_with("rag_chunks")
    assert runtime._embedder is None and runtime._generator is None
    runtime.close()


def test_runtime_readiness_fails_when_storage_is_not_accessible(tmp_path, monkeypatch):
    runtime, client = _readiness_runtime(tmp_path)
    storage = runtime.settings.document_dir
    original_access = __import__("os").access

    def access(path, mode):
        if str(path) == str(storage):
            return False
        return original_access(path, mode)

    monkeypatch.setattr("rag_agent.api.runtime.os.access", access)
    with pytest.raises(DependencyUnavailable, match="storage is unavailable"):
        runtime.check_readiness()
    client.collection_exists.assert_not_called()
    runtime.close()


def test_runtime_readiness_does_not_create_a_missing_registry(tmp_path):
    runtime, client = _readiness_runtime(tmp_path)
    registry = runtime.settings.document_dir / "inventory.sqlite3"
    registry.unlink()
    with pytest.raises(DependencyUnavailable, match="registry is unavailable"):
        runtime.check_readiness()
    assert not registry.exists()
    client.collection_exists.assert_not_called()
    runtime.close()


@pytest.mark.parametrize("collection_exists", [False, "unreachable"])
def test_runtime_readiness_fails_for_missing_or_unavailable_qdrant(tmp_path, collection_exists):
    runtime, client = _readiness_runtime(tmp_path, collection_exists=True)
    if collection_exists == "unreachable":
        client.collection_exists.side_effect = ConnectionError("private endpoint")
    else:
        client.collection_exists.return_value = collection_exists
    with pytest.raises(DependencyUnavailable):
        runtime.check_readiness()
    runtime.close()


def test_storage_permissions_are_private_on_posix(tmp_path):
    import os
    import stat

    from rag_agent.api.storage import private_directory, private_file

    if os.name != "posix":
        pytest.skip("POSIX permission modes are not enforced on Windows")
    directory = private_directory(tmp_path / "private")
    artifact = directory / "registry.sqlite3"
    artifact.write_bytes(b"temporary test registry")
    private_file(artifact)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_production_requires_explicit_https_trusted_origins(monkeypatch):
    monkeypatch.setenv("RAG_ENV", "production")
    monkeypatch.delenv("RAG_TRUSTED_ORIGINS", raising=False)
    with pytest.raises(ValueError, match="production requires RAG_TRUSTED_ORIGINS"):
        APISettings.from_env()

    monkeypatch.setenv("RAG_TRUSTED_ORIGINS", "http://localhost:5173")
    with pytest.raises(ValidationError, match="production requires one or more HTTPS"):
        APISettings.from_env()

    monkeypatch.setenv("RAG_TRUSTED_ORIGINS", "https://app.example.com")
    settings = APISettings.from_env()
    assert settings.environment == "production"
    assert settings.trusted_origins == ("https://app.example.com",)


def test_production_settings_reject_default_development_origins():
    with pytest.raises(ValidationError, match="production requires one or more HTTPS"):
        APISettings(environment="production")


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
@pytest.mark.parametrize("rerank", [False, True])
def test_real_retrievers_and_optional_reranker_compose(monkeypatch, strategy, rerank):
    # Isolated test data only; never accesses the user's running Qdrant.
    from qdrant_client.http import models

    from rag_agent.embeddings.store import point_id

    client = QdrantClient(":memory:")
    embedder = FakeEmbedder()
    QdrantStore(embedder, client=client)
    text = "Revenue increased in 2024."
    client.upsert(
        "rag_chunks",
        points=[
            models.PointStruct(
                id=str(point_id("c1")),
                vector=embedder.embed_query(text),
                payload={
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "source_file": "report.pdf",
                    "page_start": 1,
                    "page_end": 1,
                    "section_heading": None,
                    "subheading": None,
                    "chunk_type": "text",
                    "text": text,
                    "embedding_model": embedder.model_name,
                    "tenant_id": TENANT,
                },
            )
        ],
    )
    monkeypatch.setattr("rag_agent.api.runtime.QdrantClient", Mock(return_value=client))
    model = Mock(return_value=embedder)
    monkeypatch.setattr("rag_agent.api.runtime.SentenceTransformerEmbedder", model)
    encoder = Mock(return_value=FakeCrossEncoder())
    monkeypatch.setattr("rag_agent.retrieval.reranker.CrossEncoderReranker._load_model", encoder)
    runtime = Runtime(APISettings())
    request = RetrievalRequest(question="Revenue", strategy=strategy, rerank=rerank, top_k=1)
    first = runtime.retrieve(request, tenant_id=TENANT)
    assert first[0].chunk_id == "c1"
    assert first[0].source_file == "report.pdf"
    assert runtime.retrieve(request, tenant_id=TENANT)[0].chunk_id == "c1"
    assert model.call_count == (0 if strategy == "bm25" else 1)
    if not rerank:
        encoder.assert_not_called()
    runtime.close()
