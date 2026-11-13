"""Lazy per-application services and coordinated managed document mutations."""

import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from threading import RLock
from typing import TYPE_CHECKING
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from rag_agent.api.documents import DocumentManager
from rag_agent.api.schemas import RetrievalRequest
from rag_agent.api.settings import APISettings
from rag_agent.api.storage import private_directory, private_file
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.protocol import Embedder
from rag_agent.embeddings.sentence_transformer import SentenceTransformerEmbedder
from rag_agent.generation import AnswerGenerator, GenerationConfig, build_provider
from rag_agent.retrieval.bm25 import BM25Retriever
from rag_agent.retrieval.dense import DenseRetriever
from rag_agent.retrieval.hybrid import HybridRetriever
from rag_agent.retrieval.models import RetrievedChunk
from rag_agent.retrieval.protocol import Retriever
from rag_agent.retrieval.reranker import CrossEncoderReranker, RerankerConfig
from rag_agent.tenancy import validate_tenant_id

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from rag_agent.embeddings.store import QdrantStore


def _new_qdrant_client(**options):
    """Import the production Qdrant backend only when a Qdrant connection is requested."""
    from qdrant_client import QdrantClient

    return QdrantClient(**options)


class DependencyUnavailable(RuntimeError):
    """A required service, model, or configuration is unavailable."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Runtime:
    """Cache models and BM25 snapshots; serialize retrieval to protect model use.

    Clients belong to this runtime and are closed by the application lifespan.
    Failed initialization is not cached, so a subsequent request can retry.
    """

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.lock = RLock()
        self._client: QdrantClient | None = None
        self._retrievers: dict[tuple[str, str], Retriever] = {}
        self._generator: AnswerGenerator | None = None
        self._embedder: Embedder | None = None
        self._documents: DocumentManager | None = None

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Open a short-lived connection to the app's shared local registry DB."""
        path = self.settings.document_dir / "inventory.sqlite3"
        private_directory(path.parent)
        connection = sqlite3.connect(path, timeout=10)
        private_file(path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    def documents(self) -> DocumentManager:
        if self._documents is not None:
            return self._documents
        with self.lock:
            if self._documents is None:
                self._documents = DocumentManager(self)
            return self._documents

    def embedding(self) -> Embedder:
        if self._embedder is None:
            self._embedder = SentenceTransformerEmbedder()
        return self._embedder

    def invalidate_retrieval(self, tenant_id: str | None = None) -> None:
        if tenant_id is None:
            self._retrievers.clear()
        else:
            tenant_id = validate_tenant_id(tenant_id)
            self._retrievers = {
                key: retriever for key, retriever in self._retrievers.items() if key[0] != tenant_id
            }

    def _store(self, embedder: Embedder, *, validate: bool) -> "QdrantStore":
        from rag_agent.embeddings.store import QdrantStore

        client = self.qdrant_client()
        return QdrantStore(
            embedder,
            collection_name=self.settings.collection,
            client=client,
            validate_dimension=validate,
            create_if_missing=False,
        )

    def qdrant_client(self) -> "QdrantClient":
        """Return the shared client configured with authenticated, verified REST TLS."""
        if self._client is None:
            key = self.settings.qdrant_api_key
            options = {
                "url": self.settings.qdrant_url,
                "api_key": key.get_secret_value() if key else None,
                "timeout": self.settings.qdrant_timeout,
            }
            context = self.settings.qdrant_ssl_context()
            if context is not None:
                # qdrant-client's documented REST kwargs passthrough forwards this
                # verifying SSLContext to HTTPX's ``verify`` parameter.
                options["verify"] = context
            self._client = _new_qdrant_client(**options)
        return self._client

    def check_readiness(self) -> None:
        """Check local storage and Qdrant without creating files or loading models."""
        root = self.settings.document_dir
        database = root / "inventory.sqlite3"
        required_dir_access = os.R_OK | os.W_OK | os.X_OK
        if not root.is_dir() or not os.access(root, required_dir_access):
            raise DependencyUnavailable("Application storage is unavailable")
        if not database.is_file() or not os.access(database, os.R_OK | os.W_OK):
            raise DependencyUnavailable("Application registry is unavailable")

        connection: sqlite3.Connection | None = None
        try:
            # mode=rw refuses to create a missing registry during a readiness probe.
            uri = f"{database.resolve(strict=True).as_uri()}?mode=rw"
            connection = sqlite3.connect(uri, uri=True, timeout=1)
            check = connection.execute("PRAGMA quick_check(1)").fetchone()
            if check is None or check[0] != "ok":
                raise sqlite3.DatabaseError("registry integrity check failed")
        except (OSError, sqlite3.Error, ValueError) as exc:
            raise DependencyUnavailable("Application registry is unavailable") from exc
        finally:
            if connection is not None:
                connection.close()

        try:
            if not self.qdrant_client().collection_exists(self.settings.collection):
                raise DependencyUnavailable("Configured vector collection is unavailable")
        except DependencyUnavailable:
            raise
        except Exception as exc:
            raise DependencyUnavailable("Vector database is unavailable") from exc

    def check_generation_readiness(self) -> None:
        """Check that configured Ollama already serves the model, without loading or pulling it."""
        try:
            config = GenerationConfig.from_env()
            request = Request(config.base_url.rstrip("/") + "/api/tags", method="GET")
            opener = build_opener(ProxyHandler({}), _NoRedirect())
            with opener.open(request, timeout=min(config.timeout_seconds, 3.0)) as response:
                raw = response.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError("model listing exceeded probe limit")
            payload = json.loads(raw)
            models = payload.get("models") if isinstance(payload, dict) else None
            if not isinstance(models, list) or not any(
                isinstance(model, dict)
                and (model.get("name") == config.model or model.get("model") == config.model)
                for model in models
            ):
                raise DependencyUnavailable("Configured generation model is unavailable")
        except DependencyUnavailable:
            raise
        except Exception as exc:
            raise DependencyUnavailable("Generation service is unavailable") from exc

    def _base(self, strategy: str, tenant_id: str) -> Retriever:
        key = (tenant_id, strategy)
        if key not in self._retrievers:
            if strategy == "bm25":
                # The placeholder never embeds; payload-only access needs no ML model.
                store = self._store(FakeEmbedder(), validate=False)
                retriever = BM25Retriever(
                    store.scroll_payloads(tenant_id=tenant_id), tenant_id=tenant_id
                )
            elif strategy == "dense":
                # Check service/collection before the expensive local model load.
                self._store(FakeEmbedder(), validate=False)
                embedder = self.embedding()
                retriever = DenseRetriever(embedder, self._store(embedder, validate=True))
            else:
                retriever = HybridRetriever(
                    self._base("dense", tenant_id), self._base("bm25", tenant_id)
                )
            self._retrievers[key] = retriever
        return self._retrievers[key]

    def retrieve(self, request: RetrievalRequest, *, tenant_id: str) -> list[RetrievedChunk]:
        tenant_id = validate_tenant_id(tenant_id)
        with self.lock:
            try:
                if (
                    self._documents is not None
                    or (self.settings.document_dir / "inventory.sqlite3").exists()
                ):
                    self.documents().ensure_readable(tenant_id)
                retriever = self._base(request.strategy, tenant_id)
                if request.rerank:
                    key = (tenant_id, f"reranked_{request.strategy}")
                    if key not in self._retrievers:
                        config = RerankerConfig.from_env(
                            top_k=1, candidate_count=self.settings.rerank_candidates
                        )
                        self._retrievers[key] = CrossEncoderReranker(retriever, config)
                    retriever = self._retrievers[key]
                return retriever.retrieve(
                    request.question,
                    top_k=request.top_k,
                    doc_id=request.doc_id,
                    chunk_type=request.chunk_type,
                    min_score=request.min_score,
                    tenant_id=tenant_id,
                )
            except Exception as exc:
                raise DependencyUnavailable("Retrieval service or model unavailable") from exc

    def generator(self) -> AnswerGenerator:
        with self.lock:
            if self._generator is None:
                try:
                    config = GenerationConfig.from_env()
                    self._generator = AnswerGenerator(
                        build_provider(config),
                        max_evidence_chars=config.max_evidence_chars,
                        max_chunks=config.max_chunks,
                    )
                except Exception as exc:
                    raise DependencyUnavailable("Generation configuration unavailable") from exc
            return self._generator

    def close(self) -> None:
        with self.lock:
            if self._client is not None:
                self._client.close()
                self._client = None
            self._retrievers.clear()
            self._generator = None
            self._embedder = None
            if self._documents is not None:
                self._documents.close()
                self._documents = None
