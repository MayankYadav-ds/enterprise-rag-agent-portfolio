"""Durable, single-writer document lifecycle using the existing indexing pipeline.

SQLite records intent before vector mutations. Uncertain writes are compensated
by exact document-ID deletion; unresolved compensation blocks API retrieval.
Only newly registered, API-owned documents can be mutated. Never adopt CLI data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import fitz
from filelock import FileLock, Timeout
from pydantic import BaseModel

from rag_agent.api.storage import private_directory, private_file, private_tree
from rag_agent.chunking.pipeline import chunk_document
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.ingestion.models import document_id_from_hash, sha256_for_file
from rag_agent.ingestion.pipeline import ingest_pdf
from rag_agent.tenancy import validate_tenant_id

if TYPE_CHECKING:
    from rag_agent.api.runtime import Runtime
    from rag_agent.embeddings.store import QdrantStore

logger = logging.getLogger("rag_agent.api.documents")
State = Literal[
    "queued",
    "processing",
    "indexing",
    "ready",
    "failed",
    "cleanup_required",
    "deleting",
    "deleted",
    "external",
]


class DocumentError(RuntimeError):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code = status, code


class DocumentRecord(BaseModel):
    doc_id: str
    filename: str
    status: State
    managed: bool = True
    owns_vectors: bool = False
    size_bytes: int | None = None
    page_count: int | None = None
    chunk_count: int = 0
    updated_at: str | None = None
    error: str | None = None


def safe_filename(value: str) -> str:
    """Reject path syntax rather than silently renaming untrusted input."""
    if (
        not value
        or len(value) > 120
        or value != value.strip()
        or not value.lower().endswith(".pdf")
        or any(c in value for c in '/\\:<>"|?*[]')
        or any(unicodedata.category(c).startswith("C") for c in value)
        or value.startswith(".")
        or value.split(".")[0].upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"{prefix}{i}" for prefix in ("COM", "LPT") for i in range(1, 10)],
        }
    ):
        raise DocumentError(422, "invalid_filename", "Use a plain PDF filename without path syntax")
    return value


class DocumentManager:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.root = runtime.settings.document_dir.resolve()
        private_directory(self.root)
        self.guard = FileLock(str(self.root / "writer.lock"), thread_local=False)
        try:
            self.guard.acquire(timeout=0)
        except Timeout as exc:
            raise DocumentError(
                503, "writer_busy", "Document storage is owned by another worker"
            ) from exc
        self.db = self.root / "inventory.sqlite3"
        try:
            with self.connection() as db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS documents "
                    "(id TEXT PRIMARY KEY, record TEXT, file TEXT, tenant_id TEXT)"
                )
                columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
                if "tenant_id" not in columns:
                    # Existing rows remain NULL and are intentionally invisible to tenant APIs.
                    db.execute("ALTER TABLE documents ADD COLUMN tenant_id TEXT")
                db.execute("CREATE TABLE IF NOT EXISTS binding (target TEXT PRIMARY KEY)")
                target = json.dumps([runtime.settings.qdrant_url, runtime.settings.collection])
                bindings = db.execute("SELECT target FROM binding").fetchall()
                if bindings and bindings[0][0] != target:
                    raise DocumentError(
                        503, "storage_mismatch", "Document storage belongs to another collection"
                    )
                db.execute("INSERT OR IGNORE INTO binding VALUES (?)", (target,))
            self.recover_all_scoped()
        except Exception:
            self.guard.release()
            raise

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db, timeout=10)
        private_file(self.db)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def tenant_root(self, tenant_id: str) -> Path:
        tenant_id = validate_tenant_id(tenant_id)
        tenants = self.root / "tenants"
        if tenants.is_symlink():
            raise DocumentError(503, "invalid_registry", "Tenant storage requires inspection")
        private_directory(tenants)
        root = tenants / tenant_id
        if root.is_symlink():
            raise DocumentError(403, "tenant_access_denied", "Tenant storage unavailable")
        private_directory(root)
        if (
            tenants.resolve().parent != self.root.resolve()
            or root.resolve().parent != tenants.resolve()
        ):
            raise DocumentError(403, "tenant_access_denied", "Tenant storage unavailable")
        return root

    @staticmethod
    def _tenant_document_id(tenant_id: str, source_doc_id: str) -> str:
        tenant_id = validate_tenant_id(tenant_id)
        digest = hashlib.sha256(f"{tenant_id}:{source_doc_id}".encode()).hexdigest()
        return f"doc_{digest}"

    def records(self, tenant_id: str) -> list[DocumentRecord]:
        tenant_id = validate_tenant_id(tenant_id)
        with self.connection() as db:
            return [
                DocumentRecord.model_validate_json(row[0])
                for row in db.execute(
                    "SELECT record FROM documents WHERE tenant_id=?", (tenant_id,)
                )
            ]

    def get(self, doc_id: str, tenant_id: str) -> DocumentRecord:
        tenant_id = validate_tenant_id(tenant_id)
        if not re.fullmatch(r"doc_[0-9a-f]{64}", doc_id):
            raise DocumentError(404, "not_found", "Managed document not found")
        with self.connection() as db:
            row = db.execute(
                "SELECT record FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id)
            ).fetchone()
        if row is None:
            raise DocumentError(404, "not_found", "Managed document not found")
        return DocumentRecord.model_validate_json(row[0])

    def _save(self, record: DocumentRecord, tenant_id: str, file: Path | None = None) -> None:
        tenant_id = validate_tenant_id(tenant_id)
        record.updated_at = datetime.now(UTC).isoformat()
        with self.connection() as db:
            if file is not None:
                db.execute(
                    "INSERT OR REPLACE INTO documents "
                    "(id, record, file, tenant_id) VALUES (?,?,?,?)",
                    (record.doc_id, record.model_dump_json(), file.name, tenant_id),
                )
            else:
                db.execute(
                    "UPDATE documents SET record=? WHERE id=? AND tenant_id=?",
                    (record.model_dump_json(), record.doc_id, tenant_id),
                )
        logger.info(
            json.dumps(
                {"event": "document_state", "doc_id": record.doc_id, "status": record.status}
            )
        )

    def _file(self, doc_id: str, tenant_id: str) -> Path:
        tenant_id = validate_tenant_id(tenant_id)
        with self.connection() as db:
            row = db.execute(
                "SELECT file FROM documents WHERE id=? AND tenant_id=?", (doc_id, tenant_id)
            ).fetchone()
        if row is None:
            raise DocumentError(404, "not_found", "Managed document not found")
        name = row[0]
        # Even a corrupt registry must not address arbitrary filesystem paths.
        if not re.fullmatch(r"[0-9a-f]{32}\.pdf", name):
            raise DocumentError(503, "invalid_registry", "Document storage requires inspection")
        root = self.tenant_root(tenant_id)
        file = root / name
        if file.resolve().parent != root.resolve() or file.is_symlink():
            raise DocumentError(503, "invalid_registry", "Document storage requires inspection")
        return file

    def _store(self) -> QdrantStore:
        return self.runtime._store(FakeEmbedder(), validate=False)

    def ensure_readable(self, tenant_id: str) -> None:
        if any(
            r.status in {"indexing", "cleanup_required", "deleting"}
            for r in self.records(tenant_id)
        ):
            raise DocumentError(
                503, "recovery_required", "Document cleanup must complete before retrieval"
            )

    def accept(self, file: Path, filename: str, tenant_id: str) -> DocumentRecord:
        tenant_id = validate_tenant_id(tenant_id)
        with self.runtime.lock:
            tenant_root = self.tenant_root(tenant_id)
            if file.is_symlink() or file.resolve().parent != tenant_root.resolve():
                raise DocumentError(403, "tenant_access_denied", "Tenant storage unavailable")
            self.ensure_readable(tenant_id)
            source_id = document_id_from_hash(sha256_for_file(str(file)))
            doc_id = self._tenant_document_id(tenant_id, source_id)
            previous = next((r for r in self.records(tenant_id) if r.doc_id == doc_id), None)
            if previous and previous.status not in {"failed", "deleted"}:
                raise DocumentError(409, "duplicate_document", "This PDF is already registered")
            if self._store().scroll_ids(doc_id, tenant_id=tenant_id):
                raise DocumentError(409, "duplicate_document", "This PDF is already indexed")
            if previous:
                # A retry must not orphan the previous API-owned copy of the same bytes.
                self._file(doc_id, tenant_id).unlink(missing_ok=True)
            record = DocumentRecord(
                doc_id=doc_id, filename=filename, status="queued", size_bytes=file.stat().st_size
            )
            self._save(record, tenant_id, file)
            return record

    def process(self, doc_id: str, tenant_id: str) -> None:
        tenant_id = validate_tenant_id(tenant_id)
        with self.runtime.lock:
            record = self.get(doc_id, tenant_id)
            if record.status != "queued":
                return
            mutation_started = False
            try:
                self.ensure_readable(tenant_id)
                record.status = "processing"
                self._save(record, tenant_id)
                file = self._file(doc_id, tenant_id)
                # Inspect page count/encryption before full extraction, without duplicating parsing.
                with fitz.open(file) as pdf:
                    if pdf.needs_pass or pdf.page_count > self.runtime.settings.document_max_pages:
                        raise ValueError("Unsupported PDF limits")
                tenant_root = self.tenant_root(tenant_id)
                parsed_root = tenant_root / doc_id
                document, pages = ingest_pdf(file, parsed_root)
                private_tree(parsed_root)
                if document.doc_id != document_id_from_hash(sha256_for_file(str(file))):
                    raise ValueError("Content identity changed")
                chunks = chunk_document(
                    document, pages, strategy="recursive", chunk_size=400, chunk_overlap=39
                )
                chunks = [
                    replace(
                        chunk,
                        doc_id=doc_id,
                        chunk_id=chunk.chunk_id.replace(document.doc_id, doc_id, 1),
                    )
                    for chunk in chunks
                ]
                if not chunks or len(chunks) > self.runtime.settings.document_max_chunks:
                    raise ValueError("No usable chunks or chunk limit exceeded")
                record.page_count, record.chunk_count = document.num_pages, len(chunks)
                embedder = self.runtime.embedding()
                store = self.runtime._store(embedder, validate=True)
                # Never adopt or patch data introduced by an external writer.
                if store.scroll_ids(doc_id, tenant_id=tenant_id):
                    raise ValueError("Document already indexed externally")
                record.status = "indexing"
                record.owns_vectors = True
                self._save(record, tenant_id)
                mutation_started = True
                store.sync(
                    doc_id, chunks, embedder, source_file=record.filename, tenant_id=tenant_id
                )
                if len(store.scroll_ids(doc_id, tenant_id=tenant_id)) != len(chunks):
                    raise ValueError("Incomplete vector write")
                record.status, record.error = "ready", None
                self._save(record, tenant_id)
            except Exception:
                if mutation_started:
                    self._compensate(record, tenant_id)
                else:
                    record.status, record.error = (
                        "failed",
                        "PDF processing failed; check format, limits, and services",
                    )
                    self._save(record, tenant_id)
            finally:
                self.runtime.invalidate_retrieval(tenant_id)

    def _erase_vectors(self, doc_id: str, tenant_id: str) -> None:
        from qdrant_client.http import models

        tenant_id = validate_tenant_id(tenant_id)
        store = self._store()
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    models.FieldCondition(
                        key="tenant_id", match=models.MatchValue(value=tenant_id)
                    ),
                ]
            )
        )
        store.client.delete(store.collection_name, points_selector=selector, wait=True)
        if store.scroll_ids(doc_id, tenant_id=tenant_id):
            raise RuntimeError("Vector deletion not confirmed")

    def _compensate(self, record: DocumentRecord, tenant_id: str) -> None:
        try:
            self._erase_vectors(record.doc_id, tenant_id)
            record.owns_vectors = False
            record.chunk_count = 0
            record.status, record.error = (
                "failed",
                "Ingestion failed; partial vector writes were removed. Re-upload to retry.",
            )
        except Exception:
            record.status, record.error = (
                "cleanup_required",
                "Cleanup could not be confirmed. Retry recovery before searching.",
            )
        self._save(record, tenant_id)

    def delete(self, doc_id: str, tenant_id: str) -> DocumentRecord:
        tenant_id = validate_tenant_id(tenant_id)
        with self.runtime.lock:
            record = self.get(doc_id, tenant_id)
            if record.status == "deleted":
                return record
            record.status, record.error = "deleting", None
            self._save(record, tenant_id)
            try:
                if record.owns_vectors:
                    self._erase_vectors(doc_id, tenant_id)
                record.owns_vectors = False
                # Remove only known API-owned artifacts; never recursively delete directories.
                self._file(doc_id, tenant_id).unlink(missing_ok=True)
                for suffix in (".json", ".json.tmp"):
                    parsed_root = self.tenant_root(tenant_id) / doc_id
                    parsed = parsed_root / (doc_id + suffix)
                    if parsed.resolve().parent != parsed_root.resolve() or parsed.is_symlink():
                        raise ValueError("Unsafe parsed artifact path")
                    parsed.unlink(missing_ok=True)
                record.status, record.chunk_count = "deleted", 0
                self._save(record, tenant_id)
            except Exception as exc:
                record.error = "Deletion incomplete; retry deletion to finish safely"
                self._save(record, tenant_id)
                raise DocumentError(503, "deletion_incomplete", record.error) from exc
            finally:
                self.runtime.invalidate_retrieval(tenant_id)
            return record

    def recover(self, tenant_id: str) -> None:
        tenant_id = validate_tenant_id(tenant_id)
        with self.runtime.lock:
            for record in self.records(tenant_id):
                if record.status in {"indexing", "cleanup_required"}:
                    self._compensate(record, tenant_id)
                elif record.status == "deleting":
                    try:
                        self.delete(record.doc_id, tenant_id)
                    except DocumentError:
                        pass  # Durable deleting state remains visible and blocks retrieval.
                elif record.status in {"queued", "processing"}:
                    record.status, record.error = (
                        "failed",
                        "Processing was interrupted. Re-upload to retry.",
                    )
                    self._save(record, tenant_id)
            self.runtime.invalidate_retrieval(tenant_id)

    def recover_all_scoped(self) -> None:
        """Recover explicit tenant rows; legacy NULL tenant rows remain quarantined."""
        with self.connection() as db:
            tenant_ids = {
                row[0]
                for row in db.execute(
                    "SELECT DISTINCT tenant_id FROM documents WHERE tenant_id IS NOT NULL"
                )
            }
        for tenant_id in tenant_ids:
            try:
                self.recover(validate_tenant_id(tenant_id))
            except ValueError as exc:
                raise DocumentError(
                    503, "invalid_registry", "Tenant registry requires inspection"
                ) from exc

    def inventory(self, tenant_id: str) -> list[DocumentRecord]:
        tenant_id = validate_tenant_id(tenant_id)
        records = self.records(tenant_id)
        managed = {r.doc_id for r in records if r.status != "deleted"}
        external: dict[str, DocumentRecord] = {}
        for payload in self._store().scroll_payloads(tenant_id=tenant_id).values():
            doc_id = payload.get("doc_id")
            if not isinstance(doc_id, str) or doc_id in managed:
                continue
            if doc_id not in external:
                filename = Path(str(payload.get("source_file", "")).replace("\\", "/")).name
                external[doc_id] = DocumentRecord(
                    doc_id=doc_id, filename=filename, status="external", managed=False
                )
            external[doc_id].chunk_count += 1
        return [r for r in records if r.status != "deleted"] + list(external.values())

    def close(self) -> None:
        self.guard.release()
