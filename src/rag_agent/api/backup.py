"""Authenticated, offline backup/restore for the managed document registry.

This format intentionally excludes authentication tables, sessions, rate-limit
state, secrets, and Qdrant vectors. A successful restore is therefore a
SQLite/document-store component restore, not a complete RAG service recovery.
The caller must provide a private integrity key; the key is never archived.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from filelock import FileLock, Timeout

from rag_agent.tenancy import validate_tenant_id

FORMAT_VERSION = 1
REGISTRY_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
REGISTRY_NAME = "registry.sqlite3"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ARCHIVE_BYTES = 50 * 1024 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 100_000
RESTORE_STAGE_READY = ".restore-stage-ready"
_DOCUMENT_ID = re.compile(r"doc_[0-9a-f]{64}\Z")
_PDF_NAME = re.compile(r"[0-9a-f]{32}\.pdf\Z")
_PROCESSING_STATES = {"queued", "processing", "indexing", "deleting", "cleanup_required"}
_ALLOWED_TOP_LEVEL = {
    "inventory.sqlite3",
    "inventory.sqlite3-wal",
    "inventory.sqlite3-shm",
    "writer.lock",
    "tenants",
}


class BackupError(RuntimeError):
    """Backup or restore failed validation; no source data is changed."""


@dataclass(frozen=True, slots=True)
class BackupSummary:
    backup_id: str
    archive: Path
    file_count: int
    document_count: int
    tenant_count: int
    total_bytes: int


@dataclass(frozen=True, slots=True)
class RestoreSummary:
    backup_id: str
    destination: Path
    file_count: int
    document_count: int
    tenant_count: int
    total_bytes: int


def _validate_key(key: bytes) -> None:
    if not isinstance(key, bytes) or len(key) < 32:
        raise BackupError("A private integrity key of at least 32 bytes is required")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _sign_manifest(manifest: dict[str, Any], key: bytes) -> dict[str, Any]:
    unsigned = dict(manifest)
    unsigned.pop("hmac_sha256", None)
    signature = hmac.new(
        key, b"rag-backup-manifest-v1\0" + _canonical_json(unsigned), hashlib.sha256
    )
    return {**unsigned, "hmac_sha256": signature.hexdigest()}


def _verify_manifest(manifest: dict[str, Any], key: bytes) -> None:
    supplied = manifest.get("hmac_sha256")
    if not isinstance(supplied, str) or re.fullmatch(r"[0-9a-f]{64}", supplied) is None:
        raise BackupError("Backup manifest authentication failed")
    expected = _sign_manifest(manifest, key)["hmac_sha256"]
    if not hmac.compare_digest(supplied, expected):
        raise BackupError("Backup manifest authentication failed")


def _reject_symlink_components(path: Path) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise BackupError("Symlink paths are not accepted")


def _validate_source_root(source: Path) -> Path:
    if not source.is_absolute() or source.is_symlink() or not source.is_dir():
        raise BackupError("Document storage directory is unavailable or unsafe")
    _reject_symlink_components(source)
    return source.resolve(strict=True)


def _validate_destination_path(destination: Path, source: Path | None = None) -> Path:
    if not destination.is_absolute():
        raise BackupError("Destination must be an absolute path")
    _reject_symlink_components(destination.parent)
    if not destination.parent.is_dir():
        raise BackupError("Destination parent directory must already exist")
    if destination.exists() or destination.is_symlink():
        raise BackupError("Destination already exists")
    resolved_parent = destination.parent.resolve(strict=True)
    resolved = resolved_parent / destination.name
    if source is not None:
        try:
            resolved.relative_to(source)
        except ValueError:
            pass
        else:
            raise BackupError("Backup destination cannot be inside the source directory")
        try:
            source.relative_to(resolved)
        except ValueError:
            pass
        else:
            raise BackupError("Backup destination cannot contain the source directory")
    return resolved


def _is_safe_db_target(value: str) -> bool:
    try:
        parsed = json.loads(value)
        if (
            not isinstance(parsed, list)
            or len(parsed) != 2
            or not all(isinstance(item, str) and item for item in parsed)
        ):
            return False
        url = urlsplit(parsed[0])
        return bool(url.hostname) and not url.username and not url.password and not url.query
    except (json.JSONDecodeError, ValueError, TypeError):
        return False


def _snapshot_registry(source_db: Path, target_db: Path) -> tuple[int, int, int, int]:
    """Take a SQLite online snapshot in memory, then persist only safe tables."""
    if source_db.is_symlink() or not source_db.is_file():
        raise BackupError("Document registry is missing or unsafe")
    try:
        uri = source_db.resolve(strict=True).as_uri() + "?mode=ro"
        source = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            source.execute("PRAGMA query_only=ON")
            memory = sqlite3.connect(":memory:")
            try:
                source.backup(memory)
                integrity = memory.execute("PRAGMA integrity_check").fetchone()
                if integrity is None or integrity[0] != "ok":
                    raise BackupError("Source registry integrity check failed")
                tables = {
                    row[0]
                    for row in memory.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                    )
                }
                if not {"documents", "binding"}.issubset(tables):
                    raise BackupError("Source registry schema is incomplete")
                columns = frozenset(
                    row[1] for row in memory.execute("PRAGMA table_info(documents)")
                )
                required = {"id", "record", "file"}
                if columns not in (frozenset(required), frozenset(required | {"tenant_id"})):
                    raise BackupError("Source document registry schema is unsupported")
                rows = memory.execute("SELECT id, record, file FROM documents").fetchall()
                if "tenant_id" in columns:
                    rows = memory.execute(
                        "SELECT id, record, file, tenant_id FROM documents"
                    ).fetchall()
                else:
                    rows = [(*row, None) for row in rows]
                bindings = memory.execute("SELECT target FROM binding").fetchall()
                if len(bindings) > 1 or any(not _is_safe_db_target(row[0]) for row in bindings):
                    raise BackupError("Source collection binding is invalid")
                target = sqlite3.connect(target_db)
                try:
                    target.execute(
                        "CREATE TABLE documents (id TEXT PRIMARY KEY, record TEXT, file TEXT, "
                        "tenant_id TEXT)"
                    )
                    target.execute("CREATE TABLE binding (target TEXT PRIMARY KEY)")
                    target.executemany(
                        "INSERT INTO documents (id, record, file, tenant_id) VALUES (?, ?, ?, ?)",
                        rows,
                    )
                    target.executemany("INSERT INTO binding (target) VALUES (?)", bindings)
                    target.execute(f"PRAGMA user_version={REGISTRY_SCHEMA_VERSION}")
                    target.commit()
                    target_integrity = target.execute("PRAGMA integrity_check").fetchone()
                    if target_integrity is None or target_integrity[0] != "ok":
                        raise BackupError("Sanitized registry integrity check failed")
                finally:
                    target.close()
                tenant_ids: set[str] = set()
                for doc_id, record_json, filename, tenant_id in rows:
                    if not isinstance(doc_id, str) or _DOCUMENT_ID.fullmatch(doc_id) is None:
                        raise BackupError("Source document identifier is invalid")
                    try:
                        record = json.loads(record_json)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise BackupError("Source document metadata is invalid") from exc
                    if not isinstance(record, dict) or record.get("doc_id") != doc_id:
                        raise BackupError("Source document metadata is inconsistent")
                    if not isinstance(filename, str) or Path(filename).name != filename:
                        raise BackupError("Source document file reference is invalid")
                    if tenant_id is not None:
                        try:
                            tenant_ids.add(validate_tenant_id(tenant_id))
                        except ValueError as exc:
                            raise BackupError("Source tenant metadata is invalid") from exc
                    if record.get("status") in _PROCESSING_STATES:
                        raise BackupError("Document storage is not quiescent")
                return (
                    len(rows),
                    len(tenant_ids),
                    sum(
                        record.get("status") != "deleted"
                        for _, record_json, _, _ in rows
                        if isinstance(record := json.loads(record_json), dict)
                    ),
                    memory.execute("PRAGMA user_version").fetchone()[0],
                )
            finally:
                memory.close()
        finally:
            source.close()
    except BackupError:
        raise
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise BackupError("Could not create a consistent document registry snapshot") from exc


def _collect_tenant_files(
    root: Path, rows: list[tuple[str, str, str, str | None]]
) -> list[tuple[Path, str]]:
    """Collect only registered tenant files; reject unknown or symlinked paths."""
    allowed = set(_ALLOWED_TOP_LEVEL)
    try:
        for entry in root.iterdir():
            if entry.name not in allowed:
                raise BackupError("Unrecognized file in document storage; backup refused")
            if entry.is_symlink():
                raise BackupError("Symlink paths are not accepted")
        tenants_root = root / "tenants"
        registered: dict[str, dict[str, str]] = {}
        ready_documents: set[tuple[str, str]] = set()
        for doc_id, record_json, filename, tenant_id in rows:
            if tenant_id is None or not isinstance(tenant_id, str):
                continue
            status = json.loads(record_json).get("status")
            if status == "deleted":
                continue
            registered.setdefault(tenant_id, {})[filename] = doc_id
            if status == "ready":
                ready_documents.add((tenant_id, doc_id))
        collected: list[tuple[Path, str]] = []
        if not tenants_root.exists():
            if any(registered.values()):
                raise BackupError("Tenant document files are missing")
            return collected
        if tenants_root.is_symlink() or not tenants_root.is_dir():
            raise BackupError("Tenant storage is unsafe")
        seen_pdfs: set[tuple[str, str]] = set()
        seen_parsed: set[tuple[str, str]] = set()
        for tenant_dir in tenants_root.iterdir():
            if tenant_dir.is_symlink() or not tenant_dir.is_dir():
                raise BackupError("Tenant storage contains an unsafe entry")
            try:
                tenant_id = validate_tenant_id(tenant_dir.name)
            except ValueError as exc:
                raise BackupError("Tenant storage path is invalid") from exc
            expected_docs = registered.get(tenant_id, {})
            for child in tenant_dir.iterdir():
                if child.is_symlink():
                    raise BackupError("Tenant storage contains a symlink")
                if child.is_file():
                    if _PDF_NAME.fullmatch(child.name) is None or child.name not in expected_docs:
                        raise BackupError("Unregistered tenant file; backup refused")
                    seen_pdfs.add((tenant_id, child.name))
                    collected.append((child, f"files/tenants/{tenant_id}/{child.name}"))
                    continue
                if not child.is_dir() or _DOCUMENT_ID.fullmatch(child.name) is None:
                    raise BackupError("Unexpected tenant storage path")
                if child.name not in expected_docs.values():
                    raise BackupError("Unregistered parsed document; backup refused")
                for parsed in child.iterdir():
                    if parsed.is_symlink() or not parsed.is_file():
                        raise BackupError("Parsed document storage contains an unsafe entry")
                    if parsed.name != f"{child.name}.json":
                        raise BackupError("Incomplete or temporary parsed artifact; backup refused")
                    seen_parsed.add((tenant_id, child.name))
                    collected.append(
                        (parsed, f"files/tenants/{tenant_id}/{child.name}/{parsed.name}")
                    )
        for tenant_id, docs in registered.items():
            for filename in docs:
                if (tenant_id, filename) not in seen_pdfs:
                    raise BackupError("A registered document file is missing")
        if not ready_documents.issubset(seen_parsed):
            raise BackupError("A ready document's parsed metadata file is missing")
    except BackupError:
        raise
    except OSError as exc:
        raise BackupError("Could not inspect managed document files") from exc
    return sorted(collected, key=lambda pair: pair[1])


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def create_backup(
    document_dir: Path,
    destination: Path,
    *,
    integrity_key: bytes,
) -> BackupSummary:
    """Create a signed SQLite/document archive without capturing auth secrets.

    ``document_dir`` and ``destination`` must be absolute. The destination and
    its parent must be separate from the source; the parent must already exist.
    The archive is not encrypted and must be stored only on encrypted media or
    encrypted by an approved external process before leaving the host.
    """
    _validate_key(integrity_key)
    source = _validate_source_root(document_dir)
    output = _validate_destination_path(destination, source)
    lock = FileLock(str(source / "writer.lock"), thread_local=False)
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise BackupError("Document storage is active; backup requires quiescence") from exc
    partial: Path | None = None
    try:
        with tempfile.TemporaryDirectory(prefix=".rag-backup-build-", dir=output.parent) as work:
            work_dir = Path(work)
            registry = work_dir / REGISTRY_NAME
            document_count, tenant_count, active_count, source_user_version = _snapshot_registry(
                source / "inventory.sqlite3", registry
            )
            registry_connection = sqlite3.connect(registry)
            try:
                rows = registry_connection.execute(
                    "SELECT id, record, file, tenant_id FROM documents"
                ).fetchall()
            finally:
                registry_connection.close()
            documents = _collect_tenant_files(source, rows)
            for _doc_id, record_json, filename, tenant_id in rows:
                if tenant_id is None or json.loads(record_json).get("status") == "deleted":
                    continue
                if (source / "tenants" / tenant_id / filename).is_symlink():
                    raise BackupError("Document storage contains a symlink")
            payload_files: list[tuple[Path, str]] = [(registry, REGISTRY_NAME), *documents]
            records = []
            total_bytes = 0
            for path, archive_name in payload_files:
                size, digest = _hash_file(path)
                total_bytes += size
                records.append({"path": archive_name, "size": size, "sha256": digest})
            manifest: dict[str, Any] = {
                "format_version": FORMAT_VERSION,
                "registry_schema_version": REGISTRY_SCHEMA_VERSION,
                "source_sqlite_user_version": source_user_version,
                "backup_id": uuid.uuid4().hex,
                "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                "registry_row_count": document_count,
                "active_document_count": active_count,
                "tenant_count": tenant_count,
                "qdrant": {
                    "included": False,
                    "reason": "remote snapshot integration not implemented",
                },
                "authentication_state": {
                    "included": False,
                    "restore_action": "reprovision owner; sessions invalidated",
                },
                "files": records,
            }
            signed_manifest = _sign_manifest(manifest, integrity_key)
            manifest_bytes = _canonical_json(signed_manifest)
            fd, partial_name = tempfile.mkstemp(
                prefix=".rag-backup-", suffix=".partial", dir=output.parent
            )
            partial = Path(partial_name)
            os.chmod(partial, 0o600)
            with os.fdopen(fd, "w+b") as raw:
                with zipfile.ZipFile(
                    raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
                ) as archive:
                    for path, archive_name in payload_files:
                        archive.write(path, archive_name)
                    archive.writestr(MANIFEST_NAME, manifest_bytes)
                raw.flush()
                os.fsync(raw.fileno())
            # Hard-link publication is atomic and refuses to replace an existing backup.
            os.link(partial, output)
            partial.unlink()
            partial = None
            return BackupSummary(
                backup_id=manifest["backup_id"],
                archive=output,
                file_count=len(records),
                document_count=document_count,
                tenant_count=tenant_count,
                total_bytes=total_bytes,
            )
    except BackupError:
        raise
    except (OSError, sqlite3.Error, zipfile.BadZipFile, ValueError) as exc:
        raise BackupError("Backup creation failed; no completed archive was published") from exc
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)
        lock.release()


def _safe_member_path(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\\" in name or name.startswith("/"):
        raise BackupError("Backup contains an unsafe path")
    path = PurePosixPath(name)
    if any(part in {"", ".", ".."} for part in path.parts) or path.is_absolute():
        raise BackupError("Backup contains an unsafe path")
    return path


def _validate_manifest(manifest: Any, key: bytes) -> tuple[list[dict[str, Any]], int]:
    if not isinstance(manifest, dict):
        raise BackupError("Backup manifest is invalid")
    _verify_manifest(manifest, key)
    if (
        type(manifest.get("format_version")) is not int
        or manifest["format_version"] != FORMAT_VERSION
    ):
        raise BackupError("Backup format version is unsupported")
    if (
        type(manifest.get("registry_schema_version")) is not int
        or manifest["registry_schema_version"] != REGISTRY_SCHEMA_VERSION
    ):
        raise BackupError("Backup registry schema version is unsupported")
    if re.fullmatch(r"[0-9a-f]{32}", manifest.get("backup_id", "")) is None:
        raise BackupError("Backup identifier metadata is invalid")
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str) or not created_at.endswith("Z"):
        raise BackupError("Backup timestamp metadata is invalid")
    try:
        timestamp = datetime.fromisoformat(created_at[:-1] + "+00:00")
    except ValueError as exc:
        raise BackupError("Backup timestamp metadata is invalid") from exc
    if timestamp.utcoffset() is None or timestamp.utcoffset().total_seconds() != 0:
        raise BackupError("Backup timestamp metadata is invalid")
    counter_names = (
        "source_sqlite_user_version",
        "registry_row_count",
        "active_document_count",
        "tenant_count",
    )
    for name in counter_names:
        value = manifest.get(name)
        if type(value) is not int or value < 0:
            raise BackupError("Backup count metadata is invalid")
    if (
        manifest["active_document_count"] > manifest["registry_row_count"]
        or manifest["tenant_count"] > manifest["registry_row_count"]
    ):
        raise BackupError("Backup count metadata is inconsistent")
    qdrant = manifest.get("qdrant")
    authentication_state = manifest.get("authentication_state")
    if not isinstance(qdrant, dict) or qdrant.get("included") is not False:
        raise BackupError("Backup Qdrant metadata is invalid")
    if (
        not isinstance(authentication_state, dict)
        or authentication_state.get("included") is not False
    ):
        raise BackupError("Backup authentication metadata is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files or len(files) > MAX_ARCHIVE_MEMBERS:
        raise BackupError("Backup manifest file list is invalid")
    seen: set[str] = set()
    total = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise BackupError("Backup manifest file entry is invalid")
        name, size, digest = entry.get("path"), entry.get("size"), entry.get("sha256")
        path = _safe_member_path(name)
        if name in seen or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise BackupError("Backup manifest file entry is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", digest or "") is None:
            raise BackupError("Backup manifest file hash is invalid")
        allowed = name == REGISTRY_NAME or (
            len(path.parts) >= 4
            and path.parts[:2] == ("files", "tenants")
            and re.fullmatch(r"[0-9a-f]{32}", path.parts[2]) is not None
        )
        if not allowed:
            raise BackupError("Backup contains an unsupported path")
        seen.add(name)
        total += size
    if REGISTRY_NAME not in seen or total > MAX_ARCHIVE_BYTES:
        raise BackupError("Backup exceeds supported size or lacks its registry")
    return files, total


def _safe_extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    relative = _safe_member_path(info.filename)
    if info.is_dir():
        raise BackupError("Unexpected directory entry in backup")
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(unix_mode) or info.flag_bits & 0x1:
        raise BackupError("Encrypted or symlink archive entries are not supported")
    if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
        raise BackupError("Backup compression method is unsupported")
    if relative.parts == (REGISTRY_NAME,):
        out = target / REGISTRY_NAME
    else:
        if len(relative.parts) < 4 or relative.parts[:2] != ("files", "tenants"):
            raise BackupError("Backup contains an unsupported path")
        tenant = validate_tenant_id(relative.parts[2])
        out = target.joinpath(*relative.parts)
        if tenant != relative.parts[2]:
            raise BackupError("Backup tenant path is invalid")
    out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _reject_symlink_components(out.parent)
    with archive.open(info, "r") as source, out.open("xb") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    os.chmod(out, 0o600)


def _validate_restored_registry(registry: Path, files_root: Path) -> tuple[int, int, int]:
    try:
        db = sqlite3.connect(registry)
        try:
            if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise BackupError("Restored registry integrity check failed")
            if db.execute("PRAGMA user_version").fetchone()[0] != REGISTRY_SCHEMA_VERSION:
                raise BackupError("Restored registry schema version is unsupported")
            tables = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables != {"documents", "binding"}:
                raise BackupError("Restored registry contains unsupported tables")
            columns = {row[1] for row in db.execute("PRAGMA table_info(documents)")}
            if columns != {"id", "record", "file", "tenant_id"}:
                raise BackupError("Restored document registry schema is unsupported")
            rows = db.execute("SELECT id, record, file, tenant_id FROM documents").fetchall()
            bindings = db.execute("SELECT target FROM binding").fetchall()
            if len(bindings) > 1 or any(not _is_safe_db_target(row[0]) for row in bindings):
                raise BackupError("Restored collection binding is invalid")
        finally:
            db.close()
        tenant_ids: set[str] = set()
        active = 0
        expected_pdfs: set[Path] = set()
        expected_parsed: set[Path] = set()
        allowed_directories: set[Path] = {files_root, files_root / "tenants"}
        for doc_id, record_json, filename, tenant_id in rows:
            if _DOCUMENT_ID.fullmatch(str(doc_id)) is None:
                raise BackupError("Restored document identifier is invalid")
            record = json.loads(record_json)
            if not isinstance(record, dict) or record.get("doc_id") != doc_id:
                raise BackupError("Restored document metadata is inconsistent")
            if record.get("status") != "deleted":
                active += 1
            if tenant_id is None:
                continue
            tenant_id = validate_tenant_id(tenant_id)
            tenant_ids.add(tenant_id)
            if record.get("status") != "deleted":
                if _PDF_NAME.fullmatch(str(filename)) is None:
                    raise BackupError("Restored PDF reference is invalid")
                pdf = files_root / "tenants" / tenant_id / filename
                expected_pdfs.add(pdf)
                tenant_dir = files_root / "tenants" / tenant_id
                allowed_directories.add(tenant_dir)
                parsed_dir = tenant_dir / doc_id
                allowed_directories.add(parsed_dir)
                parsed_file = parsed_dir / f"{doc_id}.json"
                if record.get("status") == "ready":
                    expected_parsed.add(parsed_file)

        actual_files: set[Path] = set()
        if files_root.exists():
            for path in files_root.rglob("*"):
                if path == registry:
                    continue
                if path.is_symlink():
                    raise BackupError("Restored document storage contains a symlink")
                if path.is_dir():
                    if path not in allowed_directories:
                        raise BackupError(
                            "Restored document storage contains an unexpected directory"
                        )
                elif path.is_file():
                    if path not in expected_pdfs and path not in expected_parsed:
                        # Failed documents may retain parsed output from a prior attempt.
                        relative = path.relative_to(files_root)
                        parts = relative.parts
                        is_registered_parsed = (
                            len(parts) == 4
                            and parts[0] == "tenants"
                            and validate_tenant_id(parts[1]) == parts[1]
                            and _DOCUMENT_ID.fullmatch(parts[2]) is not None
                            and parts[3] == f"{parts[2]}.json"
                            and files_root / parts[0] / parts[1] / parts[2] in allowed_directories
                        )
                        if not is_registered_parsed:
                            raise BackupError(
                                "Restored document storage contains an unexpected file"
                            )
                    actual_files.add(path)
                else:
                    raise BackupError("Restored document storage contains an unsupported entry")
        if not expected_pdfs.issubset(actual_files) or not expected_parsed.issubset(actual_files):
            raise BackupError("Restored document files are incomplete")
        return len(rows), len(tenant_ids), active
    except BackupError:
        raise
    except (OSError, sqlite3.Error, json.JSONDecodeError, ValueError, TypeError) as exc:
        raise BackupError("Restored registry validation failed") from exc


def list_restore_staging_directories(parent: Path) -> tuple[Path, ...]:
    """List restore work directories for operator inspection; never removes them."""
    if not parent.is_absolute() or parent.is_symlink() or not parent.is_dir():
        raise BackupError("Restore staging parent is missing or unsafe")
    _reject_symlink_components(parent)
    return tuple(
        sorted(
            (
                candidate
                for candidate in parent.glob(".rag-restore-*")
                if candidate.is_dir() and not candidate.is_symlink()
            ),
            key=lambda candidate: candidate.name,
        )
    )


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry where the platform exposes directory fsync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    if os.name != "nt":
        for file_path in (path for path in root.rglob("*") if path.is_file()):
            descriptor = os.open(file_path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in [*directories, root]:
        _fsync_directory(directory)


def _write_restore_stage_ready(stage_root: Path, *, backup_id: str, destination_name: str) -> None:
    marker = stage_root / RESTORE_STAGE_READY
    payload = _canonical_json(
        {
            "format_version": FORMAT_VERSION,
            "backup_id": backup_id,
            "destination_name": destination_name,
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".restore-ready-", suffix=".partial", dir=stage_root
    )
    temporary = Path(temporary_name)
    try:
        os.chmod(temporary, 0o600)
        with os.fdopen(descriptor, "wb") as ready_file:
            ready_file.write(payload)
            ready_file.flush()
            os.fsync(ready_file.fileno())
        # The unique private staging directory makes the ready marker name exclusive.
        os.rename(temporary, marker)
        _fsync_directory(stage_root)
    finally:
        temporary.unlink(missing_ok=True)


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    """Atomically rename a staged directory without replacing a target."""
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(errno.EEXIST, "restore destination already exists", destination)
    if sys.platform == "win32":
        # Windows os.rename fails if the target already exists.
        os.rename(source, destination)
        return
    if sys.platform.startswith("linux"):
        try:
            renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
        except (AttributeError, OSError) as exc:
            raise BackupError("Atomic no-replace directory rename is unavailable") from exc
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(error, "restore destination already exists", destination)
        if error in {errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP}:
            raise BackupError("Atomic no-replace directory rename is unsupported") from OSError(
                error, os.strerror(error), destination
            )
        raise OSError(error, os.strerror(error), destination)
    raise BackupError("Atomic no-replace directory rename is unsupported on this platform")


def _publish_restore(staging: Path, destination: Path, *, expected_backup_id: str) -> None:
    """Atomically publish a fully validated staged directory without replacement."""
    if staging.is_symlink() or not staging.is_dir():
        raise BackupError("Validated restore staging directory is unavailable")
    marker = staging.parent / RESTORE_STAGE_READY
    try:
        if marker.is_symlink() or not marker.is_file() or marker.stat().st_size > 4096:
            raise BackupError("Restore staging completion marker is invalid")
        ready = json.loads(marker.read_bytes())
    except BackupError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise BackupError("Restore staging directory is not marked complete") from exc
    if (
        not isinstance(ready, dict)
        or set(ready) != {"format_version", "backup_id", "destination_name"}
        or ready.get("format_version") != FORMAT_VERSION
        or ready.get("backup_id") != expected_backup_id
        or ready.get("destination_name") != destination.name
    ):
        raise BackupError("Restore staging completion marker is invalid")
    source_parent = staging.parent
    try:
        _rename_directory_no_replace(staging, destination)
    except FileExistsError as exc:
        raise BackupError("Restore destination already exists") from exc
    except OSError as exc:
        raise BackupError(
            "Atomic restore publication failed; destination was not published"
        ) from exc
    try:
        _fsync_directory(source_parent)
        _fsync_directory(destination.parent)
    except OSError as exc:
        raise BackupError(
            "Restore was atomically published, but parent directory durability "
            "could not be confirmed"
        ) from exc


def restore_backup(
    archive_path: Path,
    destination: Path,
    *,
    integrity_key: bytes,
) -> RestoreSummary:
    """Restore verified registry/files to a new directory; never overwrite."""
    _validate_key(integrity_key)
    if not archive_path.is_absolute() or archive_path.is_symlink() or not archive_path.is_file():
        raise BackupError("Backup archive is missing or unsafe")
    _reject_symlink_components(archive_path)
    archive_path = archive_path.resolve(strict=True)
    output = _validate_destination_path(destination)
    if archive_path == output or archive_path.is_relative_to(output):
        raise BackupError("Restore destination cannot contain the backup archive")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise BackupError("Backup archive exceeds the supported size")
    staging: Path | None = None
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ARCHIVE_MEMBERS + 1:
                raise BackupError("Backup contains too many files")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or MANIFEST_NAME not in names:
                raise BackupError("Backup archive entries are invalid")
            manifest_info = archive.getinfo(MANIFEST_NAME)
            if manifest_info.file_size > MAX_MANIFEST_BYTES:
                raise BackupError("Backup manifest is too large")
            manifest = json.loads(archive.read(manifest_info))
            files, total = _validate_manifest(manifest, integrity_key)
            expected = {entry["path"] for entry in files} | {MANIFEST_NAME}
            if set(names) != expected:
                raise BackupError("Backup is missing files or contains unexpected entries")
            info_by_name = {info.filename: info for info in infos}
            for entry in files:
                info = info_by_name[entry["path"]]
                if info.file_size != entry["size"]:
                    raise BackupError("Backup file size does not match its manifest")
                if info.file_size > MAX_ARCHIVE_BYTES:
                    raise BackupError("Backup member exceeds the supported size")
                mode = (info.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    raise BackupError("Symlink archive entries are not supported")
            with tempfile.TemporaryDirectory(prefix=".rag-restore-", dir=output.parent) as work:
                staging = Path(work)
                for entry in files:
                    info = info_by_name[entry["path"]]
                    _safe_extract_member(archive, info, staging)
                    extracted = staging.joinpath(*_safe_member_path(entry["path"]).parts)
                    size, digest = _hash_file(extracted)
                    if size != entry["size"] or not hmac.compare_digest(digest, entry["sha256"]):
                        raise BackupError("Backup file checksum validation failed")
                registry = staging / REGISTRY_NAME
                files_root = staging / "files"
                row_count, tenant_count, active_count = _validate_restored_registry(
                    registry, files_root
                )
                if (
                    row_count != manifest["registry_row_count"]
                    or tenant_count != manifest["tenant_count"]
                    or active_count != manifest["active_document_count"]
                ):
                    raise BackupError("Backup manifest counts do not match the restored registry")
                # Restore only document data. Auth tables are intentionally absent.
                final_staging = staging / "final"
                final_staging.mkdir(mode=0o700)
                os.replace(registry, final_staging / "inventory.sqlite3")
                tenant_files = files_root / "tenants"
                if tenant_files.exists():
                    os.replace(tenant_files, final_staging / "tenants")
                (files_root).rmdir()
                final_rows, final_tenants, final_active = _validate_restored_registry(
                    final_staging / "inventory.sqlite3", final_staging
                )
                if (final_rows, final_tenants, final_active) != (
                    row_count,
                    tenant_count,
                    active_count,
                ):
                    raise BackupError("Final restore staging inventory changed during assembly")
                _fsync_directory_tree(final_staging)
                _write_restore_stage_ready(
                    staging,
                    backup_id=manifest["backup_id"],
                    destination_name=output.name,
                )
                _publish_restore(
                    final_staging,
                    output,
                    expected_backup_id=manifest["backup_id"],
                )
            return RestoreSummary(
                backup_id=str(manifest["backup_id"]),
                destination=output,
                file_count=len(files),
                document_count=row_count,
                tenant_count=tenant_count,
                total_bytes=total,
            )
    except BackupError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError, KeyError, TypeError) as exc:
        raise BackupError(
            "Backup validation or restore failed; destination was not accepted"
        ) from exc
