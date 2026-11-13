"""Offline backup/restore tests with synthetic tenant documents only."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from rag_agent.api.app import create_app
from rag_agent.api.backup import (
    RESTORE_STAGE_READY,
    BackupError,
    _canonical_json,
    _rename_directory_no_replace,
    _sign_manifest,
    create_backup,
    list_restore_staging_directories,
    restore_backup,
)
from rag_agent.api.documents import DocumentRecord
from rag_agent.api.settings import APISettings
from rag_agent.embeddings.fake import FakeEmbedder
from rag_agent.embeddings.store import QdrantStore

KEY = b"disposable-test-integrity-key-material-32-bytes"
TENANT_A = "a" * 32
TENANT_B = "b" * 32
DOC_A = "doc_" + "1" * 64
DOC_B = "doc_" + "2" * 64
LEGACY = "doc_" + "3" * 64


def _make_source(root: Path) -> dict[str, bytes]:
    root.mkdir()
    (root / "tenants" / TENANT_A / f"{DOC_A}").mkdir(parents=True)
    (root / "tenants" / TENANT_B / f"{DOC_B}").mkdir(parents=True)
    pdf_a_name = "1" * 32 + ".pdf"
    pdf_b_name = "2" * 32 + ".pdf"
    pdf_a = b"%PDF-1.4 synthetic tenant A document\n"
    pdf_b = b"%PDF-1.4 synthetic tenant B document\n"
    parsed_a = json.dumps({"doc_id": DOC_A, "text": "synthetic A content"}).encode()
    parsed_b = json.dumps({"doc_id": DOC_B, "text": "synthetic B content"}).encode()
    (root / "tenants" / TENANT_A / pdf_a_name).write_bytes(pdf_a)
    (root / "tenants" / TENANT_B / pdf_b_name).write_bytes(pdf_b)
    (root / "tenants" / TENANT_A / DOC_A / f"{DOC_A}.json").write_bytes(parsed_a)
    (root / "tenants" / TENANT_B / DOC_B / f"{DOC_B}.json").write_bytes(parsed_b)
    with sqlite3.connect(root / "inventory.sqlite3") as db:
        db.executescript(
            """
            CREATE TABLE documents (id TEXT PRIMARY KEY, record TEXT, file TEXT, tenant_id TEXT);
            CREATE TABLE binding (target TEXT PRIMARY KEY);
            CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT, password_hash TEXT);
            CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, csrf_hash TEXT);
            CREATE TABLE login_attempts (bucket TEXT PRIMARY KEY, failures INTEGER);
            CREATE TABLE upload_attempts (bucket TEXT PRIMARY KEY, attempts INTEGER);
            """
        )
        db.execute(
            "INSERT INTO binding VALUES (?)",
            (json.dumps(["http://localhost:6333", "rag_chunks"]),),
        )
        for doc_id, tenant_id, filename, original_name in (
            (DOC_A, TENANT_A, pdf_a_name, "tenant-a.pdf"),
            (DOC_B, TENANT_B, pdf_b_name, "tenant-b.pdf"),
        ):
            record = DocumentRecord(
                doc_id=doc_id, filename=original_name, status="ready", chunk_count=1
            )
            db.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (doc_id, record.model_dump_json(), filename, tenant_id),
            )
        legacy_record = DocumentRecord(doc_id=LEGACY, filename="legacy.pdf", status="ready")
        db.execute(
            "INSERT INTO documents VALUES (?, ?, ?, NULL)",
            (LEGACY, legacy_record.model_dump_json(), "3" * 32 + ".pdf"),
        )
        # Synthetic values make sure authentication state is omitted from the archive.
        db.execute(
            "INSERT INTO users VALUES (?, ?, ?)", (TENANT_A, "synthetic", "fake-hash-not-secret")
        )
        db.execute(
            "INSERT INTO sessions VALUES (?, ?)", ("synthetic-session-hash", "synthetic-csrf-hash")
        )
        db.execute("INSERT INTO login_attempts VALUES (?, ?)", ("synthetic-bucket", 5))
        db.execute("INSERT INTO upload_attempts VALUES (?, ?)", ("synthetic-upload", 3))
    return {
        f"tenants/{TENANT_A}/{pdf_a_name}": pdf_a,
        f"tenants/{TENANT_B}/{pdf_b_name}": pdf_b,
        f"tenants/{TENANT_A}/{DOC_A}/{DOC_A}.json": parsed_a,
        f"tenants/{TENANT_B}/{DOC_B}/{DOC_B}.json": parsed_b,
    }


def _read_members(path: Path) -> tuple[dict[str, bytes], dict]:
    with zipfile.ZipFile(path) as archive:
        values = {info.filename: archive.read(info) for info in archive.infolist()}
    manifest = json.loads(values.pop("manifest.json"))
    return values, manifest


def _write_signed_archive(path: Path, members: dict[str, bytes], manifest: dict) -> None:
    manifest = _sign_manifest(manifest, KEY)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", _canonical_json(manifest))


@pytest.fixture
def backup_case(tmp_path):
    source = tmp_path / "source"
    payloads = _make_source(source)
    archive = tmp_path / "synthetic.ragbackup"
    summary = create_backup(source, archive, integrity_key=KEY)
    return tmp_path, source, payloads, archive, summary


def test_backup_is_transactional_sanitized_and_manifested(backup_case):
    tmp_path, source, payloads, archive, summary = backup_case
    members, manifest = _read_members(archive)
    assert summary.document_count == 3
    assert summary.tenant_count == 2
    assert manifest["registry_schema_version"] == 1
    assert manifest["qdrant"]["included"] is False
    assert manifest["authentication_state"]["included"] is False
    assert set(payloads).issubset({name.removeprefix("files/") for name in members})
    assert not (tmp_path / ".rag-backup-build-").exists()
    with sqlite3.connect(":memory:") as db:
        db.deserialize(members["registry.sqlite3"])
        tables = {
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        assert tables == {"documents", "binding"}
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (LEGACY,)).fetchone()[0]
            is None
        )
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert b"fake-hash-not-secret" not in members["registry.sqlite3"]
    assert b"synthetic-session-hash" not in members["registry.sqlite3"]
    assert b"synthetic-csrf-hash" not in members["registry.sqlite3"]
    assert b"synthetic-bucket" not in members["registry.sqlite3"]
    assert archive.stat().st_size > 0


def test_restore_verifies_files_and_tenant_isolation(backup_case, tmp_path):
    _, source, original_payloads, archive, summary = backup_case
    destination = tmp_path / "restored-data"
    result = restore_backup(archive, destination, integrity_key=KEY)
    assert result.backup_id == summary.backup_id
    assert result.document_count == 3
    assert result.tenant_count == 2
    assert result.file_count == summary.file_count
    for relative, expected in original_payloads.items():
        assert (destination / relative).read_bytes() == expected
        assert (
            hashlib.sha256((destination / relative).read_bytes()).digest()
            == hashlib.sha256(expected).digest()
        )
    with sqlite3.connect(destination / "inventory.sqlite3") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (DOC_A,)).fetchone()[0]
            == TENANT_A
        )
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (DOC_B,)).fetchone()[0]
            == TENANT_B
        )
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (LEGACY,)).fetchone()[0]
            is None
        )

    # Validate restored registry visibility through the actual DocumentManager,
    # backed only by a new in-memory Qdrant client and an empty disposable collection.
    settings = APISettings(document_dir=destination)
    runtime = create_app(settings).state.runtime
    qdrant = QdrantClient(":memory:")
    runtime._client = qdrant
    QdrantStore(FakeEmbedder(), client=qdrant)
    manager = runtime.documents()
    try:
        assert [row.doc_id for row in manager.inventory(TENANT_A)] == [DOC_A]
        assert [row.doc_id for row in manager.inventory(TENANT_B)] == [DOC_B]
    finally:
        runtime.close()


def test_restore_rejects_signed_wrong_tenant_file_metadata(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    registry_path = tmp_path / "mutated-registry.sqlite3"
    registry_path.write_bytes(members["registry.sqlite3"])
    with sqlite3.connect(registry_path) as db:
        db.execute("UPDATE documents SET tenant_id=? WHERE id=?", (TENANT_B, DOC_A))
    members["registry.sqlite3"] = registry_path.read_bytes()
    registry_entry = next(item for item in manifest["files"] if item["path"] == "registry.sqlite3")
    registry_entry["size"] = len(members["registry.sqlite3"])
    registry_entry["sha256"] = hashlib.sha256(members["registry.sqlite3"]).hexdigest()
    wrong_tenant_archive = tmp_path / "wrong-tenant.ragbackup"
    _write_signed_archive(wrong_tenant_archive, members, manifest)

    destination = tmp_path / "wrong-tenant-restore"
    with pytest.raises(BackupError, match="PDF reference|file is missing|unexpected directory"):
        restore_backup(wrong_tenant_archive, destination, integrity_key=KEY)
    assert not destination.exists()
    assert list_restore_staging_directories(tmp_path) == ()


def test_manifest_tampering_is_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    manifest["created_at"] = "2000-01-01T00:00:00Z"
    tampered = tmp_path / "tampered.ragbackup"
    with zipfile.ZipFile(tampered, "w") as output:
        for name, data in members.items():
            output.writestr(name, data)
        output.writestr("manifest.json", _canonical_json(manifest))
    with pytest.raises(BackupError, match="authentication"):
        restore_backup(tampered, tmp_path / "restore-tampered", integrity_key=KEY)
    assert not (tmp_path / "restore-tampered").exists()


def test_wrong_key_and_incompatible_schema_are_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    with pytest.raises(BackupError, match="authentication"):
        restore_backup(archive, tmp_path / "wrong-key", integrity_key=b"x" * 32)
    members, manifest = _read_members(archive)
    manifest["registry_schema_version"] = 99
    bad_schema = tmp_path / "wrong-schema.ragbackup"
    _write_signed_archive(bad_schema, members, manifest)
    with pytest.raises(BackupError, match="schema version"):
        restore_backup(bad_schema, tmp_path / "wrong-schema", integrity_key=KEY)


def test_missing_archive_member_and_corrupt_payload_are_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    missing = tmp_path / "missing-member.ragbackup"
    with zipfile.ZipFile(missing, "w") as output:
        for name, data in members.items():
            if name != "files/tenants/" + TENANT_B + "/" + "2" * 32 + ".pdf":
                output.writestr(name, data)
        output.writestr("manifest.json", _canonical_json(manifest))
    with pytest.raises(BackupError, match="missing files"):
        restore_backup(missing, tmp_path / "restore-missing", integrity_key=KEY)

    corrupt_members, corrupt_manifest = _read_members(archive)
    payload_name = f"files/tenants/{TENANT_A}/" + "1" * 32 + ".pdf"
    corrupt_members[payload_name] += b"tampered"
    corrupt = tmp_path / "corrupt-payload.ragbackup"
    # Keep the authenticated manifest unchanged: the member hash must reject corruption.
    with zipfile.ZipFile(corrupt, "w") as output:
        for name, data in corrupt_members.items():
            output.writestr(name, data)
        output.writestr("manifest.json", _canonical_json(corrupt_manifest))
    with pytest.raises(BackupError, match="size|checksum"):
        restore_backup(corrupt, tmp_path / "restore-corrupt", integrity_key=KEY)


def test_truncated_manifest_and_unexpected_archive_entry_are_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)

    truncated = tmp_path / "truncated-manifest.ragbackup"
    with zipfile.ZipFile(truncated, "w") as output:
        for name, data in members.items():
            output.writestr(name, data)
        output.writestr("manifest.json", b"{truncated")
    truncated_destination = tmp_path / "restore-truncated"
    with pytest.raises(BackupError):
        restore_backup(truncated, truncated_destination, integrity_key=KEY)
    assert not truncated_destination.exists()

    unexpected = tmp_path / "unexpected-entry.ragbackup"
    extra_members = {**members, "unexpected.bin": b"synthetic unlisted data"}
    _write_signed_archive(unexpected, extra_members, manifest)
    unexpected_destination = tmp_path / "restore-unexpected"
    with pytest.raises(BackupError, match="unexpected entries"):
        restore_backup(unexpected, unexpected_destination, integrity_key=KEY)
    assert not unexpected_destination.exists()


def test_corrupt_sqlite_snapshot_is_rejected_even_with_valid_member_hash(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    members["registry.sqlite3"] = b"not a SQLite database"
    registry_entry = next(item for item in manifest["files"] if item["path"] == "registry.sqlite3")
    registry_entry["size"] = len(members["registry.sqlite3"])
    registry_entry["sha256"] = hashlib.sha256(members["registry.sqlite3"]).hexdigest()
    corrupted = tmp_path / "corrupt-registry.ragbackup"
    _write_signed_archive(corrupted, members, manifest)

    destination = tmp_path / "restore-corrupt-registry"
    with pytest.raises(BackupError, match="registry validation failed"):
        restore_backup(corrupted, destination, integrity_key=KEY)
    assert not destination.exists()


@pytest.mark.parametrize(
    "field, value",
    [
        ("backup_id", "not-a-uuid"),
        ("created_at", "not-a-timestamp"),
        ("registry_row_count", True),
        ("active_document_count", -1),
        ("qdrant", {"included": True}),
        ("authentication_state", {"included": True}),
    ],
)
def test_invalid_signed_manifest_metadata_is_rejected(backup_case, tmp_path, field, value):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    manifest[field] = value
    malformed = tmp_path / f"invalid-metadata-{field}.ragbackup"
    _write_signed_archive(malformed, members, manifest)

    destination = tmp_path / f"restore-invalid-{field}"
    with pytest.raises(BackupError, match="metadata"):
        restore_backup(malformed, destination, integrity_key=KEY)
    assert not destination.exists()


def test_manifest_record_counts_must_match_restored_registry(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    manifest["registry_row_count"] += 1
    inconsistent = tmp_path / "inconsistent-counts.ragbackup"
    _write_signed_archive(inconsistent, members, manifest)

    destination = tmp_path / "restore-inconsistent-counts"
    with pytest.raises(BackupError, match="counts do not match"):
        restore_backup(inconsistent, destination, integrity_key=KEY)
    assert not destination.exists()


def test_symlink_archive_member_is_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    symlink_name = f"files/tenants/{TENANT_A}/" + "8" * 32 + ".pdf"
    members[symlink_name] = b"../../outside.pdf"
    manifest["files"].append(
        {
            "path": symlink_name,
            "size": len(members[symlink_name]),
            "sha256": hashlib.sha256(members[symlink_name]).hexdigest(),
        }
    )
    symlink_archive = tmp_path / "symlink-member.ragbackup"
    signed_manifest = _sign_manifest(manifest, KEY)
    with zipfile.ZipFile(symlink_archive, "w") as output:
        for name, data in members.items():
            if name == symlink_name:
                info = zipfile.ZipInfo(name)
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                output.writestr(info, data)
            else:
                output.writestr(name, data)
        output.writestr("manifest.json", _canonical_json(signed_manifest))

    destination = tmp_path / "restore-symlink-member"
    with pytest.raises(BackupError, match="Symlink"):
        restore_backup(symlink_archive, destination, integrity_key=KEY)
    assert not destination.exists()


def test_missing_registered_pdf_prevents_backup(tmp_path):
    source = tmp_path / "source"
    _make_source(source)
    (source / "tenants" / TENANT_A / ("1" * 32 + ".pdf")).unlink()

    destination = tmp_path / "missing-pdf.ragbackup"
    with pytest.raises(BackupError, match="PDF|document file|incomplete"):
        create_backup(source, destination, integrity_key=KEY)
    assert not destination.exists()


def test_signed_path_traversal_entry_is_rejected(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    members, manifest = _read_members(archive)
    source_name = f"files/tenants/{TENANT_A}/" + "1" * 32 + ".pdf"
    traversal = f"files/tenants/{TENANT_A}/../../../../escape.bin"
    payload = members.pop(source_name)
    members[traversal] = payload
    entry = next(item for item in manifest["files"] if item["path"] == source_name)
    entry.update(path=traversal, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
    malicious = tmp_path / "traversal.ragbackup"
    _write_signed_archive(malicious, members, manifest)
    with pytest.raises(BackupError, match="unsafe path"):
        restore_backup(malicious, tmp_path / "restore-traversal", integrity_key=KEY)
    assert not (tmp_path / "escape.bin").exists()


def test_backup_rejects_symlinks_and_unknown_files(tmp_path):
    source = tmp_path / "source"
    _make_source(source)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"synthetic")
    link = source / "tenants" / TENANT_A / ("9" * 32 + ".pdf")
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable: {type(exc).__name__}")
    with pytest.raises(BackupError, match="symlink|unregistered"):
        create_backup(source, tmp_path / "symlink.ragbackup", integrity_key=KEY)
    assert not (tmp_path / "symlink.ragbackup").exists()


def test_backup_rejects_missing_ready_document_metadata(tmp_path):
    source = tmp_path / "source"
    _make_source(source)
    parsed = source / "tenants" / TENANT_A / DOC_A / f"{DOC_A}.json"
    parsed.unlink()
    with pytest.raises(BackupError, match="parsed metadata file is missing"):
        create_backup(source, tmp_path / "incomplete.ragbackup", integrity_key=KEY)
    assert not (tmp_path / "incomplete.ragbackup").exists()


def test_backup_fails_closed_on_transient_state_missing_database_and_source_target(tmp_path):
    source = tmp_path / "source"
    _make_source(source)
    with sqlite3.connect(source / "inventory.sqlite3") as db:
        db.execute(
            "UPDATE documents SET record=json_set(record, '$.status', 'processing') WHERE id=?",
            (DOC_A,),
        )
    with pytest.raises(BackupError, match="quiescent"):
        create_backup(source, tmp_path / "busy.ragbackup", integrity_key=KEY)
    assert not (tmp_path / "busy.ragbackup").exists()
    with pytest.raises(BackupError, match="inside the source"):
        create_backup(source, source / "nested.ragbackup", integrity_key=KEY)
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(BackupError, match="registry"):
        create_backup(missing, tmp_path / "missing.ragbackup", integrity_key=KEY)


def test_existing_destinations_are_never_overwritten(backup_case, tmp_path):
    _, source, _, archive, _ = backup_case
    existing_backup = tmp_path / "existing.ragbackup"
    existing_backup.write_bytes(b"keep this")
    with pytest.raises(BackupError, match="already exists"):
        create_backup(source, existing_backup, integrity_key=KEY)
    assert existing_backup.read_bytes() == b"keep this"
    existing_restore = tmp_path / "existing-restore"
    existing_restore.mkdir()
    sentinel = existing_restore / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(BackupError, match="already exists"):
        restore_backup(archive, existing_restore, integrity_key=KEY)
    assert sentinel.read_text(encoding="utf-8") == "preserve"

    successful_target = tmp_path / "restore-once"
    restore_backup(archive, successful_target, integrity_key=KEY)
    before = {
        path.relative_to(successful_target).as_posix(): hashlib.sha256(path.read_bytes()).digest()
        for path in successful_target.rglob("*")
        if path.is_file()
    }
    with pytest.raises(BackupError, match="already exists"):
        restore_backup(archive, successful_target, integrity_key=KEY)
    after = {
        path.relative_to(successful_target).as_posix(): hashlib.sha256(path.read_bytes()).digest()
        for path in successful_target.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_interrupted_backup_and_partial_restore_leave_no_partial_destination(
    backup_case, tmp_path, monkeypatch
):
    _, source, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    original_write = backup_module.zipfile.ZipFile.write

    def fail_write(self, *args, **kwargs):
        raise OSError("synthetic interrupted archive write")

    monkeypatch.setattr(backup_module.zipfile.ZipFile, "write", fail_write)
    interrupted_archive = tmp_path / "interrupted.ragbackup"
    with pytest.raises(BackupError, match="creation failed"):
        create_backup(source, interrupted_archive, integrity_key=KEY)
    assert not interrupted_archive.exists()
    assert not list(tmp_path.glob(".rag-backup-*.partial"))
    monkeypatch.setattr(backup_module.zipfile.ZipFile, "write", original_write)

    original_extract = backup_module._safe_extract_member
    calls = 0

    def fail_extract(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic interrupted restore")
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(backup_module, "_safe_extract_member", fail_extract)
    partial_target = tmp_path / "partial-restore"
    with pytest.raises(BackupError):
        restore_backup(archive, partial_target, integrity_key=KEY)
    assert not partial_target.exists()
    assert not list(tmp_path.glob(".rag-restore-*"))


def test_keyboard_interrupt_during_restore_publication_rolls_back_destination(
    backup_case, tmp_path, monkeypatch
):
    _, _, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    destination = tmp_path / "interrupted-publication"

    def interrupt_before_publish(source, target):
        raise KeyboardInterrupt

    monkeypatch.setattr(backup_module, "_rename_directory_no_replace", interrupt_before_publish)
    with pytest.raises(KeyboardInterrupt):
        restore_backup(archive, destination, integrity_key=KEY)

    assert not destination.exists()
    assert archive.is_file()
    assert list_restore_staging_directories(tmp_path) == ()


def test_interrupt_immediately_after_atomic_publish_keeps_complete_destination(
    backup_case, tmp_path, monkeypatch
):
    _, _, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    destination = tmp_path / "interrupt-after-move"
    original_publish = backup_module._rename_directory_no_replace

    def publish_then_interrupt(source, target):
        original_publish(source, target)
        raise KeyboardInterrupt

    monkeypatch.setattr(backup_module, "_rename_directory_no_replace", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        restore_backup(archive, destination, integrity_key=KEY)

    assert destination.is_dir()
    with sqlite3.connect(destination / "inventory.sqlite3") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
    assert archive.is_file()
    assert list_restore_staging_directories(tmp_path) == ()


def _run_restore_crash_process(
    mode: str, archive: Path, destination: Path
) -> subprocess.CompletedProcess:
    script = r"""
import json, os, sys
from pathlib import Path
import rag_agent.api.backup as module
request = json.loads(sys.stdin.read())
mode = request["mode"]
if mode == "during_extract":
    original = module._safe_extract_member
    calls = 0
    def crash_after_first_extract(*args, **kwargs):
        global calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            os._exit(71)
        return result
    module._safe_extract_member = crash_after_first_extract
elif mode == "before_publish":
    def crash_before_publish(*args, **kwargs):
        os._exit(72)
    module._publish_restore = crash_before_publish
elif mode == "after_publish":
    original = module._rename_directory_no_replace
    def publish_then_crash(source, destination):
        original(source, destination)
        os._exit(73)
    module._rename_directory_no_replace = publish_then_crash
module.restore_backup(
    Path(request["archive"]),
    Path(request["destination"]),
    integrity_key=bytes.fromhex(request["key"]),
)
"""
    environment = {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH")
        if key in os.environ
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    payload = json.dumps(
        {
            "mode": mode,
            "archive": str(archive),
            "destination": str(destination),
            "key": KEY.hex(),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        input=payload,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
        check=False,
    )


@pytest.mark.parametrize(
    "mode, expected_exit, ready",
    [("during_extract", 71, False), ("before_publish", 72, True)],
)
def test_process_crash_leaves_no_target_and_abandoned_stage_is_classifiable(
    backup_case, tmp_path, mode, expected_exit, ready
):
    _, _, _, archive, _ = backup_case
    destination = tmp_path / f"crash-{mode}"
    child = _run_restore_crash_process(mode, archive, destination)
    assert child.returncode == expected_exit, child.stderr
    assert not destination.exists()
    stages = list_restore_staging_directories(tmp_path)
    assert len(stages) == 1
    marker = stages[0] / RESTORE_STAGE_READY
    assert marker.exists() is ready
    assert ((stages[0] / "final").is_dir()) is ready
    assert archive.is_file()


def test_process_crash_immediately_after_publish_leaves_complete_destination(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    original_hash = hashlib.sha256(archive.read_bytes()).digest()
    destination = tmp_path / "crash-after-publish"
    child = _run_restore_crash_process("after_publish", archive, destination)
    assert child.returncode == 73, child.stderr
    assert destination.is_dir()
    with sqlite3.connect(destination / "inventory.sqlite3") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (DOC_A,)).fetchone()[0]
            == TENANT_A
        )
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (DOC_B,)).fetchone()[0]
            == TENANT_B
        )
    assert hashlib.sha256(archive.read_bytes()).digest() == original_hash


def test_tampered_completion_marker_backup_id_blocks_publication(
    backup_case, tmp_path, monkeypatch
):
    _, _, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    original_writer = backup_module._write_restore_stage_ready

    def write_then_tamper(stage_root, *, backup_id, destination_name):
        original_writer(stage_root, backup_id=backup_id, destination_name=destination_name)
        marker = stage_root / RESTORE_STAGE_READY
        ready = json.loads(marker.read_bytes())
        ready["backup_id"] = "f" * 32
        marker.write_bytes(_canonical_json(ready))

    monkeypatch.setattr(backup_module, "_write_restore_stage_ready", write_then_tamper)
    destination = tmp_path / "tampered-marker-restore"
    with pytest.raises(BackupError, match="completion marker"):
        restore_backup(archive, destination, integrity_key=KEY)
    assert not destination.exists()
    assert list_restore_staging_directories(tmp_path) == ()


def test_partial_extraction_write_is_not_published(backup_case, tmp_path, monkeypatch):
    _, _, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    def partial_then_fail(source, destination, length):
        destination.write(b"partial synthetic payload")
        raise OSError("synthetic write interruption")

    monkeypatch.setattr(backup_module.shutil, "copyfileobj", partial_then_fail)
    destination = tmp_path / "partial-write-restore"
    with pytest.raises(BackupError):
        restore_backup(archive, destination, integrity_key=KEY)
    assert not destination.exists()
    assert list_restore_staging_directories(tmp_path) == ()


def test_restore_rejects_symlink_in_destination_ancestry(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    link_parent = tmp_path / "linked-parent"
    try:
        link_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"directory symlinks unavailable: {type(exc).__name__}")

    with pytest.raises(BackupError, match="Symlink"):
        restore_backup(archive, link_parent / "restore", integrity_key=KEY)
    assert not (real_parent / "restore").exists()


def test_publication_failure_keeps_destination_absent(backup_case, tmp_path, monkeypatch):
    _, _, _, archive, _ = backup_case
    from rag_agent.api import backup as backup_module

    def fail_publication(source, destination):
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(backup_module, "_rename_directory_no_replace", fail_publication)
    destination = tmp_path / "rename-failure-restore"
    with pytest.raises(BackupError, match="destination was not published"):
        restore_backup(archive, destination, integrity_key=KEY)
    assert not destination.exists()
    assert list_restore_staging_directories(tmp_path) == ()


def test_two_process_restores_to_same_destination_publish_exactly_one(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    destination = tmp_path / "concurrent-restore"
    script = r"""
import json, sys
from pathlib import Path
from rag_agent.api.backup import BackupError, restore_backup
request = json.loads(sys.stdin.readline())
print("READY", flush=True)
sys.stdin.readline()
try:
    restore_backup(
        Path(request["archive"]),
        Path(request["destination"]),
        integrity_key=bytes.fromhex(request["key"]),
    )
except BackupError:
    print("REJECTED", flush=True)
else:
    print("PUBLISHED", flush=True)
"""
    environment = {
        key: os.environ[key]
        for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH")
        if key in os.environ
    }
    environment.update(
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    request = json.dumps(
        {"archive": str(archive), "destination": str(destination), "key": KEY.hex()}
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(2)
    ]
    for process in processes:
        assert process.stdin is not None and process.stdout is not None
        process.stdin.write(request + "\n")
        process.stdin.flush()
    assert [process.stdout.readline().strip() for process in processes] == ["READY", "READY"]
    for process in processes:
        assert process.stdin is not None
        process.stdin.write("go\n")
        process.stdin.flush()
        process.stdin.close()
    results = []
    for process in processes:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        results.append(stdout.strip())

    assert sorted(results) == ["PUBLISHED", "REJECTED"]
    assert destination.is_dir()
    with sqlite3.connect(destination / "inventory.sqlite3") as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert db.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 3


def test_linux_publication_uses_atomic_no_replace_renameat2(tmp_path, monkeypatch):
    calls = []

    def fake_renameat2(*args):
        calls.append(args)
        return 0

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "rag_agent.api.backup.ctypes.CDLL",
        lambda *args, **kwargs: SimpleNamespace(renameat2=fake_renameat2),
    )
    source = tmp_path / ".rag-restore-test" / "final"
    destination = tmp_path / "restored"
    _rename_directory_no_replace(source, destination)

    assert len(calls) == 1
    assert calls[0][0] == -100
    assert calls[0][1] == os.fsencode(source)
    assert calls[0][2] == -100
    assert calls[0][3] == os.fsencode(destination)
    assert calls[0][4] == 1


def test_linux_publication_fails_closed_when_renameat2_is_unsupported(tmp_path, monkeypatch):
    def unavailable_renameat2(*args):
        return -1

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        "rag_agent.api.backup.ctypes.CDLL",
        lambda *args, **kwargs: SimpleNamespace(renameat2=unavailable_renameat2),
    )
    monkeypatch.setattr("rag_agent.api.backup.ctypes.get_errno", lambda: errno.ENOSYS)
    with pytest.raises(BackupError, match="unsupported"):
        _rename_directory_no_replace(tmp_path / "stage", tmp_path / "destination")


def test_registry_backup_keeps_legacy_rows_unassigned(backup_case, tmp_path):
    _, _, _, archive, _ = backup_case
    destination = tmp_path / "restore"
    restore_backup(archive, destination, integrity_key=KEY)
    with sqlite3.connect(destination / "inventory.sqlite3") as db:
        assert (
            db.execute("SELECT tenant_id FROM documents WHERE id=?", (LEGACY,)).fetchone()[0]
            is None
        )
        assert db.execute("PRAGMA user_version").fetchone()[0] == 1
