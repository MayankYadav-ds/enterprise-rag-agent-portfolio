"""Offline legacy tenant migration planning tests with synthetic data only."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from rag_agent.api.tenant_migration import (
    MigrationPlanningError,
    OwnershipEvidence,
    VectorObservation,
    plan_legacy_tenant_migration,
)

TENANT_A = "a" * 32
TENANT_B = "b" * 32
DOC_A = "doc_" + "1" * 64
DOC_B = "doc_" + "2" * 64
DOC_ORPHAN = "doc_" + "3" * 64


def _row(
    doc_id: str,
    *,
    filename: str = "legacy.pdf",
    display_filename: str | None = None,
    tenant_id=None,
    status="ready",
    chunks=1,
):
    record = json.dumps(
        {
            "doc_id": doc_id,
            "status": status,
            "chunk_count": chunks,
            "filename": display_filename or filename,
        }
    )
    return (doc_id, record, filename, tenant_id)


def _database(path: Path, rows: list[tuple]) -> Path:
    connection = sqlite3.connect(path)
    try:
        # No primary key intentionally lets tests model malformed legacy duplicates.
        connection.execute(
            "CREATE TABLE documents (id TEXT, record TEXT, file TEXT, tenant_id TEXT)"
        )
        connection.executemany("INSERT INTO documents VALUES (?, ?, ?, ?)", rows)
        connection.commit()
    finally:
        connection.close()
    document_root = path.parent / "documents"
    for doc_id, _, filename, tenant_id in rows:
        if not isinstance(filename, str) or Path(filename).name != filename:
            continue
        storage_root = document_root
        if tenant_id is not None:
            storage_root = storage_root / "tenants" / tenant_id
        storage_root.mkdir(parents=True, exist_ok=True)
        (storage_root / filename).write_bytes(
            f"synthetic disposable document for {doc_id}".encode()
        )
    return path


def _point_id(seed: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _qdrant_points(*items: tuple[str, str | None, str | None]):
    client = QdrantClient(":memory:")
    collection = "migration_dry_run"
    client.create_collection(
        collection,
        vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE),
    )
    structures = []
    for point_id, doc_id, tenant_id in items:
        payload = {}
        if doc_id is not None:
            payload["doc_id"] = doc_id
        if tenant_id is not None:
            payload["tenant_id"] = tenant_id
        structures.append(models.PointStruct(id=point_id, vector=[1.0, 0.0], payload=payload))
    if structures:
        client.upsert(collection, points=structures, wait=True)
    observations = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        observations.extend(
            VectorObservation(
                point_id=str(point.id),
                document_id=(point.payload or {}).get("doc_id"),
                tenant_id=(point.payload or {}).get("tenant_id"),
            )
            for point in records
        )
        if offset is None:
            break
    return client, collection, observations


def _evidence(doc_id: str, tenant_id: str, ref: str | None = None, *, verified=True):
    return OwnershipEvidence(
        document_id=doc_id,
        tenant_id=tenant_id,
        evidence_reference=ref or f"audit-record-{doc_id[-4:]}",
        evidence_source="server_side_owner_record",
        verified=verified,
    )


def _plan(database, points, evidence=()):
    return plan_legacy_tenant_migration(
        database,
        document_root=database.parent / "documents",
        vector_points=points,
        vector_inventory_complete=True,
        ownership_evidence=evidence,
    )


def test_verified_owner_assignment_is_proposed_with_rollback_and_no_source_write(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A, chunks=2)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    document_file = db.parent / "documents" / "legacy.pdf"
    document_before = hashlib.sha256(document_file.read_bytes()).hexdigest()
    client, collection, points = _qdrant_points(
        (_point_id("a-1"), DOC_A, None), (_point_id("a-2"), DOC_A, None)
    )
    try:
        plan = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        assert [(x.document_id, x.tenant_id) for x in plan.registry_assignments] == [
            (DOC_A, TENANT_A)
        ]
        assert {x.point_id for x in plan.vector_assignments} == {
            _point_id("a-1"),
            _point_id("a-2"),
        }
        assert plan.decisions[0].classification == "VERIFIED_OWNER"
        assert len(plan.rollback_steps) == 3
        assert plan.automatic_reassignment_allowed is False
        # The proposal is not applied: legacy points remain unscoped and invisible
        # to a tenant-filtered Qdrant inventory.
        visible_a, _ = client.scroll(
            collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(key="tenant_id", match=models.MatchValue(value=TENANT_A))
                ]
            ),
            with_payload=True,
            with_vectors=False,
        )
        assert visible_a == []
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before
        assert hashlib.sha256(document_file.read_bytes()).hexdigest() == document_before
    finally:
        client.close()


def test_plan_stores_only_a_hash_of_the_ownership_evidence_reference(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    reference = "synthetic-audit-reference"
    try:
        plan = _plan(db, points, [_evidence(DOC_A, TENANT_A, reference)])
        assignment = plan.registry_assignments[0]
        assert (
            assignment.evidence_reference_sha256
            == hashlib.sha256(reference.encode("utf-8")).hexdigest()
        )
        assert not hasattr(assignment, "evidence_reference")
    finally:
        client.close()


def test_conflicting_or_unverified_evidence_is_quarantined(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    try:
        conflict = _plan(
            db,
            points,
            [_evidence(DOC_A, TENANT_A, "record-a"), _evidence(DOC_A, TENANT_B, "record-b")],
        )
        unverified = _plan(db, points, [_evidence(DOC_A, TENANT_A, verified=False)])
        oversized_reference = _plan(
            db,
            points,
            [_evidence(DOC_A, TENANT_A, "x" * 513)],
        )
        for plan in (conflict, unverified, oversized_reference):
            assert plan.decisions[0].classification == "AMBIGUOUS_OWNER"
            assert plan.decisions[0].quarantined
            assert not plan.registry_assignments
            assert not plan.vector_assignments
            assert not plan.automatic_reassignment_allowed
    finally:
        client.close()


def test_missing_owner_evidence_and_unscoped_vectors_remain_unowned(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    try:
        plan = _plan(db, points)
        assert plan.decisions[0].classification == "UNOWNED"
        assert plan.decisions[0].reason == "no_verified_server_side_ownership_evidence"
        assert not plan.registry_assignments and not plan.vector_assignments
    finally:
        client.close()


def test_duplicate_registry_ids_are_quarantined(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A), _row(DOC_A, filename="other.pdf")])
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    try:
        plan = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        assert plan.registry_row_count == 2
        assert plan.decisions[0].classification == "AMBIGUOUS_OWNER"
        assert plan.decisions[0].reason == "duplicate_or_conflicting_registry_rows"
        assert not plan.registry_assignments
    finally:
        client.close()


def test_shared_filenames_are_not_used_as_ownership_evidence(tmp_path):
    db = _database(
        tmp_path / "inventory.sqlite3",
        [
            _row(DOC_A, filename="doc-a-storage.pdf", display_filename="shared.pdf"),
            _row(DOC_B, filename="doc-b-storage.pdf", display_filename="shared.pdf"),
        ],
    )
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None), (_point_id("b"), DOC_B, None))
    try:
        plan = _plan(
            db,
            points,
            [_evidence(DOC_A, TENANT_A), _evidence(DOC_B, TENANT_B)],
        )
        assert {item.document_id: item.tenant_id for item in plan.registry_assignments} == {
            DOC_A: TENANT_A,
            DOC_B: TENANT_B,
        }
        assert len(plan.vector_assignments) == 2
        assert {item.document_id: item.tenant_id for item in plan.vector_assignments} == {
            DOC_A: TENANT_A,
            DOC_B: TENANT_B,
        }
    finally:
        client.close()


def test_orphan_vectors_missing_registry_and_evidence_are_quarantined(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    client, _, points = _qdrant_points(
        (_point_id("orphan"), DOC_ORPHAN, TENANT_B),
        (_point_id("no-doc"), None, None),
        (_point_id("legacy"), DOC_A, None),
    )
    try:
        plan = _plan(db, points, [_evidence(DOC_ORPHAN, TENANT_B)])
        assert {point.reason for point in plan.quarantined_points} >= {
            "orphan_vector",
            "missing_or_invalid_document_id",
        }
        assert plan.quarantined_evidence[0].reason == "ownership_evidence_without_registry_row"
        assert plan.decisions[0].classification == "UNOWNED"
        assert not plan.registry_assignments and not plan.vector_assignments
    finally:
        client.close()


def test_duplicate_point_ids_and_conflicting_vector_scopes_are_quarantined(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A, chunks=2)])
    duplicate = _point_id("same-point")
    duplicate_plan = _plan(
        db,
        [
            VectorObservation(duplicate, DOC_A, None),
            VectorObservation(duplicate, DOC_A, None),
        ],
        [_evidence(DOC_A, TENANT_A)],
    )
    assert duplicate_plan.decisions[0].classification == "AMBIGUOUS_OWNER"
    assert duplicate_plan.decisions[0].reason == "duplicate_qdrant_point_id"
    assert not duplicate_plan.vector_assignments

    client, _, points = _qdrant_points(
        (_point_id("a"), DOC_A, TENANT_A), (_point_id("b"), DOC_A, TENANT_B)
    )
    try:
        conflict_plan = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        assert conflict_plan.decisions[0].classification == "AMBIGUOUS_OWNER"
        assert conflict_plan.decisions[0].reason == "conflicting_vector_tenant_scopes"
        assert not conflict_plan.registry_assignments
    finally:
        client.close()


def test_missing_or_malformed_vector_scope_is_quarantined(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A, chunks=2)])
    one_point = VectorObservation(_point_id("only-point"), DOC_A, None)
    missing = _plan(db, [one_point], [_evidence(DOC_A, TENANT_A)])
    assert missing.decisions[0].classification == "AMBIGUOUS_OWNER"
    assert missing.decisions[0].reason == "missing_or_extra_qdrant_points"
    malformed = _plan(
        db,
        [
            VectorObservation(_point_id("a"), DOC_A, "../tenant"),
            VectorObservation(_point_id("b"), DOC_A, None),
        ],
        [_evidence(DOC_A, TENANT_A)],
    )
    assert malformed.decisions[0].classification == "AMBIGUOUS_OWNER"
    assert malformed.decisions[0].reason == "invalid_vector_tenant_id"
    partial = _plan(
        db,
        [
            VectorObservation(_point_id("scoped"), DOC_A, TENANT_A),
            VectorObservation(_point_id("unscoped"), DOC_A, None),
        ],
        [_evidence(DOC_A, TENANT_A)],
    )
    assert partial.decisions[0].classification == "AMBIGUOUS_OWNER"
    assert partial.decisions[0].reason == "partially_scoped_vectors"
    assert not partial.registry_assignments and not partial.vector_assignments


def test_existing_scopes_are_preserved_and_no_cross_tenant_plan_is_created(tmp_path):
    db = _database(
        tmp_path / "inventory.sqlite3",
        [_row(DOC_A, tenant_id=TENANT_A), _row(DOC_B, tenant_id=TENANT_B)],
    )
    client, collection, points = _qdrant_points(
        (_point_id("a"), DOC_A, TENANT_A), (_point_id("b"), DOC_B, TENANT_B)
    )
    try:
        plan = _plan(db, points)
        assert {item.document_id: item.classification for item in plan.decisions} == {
            DOC_A: "ALREADY_SCOPED",
            DOC_B: "ALREADY_SCOPED",
        }
        assert not plan.registry_assignments and not plan.vector_assignments
        for tenant, expected in ((TENANT_A, DOC_A), (TENANT_B, DOC_B)):
            found, _ = client.scroll(
                collection,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=tenant)
                        )
                    ]
                ),
                with_payload=True,
                with_vectors=False,
            )
            assert len(found) == 1 and found[0].payload["doc_id"] == expected
    finally:
        client.close()


def test_repeated_plans_are_deterministic_and_source_is_unchanged(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    try:
        first = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        second = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        assert first == second
        assert first.plan_sha256 == second.plan_sha256
        assert hashlib.sha256(db.read_bytes()).hexdigest() == before
        assert not Path(f"{db}-wal").exists()
        assert not Path(f"{db}-shm").exists()
    finally:
        client.close()


def test_registry_change_during_inventory_fails_closed(tmp_path, monkeypatch):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    original_read_bytes = Path.read_bytes
    reads = 0
    changed_source_digest = None

    def change_registry_before_verification(path):
        nonlocal reads, changed_source_digest
        if path == db:
            reads += 1
            if reads == 2:
                with sqlite3.connect(db) as connection:
                    connection.execute("UPDATE documents SET record=record || ' '")
                changed_source_digest = hashlib.sha256(original_read_bytes(db)).hexdigest()
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", change_registry_before_verification)
    with pytest.raises(MigrationPlanningError, match="Registry changed during"):
        _plan(db, [], [_evidence(DOC_A, TENANT_A)])

    assert reads == 2
    assert changed_source_digest is not None
    assert hashlib.sha256(original_read_bytes(db)).hexdigest() == changed_source_digest


def test_interrupted_inventory_fails_without_returning_partial_plan(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    before = hashlib.sha256(db.read_bytes()).hexdigest()

    def interrupted():
        yield VectorObservation(_point_id("first"), DOC_A, None)
        raise OSError("synthetic interrupted inventory")

    with pytest.raises(MigrationPlanningError, match="inventory is incomplete"):
        _plan(db, interrupted(), [_evidence(DOC_A, TENANT_A)])
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_missing_or_incomplete_vector_inventory_fails_closed(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    with pytest.raises(MigrationPlanningError, match="complete offline vector inventory"):
        plan_legacy_tenant_migration(
            db,
            document_root=db.parent / "documents",
            vector_points=[],
            vector_inventory_complete=False,
            ownership_evidence=[_evidence(DOC_A, TENANT_A)],
        )


def test_wal_registry_is_rejected_without_modification(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    wal = Path(f"{db}-wal")
    wal.write_bytes(b"synthetic uncheckpointed WAL marker")
    before = hashlib.sha256(wal.read_bytes()).hexdigest()
    with pytest.raises(MigrationPlanningError, match="active WAL"):
        _plan(db, [], [_evidence(DOC_A, TENANT_A)])
    assert hashlib.sha256(wal.read_bytes()).hexdigest() == before


def test_shared_storage_path_is_quarantined_and_missing_document_file_is_not_assigned(
    tmp_path,
):
    db = _database(
        tmp_path / "inventory.sqlite3",
        [_row(DOC_A, filename="shared.pdf"), _row(DOC_B, filename="shared.pdf")],
    )
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None), (_point_id("b"), DOC_B, None))
    try:
        collision = _plan(
            db,
            points,
            [_evidence(DOC_A, TENANT_A), _evidence(DOC_B, TENANT_B)],
        )
        assert not collision.registry_assignments
        assert all(item.reason == "document_file_path_collision" for item in collision.decisions)

        (db.parent / "documents" / "shared.pdf").unlink()
        missing = _plan(db, points, [_evidence(DOC_A, TENANT_A), _evidence(DOC_B, TENANT_B)])
        assert not missing.registry_assignments
        assert all(item.reason == "missing_or_unsafe_document_file" for item in missing.decisions)
    finally:
        client.close()


def test_document_content_change_changes_inventory_digest(tmp_path):
    db = _database(tmp_path / "inventory.sqlite3", [_row(DOC_A)])
    client, _, points = _qdrant_points((_point_id("a"), DOC_A, None))
    try:
        first = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        source = db.parent / "documents" / "legacy.pdf"
        source.write_bytes(b"changed synthetic content")
        second = _plan(db, points, [_evidence(DOC_A, TENANT_A)])
        assert first.document_inventory_sha256 != second.document_inventory_sha256
        assert first.plan_sha256 != second.plan_sha256
    finally:
        client.close()
