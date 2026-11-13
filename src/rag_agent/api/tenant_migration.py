"""Offline, non-mutating planner for legacy tenant ownership migration.

This module deliberately does not connect to Qdrant or apply its plan. Callers
must supply a complete vector inventory and explicitly verified server-side
ownership evidence. Missing, ambiguous, inconsistent, or orphaned data is
quarantined in the plan rather than assigned by heuristic.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from rag_agent.tenancy import validate_tenant_id

_DOCUMENT_ID = re.compile(r"doc_[0-9a-f]{64}\Z")
_PROCESSING_STATES = {"queued", "processing", "indexing", "deleting", "cleanup_required"}
_KNOWN_STATES = _PROCESSING_STATES | {"ready", "failed", "deleted", "external"}
_ALLOWED_EVIDENCE_SOURCES = {"server_side_owner_record", "operator_verified_server_record"}


class MigrationPlanningError(RuntimeError):
    """Planning failed or input inventories were incomplete; no plan is returned."""


@dataclass(frozen=True, slots=True)
class OwnershipEvidence:
    """An ownership claim already verified by a trusted server-side provider.

    ``evidence_reference`` is an opaque audit-record identifier, never document
    text, a username, or a secret. This data type does not authenticate its
    provider; production adapters must establish provenance before constructing
    it. No such production adapter or migration executor is provided here.
    """

    document_id: str
    tenant_id: str
    evidence_reference: str
    evidence_source: str
    verified: bool = True


@dataclass(frozen=True, slots=True)
class VectorObservation:
    """Minimal point metadata copied from a disposable/offline inventory."""

    point_id: str
    document_id: str | None
    tenant_id: str | None


@dataclass(frozen=True, slots=True)
class RecordDecision:
    document_id: str
    classification: str
    current_tenant_id: str | None
    proposed_tenant_id: str | None
    reason: str
    quarantined: bool


@dataclass(frozen=True, slots=True)
class RegistryAssignment:
    document_id: str
    tenant_id: str
    evidence_reference_sha256: str
    rollback_to: None = None


@dataclass(frozen=True, slots=True)
class VectorAssignment:
    point_id: str
    document_id: str
    tenant_id: str
    rollback_to: None = None


@dataclass(frozen=True, slots=True)
class QuarantinedPoint:
    point_id: str
    document_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class QuarantinedEvidence:
    document_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    source_registry_sha256: str
    document_inventory_sha256: str
    vector_inventory_sha256: str
    registry_row_count: int
    vector_point_count: int
    decisions: tuple[RecordDecision, ...]
    registry_assignments: tuple[RegistryAssignment, ...]
    vector_assignments: tuple[VectorAssignment, ...]
    quarantined_points: tuple[QuarantinedPoint, ...]
    quarantined_evidence: tuple[QuarantinedEvidence, ...]
    rollback_steps: tuple[str, ...]
    automatic_reassignment_allowed: bool
    plan_sha256: str


def _read_registry(database: Path) -> tuple[list[tuple], str]:
    if not database.is_absolute() or database.is_symlink() or not database.is_file():
        raise MigrationPlanningError("Registry path is missing or unsafe")
    wal = Path(f"{database}-wal")
    shm = Path(f"{database}-shm")
    if wal.is_symlink() or shm.is_symlink():
        raise MigrationPlanningError("Registry sidecar path is unsafe")
    # The dry run refuses a live/uncheckpointed WAL rather than risking a
    # misleading snapshot or causing SQLite to create sidecar files.
    if wal.exists() and wal.stat().st_size:
        raise MigrationPlanningError(
            "Registry has an active WAL; quiesce/checkpoint before planning"
        )
    digest = hashlib.sha256(database.read_bytes()).hexdigest()
    uri = database.resolve(strict=True).as_uri() + "?mode=ro&immutable=1"
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=5)
        connection.execute("PRAGMA query_only=ON")
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if "documents" not in tables:
            raise MigrationPlanningError("Registry documents table is missing")
        columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        required = {"id", "record", "file"}
        if not required.issubset(columns) or columns - (required | {"tenant_id"}):
            raise MigrationPlanningError("Registry documents schema is unsupported")
        tenant_expr = "tenant_id" if "tenant_id" in columns else "NULL"
        rows = connection.execute(
            f"SELECT id, record, file, {tenant_expr} FROM documents ORDER BY id"
        ).fetchall()
    except MigrationPlanningError:
        raise
    except (OSError, sqlite3.Error, ValueError) as exc:
        raise MigrationPlanningError("Could not read a stable registry snapshot") from exc
    finally:
        if connection is not None:
            connection.close()
    if hashlib.sha256(database.read_bytes()).hexdigest() != digest:
        raise MigrationPlanningError("Registry changed during dry-run planning")
    if wal.exists() and wal.stat().st_size:
        raise MigrationPlanningError("Registry WAL changed during dry-run planning")
    return rows, digest


def _read_document_inventory(
    root: Path, rows: list[tuple]
) -> tuple[dict[str, list[Path | None]], str, set[Path]]:
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise MigrationPlanningError("Managed document directory is missing or unsafe")
    absolute = Path(root.anchor)
    for part in root.parts[1:]:
        absolute = absolute / part
        if absolute.is_symlink():
            raise MigrationPlanningError("Managed document directory contains a symlink")
    by_document: dict[str, list[Path | None]] = defaultdict(list)
    paths: list[tuple[str, int, str]] = []
    owners_by_path: dict[Path, set[str]] = defaultdict(set)
    for doc_id, record_json, filename, tenant_id in rows:
        if not isinstance(doc_id, str) or _DOCUMENT_ID.fullmatch(doc_id) is None:
            continue
        try:
            record = json.loads(record_json)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("status") == "deleted":
            continue
        if not isinstance(filename, str) or not filename or Path(filename).name != filename:
            by_document[doc_id].append(None)
            continue
        if tenant_id is None:
            path = root / filename
        else:
            try:
                tenant_id = validate_tenant_id(tenant_id)
            except (ValueError, TypeError):
                by_document[doc_id].append(None)
                continue
            path = root / "tenants" / tenant_id / filename
        by_document[doc_id].append(path)
        owners_by_path[path].add(doc_id)
        current = root
        for part in path.relative_to(root).parts:
            current = current / part
            if current.is_symlink():
                raise MigrationPlanningError("Managed document path contains a symlink")
        if path.is_symlink() or not path.is_file():
            continue
        resolved_root = root.resolve(strict=True)
        resolved_path = path.resolve(strict=True)
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise MigrationPlanningError("Managed document path escapes its root") from exc
        before = path.stat()
        digest = hashlib.sha256()
        try:
            with path.open("rb") as document:
                for block in iter(lambda: document.read(1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise MigrationPlanningError(
                "Could not read a stable managed document inventory"
            ) from exc
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise MigrationPlanningError("Managed document changed during dry-run planning")
        paths.append(
            (
                path.relative_to(root).as_posix(),
                after.st_size,
                digest.hexdigest(),
            )
        )
    collisions = {path for path, owners in owners_by_path.items() if len(owners) > 1}
    digest = hashlib.sha256(_canonical_plan_payload(sorted(paths))).hexdigest()
    return by_document, digest, collisions


def _normalize_vectors(points: Iterable[VectorObservation]) -> list[VectorObservation]:
    try:
        values = list(points)
    except Exception as exc:  # noqa: BLE001 - incomplete iterators must fail closed
        raise MigrationPlanningError("Vector inventory is incomplete") from exc
    normalized: list[VectorObservation] = []
    for point in values:
        if not isinstance(point, VectorObservation):
            raise MigrationPlanningError("Vector inventory has an unsupported record")
        if not isinstance(point.point_id, str) or not point.point_id or len(point.point_id) > 128:
            raise MigrationPlanningError("Vector inventory contains an invalid point identifier")
        normalized.append(point)
    return sorted(normalized, key=lambda item: (item.point_id, item.document_id or ""))


def _normalize_evidence(
    evidence: Iterable[OwnershipEvidence],
) -> dict[str, list[OwnershipEvidence]]:
    try:
        values = list(evidence)
    except Exception as exc:  # noqa: BLE001 - do not emit a partial plan
        raise MigrationPlanningError("Ownership evidence inventory is incomplete") from exc
    grouped: dict[str, list[OwnershipEvidence]] = defaultdict(list)
    for item in values:
        if not isinstance(item, OwnershipEvidence):
            raise MigrationPlanningError("Ownership evidence has an unsupported record")
        grouped[item.document_id].append(item)
    return grouped


def _canonical_plan_payload(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def plan_legacy_tenant_migration(
    database: Path,
    *,
    document_root: Path,
    vector_points: Iterable[VectorObservation],
    vector_inventory_complete: bool,
    ownership_evidence: Iterable[OwnershipEvidence],
) -> MigrationPlan:
    """Build a deterministic proposal without writing to SQLite or Qdrant.

    The caller supplies a complete offline vector inventory. Evidence is
    accepted only when it explicitly states that a trusted server-side source
    was verified; filename, username, document content, and collection name
    are never consulted as ownership signals. The returned plan is not an
    executor authorization: ``automatic_reassignment_allowed`` is always false.
    """
    if vector_inventory_complete is not True:
        raise MigrationPlanningError("A complete offline vector inventory is required")
    rows, registry_digest = _read_registry(database)
    document_paths, document_digest, colliding_paths = _read_document_inventory(document_root, rows)
    points = _normalize_vectors(vector_points)
    evidence = _normalize_evidence(ownership_evidence)
    vector_digest = hashlib.sha256(
        _canonical_plan_payload([asdict(point) for point in points])
    ).hexdigest()
    row_groups: dict[str, list[tuple]] = defaultdict(list)
    invalid_rows: list[tuple] = []
    for row in rows:
        doc_id = row[0]
        if not isinstance(doc_id, str) or _DOCUMENT_ID.fullmatch(doc_id) is None:
            invalid_rows.append(row)
        else:
            row_groups[doc_id].append(row)
    point_id_counts = Counter(point.point_id for point in points)
    point_groups: dict[str, list[VectorObservation]] = defaultdict(list)
    quarantined_points: list[QuarantinedPoint] = []
    quarantined_evidence: list[QuarantinedEvidence] = []
    bad_point_documents: set[str] = set()
    for point in points:
        if point.document_id is None or _DOCUMENT_ID.fullmatch(point.document_id) is None:
            quarantined_points.append(
                QuarantinedPoint(
                    point.point_id, point.document_id, "missing_or_invalid_document_id"
                )
            )
        elif point.document_id not in row_groups:
            quarantined_points.append(
                QuarantinedPoint(point.point_id, point.document_id, "orphan_vector")
            )
        elif point_id_counts[point.point_id] > 1:
            quarantined_points.append(
                QuarantinedPoint(point.point_id, point.document_id, "duplicate_point_id")
            )
            bad_point_documents.add(point.document_id)
        else:
            point_groups[point.document_id].append(point)

    decisions: list[RecordDecision] = []
    registry_assignments: list[RegistryAssignment] = []
    vector_assignments: list[VectorAssignment] = []
    for doc_id in sorted(row_groups):
        grouped_rows = row_groups[doc_id]
        current_values = {row[3] for row in grouped_rows}
        decision: RecordDecision
        if len(grouped_rows) != 1 or len(current_values) != 1:
            decision = RecordDecision(
                doc_id,
                "AMBIGUOUS_OWNER",
                None,
                None,
                "duplicate_or_conflicting_registry_rows",
                True,
            )
            decisions.append(decision)
            continue
        if doc_id in bad_point_documents:
            decisions.append(
                RecordDecision(
                    doc_id, "AMBIGUOUS_OWNER", None, None, "duplicate_qdrant_point_id", True
                )
            )
            continue
        _, record_json, filename, current_tenant = grouped_rows[0]
        try:
            record = json.loads(record_json)
            if not isinstance(record, dict) or record.get("doc_id") != doc_id:
                raise ValueError("registry record identifier mismatch")
            if not isinstance(filename, str) or not filename or Path(filename).name != filename:
                raise ValueError("unsafe registry filename")
        except (ValueError, TypeError, json.JSONDecodeError):
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "invalid_registry_metadata",
                    True,
                )
            )
            continue
        if record.get("status") not in _KNOWN_STATES:
            decisions.append(
                RecordDecision(
                    doc_id, "AMBIGUOUS_OWNER", current_tenant, None, "unknown_document_state", True
                )
            )
            continue
        if current_tenant is not None:
            try:
                current_tenant = validate_tenant_id(current_tenant)
            except ValueError:
                decisions.append(
                    RecordDecision(
                        doc_id, "AMBIGUOUS_OWNER", None, None, "invalid_existing_tenant_id", True
                    )
                )
                continue
        if record.get("status") in _PROCESSING_STATES:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "non_quiescent_document_state",
                    True,
                )
            )
            continue
        if record.get("status") == "deleted":
            decisions.append(
                RecordDecision(
                    doc_id,
                    "UNOWNED",
                    current_tenant,
                    None,
                    "inactive_deleted_record_not_migrated",
                    True,
                )
            )
            continue
        registered_files = document_paths.get(doc_id, [])
        if (
            len(registered_files) != 1
            or registered_files[0] is None
            or registered_files[0].is_symlink()
            or not registered_files[0].is_file()
        ):
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "missing_or_unsafe_document_file",
                    True,
                )
            )
            continue
        if registered_files[0] in colliding_paths:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "document_file_path_collision",
                    True,
                )
            )
            continue

        doc_points = point_groups.get(doc_id, [])
        raw_evidence = evidence.get(doc_id, [])
        evidence_tenants: set[str] = set()
        evidence_refs: set[str] = set()
        evidence_invalid = False
        for item in raw_evidence:
            try:
                if (
                    not item.verified
                    or item.evidence_source not in _ALLOWED_EVIDENCE_SOURCES
                    or not isinstance(item.evidence_reference, str)
                    or not item.evidence_reference.strip()
                    or len(item.evidence_reference) > 512
                ):
                    evidence_invalid = True
                    continue
                evidence_tenants.add(validate_tenant_id(item.tenant_id))
                evidence_refs.add(item.evidence_reference.strip())
            except (ValueError, TypeError):
                evidence_invalid = True
        point_tenants: set[str] = set()
        point_invalid = False
        for point in doc_points:
            if point.tenant_id is None:
                continue
            try:
                point_tenants.add(validate_tenant_id(point.tenant_id))
            except ValueError:
                point_invalid = True
        expected_chunks = record.get("chunk_count", 0)
        if (
            not isinstance(expected_chunks, int)
            or isinstance(expected_chunks, bool)
            or expected_chunks < 0
        ):
            decisions.append(
                RecordDecision(
                    doc_id, "AMBIGUOUS_OWNER", current_tenant, None, "invalid_chunk_count", True
                )
            )
            continue
        if record.get("status") == "ready" and len(doc_points) != expected_chunks:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "missing_or_extra_qdrant_points",
                    True,
                )
            )
            continue
        if point_invalid:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "invalid_vector_tenant_id",
                    True,
                )
            )
            continue
        if len(point_tenants) > 1:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "conflicting_vector_tenant_scopes",
                    True,
                )
            )
            continue
        if evidence_invalid or len(evidence_tenants) > 1:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    current_tenant,
                    None,
                    "unverified_or_conflicting_ownership_evidence",
                    True,
                )
            )
            continue

        if current_tenant is not None:
            if evidence_tenants and evidence_tenants != {current_tenant}:
                decisions.append(
                    RecordDecision(
                        doc_id,
                        "AMBIGUOUS_OWNER",
                        current_tenant,
                        None,
                        "evidence_conflicts_with_existing_scope",
                        True,
                    )
                )
            elif point_tenants and point_tenants != {current_tenant}:
                decisions.append(
                    RecordDecision(
                        doc_id,
                        "AMBIGUOUS_OWNER",
                        current_tenant,
                        None,
                        "vectors_conflict_with_existing_scope",
                        True,
                    )
                )
            elif any(point.tenant_id is None for point in doc_points):
                decisions.append(
                    RecordDecision(
                        doc_id,
                        "AMBIGUOUS_OWNER",
                        current_tenant,
                        None,
                        "partially_scoped_vectors",
                        True,
                    )
                )
            elif record.get("status") == "ready" and not doc_points:
                decisions.append(
                    RecordDecision(
                        doc_id,
                        "AMBIGUOUS_OWNER",
                        current_tenant,
                        None,
                        "ready_record_has_no_vectors",
                        True,
                    )
                )
            else:
                decisions.append(
                    RecordDecision(
                        doc_id,
                        "ALREADY_SCOPED",
                        current_tenant,
                        None,
                        "existing_registry_and_vectors_consistent",
                        False,
                    )
                )
            continue

        if not evidence_tenants:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "UNOWNED",
                    None,
                    None,
                    "no_verified_server_side_ownership_evidence",
                    True,
                )
            )
            continue
        target_tenant = next(iter(evidence_tenants))
        if point_tenants and any(point.tenant_id is None for point in doc_points):
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    None,
                    None,
                    "partially_scoped_vectors",
                    True,
                )
            )
            continue
        if point_tenants and point_tenants != {target_tenant}:
            decisions.append(
                RecordDecision(
                    doc_id,
                    "AMBIGUOUS_OWNER",
                    None,
                    None,
                    "verified_owner_conflicts_with_vector_scope",
                    True,
                )
            )
            continue
        if record.get("status") == "ready" and not doc_points:
            decisions.append(
                RecordDecision(
                    doc_id, "AMBIGUOUS_OWNER", None, None, "ready_record_has_no_vectors", True
                )
            )
            continue
        evidence_ref = hashlib.sha256(sorted(evidence_refs)[0].encode("utf-8")).hexdigest()
        decisions.append(
            RecordDecision(
                doc_id,
                "VERIFIED_OWNER",
                None,
                target_tenant,
                "verified_server_side_evidence",
                False,
            )
        )
        registry_assignments.append(RegistryAssignment(doc_id, target_tenant, evidence_ref))
        vector_assignments.extend(
            VectorAssignment(point.point_id, doc_id, target_tenant)
            for point in doc_points
            if point.tenant_id is None
        )

    for row in invalid_rows:
        decisions.append(
            RecordDecision(
                str(row[0])[:128],
                "AMBIGUOUS_OWNER",
                None,
                None,
                "invalid_registry_document_id",
                True,
            )
        )
    known_ids = set(row_groups)
    for doc_id in sorted(set(evidence) - known_ids):
        # Evidence for absent rows cannot create ownership or authorize points.
        quarantined_evidence.append(
            QuarantinedEvidence(doc_id, "ownership_evidence_without_registry_row")
        )

    decisions.sort(key=lambda item: (item.document_id, item.classification, item.reason))
    registry_assignments.sort(key=lambda item: item.document_id)
    vector_assignments.sort(key=lambda item: (item.document_id, item.point_id))
    quarantined_points.sort(key=lambda item: (item.document_id or "", item.point_id, item.reason))
    quarantined_evidence.sort(key=lambda item: (item.document_id, item.reason))
    rollback_steps = tuple(
        [
            f"restore documents.tenant_id to NULL for {item.document_id}"
            for item in registry_assignments
        ]
        + [
            f"remove tenant_id assignment from Qdrant point {item.point_id}"
            for item in vector_assignments
        ]
    )
    body = {
        "source_registry_sha256": registry_digest,
        "document_inventory_sha256": document_digest,
        "vector_inventory_sha256": vector_digest,
        "registry_row_count": len(rows),
        "vector_point_count": len(points),
        "decisions": [asdict(item) for item in decisions],
        "registry_assignments": [asdict(item) for item in registry_assignments],
        "vector_assignments": [asdict(item) for item in vector_assignments],
        "quarantined_points": [asdict(item) for item in quarantined_points],
        "quarantined_evidence": [asdict(item) for item in quarantined_evidence],
        "rollback_steps": rollback_steps,
        "automatic_reassignment_allowed": False,
    }
    plan_hash = hashlib.sha256(_canonical_plan_payload(body)).hexdigest()
    return MigrationPlan(
        source_registry_sha256=registry_digest,
        document_inventory_sha256=document_digest,
        vector_inventory_sha256=vector_digest,
        registry_row_count=len(rows),
        vector_point_count=len(points),
        decisions=tuple(decisions),
        registry_assignments=tuple(registry_assignments),
        vector_assignments=tuple(vector_assignments),
        quarantined_points=tuple(quarantined_points),
        quarantined_evidence=tuple(quarantined_evidence),
        rollback_steps=rollback_steps,
        automatic_reassignment_allowed=False,
        plan_sha256=plan_hash,
    )
