# Legacy tenant migration dry run

## Purpose and boundary

`rag_agent.api.tenant_migration.plan_legacy_tenant_migration` creates a deterministic, read-only proposal from a SQLite registry, a complete offline vector inventory, and ownership evidence already verified by a trusted server-side source. It does not connect to Qdrant, update SQLite, change payloads, move files, or execute its returned assignments. There is no production evidence adapter and no migration executor in this phase.

The planner accepts no ownership inference from usernames, filenames, document text, shared collection names, upload ordering, or the first/admin account. `OwnershipEvidence` is a contract for a verified source record and an opaque audit reference. The type does not authenticate the provider; a production adapter must independently establish provenance. Synthetic tests use fabricated evidence only to exercise the decision logic.

The registry is opened read-only and immutable. A non-empty SQLite WAL causes the planner to fail closed; quiesce and checkpoint a disposable database before planning. The caller must explicitly attest `vector_inventory_complete=True`. That flag is a completeness assertion, not proof that a remote Qdrant scroll was complete. This planner is not a substitute for a remote snapshot or a consistent database/vector snapshot. It hashes managed document contents into the inventory digest and rejects missing files, symlinks, path escapes, shared storage-path collisions, and files that change while being scanned. Proposed assignments retain only a SHA-256 of the opaque evidence reference, not the supplied reference itself.

## Ownership classifications

| Classification | Meaning | Planner behavior |
|---|---|---|
| `VERIFIED_OWNER` | A legacy SQLite row has one validated, verified server-side owner record and matching complete vector inventory. | Emits proposed registry/vector assignments and rollback-to-NULL steps. It still does not execute them. |
| `AMBIGUOUS_OWNER` | Conflicting/invalid evidence, duplicate registry IDs or point IDs, inconsistent scopes, malformed metadata, non-quiescent state, or vector count mismatch. | Quarantines; emits no assignment for that document. |
| `UNOWNED` | No verified server-side owner evidence, or a deleted/inactive legacy record. | Quarantines; emits no assignment. |
| `ALREADY_SCOPED` | Registry and all observed document vectors have the same valid tenant scope. | Preserves the scope and emits no assignment. |

Orphan vectors, vectors without a valid document ID, and evidence without a registry row are separately listed for quarantine. Unscoped vectors remain inaccessible to tenant filters until a separately approved migration is completed.

## Offline test data and output

Tests create temporary SQLite registries and use `QdrantClient(":memory:")` with tiny deterministic two-dimensional vectors. Inputs deliberately include shared filenames, conflicting owners, missing owners, duplicate rows/point IDs, unscoped points, orphan points, and existing tenant scopes. The planner hashes the registry and normalized vector inventory into a repeatable plan ID, records proposed changes and rollback-to-original-scope steps, and always sets `automatic_reassignment_allowed` to false.

The SQLite fixture is hashed before and after each relevant dry run. No source rows are changed. The tests do not demonstrate remote Qdrant scroll completeness, snapshots, or server-side evidence provenance.

## Future real migration procedure (not authorized or implemented)

1. **Preconditions:** obtain explicit migration approval; identify exact registry, collection, and managed-file roots; stop API writers and background tasks; establish a maintenance window; pin compatible Qdrant client/server versions; confirm remote TLS and API-key authentication.
2. **Immutable backups:** capture a transaction-consistent SQLite backup and managed-file archive, plus a Qdrant snapshot from the same quiescent window. Hash and independently verify each artifact. Keep the original database, collection, files, and backup keys untouched until acceptance.
3. **Authoritative ownership:** obtain per-document owner evidence from an auditable server-side source. Require two-person review for mappings. Quarantine absent, ambiguous, conflicting, or unverifiable records. Never infer ownership from names, filenames, document contents, or collection membership.
4. **Dry-run review:** execute a read-only inventory against disposable copies. Review all `VERIFIED_OWNER` proposals, quarantines, duplicates, orphans, missing records/files/vectors, counts, payload scopes, source hashes, and rollback actions. Require zero unexplained discrepancies and explicit approval of the exact plan digest.
5. **Staged migration:** restore/copy to a new isolated target or shadow collection. Apply tenant payloads only to points whose ownership is verified. Do not overwrite the source collection or reuse its storage. Preserve all existing scopes and abort on any mismatch.
6. **Reconciliation:** compare per-tenant SQLite records, files, document IDs, chunk IDs, vector IDs, payloads, and counts. Run cross-tenant list/search/delete-denial tests and restore tests. Keep unowned records quarantined and invisible.
7. **Authorization and cutover:** independently review the plan and reconciliation evidence, then require explicit operator approval before changing the target application binding. Verify authenticated tenant filters and session-derived identity before enabling access.
8. **Rollback:** if any count, scope, checksum, snapshot version, or authorization check differs, keep the target offline and switch back to the preserved original configuration/collection. Never perform guessed in-place reversal. Retain an append-only audit record of plan hash, approvals, actions, and verification results.
9. **Final audit:** retain backups and quarantines according to an approved retention policy; record all residual ambiguity. No real migration should proceed until remote Qdrant snapshot/restore and end-to-end recovery have been validated on disposable infrastructure.

## Current readiness

Only the offline planner and synthetic tests are implemented. This dry run does not establish that any actual legacy document has a verified owner, and it cannot authorize or perform a real migration. Remote Qdrant behavior and complete recovery remain blocked.
