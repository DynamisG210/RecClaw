"""Immutable SQLite snapshots and association-free neutral audit projections."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

from .canonical import canonical_json_bytes, sha256_digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sidecars(path: Path) -> tuple[str, ...]:
    candidates = (
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    )
    return tuple(item.name for item in candidates if item.exists())


@dataclass(frozen=True, slots=True)
class ImmutableAuditSnapshotManifestV1:
    source_database_identity: str
    source_schema_identity: str
    snapshot_sha256: str
    snapshot_size_bytes: int
    sqlite_schema_version: int
    sqlite_user_version: int
    created_at: str
    audit_purpose: str
    creation_method: str
    external_wal_required: bool
    manifest_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def create_immutable_audit_snapshot(
    *,
    writer_connection: sqlite3.Connection,
    source_db_path: Path,
    snapshot_path: Path,
    source_schema_identity: str,
    audit_purpose: str,
) -> ImmutableAuditSnapshotManifestV1:
    """Create one writer-owned, WAL-independent SQLite backup."""

    source_db_path = source_db_path.resolve()
    snapshot_path = snapshot_path.resolve()
    if writer_connection.in_transaction:
        raise RuntimeError("audit snapshot requires no active writer transaction")
    if snapshot_path.exists():
        raise FileExistsError("immutable audit snapshot already exists")
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    writer_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    schema_version = int(
        writer_connection.execute("PRAGMA schema_version").fetchone()[0]
    )
    user_version = int(
        writer_connection.execute("PRAGMA user_version").fetchone()[0]
    )
    source_database_identity = sha256_digest(
        {
            "database_sha256": _file_sha256(source_db_path),
            "schema_identity": source_schema_identity,
            "schema_version": schema_version,
            "user_version": user_version,
        }
    )
    target = sqlite3.connect(snapshot_path)
    try:
        writer_connection.backup(target)
        target.execute("PRAGMA journal_mode=DELETE")
        target.execute("PRAGMA synchronous=FULL")
        integrity = str(target.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity != "ok":
            raise RuntimeError("immutable audit snapshot integrity check failed")
        target.commit()
    finally:
        target.close()
    with snapshot_path.open("rb") as handle:
        os.fsync(handle.fileno())
    if _sidecars(snapshot_path):
        raise RuntimeError("snapshot creation left SQLite sidecars")
    payload = {
        "audit_purpose": audit_purpose,
        "created_at": _utc_now(),
        "creation_method": "SQLITE_BACKUP_API_V1",
        "external_wal_required": False,
        "snapshot_sha256": _file_sha256(snapshot_path),
        "snapshot_size_bytes": snapshot_path.stat().st_size,
        "source_database_identity": source_database_identity,
        "source_schema_identity": source_schema_identity,
        "sqlite_schema_version": schema_version,
        "sqlite_user_version": user_version,
    }
    manifest = ImmutableAuditSnapshotManifestV1(
        **payload, manifest_digest=sha256_digest(payload)
    )
    manifest_path = snapshot_path.with_suffix(snapshot_path.suffix + ".manifest.json")
    with manifest_path.open("xb") as handle:
        handle.write(canonical_json_bytes(manifest.to_dict()) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def open_immutable_snapshot(path: Path) -> sqlite3.Connection:
    """Open a snapshot without permitting journal or sidecar creation."""

    path = path.resolve()
    before = _sidecars(path)
    uri = f"file:{quote(str(path))}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only=ON")
    after = _sidecars(path)
    if after != before:
        connection.close()
        raise RuntimeError("immutable audit created a SQLite sidecar")
    return connection


def _rows_as_counts(
    connection: sqlite3.Connection, query: str
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in connection.execute(query).fetchall()
    }


def association_free_neutral_audit(
    *,
    state_snapshot: Path,
    broker_snapshot: Path,
    guard_snapshot: Path,
) -> dict[str, Any]:
    """Return only aggregate failure/readiness counts without treatment joins."""

    state = open_immutable_snapshot(state_snapshot)
    broker = open_immutable_snapshot(broker_snapshot)
    guard = open_immutable_snapshot(guard_snapshot)
    try:
        projection: dict[str, Any] = {
            "broker_call_count_by_status": _rows_as_counts(
                broker,
                "SELECT status, COUNT(*) FROM calls GROUP BY status ORDER BY status",
            ),
            "broker_failure_count_by_class": _rows_as_counts(
                broker,
                "SELECT COALESCE(error_type, 'NONE'), COUNT(*) "
                "FROM calls GROUP BY COALESCE(error_type, 'NONE') "
                "ORDER BY COALESCE(error_type, 'NONE')",
            ),
            "broker_receipt_coverage_count": int(
                broker.execute(
                    "SELECT COUNT(*) FROM calls "
                    "WHERE exit_receipt_digest IS NOT NULL "
                    "AND length(exit_receipt_digest)=64"
                ).fetchone()[0]
            ),
            "execution_claim_count": int(
                state.execute("SELECT COUNT(*) FROM execution_claims").fetchone()[0]
            ),
            "guard_call_count": int(
                guard.execute("SELECT COUNT(*) FROM guard_calls").fetchone()[0]
            ),
            "resource_totals_by_dimension": _rows_as_counts(
                state,
                "SELECT dimension, COALESCE(SUM(quantity), 0) "
                "FROM resource_ledger GROUP BY dimension ORDER BY dimension",
            ),
            "round_count_by_status": _rows_as_counts(
                state,
                "SELECT status, COUNT(*) FROM rounds GROUP BY status ORDER BY status",
            ),
            "round_count_by_terminal_class": _rows_as_counts(
                state,
                "SELECT COALESCE(terminal_class, 'NONE'), COUNT(*) "
                "FROM rounds GROUP BY COALESCE(terminal_class, 'NONE') "
                "ORDER BY COALESCE(terminal_class, 'NONE')",
            ),
            "scheduled_slot_count_by_status": _rows_as_counts(
                state,
                "SELECT slot_status, COUNT(*) FROM scheduled_slots "
                "GROUP BY slot_status ORDER BY slot_status",
            ),
        }
    finally:
        state.close()
        broker.close()
        guard.close()
    encoded = json.dumps(projection, sort_keys=True, separators=(",", ":")).lower()
    forbidden = (
        "arm_code",
        "arm_instance",
        "assignment_key",
        "candidate_id",
        "metric",
        "ndcg",
        "treatment",
    )
    if any(token in encoded for token in forbidden):
        raise RuntimeError("neutral audit projection exposes a forbidden association")
    projection["audit_projection_digest"] = sha256_digest(projection)
    return projection


def verify_immutable_snapshot(
    path: Path, manifest: ImmutableAuditSnapshotManifestV1
) -> dict[str, Any]:
    before = _sidecars(path)
    connection = open_immutable_snapshot(path)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()
    after = _sidecars(path)
    return {
        "immutable_open": True,
        "integrity_check": integrity,
        "manifest_digest": manifest.manifest_digest,
        "sha256_match": _file_sha256(path) == manifest.snapshot_sha256,
        "sidecars_before": list(before),
        "sidecars_after": list(after),
        "sidecars_created": len(set(after) - set(before)),
    }


__all__ = [
    "ImmutableAuditSnapshotManifestV1",
    "association_free_neutral_audit",
    "create_immutable_audit_snapshot",
    "open_immutable_snapshot",
    "verify_immutable_snapshot",
]
