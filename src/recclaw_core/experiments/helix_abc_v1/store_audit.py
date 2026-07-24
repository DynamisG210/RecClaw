"""Canonical read-only integrity audit for both experiment-store variants."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .canonical import sha256_digest, validate_relative_artifact_path


STORE_AUDIT_CONTRACT_VERSION = "ExperimentStoreAuditPortV1"
STORE_AUDIT_REPORT_FIELDS = (
    "artifact_index_check",
    "execution_claim_uniqueness_check",
    "experiment_identity_digest",
    "feedback_uniqueness_check",
    "foreign_key_violation_count",
    "migration_digest",
    "required_table_set",
    "round_uniqueness_check",
    "schema_version",
    "sqlite_integrity",
    "triplet_barrier_check",
)


def store_audit_contract_digest() -> str:
    return sha256_digest(
        {
            "operation": "audit_store",
            "port": STORE_AUDIT_CONTRACT_VERSION,
            "read_only": True,
            "report_fields": list(STORE_AUDIT_REPORT_FIELDS),
        }
    )


@dataclass(frozen=True, slots=True)
class StoreIntegrityReportV1:
    sqlite_integrity: str
    foreign_key_violation_count: int
    migration_digest: str
    experiment_identity_digest: str
    schema_version: int
    required_table_set: tuple[str, ...]
    round_uniqueness_check: bool
    execution_claim_uniqueness_check: bool
    feedback_uniqueness_check: bool
    triplet_barrier_check: bool
    artifact_index_check: bool
    report_digest: str

    @classmethod
    def create(cls, payload: dict[str, Any]) -> "StoreIntegrityReportV1":
        if set(payload) != set(STORE_AUDIT_REPORT_FIELDS):
            raise ValueError("store audit payload does not match the closed contract")
        normalized = {
            **payload,
            "required_table_set": tuple(payload["required_table_set"]),
        }
        digest_payload = {
            **normalized,
            "required_table_set": list(normalized["required_table_set"]),
        }
        return cls(**normalized, report_digest=sha256_digest(digest_payload))

    @property
    def passed(self) -> bool:
        return (
            self.sqlite_integrity == "ok"
            and self.foreign_key_violation_count == 0
            and self.round_uniqueness_check
            and self.execution_claim_uniqueness_check
            and self.feedback_uniqueness_check
            and self.triplet_barrier_check
            and self.artifact_index_check
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_index_check": self.artifact_index_check,
            "execution_claim_uniqueness_check": (
                self.execution_claim_uniqueness_check
            ),
            "experiment_identity_digest": self.experiment_identity_digest,
            "feedback_uniqueness_check": self.feedback_uniqueness_check,
            "foreign_key_violation_count": self.foreign_key_violation_count,
            "migration_digest": self.migration_digest,
            "report_digest": self.report_digest,
            "required_table_set": list(self.required_table_set),
            "round_uniqueness_check": self.round_uniqueness_check,
            "schema_version": self.schema_version,
            "sqlite_integrity": self.sqlite_integrity,
            "triplet_barrier_check": self.triplet_barrier_check,
        }


@runtime_checkable
class ExperimentStoreAuditPortV1(Protocol):
    def audit_store(self) -> StoreIntegrityReportV1:
        """Return the canonical, read-only integrity result."""


def require_store_audit_port(value: object) -> ExperimentStoreAuditPortV1:
    if not isinstance(value, ExperimentStoreAuditPortV1):
        raise TypeError("experiment store does not provide ExperimentStoreAuditPortV1")
    return value


class ExperimentStoreAuditAdapterV1:
    """Typed adapter over the common private store connection contract."""

    def __init__(self, store: object) -> None:
        required = ("_connection", "_lock", "artifact_root", "migration_sha256")
        if any(not hasattr(store, name) for name in required):
            raise TypeError(
                "experiment store cannot delegate ExperimentStoreAuditPortV1"
            )
        self._store = store

    def audit_store(self) -> StoreIntegrityReportV1:
        store = self._store
        with store._lock:
            return audit_experiment_store(
                store._connection,
                artifact_root=Path(store.artifact_root),
                migration_digest=str(store.migration_sha256),
                required_tables=frozenset(
                    {
                        "arm_state",
                        "artifact_index",
                        "execution_claims",
                        "resource_ledger",
                        "round_events",
                        "rounds",
                        "scheduled_slots",
                        "triplet_barrier",
                    }
                ),
            )


def experiment_store_audit_port(store: object) -> ExperimentStoreAuditPortV1:
    if isinstance(store, ExperimentStoreAuditPortV1):
        return store
    return ExperimentStoreAuditAdapterV1(store)


def _has_duplicate(connection: sqlite3.Connection, query: str) -> bool:
    return connection.execute(query).fetchone() is not None


def _artifact_index_is_exact(
    connection: sqlite3.Connection, artifact_root: Path
) -> bool:
    root = artifact_root.resolve()
    rows = connection.execute(
        "SELECT relative_path, size_bytes, sha256 FROM artifact_index"
    ).fetchall()
    for relative_path, size_bytes, expected_sha256 in rows:
        try:
            normalized = validate_relative_artifact_path(str(relative_path))
            path = artifact_root.joinpath(*normalized.split("/"))
            if path.is_symlink() or not path.is_file():
                return False
            resolved = path.resolve()
            resolved.relative_to(root)
            data = resolved.read_bytes()
        except (OSError, ValueError):
            return False
        if (
            len(data) != int(size_bytes)
            or hashlib.sha256(data).hexdigest() != str(expected_sha256)
        ):
            return False
    return True


def audit_experiment_store(
    connection: sqlite3.Connection,
    *,
    artifact_root: Path,
    migration_digest: str,
    required_tables: frozenset[str],
) -> StoreIntegrityReportV1:
    """Execute the one canonical set of store-integrity queries."""

    observed_tables = tuple(
        sorted(
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        )
    )
    identity_rows = [
        tuple(row)
        for row in connection.execute(
            "SELECT experiment_id, arm_instance_id, arm_code, search_seed, "
            "experiment_contract_digest FROM arm_state "
            "ORDER BY experiment_id, arm_instance_id, search_seed"
        )
    ]
    foreign_key_violations = list(
        connection.execute("PRAGMA foreign_key_check")
    )
    round_duplicates = _has_duplicate(
        connection,
        "SELECT 1 FROM rounds GROUP BY experiment_id, arm_instance_id, "
        "search_seed, round_index HAVING COUNT(*) > 1 LIMIT 1",
    )
    claim_duplicates = _has_duplicate(
        connection,
        "SELECT 1 FROM execution_claims GROUP BY round_id "
        "HAVING COUNT(*) > 1 LIMIT 1",
    ) or _has_duplicate(
        connection,
        "SELECT 1 FROM execution_claims GROUP BY claim_id "
        "HAVING COUNT(*) > 1 LIMIT 1",
    )
    feedback_duplicates = _has_duplicate(
        connection,
        "SELECT 1 FROM round_events WHERE event_type='ROUND_FEEDBACK' "
        "GROUP BY round_id HAVING COUNT(*) > 1 LIMIT 1",
    )
    invalid_barrier = _has_duplicate(
        connection,
        "SELECT 1 FROM triplet_barrier WHERE "
        "arm_a_instance_id=arm_b_instance_id "
        "OR arm_a_instance_id=arm_c_instance_id "
        "OR arm_b_instance_id=arm_c_instance_id "
        "OR closed_bitmap < 0 OR closed_bitmap > 7 "
        "OR (next_index_authorized=1 AND "
        "(closed_bitmap<>7 OR stop_requested<>0)) LIMIT 1",
    )
    return StoreIntegrityReportV1.create(
        {
            "artifact_index_check": (
                observed_tables == tuple(sorted(required_tables))
                and _artifact_index_is_exact(connection, artifact_root)
            ),
            "execution_claim_uniqueness_check": not claim_duplicates,
            "experiment_identity_digest": sha256_digest(identity_rows),
            "feedback_uniqueness_check": not feedback_duplicates,
            "foreign_key_violation_count": len(foreign_key_violations),
            "migration_digest": migration_digest,
            "required_table_set": observed_tables,
            "round_uniqueness_check": not round_duplicates,
            "schema_version": int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            ),
            "sqlite_integrity": str(
                connection.execute("PRAGMA integrity_check").fetchone()[0]
            ),
            "triplet_barrier_check": not invalid_barrier,
        }
    )


__all__ = [
    "ExperimentStoreAuditPortV1",
    "ExperimentStoreAuditAdapterV1",
    "StoreIntegrityReportV1",
    "audit_experiment_store",
    "experiment_store_audit_port",
    "require_store_audit_port",
    "store_audit_contract_digest",
]
