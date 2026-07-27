"""C-private single-writer development audit ledger."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.helix.scientific_attribution import (
    GuardEvidenceObservationV1,
    GuardEvidenceSnapshotV1,
)


class GuardLedgerError(RuntimeError):
    pass


class EvidenceGuardLedgerWriterV1:
    namespace = "DEVELOPMENT_ONLY/EVIDENCE_AUDIT"

    def __init__(self, private_root: Path) -> None:
        self.private_root = Path(private_root).resolve()
        self.private_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.private_root / "evidence_guard.sqlite3"
        self._connection = sqlite3.connect(self.db_path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_calls (
                guard_call_id TEXT PRIMARY KEY,
                phase TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                request_digest TEXT NOT NULL,
                request_json TEXT NOT NULL,
                full_event_digest TEXT NOT NULL,
                full_event_json TEXT NOT NULL
            )
            """
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_evidence_observations (
                candidate_semantic_digest TEXT NOT NULL,
                protocol_digest TEXT NOT NULL,
                comparator_identity TEXT NOT NULL,
                observation_seed TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                raw_result_digest TEXT NOT NULL,
                raw_result_json TEXT NOT NULL,
                PRIMARY KEY(
                    candidate_semantic_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed
                )
            )
            """
        )
        self._connection.commit()

    def commit_create_once(
        self,
        *,
        guard_call_id: str,
        phase: str,
        candidate_id: str,
        request: Mapping[str, Any],
        full_event: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        request_digest = sha256_digest(request)
        event_digest = sha256_digest(full_event)
        request_json = json.dumps(request, sort_keys=True, separators=(",", ":"))
        event_json = json.dumps(full_event, sort_keys=True, separators=(",", ":"))
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                "SELECT request_digest, full_event_json FROM guard_calls WHERE guard_call_id=?",
                (guard_call_id,),
            ).fetchone()
            if row is not None:
                if row[0] != request_digest:
                    raise GuardLedgerError("guard_call_id request digest mismatch")
                self._connection.commit()
                return json.loads(row[1]), False
            self._connection.execute(
                """
                INSERT INTO guard_calls(
                    guard_call_id, phase, candidate_id, request_digest,
                    request_json, full_event_digest, full_event_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    guard_call_id,
                    phase,
                    candidate_id,
                    request_digest,
                    request_json,
                    event_digest,
                    event_json,
                ),
            )
            self._connection.commit()
            return dict(full_event), True
        except Exception:
            self._connection.rollback()
            raise

    def event_for(self, guard_call_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT full_event_json FROM guard_calls WHERE guard_call_id=?",
            (guard_call_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def request_and_event_for_candidate(
        self, *, phase: str, candidate_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        row = self._connection.execute(
            """
            SELECT request_json, full_event_json
            FROM guard_calls
            WHERE phase=? AND candidate_id=?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            (phase, candidate_id),
        ).fetchone()
        return (json.loads(row[0]), json.loads(row[1])) if row else None

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM guard_calls").fetchone()[0])

    def record_evidence_observation(
        self,
        *,
        candidate_semantic_digest: str,
        protocol_digest: str,
        comparator_identity: str,
        observation_seed: str,
        observation_id: str,
        raw_result: Mapping[str, Any],
    ) -> bool:
        raw_result_digest = sha256_digest(raw_result)
        raw_result_json = json.dumps(
            raw_result, sort_keys=True, separators=(",", ":")
        )
        key = (
            candidate_semantic_digest,
            protocol_digest,
            comparator_identity,
            observation_seed,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT observation_id, raw_result_digest
                FROM guard_evidence_observations
                WHERE candidate_semantic_digest=?
                  AND protocol_digest=?
                  AND comparator_identity=?
                  AND observation_seed=?
                """,
                key,
            ).fetchone()
            if row is not None:
                if (
                    str(row[0]) != observation_id
                    or str(row[1]) != raw_result_digest
                ):
                    raise GuardLedgerError(
                        "exact evidence key identity substitution"
                    )
                self._connection.commit()
                return False
            self._connection.execute(
                """
                INSERT INTO guard_evidence_observations(
                    candidate_semantic_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id,
                    raw_result_digest,
                    raw_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                key
                + (
                    observation_id,
                    raw_result_digest,
                    raw_result_json,
                ),
            )
            self._connection.commit()
            return True
        except Exception:
            self._connection.rollback()
            raise

    def evidence_snapshot(
        self,
        *,
        candidate_semantic_digest: str | None = None,
        protocol_digest: str | None = None,
        comparator_identity: str | None = None,
    ) -> GuardEvidenceSnapshotV1:
        identity = (
            candidate_semantic_digest,
            protocol_digest,
            comparator_identity,
        )
        if all(value is None for value in identity):
            rows = self._connection.execute(
                """
                SELECT
                    candidate_semantic_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id
                FROM guard_evidence_observations
                ORDER BY
                    candidate_semantic_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed
                """
            ).fetchall()
        elif all(value is not None for value in identity):
            rows = self._connection.execute(
                """
                SELECT
                    candidate_semantic_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id
                FROM guard_evidence_observations
                WHERE candidate_semantic_digest=?
                  AND protocol_digest=?
                  AND comparator_identity=?
                ORDER BY observation_seed
                """,
                identity,
            ).fetchall()
        else:
            raise ValueError("evidence snapshot requires one complete exact identity")
        return GuardEvidenceSnapshotV1(
            tuple(
                GuardEvidenceObservationV1(
                    candidate_semantic_digest=str(row[0]),
                    protocol_digest=str(row[1]),
                    comparator_identity=str(row[2]),
                    observation_seed=str(row[3]),
                    observation_id=str(row[4]),
                )
                for row in rows
            )
        )

    def raw_results_for_identity(
        self,
        *,
        candidate_semantic_digest: str,
        protocol_digest: str,
        comparator_identity: str,
    ) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            """
            SELECT raw_result_json
            FROM guard_evidence_observations
            WHERE candidate_semantic_digest=?
              AND protocol_digest=?
              AND comparator_identity=?
            ORDER BY observation_seed
            """,
            (
                candidate_semantic_digest,
                protocol_digest,
                comparator_identity,
            ),
        ).fetchall()
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = json.loads(row[0])
            unique[sha256_digest(payload)] = payload
        return tuple(unique[key] for key in sorted(unique))

    def close(self) -> None:
        self._connection.close()
