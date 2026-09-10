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
        self.db_path = self.private_root / "evidence_guard.sqlite3"
        self.__connection: sqlite3.Connection | None = None

    @property
    def _connection(self) -> sqlite3.Connection:
        """Open the ledger only when the first Guard operation is executed.

        Standalone Research requires a nonexistent run root when composing a
        new campaign.  Bridge construction is part of sealed config identity,
        so it must remain side-effect free until Research owns that root.
        """

        if self.__connection is not None:
            return self.__connection
        self.private_root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(
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
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_evidence_observations_v2 (
                candidate_semantic_digest TEXT NOT NULL,
                mechanism_program_digest TEXT NOT NULL,
                protocol_digest TEXT NOT NULL,
                comparator_identity TEXT NOT NULL,
                observation_seed TEXT NOT NULL,
                observation_id TEXT NOT NULL,
                raw_result_digest TEXT NOT NULL,
                raw_result_json TEXT NOT NULL,
                PRIMARY KEY(
                    candidate_semantic_digest,
                    mechanism_program_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed
                )
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_control_states (
                control_identity_digest TEXT PRIMARY KEY,
                evidence_count INTEGER NOT NULL,
                state_digest TEXT NOT NULL,
                state_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_control_history (
                control_identity_digest TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                state_digest TEXT NOT NULL,
                state_json TEXT NOT NULL,
                PRIMARY KEY(control_identity_digest, evidence_count)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guard_allocation_actions_v31 (
                action_id TEXT PRIMARY KEY,
                control_identity_digest TEXT NOT NULL,
                action TEXT NOT NULL,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                decision_digest TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                closure_status TEXT,
                closure_digest TEXT,
                closure_json TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS guard_allocation_identity_status_v31
            ON guard_allocation_actions_v31(control_identity_digest, status)
            """
        )
        connection.commit()
        self.__connection = connection
        return connection

    def upsert_control_state(
        self,
        *,
        control_identity_digest: str,
        evidence_count: int,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Persist an append-only-by-evidence candidate control snapshot.

        A replay with the same evidence count must be byte-equivalent after
        canonical JSON serialization.  This prevents budget, routing, or claim
        fields from being rewritten without a new admissible observation.
        """

        if (
            isinstance(evidence_count, bool)
            or not isinstance(evidence_count, int)
            or evidence_count < 0
        ):
            raise ValueError("evidence_count must be a non-negative integer")
        state_digest = sha256_digest(state)
        state_json = json.dumps(state, sort_keys=True, separators=(",", ":"))
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT evidence_count, state_digest, state_json
                FROM guard_control_states
                WHERE control_identity_digest=?
                """,
                (control_identity_digest,),
            ).fetchone()
            if row is not None:
                prior_count = int(row[0])
                if evidence_count < prior_count:
                    raise GuardLedgerError("guard control evidence count regressed")
                if evidence_count == prior_count:
                    if str(row[1]) != state_digest:
                        raise GuardLedgerError(
                            "guard control changed without new admissible evidence"
                        )
                    self._connection.commit()
                    return json.loads(row[2])
                self._connection.execute(
                    """
                    UPDATE guard_control_states
                    SET evidence_count=?, state_digest=?, state_json=?
                    WHERE control_identity_digest=?
                    """,
                    (
                        evidence_count,
                        state_digest,
                        state_json,
                        control_identity_digest,
                    ),
                )
            else:
                self._connection.execute(
                    """
                    INSERT INTO guard_control_states(
                        control_identity_digest, evidence_count,
                        state_digest, state_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (
                        control_identity_digest,
                        evidence_count,
                        state_digest,
                        state_json,
                    ),
                )
            self._connection.execute(
                """
                INSERT OR IGNORE INTO guard_control_history(
                    control_identity_digest, evidence_count,
                    state_digest, state_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    control_identity_digest,
                    evidence_count,
                    state_digest,
                    state_json,
                ),
            )
            self._connection.commit()
            return dict(state)
        except Exception:
            self._connection.rollback()
            raise

    def control_states(self) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            """
            SELECT state_json
            FROM guard_control_states
            ORDER BY control_identity_digest
            """
        ).fetchall()
        return tuple(json.loads(row[0]) for row in rows)

    def control_history(self) -> tuple[dict[str, Any], ...]:
        """Return every byte-stable control snapshot for offline replay."""

        rows = self._connection.execute(
            """
            SELECT control_identity_digest, evidence_count,
                   state_digest, state_json
            FROM guard_control_history
            ORDER BY control_identity_digest, evidence_count
            """
        ).fetchall()
        return tuple(
            {
                "control_identity_digest": str(row[0]),
                "evidence_count": int(row[1]),
                "state_digest": str(row[2]),
                "state": json.loads(row[3]),
            }
            for row in rows
        )

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
        mechanism_program_digest: str,
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
            mechanism_program_digest,
            protocol_digest,
            comparator_identity,
            observation_seed,
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT observation_id, raw_result_digest
                FROM guard_evidence_observations_v2
                WHERE candidate_semantic_digest=?
                  AND mechanism_program_digest=?
                  AND protocol_digest=?
                  AND comparator_identity=?
                  AND observation_seed=?
                """,
                key,
            ).fetchone()
            if row is not None:
                if (
                    str(row[0]) == observation_id
                    and str(row[1]) == raw_result_digest
                ):
                    self._connection.commit()
                    return False
                # The exact seed key is a current-evidence index, not an
                # attempt identity.  Guard calls retain the prior attempt;
                # a later retry may replace this index so current science is
                # classified from the current attestation/result.
                self._connection.execute(
                    """
                    UPDATE guard_evidence_observations_v2
                    SET observation_id=?, raw_result_digest=?, raw_result_json=?
                    WHERE candidate_semantic_digest=?
                      AND mechanism_program_digest=?
                      AND protocol_digest=?
                      AND comparator_identity=?
                      AND observation_seed=?
                    """,
                    (observation_id, raw_result_digest, raw_result_json) + key,
                )
                self._connection.commit()
                return True
            self._connection.execute(
                """
                INSERT INTO guard_evidence_observations_v2(
                    candidate_semantic_digest,
                    mechanism_program_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id,
                    raw_result_digest,
                    raw_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
        mechanism_program_digest: str | None = None,
        protocol_digest: str | None = None,
        comparator_identity: str | None = None,
    ) -> GuardEvidenceSnapshotV1:
        identity = (
            candidate_semantic_digest,
            mechanism_program_digest,
            protocol_digest,
            comparator_identity,
        )
        if all(value is None for value in identity):
            rows = self._connection.execute(
                """
                SELECT
                    candidate_semantic_digest,
                    mechanism_program_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id
                FROM guard_evidence_observations_v2
                ORDER BY
                    candidate_semantic_digest,
                    mechanism_program_digest,
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
                    mechanism_program_digest,
                    protocol_digest,
                    comparator_identity,
                    observation_seed,
                    observation_id
                FROM guard_evidence_observations_v2
                WHERE candidate_semantic_digest=?
                  AND mechanism_program_digest=?
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
                    mechanism_program_digest=str(row[1]),
                    protocol_digest=str(row[2]),
                    comparator_identity=str(row[3]),
                    observation_seed=str(row[4]),
                    observation_id=str(row[5]),
                )
                for row in rows
            )
        )

    def raw_results_for_identity(
        self,
        *,
        candidate_semantic_digest: str,
        mechanism_program_digest: str,
        protocol_digest: str,
        comparator_identity: str,
    ) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            """
            SELECT raw_result_json
            FROM guard_evidence_observations_v2
            WHERE candidate_semantic_digest=?
              AND mechanism_program_digest=?
              AND protocol_digest=?
              AND comparator_identity=?
            ORDER BY observation_seed
            """,
            (
                candidate_semantic_digest,
                mechanism_program_digest,
                protocol_digest,
                comparator_identity,
            ),
        ).fetchall()
        unique: dict[str, dict[str, Any]] = {}
        for row in rows:
            payload = json.loads(row[0])
            unique[sha256_digest(payload)] = payload
        return tuple(unique[key] for key in sorted(unique))

    def raw_result_attempts_for_identity(
        self,
        *,
        candidate_semantic_digest: str,
        mechanism_program_digest: str,
        protocol_digest: str,
        comparator_identity: str,
    ) -> tuple[dict[str, Any], ...]:
        """Return POST attempts, including superseded retries, for replay."""

        rows = self._connection.execute(
            """
            SELECT request_json
            FROM guard_calls
            WHERE phase='POST'
            ORDER BY rowid
            """
        ).fetchall()
        attempts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            request = json.loads(row[0])
            if (
                request.get("candidate_semantic_digest") != candidate_semantic_digest
                or request.get("mechanism_program_digest")
                != mechanism_program_digest
                or request.get("protocol_digest") != protocol_digest
                or request.get("comparator_identity") != comparator_identity
            ):
                continue
            raw = request.get("raw_result")
            if not isinstance(raw, dict):
                continue
            digest = str(raw.get("raw_result_digest"))
            if digest in seen:
                continue
            seen.add(digest)
            attempts.append(raw)
        return tuple(attempts)

    def allocation_budget_snapshot(
        self, *, max_actions: int
    ) -> dict[str, Any]:
        if isinstance(max_actions, bool) or max_actions < 0:
            raise ValueError("max_actions must be a non-negative integer")
        rows = self._connection.execute(
            """
            SELECT status, COUNT(*)
            FROM guard_allocation_actions_v31
            GROUP BY status
            """
        ).fetchall()
        counts = {str(status): int(count) for status, count in rows}
        reserved = sum(counts.values())
        open_count = counts.get("RESERVED", 0)
        closed_count = reserved - open_count
        return {
            "schema": "recclaw.helix.frontier-allocation-budget.v31",
            "max_actions": max_actions,
            "reserved_action_count": reserved,
            "open_action_count": open_count,
            "closed_action_count": closed_count,
            "remaining_actions": max(0, max_actions - reserved),
            "matched_budget_semantics": (
                "SUBSET_OF_FROZEN_METRIC_OPPORTUNITIES"
            ),
        }

    def open_allocation_for_identity(
        self, *, control_identity_digest: str
    ) -> dict[str, Any] | None:
        row = self._connection.execute(
            """
            SELECT action_id, action, target, decision_json
            FROM guard_allocation_actions_v31
            WHERE control_identity_digest=? AND status='RESERVED'
            ORDER BY rowid
            LIMIT 1
            """,
            (control_identity_digest,),
        ).fetchone()
        if row is None:
            return None
        return {
            "action_id": str(row[0]),
            "action": str(row[1]),
            "target": str(row[2]),
            "decision": json.loads(row[3]),
        }

    def reserve_allocation_action(
        self,
        *,
        action_id: str,
        control_identity_digest: str,
        action: str,
        target: str,
        max_actions: int,
        decision: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if action not in {"REPLICATE", "CONTROL"}:
            raise ValueError("allocation action must be REPLICATE or CONTROL")
        if isinstance(max_actions, bool) or max_actions < 0:
            raise ValueError("max_actions must be a non-negative integer")
        decision_digest = sha256_digest(decision)
        decision_json = json.dumps(
            decision, sort_keys=True, separators=(",", ":")
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT control_identity_digest, action, target,
                       decision_digest, decision_json, status
                FROM guard_allocation_actions_v31
                WHERE action_id=?
                """,
                (action_id,),
            ).fetchone()
            if row is not None:
                if (
                    str(row[0]) != control_identity_digest
                    or str(row[1]) != action
                    or str(row[2]) != target
                    or str(row[3]) != decision_digest
                ):
                    raise GuardLedgerError("allocation action identity drift")
                self._connection.commit()
                return {
                    "action_id": action_id,
                    "control_identity_digest": str(row[0]),
                    "action": str(row[1]),
                    "target": str(row[2]),
                    "status": str(row[5]),
                    "decision": json.loads(row[4]),
                }, False
            current_count = int(
                self._connection.execute(
                    "SELECT COUNT(*) FROM guard_allocation_actions_v31"
                ).fetchone()[0]
            )
            if current_count >= max_actions:
                raise GuardLedgerError("matched allocation budget exhausted")
            open_for_identity = self._connection.execute(
                """
                SELECT action_id
                FROM guard_allocation_actions_v31
                WHERE control_identity_digest=? AND status='RESERVED'
                LIMIT 1
                """,
                (control_identity_digest,),
            ).fetchone()
            if open_for_identity is not None:
                raise GuardLedgerError(
                    "candidate already has an open allocation action"
                )
            self._connection.execute(
                """
                INSERT INTO guard_allocation_actions_v31(
                    action_id, control_identity_digest, action, target, status,
                    decision_digest, decision_json
                ) VALUES (?, ?, ?, ?, 'RESERVED', ?, ?)
                """,
                (
                    action_id,
                    control_identity_digest,
                    action,
                    target,
                    decision_digest,
                    decision_json,
                ),
            )
            self._connection.commit()
            return {
                "action_id": action_id,
                "control_identity_digest": control_identity_digest,
                "action": action,
                "target": target,
                "status": "RESERVED",
                "decision": dict(decision),
            }, True
        except Exception:
            self._connection.rollback()
            raise

    def close_allocation_action(
        self,
        *,
        action_id: str,
        closure_status: str,
        closure: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        if closure_status not in {"SATISFIED", "FAILED", "CANCELLED"}:
            raise ValueError("allocation closure_status is invalid")
        closure_digest = sha256_digest(closure)
        closure_json = json.dumps(
            closure, sort_keys=True, separators=(",", ":")
        )
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            row = self._connection.execute(
                """
                SELECT status, closure_status, closure_digest, closure_json,
                       control_identity_digest, action, target, decision_json
                FROM guard_allocation_actions_v31
                WHERE action_id=?
                """,
                (action_id,),
            ).fetchone()
            if row is None:
                raise GuardLedgerError("cannot close an unknown allocation action")
            decision = json.loads(row[7])
            for name in (
                "candidate_semantic_digest",
                "mechanism_program_digest",
            ):
                if closure.get(name) != decision.get(name):
                    raise GuardLedgerError(
                        f"allocation action closure {name} drift"
                    )
            if closure.get("allocation_target") != str(row[6]):
                raise GuardLedgerError("allocation action closure target drift")
            if str(row[0]) != "RESERVED":
                if str(row[1]) != closure_status or str(row[2]) != closure_digest:
                    raise GuardLedgerError("allocation action closure drift")
                self._connection.commit()
                return {
                    "action_id": action_id,
                    "control_identity_digest": str(row[4]),
                    "action": str(row[5]),
                    "target": str(row[6]),
                    "status": str(row[0]),
                    "decision": decision,
                    "closure": json.loads(row[3]),
                }, False
            self._connection.execute(
                """
                UPDATE guard_allocation_actions_v31
                SET status=?, closure_status=?, closure_digest=?, closure_json=?
                WHERE action_id=?
                """,
                (
                    closure_status,
                    closure_status,
                    closure_digest,
                    closure_json,
                    action_id,
                ),
            )
            self._connection.commit()
            return {
                "action_id": action_id,
                "control_identity_digest": str(row[4]),
                "action": str(row[5]),
                "target": str(row[6]),
                "status": closure_status,
                "decision": decision,
                "closure": dict(closure),
            }, True
        except Exception:
            self._connection.rollback()
            raise

    def allocation_actions(self) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            """
            SELECT action_id, control_identity_digest, action, target, status,
                   decision_json, closure_status, closure_json
            FROM guard_allocation_actions_v31
            ORDER BY rowid
            """
        ).fetchall()
        return tuple(
            {
                "action_id": str(row[0]),
                "control_identity_digest": str(row[1]),
                "action": str(row[2]),
                "target": str(row[3]),
                "status": str(row[4]),
                "decision": json.loads(row[5]),
                "closure_status": str(row[6]) if row[6] is not None else None,
                "closure": json.loads(row[7]) if row[7] is not None else None,
            }
            for row in rows
        )

    def close(self) -> None:
        if self.__connection is not None:
            self.__connection.close()
            self.__connection = None
