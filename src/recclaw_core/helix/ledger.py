"""C-private single-writer development audit ledger."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest


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
            ORDER BY guard_call_id
            LIMIT 1
            """,
            (phase, candidate_id),
        ).fetchone()
        return (json.loads(row[0]), json.loads(row[1])) if row else None

    def count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM guard_calls").fetchone()[0])

    def close(self) -> None:
        self._connection.close()
