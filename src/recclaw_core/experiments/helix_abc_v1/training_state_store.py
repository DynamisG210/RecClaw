"""Additive single-writer transitions for the M6R training release."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .canonical import content_id, sha256_digest, validate_sha256
from .state_store import (
    EXPECTED_TABLES,
    InvariantViolation,
    ResourceDebitV1,
    SingleWriterExperimentStoreV1,
)
from .training_runtime_contracts import (
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    TrainingRawRunOutputV2,
)
from .training_runtime_release import resolve_bound_training_release


TRAINING_SCHEMA_VERSION = 2
TRAINING_CLAIM_IDENTITY_FIELDS = (
    "runtime_release_digest",
    "runtime_binding_digest",
    "runner_abi",
    "execution_purpose",
    "experiment_id",
    "candidate_id",
    "run_id",
    "budget_digest",
    "metric_contract_digest",
    "resource_contract_digest",
)


@dataclass(frozen=True, slots=True)
class ClaimTrainingExecutionCommandV1:
    round_id: str
    experiment_id: str
    candidate_id: str
    run_id: str
    permit_digest: str
    binding_digest: str
    budget_digest: str
    runtime_release_digest: str
    runtime_binding_digest: str
    runner_abi: str
    execution_purpose: str
    metric_contract_digest: str
    resource_contract_digest: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class PrepareTrainingAttemptCommandV1:
    round_id: str
    claim_id: str
    permit_digest: str
    binding_digest: str
    runtime_release_digest: str
    runtime_binding_digest: str
    runner_abi: str
    execution_purpose: str


@dataclass(frozen=True, slots=True)
class MarkTrainingExecutionStartedCommandV1:
    round_id: str
    claim_id: str
    receipt_artifact_id: str
    confirmation_artifact_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class MarkTrainingExecutionFinishedCommandV1:
    round_id: str
    claim_id: str
    raw_output_artifact_id: str


class TrainingSingleWriterExperimentStoreV1(SingleWriterExperimentStoreV1):
    """The same eight-table store with an additive training-claim closure."""

    def __init__(self, db_path: str | Path, artifact_root: str | Path) -> None:
        super().__init__(
            db_path,
            artifact_root,
            migration_path=(
                Path(__file__).with_name("migrations")
                / "002_training_runtime_release.sql"
            ),
        )

    def _configure_and_migrate(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            journal_mode = self._connection.execute(
                "PRAGMA journal_mode = WAL"
            ).fetchone()[0]
            self._connection.execute("PRAGMA synchronous = FULL")
            if str(journal_mode).lower() != "wal":
                raise InvariantViolation("SQLite did not enter WAL mode")
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                base_path = Path(__file__).with_name("migrations") / (
                    "001_minimum_sufficient.sql"
                )
                self._connection.executescript(base_path.read_text(encoding="utf-8"))
                self._connection.executescript(
                    self.migration_bytes.decode("utf-8")
                )
            elif version != 2:
                raise InvariantViolation(
                    "training store refuses to migrate a historical store"
                )
            report = self.integrity_report()
            if report["tables"] != sorted(EXPECTED_TABLES):
                raise InvariantViolation("training migration must retain eight tables")
            if report["integrity_check"] != "ok" or report["foreign_key_violations"]:
                raise InvariantViolation("training SQLite integrity preflight failed")

    def claim_training_execution(
        self, command: ClaimTrainingExecutionCommandV1
    ) -> dict[str, object]:
        for name in (
            "permit_digest",
            "binding_digest",
            "budget_digest",
            "runtime_release_digest",
            "runtime_binding_digest",
            "metric_contract_digest",
            "resource_contract_digest",
        ):
            validate_sha256(str(getattr(command, name)), field_name=name)
        try:
            release = resolve_bound_training_release(
                runner_abi=command.runner_abi,
                runtime_release_digest=command.runtime_release_digest,
                execution_purpose=command.execution_purpose,
            )
        except ValueError as error:
            raise InvariantViolation(
                "training claim requires a closed package-owned release"
            ) from error
        if (
            command.metric_contract_digest
            != sha256_digest(release.metric_contract)
            or command.resource_contract_digest
            != sha256_digest(release.resource_contract)
        ):
            raise InvariantViolation(
                "training claim contract digests do not match the release"
            )
        payload = {
            "binding_digest": command.binding_digest,
            "budget_digest": command.budget_digest,
            "candidate_id": command.candidate_id,
            "execution_purpose": command.execution_purpose,
            "experiment_id": command.experiment_id,
            "metric_contract_digest": command.metric_contract_digest,
            "permit_digest": command.permit_digest,
            "resource_contract_digest": command.resource_contract_digest,
            "round_id": command.round_id,
            "run_id": command.run_id,
            "runner_abi": command.runner_abi,
            "runtime_binding_digest": command.runtime_binding_digest,
            "runtime_release_digest": command.runtime_release_digest,
        }
        payload_digest = sha256_digest(payload)
        claim_id = content_id("training-execution-claim", payload)
        with self._transaction() as cursor:
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="claim_execution",
                payload_digest=payload_digest,
            )
            if prior is not None:
                return self.get_execution_claim(command.round_id, cursor=cursor)
            round_row = cursor.execute(
                "SELECT status FROM rounds WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            if round_row is None or round_row["status"] != "OPEN":
                raise InvariantViolation("training claim requires an OPEN round")
            try:
                cursor.execute(
                    """
                    INSERT INTO execution_claims (
                        round_id, claim_id, idempotency_key, payload_digest,
                        permit_digest, binding_digest, claim_state,
                        experiment_id, candidate_id, run_id, budget_digest,
                        runtime_release_digest, runtime_binding_digest,
                        runner_abi, execution_purpose, metric_contract_digest,
                        resource_contract_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, 'CLAIMED', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        command.round_id,
                        claim_id,
                        command.idempotency_key,
                        payload_digest,
                        command.permit_digest,
                        command.binding_digest,
                        command.experiment_id,
                        command.candidate_id,
                        command.run_id,
                        command.budget_digest,
                        command.runtime_release_digest,
                        command.runtime_binding_digest,
                        command.runner_abi,
                        command.execution_purpose,
                        command.metric_contract_digest,
                        command.resource_contract_digest,
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise InvariantViolation(
                    "one execution claim per round"
                ) from error
            self._append_event(
                cursor,
                round_id=command.round_id,
                event_type="EXECUTION_CLAIMED",
                payload=payload,
                idempotency_key=f"{command.idempotency_key}:event:claimed",
            )
            return self.get_execution_claim(command.round_id, cursor=cursor)

    @staticmethod
    def _claim_matches(
        claim: sqlite3.Row, command: PrepareTrainingAttemptCommandV1
    ) -> bool:
        return all(
            claim[field] == getattr(command, field)
            for field in (
                "claim_id",
                "permit_digest",
                "binding_digest",
                "runtime_release_digest",
                "runtime_binding_digest",
                "runner_abi",
                "execution_purpose",
            )
        )

    def prepare_training_attempt(
        self, command: PrepareTrainingAttemptCommandV1
    ) -> dict[str, object]:
        with self._transaction() as cursor:
            claim = cursor.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            if claim is None or not self._claim_matches(claim, command):
                raise InvariantViolation(
                    "training attempt requires the exact claim-bound release"
                )
            try:
                resolve_bound_training_release(
                    runner_abi=str(claim["runner_abi"]),
                    runtime_release_digest=str(claim["runtime_release_digest"]),
                    execution_purpose=str(claim["execution_purpose"]),
                )
            except ValueError as error:
                raise InvariantViolation(
                    "training attempt claim release is not registered"
                ) from error
            if claim["attempt_state"] == "PREPARED":
                return self.get_execution_claim(command.round_id, cursor=cursor)
            if (
                claim["claim_state"] != "CLAIMED"
                or claim["attempt_state"] != "NOT_PREPARED"
            ):
                raise InvariantViolation("training attempt cannot be prepared")
            cursor.execute(
                """
                UPDATE execution_claims
                SET attempt_state = 'PREPARED',
                    ordinary_launch_attempt_ordinal = 1
                WHERE round_id = ? AND claim_state = 'CLAIMED'
                  AND attempt_state = 'NOT_PREPARED'
                """,
                (command.round_id,),
            )
            if cursor.rowcount != 1:
                raise InvariantViolation("training attempt preparation lost transition")
            return self.get_execution_claim(command.round_id, cursor=cursor)

    def mark_training_execution_started(
        self, command: MarkTrainingExecutionStartedCommandV1
    ) -> dict[str, object]:
        with self._transaction() as cursor:
            claim = cursor.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            if claim is None or claim["claim_id"] != command.claim_id:
                raise InvariantViolation("training start requires the exact claim")
            try:
                resolve_bound_training_release(
                    runner_abi=str(claim["runner_abi"]),
                    runtime_release_digest=str(claim["runtime_release_digest"]),
                    execution_purpose=str(claim["execution_purpose"]),
                )
            except ValueError as error:
                raise InvariantViolation(
                    "training start claim release is not registered"
                ) from error
            receipt_row = cursor.execute(
                "SELECT * FROM artifact_index WHERE artifact_id = ?",
                (command.receipt_artifact_id,),
            ).fetchone()
            confirmation_row = cursor.execute(
                "SELECT * FROM artifact_index WHERE artifact_id = ?",
                (command.confirmation_artifact_id,),
            ).fetchone()
            if (
                receipt_row is None
                or receipt_row["round_id"] != command.round_id
                or receipt_row["artifact_type"] != "EXECUTION_START_RECEIPT_V2"
                or confirmation_row is None
                or confirmation_row["round_id"] != command.round_id
                or confirmation_row["artifact_type"]
                != "EXECUTION_START_CONFIRMATION_V1"
            ):
                raise InvariantViolation(
                    "training start requires indexed confirmation and receipt"
                )
            try:
                receipt = ExecutionStartReceiptV2(
                    json.loads(
                        self._artifact_target(
                            str(receipt_row["relative_path"])
                        ).read_bytes()
                    )
                )
                confirmation = ExecutionStartConfirmationV1(
                    json.loads(
                        self._artifact_target(
                            str(confirmation_row["relative_path"])
                        ).read_bytes()
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise InvariantViolation(
                    "training start artifacts are unreadable"
                ) from error
            expected = {
                "binding_digest": claim["binding_digest"],
                "claim_id": claim["claim_id"],
                "execution_purpose": claim["execution_purpose"],
                "permit_digest": claim["permit_digest"],
                "round_id": command.round_id,
                "runner_abi": claim["runner_abi"],
                "runtime_binding_digest": claim["runtime_binding_digest"],
                "runtime_release_digest": claim["runtime_release_digest"],
            }
            if any(getattr(receipt, key) != value for key, value in expected.items()):
                raise InvariantViolation("receipt does not match the claim source of truth")
            if any(
                getattr(confirmation, key) != value
                for key, value in expected.items()
            ):
                raise InvariantViolation(
                    "start confirmation does not match the claim source of truth"
                )
            if (
                receipt.start_confirmation_digest != confirmation.digest
                or receipt.start_status != "STARTED"
                or confirmation.start_status != "START_CONFIRMED"
                or receipt.ordinary_launch_attempt_ordinal != 1
                or confirmation.ordinary_launch_attempt_ordinal != 1
            ):
                raise InvariantViolation("training start ordering is not closed")
            if claim["claim_state"] in {"STARTED", "FINISHED"}:
                debits = cursor.execute(
                    "SELECT idempotency_key FROM resource_ledger "
                    "WHERE round_id = ? AND dimension = 'ORDINARY_EXECUTION'",
                    (command.round_id,),
                ).fetchall()
                if (
                    claim["attempt_state"] != "START_CONFIRMED"
                    or len(debits) != 1
                    or debits[0]["idempotency_key"] != command.idempotency_key
                ):
                    raise InvariantViolation(
                        "replayed training start differs from committed bytes"
                    )
                return self.get_execution_claim(command.round_id, cursor=cursor)
            if (
                claim["claim_state"] != "CLAIMED"
                or claim["attempt_state"] != "PREPARED"
            ):
                raise InvariantViolation("training claim cannot enter STARTED")
            self._append_resource(
                cursor,
                round_id=command.round_id,
                debit=ResourceDebitV1("ORDINARY_EXECUTION", 1),
                idempotency_key=command.idempotency_key,
            )
            cursor.execute(
                """
                UPDATE execution_claims
                SET claim_state = 'STARTED', execution_debited = 1,
                    attempt_state = 'START_CONFIRMED'
                WHERE round_id = ? AND claim_id = ?
                  AND claim_state = 'CLAIMED' AND attempt_state = 'PREPARED'
                """,
                (command.round_id, command.claim_id),
            )
            if cursor.rowcount != 1:
                raise InvariantViolation("training start lost its transition")
            return self.get_execution_claim(command.round_id, cursor=cursor)

    def mark_training_execution_finished(
        self, command: MarkTrainingExecutionFinishedCommandV1
    ) -> dict[str, object]:
        with self._transaction() as cursor:
            claim = cursor.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            raw_row = cursor.execute(
                "SELECT * FROM artifact_index WHERE artifact_id = ?",
                (command.raw_output_artifact_id,),
            ).fetchone()
            if (
                claim is None
                or claim["claim_id"] != command.claim_id
                or raw_row is None
                or raw_row["round_id"] != command.round_id
                or raw_row["artifact_type"] != "TRAINING_RAW_RUN_OUTPUT_V2"
            ):
                raise InvariantViolation(
                    "training finish requires its exact claim and raw output"
                )
            try:
                raw_output = TrainingRawRunOutputV2(
                    json.loads(
                        self._artifact_target(
                            str(raw_row["relative_path"])
                        ).read_bytes()
                    )
                )
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise InvariantViolation("training raw output is unreadable") from error
            expected = {
                "binding_digest": claim["binding_digest"],
                "budget_digest": claim["budget_digest"],
                "candidate_id": claim["candidate_id"],
                "execution_purpose": claim["execution_purpose"],
                "experiment_id": claim["experiment_id"],
                "metric_contract_digest": claim["metric_contract_digest"],
                "permit_digest": claim["permit_digest"],
                "round_id": command.round_id,
                "run_id": claim["run_id"],
                "runner_abi": claim["runner_abi"],
                "runtime_binding_digest": claim["runtime_binding_digest"],
                "runtime_release_digest": claim["runtime_release_digest"],
            }
            if any(getattr(raw_output, key) != value for key, value in expected.items()):
                raise InvariantViolation(
                    "training raw output does not match the claim source of truth"
                )
            if claim["claim_state"] == "FINISHED":
                return self.get_execution_claim(command.round_id, cursor=cursor)
            if (
                claim["claim_state"] != "STARTED"
                or claim["execution_debited"] != 1
                or claim["attempt_state"] != "START_CONFIRMED"
            ):
                raise InvariantViolation("only a confirmed training start may finish")
            cursor.execute(
                """
                UPDATE execution_claims
                SET claim_state = 'FINISHED'
                WHERE round_id = ? AND claim_id = ? AND claim_state = 'STARTED'
                """,
                (command.round_id, command.claim_id),
            )
            if cursor.rowcount != 1:
                raise InvariantViolation("training finish lost its transition")
            return self.get_execution_claim(command.round_id, cursor=cursor)


__all__ = [
    "ClaimTrainingExecutionCommandV1",
    "MarkTrainingExecutionFinishedCommandV1",
    "MarkTrainingExecutionStartedCommandV1",
    "PrepareTrainingAttemptCommandV1",
    "TRAINING_CLAIM_IDENTITY_FIELDS",
    "TRAINING_SCHEMA_VERSION",
    "TrainingSingleWriterExperimentStoreV1",
]
