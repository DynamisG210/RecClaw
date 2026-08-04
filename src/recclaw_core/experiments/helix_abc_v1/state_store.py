"""SQLite WAL single-writer state store for the M0 experiment slice."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    content_id,
    sha256_digest,
    validate_relative_artifact_path,
    validate_sha256,
)
from .contracts import (
    ArmCode,
    ExperimentContractV1,
    ResourceCeilingsV1,
)


EXPECTED_TABLES = frozenset(
    {
        "scheduled_slots",
        "rounds",
        "round_events",
        "arm_state",
        "resource_ledger",
        "execution_claims",
        "triplet_barrier",
        "artifact_index",
    }
)

TERMINAL_CLASSES = frozenset(
    {
        "COMPLETED",
        "NO_EXECUTION",
        "ABORTED",
        "ABORTED_RECOVERY_BEFORE_EXECUTION",
        "ABORTED_RECOVERY_START_AMBIGUOUS",
    }
)

RESOURCE_CEILING_FIELDS = {
    "INPUT_TOKEN": "total_input_tokens",
    "OUTPUT_TOKEN": "total_output_tokens",
    "BILLED_TOKEN_DEBIT": "total_billed_token_debit",
    "PROPOSAL": "total_proposal_count",
    "WALL_TIME_MS": "wall_time_ms",
    "RETRY": "retry_debit",
    "PROPOSAL_ATTEMPT": "proposal_attempt_debit",
    "ORDINARY_EXECUTION": "ordinary_executions",
    "COMMON_VALIDATION": "common_validation_count",
    "GPU_DEVICE_TIME_MS": "gpu_device_time_ms",
    "GPU_COST_MICROUNITS": "gpu_cost_microunits",
}

RESOURCE_UNITS = {
    "PROPOSAL_GENERATION_SESSION": "count",
    "PHYSICAL_LLM_CALL": "count",
    "INPUT_TOKEN": "token",
    "OUTPUT_TOKEN": "token",
    "BILLED_TOKEN_DEBIT": "token",
    "PROPOSAL": "count",
    "WALL_TIME_MS": "millisecond",
    "RETRY": "count",
    "PROPOSAL_ATTEMPT": "count",
    "ORDINARY_EXECUTION": "count",
    "COMMON_VALIDATION": "count",
    "GPU_DEVICE_TIME_MS": "millisecond",
    "GPU_COST_MICROUNITS": "microunit",
}


class StateStoreError(RuntimeError):
    pass


class IdempotencyConflict(StateStoreError):
    pass


class InvariantViolation(StateStoreError):
    pass


@dataclass(frozen=True, slots=True)
class OpenRoundCommand:
    experiment_id: str
    arm_instance_id: str
    arm_code: ArmCode
    search_seed: int
    round_index: int
    budget_snapshot: ResourceCeilingsV1
    controller_state_before_digest: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ClaimExecutionCommand:
    round_id: str
    permit_digest: str
    binding_digest: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class MarkExecutionStartedCommand:
    round_id: str
    claim_id: str
    receipt_artifact_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class MarkExecutionFinishedCommand:
    round_id: str
    claim_id: str
    raw_output_artifact_id: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class RegisterArtifactCommand:
    round_id: str | None
    artifact_type: str
    relative_path: str
    producer: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ResourceDebitV1:
    dimension: str
    quantity: int

    def __post_init__(self) -> None:
        if self.dimension not in RESOURCE_UNITS:
            raise ValueError(f"unknown resource dimension: {self.dimension}")
        if not isinstance(self.quantity, int) or isinstance(self.quantity, bool):
            raise ValueError("resource quantity must be an integer")
        if self.quantity < 0:
            raise ValueError("resource quantity must be non-negative")


@dataclass(frozen=True, slots=True)
class CloseRoundCommand:
    round_id: str
    terminal_class: str
    feedback_payload: Mapping[str, Any]
    controller_state_after_digest: str
    resource_debits: tuple[ResourceDebitV1, ...]
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class StopAndFillCommand:
    experiment_id: str
    search_seed: int
    current_round_index: int
    reason: str
    idempotency_key: str


@dataclass(frozen=True, slots=True)
class ConservativeRecoveryCommand:
    experiment_id: str
    search_seed: int
    current_round_index: int
    idempotency_key: str


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class SingleWriterExperimentStoreV1:
    """One package-owned writer using BEGIN IMMEDIATE for every transition."""

    def __init__(
        self,
        db_path: str | Path,
        artifact_root: str | Path,
        *,
        migration_path: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.artifact_root = Path(artifact_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._artifact_root_resolved = self.artifact_root.resolve()
        self.migration_path = (
            Path(migration_path)
            if migration_path is not None
            else Path(__file__).with_name("migrations") / "001_minimum_sufficient.sql"
        )
        self.migration_bytes = self.migration_path.read_bytes()
        self.migration_sha256 = bytes_sha256(self.migration_bytes)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._configure_and_migrate()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SingleWriterExperimentStoreV1":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _configure_and_migrate(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            journal_mode = self._connection.execute(
                "PRAGMA journal_mode = WAL"
            ).fetchone()[0]
            self._connection.execute("PRAGMA synchronous = FULL")
            if str(journal_mode).lower() != "wal":
                raise InvariantViolation("SQLite did not enter WAL mode")
            self._connection.executescript(self.migration_bytes.decode("utf-8"))
            report = self.integrity_report()
            if report["tables"] != sorted(EXPECTED_TABLES):
                raise InvariantViolation("migration must create exactly the M0 eight tables")
            if report["integrity_check"] != "ok" or report["foreign_key_violations"]:
                raise InvariantViolation("SQLite integrity preflight failed")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                yield cursor
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @staticmethod
    def _json(value: Any) -> str:
        return canonical_json_bytes(value).decode("utf-8")

    def _find_idempotency(
        self, cursor: sqlite3.Cursor, key: str
    ) -> tuple[str, str, str] | None:
        lookups = (
            (
                "open_round",
                "SELECT round_id AS object_id, open_payload_digest AS payload_digest "
                "FROM rounds WHERE idempotency_key = ?",
            ),
            (
                "claim_execution",
                "SELECT round_id AS object_id, payload_digest "
                "FROM execution_claims WHERE idempotency_key = ?",
            ),
            (
                "register_artifact",
                "SELECT artifact_id AS object_id, payload_digest "
                "FROM artifact_index WHERE idempotency_key = ?",
            ),
            (
                "event",
                "SELECT round_id AS object_id, payload_digest "
                "FROM round_events WHERE idempotency_key = ?",
            ),
            (
                "stop_and_fill",
                "SELECT CAST(round_index AS TEXT) AS object_id, "
                "stop_payload_digest AS payload_digest "
                "FROM triplet_barrier WHERE stop_idempotency_key = ?",
            ),
            (
                "conservative_recovery",
                "SELECT CAST(round_index AS TEXT) AS object_id, "
                "recovery_payload_digest AS payload_digest "
                "FROM triplet_barrier WHERE recovery_idempotency_key = ?",
            ),
        )
        for operation, query in lookups:
            row = cursor.execute(query, (key,)).fetchone()
            if row is not None:
                return operation, str(row["object_id"]), str(row["payload_digest"])
        return None

    def _check_idempotency(
        self,
        cursor: sqlite3.Cursor,
        *,
        key: str,
        operation: str,
        payload_digest: str,
    ) -> str | None:
        found = self._find_idempotency(cursor, key)
        if found is None:
            return None
        found_operation, object_id, found_digest = found
        event_operation = (
            operation == "close_round" and found_operation == "event"
        )
        if (
            found_digest != payload_digest
            or (found_operation != operation and not event_operation)
        ):
            raise IdempotencyConflict(
                f"idempotency key {key!r} already binds different command bytes"
            )
        return object_id

    @staticmethod
    def _next_event_seq(cursor: sqlite3.Cursor, round_id: str) -> int:
        row = cursor.execute(
            "SELECT COALESCE(MAX(event_seq), 0) + 1 FROM round_events WHERE round_id = ?",
            (round_id,),
        ).fetchone()
        return int(row[0])

    def _append_event(
        self,
        cursor: sqlite3.Cursor,
        *,
        round_id: str,
        event_type: str,
        payload: Mapping[str, Any],
        idempotency_key: str,
    ) -> None:
        seq = self._next_event_seq(cursor, round_id)
        digest = sha256_digest(payload)
        cursor.execute(
            """
            INSERT INTO round_events (
                round_id, event_seq, event_type, idempotency_key,
                payload_json, payload_digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                round_id,
                seq,
                event_type,
                idempotency_key,
                self._json(payload),
                digest,
            ),
        )

    def _round_budget(
        self, cursor: sqlite3.Cursor, round_id: str
    ) -> ResourceCeilingsV1:
        row = cursor.execute(
            "SELECT budget_snapshot_json FROM rounds WHERE round_id = ?",
            (round_id,),
        ).fetchone()
        if row is None:
            raise InvariantViolation("unknown round")
        return ResourceCeilingsV1(**json.loads(row["budget_snapshot_json"]))

    def _append_resource(
        self,
        cursor: sqlite3.Cursor,
        *,
        round_id: str,
        debit: ResourceDebitV1,
        idempotency_key: str,
    ) -> None:
        if debit.dimension == "PROPOSAL_GENERATION_SESSION":
            prior = cursor.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM resource_ledger "
                "WHERE round_id = ? AND dimension = ?",
                (round_id, debit.dimension),
            ).fetchone()[0]
            if int(prior) + debit.quantity > 1:
                raise InvariantViolation("one round permits exactly one proposal session")
        ceiling_field = RESOURCE_CEILING_FIELDS.get(debit.dimension)
        if ceiling_field is not None:
            ceiling = getattr(self._round_budget(cursor, round_id), ceiling_field)
            prior = cursor.execute(
                "SELECT COALESCE(SUM(quantity), 0) FROM resource_ledger "
                "WHERE round_id = ? AND dimension = ?",
                (round_id, debit.dimension),
            ).fetchone()[0]
            if int(prior) + debit.quantity > ceiling:
                raise InvariantViolation(
                    f"{debit.dimension} debit exceeds the frozen round ceiling"
                )
        payload = {
            "dimension": debit.dimension,
            "quantity": debit.quantity,
            "round_id": round_id,
            "unit": RESOURCE_UNITS[debit.dimension],
        }
        cursor.execute(
            """
            INSERT INTO resource_ledger (
                ledger_id, round_id, dimension, quantity, unit,
                idempotency_key, payload_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content_id("resource-ledger", payload),
                round_id,
                debit.dimension,
                debit.quantity,
                RESOURCE_UNITS[debit.dimension],
                idempotency_key,
                sha256_digest(payload),
            ),
        )

    def initialize_experiment(
        self,
        contract: ExperimentContractV1,
        *,
        arm_instance_ids: Mapping[ArmCode, str] | None = None,
    ) -> dict[ArmCode, str]:
        arm_ids = dict(
            arm_instance_ids
            or {
                arm: content_id(
                    "arm-instance",
                    {"arm": arm.value, "experiment_id": contract.experiment_id},
                )
                for arm in ArmCode
            }
        )
        if set(arm_ids) != set(ArmCode):
            raise ValueError("arm_instance_ids must define exactly A/B/C")
        initial_controller_digest = sha256_digest(
            {"experiment_contract_digest": contract.identity_digest, "state": "GENESIS"}
        )
        initial_search_memory_digest = sha256_digest(
            {
                "contract": contract.identity_digest,
                "search_memory": "INITIAL_EMPTY",
            }
        )
        with self._transaction() as cursor:
            for seed in contract.search_seeds:
                for arm in ArmCode:
                    policy = next(item for item in contract.arm_policies if item.arm is arm)
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO arm_state (
                            experiment_id, arm_instance_id, arm_code, search_seed,
                            experiment_contract_digest, state, next_round_index,
                            controller_state_digest, meta_policy_digest,
                            search_memory_digest
                        ) VALUES (?, ?, ?, ?, ?, 'ACTIVE', 1, ?, ?, ?)
                        """,
                        (
                            contract.experiment_id,
                            arm_ids[arm],
                            arm.value,
                            seed,
                            contract.identity_digest,
                            initial_controller_digest,
                            policy.controller_policy_digest
                            if policy.meta_policy_mode is not None
                            else None,
                            initial_search_memory_digest
                            if arm is not ArmCode.A
                            else None,
                        ),
                    )
                    for round_index in range(
                        1, contract.scheduled_slots_per_arm_seed + 1
                    ):
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO scheduled_slots (
                                experiment_id, arm_instance_id, arm_code,
                                search_seed, round_index, slot_status
                            ) VALUES (?, ?, ?, ?, ?, 'PLANNED')
                            """,
                            (
                                contract.experiment_id,
                                arm_ids[arm],
                                arm.value,
                                seed,
                                round_index,
                            ),
                        )
                for round_index in range(
                    1, contract.scheduled_slots_per_arm_seed + 1
                ):
                    cursor.execute(
                        """
                        INSERT OR IGNORE INTO triplet_barrier (
                            experiment_id, search_seed, round_index,
                            arm_a_instance_id, arm_b_instance_id, arm_c_instance_id
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            contract.experiment_id,
                            seed,
                            round_index,
                            arm_ids[ArmCode.A],
                            arm_ids[ArmCode.B],
                            arm_ids[ArmCode.C],
                        ),
                    )
            bad = cursor.execute(
                "SELECT COUNT(*) FROM arm_state "
                "WHERE experiment_id = ? AND experiment_contract_digest != ?",
                (contract.experiment_id, contract.identity_digest),
            ).fetchone()[0]
            if int(bad):
                raise InvariantViolation("experiment identity conflicts with existing state")
            for seed in contract.search_seeds:
                state_rows = cursor.execute(
                    """
                    SELECT arm_instance_id, arm_code, experiment_contract_digest
                    FROM arm_state
                    WHERE experiment_id = ? AND search_seed = ?
                    """,
                    (contract.experiment_id, seed),
                ).fetchall()
                expected_states = {
                    (
                        arm_ids[arm],
                        arm.value,
                        contract.identity_digest,
                    )
                    for arm in ArmCode
                }
                actual_states = {
                    (
                        str(row["arm_instance_id"]),
                        str(row["arm_code"]),
                        str(row["experiment_contract_digest"]),
                    )
                    for row in state_rows
                }
                if actual_states != expected_states:
                    raise InvariantViolation(
                        "experiment already binds a different Arm-instance tuple"
                    )
                slot_count = cursor.execute(
                    """
                    SELECT COUNT(*) FROM scheduled_slots
                    WHERE experiment_id = ? AND search_seed = ?
                    """,
                    (contract.experiment_id, seed),
                ).fetchone()[0]
                if int(slot_count) != 3 * contract.scheduled_slots_per_arm_seed:
                    raise InvariantViolation(
                        "scheduled slot set is not the exact frozen A/B/C matrix"
                    )
        return arm_ids

    def open_round(self, command: OpenRoundCommand) -> dict[str, Any]:
        validate_sha256(
            command.controller_state_before_digest,
            field_name="controller_state_before_digest",
        )
        budget_json = self._json(command.budget_snapshot)
        payload = {
            "arm_code": command.arm_code.value,
            "arm_instance_id": command.arm_instance_id,
            "budget_snapshot_digest": sha256_digest(command.budget_snapshot),
            "controller_state_before_digest": command.controller_state_before_digest,
            "experiment_id": command.experiment_id,
            "round_index": command.round_index,
            "search_seed": command.search_seed,
        }
        payload_digest = sha256_digest(payload)
        round_id = content_id("search-round", payload)
        with self._transaction() as cursor:
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="open_round",
                payload_digest=payload_digest,
            )
            if prior is not None:
                return self.get_round(prior, cursor=cursor)
            arm_state = cursor.execute(
                """
                SELECT * FROM arm_state
                WHERE experiment_id = ? AND arm_instance_id = ? AND search_seed = ?
                """,
                (
                    command.experiment_id,
                    command.arm_instance_id,
                    command.search_seed,
                ),
            ).fetchone()
            if arm_state is None or arm_state["arm_code"] != command.arm_code.value:
                raise InvariantViolation("unknown or mismatched Arm instance")
            if arm_state["state"] != "ACTIVE":
                raise InvariantViolation("no round may open after stop")
            if int(arm_state["next_round_index"]) != command.round_index:
                raise InvariantViolation("round index is not the Arm's exact next index")
            if (
                str(arm_state["controller_state_digest"])
                != command.controller_state_before_digest
            ):
                raise InvariantViolation(
                    "round before-state does not match the committed Arm state"
                )
            if command.round_index > 1:
                prior_barrier = cursor.execute(
                    """
                    SELECT next_index_authorized FROM triplet_barrier
                    WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                    """,
                    (
                        command.experiment_id,
                        command.search_seed,
                        command.round_index - 1,
                    ),
                ).fetchone()
                if prior_barrier is None or int(prior_barrier[0]) != 1:
                    raise InvariantViolation(
                        "triplet barrier blocks the next round index"
                    )
            stop_count = cursor.execute(
                """
                SELECT COUNT(*) FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND stop_requested = 1
                """,
                (command.experiment_id, command.search_seed),
            ).fetchone()[0]
            if int(stop_count):
                raise InvariantViolation("stop transaction blocks all future opens")
            slot = cursor.execute(
                """
                SELECT * FROM scheduled_slots
                WHERE experiment_id = ? AND arm_instance_id = ?
                  AND search_seed = ? AND round_index = ?
                """,
                (
                    command.experiment_id,
                    command.arm_instance_id,
                    command.search_seed,
                    command.round_index,
                ),
            ).fetchone()
            if slot is None or slot["slot_status"] != "PLANNED":
                raise InvariantViolation("scheduled slot is not PLANNED")
            try:
                cursor.execute(
                    """
                    INSERT INTO rounds (
                        round_id, experiment_id, arm_instance_id, arm_code,
                        search_seed, round_index, idempotency_key,
                        open_payload_digest, budget_snapshot_json,
                        budget_snapshot_digest, controller_state_before_digest,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')
                    """,
                    (
                        round_id,
                        command.experiment_id,
                        command.arm_instance_id,
                        command.arm_code.value,
                        command.search_seed,
                        command.round_index,
                        command.idempotency_key,
                        payload_digest,
                        budget_json,
                        sha256_digest(command.budget_snapshot),
                        command.controller_state_before_digest,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise InvariantViolation("one round per Arm/seed/index") from exc
            cursor.execute(
                """
                UPDATE scheduled_slots
                SET slot_status = 'OPENED', round_id = ?
                WHERE experiment_id = ? AND arm_instance_id = ?
                  AND search_seed = ? AND round_index = ? AND slot_status = 'PLANNED'
                """,
                (
                    round_id,
                    command.experiment_id,
                    command.arm_instance_id,
                    command.search_seed,
                    command.round_index,
                ),
            )
            self._append_event(
                cursor,
                round_id=round_id,
                event_type="ROUND_OPENED",
                payload=payload,
                idempotency_key=f"{command.idempotency_key}:event:opened",
            )
            self._append_resource(
                cursor,
                round_id=round_id,
                debit=ResourceDebitV1("PROPOSAL_GENERATION_SESSION", 1),
                idempotency_key=f"{command.idempotency_key}:resource:session",
            )
            return self.get_round(round_id, cursor=cursor)

    def claim_execution(self, command: ClaimExecutionCommand) -> dict[str, Any]:
        validate_sha256(command.permit_digest, field_name="permit_digest")
        validate_sha256(command.binding_digest, field_name="binding_digest")
        payload = {
            "binding_digest": command.binding_digest,
            "permit_digest": command.permit_digest,
            "round_id": command.round_id,
        }
        payload_digest = sha256_digest(payload)
        claim_id = content_id("execution-claim", payload)
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
                "SELECT status FROM rounds WHERE round_id = ?", (command.round_id,)
            ).fetchone()
            if round_row is None or round_row["status"] != "OPEN":
                raise InvariantViolation("execution claim requires an OPEN round")
            try:
                cursor.execute(
                    """
                    INSERT INTO execution_claims (
                        round_id, claim_id, idempotency_key, payload_digest,
                        permit_digest, binding_digest, claim_state
                    ) VALUES (?, ?, ?, ?, ?, ?, 'CLAIMED')
                    """,
                    (
                        command.round_id,
                        claim_id,
                        command.idempotency_key,
                        payload_digest,
                        command.permit_digest,
                        command.binding_digest,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise InvariantViolation("one execution claim per round") from exc
            self._append_event(
                cursor,
                round_id=command.round_id,
                event_type="EXECUTION_CLAIMED",
                payload=payload,
                idempotency_key=f"{command.idempotency_key}:event:claimed",
            )
            return self.get_execution_claim(command.round_id, cursor=cursor)

    def mark_execution_started(
        self, command: MarkExecutionStartedCommand
    ) -> dict[str, Any]:
        """Bind the committed claim to an append-only start-receipt artifact."""

        with self._transaction() as cursor:
            claim = cursor.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            if claim is None or claim["claim_id"] != command.claim_id:
                raise InvariantViolation("execution start requires the exact committed claim")
            artifact = cursor.execute(
                "SELECT * FROM artifact_index WHERE artifact_id = ?",
                (command.receipt_artifact_id,),
            ).fetchone()
            if (
                artifact is None
                or artifact["round_id"] != command.round_id
                or artifact["artifact_type"] != "EXECUTION_START_RECEIPT_V1"
            ):
                raise InvariantViolation("execution start requires its indexed receipt")
            receipt_path = self._artifact_target(str(artifact["relative_path"]))
            try:
                receipt = json.loads(receipt_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise InvariantViolation("execution start receipt is unreadable") from exc
            expected_receipt_keys = {
                "binding_digest",
                "claim_id",
                "ordinary_launch_attempt_ordinal",
                "permit_digest",
                "round_id",
                "run_id",
                "runner_abi",
                "start_status",
            }
            if (
                not isinstance(receipt, dict)
                or set(receipt) != expected_receipt_keys
                or receipt["binding_digest"] != claim["binding_digest"]
                or receipt["claim_id"] != claim["claim_id"]
                or receipt["permit_digest"] != claim["permit_digest"]
                or receipt["round_id"] != command.round_id
                or receipt["ordinary_launch_attempt_ordinal"] != 1
                or receipt["start_status"] != "STARTED"
                or receipt["runner_abi"] != "recclaw.fake-non-training-runner.v1"
            ):
                raise InvariantViolation("execution start receipt does not bind the claim")
            if claim["claim_state"] in {"STARTED", "FINISHED"}:
                debit = cursor.execute(
                    "SELECT idempotency_key FROM resource_ledger "
                    "WHERE round_id = ? AND dimension = 'ORDINARY_EXECUTION'",
                    (command.round_id,),
                ).fetchall()
                if len(debit) != 1 or debit[0]["idempotency_key"] != command.idempotency_key:
                    raise InvariantViolation("started execution is missing its debit")
                return self.get_execution_claim(command.round_id, cursor=cursor)
            if claim["claim_state"] != "CLAIMED":
                raise InvariantViolation("execution claim cannot transition to STARTED")
            self._append_resource(
                cursor,
                round_id=command.round_id,
                debit=ResourceDebitV1("ORDINARY_EXECUTION", 1),
                idempotency_key=command.idempotency_key,
            )
            cursor.execute(
                """
                UPDATE execution_claims
                SET claim_state = 'STARTED', execution_debited = 1
                WHERE round_id = ? AND claim_id = ? AND claim_state = 'CLAIMED'
                """,
                (command.round_id, command.claim_id),
            )
            if cursor.rowcount != 1:
                raise InvariantViolation("execution start lost its claim transition")
            return self.get_execution_claim(command.round_id, cursor=cursor)

    def mark_execution_finished(
        self, command: MarkExecutionFinishedCommand
    ) -> dict[str, Any]:
        """Close one started fake execution against its raw-output artifact."""

        with self._transaction() as cursor:
            claim = cursor.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (command.round_id,),
            ).fetchone()
            if claim is None or claim["claim_id"] != command.claim_id:
                raise InvariantViolation("execution finish requires the exact committed claim")
            artifacts = cursor.execute(
                "SELECT * FROM artifact_index "
                "WHERE round_id = ? AND artifact_type = 'RAW_RUN_OUTPUT_V1'",
                (command.round_id,),
            ).fetchall()
            artifact = artifacts[0] if len(artifacts) == 1 else None
            if (
                artifact is None
                or artifact["artifact_id"] != command.raw_output_artifact_id
                or artifact["round_id"] != command.round_id
                or artifact["artifact_type"] != "RAW_RUN_OUTPUT_V1"
            ):
                raise InvariantViolation("execution finish requires its indexed raw output")
            raw_path = self._artifact_target(str(artifact["relative_path"]))
            try:
                raw_output = json.loads(raw_path.read_bytes())
            except (OSError, json.JSONDecodeError) as exc:
                raise InvariantViolation("raw execution output is unreadable") from exc
            expected_raw_keys = {
                "binding_digest",
                "candidate_id",
                "checks",
                "evaluation_purpose",
                "exit_status",
                "interface_loss",
                "mechanism_axes_exercised",
                "normalized_metrics",
                "optimizer_steps",
                "permit_digest",
                "round_id",
                "run_id",
                "runner_abi",
                "training_backend_started",
            }
            if (
                not isinstance(raw_output, dict)
                or set(raw_output) != expected_raw_keys
                or raw_output["binding_digest"] != claim["binding_digest"]
                or raw_output["permit_digest"] != claim["permit_digest"]
                or raw_output["round_id"] != command.round_id
                or raw_output["runner_abi"] != "recclaw.fake-non-training-runner.v1"
                or raw_output["evaluation_purpose"]
                != "NON_OUTCOME_BEARING_INTERFACE_SMOKE"
                or raw_output["normalized_metrics"] != {}
                or raw_output["optimizer_steps"] != 0
                or raw_output["training_backend_started"] is not False
            ):
                raise InvariantViolation("raw execution output does not bind the claim")
            if claim["claim_state"] == "FINISHED":
                return self.get_execution_claim(command.round_id, cursor=cursor)
            if claim["claim_state"] != "STARTED" or claim["execution_debited"] != 1:
                raise InvariantViolation("only a started, debited execution may finish")
            cursor.execute(
                """
                UPDATE execution_claims
                SET claim_state = 'FINISHED'
                WHERE round_id = ? AND claim_id = ? AND claim_state = 'STARTED'
                """,
                (command.round_id, command.claim_id),
            )
            if cursor.rowcount != 1:
                raise InvariantViolation("execution finish lost its claim transition")
            return self.get_execution_claim(command.round_id, cursor=cursor)

    def _artifact_target(self, relative_path: str) -> Path:
        normalized = validate_relative_artifact_path(relative_path)
        target = self.artifact_root.joinpath(*normalized.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        for parent in (target.parent, *target.parent.parents):
            if parent == self._artifact_root_resolved:
                break
            if parent.is_symlink():
                raise InvariantViolation("artifact path traverses a symlink")
        resolved_parent = target.parent.resolve()
        try:
            resolved_parent.relative_to(self._artifact_root_resolved)
        except ValueError as exc:
            raise InvariantViolation("artifact path escapes the private root") from exc
        return target

    @staticmethod
    def _atomic_write(target: Path, data: bytes) -> None:
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, target)
            try:
                directory_fd = os.open(target.parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is None:
                return
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    def register_artifact(
        self, command: RegisterArtifactCommand, data: bytes
    ) -> dict[str, Any]:
        if not isinstance(data, bytes):
            raise TypeError("artifact data must be bytes")
        relative_path = validate_relative_artifact_path(command.relative_path)
        artifact_sha256 = bytes_sha256(data)
        payload = {
            "artifact_type": command.artifact_type,
            "producer": command.producer,
            "relative_path": relative_path,
            "round_id": command.round_id,
            "sha256": artifact_sha256,
            "size_bytes": len(data),
        }
        payload_digest = sha256_digest(payload)
        artifact_id = content_id("artifact", payload)
        with self._lock:
            cursor = self._connection.cursor()
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="register_artifact",
                payload_digest=payload_digest,
            )
            if prior is not None:
                row = self.get_artifact(prior, cursor=cursor)
                target = self._artifact_target(row["relative_path"])
                if not target.exists() or bytes_sha256(target.read_bytes()) != row["sha256"]:
                    raise InvariantViolation("indexed artifact bytes are missing or changed")
                return row
            if command.round_id is not None:
                round_row = cursor.execute(
                    "SELECT status FROM rounds WHERE round_id = ?",
                    (command.round_id,),
                ).fetchone()
                if round_row is None or round_row["status"] != "OPEN":
                    raise InvariantViolation("round artifact requires an OPEN round")
            target = self._artifact_target(relative_path)
            if target.exists():
                if bytes_sha256(target.read_bytes()) != artifact_sha256:
                    raise InvariantViolation("artifact path already contains different bytes")
            else:
                self._atomic_write(target, data)
            if bytes_sha256(target.read_bytes()) != artifact_sha256:
                raise InvariantViolation("artifact digest changed after atomic rename")
            with self._transaction() as tx:
                prior = self._check_idempotency(
                    tx,
                    key=command.idempotency_key,
                    operation="register_artifact",
                    payload_digest=payload_digest,
                )
                if prior is not None:
                    return self.get_artifact(prior, cursor=tx)
                tx.execute(
                    """
                    INSERT INTO artifact_index (
                        artifact_id, round_id, artifact_type, relative_path,
                        size_bytes, sha256, producer, idempotency_key,
                        payload_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        command.round_id,
                        command.artifact_type,
                        relative_path,
                        len(data),
                        artifact_sha256,
                        command.producer,
                        command.idempotency_key,
                        payload_digest,
                    ),
                )
                if command.round_id is not None:
                    self._append_event(
                        tx,
                        round_id=command.round_id,
                        event_type="ARTIFACT_REGISTERED",
                        payload=payload,
                        idempotency_key=f"{command.idempotency_key}:event:artifact",
                    )
                return self.get_artifact(artifact_id, cursor=tx)

    def close_round(self, command: CloseRoundCommand) -> dict[str, Any]:
        if command.terminal_class not in TERMINAL_CLASSES:
            raise ValueError("terminal_class is outside the M0 closed domain")
        validate_sha256(
            command.controller_state_after_digest,
            field_name="controller_state_after_digest",
        )
        payload = {
            "controller_state_after_digest": command.controller_state_after_digest,
            "feedback_digest": sha256_digest(command.feedback_payload),
            "resource_debits": [
                {"dimension": item.dimension, "quantity": item.quantity}
                for item in command.resource_debits
            ],
            "round_id": command.round_id,
            "terminal_class": command.terminal_class,
        }
        payload_digest = sha256_digest(payload)
        with self._transaction() as cursor:
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="close_round",
                payload_digest=payload_digest,
            )
            if prior is not None:
                return self.get_round(prior, cursor=cursor)
            round_row = cursor.execute(
                "SELECT * FROM rounds WHERE round_id = ?", (command.round_id,)
            ).fetchone()
            if round_row is None:
                raise InvariantViolation("unknown round")
            if round_row["status"] != "OPEN":
                raise IdempotencyConflict(
                    "round is already terminal under a different command"
                )
            for index, debit in enumerate(command.resource_debits):
                self._append_resource(
                    cursor,
                    round_id=command.round_id,
                    debit=debit,
                    idempotency_key=f"{command.idempotency_key}:resource:{index}",
                )
            feedback_payload = {
                "feedback": command.feedback_payload,
                "round_id": command.round_id,
                "terminal_class": command.terminal_class,
            }
            self._append_event(
                cursor,
                round_id=command.round_id,
                event_type="ROUND_FEEDBACK",
                payload=feedback_payload,
                idempotency_key=f"{command.idempotency_key}:feedback",
            )
            self._append_event(
                cursor,
                round_id=command.round_id,
                event_type="ROUND_CLOSED",
                payload=payload,
                idempotency_key=command.idempotency_key,
            )
            status = (
                "ABORTED"
                if command.terminal_class.startswith("ABORTED")
                else "CLOSED"
            )
            cursor.execute(
                """
                UPDATE rounds
                SET status = ?, terminal_class = ?, feedback_digest = ?,
                    controller_state_after_digest = ?
                WHERE round_id = ? AND status = 'OPEN'
                """,
                (
                    status,
                    command.terminal_class,
                    sha256_digest(command.feedback_payload),
                    command.controller_state_after_digest,
                    command.round_id,
                ),
            )
            cursor.execute(
                """
                UPDATE scheduled_slots
                SET slot_status = ?
                WHERE round_id = ? AND slot_status = 'OPENED'
                """,
                (status, command.round_id),
            )
            bit = {"A": 1, "B": 2, "C": 4}[round_row["arm_code"]]
            barrier = cursor.execute(
                """
                SELECT * FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    round_row["experiment_id"],
                    round_row["search_seed"],
                    round_row["round_index"],
                ),
            ).fetchone()
            if barrier is None:
                raise InvariantViolation("missing triplet barrier")
            bitmap = int(barrier["closed_bitmap"]) | bit
            authorized = int(bitmap == 7 and int(barrier["stop_requested"]) == 0)
            cursor.execute(
                """
                UPDATE triplet_barrier
                SET closed_bitmap = ?, next_index_authorized = ?
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    bitmap,
                    authorized,
                    round_row["experiment_id"],
                    round_row["search_seed"],
                    round_row["round_index"],
                ),
            )
            cursor.execute(
                """
                UPDATE arm_state
                SET next_round_index = ?, controller_state_digest = ?,
                    revision = revision + 1
                WHERE experiment_id = ? AND arm_instance_id = ? AND search_seed = ?
                """,
                (
                    int(round_row["round_index"]) + 1,
                    command.controller_state_after_digest,
                    round_row["experiment_id"],
                    round_row["arm_instance_id"],
                    round_row["search_seed"],
                ),
            )
            open_count = cursor.execute(
                """
                SELECT COUNT(*) FROM rounds
                WHERE experiment_id = ? AND search_seed = ? AND status = 'OPEN'
                """,
                (round_row["experiment_id"], round_row["search_seed"]),
            ).fetchone()[0]
            stopped = cursor.execute(
                """
                SELECT COUNT(*) FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND stop_requested = 1
                """,
                (round_row["experiment_id"], round_row["search_seed"]),
            ).fetchone()[0]
            if int(stopped) and int(open_count) == 0:
                cursor.execute(
                    """
                    UPDATE arm_state SET state = 'STOPPED', revision = revision + 1
                    WHERE experiment_id = ? AND search_seed = ?
                    """,
                    (round_row["experiment_id"], round_row["search_seed"]),
                )
            return self.get_round(command.round_id, cursor=cursor)

    def stop_and_fill_remaining(
        self, command: StopAndFillCommand
    ) -> dict[str, Any]:
        if not command.reason:
            raise ValueError("stop reason is required")
        payload = {
            "current_round_index": command.current_round_index,
            "experiment_id": command.experiment_id,
            "reason": command.reason,
            "search_seed": command.search_seed,
        }
        payload_digest = sha256_digest(payload)
        with self._transaction() as cursor:
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="stop_and_fill",
                payload_digest=payload_digest,
            )
            if prior is not None:
                return self._stop_result(cursor, command)
            barrier = cursor.execute(
                """
                SELECT * FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            ).fetchone()
            if barrier is None:
                raise InvariantViolation("unknown triplet barrier")
            if barrier["stop_idempotency_key"] is not None:
                raise IdempotencyConflict("a different stop command already won")
            cursor.execute(
                """
                UPDATE triplet_barrier
                SET stop_requested = 1, next_index_authorized = 0,
                    stop_reason = ?
                WHERE experiment_id = ? AND search_seed = ?
                  AND round_index >= ?
                """,
                (
                    command.reason,
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            )
            cursor.execute(
                """
                UPDATE triplet_barrier
                SET stop_idempotency_key = ?, stop_payload_digest = ?
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    command.idempotency_key,
                    payload_digest,
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            )
            cursor.execute(
                """
                UPDATE scheduled_slots
                SET slot_status = 'NOT_STARTED_STOP', stop_reason = ?
                WHERE experiment_id = ? AND search_seed = ?
                  AND round_index >= ? AND slot_status = 'PLANNED'
                """,
                (
                    command.reason,
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            )
            cursor.execute(
                """
                UPDATE arm_state
                SET state = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM rounds
                            WHERE rounds.experiment_id = arm_state.experiment_id
                              AND rounds.arm_instance_id = arm_state.arm_instance_id
                              AND rounds.search_seed = arm_state.search_seed
                              AND rounds.status = 'OPEN'
                        ) THEN 'STOPPING'
                        ELSE 'STOPPED'
                    END,
                    stop_reason = ?,
                    revision = revision + 1
                WHERE experiment_id = ? AND search_seed = ?
                """,
                (command.reason, command.experiment_id, command.search_seed),
            )
            return self._stop_result(cursor, command)

    @staticmethod
    def _stop_result(
        cursor: sqlite3.Cursor, command: StopAndFillCommand
    ) -> dict[str, Any]:
        counts = {
            row["slot_status"]: int(row["count"])
            for row in cursor.execute(
                """
                SELECT slot_status, COUNT(*) AS count
                FROM scheduled_slots
                WHERE experiment_id = ? AND search_seed = ?
                GROUP BY slot_status
                """,
                (command.experiment_id, command.search_seed),
            )
        }
        return {
            "current_round_index": command.current_round_index,
            "experiment_id": command.experiment_id,
            "search_seed": command.search_seed,
            "slot_status_counts": counts,
            "stop_reason": command.reason,
        }

    def conservative_recovery(
        self, command: ConservativeRecoveryCommand
    ) -> dict[str, Any]:
        payload = {
            "current_round_index": command.current_round_index,
            "experiment_id": command.experiment_id,
            "search_seed": command.search_seed,
        }
        payload_digest = sha256_digest(payload)
        with self._transaction() as cursor:
            prior = self._check_idempotency(
                cursor,
                key=command.idempotency_key,
                operation="conservative_recovery",
                payload_digest=payload_digest,
            )
            if prior is not None:
                row = cursor.execute(
                    """
                    SELECT recovery_result_json FROM triplet_barrier
                    WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                    """,
                    (
                        command.experiment_id,
                        command.search_seed,
                        command.current_round_index,
                    ),
                ).fetchone()
                if row is None or row["recovery_result_json"] is None:
                    raise InvariantViolation("recovery replay record is incomplete")
                return json.loads(row["recovery_result_json"])
            barrier = cursor.execute(
                """
                SELECT recovery_idempotency_key FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            ).fetchone()
            if barrier is None:
                raise InvariantViolation("unknown triplet barrier")
            if barrier["recovery_idempotency_key"] is not None:
                raise IdempotencyConflict("a different recovery command already won")
        open_rounds = [
            dict(row)
            for row in self._connection.execute(
                """
                SELECT * FROM rounds
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                  AND status = 'OPEN'
                ORDER BY round_id
                """,
                (
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            )
        ]
        for round_row in open_rounds:
            claim = self._connection.execute(
                "SELECT * FROM execution_claims WHERE round_id = ?",
                (round_row["round_id"],),
            ).fetchone()
            if claim is None:
                terminal_class = "ABORTED_RECOVERY_BEFORE_EXECUTION"
            else:
                terminal_class = "ABORTED_RECOVERY_START_AMBIGUOUS"
                if claim["claim_state"] == "CLAIMED":
                    with self._transaction() as cursor:
                        cursor.execute(
                            """
                            UPDATE execution_claims
                            SET claim_state = 'START_AMBIGUOUS', execution_debited = 1
                            WHERE round_id = ? AND claim_state = 'CLAIMED'
                            """,
                            (round_row["round_id"],),
                        )
                        if cursor.rowcount:
                            self._append_resource(
                                cursor,
                                round_id=round_row["round_id"],
                                debit=ResourceDebitV1("ORDINARY_EXECUTION", 1),
                                idempotency_key=(
                                    f"recovery:{round_row['round_id']}:resource:execution"
                                ),
                            )
                            self._append_event(
                                cursor,
                                round_id=round_row["round_id"],
                                event_type="EXECUTION_START_AMBIGUOUS",
                                payload={
                                    "recovery_class": "START_AMBIGUOUS",
                                    "round_id": round_row["round_id"],
                                },
                                idempotency_key=(
                                    f"recovery:{round_row['round_id']}:event:ambiguous"
                                ),
                            )
            result = self.close_round(
                CloseRoundCommand(
                    round_id=round_row["round_id"],
                    terminal_class=terminal_class,
                    feedback_payload={
                        "outcome_class": terminal_class,
                        "recovery": "CONSERVATIVE",
                    },
                    controller_state_after_digest=round_row[
                        "controller_state_before_digest"
                    ],
                    resource_debits=(),
                    idempotency_key=(
                        f"{command.idempotency_key}:{round_row['round_id']}:close"
                    ),
                )
            )
            if result["terminal_class"] != terminal_class:
                raise InvariantViolation("recovery close returned a conflicting terminal")
        recovered = [
            {
                "round_id": str(row["round_id"]),
                "terminal_class": str(row["terminal_class"]),
            }
            for row in self._connection.execute(
                """
                SELECT round_id, terminal_class FROM rounds
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                  AND terminal_class IN (
                      'ABORTED_RECOVERY_BEFORE_EXECUTION',
                      'ABORTED_RECOVERY_START_AMBIGUOUS'
                  )
                ORDER BY round_id
                """,
                (
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            )
        ]
        quarantined_temp_files = sorted(
            path.relative_to(self.artifact_root).as_posix()
            for path in self.artifact_root.rglob("*.tmp")
            if path.is_file()
        )
        report = {
            "quarantined_temp_files": quarantined_temp_files,
            "recovered_rounds": recovered,
        }
        with self._transaction() as cursor:
            existing = cursor.execute(
                """
                SELECT recovery_idempotency_key, recovery_payload_digest,
                       recovery_result_json
                FROM triplet_barrier
                WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                """,
                (
                    command.experiment_id,
                    command.search_seed,
                    command.current_round_index,
                ),
            ).fetchone()
            if existing is None:
                raise InvariantViolation("triplet barrier disappeared during recovery")
            if existing["recovery_idempotency_key"] is not None:
                if (
                    existing["recovery_idempotency_key"] != command.idempotency_key
                    or existing["recovery_payload_digest"] != payload_digest
                    or json.loads(existing["recovery_result_json"]) != report
                ):
                    raise IdempotencyConflict("recovery completion conflicts")
            else:
                cursor.execute(
                    """
                    UPDATE triplet_barrier
                    SET recovery_idempotency_key = ?,
                        recovery_payload_digest = ?,
                        recovery_result_json = ?
                    WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
                    """,
                    (
                        command.idempotency_key,
                        payload_digest,
                        self._json(report),
                        command.experiment_id,
                        command.search_seed,
                        command.current_round_index,
                    ),
                )
        return report

    def get_round(
        self, round_id: str, *, cursor: sqlite3.Cursor | None = None
    ) -> dict[str, Any]:
        active = cursor or self._connection.cursor()
        row = active.execute(
            "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        result = _row_dict(row)
        if result is None:
            raise InvariantViolation("unknown round")
        return result

    def get_execution_claim(
        self, round_id: str, *, cursor: sqlite3.Cursor | None = None
    ) -> dict[str, Any]:
        active = cursor or self._connection.cursor()
        row = active.execute(
            "SELECT * FROM execution_claims WHERE round_id = ?", (round_id,)
        ).fetchone()
        result = _row_dict(row)
        if result is None:
            raise InvariantViolation("unknown execution claim")
        return result

    def get_artifact(
        self, artifact_id: str, *, cursor: sqlite3.Cursor | None = None
    ) -> dict[str, Any]:
        active = cursor or self._connection.cursor()
        row = active.execute(
            "SELECT * FROM artifact_index WHERE artifact_id = ?", (artifact_id,)
        ).fetchone()
        result = _row_dict(row)
        if result is None:
            raise InvariantViolation("unknown artifact")
        return result

    def integrity_report(self) -> dict[str, Any]:
        with self._lock:
            tables = sorted(
                row[0]
                for row in self._connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            )
            integrity = self._connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
            foreign_keys = [
                tuple(row)
                for row in self._connection.execute("PRAGMA foreign_key_check")
            ]
            return {
                "foreign_key_violations": foreign_keys,
                "foreign_keys": int(
                    self._connection.execute("PRAGMA foreign_keys").fetchone()[0]
                ),
                "integrity_check": str(integrity),
                "journal_mode": str(
                    self._connection.execute("PRAGMA journal_mode").fetchone()[0]
                ).lower(),
                "migration_sha256": self.migration_sha256,
                "synchronous": int(
                    self._connection.execute("PRAGMA synchronous").fetchone()[0]
                ),
                "tables": tables,
                "user_version": int(
                    self._connection.execute("PRAGMA user_version").fetchone()[0]
                ),
            }
