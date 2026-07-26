"""Exactly-once SearchRound closure for terminal Broker process failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .broker_process import BrokerCallOutcomeV2, BrokerProcessExitReceiptV2
from .canonical import canonical_json_bytes, sha256_digest
from .contracts import ResourceCeilingsV1
from .state_store import (
    IdempotencyConflict,
    InvariantViolation,
    RegisterArtifactCommand,
    ResourceDebitV1,
    SingleWriterExperimentStoreV1,
    StopAndFillCommand,
)


@dataclass(frozen=True, slots=True)
class BrokerFailureClosureV1:
    round_id: str
    receipt_digest: str
    broker_outcome_digest: str
    failure_class: str
    classifier_rule_id: str
    proposal_generation_session_consumed: bool
    proposal_response_present: bool
    execution_claim_present: bool
    training_started: bool
    guard_called: bool
    search_memory_updated: bool
    frontier_updated: bool
    retry_count: int
    refund_applied: bool
    fallback_used: bool
    physical_call_count: int
    input_token_debit: int
    output_token_debit: int
    billed_token_debit: int
    wall_time_ms: int
    round_terminal_class: str
    feedback_class: str
    stdout_sha256: str
    stdout_size_bytes: int
    stdout_truncated: bool
    stderr_sha256: str
    stderr_size_bytes: int
    stderr_truncated: bool
    redacted_excerpt: str | None
    closure_digest: str

    @classmethod
    def create(
        cls,
        *,
        round_id: str,
        receipt: BrokerProcessExitReceiptV2,
        outcome: BrokerCallOutcomeV2,
        physical_call_count: int,
        input_tokens: int,
        output_tokens: int,
        billed_tokens: int,
        wall_time_ms: int,
    ) -> "BrokerFailureClosureV1":
        if outcome.status != "PROCESS_FAILURE":
            raise ValueError("Broker failure closure requires a process failure")
        payload = {
            "broker_outcome_digest": outcome.outcome_digest,
            "classifier_rule_id": outcome.classifier_rule_id,
            "execution_claim_present": False,
            "failure_class": outcome.failure_class,
            "fallback_used": False,
            "feedback_class": "COMMON_NO_EXECUTION",
            "frontier_updated": False,
            "guard_called": False,
            "proposal_generation_session_consumed": True,
            "proposal_response_present": False,
            "physical_call_count": physical_call_count,
            "input_token_debit": input_tokens,
            "output_token_debit": output_tokens,
            "billed_token_debit": billed_tokens,
            "wall_time_ms": wall_time_ms,
            "receipt_digest": receipt.receipt_digest,
            "redacted_excerpt": outcome.redacted_excerpt,
            "refund_applied": False,
            "retry_count": 0,
            "round_id": round_id,
            "round_terminal_class": "BROKER_PROCESS_FAILURE",
            "search_memory_updated": False,
            "stderr_sha256": receipt.stderr_sha256,
            "stderr_size_bytes": receipt.stderr_size_bytes,
            "stderr_truncated": receipt.stderr_truncated,
            "stdout_sha256": receipt.stdout_sha256,
            "stdout_size_bytes": receipt.stdout_size_bytes,
            "stdout_truncated": receipt.stdout_truncated,
            "training_started": False,
        }
        return cls(**payload, closure_digest=sha256_digest(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def close_broker_failure(
    *,
    store: SingleWriterExperimentStoreV1,
    experiment_id: str,
    search_seed: int,
    round_index: int,
    round_id: str,
    controller_state_digest: str,
    ceilings: ResourceCeilingsV1,
    receipt: BrokerProcessExitReceiptV2,
    outcome: BrokerCallOutcomeV2,
    physical_call_count: int,
    input_tokens: int,
    output_tokens: int,
    billed_tokens: int,
    wall_time_ms: int,
) -> BrokerFailureClosureV1:
    """Persist and replay one fail-closed Broker terminal transaction."""

    closure = BrokerFailureClosureV1.create(
        round_id=round_id,
        receipt=receipt,
        outcome=outcome,
        physical_call_count=physical_call_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        billed_tokens=billed_tokens,
        wall_time_ms=wall_time_ms,
    )
    artifact_path = (
        "broker_failures/"
        + sha256_digest({"round_id": round_id})
        + "/BROKER_FAILURE_CLOSURE_V1.json"
    )
    store.register_artifact(
        RegisterArtifactCommand(
            round_id=round_id,
            artifact_type="BROKER_FAILURE_CLOSURE_V1",
            relative_path=artifact_path,
            producer="M6F_BROKER_FAILURE_CLOSER_V1",
            idempotency_key=f"m6f:broker-failure-artifact:{round_id}",
        ),
        canonical_json_bytes(closure.to_dict()) + b"\n",
    )
    store.stop_and_fill_remaining(
        StopAndFillCommand(
            experiment_id=experiment_id,
            search_seed=search_seed,
            current_round_index=round_index,
            reason="BROKER_PROCESS_FAILURE",
            idempotency_key=f"m6f:broker-failure-stop:{search_seed}:{round_index}",
        )
    )
    _commit_broker_failure_terminal(
        store=store,
        round_id=round_id,
        controller_state_digest=controller_state_digest,
        feedback_payload={
            "broker_failure_closure_digest": closure.closure_digest,
            "feedback_class": "COMMON_NO_EXECUTION",
            "failure_class": closure.failure_class,
            "retry_count": 0,
        },
        resource_debits=(
            ResourceDebitV1("PHYSICAL_LLM_CALL", physical_call_count),
            ResourceDebitV1(
                "PROPOSAL_ATTEMPT", ceilings.proposal_attempt_debit
            ),
            ResourceDebitV1("INPUT_TOKEN", input_tokens),
            ResourceDebitV1("OUTPUT_TOKEN", output_tokens),
            ResourceDebitV1("BILLED_TOKEN_DEBIT", billed_tokens),
            ResourceDebitV1("ORDINARY_EXECUTION", 0),
            ResourceDebitV1("GPU_DEVICE_TIME_MS", 0),
            ResourceDebitV1("GPU_COST_MICROUNITS", 0),
            ResourceDebitV1("WALL_TIME_MS", wall_time_ms),
            ResourceDebitV1("RETRY", 0),
        ),
        idempotency_key=f"m6f:broker-failure-close:{round_id}",
    )
    return closure


def _commit_broker_failure_terminal(
    *,
    store: SingleWriterExperimentStoreV1,
    round_id: str,
    controller_state_digest: str,
    feedback_payload: dict[str, Any],
    resource_debits: tuple[ResourceDebitV1, ...],
    idempotency_key: str,
) -> dict[str, Any]:
    """M6F additive terminal transaction without changing the frozen M0 store."""

    payload = {
        "controller_state_after_digest": controller_state_digest,
        "feedback_digest": sha256_digest(feedback_payload),
        "resource_debits": [
            {"dimension": item.dimension, "quantity": item.quantity}
            for item in resource_debits
        ],
        "round_id": round_id,
        "terminal_class": "BROKER_PROCESS_FAILURE",
    }
    payload_digest = sha256_digest(payload)
    with store._transaction() as cursor:
        prior = store._check_idempotency(
            cursor,
            key=idempotency_key,
            operation="close_round",
            payload_digest=payload_digest,
        )
        if prior is not None:
            return store.get_round(prior, cursor=cursor)
        round_row = cursor.execute(
            "SELECT * FROM rounds WHERE round_id = ?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise InvariantViolation("unknown round")
        if round_row["status"] != "OPEN":
            raise IdempotencyConflict(
                "round is already terminal under a different command"
            )
        for index, debit in enumerate(resource_debits):
            store._append_resource(
                cursor,
                round_id=round_id,
                debit=debit,
                idempotency_key=f"{idempotency_key}:resource:{index}",
            )
        store._append_event(
            cursor,
            round_id=round_id,
            event_type="ROUND_FEEDBACK",
            payload={
                "feedback": feedback_payload,
                "round_id": round_id,
                "terminal_class": "BROKER_PROCESS_FAILURE",
            },
            idempotency_key=f"{idempotency_key}:feedback",
        )
        store._append_event(
            cursor,
            round_id=round_id,
            event_type="ROUND_CLOSED",
            payload=payload,
            idempotency_key=idempotency_key,
        )
        cursor.execute(
            """
            UPDATE rounds
            SET status = 'CLOSED', terminal_class = ?,
                feedback_digest = ?, controller_state_after_digest = ?
            WHERE round_id = ? AND status = 'OPEN'
            """,
            (
                "BROKER_PROCESS_FAILURE",
                sha256_digest(feedback_payload),
                controller_state_digest,
                round_id,
            ),
        )
        cursor.execute(
            """
            UPDATE scheduled_slots SET slot_status = 'CLOSED'
            WHERE round_id = ? AND slot_status = 'OPENED'
            """,
            (round_id,),
        )
        bit = {"A": 1, "B": 2, "C": 4}[str(round_row["arm_code"])]
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
        cursor.execute(
            """
            UPDATE triplet_barrier
            SET closed_bitmap = ?, next_index_authorized = 0
            WHERE experiment_id = ? AND search_seed = ? AND round_index = ?
            """,
            (
                bitmap,
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
                controller_state_digest,
                round_row["experiment_id"],
                round_row["arm_instance_id"],
                round_row["search_seed"],
            ),
        )
        open_count = int(
            cursor.execute(
                """
                SELECT COUNT(*) FROM rounds
                WHERE experiment_id = ? AND search_seed = ? AND status = 'OPEN'
                """,
                (round_row["experiment_id"], round_row["search_seed"]),
            ).fetchone()[0]
        )
        if open_count == 0:
            cursor.execute(
                """
                UPDATE arm_state SET state = 'STOPPED', revision = revision + 1
                WHERE experiment_id = ? AND search_seed = ?
                """,
                (round_row["experiment_id"], round_row["search_seed"]),
            )
        return store.get_round(round_id, cursor=cursor)


__all__ = ["BrokerFailureClosureV1", "close_broker_failure"]
