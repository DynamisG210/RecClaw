"""Common release closure for package-owned training in every experiment Arm."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .canonical import bytes_sha256, canonical_json_bytes
from .runtime_contracts import (
    CandidateExecutionBindingV2,
    CommonExecutionPermitV1,
    CommonPlanDecisionV1,
    CommonDecision,
)
from .training_materialization import verify_training_binding_v3
from .training_runtime_contracts import (
    CandidateExecutionBindingV3,
    CommonExecutionPermitV2,
    CommonResultClosureV2,
    ExecutionStartConfirmationV1,
    ExecutionStartReceiptV2,
    RawResultEnvelopeV2,
    TrainingClosureDecisionV1,
    TrainingRawRunOutputV2,
    TrainingResourceAccountingV1,
    TrainingRuntimeBindingV2,
    TrainingRuntimePlanDecisionV1,
)
from .training_runtime_release import (
    CAMPAIGN_TRAINING_RUNNER_ABI,
    TRAINING_RUNNER_ABI,
    resolve_runtime_release,
    training_release_for_abi,
)


def _subcheck(name: str, verified: bool) -> dict[str, str]:
    return {
        "name": name,
        "status": "VERIFIED" if verified else "REJECTED",
    }


class CommonTrainingExecutionGuardV1:
    """Mechanical training guard with no Arm or Evidence Guard branch."""

    def plan_check(
        self,
        *,
        base_plan: CommonPlanDecisionV1,
        runtime_binding: TrainingRuntimeBindingV2,
    ) -> TrainingRuntimePlanDecisionV1:
        failures: list[str] = []
        try:
            resolved = resolve_runtime_release(runtime_binding.runner_abi)
        except ValueError:
            resolved = None
            failures.append("TRAINING_PLAN_UNKNOWN_RUNTIME_RELEASE")
        release = training_release_for_abi(runtime_binding.runner_abi)
        if (
            base_plan.decision != CommonDecision.PASS.value
            or resolved is None
            or resolved["release_digest"] != release.digest
            or runtime_binding.release_digest != release.digest
            or runtime_binding.runner_abi
            not in {
                TRAINING_RUNNER_ABI,
                CAMPAIGN_TRAINING_RUNNER_ABI,
            }
            or runtime_binding.execution_purpose
            not in set(release.supported_execution_purposes)
        ):
            failures.append("TRAINING_PLAN_RELEASE_CLOSURE_MISMATCH")
        return TrainingRuntimePlanDecisionV1(
            {
                "base_plan_decision_digest": base_plan.digest,
                "decision": (
                    CommonDecision.PASS.value
                    if not failures
                    else CommonDecision.DENY.value
                ),
                "execution_purpose": runtime_binding.execution_purpose,
                "reason_codes": sorted(set(failures)),
                "runner_abi": runtime_binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": runtime_binding.release_digest,
            }
        )

    def pre_execute(
        self,
        *,
        base_permit: CommonExecutionPermitV1,
        base_binding: CandidateExecutionBindingV2,
        binding: CandidateExecutionBindingV3,
        runtime_binding: TrainingRuntimeBindingV2,
        training_plan: TrainingRuntimePlanDecisionV1,
    ) -> CommonExecutionPermitV2:
        valid_binding, reasons = verify_training_binding_v3(
            binding,
            base_binding=base_binding,
            runtime_binding=runtime_binding,
        )
        resolved = resolve_runtime_release(binding.runner_abi)
        release = training_release_for_abi(binding.runner_abi)
        valid = (
            valid_binding
            and base_permit.binding_digest == base_binding.digest
            and base_permit.budget_digest == binding.budget_digest
            and binding.runtime_binding_digest == runtime_binding.digest
            and binding.runtime_release_digest == release.digest
            and runtime_binding.release_digest == release.digest
            and resolved["release_digest"] == release.digest
            and binding.runner_abi
            in {TRAINING_RUNNER_ABI, CAMPAIGN_TRAINING_RUNNER_ABI}
            and training_plan.decision == CommonDecision.PASS.value
            and training_plan.runtime_binding_digest == runtime_binding.digest
            and training_plan.runtime_release_digest == release.digest
        )
        if not valid:
            detail = ",".join(reasons) or "TRAINING_RELEASE_CLOSURE_MISMATCH"
            raise ValueError(detail)
        return CommonExecutionPermitV2(
            {
                "backend_digest": runtime_binding.backend_identity_digest,
                "base_permit_digest": base_permit.digest,
                "binding_digest": binding.digest,
                "budget_digest": binding.budget_digest,
                "candidate_id": binding.candidate_id,
                "execution_purpose": binding.execution_purpose,
                "ordinary_launch_attempt_ordinal": 1,
                "pre_execution_decision_digest": (
                    base_permit.pre_execution_decision_digest
                ),
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": release.digest,
                "training_plan_decision_digest": training_plan.digest,
            }
        )

    def close_result(
        self,
        *,
        permit: CommonExecutionPermitV2,
        binding: CandidateExecutionBindingV3,
        runtime_binding: TrainingRuntimeBindingV2,
        claim: Mapping[str, Any],
        confirmation: ExecutionStartConfirmationV1,
        receipt: ExecutionStartReceiptV2,
        raw_output: TrainingRawRunOutputV2,
        resource_accounting: TrainingResourceAccountingV1,
        artifact_closure: list[Mapping[str, Any]],
        seed: int,
    ) -> tuple[CommonResultClosureV2, RawResultEnvelopeV2 | None]:
        failures: list[str] = []
        subchecks: list[dict[str, str]] = []
        release = training_release_for_abi(binding.runner_abi)

        exact = {
            "binding_digest": binding.digest,
            "execution_purpose": binding.execution_purpose,
            "permit_digest": permit.digest,
            "round_id": binding.round_id,
            "runner_abi": binding.runner_abi,
            "runtime_binding_digest": runtime_binding.digest,
            "runtime_release_digest": release.digest,
        }
        permit_ok = (
            permit.binding_digest == binding.digest
            and permit.runtime_binding_digest == runtime_binding.digest
            and permit.runtime_release_digest == release.digest
            and permit.runner_abi
            in {TRAINING_RUNNER_ABI, CAMPAIGN_TRAINING_RUNNER_ABI}
            and permit.execution_purpose == binding.execution_purpose
        )
        subchecks.append(_subcheck("PERMIT_RELEASE", permit_ok))
        if not permit_ok:
            failures.append("TRAINING_PERMIT_RELEASE_MISMATCH")

        claim_ok = (
            all(claim.get(key) == value for key, value in exact.items())
            and claim.get("claim_state") == "FINISHED"
            and claim.get("attempt_state") == "START_CONFIRMED"
            and claim.get("execution_debited") == 1
        )
        subchecks.append(_subcheck("CLAIM_RELEASE", claim_ok))
        if not claim_ok:
            failures.append("TRAINING_CLAIM_RELEASE_MISMATCH")

        confirmation_ok = (
            all(getattr(confirmation, key) == value for key, value in exact.items())
            and confirmation.claim_id == claim.get("claim_id")
            and confirmation.start_status == "START_CONFIRMED"
            and confirmation.ordinary_launch_attempt_ordinal == 1
        )
        receipt_ok = (
            all(getattr(receipt, key) == value for key, value in exact.items())
            and receipt.claim_id == claim.get("claim_id")
            and receipt.start_confirmation_digest == confirmation.digest
            and receipt.start_status == "STARTED"
            and receipt.ordinary_launch_attempt_ordinal == 1
        )
        subchecks.append(
            _subcheck("CONFIRMED_START_RECEIPT", confirmation_ok and receipt_ok)
        )
        if not confirmation_ok or not receipt_ok:
            failures.append("TRAINING_START_ORDER_MISMATCH")

        raw_ok = (
            all(getattr(raw_output, key) == value for key, value in exact.items())
            and seed == int(binding.search_seed)
            and raw_output.budget_digest == binding.budget_digest
            and raw_output.candidate_id == binding.candidate_id
            and raw_output.experiment_id == runtime_binding.experiment_id
            and raw_output.training_backend_started is True
            and raw_output.environment_lock_digest
            == runtime_binding.environment_lock_digest
            and raw_output.lineage_digest == runtime_binding.lineage_digest
            and raw_output.metric_contract_digest
            == runtime_binding.metric_contract_digest
            and raw_output.partition_purpose
            == runtime_binding.partition_purpose
            and raw_output.protocol_digest == runtime_binding.protocol_digest
            and raw_output.seed == int(binding.search_seed)
            and raw_output.training_config_budget_digest
            == runtime_binding.training_config_budget_digest
            and raw_output.filesystem_capability_digest
            == runtime_binding.filesystem_capability_digest
            and raw_output.filesystem_confinement_status == "PASS"
            and raw_output.gpu_device_time_ms >= 0
            and raw_output.gpu_cost_microunits >= 0
            and raw_output.wall_time_ms >= 0
            and all(
                math.isfinite(float(value))
                for value in raw_output.normalized_metrics.values()
            )
            and (
                (
                    raw_output.termination_class == "SUCCESS"
                    and raw_output.exit_status == "SUCCESS"
                    and raw_output.launcher_return_code == 0
                    and "ndcg" in raw_output.normalized_metrics
                )
                or (
                    raw_output.termination_class == "TIMEOUT"
                    and raw_output.exit_status == "RUNTIME_FAILURE"
                    and raw_output.launcher_return_code == 124
                )
                or (
                    raw_output.termination_class == "CRASH_OR_RUNTIME_FAILURE"
                    and raw_output.exit_status == "RUNTIME_FAILURE"
                    and raw_output.launcher_return_code != 0
                    and raw_output.launcher_return_code != 124
                )
            )
        )
        subchecks.append(_subcheck("RAW_RESULT_RELEASE", raw_ok))
        if not raw_ok:
            failures.append("TRAINING_RAW_RESULT_MISMATCH")
        resource_ceiling_ok = (
            raw_output.gpu_device_time_ms
            <= runtime_binding.gpu_device_time_ceiling_ms
            and raw_output.gpu_cost_microunits
            <= runtime_binding.gpu_cost_ceiling_microunits
        )
        subchecks.append(
            _subcheck("RESOURCE_CEILING", resource_ceiling_ok)
        )
        if not resource_ceiling_ok:
            failures.append("TRAINING_RESOURCE_CEILING_EXCEEDED")

        expected_debits = [
            {"dimension": "GPU_COST_MICROUNITS", "quantity": raw_output.gpu_cost_microunits},
            {"dimension": "GPU_DEVICE_TIME_MS", "quantity": raw_output.gpu_device_time_ms},
            {"dimension": "ORDINARY_EXECUTION", "quantity": 1},
            {"dimension": "WALL_TIME_MS", "quantity": raw_output.wall_time_ms},
        ]
        accounting_ok = (
            resource_accounting.binding_digest == binding.digest
            and resource_accounting.claim_id == claim.get("claim_id")
            and resource_accounting.permit_digest == permit.digest
            and resource_accounting.raw_output_digest == raw_output.digest
            and resource_accounting.runtime_binding_digest == runtime_binding.digest
            and resource_accounting.runtime_release_digest == release.digest
            and [dict(item) for item in resource_accounting.debits]
            == expected_debits
        )
        subchecks.append(_subcheck("RESOURCE_RELEASE", accounting_ok))
        if not accounting_ok:
            failures.append("TRAINING_RESOURCE_ACCOUNTING_MISMATCH")

        expected_artifacts = {
            "EXECUTION_START_CONFIRMATION_V1": confirmation,
            "EXECUTION_START_RECEIPT_V2": receipt,
            "TRAINING_RAW_RUN_OUTPUT_V2": raw_output,
            "TRAINING_RESOURCE_ACCOUNTING_V1": resource_accounting,
        }
        artifact_ok = True
        for artifact_type, record in expected_artifacts.items():
            rows = [
                row
                for row in artifact_closure
                if row.get("artifact_type") == artifact_type
            ]
            artifact_ok = (
                artifact_ok
                and len(rows) == 1
                and rows[0].get("round_id") == binding.round_id
                and rows[0].get("sha256")
                == bytes_sha256(canonical_json_bytes(record.to_dict()))
            )
        subchecks.append(_subcheck("ARTIFACT_CLOSURE", artifact_ok))
        if not artifact_ok:
            failures.append("TRAINING_ARTIFACT_CLOSURE_MISMATCH")

        closure = CommonResultClosureV2(
            {
                "binding_digest": binding.digest,
                "claim_id": str(claim.get("claim_id") or ""),
                "decision": (
                    TrainingClosureDecisionV1.CLOSED.value
                    if not failures
                    else TrainingClosureDecisionV1.REJECTED.value
                ),
                "execution_purpose": binding.execution_purpose,
                "permit_digest": permit.digest,
                "raw_output_digest": raw_output.digest,
                "reason_codes": sorted(failures),
                "resource_accounting_digest": resource_accounting.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": release.digest,
                "start_receipt_digest": receipt.digest,
                "subchecks": subchecks,
            }
        )
        if failures == ["TRAINING_RESOURCE_CEILING_EXCEEDED"]:
            envelope = RawResultEnvelopeV2(
                {
                    "artifact_closure": artifact_closure,
                    "binding_digest": binding.digest,
                    "candidate_id": binding.candidate_id,
                    "common_result_closure_digest": closure.digest,
                    "evaluation_purpose": binding.execution_purpose,
                    "exit_status": "COMMON_EXECUTION_FAILURE",
                    "metric_source": "NOT_ADMITTED_RESOURCE_CEILING",
                    "normalized_metrics": {},
                    "ordinary_execution_start_index": 1,
                    "partition_role": runtime_binding.partition_purpose,
                    "raw_output_digest": raw_output.digest,
                    "resource_accounting_digest": (
                        resource_accounting.digest
                    ),
                    "round_id": binding.round_id,
                    "run_id": binding.run_id,
                    "runner_abi": binding.runner_abi,
                    "runtime_binding_digest": runtime_binding.digest,
                    "runtime_release_digest": release.digest,
                    "seed": seed,
                }
            )
            return closure, envelope
        if failures:
            return closure, None
        envelope = RawResultEnvelopeV2(
            {
                "artifact_closure": artifact_closure,
                "binding_digest": binding.digest,
                "candidate_id": binding.candidate_id,
                "common_result_closure_digest": closure.digest,
                "evaluation_purpose": binding.execution_purpose,
                "exit_status": raw_output.exit_status,
                "metric_source": "RECBOLE_FULL_SORT_TEST_RESULT",
                "normalized_metrics": raw_output.normalized_metrics,
                "ordinary_execution_start_index": 1,
                "partition_role": runtime_binding.partition_purpose,
                "raw_output_digest": raw_output.digest,
                "resource_accounting_digest": resource_accounting.digest,
                "round_id": binding.round_id,
                "run_id": binding.run_id,
                "runner_abi": binding.runner_abi,
                "runtime_binding_digest": runtime_binding.digest,
                "runtime_release_digest": release.digest,
                "seed": seed,
            }
        )
        return closure, envelope


__all__ = ["CommonTrainingExecutionGuardV1"]
