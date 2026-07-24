"""Closed M6R contracts for the package-owned training runtime release."""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping

from .runtime_contracts import AUTHORITY_FIELDS, ClosedRuntimeRecord


class RuntimeProfileIdV1(str, Enum):
    FAKE_NON_TRAINING = "FAKE_NON_TRAINING_RELEASE_V1"
    PACKAGE_TRAINING = "PACKAGE_OWNED_TRAINING_RELEASE_V1"
    PACKAGE_TRAINING_V2 = "PACKAGE_OWNED_TRAINING_RELEASE_V2"


class TrainingExecutionPurposeV1(str, Enum):
    FIXED_CANARY = "DEVELOPMENT_FIXED_TRAINING_CANARY"
    PILOT = "DEVELOPMENT_PILOT_OFFLINE_TOPN"
    MAIN = "DEVELOPMENT_MAIN_OFFLINE_TOPN"


class TrainingCompatibilityStatusV1(str, Enum):
    COMPATIBLE = "TRAINING_RUNTIME_COMPATIBLE"
    INCOMPATIBLE = "TRAINING_RUNTIME_INCOMPATIBLE"


class TrainingClosureDecisionV1(str, Enum):
    CLOSED = "TRAINING_RESULT_CLOSED"
    REJECTED = "TRAINING_RESULT_REJECTED"


class TrainingRuntimeReleaseV1(ClosedRuntimeRecord):
    record_type = "TrainingRuntimeReleaseV1"
    required_fields = frozenset(
        {
            "allowed_read_roots_policy_digest",
            "allowed_write_roots_policy_digest",
            "backend_identity",
            "candidate_binding_schema_digest",
            "close_contract",
            "close_result_policy_digest",
            "close_result_schema_digest",
            "common_guard_policy_digest",
            "config_schema_digest",
            "confinement_contract",
            "confinement_policy_digest",
            "environment_lock",
            "launch_protocol_id",
            "launcher_abi",
            "launcher_source_digest",
            "metric_contract",
            "metric_parser_digest",
            "package_owned_handler_registry_digest",
            "permit_schema_digest",
            "python_environment_lock_digest",
            "profile_id",
            "read_contract",
            "raw_output_schema_digest",
            "raw_result_schema_digest",
            "recbole_identity_digest",
            "receipt_schema_digest",
            "release_id",
            "resource_contract",
            "resource_accounting_schema_digest",
            "resource_meter_policy_digest",
            "runner_entrypoint_digest",
            "runner_abi",
            "runner_source_digest",
            "runtime_binding_schema_digest",
            "source_manifest",
            "start_confirmation_schema_digest",
            "state_store_claim_schema_digest",
            "supported_execution_purposes",
            "training_config",
            "training_plan_schema_digest",
            "write_contract",
        }
    )


class TrainingRuntimeReleaseV2(ClosedRuntimeRecord):
    record_type = "TrainingRuntimeReleaseV2"
    required_fields = TrainingRuntimeReleaseV1.required_fields | frozenset(
        {
            "filesystem_capability_policy_digest",
            "store_audit_contract_digest",
            "training_profile_digest",
        }
    )


class TrainingRuntimeBindingV1(ClosedRuntimeRecord):
    record_type = "TrainingRuntimeBindingV1"
    required_fields = frozenset(
        {
            "accepted_evidence_eligibility",
            "arm_common_projection_digest",
            "backend_identity_digest",
            "budget_digest",
            "candidate_id",
            "checkpoint_root",
            "confinement_contract_digest",
            "environment_lock_digest",
            "evaluation_purpose",
            "execution_purpose",
            "experiment_id",
            "frontier_eligibility",
            "gpu_cost_ceiling_microunits",
            "gpu_device_time_ceiling_ms",
            "implementation_digest",
            "instance_private_root_digest",
            "lineage_digest",
            "metric_contract_digest",
            "opaque_arm_instance_id",
            "partition_purpose",
            "profile_id",
            "protocol_digest",
            "protocol_profile_ref",
            "read_contract_digest",
            "release_digest",
            "release_id",
            "resource_contract_digest",
            "result_root",
            "round_id",
            "run_id",
            "runner_abi",
            "search_memory_eligibility",
            "seed_policy_digest",
            "source_manifest_digest",
            "training_config_budget_digest",
            "training_config_digest",
            "write_contract_digest",
        }
    )


class TrainingRuntimeBindingV2(ClosedRuntimeRecord):
    record_type = "TrainingRuntimeBindingV2"
    required_fields = TrainingRuntimeBindingV1.required_fields | frozenset(
        {"filesystem_capability_digest"}
    )


class CandidateExecutionBindingV3(ClosedRuntimeRecord):
    record_type = "CandidateExecutionBindingV3"
    required_fields = frozenset(
        {
            "arm_private_root",
            "base_binding_digest",
            "budget_digest",
            "candidate_id",
            "execution_purpose",
            "implementation_digest",
            "mechanism_program_digest",
            "mechanism_semantics_digest",
            "opaque_arm_instance_id",
            "profile_digest",
            "protocol_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "search_seed",
        }
    )


class CommonExecutionPermitV2(ClosedRuntimeRecord):
    record_type = "CommonExecutionPermitV2"
    required_fields = frozenset(
        {
            "backend_digest",
            "base_permit_digest",
            "binding_digest",
            "budget_digest",
            "candidate_id",
            "execution_purpose",
            "ordinary_launch_attempt_ordinal",
            "pre_execution_decision_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "training_plan_decision_digest",
        }
    )


class TrainingRuntimePlanDecisionV1(ClosedRuntimeRecord):
    record_type = "TrainingRuntimePlanDecisionV1"
    required_fields = frozenset(
        {
            "base_plan_decision_digest",
            "decision",
            "execution_purpose",
            "reason_codes",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
        }
    )


class TrainingRuntimeCompatibilityPreflightV1(ClosedRuntimeRecord):
    record_type = "TrainingRuntimeCompatibilityPreflightV1"
    required_fields = frozenset(
        {
            "checked_release_digest",
            "closure_projection_digest",
            "component_checks",
            "expected_runner_abi",
            "failure_codes",
            "fixture_digest",
            "profile_id",
            "runtime_binding_digest",
            "status",
        }
    )


class TrainingRuntimeCompatibilityFixtureV1(ClosedRuntimeRecord):
    """Exact campaign-to-close inputs checked before any proposal broker call."""

    record_type = "TrainingRuntimeCompatibilityFixtureV1"
    required_fields = frozenset(
        {
            "accepted_evidence_eligibility",
            "arm_common_projection_digest",
            "budget_digest",
            "candidate_id",
            "checkpoint_root",
            "component_runner_abis",
            "evaluation_purpose",
            "execution_purpose",
            "experiment_id",
            "frontier_eligibility",
            "gpu_cost_ceiling_microunits",
            "gpu_device_time_ceiling_ms",
            "implementation_digest",
            "instance_private_root",
            "lineage_digest",
            "opaque_arm_instance_id",
            "partition_purpose",
            "protocol_digest",
            "protocol_profile_ref",
            "result_root",
            "round_id",
            "run_id",
            "search_memory_eligibility",
            "seed_policy_digest",
            "training_config_budget_digest",
        }
    )


class ExecutionStartConfirmationV1(ClosedRuntimeRecord):
    record_type = "ExecutionStartConfirmationV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "claim_id",
            "execution_purpose",
            "ordinary_launch_attempt_ordinal",
            "permit_digest",
            "pid",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "start_status",
        }
    )


class ExecutionStartReceiptV2(ClosedRuntimeRecord):
    record_type = "ExecutionStartReceiptV2"
    required_fields = frozenset(
        {
            "binding_digest",
            "claim_id",
            "execution_purpose",
            "ordinary_launch_attempt_ordinal",
            "permit_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "start_confirmation_digest",
            "start_status",
        }
    )


class TrainingRawRunOutputV1(ClosedRuntimeRecord):
    record_type = "TrainingRawRunOutputV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "budget_digest",
            "candidate_id",
            "epochs_requested",
            "environment_lock_digest",
            "execution_purpose",
            "experiment_id",
            "exit_status",
            "gpu_cost_microunits",
            "gpu_device_time_ms",
            "launcher_return_code",
            "lineage_digest",
            "metric_contract_digest",
            "model",
            "normalized_metrics",
            "permit_digest",
            "partition_purpose",
            "protocol_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "seed",
            "training_backend_started",
            "termination_class",
            "training_config_budget_digest",
            "wall_time_ms",
            "worker_result_digest",
        }
    )


class TrainingRawRunOutputV2(ClosedRuntimeRecord):
    record_type = "TrainingRawRunOutputV2"
    required_fields = TrainingRawRunOutputV1.required_fields | frozenset(
        {
            "filesystem_capability_digest",
            "filesystem_confinement_status",
            "side_effect_audit_digest",
        }
    )


class TrainingResourceAccountingV1(ClosedRuntimeRecord):
    record_type = "TrainingResourceAccountingV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "claim_id",
            "debits",
            "permit_digest",
            "raw_output_digest",
            "round_id",
            "run_id",
            "runtime_binding_digest",
            "runtime_release_digest",
        }
    )


class CommonResultClosureV2(ClosedRuntimeRecord):
    record_type = "CommonResultClosureV2"
    required_fields = frozenset(
        {
            "binding_digest",
            "claim_id",
            "decision",
            "execution_purpose",
            "permit_digest",
            "raw_output_digest",
            "reason_codes",
            "resource_accounting_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "start_receipt_digest",
            "subchecks",
        }
    )


class RawResultEnvelopeV2(ClosedRuntimeRecord):
    record_type = "RawResultEnvelopeV2"
    required_fields = frozenset(
        {
            "artifact_closure",
            "binding_digest",
            "candidate_id",
            "common_result_closure_digest",
            "evaluation_purpose",
            "exit_status",
            "metric_source",
            "normalized_metrics",
            "ordinary_execution_start_index",
            "partition_role",
            "raw_output_digest",
            "resource_accounting_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_binding_digest",
            "runtime_release_digest",
            "seed",
        }
    )
    optional_fields = frozenset(AUTHORITY_FIELDS)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        super().__init__({**AUTHORITY_FIELDS, **dict(payload)})


__all__ = [
    "CandidateExecutionBindingV3",
    "CommonExecutionPermitV2",
    "CommonResultClosureV2",
    "ExecutionStartConfirmationV1",
    "ExecutionStartReceiptV2",
    "RawResultEnvelopeV2",
    "RuntimeProfileIdV1",
    "TrainingClosureDecisionV1",
    "TrainingCompatibilityStatusV1",
    "TrainingExecutionPurposeV1",
    "TrainingRawRunOutputV1",
    "TrainingRawRunOutputV2",
    "TrainingResourceAccountingV1",
    "TrainingRuntimeBindingV1",
    "TrainingRuntimeBindingV2",
    "TrainingRuntimeCompatibilityFixtureV1",
    "TrainingRuntimeCompatibilityPreflightV1",
    "TrainingRuntimePlanDecisionV1",
    "TrainingRuntimeReleaseV1",
    "TrainingRuntimeReleaseV2",
]
