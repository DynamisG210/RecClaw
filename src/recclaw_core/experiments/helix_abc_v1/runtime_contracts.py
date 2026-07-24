"""Closed M1 contracts for the common executable vertical slice.

These records are development-only mechanical contracts. They do not decide
evidence admission, claim state, or scientific authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar

from .canonical import canonical_value, sha256_digest, validate_sha256


AUTHORITY_FIELDS = {
    "authority": "NONE",
    "evidence_class": "DEVELOPMENT_ONLY",
    "formal_acceptance": False,
}

_FORBIDDEN_FIELD_TOKENS = frozenset(
    {
        "admissible",
        "claimceiling",
        "evidenceadmission",
        "evidenceuse",
        "metaupdate",
        "producerquota",
        "producerreward",
        "protocolbranch",
        "routerscore",
    }
)


class CommonDecision(str, Enum):
    PASS = "COMMON_PASS"
    DENY = "COMMON_DENY"
    INCONCLUSIVE = "COMMON_INCONCLUSIVE"


class TrustClass(str, Enum):
    PACKAGE_OWNED_TYPED_TEMPLATE = "PACKAGE_OWNED_TYPED_TEMPLATE"
    CANDIDATE_CONTROLLED_EXECUTABLE = "CANDIDATE_CONTROLLED_EXECUTABLE"
    INCONCLUSIVE = "INCONCLUSIVE"


class GateStatus(str, Enum):
    ALLOW = "ALLOW_DEVELOPMENT_FAKE_RUN"
    DENY = "DENY"
    INCONCLUSIVE = "INCONCLUSIVE"


class StartStatus(str, Enum):
    STARTED = "STARTED"
    START_AMBIGUOUS = "START_AMBIGUOUS"
    NOT_STARTED = "NOT_STARTED"


def _field_token(name: str) -> str:
    return "".join(character.lower() for character in name if character.isalnum())


def _assert_no_forbidden_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _field_token(str(key)) in _FORBIDDEN_FIELD_TOKENS:
                raise ValueError(f"Evidence/Search authority field is forbidden: {key}")
            _assert_no_forbidden_fields(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_forbidden_fields(item)


def _freeze(value: Any) -> Any:
    normalized = canonical_value(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze(item) for item in normalized)
    return normalized


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


class ClosedRuntimeRecord:
    """Deeply immutable, closed-field record with a content identity."""

    __slots__ = ("_payload",)

    record_type: ClassVar[str]
    required_fields: ClassVar[frozenset[str]]
    optional_fields: ClassVar[frozenset[str]] = frozenset()
    allow_authority_terms: ClassVar[bool] = False

    def __init__(self, payload: Mapping[str, Any]) -> None:
        raw = dict(payload)
        expected = self.required_fields | self.optional_fields
        if set(raw) - expected:
            raise ValueError(
                f"{self.record_type} has unknown fields: {sorted(set(raw) - expected)}"
            )
        missing = self.required_fields - set(raw)
        if missing:
            raise ValueError(f"{self.record_type} is missing fields: {sorted(missing)}")
        if not self.allow_authority_terms:
            _assert_no_forbidden_fields(raw)
        object.__setattr__(self, "_payload", _freeze(raw))

    def __setattr__(self, _name: str, _value: Any) -> None:
        raise AttributeError(f"{self.record_type} is immutable")

    def __getattr__(self, name: str) -> Any:
        try:
            return self._payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def to_dict(self) -> dict[str, Any]:
        return _thaw(self._payload)

    @property
    def digest(self) -> str:
        return sha256_digest({"record_type": self.record_type, **self.to_dict()})


class DevelopmentRecSysProtocolV1(ClosedRuntimeRecord):
    record_type = "DevelopmentRecSysProtocolV1"
    required_fields = frozenset(
        {
            "aggregation",
            "candidate_universe",
            "comparator_identity",
            "dataset_identity",
            "dataset_snapshot_digest",
            "derivation_read_scope",
            "estimand",
            "evaluation_candidate_mode",
            "feature_as_of",
            "fit_scope",
            "item_availability",
            "metric",
            "nonfinite_policy",
            "partition_role",
            "population",
            "preprocessing",
            "profile_family",
            "protocol_id",
            "repeat_policy",
            "short_list_policy",
            "split",
            "tie_policy",
            "training_negative_sampling",
            "transitive_reads",
            "zero_relevant_policy",
        }
    )


class CommonPlanDecisionV1(ClosedRuntimeRecord):
    record_type = "CommonPlanDecisionV1"
    required_fields = frozenset(
        {
            "candidate_id",
            "compile_report_digest",
            "decision",
            "mechanism_program_digest",
            "mechanism_semantics_digest",
            "profile_digest",
            "protocol_digest",
            "reason_codes",
            "release_projection_digest",
            "subchecks",
        }
    )


_ELIGIBLE_ACTION_TOKEN = object()


class CommonEligibleActionV1(ClosedRuntimeRecord):
    record_type = "CommonEligibleActionV1"
    required_fields = frozenset(
        {
            "budget_digest",
            "candidate_id",
            "compile_projection",
            "compile_report_digest",
            "plan_decision_digest",
            "program_digest",
            "protocol_digest",
            "release_projection_digest",
        }
    )

    def __init__(self, payload: Mapping[str, Any], *, _token: object | None = None) -> None:
        if _token is not _ELIGIBLE_ACTION_TOKEN:
            raise ValueError("only CommonExecutionGuardV1.plan_check may construct this type")
        super().__init__(payload)


def create_common_eligible_action(payload: Mapping[str, Any]) -> CommonEligibleActionV1:
    return CommonEligibleActionV1(payload, _token=_ELIGIBLE_ACTION_TOKEN)


class MaterializationReportV1(ClosedRuntimeRecord):
    record_type = "MaterializationReportV1"
    required_fields = frozenset(
        {
            "candidate_id",
            "campaign_profile_digest",
            "compile_report_digest",
            "dependencies",
            "diagnostics",
            "entrypoint",
            "files",
            "implementation_digest",
            "materializer_digest",
            "mechanism_program_digest",
            "mechanism_semantics_digest",
            "required_capabilities",
            "runner_abi",
            "runtime_release_digest",
            "status",
            "template_id",
        }
    )


class ExecutionTrustClassificationV1(ClosedRuntimeRecord):
    record_type = "ExecutionTrustClassificationV1"
    required_fields = frozenset(
        {
            "classification",
            "classifier_digest",
            "implementation_digest",
            "materialization_digest",
            "reason_codes",
        }
    )


class CandidateExecutionBindingV2(ClosedRuntimeRecord):
    record_type = "CandidateExecutionBindingV2"
    required_fields = frozenset(
        {
            "arm_private_root",
            "budget_digest",
            "candidate_id",
            "implementation_digest",
            "materialization_digest",
            "mechanism_program_digest",
            "mechanism_semantics_digest",
            "opaque_arm_instance_id",
            "profile_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "runtime_release_digest",
            "search_seed",
            "trust_classification_digest",
        }
    )


class DevelopmentExecutionGateDecisionV1(ClosedRuntimeRecord):
    record_type = "DevelopmentExecutionGateDecisionV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "decision",
            "gate_source_digest",
            "materialization_digest",
            "reason_codes",
            "runtime_release_digest",
            "task_authorization_ref",
            "trust_classification_digest",
        }
    )


class CommonPreExecutionDecisionV1(ClosedRuntimeRecord):
    record_type = "CommonPreExecutionDecisionV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "decision",
            "gate_decision_digest",
            "materialization_digest",
            "plan_decision_digest",
            "reason_codes",
            "release_projection_digest",
            "subchecks",
        }
    )


class CommonExecutionPermitV1(ClosedRuntimeRecord):
    record_type = "CommonExecutionPermitV1"
    required_fields = frozenset(
        {
            "backend_digest",
            "binding_digest",
            "budget_digest",
            "candidate_id",
            "gate_decision_digest",
            "ordinary_launch_attempt_ordinal",
            "pre_execution_decision_digest",
            "round_id",
            "run_id",
            "runner_abi",
        }
    )


class ExecutionStartReceiptV1(ClosedRuntimeRecord):
    record_type = "ExecutionStartReceiptV1"
    required_fields = frozenset(
        {
            "binding_digest",
            "claim_id",
            "ordinary_launch_attempt_ordinal",
            "permit_digest",
            "round_id",
            "run_id",
            "runner_abi",
            "start_status",
        }
    )


class RawRunOutputV1(ClosedRuntimeRecord):
    record_type = "RawRunOutputV1"
    required_fields = frozenset(
        {
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
    )


class CommonResultClosureV1(ClosedRuntimeRecord):
    record_type = "CommonResultClosureV1"
    required_fields = frozenset(
        {
            "claim_id",
            "decision",
            "permit_digest",
            "raw_output_digest",
            "reason_codes",
            "release_projection_digest",
            "round_id",
            "run_id",
            "start_receipt_digest",
            "subchecks",
        }
    )


class RawResultEnvelopeV1(ClosedRuntimeRecord):
    record_type = "RawResultEnvelopeV1"
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
            "round_id",
            "run_id",
            "seed",
        }
    )
    optional_fields = frozenset(AUTHORITY_FIELDS)

    def __init__(self, payload: Mapping[str, Any]) -> None:
        merged = {**AUTHORITY_FIELDS, **dict(payload)}
        super().__init__(merged)


def require_sha256_fields(record: ClosedRuntimeRecord, names: Sequence[str]) -> None:
    for name in names:
        validate_sha256(str(getattr(record, name)), field_name=name)


__all__ = [
    "CandidateExecutionBindingV2",
    "CommonDecision",
    "CommonEligibleActionV1",
    "CommonExecutionPermitV1",
    "CommonPlanDecisionV1",
    "CommonPreExecutionDecisionV1",
    "CommonResultClosureV1",
    "DevelopmentExecutionGateDecisionV1",
    "DevelopmentRecSysProtocolV1",
    "ExecutionStartReceiptV1",
    "ExecutionTrustClassificationV1",
    "GateStatus",
    "MaterializationReportV1",
    "RawResultEnvelopeV1",
    "RawRunOutputV1",
    "StartStatus",
    "TrustClass",
]
