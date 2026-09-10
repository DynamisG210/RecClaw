"""V13 scientific-attribution contracts and deterministic admission policy.

This is the single typed source of truth for the data that may cross from the
Helix-private Evidence Guard plane into Research search control.  Guard events
and CompactFeedback remain private inputs; the returned FusedSearchFeedbackV2
contains only common search utility and generic task state.
"""

from __future__ import annotations

import types
import math
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass, fields, replace
from enum import Enum
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.mechanism_space.canonical import (
    deep_freeze,
    deep_thaw,
    snapshot_json,
)

from .contracts import CompactFeedback, PortAdjudication, PortStage, PortStatus


ComparatorDeltaV2 = float | str
NOT_AVAILABLE = "NOT_AVAILABLE"


class FrontierEligibilityV2(str, Enum):
    SEARCH_ELIGIBLE = "SEARCH_ELIGIBLE"
    SEARCH_ELIGIBLE_PRELIMINARY = "SEARCH_ELIGIBLE_PRELIMINARY"
    EXCLUDED = "EXCLUDED"


class SearchFeedbackClassV2(str, Enum):
    BASELINE_RESULT = "BASELINE_RESULT"
    ADMITTED_SEARCH_RESULT = "ADMITTED_SEARCH_RESULT"
    PRELIMINARY_SEARCH_SIGNAL = "PRELIMINARY_SEARCH_SIGNAL"
    NEGATIVE_PRELIMINARY_SIGNAL = "NEGATIVE_PRELIMINARY_SIGNAL"
    REPLICATED_INCONCLUSIVE_SIGNAL = "REPLICATED_INCONCLUSIVE_SIGNAL"
    NEGATIVE_DEVELOPMENT_RESULT = "NEGATIVE_DEVELOPMENT_RESULT"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    ENGINEERING_ONLY = "ENGINEERING_ONLY"
    PROTOCOL_BRANCH_TASK = "PROTOCOL_BRANCH_TASK"
    NO_SEARCH_UPDATE = "NO_SEARCH_UPDATE"
    COMMON_NO_EXECUTION = "COMMON_NO_EXECUTION"
    COMMON_FAILED_EXECUTION = "COMMON_FAILED_EXECUTION"


class ResearchTaskTypeV1(str, Enum):
    VALIDATE_SAME_CANDIDATE = "VALIDATE_SAME_CANDIDATE"
    RUN_MATCHED_CONTROL = "RUN_MATCHED_CONTROL"
    RUN_ABLATION = "RUN_ABLATION"
    REPAIR_IMPLEMENTATION = "REPAIR_IMPLEMENTATION"
    PROTOCOL_BRANCH_DIAGNOSTIC = "PROTOCOL_BRANCH_DIAGNOSTIC"


class ResearchTaskStatusV1(str, Enum):
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


def _closed_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return deep_freeze(snapshot_json(deep_thaw(value)))


def _delta(value: ComparatorDeltaV2) -> ComparatorDeltaV2:
    if value == NOT_AVAILABLE:
        return NOT_AVAILABLE
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("comparator_delta must be numeric or NOT_AVAILABLE")
    return float(value)


@dataclass(frozen=True, slots=True)
class SearchUtilityEventV2:
    candidate_semantic_digest: str
    candidate_id: str
    mechanism_axis: str
    common_outcome_class: str
    runnable_observation: str
    comparator_delta: ComparatorDeltaV2
    metric_contract_digest: str
    resource_cost_projection: Mapping[str, Any]
    typed_blocker_class: str
    observation_seed: str
    # Research-side attribution is attached after the common event is joined
    # with the selected executable program.  Defaults preserve the ABI for
    # Helix events created before Research has seen the selected outcome.
    mechanism_axis_footprint: tuple[str, ...] = ()
    evidence_class: str = "UNCLASSIFIED"
    failure_class: str | None = None
    unresolved_confounding: tuple[str, ...] = ()
    core_mechanism_contrast: str | None = None
    causal_credit_allowed: bool = False
    # Absolute value observed under ``metric_contract_digest``.  The
    # comparator delta is anchored to the frozen comparison and therefore
    # cannot safely be added to an already-advanced search frontier.
    candidate_value: float | None = None

    def __post_init__(self) -> None:
        validate_sha256(
            self.candidate_semantic_digest,
            field_name="candidate_semantic_digest",
        )
        validate_sha256(
            self.metric_contract_digest,
            field_name="metric_contract_digest",
        )
        if not self.candidate_id or not self.mechanism_axis:
            raise ValueError("SearchUtilityEventV2 requires candidate and axis")
        object.__setattr__(self, "comparator_delta", _delta(self.comparator_delta))
        if self.candidate_value is not None:
            if (
                isinstance(self.candidate_value, bool)
                or not isinstance(self.candidate_value, (int, float))
                or not math.isfinite(float(self.candidate_value))
            ):
                raise ValueError("candidate_value must be finite numeric or None")
            object.__setattr__(self, "candidate_value", float(self.candidate_value))
        object.__setattr__(
            self,
            "resource_cost_projection",
            _closed_mapping(self.resource_cost_projection),
        )
        object.__setattr__(
            self,
            "mechanism_axis_footprint",
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in self.mechanism_axis_footprint
                    if str(item).strip()
                )
            ),
        )
        object.__setattr__(
            self,
            "unresolved_confounding",
            tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in self.unresolved_confounding
                    if str(item).strip()
                )
            ),
        )
        if not isinstance(self.evidence_class, str) or not self.evidence_class:
            raise ValueError("evidence_class must be a non-empty string")
        if self.failure_class is not None and (
            not isinstance(self.failure_class, str) or not self.failure_class
        ):
            raise ValueError("failure_class must be a non-empty string or None")
        if self.core_mechanism_contrast is not None and (
            not isinstance(self.core_mechanism_contrast, str)
            or not self.core_mechanism_contrast.strip()
        ):
            raise ValueError(
                "core_mechanism_contrast must be a non-empty string or None"
            )
        if not isinstance(self.causal_credit_allowed, bool):
            raise ValueError("causal_credit_allowed must be boolean")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "candidate_id": self.candidate_id,
                "mechanism_axis": self.mechanism_axis,
                "common_outcome_class": self.common_outcome_class,
                "runnable_observation": self.runnable_observation,
                "comparator_delta": self.comparator_delta,
                "candidate_value": self.candidate_value,
                "metric_contract_digest": self.metric_contract_digest,
                "resource_cost_projection": deep_thaw(
                    self.resource_cost_projection
                ),
                "typed_blocker_class": self.typed_blocker_class,
                "observation_seed": self.observation_seed,
                "mechanism_axis_footprint": self.mechanism_axis_footprint,
                "evidence_class": self.evidence_class,
                "failure_class": self.failure_class,
                "unresolved_confounding": self.unresolved_confounding,
                "core_mechanism_contrast": self.core_mechanism_contrast,
                "causal_credit_allowed": self.causal_credit_allowed,
            }
        )


@dataclass(frozen=True, slots=True)
class ResearchTaskV1:
    task_id: str
    task_type: ResearchTaskTypeV1
    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str
    parent_candidate_id: str | None
    comparator_identity: str
    protocol_digest: str
    required_seed_or_control: str
    task_status: ResearchTaskStatusV1
    created_round: int
    utility_priority: float
    missing_seed_count: int
    mechanism_program: Mapping[str, Any]
    owner_arm_instance_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.candidate_id:
            raise ValueError("ResearchTaskV1 requires task and candidate identities")
        if self.owner_arm_instance_id == "":
            raise ValueError("Research task owner cannot be empty")
        for name in (
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "protocol_digest",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.created_round < 1 or self.missing_seed_count < 0:
            raise ValueError("Research task round/count is invalid")
        if not 0.0 <= float(self.utility_priority) <= 1.0:
            raise ValueError("Research task utility_priority must be in [0,1]")
        object.__setattr__(
            self,
            "mechanism_program",
            _closed_mapping(self.mechanism_program),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "task_id": self.task_id,
                "task_type": self.task_type.value,
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "parent_candidate_id": self.parent_candidate_id,
                "comparator_identity": self.comparator_identity,
                "protocol_digest": self.protocol_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "task_status": self.task_status.value,
                "created_round": self.created_round,
                "utility_priority": float(self.utility_priority),
                "missing_seed_count": self.missing_seed_count,
                "mechanism_program": deep_thaw(self.mechanism_program),
                "owner_arm_instance_id": self.owner_arm_instance_id,
            }
        )

    def prompt_projection(self) -> dict[str, Any]:
        return canonical_value(
            {
                "task_type": self.task_type.value,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "task_status": self.task_status.value,
            }
        )


class ResearchTaskQueueV1:
    """Arm-private deterministic queue; every task consumes a normal round."""

    _TYPE_ORDER = {
        ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE: 0,
        ResearchTaskTypeV1.RUN_MATCHED_CONTROL: 1,
        ResearchTaskTypeV1.RUN_ABLATION: 2,
        ResearchTaskTypeV1.REPAIR_IMPLEMENTATION: 3,
        ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC: 4,
    }

    def __init__(self, owner_arm_instance_id: str | None = None) -> None:
        if owner_arm_instance_id == "":
            raise ValueError("Research task queue owner cannot be empty")
        self.owner_arm_instance_id = owner_arm_instance_id
        self._tasks: dict[str, ResearchTaskV1] = {}

    @property
    def tasks(self) -> tuple[ResearchTaskV1, ...]:
        return tuple(
            sorted(self._tasks.values(), key=lambda item: item.task_id)
        )

    @property
    def digest(self) -> str:
        return sha256_digest([item.to_dict() for item in self.tasks])

    def enqueue(self, task: ResearchTaskV1) -> ResearchTaskV1:
        if (
            self.owner_arm_instance_id is not None
            and task.owner_arm_instance_id
            != self.owner_arm_instance_id
        ):
            raise ValueError("Research task crossed its Arm owner boundary")
        prior = self._tasks.get(task.task_id)
        if prior is not None and (
            prior.candidate_semantic_digest != task.candidate_semantic_digest
            or prior.mechanism_program_digest != task.mechanism_program_digest
            or prior.protocol_digest != task.protocol_digest
        ):
            raise ValueError("Research task identity substitution")
        if prior is None:
            self._tasks[task.task_id] = task
        elif prior.task_status in {
            ResearchTaskStatusV1.PENDING,
            ResearchTaskStatusV1.ACTIVE,
        }:
            self._tasks[task.task_id] = replace(
                prior,
                utility_priority=max(
                    float(prior.utility_priority),
                    float(task.utility_priority),
                ),
                missing_seed_count=min(
                    prior.missing_seed_count,
                    task.missing_seed_count,
                ),
            )
        return self._tasks[task.task_id]

    def select_next(
        self,
        *,
        allowed_types: frozenset[ResearchTaskTypeV1] | None = None,
    ) -> ResearchTaskV1 | None:
        pending = [
            item
            for item in self._tasks.values()
            if item.task_status is ResearchTaskStatusV1.PENDING
            and (
                allowed_types is None
                or item.task_type in allowed_types
            )
        ]
        if not pending:
            return None
        return min(
            pending,
            key=lambda item: (
                self._TYPE_ORDER[item.task_type],
                -float(item.utility_priority),
                item.created_round,
                item.missing_seed_count,
                item.task_id,
            ),
        )

    def activate(self, task_id: str) -> ResearchTaskV1:
        task = self._tasks[task_id]
        if task.task_status is not ResearchTaskStatusV1.PENDING:
            raise ValueError("Only a pending Research task can be activated")
        active = replace(task, task_status=ResearchTaskStatusV1.ACTIVE)
        self._tasks[task_id] = active
        return active

    def complete(self, task_id: str) -> ResearchTaskV1:
        task = self._tasks[task_id]
        if task.task_status is not ResearchTaskStatusV1.ACTIVE:
            raise ValueError("Only an active Research task can be completed")
        completed = replace(task, task_status=ResearchTaskStatusV1.COMPLETED)
        self._tasks[task_id] = completed
        return completed

    def cancel(self, task_id: str) -> ResearchTaskV1:
        task = self._tasks[task_id]
        if task.task_status is not ResearchTaskStatusV1.ACTIVE:
            raise ValueError("Only an active Research task can be cancelled")
        cancelled = replace(
            task, task_status=ResearchTaskStatusV1.CANCELLED
        )
        self._tasks[task_id] = cancelled
        return cancelled

    def cancel_without_execution(self, task_id: str) -> ResearchTaskV1:
        return self.cancel(task_id)


@dataclass(frozen=True, slots=True)
class GuardEvidenceObservationV1:
    candidate_semantic_digest: str
    mechanism_program_digest: str
    protocol_digest: str
    comparator_identity: str
    observation_seed: str
    observation_id: str

    def __post_init__(self) -> None:
        validate_sha256(
            self.candidate_semantic_digest,
            field_name="candidate_semantic_digest",
        )
        validate_sha256(
            self.mechanism_program_digest,
            field_name="mechanism_program_digest",
        )
        validate_sha256(self.protocol_digest, field_name="protocol_digest")
        validate_sha256(self.observation_id, field_name="observation_id")

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (
            self.candidate_semantic_digest,
            self.mechanism_program_digest,
            self.protocol_digest,
            self.comparator_identity,
            self.observation_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_semantic_digest": (
                    self.candidate_semantic_digest
                ),
                "mechanism_program_digest": self.mechanism_program_digest,
                "protocol_digest": self.protocol_digest,
                "comparator_identity": self.comparator_identity,
                "observation_seed": self.observation_seed,
                "observation_id": self.observation_id,
            }
        )


@dataclass(frozen=True, slots=True)
class GuardEvidenceSnapshotV1:
    observations: tuple[GuardEvidenceObservationV1, ...]

    def __post_init__(self) -> None:
        ordered = tuple(
            sorted(self.observations, key=lambda item: (item.key, item.observation_id))
        )
        if len({item.key for item in ordered}) != len(ordered):
            raise ValueError("Guard evidence snapshot contains a duplicate exact key")
        object.__setattr__(self, "observations", ordered)

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [item.to_dict() for item in self.observations]
        }

    def guard_core_projection(
        self, *, snapshot_id: str, claim_id: str, protocol_id: str
    ) -> dict[str, Any]:
        return {
            "snapshot_id": snapshot_id,
            "claim_id": claim_id,
            "protocol_id": protocol_id,
            "observation_ids": list(
                dict.fromkeys(
                    item.observation_id for item in self.observations
                )
            ),
        }


@dataclass(frozen=True, slots=True)
class ValidationResultBundleV1:
    task_id: str
    candidate_semantic_digest: str
    observation_seeds: tuple[str, ...]
    search_utility_events: tuple[SearchUtilityEventV2, ...]
    development_validation_status: str

    def __post_init__(self) -> None:
        validate_sha256(
            self.candidate_semantic_digest,
            field_name="candidate_semantic_digest",
        )
        if len(set(self.observation_seeds)) != len(self.observation_seeds):
            raise ValueError("ValidationResultBundleV1 seeds must be unique")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "task_id": self.task_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "observation_seeds": self.observation_seeds,
                "search_utility_events": [
                    item.to_dict() for item in self.search_utility_events
                ],
                "development_validation_status": self.development_validation_status,
            }
        )


@dataclass(frozen=True, slots=True)
class EvidenceSummaryV1:
    """Closed, descriptive summary consumed by the V13 control operator.

    This is deliberately a development record.  The interval is descriptive
    and is never a formal or adaptive-valid claim interval.
    """

    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str | None
    protocol_digest: str
    comparator_identity: str
    mechanism_axis: str
    verified_seed_ids: tuple[str, ...]
    invalid_seed_ids: tuple[str, ...]
    protocol_drift_seed_ids: tuple[str, ...]
    engineering_failure_seed_ids: tuple[str, ...]
    missing_seed_ids: tuple[str, ...]
    comparator_deltas: tuple[float, ...]
    mean_comparator_delta: float | None
    dispersion: float | None
    standard_error: float | None
    descriptive_t_interval_95: tuple[float, float] | None
    sign_consistency: str
    scientific_conclusion_strength: str
    current_attempt_class: str
    required_seed_count: int
    evidence_count: int
    minimum_effect_delta: float
    protocol_status: str
    next_eligible_seed: str | None = None

    def __post_init__(self) -> None:
        validate_sha256(self.protocol_digest, field_name="protocol_digest")
        validate_sha256(
            self.candidate_semantic_digest,
            field_name="candidate_semantic_digest",
        )
        if self.mechanism_program_digest is not None:
            validate_sha256(
                self.mechanism_program_digest,
                field_name="mechanism_program_digest",
            )
        if not self.candidate_id or not self.comparator_identity:
            raise ValueError("EvidenceSummaryV1 requires candidate identity")
        if not self.mechanism_axis:
            raise ValueError("EvidenceSummaryV1 requires mechanism axis")
        if self.scientific_conclusion_strength not in SCIENTIFIC_CONCLUSION_STATES_V1:
            raise ValueError("EvidenceSummaryV1 scientific state is outside the closed set")
        if self.current_attempt_class not in CURRENT_ATTEMPT_CLASSES_V1:
            raise ValueError("EvidenceSummaryV1 attempt class is outside the closed set")
        if self.protocol_status not in PROTOCOL_STATUSES_V1:
            raise ValueError("EvidenceSummaryV1 protocol status is outside the closed set")
        if self.sign_consistency not in SIGN_CONSISTENCY_CLASSES_V1:
            raise ValueError("EvidenceSummaryV1 sign consistency is outside the closed set")
        if self.required_seed_count < 1 or self.evidence_count < 0:
            raise ValueError("EvidenceSummaryV1 seed counts are invalid")
        if self.evidence_count != len(self.verified_seed_ids):
            raise ValueError("EvidenceSummaryV1 evidence count is not verified seeds")
        if len(set(self.verified_seed_ids)) != len(self.verified_seed_ids):
            raise ValueError("EvidenceSummaryV1 verified seeds must be unique")
        if len(set(self.invalid_seed_ids)) != len(self.invalid_seed_ids):
            raise ValueError("EvidenceSummaryV1 invalid seeds must be unique")
        verified = set(self.verified_seed_ids)
        for name in (
            "invalid_seed_ids",
            "protocol_drift_seed_ids",
            "engineering_failure_seed_ids",
        ):
            if verified.intersection(getattr(self, name)):
                raise ValueError(
                    f"EvidenceSummaryV1 verified seed overlaps {name}"
                )
        if self.mean_comparator_delta is not None and not math.isfinite(
            float(self.mean_comparator_delta)
        ):
            raise ValueError("EvidenceSummaryV1 mean delta must be finite")
        for value in self.comparator_deltas:
            if not math.isfinite(float(value)):
                raise ValueError("EvidenceSummaryV1 deltas must be finite")
        for name in ("dispersion", "standard_error"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"EvidenceSummaryV1 {name} must be finite")
        interval = self.descriptive_t_interval_95
        if interval is not None:
            if len(interval) != 2 or not all(math.isfinite(float(item)) for item in interval):
                raise ValueError("EvidenceSummaryV1 t interval must be finite")
        object.__setattr__(self, "verified_seed_ids", tuple(self.verified_seed_ids))
        object.__setattr__(self, "invalid_seed_ids", tuple(self.invalid_seed_ids))
        object.__setattr__(self, "protocol_drift_seed_ids", tuple(self.protocol_drift_seed_ids))
        object.__setattr__(self, "engineering_failure_seed_ids", tuple(self.engineering_failure_seed_ids))
        object.__setattr__(self, "missing_seed_ids", tuple(self.missing_seed_ids))
        object.__setattr__(self, "comparator_deltas", tuple(float(item) for item in self.comparator_deltas))

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "protocol_digest": self.protocol_digest,
                "comparator_identity": self.comparator_identity,
                "mechanism_axis": self.mechanism_axis,
                "verified_seed_ids": self.verified_seed_ids,
                "invalid_seed_ids": self.invalid_seed_ids,
                "protocol_drift_seed_ids": self.protocol_drift_seed_ids,
                "engineering_failure_seed_ids": self.engineering_failure_seed_ids,
                "missing_seed_ids": self.missing_seed_ids,
                "comparator_deltas": self.comparator_deltas,
                "mean_comparator_delta": self.mean_comparator_delta,
                "dispersion": self.dispersion,
                "standard_error": self.standard_error,
                "descriptive_t_interval_95": self.descriptive_t_interval_95,
                "sign_consistency": self.sign_consistency,
                "scientific_conclusion_strength": self.scientific_conclusion_strength,
                "current_attempt_class": self.current_attempt_class,
                # Compatibility projection for consumers that only understand
                # the cumulative scientific state.  It never carries the
                # current attempt classification.
                "conclusion_strength": self.scientific_conclusion_strength,
                "required_seed_count": self.required_seed_count,
                "evidence_count": self.evidence_count,
                "minimum_effect_delta": self.minimum_effect_delta,
                "protocol_status": self.protocol_status,
                "next_eligible_seed": self.next_eligible_seed,
                "descriptive_only": True,
                "formal_inference": False,
                "adaptive_valid": False,
            }
        )

    @property
    def conclusion_strength(self) -> str:
        """Legacy name for the cumulative scientific state only."""

        return self.scientific_conclusion_strength


REQUESTED_CONTROL_KINDS_V1 = ("MATCHED_CONTROL", "MECHANISM_OFF", "NONE")
SCIENTIFIC_CONCLUSION_STATES_V1 = (
    "INCONCLUSIVE",
    "PRELIMINARY_POSITIVE",
    "PRELIMINARY_NONPOSITIVE",
    "SUPPORTED",
    "REPLICATED_INCONCLUSIVE",
    "REFUTED",
)
CURRENT_ATTEMPT_CLASSES_V1 = (
    "VALID_METRIC",
    "ENGINEERING_FAILURE",
    "PROTOCOL_DRIFT",
    "INVALID",
)
PROTOCOL_STATUSES_V1 = ("CURRENT_PROTOCOL", "PROTOCOL_BRANCH")
SIGN_CONSISTENCY_CLASSES_V1 = (
    "NO_VERIFIED_SEEDS",
    "ALL_POSITIVE",
    "ALL_NONPOSITIVE",
    "MIXED_SIGNS",
)
ADMISSIBILITY_OPERATOR_DIGEST_V1 = sha256_digest(
    {"operator": "admissibility", "policy": "DeterministicHelixAdmissionV13"}
)
NEXT_SEED_OPERATOR_DIGEST_V1 = sha256_digest(
    {"operator": "uncertainty_aware_next_seed", "budget": "same_budget"}
)
CONTROL_REQUEST_OPERATOR_DIGEST_V1 = sha256_digest(
    {"operator": "restricted_control_request", "binding": "research_interpreter"}
)
MEMORY_CREDIT_OPERATOR_DIGEST_V1 = sha256_digest(
    {"operator": "calibrated_memory_axis_credit", "formal_claim": False}
)


@dataclass(frozen=True, slots=True)
class HelixControlProjectionV1:
    """Typed closed-loop output of the sole V13 admission authority."""

    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str | None
    evidence_summary: EvidenceSummaryV1
    admissibility_decision: str
    next_seed_allocation: str
    next_seed: str | None
    requested_control_kind: str
    control_binding_required: bool
    task_binding_required: bool
    control_binding_status: str
    memory_update: str
    evidence_class: str
    confidence_weight: float
    development_promotion: str
    formal_claim_authority: str

    def __post_init__(self) -> None:
        if self.candidate_id != self.evidence_summary.candidate_id:
            raise ValueError("control projection candidate identity mismatch")
        if self.candidate_semantic_digest != self.evidence_summary.candidate_semantic_digest:
            raise ValueError("control projection semantic identity mismatch")
        if self.mechanism_program_digest != self.evidence_summary.mechanism_program_digest:
            raise ValueError("control projection program identity mismatch")
        if self.requested_control_kind not in REQUESTED_CONTROL_KINDS_V1:
            raise ValueError("control projection requested kind is outside the closed request set")
        if self.requested_control_kind != "NONE" and not self.control_binding_required:
            raise ValueError("control projection must require binding for a control request")
        if not 0.0 <= float(self.confidence_weight) <= 1.0:
            raise ValueError("control projection confidence weight must be in [0,1]")
        if self.evidence_summary.scientific_conclusion_strength.startswith("PRELIMINARY_") and not (
            float(self.confidence_weight) < 1.0
        ):
            raise ValueError("preliminary control projection cannot have full confidence")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "evidence_summary": self.evidence_summary.to_dict(),
                "admissibility_decision": self.admissibility_decision,
                "next_seed_allocation": self.next_seed_allocation,
                "next_seed": self.next_seed,
                "requested_control_kind": self.requested_control_kind,
                "control_binding_required": self.control_binding_required,
                "task_binding_required": self.task_binding_required,
                "control_binding_status": self.control_binding_status,
                "memory_update": self.memory_update,
                "evidence_class": self.evidence_class,
                "confidence_weight": float(self.confidence_weight),
                "development_promotion": self.development_promotion,
                "formal_claim_authority": self.formal_claim_authority,
                "operator_digests": {
                    "admissibility": ADMISSIBILITY_OPERATOR_DIGEST_V1,
                    "next_seed_allocation": NEXT_SEED_OPERATOR_DIGEST_V1,
                    "control_request": CONTROL_REQUEST_OPERATOR_DIGEST_V1,
                    "calibrated_memory_credit": MEMORY_CREDIT_OPERATOR_DIGEST_V1,
                },
            }
        )


@dataclass(frozen=True, slots=True)
class FusedSearchFeedbackV2:
    candidate_id: str | None
    search_feedback_class: SearchFeedbackClassV2
    search_utility_event: SearchUtilityEventV2 | None
    frontier_eligibility: FrontierEligibilityV2
    research_task: ResearchTaskV1 | None
    controller_update_allowed: bool
    meta_update_allowed: bool
    search_memory_update_allowed: bool
    control_projection: HelixControlProjectionV1 | None = None

    def __post_init__(self) -> None:
        if self.search_feedback_class is SearchFeedbackClassV2.NO_SEARCH_UPDATE:
            if (
                self.search_utility_event is not None
                or self.research_task is not None
                or self.controller_update_allowed
                or self.meta_update_allowed
                or self.search_memory_update_allowed
                or self.frontier_eligibility is not FrontierEligibilityV2.EXCLUDED
                or self.control_projection is not None
            ):
                raise ValueError("NO_SEARCH_UPDATE must preserve every search state")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "search_feedback_class": self.search_feedback_class.value,
                "search_utility_event": (
                    self.search_utility_event.to_dict()
                    if self.search_utility_event is not None
                    else None
                ),
                "frontier_eligibility": self.frontier_eligibility.value,
                "research_task": (
                    self.research_task.to_dict()
                    if self.research_task is not None
                    else None
                ),
                "controller_update_allowed": (
                    self.controller_update_allowed
                ),
                "meta_update_allowed": self.meta_update_allowed,
                "search_memory_update_allowed": (
                    self.search_memory_update_allowed
                ),
                "control_projection": (
                    self.control_projection.to_dict()
                    if self.control_projection is not None
                    else None
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class PromptFeedbackProjectionV2:
    common_search_utility_slot: SearchUtilityEventV2 | None
    research_task_slot: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.research_task_slot is not None:
            allowed = {
                "task_type",
                "candidate_semantic_digest",
                "required_seed_or_control",
                "task_status",
            }
            if set(self.research_task_slot) != allowed:
                raise ValueError("Prompt task slot is not the closed V13 projection")
            object.__setattr__(
                self,
                "research_task_slot",
                _closed_mapping(self.research_task_slot),
            )

    @classmethod
    def from_fused(
        cls, feedback: FusedSearchFeedbackV2
    ) -> "PromptFeedbackProjectionV2":
        return cls(
            common_search_utility_slot=feedback.search_utility_event,
            research_task_slot=(
                feedback.research_task.prompt_projection()
                if feedback.research_task is not None
                else None
            ),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "common_search_utility_slot": (
                    self.common_search_utility_slot.to_dict()
                    if self.common_search_utility_slot is not None
                    else "ABSENT"
                ),
                "research_task_slot": (
                    deep_thaw(self.research_task_slot)
                    if self.research_task_slot is not None
                    else "ABSENT"
                ),
            }
        )


class DeterministicHelixAdmissionV13:
    """Pure Guard-to-search admission mapping with no writer capability."""

    policy_digest = sha256_digest(
        {
            "policy": "DeterministicHelixAdmissionV13",
            "preliminary_positive": (
                "SEARCH_ELIGIBLE_PRELIMINARY_AND_VALIDATION_TASK"
            ),
            "preliminary_nonpositive": (
                "WEIGHTED_NEGATIVE_AND_MOVE_ON"
            ),
            "supported": "FULL_DEVELOPMENT_CREDIT_AND_PROMOTION",
            "replicated_inconclusive": "MATCHED_CONTROL_OR_MECHANISM_OFF",
            "engineering_failure": "RESOURCE_FEASIBILITY_MEMORY_ONLY",
            "protocol_drift": "ZERO_SCIENCE_UPDATE",
            "invalid": "NO_SEARCH_UPDATE",
        }
    )

    @staticmethod
    def no_search_update(candidate_id: str | None) -> FusedSearchFeedbackV2:
        return FusedSearchFeedbackV2(
            candidate_id=candidate_id,
            search_feedback_class=SearchFeedbackClassV2.NO_SEARCH_UPDATE,
            search_utility_event=None,
            frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
            research_task=None,
            controller_update_allowed=False,
            meta_update_allowed=False,
            search_memory_update_allowed=False,
        )

    @staticmethod
    def admit_pre(adjudication: PortAdjudication) -> str:
        """Map PRE authority to a same-slate action without search feedback."""

        if adjudication.stage is not PortStage.PRE:
            raise ValueError("V13 PRE admission requires PRE adjudication")
        if adjudication.status is PortStatus.BLOCK:
            return "NEXT_FROM_SAME_SLATE"
        if adjudication.status is PortStatus.NOT_ADJUDICATED:
            return "SELECT_CURRENT_NOT_ADJUDICATED"
        if adjudication.status in {PortStatus.ALLOW, PortStatus.ADJUDICATED}:
            return "SELECT_CURRENT"
        raise ValueError("EvidencePort PRE returned an unusable status")

    @staticmethod
    def _control_projection(
        summary: EvidenceSummaryV1,
    ) -> HelixControlProjectionV1:
        state = summary.scientific_conclusion_strength
        attempt = summary.current_attempt_class
        if attempt in {"ENGINEERING_FAILURE", "PROTOCOL_DRIFT", "INVALID"}:
            admissibility = "EXCLUDE_SCIENCE_UPDATE"
            allocation = {
                "ENGINEERING_FAILURE": "RESOURCE_OR_FEASIBILITY_DIAGNOSTIC_ONLY",
                "PROTOCOL_DRIFT": "ZERO_SCIENCE_UPDATE",
                "INVALID": "INVALID_EVIDENCE_DIAGNOSTIC_ONLY",
            }[attempt]
            next_seed = None
            requested_control = "NONE"
            memory = {
                "ENGINEERING_FAILURE": "RESOURCE_FEASIBILITY_MEMORY_ONLY",
                "PROTOCOL_DRIFT": "PROTOCOL_DIAGNOSTIC_ONLY",
                "INVALID": "INVALID_EVIDENCE_MEMORY_ONLY",
            }[attempt]
            promotion = "HOLD"
        elif state == "PRELIMINARY_POSITIVE":
            admissibility = "ADMIT_PRELIMINARY_DEVELOPMENT_SIGNAL"
            allocation = "NEXT_UNSEEN_SEED_SAME_BUDGET"
            next_seed = summary.next_eligible_seed
            requested_control = "NONE"
            memory = "RETAIN_PRELIMINARY_SIGNAL"
            promotion = "HOLD"
        elif state in {"PRELIMINARY_NONPOSITIVE", "REFUTED"}:
            admissibility = "ADMIT_WEIGHTED_NEGATIVE_DEVELOPMENT_SIGNAL"
            allocation = "WEIGHTED_NEGATIVE_MOVE_ON"
            next_seed = None
            requested_control = "NONE"
            memory = "WEIGHTED_NEGATIVE_MOVE_ON"
            promotion = "HOLD"
        elif state == "SUPPORTED":
            admissibility = "ADMIT_SUPPORTED_DEVELOPMENT_SIGNAL"
            allocation = "NO_ADDITIONAL_VALIDATION_REQUIRED"
            next_seed = None
            requested_control = "NONE"
            memory = "FULL_DEVELOPMENT_CREDIT"
            promotion = "ALLOW_DEVELOPMENT_PROMOTION"
        elif state == "REPLICATED_INCONCLUSIVE":
            admissibility = "ADMIT_UNCERTAIN_DEVELOPMENT_SIGNAL_NO_PROMOTION"
            allocation = "MATCHED_CONTROL_OR_MECHANISM_OFF"
            next_seed = None
            requested_control = "MATCHED_CONTROL"
            memory = "RETAIN_UNCERTAIN_NO_PROMOTION"
            promotion = "HOLD"
        elif state == "ENGINEERING_FAILURE":
            admissibility = "EXCLUDE_SCIENCE_UPDATE"
            allocation = "RESOURCE_OR_FEASIBILITY_DIAGNOSTIC_ONLY"
            next_seed = None
            requested_control = "NONE"
            memory = "RESOURCE_FEASIBILITY_MEMORY_ONLY"
            promotion = "HOLD"
        elif state == "PROTOCOL_DRIFT":
            admissibility = "EXCLUDE_SCIENCE_UPDATE"
            allocation = "ZERO_SCIENCE_UPDATE"
            next_seed = None
            requested_control = "NONE"
            memory = "PROTOCOL_DIAGNOSTIC_ONLY"
            promotion = "HOLD"
        else:
            admissibility = "NO_CLAIM_UPDATE"
            allocation = "NO_SEARCH_UPDATE"
            next_seed = None
            requested_control = "NONE"
            memory = "DIAGNOSTIC_ONLY"
            promotion = "HOLD"
        confidence_weight = (
            0.0
            if attempt in {"ENGINEERING_FAILURE", "PROTOCOL_DRIFT", "INVALID"}
            or state == "INCONCLUSIVE"
            else min(
                1.0,
                float(summary.evidence_count) / float(summary.required_seed_count),
            )
        )
        evidence_class = {
            "PRELIMINARY_POSITIVE": "DEVELOPMENT_PRELIMINARY_POSITIVE",
            "PRELIMINARY_NONPOSITIVE": "DEVELOPMENT_NEGATIVE",
            "REFUTED": "DEVELOPMENT_NEGATIVE",
            "SUPPORTED": "DEVELOPMENT_SUPPORTED",
            "REPLICATED_INCONCLUSIVE": "DEVELOPMENT_REPLICATED_INCONCLUSIVE",
            "ENGINEERING_FAILURE": "ENGINEERING_ONLY",
            "PROTOCOL_DRIFT": "PROTOCOL_DRIFT_ONLY",
            "INVALID": "INVALID_ONLY",
        }.get(state, "DEVELOPMENT_INCONCLUSIVE")
        if attempt in {"ENGINEERING_FAILURE", "PROTOCOL_DRIFT", "INVALID"}:
            evidence_class = {
                "ENGINEERING_FAILURE": "ENGINEERING_ONLY",
                "PROTOCOL_DRIFT": "PROTOCOL_DRIFT_ONLY",
                "INVALID": "INVALID_ONLY",
            }[attempt]
        return HelixControlProjectionV1(
            candidate_id=summary.candidate_id,
            candidate_semantic_digest=summary.candidate_semantic_digest,
            mechanism_program_digest=summary.mechanism_program_digest,
            evidence_summary=summary,
            admissibility_decision=admissibility,
            next_seed_allocation=allocation,
            next_seed=next_seed,
            requested_control_kind=requested_control,
            control_binding_required=requested_control != "NONE",
            task_binding_required=(
                attempt == "VALID_METRIC"
                and state
                in {
                    "PRELIMINARY_POSITIVE",
                    "REPLICATED_INCONCLUSIVE",
                }
            ),
            control_binding_status=(
                "PENDING_RESEARCH_INTERPRETER"
                if requested_control != "NONE"
                else "NOT_REQUESTED"
            ),
            memory_update=memory,
            evidence_class=evidence_class,
            confidence_weight=confidence_weight,
            development_promotion=promotion,
            formal_claim_authority="NONE",
        )

    @staticmethod
    def _validate_task_input(
        task: ResearchTaskV1 | None,
        *,
        summary: EvidenceSummaryV1,
        allowed_types: frozenset[ResearchTaskTypeV1],
    ) -> None:
        if task is None:
            return
        if (
            summary.mechanism_program_digest is None
            or task.candidate_id != summary.candidate_id
            or task.candidate_semantic_digest != summary.candidate_semantic_digest
            or task.protocol_digest != summary.protocol_digest
            or task.comparator_identity != summary.comparator_identity
            or task.task_type not in allowed_types
            or (
                summary.mechanism_program_digest is not None
                and task.mechanism_program_digest
                != summary.mechanism_program_digest
            )
        ):
            raise ValueError("Research task input is not bound to the V13 request")

    def admit_post(
        self,
        *,
        adjudication: PortAdjudication,
        search_utility_event: SearchUtilityEventV2,
        validation_task: ResearchTaskV1 | None = None,
        protocol_branch_task: ResearchTaskV1 | None = None,
        matched_control_task: ResearchTaskV1 | None = None,
        evidence_summary: EvidenceSummaryV1 | None = None,
    ) -> tuple[FusedSearchFeedbackV2, CompactFeedback | None]:
        if adjudication.stage is not PortStage.POST:
            raise ValueError("V13 POST admission requires POST adjudication")
        if adjudication.candidate_id != search_utility_event.candidate_id:
            raise ValueError("adjudication/search utility candidate mismatch")
        control_projection = (
            self._control_projection(evidence_summary)
            if evidence_summary is not None
            else None
        )
        if evidence_summary is not None and (
            evidence_summary.candidate_id != adjudication.candidate_id
            or evidence_summary.protocol_status == "PROTOCOL_BRANCH"
            and adjudication.protocol_status != "PROTOCOL_BRANCH"
        ):
            raise ValueError("V13 evidence summary/adjudication mismatch")
        if evidence_summary is not None:
            self._validate_task_input(
                validation_task,
                summary=evidence_summary,
                allowed_types=frozenset({ResearchTaskTypeV1.VALIDATE_SAME_CANDIDATE}),
            )
            self._validate_task_input(
                matched_control_task,
                summary=evidence_summary,
                allowed_types=frozenset(
                    {
                        ResearchTaskTypeV1.RUN_MATCHED_CONTROL,
                        ResearchTaskTypeV1.RUN_ABLATION,
                    }
                ),
            )
            self._validate_task_input(
                protocol_branch_task,
                summary=evidence_summary,
                allowed_types=frozenset(
                    {ResearchTaskTypeV1.PROTOCOL_BRANCH_DIAGNOSTIC}
                ),
            )
        if adjudication.status is PortStatus.NOT_ADJUDICATED:
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=SearchFeedbackClassV2.BASELINE_RESULT,
                    search_utility_event=search_utility_event,
                    frontier_eligibility=FrontierEligibilityV2.SEARCH_ELIGIBLE,
                    research_task=None,
                    controller_update_allowed=True,
                    meta_update_allowed=True,
                    search_memory_update_allowed=True,
                    control_projection=control_projection,
                ),
                None,
            )
        compact = CompactFeedback(
            candidate_id=adjudication.candidate_id,
            protocol_status=adjudication.protocol_status,
            outcome_class=adjudication.outcome_class,
            claim_ceiling=adjudication.claim_ceiling,
            reason_codes=adjudication.reason_codes,
            comparator_delta=adjudication.comparator_delta,
            evidence_use=adjudication.evidence_use,
            recommended_validation=adjudication.recommended_validation,
        )
        if adjudication.status in {PortStatus.ERROR, PortStatus.BLOCK}:
            return self.no_search_update(adjudication.candidate_id), compact
        if adjudication.protocol_status == "PROTOCOL_BRANCH":
            if protocol_branch_task is None and evidence_summary is None:
                raise ValueError("Protocol branch disposition requires a typed task")
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=(
                        SearchFeedbackClassV2.PROTOCOL_BRANCH_TASK
                    ),
                    search_utility_event=None,
                    frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                    research_task=protocol_branch_task,
                    controller_update_allowed=False,
                    meta_update_allowed=False,
                    search_memory_update_allowed=True,
                    control_projection=control_projection,
                ),
                compact,
            )
        if evidence_summary is not None:
            state = evidence_summary.conclusion_strength
            if evidence_summary.current_attempt_class == "PROTOCOL_DRIFT":
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=SearchFeedbackClassV2.PROTOCOL_BRANCH_TASK,
                        search_utility_event=None,
                        frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                        research_task=protocol_branch_task,
                        controller_update_allowed=False,
                        meta_update_allowed=False,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
            if evidence_summary.current_attempt_class in {"ENGINEERING_FAILURE", "INVALID"}:
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=SearchFeedbackClassV2.ENGINEERING_ONLY,
                        search_utility_event=None,
                        frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                        research_task=None,
                        controller_update_allowed=False,
                        meta_update_allowed=False,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
            if state == "PRELIMINARY_POSITIVE":
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=SearchFeedbackClassV2.PRELIMINARY_SEARCH_SIGNAL,
                        search_utility_event=search_utility_event,
                        frontier_eligibility=FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY,
                        research_task=validation_task,
                        controller_update_allowed=False,
                        meta_update_allowed=False,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
            if state in {"PRELIMINARY_NONPOSITIVE", "REFUTED"}:
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=(
                            SearchFeedbackClassV2.NEGATIVE_PRELIMINARY_SIGNAL
                            if state == "PRELIMINARY_NONPOSITIVE"
                            else SearchFeedbackClassV2.NEGATIVE_DEVELOPMENT_RESULT
                        ),
                        search_utility_event=search_utility_event,
                        frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                        research_task=None,
                        # Weighted memory/ranking credit is the only allowed
                        # preliminary-negative influence.  Full controller
                        # state is reserved for supported evidence.
                        controller_update_allowed=False,
                        meta_update_allowed=False,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
            if state == "REPLICATED_INCONCLUSIVE":
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=SearchFeedbackClassV2.REPLICATED_INCONCLUSIVE_SIGNAL,
                        search_utility_event=search_utility_event,
                        frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                        research_task=matched_control_task,
                        # The bound control task is the intervention.  An
                        # inconclusive candidate must not receive full policy
                        # or incumbent credit before that task resolves it.
                        controller_update_allowed=False,
                        meta_update_allowed=False,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
            if state == "SUPPORTED":
                return (
                    FusedSearchFeedbackV2(
                        candidate_id=adjudication.candidate_id,
                        search_feedback_class=SearchFeedbackClassV2.ADMITTED_SEARCH_RESULT,
                        search_utility_event=search_utility_event,
                        frontier_eligibility=FrontierEligibilityV2.SEARCH_ELIGIBLE,
                        research_task=None,
                        controller_update_allowed=True,
                        meta_update_allowed=True,
                        search_memory_update_allowed=True,
                        control_projection=control_projection,
                    ),
                    compact,
                )
        disposition = adjudication.evidence_use
        if adjudication.recommended_validation == "REQUIRES_CONFIRMATION":
            if validation_task is None:
                raise ValueError("Preliminary result requires a validation task")
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=(
                        SearchFeedbackClassV2.PRELIMINARY_SEARCH_SIGNAL
                    ),
                    search_utility_event=search_utility_event,
                    frontier_eligibility=(
                        FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY
                    ),
                    research_task=validation_task,
                    controller_update_allowed=False,
                    meta_update_allowed=False,
                    search_memory_update_allowed=True,
                ),
                compact,
            )
        if disposition == "COUNT_AS_LOCAL_PRELIMINARY_SIGNAL":
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=(
                        SearchFeedbackClassV2.PRELIMINARY_SEARCH_SIGNAL
                    ),
                    search_utility_event=search_utility_event,
                    frontier_eligibility=(
                        FrontierEligibilityV2.SEARCH_ELIGIBLE_PRELIMINARY
                    ),
                    research_task=None,
                    controller_update_allowed=True,
                    meta_update_allowed=True,
                    search_memory_update_allowed=True,
                ),
                compact,
            )
        if disposition == "COUNT_AS_SAME_PROTOCOL_MULTI_SEED_DEVELOPMENT_SIGNAL":
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=(
                        SearchFeedbackClassV2.ADMITTED_SEARCH_RESULT
                    ),
                    search_utility_event=search_utility_event,
                    frontier_eligibility=FrontierEligibilityV2.SEARCH_ELIGIBLE,
                    research_task=None,
                    controller_update_allowed=True,
                    meta_update_allowed=True,
                    search_memory_update_allowed=True,
                ),
                compact,
            )
        if disposition in {
            "RECORD_EXECUTABILITY_ONLY",
            "RECORD_RUNTIME_BLOCKER_ONLY",
            "RECORD_DIAGNOSTIC_ONLY",
            "QUARANTINE_METRIC_MISSING",
            "QUARANTINE_PROVENANCE_INCOMPLETE",
        }:
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=SearchFeedbackClassV2.DIAGNOSTIC_ONLY,
                    search_utility_event=None,
                    frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                    research_task=None,
                    controller_update_allowed=False,
                    meta_update_allowed=False,
                    search_memory_update_allowed=False,
                ),
                compact,
            )
        if disposition == "EXCLUDE_FROM_CURRENT_CLAIM":
            return (
                FusedSearchFeedbackV2(
                    candidate_id=adjudication.candidate_id,
                    search_feedback_class=SearchFeedbackClassV2.ENGINEERING_ONLY,
                    search_utility_event=None,
                    frontier_eligibility=FrontierEligibilityV2.EXCLUDED,
                    research_task=None,
                    controller_update_allowed=False,
                    meta_update_allowed=False,
                    search_memory_update_allowed=False,
                ),
                compact,
            )
        return self.no_search_update(adjudication.candidate_id), compact


class ValueOfInformationHelixAdmissionV30(DeterministicHelixAdmissionV13):
    """Keep Research learning native; use Guard only for extra allocation.

    V13 made multi-seed support a prerequisite for controller, meta, and
    frontier updates.  That coupled claim confidence to search learning and
    could deadlock a campaign: a candidate needed replication before Research
    could learn from it, while only a sufficiently large first result could
    request replication.  V30 separates those concerns.  Every exact valid
    development metric remains a normal Research observation.  Guard may add a
    replication/control task, but it cannot erase or down-weight the metric.
    """

    policy_digest = sha256_digest(
        {
            "policy": "ValueOfInformationHelixAdmissionV30",
            "valid_metric": "PRESERVE_NATIVE_RESEARCH_UPDATE",
            "guard_authority": "EXTRA_ALLOCATION_ONLY",
            "invalid_or_drifted": "EXCLUDE_SCIENCE_UPDATE",
        }
    )

    def admit_post(
        self,
        *,
        adjudication: PortAdjudication,
        search_utility_event: SearchUtilityEventV2,
        validation_task: ResearchTaskV1 | None = None,
        protocol_branch_task: ResearchTaskV1 | None = None,
        matched_control_task: ResearchTaskV1 | None = None,
        evidence_summary: EvidenceSummaryV1 | None = None,
    ) -> tuple[FusedSearchFeedbackV2, CompactFeedback | None]:
        fused, compact = super().admit_post(
            adjudication=adjudication,
            search_utility_event=search_utility_event,
            validation_task=validation_task,
            protocol_branch_task=protocol_branch_task,
            matched_control_task=matched_control_task,
            evidence_summary=evidence_summary,
        )
        if (
            evidence_summary is not None
            and evidence_summary.current_attempt_class == "VALID_METRIC"
        ):
            fused = replace(
                fused,
                search_feedback_class=SearchFeedbackClassV2.ADMITTED_SEARCH_RESULT,
                search_utility_event=search_utility_event,
                frontier_eligibility=FrontierEligibilityV2.SEARCH_ELIGIBLE,
                controller_update_allowed=True,
                meta_update_allowed=True,
                search_memory_update_allowed=True,
            )
        return fused, compact

def _schema_for_annotation(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any:
        return {}
    if annotation is type(None):
        return {"type": "null"}
    if origin in {Union, types.UnionType}:
        return {
            "anyOf": [
                _schema_for_annotation(item) for item in arguments
            ]
        }
    if origin in {tuple, Sequence, SequenceABC}:
        item_type = arguments[0] if arguments else Any
        return {
            "type": "array",
            "items": _schema_for_annotation(item_type),
        }
    if origin in {dict, Mapping, MappingABC}:
        value_type = arguments[1] if len(arguments) > 1 else Any
        return {
            "type": "object",
            "additionalProperties": _schema_for_annotation(value_type),
        }
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return {
            "type": "string",
            "enum": [item.value for item in annotation],
        }
    if hasattr(annotation, "__dataclass_fields__"):
        hints = get_type_hints(annotation)
        names = [item.name for item in fields(annotation)]
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: _schema_for_annotation(hints[name])
                for name in names
            },
            "required": names,
        }
    primitive = {
        str: "string",
        bool: "boolean",
        int: "integer",
        float: "number",
    }.get(annotation)
    return {"type": primitive} if primitive is not None else {}


def schema_for(record_type: type[Any]) -> dict[str, Any]:
    """Generate a closed JSON Schema from the authoritative dataclass."""

    if not hasattr(record_type, "__dataclass_fields__"):
        raise TypeError("schema_for accepts only V13 dataclass record types")
    schema = _schema_for_annotation(record_type)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": record_type.__name__,
        **schema,
        "x-recclaw-generated-from": (
            "recclaw_core.helix.scientific_attribution."
            + record_type.__name__
        ),
    }


V13_SCHEMA_TYPES: tuple[type[Any], ...] = (
    SearchUtilityEventV2,
    FusedSearchFeedbackV2,
    EvidenceSummaryV1,
    HelixControlProjectionV1,
    PromptFeedbackProjectionV2,
    GuardEvidenceSnapshotV1,
    ResearchTaskV1,
    ValidationResultBundleV1,
)


__all__ = [
    "ComparatorDeltaV2",
    "REQUESTED_CONTROL_KINDS_V1",
    "CURRENT_ATTEMPT_CLASSES_V1",
    "PROTOCOL_STATUSES_V1",
    "SCIENTIFIC_CONCLUSION_STATES_V1",
    "SIGN_CONSISTENCY_CLASSES_V1",
    "DeterministicHelixAdmissionV13",
    "EvidenceSummaryV1",
    "FrontierEligibilityV2",
    "FusedSearchFeedbackV2",
    "GuardEvidenceObservationV1",
    "GuardEvidenceSnapshotV1",
    "HelixControlProjectionV1",
    "NOT_AVAILABLE",
    "PromptFeedbackProjectionV2",
    "ResearchTaskQueueV1",
    "ResearchTaskStatusV1",
    "ResearchTaskTypeV1",
    "ResearchTaskV1",
    "SearchFeedbackClassV2",
    "SearchUtilityEventV2",
    "V13_SCHEMA_TYPES",
    "ValidationResultBundleV1",
    "ValueOfInformationHelixAdmissionV30",
    "schema_for",
]
