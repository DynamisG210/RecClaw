"""V13 scientific-attribution contracts and deterministic admission policy.

This is the single typed source of truth for the data that may cross from the
Helix-private Evidence Guard plane into Research search control.  Guard events
and CompactFeedback remain private inputs; the returned FusedSearchFeedbackV2
contains only common search utility and generic task state.
"""

from __future__ import annotations

import types
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
    return deep_freeze(snapshot_json(dict(value)))


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
        object.__setattr__(
            self,
            "resource_cost_projection",
            _closed_mapping(self.resource_cost_projection),
        )

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
                "metric_contract_digest": self.metric_contract_digest,
                "resource_cost_projection": deep_thaw(
                    self.resource_cost_projection
                ),
                "typed_blocker_class": self.typed_blocker_class,
                "observation_seed": self.observation_seed,
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

    def __post_init__(self) -> None:
        if not self.task_id or not self.candidate_id:
            raise ValueError("ResearchTaskV1 requires task and candidate identities")
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

    def __init__(self) -> None:
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

    def select_next(self) -> ResearchTaskV1 | None:
        pending = [
            item
            for item in self._tasks.values()
            if item.task_status is ResearchTaskStatusV1.PENDING
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


@dataclass(frozen=True, slots=True)
class GuardEvidenceObservationV1:
    candidate_semantic_digest: str
    protocol_digest: str
    comparator_identity: str
    observation_seed: str
    observation_id: str

    def __post_init__(self) -> None:
        validate_sha256(
            self.candidate_semantic_digest,
            field_name="candidate_semantic_digest",
        )
        validate_sha256(self.protocol_digest, field_name="protocol_digest")
        validate_sha256(self.observation_id, field_name="observation_id")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (
            self.candidate_semantic_digest,
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
            "observation_ids": [
                item.observation_id for item in self.observations
            ],
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
                "candidate_semantic_digest": (
                    self.candidate_semantic_digest
                ),
                "observation_seeds": self.observation_seeds,
                "search_utility_events": [
                    item.to_dict()
                    for item in self.search_utility_events
                ],
                "development_validation_status": (
                    self.development_validation_status
                ),
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

    def __post_init__(self) -> None:
        if self.search_feedback_class is SearchFeedbackClassV2.NO_SEARCH_UPDATE:
            if (
                self.search_utility_event is not None
                or self.research_task is not None
                or self.controller_update_allowed
                or self.meta_update_allowed
                or self.search_memory_update_allowed
                or self.frontier_eligibility is not FrontierEligibilityV2.EXCLUDED
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
            "preliminary": "SEARCH_ELIGIBLE_PRELIMINARY_AND_VALIDATION_TASK",
            "diagnostic": "NO_FRONTIER_NO_META",
            "invalid": "NO_SEARCH_UPDATE",
        }
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

    def admit_post(
        self,
        *,
        adjudication: PortAdjudication,
        search_utility_event: SearchUtilityEventV2,
        validation_task: ResearchTaskV1 | None = None,
        protocol_branch_task: ResearchTaskV1 | None = None,
    ) -> tuple[FusedSearchFeedbackV2, CompactFeedback | None]:
        if adjudication.stage is not PortStage.POST:
            raise ValueError("V13 POST admission requires POST adjudication")
        if adjudication.candidate_id != search_utility_event.candidate_id:
            raise ValueError("adjudication/search utility candidate mismatch")
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
            if protocol_branch_task is None:
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
                    controller_update_allowed=True,
                    meta_update_allowed=False,
                    search_memory_update_allowed=True,
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
    PromptFeedbackProjectionV2,
    GuardEvidenceSnapshotV1,
    ResearchTaskV1,
    ValidationResultBundleV1,
)


__all__ = [
    "ComparatorDeltaV2",
    "DeterministicHelixAdmissionV13",
    "FrontierEligibilityV2",
    "FusedSearchFeedbackV2",
    "GuardEvidenceObservationV1",
    "GuardEvidenceSnapshotV1",
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
    "schema_for",
]
