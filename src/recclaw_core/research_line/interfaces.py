"""Minimal shared inputs for the unified Research Line runtime.

These objects join existing production components; they do not replace the
OpenSpec, capability, Search, Episode, memory, or policy contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    OpenResearchSpecV1,
)


class ResearchLineInterfaceError(ValueError):
    """Raised when a shared production input is internally inconsistent."""


class ResearchTaskOperationV2(str, Enum):
    """Durable scientific work units understood by the Research Line."""

    NEW_SEED = "NEW_SEED"
    MATCHED_CONTROL = "MATCHED_CONTROL"
    MECHANISM_OFF = "MECHANISM_OFF"
    REPAIR = "REPAIR"
    REPRODUCE = "REPRODUCE"
    MOVE_ON = "MOVE_ON"


class ResearchTaskStatusV2(str, Enum):
    """Lifecycle state for a durable task, separate from legacy task slots."""

    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    SATISFIED = "SATISFIED"
    CLOSED = "CLOSED"


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ResearchLineInterfaceError(f"{field_name} must be normalized and non-empty")
    return value


def _snapshot(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ResearchLineInterfaceError(f"{field_name} must be a mapping")
    try:
        return canonical_value(dict(value))
    except (TypeError, ValueError) as error:
        raise ResearchLineInterfaceError(f"{field_name} is not canonicalizable") from error


@dataclass(frozen=True, slots=True)
class ResearchTaskRecordV2:
    """One durable, provenance-bearing task in the shared Research queue.

    The existing ``ResearchTaskV1`` remains the prompt/runtime compatibility
    projection.  This record carries the operation identity and closure state
    that the old single task slot cannot represent.
    """

    task_id: str
    operation: ResearchTaskOperationV2 | str
    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str
    parent_candidate_id: str | None
    comparator_identity: str
    protocol_digest: str
    required_seed_or_control: str
    priority: float
    created_round: int
    evidence_present: tuple[str, ...] = ()
    missing_seed_count: int = 1
    mechanism_program: Mapping[str, Any] = field(default_factory=dict)
    status: ResearchTaskStatusV2 | str = ResearchTaskStatusV2.PENDING
    producer_role: str | None = None
    provenance_digest: str | None = None
    deadline_round: int | None = None
    close_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    schema = "recclaw.research-line.task-record.v2"

    def __post_init__(self) -> None:
        _nonempty(self.task_id, field_name="task_id")
        _nonempty(self.candidate_id, field_name="candidate_id")
        _nonempty(self.comparator_identity, field_name="comparator_identity")
        _nonempty(
            self.required_seed_or_control,
            field_name="required_seed_or_control",
        )
        try:
            operation = (
                self.operation
                if isinstance(self.operation, ResearchTaskOperationV2)
                else ResearchTaskOperationV2(str(self.operation))
            )
        except ValueError as error:
            raise ResearchLineInterfaceError(
                "operation is not a ResearchTaskOperationV2"
            ) from error
        try:
            status = (
                self.status
                if isinstance(self.status, ResearchTaskStatusV2)
                else ResearchTaskStatusV2(str(self.status))
            )
        except ValueError as error:
            raise ResearchLineInterfaceError(
                "status is not a ResearchTaskStatusV2"
            ) from error
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "status", status)
        for field_name in (
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "protocol_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        if self.parent_candidate_id is not None:
            _nonempty(self.parent_candidate_id, field_name="parent_candidate_id")
        if self.producer_role is not None and self.producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError(
                "producer_role is outside the four-role portfolio"
            )
        if self.provenance_digest is not None:
            object.__setattr__(
                self,
                "provenance_digest",
                validate_sha256(self.provenance_digest, field_name="provenance_digest"),
            )
        if self.close_reason is not None:
            _nonempty(self.close_reason, field_name="close_reason")
        if self.created_round < 1 or self.missing_seed_count < 0:
            raise ResearchLineInterfaceError("task round/count is invalid")
        if self.deadline_round is not None and self.deadline_round < self.created_round:
            raise ResearchLineInterfaceError("deadline_round precedes created_round")
        if isinstance(self.priority, bool):
            raise ResearchLineInterfaceError("priority must be numeric")
        try:
            priority = float(self.priority)
        except (TypeError, ValueError) as error:
            raise ResearchLineInterfaceError("priority must be numeric") from error
        if not math.isfinite(priority) or not 0.0 <= priority <= 1.0:
            raise ResearchLineInterfaceError("priority must be finite and in [0,1]")
        object.__setattr__(self, "priority", priority)
        object.__setattr__(
            self,
            "evidence_present",
            tuple(dict.fromkeys(str(item) for item in self.evidence_present)),
        )
        object.__setattr__(
            self,
            "metadata",
            _snapshot(self.metadata, field_name="metadata"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchTaskRecordV2":
        if not isinstance(value, Mapping):
            raise ResearchLineInterfaceError("task record must be a mapping")
        return cls(
            task_id=value.get("task_id"),
            operation=value.get("operation"),
            candidate_id=value.get("candidate_id"),
            candidate_semantic_digest=value.get("candidate_semantic_digest"),
            mechanism_program_digest=value.get("mechanism_program_digest"),
            parent_candidate_id=value.get("parent_candidate_id"),
            comparator_identity=value.get("comparator_identity"),
            protocol_digest=value.get("protocol_digest"),
            required_seed_or_control=value.get("required_seed_or_control"),
            priority=value.get("priority", 0.5),
            created_round=value.get("created_round", 1),
            evidence_present=tuple(value.get("evidence_present", ())),
            missing_seed_count=value.get("missing_seed_count", 1),
            mechanism_program=value.get("mechanism_program", {}),
            status=value.get("status", ResearchTaskStatusV2.PENDING),
            producer_role=value.get("producer_role"),
            provenance_digest=value.get("provenance_digest"),
            deadline_round=value.get("deadline_round"),
            close_reason=value.get("close_reason"),
            metadata=value.get("metadata", {}),
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "task_id": self.task_id,
                "operation": self.operation.value,
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "parent_candidate_id": self.parent_candidate_id,
                "comparator_identity": self.comparator_identity,
                "protocol_digest": self.protocol_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "priority": self.priority,
                "created_round": self.created_round,
                "evidence_present": self.evidence_present,
                "missing_seed_count": self.missing_seed_count,
                "mechanism_program": self.mechanism_program,
                "status": self.status.value,
                "producer_role": self.producer_role,
                "provenance_digest": self.provenance_digest,
                "deadline_round": self.deadline_round,
                "close_reason": self.close_reason,
                "metadata": self.metadata,
            }
        )

    def legacy_task_type(self) -> str:
        return {
            ResearchTaskOperationV2.NEW_SEED: "VALIDATE_SAME_CANDIDATE",
            ResearchTaskOperationV2.MATCHED_CONTROL: "RUN_MATCHED_CONTROL",
            ResearchTaskOperationV2.MECHANISM_OFF: "RUN_ABLATION",
            ResearchTaskOperationV2.REPAIR: "REPAIR_IMPLEMENTATION",
            ResearchTaskOperationV2.REPRODUCE: "VALIDATE_SAME_CANDIDATE",
            ResearchTaskOperationV2.MOVE_ON: "PROTOCOL_BRANCH_DIAGNOSTIC",
        }[self.operation]

    def prompt_projection(self) -> dict[str, Any]:
        """Return the closed legacy task slot consumed by current runtime code."""

        status = {
            ResearchTaskStatusV2.PENDING: "PENDING",
            ResearchTaskStatusV2.ACTIVE: "ACTIVE",
            ResearchTaskStatusV2.SATISFIED: "COMPLETED",
            ResearchTaskStatusV2.CLOSED: "CANCELLED",
        }[self.status]
        return canonical_value(
            {
                "task_type": self.legacy_task_type(),
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "required_seed_or_control": self.required_seed_or_control,
                "task_status": status,
            }
        )

    def to_legacy_task(self):
        """Materialize the existing ResearchTaskV1 compatibility projection."""

        from recclaw_core.helix.scientific_attribution import (
            ResearchTaskStatusV1,
            ResearchTaskTypeV1,
            ResearchTaskV1,
        )

        legacy_status = {
            ResearchTaskStatusV2.PENDING: ResearchTaskStatusV1.PENDING,
            ResearchTaskStatusV2.ACTIVE: ResearchTaskStatusV1.ACTIVE,
            ResearchTaskStatusV2.SATISFIED: ResearchTaskStatusV1.COMPLETED,
            ResearchTaskStatusV2.CLOSED: ResearchTaskStatusV1.CANCELLED,
        }[self.status]
        return ResearchTaskV1(
            task_id=self.task_id,
            task_type=ResearchTaskTypeV1(self.legacy_task_type()),
            candidate_id=self.candidate_id,
            candidate_semantic_digest=self.candidate_semantic_digest,
            mechanism_program_digest=self.mechanism_program_digest,
            parent_candidate_id=self.parent_candidate_id,
            comparator_identity=self.comparator_identity,
            protocol_digest=self.protocol_digest,
            required_seed_or_control=self.required_seed_or_control,
            task_status=legacy_status,
            created_round=self.created_round,
            utility_priority=self.priority,
            missing_seed_count=self.missing_seed_count,
            mechanism_program=self.mechanism_program,
            owner_arm_instance_id=self.metadata.get("owner_arm_instance_id"),
        )


@dataclass(frozen=True, slots=True)
class ResearchTaskQueueV2:
    """Immutable durable queue with deterministic priority and lifecycle APIs."""

    tasks: tuple[ResearchTaskRecordV2, ...] = ()

    schema = "recclaw.research-line.task-queue.v2"

    _OPERATION_ORDER = {
        ResearchTaskOperationV2.MATCHED_CONTROL: 0,
        ResearchTaskOperationV2.MECHANISM_OFF: 1,
        ResearchTaskOperationV2.NEW_SEED: 2,
        ResearchTaskOperationV2.REPRODUCE: 3,
        ResearchTaskOperationV2.REPAIR: 0,
        ResearchTaskOperationV2.MOVE_ON: 4,
    }

    def __post_init__(self) -> None:
        normalized = tuple(
            item
            if isinstance(item, ResearchTaskRecordV2)
            else ResearchTaskRecordV2.from_dict(item)
            for item in self.tasks
        )
        if len({item.task_id for item in normalized}) != len(normalized):
            raise ResearchLineInterfaceError("task queue repeats a task_id")
        object.__setattr__(
            self,
            "tasks",
            tuple(sorted(normalized, key=lambda item: item.task_id)),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any] | None) -> "ResearchTaskQueueV2":
        if value is None:
            return cls()
        if not isinstance(value, Mapping):
            raise ResearchLineInterfaceError("task_queue must be a mapping")
        raw_tasks = value.get("tasks", ())
        if isinstance(raw_tasks, (str, bytes)) or not isinstance(raw_tasks, (tuple, list)):
            raise ResearchLineInterfaceError("task_queue.tasks must be a sequence")
        return cls(tuple(ResearchTaskRecordV2.from_dict(item) for item in raw_tasks))

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "tasks": tuple(item.to_dict() for item in self.tasks),
            }
        )

    def get(self, task_id: str) -> ResearchTaskRecordV2 | None:
        return next((item for item in self.tasks if item.task_id == task_id), None)

    def enqueue(self, task: ResearchTaskRecordV2) -> "ResearchTaskQueueV2":
        if not isinstance(task, ResearchTaskRecordV2):
            raise ResearchLineInterfaceError("task queue accepts ResearchTaskRecordV2")
        prior = self.get(task.task_id)
        if prior is None:
            return ResearchTaskQueueV2((*self.tasks, task))
        identity_fields = (
            "operation",
            "candidate_id",
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "parent_candidate_id",
            "comparator_identity",
            "required_seed_or_control",
            "protocol_digest",
        )
        if any(getattr(prior, field_name) != getattr(task, field_name) for field_name in identity_fields):
            raise ResearchLineInterfaceError("task identity substitution")
        if prior.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            return self
        merged = replace(
            prior,
            priority=max(prior.priority, task.priority),
            evidence_present=tuple(
                dict.fromkeys((*prior.evidence_present, *task.evidence_present))
            ),
            metadata={**prior.metadata, **task.metadata},
        )
        return ResearchTaskQueueV2(
            tuple(merged if item.task_id == task.task_id else item for item in self.tasks)
        )

    def select_next(self) -> ResearchTaskRecordV2 | None:
        pending = [
            item for item in self.tasks if item.status is ResearchTaskStatusV2.PENDING
        ]
        if not pending:
            return None
        return min(
            pending,
            key=lambda item: (
                self._OPERATION_ORDER[item.operation],
                -item.priority,
                item.created_round,
                item.deadline_round if item.deadline_round is not None else 2**31,
                item.task_id,
            ),
        )

    def activate(self, task_id: str) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None or task.status is not ResearchTaskStatusV2.PENDING:
            raise ResearchLineInterfaceError("only a pending task can be activated")
        return self._replace(
            replace(task, status=ResearchTaskStatusV2.ACTIVE)
        )

    def satisfy(
        self,
        task_id: str,
        *,
        evidence: tuple[str, ...] = (),
        reason: str = "REQUIRED_EVIDENCE_OBSERVED",
    ) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None or task.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            raise ResearchLineInterfaceError("only a pending or active task can be satisfied")
        return self._replace(
            replace(
                task,
                status=ResearchTaskStatusV2.SATISFIED,
                evidence_present=tuple(dict.fromkeys((*task.evidence_present, *evidence))),
                close_reason=reason,
            )
        )

    def close(
        self,
        task_id: str,
        *,
        reason: str = "CLOSED_BY_RESEARCH_POLICY",
    ) -> "ResearchTaskQueueV2":
        task = self.get(task_id)
        if task is None:
            raise ResearchLineInterfaceError("cannot close an unknown task")
        if task.status is ResearchTaskStatusV2.CLOSED:
            return self
        if task.status is ResearchTaskStatusV2.SATISFIED:
            return self
        return self._replace(
            replace(task, status=ResearchTaskStatusV2.CLOSED, close_reason=reason)
        )

    def _replace(self, task: ResearchTaskRecordV2) -> "ResearchTaskQueueV2":
        return ResearchTaskQueueV2(
            tuple(task if item.task_id == task.task_id else item for item in self.tasks)
        )


@dataclass(frozen=True, slots=True)
class ResearchContext:
    """One arm-local context consumed throughout one Research Line round."""

    campaign_id: str
    round_index: int
    knowledge_base: Mapping[str, Any]
    frozen_goal: Mapping[str, Any]
    frontier: Mapping[str, Any]
    scientific_memory: Mapping[str, Any]
    unresolved_questions: tuple[Mapping[str, Any], ...]
    policy: Mapping[str, Any]
    budget: Mapping[str, Any]
    active_profile_ref: str
    active_profile_digest: str
    protocol_ref: str
    protocol_digest: str

    schema = "recclaw.research-line.context.v1"

    def __post_init__(self) -> None:
        _nonempty(self.campaign_id, field_name="campaign_id")
        if self.round_index < 1:
            raise ResearchLineInterfaceError("round_index must be positive")
        for field_name in ("active_profile_ref", "protocol_ref"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for field_name in ("active_profile_digest", "protocol_digest"):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "knowledge_base",
            "frozen_goal",
            "frontier",
            "scientific_memory",
            "policy",
            "budget",
        ):
            object.__setattr__(
                self,
                field_name,
                _snapshot(getattr(self, field_name), field_name=field_name),
            )
        questions = []
        for index, question in enumerate(self.unresolved_questions):
            questions.append(
                _snapshot(question, field_name=f"unresolved_questions[{index}]")
            )
        object.__setattr__(self, "unresolved_questions", tuple(questions))

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "knowledge_base": self.knowledge_base,
                "frozen_goal": self.frozen_goal,
                "frontier": self.frontier,
                "scientific_memory": self.scientific_memory,
                "unresolved_questions": self.unresolved_questions,
                "policy": self.policy,
                "budget": self.budget,
                "active_profile_ref": self.active_profile_ref,
                "active_profile_digest": self.active_profile_digest,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    @property
    def context_ref(self) -> str:
        return f"research-context:{self.campaign_id}:round-{self.round_index}"

    def producer_view(self, producer_role: str) -> dict[str, Any]:
        """Return shared context plus only that role's provenance memory.

        Older Context payloads used ``by_role`` as a partial projection and
        omitted a role when it had no history.  Those payloads retain their
        historical fallback behavior.  Successor Contexts use the explicit
        ``global_memory``/``by_role`` split, so common feedback is no longer
        copied into all four role tails.
        """

        if producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError("producer_role is outside the four-role portfolio")
        allocation = dict(self.policy.get("producer_token_allocation", ()))
        explicit_global = self.scientific_memory.get("global_memory")
        if not isinstance(explicit_global, Mapping):
            explicit_global = self.scientific_memory.get("global")
        has_explicit_global = isinstance(explicit_global, Mapping)
        shared_global = (
            dict(explicit_global)
            if has_explicit_global
            else {
                key: value
                for key, value in self.scientific_memory.items()
                if key != "by_role"
            }
        )
        role_table = self.scientific_memory.get("by_role")
        if isinstance(role_table, Mapping) and producer_role in role_table:
            candidate_role_memory = role_table.get(producer_role)
            role_memory = (
                candidate_role_memory
                if isinstance(candidate_role_memory, Mapping)
                else {}
            )
        elif has_explicit_global:
            role_memory = {
                "producer_role": producer_role,
                "provenance": {"history": "EMPTY"},
            }
        else:
            # Compatibility for pre-V2 contexts that did not materialize every
            # role's independent history yet.
            role_memory = self.scientific_memory
        return canonical_value(
            {
                "context_ref": self.context_ref,
                "context_digest": self.digest,
                "campaign_id": self.campaign_id,
                "round_index": self.round_index,
                "producer_role": producer_role,
                "knowledge_base": self.knowledge_base,
                "frozen_goal": self.frozen_goal,
                "frontier": self.frontier,
                "scientific_memory": self.scientific_memory,
                "memory": role_memory,
                "global_memory": shared_global,
                "role_memory": role_memory,
                "unresolved_questions": self.unresolved_questions,
                "policy": self.policy,
                "budget": self.budget,
                "active_profile_ref": self.active_profile_ref,
                "active_profile_digest": self.active_profile_digest,
                "protocol_ref": self.protocol_ref,
                "protocol_digest": self.protocol_digest,
                "producer_token_fraction": float(allocation.get(producer_role, 0.0)),
                "mechanism_axis_targeting": self.policy.get(
                    "mechanism_axis_targeting", ()
                ),
                "memory_retrieval_policy": self.policy.get(
                    "memory_retrieval_policy", "UNSPECIFIED"
                ),
            }
        )

    @property
    def producer_inputs_digest(self) -> str:
        """Digest the exact four Producer inputs used in this round."""

        return sha256_digest(
            tuple(self.producer_view(role) for role in DISCOVERY_PRODUCERS)
        )


@dataclass(frozen=True, slots=True)
class ProducerOutcome:
    """One role's OpenSpec or typed failure, ready for Capability resolution."""

    producer_role: str
    context_ref: str
    context_digest: str
    spec: OpenResearchSpecV1 | None
    resolution_facts: Mapping[str, Any]
    source_proposal: CandidateProposalV4 | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    schema = "recclaw.research-line.producer-outcome.v1"

    def __post_init__(self) -> None:
        if self.producer_role not in DISCOVERY_PRODUCERS:
            raise ResearchLineInterfaceError("producer_role is outside the four-role portfolio")
        _nonempty(self.context_ref, field_name="context_ref")
        object.__setattr__(
            self,
            "context_digest",
            validate_sha256(self.context_digest, field_name="context_digest"),
        )
        has_spec = self.spec is not None
        has_failure = self.failure_code is not None
        if has_spec == has_failure:
            raise ResearchLineInterfaceError("ProducerOutcome requires exactly one spec or failure")
        if has_spec:
            if not isinstance(self.spec, OpenResearchSpecV1):
                raise ResearchLineInterfaceError("spec must be OpenResearchSpecV1")
            if (
                self.spec.producer_role != self.producer_role
                or self.spec.context_ref != self.context_ref
                or self.spec.context_digest != self.context_digest
            ):
                raise ResearchLineInterfaceError("OpenSpec is not bound to its Producer context")
            if self.source_proposal is not None and not isinstance(
                self.source_proposal, CandidateProposalV4
            ):
                raise ResearchLineInterfaceError("source_proposal must be CandidateProposalV4")
        else:
            _nonempty(str(self.failure_code), field_name="failure_code")
            if self.source_proposal is not None:
                raise ResearchLineInterfaceError("failed Producer cannot carry a source proposal")
        object.__setattr__(
            self,
            "resolution_facts",
            _snapshot(self.resolution_facts, field_name="resolution_facts"),
        )
        object.__setattr__(
            self,
            "provenance",
            _snapshot(self.provenance, field_name="provenance"),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "producer_role": self.producer_role,
                "context_ref": self.context_ref,
                "context_digest": self.context_digest,
                "spec": self.spec.to_dict() if self.spec is not None else None,
                "resolution_facts": self.resolution_facts,
                "source_proposal": (
                    self.source_proposal.to_dict()
                    if self.source_proposal is not None
                    else None
                ),
                "failure_code": self.failure_code,
                "failure_detail": self.failure_detail,
                "provenance": self.provenance,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())


@dataclass(frozen=True, slots=True)
class BehaviorProjection:
    """Decision inputs whose change proves that feedback affected a later round."""

    round_index: int
    context_ref: str
    context_digest: str
    profile_ref: str
    profile_digest: str
    policy_digest: str
    producer_inputs_digest: str
    producer_allocation: tuple[tuple[str, float], ...]
    axis_priorities: tuple[str, ...]
    memory_retrieval_policy: str
    acquisition_parameters: Mapping[str, Any]
    implementation_risk: Mapping[str, Any]

    schema = "recclaw.research-line.behavior-projection.v1"

    def __post_init__(self) -> None:
        if self.round_index < 1:
            raise ResearchLineInterfaceError("round_index must be positive")
        for field_name in ("context_ref", "profile_ref", "memory_retrieval_policy"):
            _nonempty(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "context_digest",
            "profile_digest",
            "policy_digest",
            "producer_inputs_digest",
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), field_name=field_name),
            )
        allocation = tuple(
            sorted((str(role), float(weight)) for role, weight in self.producer_allocation)
        )
        if set(role for role, _weight in allocation) != set(DISCOVERY_PRODUCERS):
            raise ResearchLineInterfaceError("producer_allocation must cover all four roles")
        object.__setattr__(self, "producer_allocation", allocation)
        object.__setattr__(
            self,
            "axis_priorities",
            tuple(dict.fromkeys(str(axis) for axis in self.axis_priorities)),
        )
        object.__setattr__(
            self,
            "acquisition_parameters",
            _snapshot(self.acquisition_parameters, field_name="acquisition_parameters"),
        )
        object.__setattr__(
            self,
            "implementation_risk",
            _snapshot(self.implementation_risk, field_name="implementation_risk"),
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "round_index": self.round_index,
                "context_ref": self.context_ref,
                "context_digest": self.context_digest,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "policy_digest": self.policy_digest,
                "producer_inputs_digest": self.producer_inputs_digest,
                "producer_allocation": self.producer_allocation,
                "axis_priorities": self.axis_priorities,
                "memory_retrieval_policy": self.memory_retrieval_policy,
                "acquisition_parameters": self.acquisition_parameters,
                "implementation_risk": self.implementation_risk,
            }
        )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def changed_fields(self, successor: "BehaviorProjection") -> tuple[str, ...]:
        if successor.round_index <= self.round_index:
            raise ResearchLineInterfaceError("successor must belong to a later round")
        ignored = {"schema", "round_index", "context_ref", "context_digest"}
        before = self.to_dict()
        after = successor.to_dict()
        return tuple(
            key
            for key in sorted(before)
            if key not in ignored and before[key] != after[key]
        )
