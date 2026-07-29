"""Canonical M6I ownership, identity, sharing, and round-state contracts.

This module is intentionally independent of Pilot version wrappers.  The
three-arm scheduler, real Pilot adapter, and future Main adapter all consume
the same state transitions and identity constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping

from .canonical import canonical_value, content_id, sha256_digest, validate_sha256
from .contracts import ArmCode


class IntegratedStateError(RuntimeError):
    """Base class for M6I contract violations."""


class OwnershipViolation(IntegratedStateError):
    """Raised when an Arm accesses state owned by another Arm."""


class CallSharingViolation(IntegratedStateError):
    """Raised when a Provider response would cross an unsafe boundary."""


class RoundTransitionViolation(IntegratedStateError):
    """Raised when a round takes a transition outside the canonical graph."""


class OwnershipScopeV1(str, Enum):
    COMMON_IMMUTABLE = "COMMON_IMMUTABLE"
    EXPERIMENT_SHARED_APPEND_ONLY = "EXPERIMENT_SHARED_APPEND_ONLY"
    ARM_PRIVATE = "ARM_PRIVATE"
    ROUND_LOCAL = "ROUND_LOCAL"
    EXPLICIT_PAIRED_CALL = "EXPLICIT_PAIRED_CALL"
    FORBIDDEN_GLOBAL_MUTABLE = "FORBIDDEN_GLOBAL_MUTABLE"


class CallSharingPolicyV1(str, Enum):
    ARM_PRIVATE = "ARM_PRIVATE"
    PAIRED_BC_EXACT_CONTEXT = "PAIRED_BC_EXACT_CONTEXT"
    COMMON_IMMUTABLE = "COMMON_IMMUTABLE"


class ParentBindingPolicyV1(str, Enum):
    EXPLICIT_ROOT_REQUEST = "EXPLICIT_ROOT_REQUEST"
    REQUIRE_EXACT_PRIOR_PARENT = "REQUIRE_EXACT_PRIOR_PARENT"
    OPTIONAL = "OPTIONAL"


class RoundStateV1(str, Enum):
    ROUND_READY = "ROUND_READY"
    PROPOSAL_PENDING = "PROPOSAL_PENDING"
    ROUTE_CREATED = "ROUTE_CREATED"
    ACTIVE_TASK_BOUND = "ACTIVE_TASK_BOUND"
    CANDIDATE_SELECTED = "CANDIDATE_SELECTED"
    EXECUTION_STARTED = "EXECUTION_STARTED"
    RESULT_CLOSED = "RESULT_CLOSED"
    OBSERVATION_ADMITTED = "OBSERVATION_ADMITTED"
    OBSERVATION_WITHHELD = "OBSERVATION_WITHHELD"
    NO_EXECUTION = "NO_EXECUTION"
    ROUND_TERMINAL = "ROUND_TERMINAL"


class ProposalSourceV1(str, Enum):
    NORMAL_ROUTED_PROPOSAL = "NORMAL_ROUTED_PROPOSAL"
    ACTIVE_BOUND_TASK = "ACTIVE_BOUND_TASK"
    ORIGINAL_CONTROLLER_PATH = "ORIGINAL_CONTROLLER_PATH"
    NO_PROPOSAL_TERMINAL = "NO_PROPOSAL_TERMINAL"


class ObservationPathV1(str, Enum):
    ADMITTED_OBSERVATION = "ADMITTED_OBSERVATION"
    WITHHELD_OBSERVATION = "WITHHELD_OBSERVATION"
    DIAGNOSTIC_OR_ENGINEERING_ONLY = "DIAGNOSTIC_OR_ENGINEERING_ONLY"
    NO_OBSERVATION = "NO_OBSERVATION"


def canonical_observation_path(
    *,
    meta_update_allowed: bool,
    search_feedback_class: str,
) -> ObservationPathV1:
    """Map fused feedback to the one Pilot/Main observation taxonomy."""

    if meta_update_allowed:
        return ObservationPathV1.ADMITTED_OBSERVATION
    if search_feedback_class in {
        "COMMON_FAILED_EXECUTION",
        "DIAGNOSTIC_ONLY",
        "ENGINEERING_ONLY",
        "PROTOCOL_BRANCH_TASK",
    }:
        return ObservationPathV1.DIAGNOSTIC_OR_ENGINEERING_ONLY
    return ObservationPathV1.WITHHELD_OBSERVATION


@dataclass(frozen=True, slots=True)
class ArmOwnerTokenV1:
    experiment_id: str
    arm: ArmCode
    opaque_arm_instance_id: str

    def __post_init__(self) -> None:
        if not self.experiment_id or not self.opaque_arm_instance_id:
            raise OwnershipViolation("Arm owner identities must be non-empty")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "arm": self.arm.value,
            "opaque_arm_instance_id": self.opaque_arm_instance_id,
        }


def _validate_context_digest(value: str, *, field_name: str) -> str:
    if value == "ABSENT":
        return value
    return validate_sha256(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class ProviderRequestContextV1:
    """Complete behaviorally relevant Provider-call identity preimage."""

    model_release_digest: str
    response_schema_digest: str
    temperature: float
    timeout_policy_digest: str
    producer_role: str
    prompt_bytes_digest: str
    complete_context_digest: str
    memory_view_digest: str
    meta_fast_state_digest: str
    lineage_view_digest: str
    active_task_digest: str
    research_task_queue_digest: str
    round_index: int
    search_seed: int
    response_arm_neutral: bool

    def __post_init__(self) -> None:
        if not self.producer_role or self.round_index < 1:
            raise CallSharingViolation("Provider context role/round is invalid")
        for name in (
            "model_release_digest",
            "response_schema_digest",
            "timeout_policy_digest",
            "prompt_bytes_digest",
            "complete_context_digest",
            "memory_view_digest",
            "meta_fast_state_digest",
            "lineage_view_digest",
            "active_task_digest",
            "research_task_queue_digest",
        ):
            _validate_context_digest(str(getattr(self, name)), field_name=name)

    @property
    def exact_request_digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "model_release_digest": self.model_release_digest,
                "response_schema_digest": self.response_schema_digest,
                "temperature": float(self.temperature),
                "timeout_policy_digest": self.timeout_policy_digest,
                "producer_role": self.producer_role,
                "prompt_bytes_digest": self.prompt_bytes_digest,
                "complete_context_digest": self.complete_context_digest,
                "memory_view_digest": self.memory_view_digest,
                "meta_fast_state_digest": self.meta_fast_state_digest,
                "lineage_view_digest": self.lineage_view_digest,
                "active_task_digest": self.active_task_digest,
                "research_task_queue_digest": self.research_task_queue_digest,
                "round_index": self.round_index,
                "search_seed": self.search_seed,
                "response_arm_neutral": self.response_arm_neutral,
            }
        )


@dataclass(frozen=True, slots=True)
class ProviderPhysicalCallIdV1:
    value: str
    sharing_policy: CallSharingPolicyV1
    exact_request_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ConsumerLogicalCallIdV1:
    value: str
    experiment_id: str
    opaque_arm_instance_id: str
    search_seed: int
    round_index: int
    producer_role: str
    provider_physical_call_id: str
    consumer_context_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class CandidateInstanceIdV1:
    value: str
    opaque_arm_instance_id: str
    round_index: int
    producer_role: str
    semantic_program_digest: str
    local_parent_or_task_identity: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class CanonicalParentBindingV1:
    """Runtime-owned candidate-parent binding for one logical call."""

    policy: ParentBindingPolicyV1
    runtime_parent_candidate_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(slots=True)
class CallSharingRegistryV1:
    """Append-only physical/consumer/candidate identity registry."""

    policy: CallSharingPolicyV1 = CallSharingPolicyV1.ARM_PRIVATE
    _physical: dict[str, ProviderPhysicalCallIdV1] = field(default_factory=dict)
    _physical_owner_digests: dict[str, set[str]] = field(default_factory=dict)
    _consumer: dict[str, ConsumerLogicalCallIdV1] = field(default_factory=dict)
    _candidates: dict[str, CandidateInstanceIdV1] = field(default_factory=dict)
    _candidate_owners: dict[str, str] = field(default_factory=dict)
    _audit: list[dict[str, Any]] = field(default_factory=list)

    def _lookup_preimage(
        self,
        *,
        owner: ArmOwnerTokenV1,
        context: ProviderRequestContextV1,
    ) -> dict[str, Any]:
        if self.policy is CallSharingPolicyV1.ARM_PRIVATE:
            return {
                "owner_digest": owner.digest,
                "exact_request_digest": context.exact_request_digest,
                "policy": self.policy.value,
            }
        if not context.response_arm_neutral:
            raise CallSharingViolation(
                "shared physical calls require an arm-neutral response"
            )
        if (
            self.policy is CallSharingPolicyV1.PAIRED_BC_EXACT_CONTEXT
            and owner.arm is ArmCode.A
        ):
            raise CallSharingViolation("Arm A cannot consume paired B/C calls")
        return {
            "exact_request_digest": context.exact_request_digest,
            "policy": self.policy.value,
        }

    def register_request(
        self,
        *,
        owner: ArmOwnerTokenV1,
        context: ProviderRequestContextV1,
    ) -> tuple[ProviderPhysicalCallIdV1, ConsumerLogicalCallIdV1, str]:
        lookup = self._lookup_preimage(owner=owner, context=context)
        lookup_digest = sha256_digest(lookup)
        physical = self._physical.get(lookup_digest)
        decision = "HIT" if physical is not None else "MISS"
        if physical is None:
            physical = ProviderPhysicalCallIdV1(
                value=content_id("provider-physical-v1", lookup),
                sharing_policy=self.policy,
                exact_request_digest=context.exact_request_digest,
            )
            self._physical[lookup_digest] = physical
            self._physical_owner_digests[physical.value] = set()
        elif physical.exact_request_digest != context.exact_request_digest:
            raise CallSharingViolation(
                "physical-call lookup collided across different contexts"
            )
        owners = self._physical_owner_digests[physical.value]
        if (
            owners
            and owner.digest not in owners
            and self.policy is CallSharingPolicyV1.ARM_PRIVATE
        ):
            raise CallSharingViolation("ARM_PRIVATE physical call crossed Arms")
        owners.add(owner.digest)
        consumer_payload = {
            "experiment_id": owner.experiment_id,
            "opaque_arm_instance_id": owner.opaque_arm_instance_id,
            "search_seed": context.search_seed,
            "round_index": context.round_index,
            "producer_role": context.producer_role,
            "provider_physical_call_id": physical.value,
            "consumer_context_digest": context.exact_request_digest,
        }
        consumer = ConsumerLogicalCallIdV1(
            value=content_id("consumer-logical-v1", consumer_payload),
            **consumer_payload,
        )
        prior = self._consumer.get(consumer.value)
        if prior is not None and prior != consumer:
            raise CallSharingViolation("consumer logical-call substitution")
        self._consumer[consumer.value] = consumer
        self._audit.append(
            canonical_value(
                {
                    "audit_index": len(self._audit),
                    "decision": decision,
                    "owner": owner.to_dict(),
                    "provider": physical.to_dict(),
                    "consumer": consumer.to_dict(),
                }
            )
        )
        return physical, consumer, decision

    def register_candidate(
        self,
        *,
        owner: ArmOwnerTokenV1,
        round_index: int,
        producer_role: str,
        semantic_program_digest: str,
        local_parent_or_task_identity: str | None,
    ) -> CandidateInstanceIdV1:
        validate_sha256(
            semantic_program_digest, field_name="semantic_program_digest"
        )
        local_parent = local_parent_or_task_identity or "ROOT"
        known_parent_owner = self._candidate_owners.get(local_parent)
        if local_parent.startswith("cand-") and known_parent_owner is None:
            raise CallSharingViolation(
                "candidate instance references an absent candidate parent"
            )
        if known_parent_owner is not None and known_parent_owner != owner.digest:
            raise CallSharingViolation(
                "candidate instance references a foreign Arm parent"
            )
        payload = {
            "opaque_arm_instance_id": owner.opaque_arm_instance_id,
            "round_index": int(round_index),
            "producer_role": producer_role,
            "semantic_program_digest": semantic_program_digest,
            "local_parent_or_task_identity": local_parent,
        }
        candidate = CandidateInstanceIdV1(
            value=f"cand-{sha256_digest(payload)}",
            **payload,
        )
        prior = self._candidates.get(candidate.value)
        if prior is not None and prior != candidate:
            raise CallSharingViolation("candidate-instance substitution")
        self._candidates[candidate.value] = candidate
        self._candidate_owners[candidate.value] = owner.digest
        self._audit.append(
            canonical_value(
                {
                    "audit_index": len(self._audit),
                    "candidate": candidate.to_dict(),
                    "decision": "CANDIDATE_INSTANCE_REGISTERED",
                    "owner": owner.to_dict(),
                }
            )
        )
        return candidate

    def resolve_candidate_parent(
        self,
        *,
        owner: ArmOwnerTokenV1,
        binding: CanonicalParentBindingV1,
        provider_parent_candidate_id: str | None,
    ) -> str | None:
        """Resolve a parent without letting Provider output author identity."""

        if provider_parent_candidate_id is not None:
            raise CallSharingViolation(
                "Provider-authored candidate parent identity is forbidden"
            )
        runtime_parent = binding.runtime_parent_candidate_id
        if (
            binding.policy is ParentBindingPolicyV1.EXPLICIT_ROOT_REQUEST
            and runtime_parent is not None
        ):
            raise CallSharingViolation(
                "explicit-root binding cannot carry a runtime parent"
            )
        if (
            binding.policy
            is ParentBindingPolicyV1.REQUIRE_EXACT_PRIOR_PARENT
            and runtime_parent is None
        ):
            raise CallSharingViolation(
                "exact-prior-parent binding requires a runtime parent"
            )
        if runtime_parent is None:
            return None
        known_parent_owner = self._candidate_owners.get(runtime_parent)
        if known_parent_owner is None:
            raise CallSharingViolation(
                "runtime parent is absent from the candidate registry"
            )
        if known_parent_owner != owner.digest:
            raise CallSharingViolation(
                "runtime parent belongs to a foreign Arm"
            )
        return runtime_parent

    @property
    def audit_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(canonical_value(item) for item in self._audit)

    def audit_projection(self) -> dict[str, Any]:
        cross_arm_physical = sum(
            len(owners) > 1
            for owners in self._physical_owner_digests.values()
        )
        return canonical_value(
            {
                "policy": self.policy.value,
                "physical_call_identities": len(self._physical),
                "consumer_logical_identities": len(self._consumer),
                "candidate_instance_identities": len(self._candidates),
                "cross_arm_physical_identities": cross_arm_physical,
                "records": self.audit_records,
            }
        )


@dataclass(frozen=True, slots=True)
class RoundBoundaryEventV1:
    owner: ArmOwnerTokenV1
    search_seed: int
    round_index: int
    proposal_source: ProposalSourceV1
    observation_path: ObservationPathV1
    terminal_class: str
    route_digest: str | None
    active_task_digest: str | None
    candidate_instance_id: str | None

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(slots=True)
class _MutableRoundStateV1:
    owner: ArmOwnerTokenV1
    search_seed: int
    round_index: int
    state: RoundStateV1 = RoundStateV1.ROUND_READY
    proposal_source: ProposalSourceV1 | None = None
    route_digest: str | None = None
    active_task_digest: str | None = None
    candidate_instance_id: str | None = None
    observation_path: ObservationPathV1 | None = None
    terminal_class: str | None = None
    meta_boundary_count: int = 0
    transitions: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class IntegratedCampaignStateCoreV1:
    """Single canonical state machine for Pilot and future Main adapters."""

    experiment_id: str
    _owners: dict[ArmCode, ArmOwnerTokenV1] = field(default_factory=dict)
    _rounds: dict[tuple[str, int, int], _MutableRoundStateV1] = field(
        default_factory=dict
    )
    _access_audit: list[dict[str, Any]] = field(default_factory=list)
    _triplet_barriers: set[tuple[int, int]] = field(default_factory=set)

    def bind_arms(self, arm_to_instance: Mapping[ArmCode, str]) -> None:
        if self._owners:
            raise OwnershipViolation("Arm owners are already bound")
        if set(arm_to_instance) != {ArmCode.A, ArmCode.B, ArmCode.C}:
            raise OwnershipViolation("exact A/B/C owner mapping is required")
        tokens = {
            arm: ArmOwnerTokenV1(
                experiment_id=self.experiment_id,
                arm=arm,
                opaque_arm_instance_id=str(arm_to_instance[arm]),
            )
            for arm in ArmCode
        }
        if len({item.opaque_arm_instance_id for item in tokens.values()}) != 3:
            raise OwnershipViolation("Arm instance identities must be distinct")
        self._owners.update(tokens)

    def owner(self, arm: ArmCode) -> ArmOwnerTokenV1:
        try:
            return self._owners[arm]
        except KeyError as error:
            raise OwnershipViolation("Arm owner is not bound") from error

    def assert_owner(
        self,
        *,
        accessor_arm: ArmCode,
        owner_token: ArmOwnerTokenV1,
        operation: str,
    ) -> None:
        expected = self.owner(accessor_arm)
        allowed = expected == owner_token
        self._access_audit.append(
            canonical_value(
                {
                    "accessor_arm": accessor_arm.value,
                    "allowed": allowed,
                    "operation": operation,
                    "owner_digest": owner_token.digest,
                }
            )
        )
        if not allowed:
            raise OwnershipViolation(
                f"{accessor_arm.value} attempted foreign Arm access: {operation}"
            )

    def _key(self, arm: ArmCode, search_seed: int, round_index: int) -> tuple[str, int, int]:
        return (self.owner(arm).digest, int(search_seed), int(round_index))

    def _record(
        self,
        item: _MutableRoundStateV1,
        state: RoundStateV1,
        *,
        reason: str,
    ) -> None:
        item.state = state
        item.transitions.append(
            canonical_value(
                {
                    "index": len(item.transitions),
                    "reason": reason,
                    "state": state.value,
                }
            )
        )

    def open_round(
        self, *, arm: ArmCode, search_seed: int, round_index: int
    ) -> ArmOwnerTokenV1:
        if round_index < 1:
            raise RoundTransitionViolation("round index must be positive")
        key = self._key(arm, search_seed, round_index)
        if key in self._rounds:
            raise RoundTransitionViolation("round is create-once")
        owner = self.owner(arm)
        item = _MutableRoundStateV1(
            owner=owner,
            search_seed=int(search_seed),
            round_index=int(round_index),
        )
        self._record(item, RoundStateV1.ROUND_READY, reason="OPEN_ROUND")
        self._record(
            item, RoundStateV1.PROPOSAL_PENDING, reason="REQUEST_PROPOSAL_SOURCE"
        )
        self._rounds[key] = item
        return owner

    def bind_proposal_source(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        source: ProposalSourceV1,
        route_digest: str | None = None,
        active_task_digest: str | None = None,
    ) -> None:
        item = self._require_state(
            arm, search_seed, round_index, {RoundStateV1.PROPOSAL_PENDING}
        )
        item.proposal_source = source
        item.route_digest = route_digest
        item.active_task_digest = active_task_digest
        if source is ProposalSourceV1.ACTIVE_BOUND_TASK:
            if active_task_digest is None:
                raise RoundTransitionViolation("active task source lacks task digest")
            self._record(
                item, RoundStateV1.ACTIVE_TASK_BOUND, reason=source.value
            )
        elif source is ProposalSourceV1.NO_PROPOSAL_TERMINAL:
            self._record(item, RoundStateV1.NO_EXECUTION, reason=source.value)
        else:
            if source is ProposalSourceV1.NORMAL_ROUTED_PROPOSAL:
                if route_digest is None:
                    raise RoundTransitionViolation(
                        "normal Research source lacks route digest"
                    )
            elif route_digest is not None:
                raise RoundTransitionViolation(
                    "Original controller source cannot bind a Research route"
                )
            self._record(item, RoundStateV1.ROUTE_CREATED, reason=source.value)

    def select_candidate(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        candidate_instance_id: str,
    ) -> None:
        item = self._require_state(
            arm,
            search_seed,
            round_index,
            {RoundStateV1.ROUTE_CREATED, RoundStateV1.ACTIVE_TASK_BOUND},
        )
        if not candidate_instance_id:
            raise RoundTransitionViolation("candidate instance identity is empty")
        item.candidate_instance_id = candidate_instance_id
        self._record(
            item, RoundStateV1.CANDIDATE_SELECTED, reason="SELECT_CANDIDATE"
        )

    def start_execution(
        self, *, arm: ArmCode, search_seed: int, round_index: int
    ) -> None:
        item = self._require_state(
            arm, search_seed, round_index, {RoundStateV1.CANDIDATE_SELECTED}
        )
        self._record(
            item, RoundStateV1.EXECUTION_STARTED, reason="START_EXECUTION"
        )

    def close_result(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        observation_path: ObservationPathV1,
    ) -> None:
        if observation_path is ObservationPathV1.NO_OBSERVATION:
            raise RoundTransitionViolation(
                "executed results require an explicit result observation path"
            )
        item = self._require_state(
            arm, search_seed, round_index, {RoundStateV1.EXECUTION_STARTED}
        )
        self._record(item, RoundStateV1.RESULT_CLOSED, reason="CLOSE_RESULT")
        item.observation_path = observation_path
        target = (
            RoundStateV1.OBSERVATION_ADMITTED
            if observation_path is ObservationPathV1.ADMITTED_OBSERVATION
            else RoundStateV1.OBSERVATION_WITHHELD
        )
        self._record(item, target, reason=observation_path.value)

    def close_no_execution(
        self, *, arm: ArmCode, search_seed: int, round_index: int
    ) -> None:
        item = self._round(arm, search_seed, round_index)
        if item.state not in {
            RoundStateV1.PROPOSAL_PENDING,
            RoundStateV1.ROUTE_CREATED,
            RoundStateV1.ACTIVE_TASK_BOUND,
            RoundStateV1.CANDIDATE_SELECTED,
            RoundStateV1.NO_EXECUTION,
        }:
            raise RoundTransitionViolation(
                f"no-execution closure is invalid from {item.state.value}"
            )
        item.observation_path = ObservationPathV1.NO_OBSERVATION
        if item.state is not RoundStateV1.NO_EXECUTION:
            self._record(item, RoundStateV1.NO_EXECUTION, reason="NO_EXECUTION")

    def terminalize(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        terminal_class: str,
        meta_boundary: Callable[[RoundBoundaryEventV1], None] | None = None,
    ) -> RoundBoundaryEventV1:
        item = self._round(arm, search_seed, round_index)
        if item.state not in {
            RoundStateV1.OBSERVATION_ADMITTED,
            RoundStateV1.OBSERVATION_WITHHELD,
            RoundStateV1.NO_EXECUTION,
        }:
            raise RoundTransitionViolation(
                f"terminal closure is invalid from {item.state.value}"
            )
        if item.proposal_source is None or item.observation_path is None:
            raise RoundTransitionViolation(
                "terminal round lacks proposal/observation classification"
            )
        if not terminal_class:
            raise RoundTransitionViolation("terminal class must be non-empty")
        event = RoundBoundaryEventV1(
            owner=item.owner,
            search_seed=item.search_seed,
            round_index=item.round_index,
            proposal_source=item.proposal_source,
            observation_path=item.observation_path,
            terminal_class=terminal_class,
            route_digest=item.route_digest,
            active_task_digest=item.active_task_digest,
            candidate_instance_id=item.candidate_instance_id,
        )
        if arm in {ArmCode.B, ArmCode.C}:
            if item.meta_boundary_count != 0:
                raise RoundTransitionViolation("Meta boundary is create-once")
            if meta_boundary is not None:
                meta_boundary(event)
            item.meta_boundary_count = 1
        elif meta_boundary is not None:
            raise RoundTransitionViolation("Arm A cannot receive a Meta boundary")
        item.terminal_class = terminal_class
        self._record(item, RoundStateV1.ROUND_TERMINAL, reason=terminal_class)
        return event

    def close_triplet(self, *, search_seed: int, round_index: int) -> None:
        if any(
            self._round(arm, search_seed, round_index).state
            is not RoundStateV1.ROUND_TERMINAL
            for arm in ArmCode
        ):
            raise RoundTransitionViolation(
                "triplet barrier requires terminal A/B/C rounds"
            )
        key = (int(search_seed), int(round_index))
        if key in self._triplet_barriers:
            raise RoundTransitionViolation("triplet barrier is create-once")
        self._triplet_barriers.add(key)

    def _round(
        self, arm: ArmCode, search_seed: int, round_index: int
    ) -> _MutableRoundStateV1:
        try:
            return self._rounds[self._key(arm, search_seed, round_index)]
        except KeyError as error:
            raise RoundTransitionViolation("round has not been opened") from error

    def _require_state(
        self,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        allowed: set[RoundStateV1],
    ) -> _MutableRoundStateV1:
        item = self._round(arm, search_seed, round_index)
        if item.state not in allowed:
            expected = ",".join(sorted(value.value for value in allowed))
            raise RoundTransitionViolation(
                f"{item.state.value} is not one of [{expected}]"
            )
        self.assert_owner(
            accessor_arm=arm,
            owner_token=item.owner,
            operation=f"round:{round_index}:{item.state.value}",
        )
        return item

    def round_projection(
        self, *, arm: ArmCode, search_seed: int, round_index: int
    ) -> dict[str, Any]:
        item = self._round(arm, search_seed, round_index)
        return canonical_value(
            {
                "owner": item.owner.to_dict(),
                "search_seed": item.search_seed,
                "round_index": item.round_index,
                "state": item.state.value,
                "proposal_source": (
                    item.proposal_source.value
                    if item.proposal_source is not None
                    else None
                ),
                "route_digest": item.route_digest,
                "active_task_digest": item.active_task_digest,
                "candidate_instance_id": item.candidate_instance_id,
                "observation_path": (
                    item.observation_path.value
                    if item.observation_path is not None
                    else None
                ),
                "terminal_class": item.terminal_class,
                "meta_boundary_count": item.meta_boundary_count,
                "transitions": item.transitions,
            }
        )

    def audit_projection(self) -> dict[str, Any]:
        rounds = [
            self.round_projection(
                arm=item.owner.arm,
                search_seed=item.search_seed,
                round_index=item.round_index,
            )
            for item in self._rounds.values()
        ]
        rounds.sort(
            key=lambda item: (
                item["search_seed"],
                item["round_index"],
                item["owner"]["arm"],
            )
        )
        return canonical_value(
            {
                "schema": "recclaw.m6i.integrated-campaign-state.v1",
                "experiment_id": self.experiment_id,
                "owners": [
                    self._owners[arm].to_dict() for arm in ArmCode
                ],
                "rounds": rounds,
                "triplet_barriers": [
                    {"search_seed": seed, "round_index": round_index}
                    for seed, round_index in sorted(self._triplet_barriers)
                ],
                "access_records": self._access_audit,
                "cross_arm_reads": sum(
                    not item["allowed"] for item in self._access_audit
                ),
            }
        )


__all__ = [
    "ArmOwnerTokenV1",
    "CallSharingPolicyV1",
    "CallSharingRegistryV1",
    "CallSharingViolation",
    "CanonicalParentBindingV1",
    "CandidateInstanceIdV1",
    "canonical_observation_path",
    "ConsumerLogicalCallIdV1",
    "IntegratedCampaignStateCoreV1",
    "IntegratedStateError",
    "ObservationPathV1",
    "OwnershipScopeV1",
    "OwnershipViolation",
    "ParentBindingPolicyV1",
    "ProposalSourceV1",
    "ProviderPhysicalCallIdV1",
    "ProviderRequestContextV1",
    "RoundBoundaryEventV1",
    "RoundStateV1",
    "RoundTransitionViolation",
]
