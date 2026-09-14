"""Small checkpointed multi-round boundary for the Research Line runtime.

This module owns only arm-local state and round persistence.  Provider,
qualification, and experiment composition remain injected by the caller; one
round is still executed by :func:`run_research_round`.
"""

from __future__ import annotations

import json
import os
import copyreg
from dataclasses import dataclass, field, replace
import hashlib
import math
from pathlib import Path
import pickle
import tempfile
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    content_id,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_EVALUATOR,
    COMMON_SPLIT,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemorySnapshotV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DISCOVERY_PRODUCERS,
)
from recclaw_core.helix.scientific_attribution import SearchUtilityEventV2
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityResolutionResultV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    is_qualified_open_spec_candidate,
    QualifiedSearchCandidateProtocolV1,
    SearchExecutableProfileV1,
    SearchProfileActivationV1,
    activate_next_fresh_search_profile,
    freeze_experiment_slate,
)

from .interfaces import (
    ProducerOutcome,
    ResearchContext,
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
    ResearchTaskStatusV2,
    project_provider_context_view,
)
from .bootstrap import bootstrap_search_pool
from .interpreter import (
    MissingSearchInterpretation,
    _behavior,
    interpret_missing_search_opportunity,
)
from .portfolio import PortfolioCandidateV2
from .profile_source import ResearchProfileSourceV1
from .producers import ResearchProducer, produce_research_specs
from .search_space_adapter import SearchSpaceAdapter
from .single_parent_search import is_single_parent_context
from .runtime import (
    CandidateHandoffFactory,
    EvidenceGuardPort,
    ExperimentRunner,
    ImplementerGateway,
    InnovationRuntimeInputs,
    MetaResearchInputs,
    PreparedResearchRoundV1,
    RoundAttemptV1,
    ResearchRoundResult,
    _attempt_retires_executable_identity,
    _default_search_space_adapter,
    _discovery_feedback_task,
    _existing_exact_feedback_observation_rebind,
    _metric_executable_identity_digest,
    _prepared_has_unfinished_producer_failure,
    _prepared_producer_roles,
    _prepared_has_recoverable_resource_failure,
    _prepared_has_external_implementation_failure,
    _is_implementation_admission_failure,
    _prepared_has_untried_candidate_local_innovation,
    _innovation_resource_failure_scope,
    _innovation_attempt_proposal_digest,
    _resolve_confirmation,
    _search_ranking_inputs,
    _verification_feedback_task,
    activate_promoted_meta_strategy,
    activate_staged_innovation,
    normalize_fidelity_utility_state,
    run_research_round,
)


def _reduce_mapping_proxy(value: MappingProxyType) -> tuple[type[dict[Any, Any]], tuple[dict[Any, Any]]]:
    return dict, (dict(value),)


copyreg.pickle(MappingProxyType, _reduce_mapping_proxy)


class CampaignError(RuntimeError):
    """Raised when a persisted campaign cannot be resumed safely."""


MAX_DISCOVERY_GENERATIONS_PER_ROUND = 6
PHYSICAL_CONTEXT_SCHEMA = "recclaw.research-line.physical-execution-context.v1"
RESEARCH_CONTEXT_PROJECTION_VERSION = 1
RESEARCH_CONTEXT_PROJECTION_SCHEMA = (
    "recclaw.research-line.effect-feedback-projection.v1"
)
ORIGINAL_ACTIVE_PARENT_PROJECTION_VERSION = 2
DEFERRED_INNOVATION_BACKLOG_LIMIT = 16


def _discovery_generation(context: ResearchContext) -> int:
    """Return the same-round discovery generation carried by Context memory."""

    value = context.scientific_memory.get("discovery_generation", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignError("discovery_generation must be a non-negative integer")
    return value


def _research_context_projection_digest(context: ResearchContext) -> str:
    """Identify the effect feedback whose Provider projection this source owns."""

    provider_view = project_provider_context_view(
        context.producer_view(DISCOVERY_PRODUCERS[0])
    )
    return sha256_digest(
        canonical_value(
            {
                "schema": RESEARCH_CONTEXT_PROJECTION_SCHEMA,
                "directional_search_utility": provider_view.get(
                    "memory", {}
                ).get("directional_search_utility"),
            }
        )
    )


def _is_original_matched_context(context: ResearchContext) -> bool:
    global_memory = context.scientific_memory.get("global_memory")
    return isinstance(global_memory, Mapping) and isinstance(
        global_memory.get("original_matched_controller_state"), Mapping
    )


def _stale_original_active_parent_projection(
    *,
    context: ResearchContext,
    prepared: PreparedResearchRoundV1,
    checkpoint_payload: Mapping[str, Any],
) -> bool:
    """Identify the pre-fix Original portfolio that ignored an active parent."""

    if (
        checkpoint_payload.get("original_active_parent_projection_version")
        == ORIGINAL_ACTIVE_PARENT_PROJECTION_VERSION
        or not _is_original_matched_context(context)
        or prepared.search_bindings
        or len(prepared.producer_outcomes) != len(DISCOVERY_PRODUCERS)
    ):
        return False
    return {outcome.producer_role for outcome in prepared.producer_outcomes} == set(
        DISCOVERY_PRODUCERS
    ) and all(
        outcome.failure_code == "OPEN_SPEC_PROJECTION_FAILED"
        and isinstance(outcome.failure_detail, str)
        and "parent binding differs from the activated lineage parent"
        in outcome.failure_detail
        for outcome in prepared.producer_outcomes
    )

_OPEN_SPEC_DRAFT_BASE_FIELDS = (
    "producer_role",
    "hypothesis",
    "mechanism_change",
    "competing_explanation",
    "matched_control_requirement",
    "implementation_requirements",
    "expected_evidence",
    "falsifier",
    "compatibility_requirements",
    "high_change_justification",
    "current_profile_expressibility_claim",
)
_OPEN_SPEC_DRAFT_ENRICHED_FIELDS = (
    "idea_mode",
    "research_question",
    "observed_failure_mode",
    "closest_parent",
    "minimal_testable_wedge",
    "causal_chain",
    "discriminative_predictions",
    "mechanism_off_definition",
    "resource_hypothesis",
    "realization_mode",
    "execution_contract",
)


def _deferred_open_spec_draft(outcome: ProducerOutcome) -> Mapping[str, Any]:
    if outcome.spec is None or outcome.source_proposal is not None:
        raise CampaignError("deferred Innovation backlog requires an open spec")
    spec = outcome.spec.to_dict()
    draft = {
        field_name: spec[field_name]
        for field_name in _OPEN_SPEC_DRAFT_BASE_FIELDS
    }
    draft.update(
        {
            field_name: spec[field_name]
            for field_name in _OPEN_SPEC_DRAFT_ENRICHED_FIELDS
            if field_name in spec
        }
    )
    draft["resolution_facts"] = outcome.resolution_facts
    return canonical_value(draft)


def _deferred_open_spec_key(outcome: ProducerOutcome) -> str:
    return sha256_digest(_deferred_open_spec_draft(outcome))


def _bounded_deferred_innovation_backlog(
    outcomes: Sequence[ProducerOutcome],
) -> tuple[ProducerOutcome, ...]:
    retained: list[ProducerOutcome] = []
    seen_spec_digests: set[str] = set()
    for outcome in outcomes:
        if not isinstance(outcome, ProducerOutcome):
            raise CampaignError(
                "deferred_innovation_backlog contains an invalid outcome"
            )
        if outcome.spec is None or outcome.source_proposal is not None:
            raise CampaignError(
                "deferred_innovation_backlog must contain open-spec outcomes"
            )
        if outcome.spec.digest in seen_spec_digests:
            continue
        seen_spec_digests.add(outcome.spec.digest)
        retained.append(outcome)
        if len(retained) == DEFERRED_INNOVATION_BACKLOG_LIMIT:
            break
    return tuple(retained)


def _next_deferred_innovation_backlog(
    before: "CampaignState",
    result: ResearchRoundResult,
) -> tuple[ProducerOutcome, ...]:
    prior = tuple(getattr(before, "deferred_innovation_backlog", ()))
    observed_open = tuple(
        outcome
        for outcome in result.producer_outcomes
        if outcome.spec is not None and outcome.source_proposal is None
    )
    still_deferred = tuple(
        outcome
        for outcome, _resolution in result.deferred_innovation_outcomes
        if outcome.spec is not None and outcome.source_proposal is None
    )
    observed_keys = {_deferred_open_spec_key(item) for item in observed_open}
    deferred_keys = {_deferred_open_spec_key(item) for item in still_deferred}
    retained = [
        item
        for item in prior
        if (
            _deferred_open_spec_key(item) not in observed_keys
            or _deferred_open_spec_key(item) in deferred_keys
        )
    ]
    retained_keys = {_deferred_open_spec_key(item) for item in retained}
    for outcome in still_deferred:
        key = _deferred_open_spec_key(outcome)
        if key not in retained_keys:
            retained.append(outcome)
            retained_keys.add(key)
    return _bounded_deferred_innovation_backlog(retained)


class _DeferredInnovationPending(RuntimeError):
    pass


class _DeferredInnovationProducer:
    """Replay persisted role output before issuing unrelated Provider work."""

    def __init__(
        self,
        producer: ResearchProducer,
        backlog: Sequence[ProducerOutcome],
    ) -> None:
        self._producer = producer
        self._backlog = tuple(backlog)
        self._by_role: dict[str, ProducerOutcome] = {}
        for outcome in self._backlog:
            self._by_role.setdefault(outcome.producer_role, outcome)

    @property
    def call_traces(self) -> tuple[Mapping[str, Any], ...]:
        traces = getattr(self._producer, "call_traces", ())
        return tuple(traces) if isinstance(traces, (tuple, list)) else ()

    def __call__(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
    ) -> Any:
        del context_view
        outcome = self._by_role.get(producer_role)
        if outcome is not None:
            return _deferred_open_spec_draft(outcome)
        raise _DeferredInnovationPending(
            "deferred Innovation backlog blocks unrelated Provider exploration"
        )


def _producer_for_round(
    producer: ResearchProducer,
    backlog: Sequence[ProducerOutcome],
    *,
    round_role: str,
) -> ResearchProducer:
    """Keep discovery rounds on the live Provider search path."""

    if round_role == "DISCOVERY" or not backlog:
        return producer
    return _DeferredInnovationProducer(producer, backlog)


def _immutable_canonical_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    canonical = canonical_value(dict(value))

    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            return MappingProxyType({key: freeze(child) for key, child in item.items()})
        if isinstance(item, tuple):
            return tuple(freeze(child) for child in item)
        return item

    return freeze(canonical)


def _physical_identity_fields(
    candidate_run: Mapping[str, Any],
    *,
    context_digest: str,
    context_applied: bool,
) -> dict[str, Any]:
    physical = candidate_run.get("physical_identity")
    physical = physical if isinstance(physical, Mapping) else {}
    experiment_binding = candidate_run.get("experiment_binding")
    experiment_binding = (
        experiment_binding if isinstance(experiment_binding, Mapping) else {}
    )
    evidence = candidate_run.get("gpu_reservation_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}

    def first(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    actual_context_digest = first(physical.get("context_digest"), context_digest)
    if actual_context_digest != context_digest:
        raise CampaignError("physical runner context identity drift")
    return canonical_value(
        {
            "physical_run_id": first(
                physical.get("run_id"),
                candidate_run.get("physical_run_id"),
                experiment_binding.get("run_id"),
                candidate_run.get("run_id"),
            ),
            "physical_seed": first(
                physical.get("seed"),
                candidate_run.get("physical_seed"),
                candidate_run.get("seed"),
                experiment_binding.get("seed"),
            ),
            "physical_context_digest": actual_context_digest,
            "research_context_digest": candidate_run.get(
                "research_context_digest"
            ),
            "cuda_visible_devices": first(
                physical.get("cuda_visible_devices"),
                candidate_run.get("cuda_visible_devices"),
            ),
            "reservation_digest": first(
                physical.get("reservation_digest"),
                candidate_run.get("reservation_digest"),
                evidence.get("reservation_digest"),
            ),
            "reservation_status": first(
                physical.get("reservation_status"),
                candidate_run.get("gpu_reservation_status"),
                candidate_run.get("reservation_status"),
            ),
            "final_worker_ceiling_seconds": first(
                physical.get("final_worker_ceiling_seconds"),
                candidate_run.get("final_worker_ceiling_seconds"),
            ),
            "physical_context_applied": context_applied,
        }
    )


@dataclass(frozen=True, slots=True)
class CampaignState:
    """The minimum arm-local state needed to build the next Research round."""

    campaign_id: str
    next_round_index: int
    context: ResearchContext
    active_profile: SearchExecutableProfileV1
    policy: VersionedResearchPolicyV1
    search_memory_head: SearchMemorySnapshotV1 | None
    carryover_proposals: tuple[CandidateProposalV4, ...]
    carryover_open_candidates: tuple[QualifiedSearchCandidateProtocolV1, ...]
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]]
    candidate_root_by_capability: Mapping[str, str]
    incumbent_observation: Mapping[str, Any]
    frontier: Mapping[str, Any]
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] | None = None
    deferred_innovation_backlog: tuple[ProducerOutcome, ...] = ()
    last_round_result_digest: str | None = None

    schema = "recclaw.research-line.campaign-state.v1"

    @classmethod
    def initial(
        cls,
        *,
        context: ResearchContext,
        active_profile: SearchExecutableProfileV1,
        policy: VersionedResearchPolicyV1,
        incumbent_observation: Mapping[str, Any],
        carryover_proposals: Sequence[CandidateProposalV4] = (),
        carryover_open_candidates: Sequence[QualifiedSearchCandidateProtocolV1] = (),
        search_memory_head: SearchMemorySnapshotV1 | None = None,
        qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] = (),
        candidate_root_by_capability: Mapping[str, str] = (),
        resource_profile_by_capability: Mapping[str, Mapping[str, Any]] = (),
        deferred_innovation_backlog: Sequence[ProducerOutcome] = (),
        frontier: Mapping[str, Any] | None = None,
    ) -> "CampaignState":
        return cls(
            campaign_id=context.campaign_id,
            next_round_index=context.round_index,
            context=context,
            active_profile=active_profile,
            policy=policy,
            search_memory_head=search_memory_head,
            carryover_proposals=tuple(carryover_proposals),
            carryover_open_candidates=tuple(carryover_open_candidates),
            qualified_execution_by_capability=qualified_execution_by_capability,
            candidate_root_by_capability=candidate_root_by_capability,
            resource_profile_by_capability=resource_profile_by_capability,
            deferred_innovation_backlog=tuple(deferred_innovation_backlog),
            incumbent_observation=incumbent_observation,
            frontier=context.frontier if frontier is None else frontier,
        )

    def __post_init__(self) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id.strip():
            raise CampaignError("campaign_id must be non-empty")
        if isinstance(self.next_round_index, bool) or self.next_round_index < 1:
            raise CampaignError("next_round_index must be positive")
        if not isinstance(self.context, ResearchContext):
            raise CampaignError("context must be ResearchContext")
        if not isinstance(self.active_profile, SearchExecutableProfileV1):
            raise CampaignError("active_profile must be SearchExecutableProfileV1")
        if not isinstance(self.policy, VersionedResearchPolicyV1):
            raise CampaignError("policy must be VersionedResearchPolicyV1")
        if self.context.round_index != self.next_round_index:
            raise CampaignError("context and next_round_index differ")
        if (
            self.context.campaign_id,
            self.context.active_profile_ref,
            self.context.active_profile_digest,
        ) != (
            self.campaign_id,
            self.active_profile.profile_ref,
            self.active_profile.profile_digest,
        ):
            raise CampaignError("campaign state profile/context identity drift")
        if canonical_value(self.context.policy) != canonical_value(self.policy.to_dict()):
            raise CampaignError("campaign state policy is not bound to Context")
        if (
            self.context.protocol_ref != self.active_profile.protocol_ref
            or self.context.protocol_digest != self.active_profile.protocol_digest
        ):
            raise CampaignError("campaign state protocol identity drift")
        if self.search_memory_head is not None and not isinstance(
            self.search_memory_head, SearchMemorySnapshotV1
        ):
            raise CampaignError("search_memory_head must be SearchMemorySnapshotV1")
        for item in self.carryover_proposals:
            if not isinstance(item, CandidateProposalV4):
                raise CampaignError("carryover_proposals contain an invalid proposal")
        for item in self.carryover_open_candidates:
            if not is_qualified_open_spec_candidate(item):
                raise CampaignError("carryover_open_candidates contain an invalid candidate")
        object.__setattr__(self, "carryover_proposals", tuple(self.carryover_proposals))
        object.__setattr__(
            self,
            "carryover_open_candidates",
            tuple(self.carryover_open_candidates),
        )
        object.__setattr__(
            self,
            "qualified_execution_by_capability",
            canonical_value(dict(self.qualified_execution_by_capability)),
        )
        object.__setattr__(
            self,
            "candidate_root_by_capability",
            canonical_value(dict(self.candidate_root_by_capability)),
        )
        object.__setattr__(
            self,
            "resource_profile_by_capability",
            canonical_value(dict(self.resource_profile_by_capability or {})),
        )
        object.__setattr__(
            self,
            "deferred_innovation_backlog",
            _bounded_deferred_innovation_backlog(
                getattr(self, "deferred_innovation_backlog", ())
            ),
        )
        object.__setattr__(
            self,
            "incumbent_observation",
            canonical_value(dict(self.incumbent_observation)),
        )
        object.__setattr__(self, "frontier", canonical_value(dict(self.frontier)))

    @property
    def active_executable_profile(self) -> SearchExecutableProfileV1:
        return self.active_profile

    @property
    def round_index(self) -> int:
        return self.next_round_index

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "campaign_id": self.campaign_id,
                "next_round_index": self.next_round_index,
                "context": self.context.to_dict(),
                "active_profile": self.active_profile.canonical_dict(),
                "policy": self.policy.to_dict(),
                "search_memory_head": (
                    self.search_memory_head.to_dict()
                    if self.search_memory_head is not None
                    else None
                ),
                "carryover_proposals": tuple(
                    item.to_dict() for item in self.carryover_proposals
                ),
                "carryover_open_candidates": tuple(
                    item.to_dict() for item in self.carryover_open_candidates
                ),
                "qualified_execution_by_capability": self.qualified_execution_by_capability,
                "candidate_root_by_capability": self.candidate_root_by_capability,
                "resource_profile_by_capability": getattr(
                    self, "resource_profile_by_capability", {}
                ),
                "deferred_innovation_backlog": tuple(
                    {
                        "spec_digest": item.spec.digest,
                        "outcome": item.to_dict(),
                    }
                    for item in getattr(
                        self, "deferred_innovation_backlog", ()
                    )
                    if item.spec is not None
                ),
                "incumbent_observation": self.incumbent_observation,
                "frontier": self.frontier,
                "last_round_result_digest": self.last_round_result_digest,
            }
        )


ResearchCampaignState = CampaignState


@dataclass(frozen=True, slots=True)
class CampaignRoundInputs:
    """Round-local inputs supplied by the production composition."""

    producer_bindings: Mapping[str, Any]
    resolver_environment: Mapping[str, Any]
    budget_snapshot: Mapping[str, Any]
    router: StrongStaticRouterV1
    metric_contract_digest: str
    observation_seed: str
    next_discriminative_test: str
    confirmation_seed: str | None = None
    qualified_execution_by_capability: Mapping[str, Mapping[str, Any]] = ()
    innovation_inputs: InnovationRuntimeInputs | None = None
    meta_research_inputs: MetaResearchInputs | None = None
    attempt_scheduler: bool = False
    max_attempts_per_round: int | None = None
    prebinding_token_ceiling_retry: bool = True
    close_exhausted_no_metric_slot: bool = False
    bootstrap_fixed_candidates: bool = False
    portfolio_candidates: tuple[PortfolioCandidateV2, ...] = ()
    research_profile_source: ResearchProfileSourceV1 | None = None
    candidate_handoff_factory: CandidateHandoffFactory | None = None
    candidate_root_by_capability: Mapping[str, str] = ()
    resource_profile_by_capability: Mapping[str, Mapping[str, Any]] = ()
    evidence_port: EvidenceGuardPort | None = None
    observation_seed_schedule: tuple[str, ...] | None = None
    verification_seed_schedule: tuple[str, ...] | None = None
    evaluator: Mapping[str, Any] = field(
        default_factory=lambda: canonical_value(COMMON_EVALUATOR)
    )
    split: str = COMMON_SPLIT
    frozen_profile_ref: Mapping[str, Any] | None = None
    round_role: str = "LEGACY_MIXED"
    search_space_adapter: SearchSpaceAdapter | None = None
    feedback_proposal_generation_exhausted: bool = False
    engineering_source_identity: Mapping[str, Any] | None = None


def _configured_attempt_budget(inputs: CampaignRoundInputs) -> int:
    """Return the frozen configuration cap for this round's attempts."""

    value: Any = inputs.max_attempts_per_round
    if value is None:
        for field_name in (
            "max_attempts_per_round",
            "round_attempt_budget",
            "remaining_attempt_budget",
            "attempt_budget",
        ):
            candidate = inputs.budget_snapshot.get(field_name)
            if candidate is not None:
                value = candidate
                break
    if value is None:
        raise CampaignError(
            "attempt_scheduler=True requires an explicit frozen "
            "max_attempts_per_round or round attempt budget"
        )
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CampaignError("round attempt budget must be a non-negative integer")
    return value


class CampaignRoundInputFactory(Protocol):
    def __call__(self, state: CampaignState) -> CampaignRoundInputs: ...


@dataclass(frozen=True, slots=True)
class CampaignRoundRecord:
    """Durable in-process result and before/after state for one round."""

    round_index: int
    opportunity_ref: str
    status: str
    state_before: CampaignState
    result: ResearchRoundResult
    state_after: CampaignState

    schema = "recclaw.research-line.campaign-round.v1"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    @property
    def outcome_kind(self) -> str:
        return self.status

    def to_dict(self) -> dict[str, Any]:
        interpretation = self.result.interpretation
        return canonical_value(
            {
                "schema": self.schema,
                "round_index": self.round_index,
                "opportunity_ref": self.opportunity_ref,
                "status": self.status,
                "failure_taxonomy": (
                    interpretation.failure_taxonomy if interpretation is not None else None
                ),
                "experiment_executed": self.result.candidate_run is not None,
                "metric_bearing_experiment": self.result.has_metric_bearing_attempt,
                "attempt_count": self.result.attempt_count,
                "metric_bearing_attempt_index": self.result.metric_bearing_attempt_index,
                "state_before": self.state_before.to_dict(),
                "result": self.result.to_dict(),
                "state_after": self.state_after.to_dict(),
            }
        )


def compact_campaign_round_trace(
    record: CampaignRoundRecord,
    *,
    checkpoint_sha256: str,
    checkpoint_ref: str,
) -> dict[str, Any]:
    """Build the small audit index for an authoritative round checkpoint."""

    attempt_summaries = tuple(
        canonical_value(
            {
                "attempt_index": attempt.attempt_index,
                "attempt_digest": attempt.digest,
                "candidate_id": attempt.candidate_id,
                "engineering_disposition": attempt.engineering_disposition,
                "metric_bearing": attempt.metric_bearing,
                "failure_scope": attempt.failure_scope,
                "failure_reason_code": (
                    attempt.failure_detail.get("reason_code")
                    if isinstance(attempt.failure_detail, Mapping)
                    else None
                ),
                "observation_ref": attempt.observation_ref,
                "observation_digest": attempt.observation_digest,
                "candidate_run_digest": sha256_digest(attempt.candidate_run),
                "metric_summary": {
                    key: value
                    for key, value in attempt.candidate_run.items()
                    if key
                    in {
                        "best_epoch",
                        "best_valid_result",
                        "epochs_completed",
                        "metric",
                        "metric_name",
                        "metric_value",
                        "metrics",
                        "seed",
                        "split",
                        "status",
                    }
                },
            }
        )
        for attempt in record.result.attempts
    )
    interpretation = record.result.interpretation
    feedback_projection = getattr(interpretation, "feedback_projection", None)
    diagnostic_detail = (
        feedback_projection.get("diagnostic_detail")
        if isinstance(feedback_projection, Mapping)
        else None
    )
    return canonical_value(
        {
            "schema": "recclaw.research-line.campaign-round-trace.v2",
            "record_digest": record.digest,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_ref": checkpoint_ref,
            "summary": {
                "round_index": record.round_index,
                "opportunity_ref": record.opportunity_ref,
                "status": record.status,
                "state_before_digest": record.state_before.digest,
                "state_after_digest": record.state_after.digest,
                "result_digest": sha256_digest(record.result.to_dict()),
                "incomplete_reason": record.result.incomplete_reason,
                "failure_taxonomy": (
                    interpretation.failure_taxonomy
                    if interpretation is not None
                    else None
                ),
                "failure_code": (
                    diagnostic_detail.get("failure_code")
                    if isinstance(diagnostic_detail, Mapping)
                    else None
                ),
                "selected_outcome_digest": (
                    record.result.selected_outcome.digest
                    if record.result.selected_outcome is not None
                    else None
                ),
                "prepared_digest": (
                    record.result.prepared.digest
                    if record.result.prepared is not None
                    else None
                ),
                "provider_trace_digests": tuple(
                    sha256_digest(item) for item in record.result.provider_traces
                ),
                "metric_bearing_attempt_index": (
                    record.result.metric_bearing_attempt_index
                ),
                "attempt_count": record.result.attempt_count,
                "attempts": attempt_summaries,
            },
        }
    )


@dataclass(frozen=True, slots=True)
class CampaignVerificationRecord:
    """Durable auxiliary evidence that does not consume a discovery round."""

    verification_ref: str
    task_id: str
    operation: str
    observation_seed: str
    status: str
    state_before: CampaignState
    result: ResearchRoundResult
    state_after: CampaignState

    schema = "recclaw.research-line.campaign-verification.v1"

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        interpretation = self.result.interpretation
        return canonical_value(
            {
                "schema": self.schema,
                "verification_ref": self.verification_ref,
                "task_id": self.task_id,
                "operation": self.operation,
                "observation_seed": self.observation_seed,
                "status": self.status,
                "failure_taxonomy": (
                    interpretation.failure_taxonomy
                    if interpretation is not None
                    else None
                ),
                "metric_bearing_experiment": self.result.has_metric_bearing_attempt,
                "attempt_count": len(self.result.attempts),
                "metric_bearing_attempt_index": (
                    self.result.metric_bearing_attempt_index
                ),
                "state_before": self.state_before.to_dict(),
                "result": self.result.to_dict(),
                "state_after": self.state_after.to_dict(),
            }
        )


def _verification_identity(
    task: ResearchTaskRecordV2,
    observation_seed: str,
) -> str:
    return sha256_digest(
        {
            "schema": "recclaw.research-line.verification-identity.v1",
            "task_id": task.task_id,
            "task_digest": task.digest,
            "operation": task.operation.value,
            "observation_seed": observation_seed,
        }
    )


def _write_once(path: Path, payload: bytes) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        return False
    return True


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"cannot read campaign JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise CampaignError(f"campaign JSON root is not an object: {path}")
    return value


def _read_pickle(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError, ImportError, AttributeError) as error:
        raise CampaignError(f"cannot read campaign checkpoint {path}: {error}") from error


def _unique_by_id(
    items: Sequence[CandidateProposalV4 | QualifiedSearchCandidateProtocolV1],
) -> tuple[CandidateProposalV4 | QualifiedSearchCandidateProtocolV1, ...]:
    by_id: dict[str, CandidateProposalV4 | QualifiedSearchCandidateProtocolV1] = {}
    for item in items:
        by_id[item.candidate_id] = item
    return tuple(by_id.values())


def _resource_efficiency_repair_context(
    candidate_run: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Keep only measured timeout facts needed for one implementation repair.

    A healthy worker that reaches the fixed ceiling has not falsified its
    mechanism.  Its durable observation may guide the next logical generation
    to optimize the implementation, but it must not become a scientific metric
    or authorize changing the mechanism to make the timeout disappear.
    """

    if str(candidate_run.get("exit_status", "")).upper() != "RESOURCE_CENSORED":
        return None
    telemetry = candidate_run.get("resource_telemetry")
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    phase_records = telemetry.get("phase_records")
    phase_records = (
        tuple(item for item in phase_records if isinstance(item, Mapping))
        if isinstance(phase_records, (tuple, list))
        else ()
    )

    def mean_phase_ms(phase: str) -> float | None:
        values = [
            float(item["wall_time_ms"])
            for item in phase_records
            if str(item.get("phase", "")).upper() == phase
            and isinstance(item.get("wall_time_ms"), (int, float))
            and not isinstance(item.get("wall_time_ms"), bool)
        ]
        return sum(values) / len(values) if values else None

    prediction = candidate_run.get("resource_prediction")
    prediction = prediction if isinstance(prediction, Mapping) else {}
    nested_prediction = prediction.get("prediction")
    nested_prediction = (
        nested_prediction if isinstance(nested_prediction, Mapping) else {}
    )
    ceiling = next(
        (
            value
            for value in (
                candidate_run.get("final_worker_ceiling_seconds"),
                candidate_run.get("assigned_deadline_seconds"),
                candidate_run.get("resource_deadline_seconds"),
                nested_prediction.get("worker_ceiling_seconds"),
            )
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ),
        None,
    )
    wall_time_ms = candidate_run.get("wall_time_ms")
    return canonical_value(
        {
            "schema": (
                "recclaw.research-line.implementation-efficiency-repair-context.v1"
            ),
            "reason_code": "MEASURED_WORKER_RESOURCE_CEILING",
            "next_attempt_scope": "IMPLEMENTATION_EFFICIENCY_ONLY",
            "preserve_mechanism_program": True,
            "mechanism_effect_update_allowed": False,
            "epochs_completed": telemetry.get("epochs_completed"),
            "best_observed_epoch": telemetry.get("best_observed_epoch"),
            "active_progress": telemetry.get("active_progress"),
            "elapsed_seconds": (
                float(wall_time_ms) / 1000
                if isinstance(wall_time_ms, (int, float))
                and not isinstance(wall_time_ms, bool)
                else None
            ),
            "worker_ceiling_seconds": ceiling,
            "censoring_trigger": candidate_run.get("censoring_trigger"),
            "mean_train_epoch_wall_time_ms": mean_phase_ms("TRAIN"),
            "mean_eval_epoch_wall_time_ms": mean_phase_ms("EVAL"),
            "full_train_batches_per_epoch": telemetry.get(
                "full_train_batches_per_epoch"
            ),
            "full_validation_batches_per_eval": telemetry.get(
                "full_validation_batches_per_eval"
            ),
        }
    )


def _measured_execution_cost(candidate_run: Mapping[str, Any]) -> Mapping[str, Any]:
    """Retain observed cost for successful and censored runs alike."""

    telemetry = candidate_run.get("resource_telemetry")
    telemetry = telemetry if isinstance(telemetry, Mapping) else {}
    cost = {
        key: telemetry[key]
        for key in (
            "epochs_completed",
            "best_observed_epoch",
            "initialization_wall_time_ms",
            "full_train_batches_per_epoch",
            "full_validation_batches_per_eval",
        )
        if telemetry.get(key) is not None
    }
    wall_time_ms = candidate_run.get("wall_time_ms")
    if isinstance(wall_time_ms, (int, float)) and not isinstance(wall_time_ms, bool):
        cost["elapsed_seconds"] = float(wall_time_ms) / 1000
    for key in ("final_worker_ceiling_seconds", "censoring_trigger", "exit_status", "epochs_requested", "physical_gpu_id"):
        if candidate_run.get(key) is not None:
            cost[key] = candidate_run[key]
    phase_records = telemetry.get("phase_records", ())
    if isinstance(phase_records, (tuple, list)):
        for phase in ("TRAIN", "EVAL"):
            values = [float(row["wall_time_ms"]) for row in phase_records
                if isinstance(row, Mapping) and row.get("phase") == phase
                and isinstance(row.get("wall_time_ms"), (int, float)) and not isinstance(row["wall_time_ms"], bool)]
            if values:
                cost[f"mean_{phase.lower()}_phase_wall_time_ms"] = sum(values) / len(values)
    return canonical_value(cost)


def _completed_execution_facts(
    candidate_run: Mapping[str, Any], *, metric_bearing: bool,
) -> Mapping[str, Any] | None:
    """Keep native execution facts, without code or inferred mechanism claims."""
    binding = candidate_run.get("experiment_binding")
    if not isinstance(binding, Mapping):
        return None
    facts = {
        "status": candidate_run.get("exit_status"),
        "metric_bearing": metric_bearing,
        "seed": candidate_run.get("seed"),
        "configuration_source": "worker.experiment_binding",
        "configuration": {
            key: binding[key] for key in ("model", "base_model_config", "entrypoint", "config", "split")
            if key in binding
        },
        **{
            key: binding[key] for key in (
                "candidate_root_path", "candidate_source_tree_digest", "entrypoint_source_sha256",
            ) if key in binding
        },
    }
    if metric_bearing and isinstance(candidate_run.get("metrics"), Mapping):
        facts["development_metrics"] = candidate_run["metrics"]
    telemetry = candidate_run.get("resource_telemetry")
    records = telemetry.get("phase_records", ()) if isinstance(telemetry, Mapping) else ()
    records = [
        row for row in records if isinstance(row, Mapping)
        and isinstance(row.get("epoch"), int) and row.get("phase") in {"TRAIN", "EVAL"}
    ] if isinstance(records, (tuple, list)) else []
    if records:
        epochs = sorted({row["epoch"] for row in records})
        checkpoints = {"first": epochs[0], "last": epochs[-1]}
        evaluations = [
            row for row in records if row["phase"] == "EVAL"
            and isinstance(row.get("valid_score"), (int, float))
            and math.isfinite(row["valid_score"])
        ]
        if evaluations:
            checkpoints["best_dev"] = max(evaluations, key=lambda row: row["valid_score"])["epoch"]
        facts["native_training"] = {
            "epoch_selection": checkpoints,
            "phase_records": [
                {key: row[key] for key in ("epoch", "phase", "status", "batch_count", "loss", "valid_score")
                 if row.get(key) is not None}
                for row in records if row["epoch"] in checkpoints.values()
            ],
            "semantics": "Native phase values and loss reduction unchanged; observed progress is not a terminal metric for censored runs.",
        }
    return canonical_value(facts)


def _compact_round_attempt_summary(
    attempt: RoundAttemptV1,
    *,
    round_index: int,
    effect_observation: Mapping[str, Any] | None = None,
    context: ResearchContext | None = None,
) -> Mapping[str, Any]:
    """Keep compact attempt identity in successor scientific memory.

    The lossless attempt, including diagnostic successor Context and Memory
    snapshots, remains in the current ``CampaignRoundRecord`` checkpoint and
    trace.  Copying those snapshots into the successor Context would make the
    next round embed the complete prior ``round_attempts`` history again.
    """

    candidate_run = attempt.candidate_run
    acquisition = attempt.acquisition
    effective_identity = attempt.search_space_attestation
    if not isinstance(effective_identity, Mapping):
        raise CampaignError("round attempt lacks Runtime search-space attestation")
    development_metrics = candidate_run.get("metrics")
    development_metrics = (
        canonical_value(dict(development_metrics))
        if attempt.metric_bearing and isinstance(development_metrics, Mapping)
        else None
    )
    effect_observation = (
        effect_observation
        if isinstance(effect_observation, Mapping)
        else {}
    )

    def digest_or_none(value: Any) -> str | None:
        return sha256_digest(value) if value is not None else None

    proposal = attempt.binding.proposal
    producer_role = (
        proposal.spec.producer_role
        if is_qualified_open_spec_candidate(proposal)
        else proposal.producer_role
    )
    program = proposal.mechanism_program
    payload = program.get("program_payload", {}) if isinstance(program, Mapping) else {}
    parent_refs = payload.get("parent_refs", ())
    parent_comparison = None
    if len(parent_refs) == 1 and isinstance(parent_refs[0], Mapping):
        parent_candidate_id = parent_refs[0].get("candidate_id")
        parent_comparison = {**dict(parent_refs[0]), "value": None, "delta": None}
        if context is not None:
            history = context.scientific_memory.get("mechanism_experiences", ())
            for row in reversed(history):
                if (
                    isinstance(row, Mapping)
                    and isinstance(parent_candidate_id, str)
                    and bool(parent_candidate_id)
                    and (row.get("compiler_candidate_id") or row.get("candidate_id"))
                    == parent_candidate_id
                    and str(row.get("observation_seed")) == str(candidate_run.get("seed"))
                    and row.get("protocol_digest") == context.protocol_digest
                ):
                    parent_metrics = row.get("development_metrics") or {}
                    value = parent_metrics.get("ndcg@10")
                    if isinstance(value, (int, float)):
                        parent_comparison["value"] = value
                        parent_comparison["seed"] = row["observation_seed"]
                        break
            baseline = context.knowledge_base.get("baseline_context", {})
            anchor = baseline.get("parent_anchor", {}) if isinstance(baseline, Mapping) else {}
            paired = anchor.get("paired_metric", {})
            if (
                anchor.get("binding") == parent_refs[0]
                and isinstance(paired, Mapping)
                and str(paired.get("seed")) == str(candidate_run.get("seed"))
            ):
                parent_comparison["value"] = paired.get("value")
                parent_comparison["seed"] = paired.get("seed")
            value = (development_metrics or {}).get("ndcg@10")
            if value is not None and parent_comparison["value"] is not None:
                parent_comparison["delta"] = value - parent_comparison["value"]

    return canonical_value(
        {
            "schema": "recclaw.research-line.round-attempt-summary.v1",
            "round_index": round_index,
            "attempt_index": attempt.attempt_index,
            "candidate_id": attempt.candidate_id,
            "compiler_candidate_id": getattr(proposal, "compiler_candidate_id", None),
            "protocol_digest": context.protocol_digest if context else None,
            "construction_parent": parent_comparison,
            "producer_role": producer_role,
            "mechanism_axis": attempt.binding.proposal.mechanism_axis,
            "candidate_semantic_digest": attempt.binding.mechanism_semantics_digest,
            "effective_experiment_digest": effective_identity[
                "effective_experiment_digest"
            ],
            "effective_family_digest": effective_identity[
                "effective_family_digest"
            ],
            "primitive_ids": effective_identity.get("primitive_ids", ()),
            "executable_identity_digest": _metric_executable_identity_digest(
                attempt.execution_recipe
            ),
            "candidate_package_digest": attempt.execution_recipe.get(
                "candidate_package_digest"
            ),
            "candidate_source_tree_digest": attempt.execution_recipe.get(
                "candidate_source_tree_digest"
            ),
            "candidate_source_content_digest": attempt.execution_recipe.get(
                "candidate_source_content_digest"
            ),
            "candidate_root_path": attempt.execution_recipe.get("candidate_root_path"),
            "execution_contract": getattr(proposal, "execution_contract", None),
            "completed_execution": _completed_execution_facts(
                candidate_run, metric_bearing=attempt.metric_bearing,
            ),
            "entrypoint_source_sha256": attempt.execution_recipe.get(
                "entrypoint_source_sha256"
            ),
            "observation_seed": candidate_run.get("seed"),
            "outcome": candidate_run.get(
                "exit_status", candidate_run.get("status")
            ),
            "outcome_digest": sha256_digest(candidate_run),
            "metric_digest": digest_or_none(candidate_run.get("metrics")),
            "development_metrics": development_metrics,
            "comparator_delta": effect_observation.get("comparator_delta"),
            "core_mechanism_contrast": effect_observation.get(
                "core_mechanism_contrast"
            ) or (
                proposal.spec.mechanism_change
                if is_qualified_open_spec_candidate(proposal)
                else None
            ),
            "next_discriminative_task": (
                proposal.spec.falsifier if is_qualified_open_spec_candidate(proposal) else None
            ),
            "frontier_updated": effect_observation.get("frontier_updated"),
            "frontier_delta": effect_observation.get("frontier_delta"),
            "measured_execution_cost": _measured_execution_cost(candidate_run),
            "mechanism_axis_footprint": effect_observation.get(
                "mechanism_axis_footprint", ()
            ),
            "evidence_class": effect_observation.get("evidence_class"),
            "unresolved_confounding": effect_observation.get(
                "unresolved_confounding", ()
            ),
            "failure_scope": attempt.failure_scope,
            "engineering_disposition": attempt.engineering_disposition,
            "metric_bearing": attempt.metric_bearing,
            "physical_observation_ref": attempt.observation_ref,
            "physical_observation_digest": attempt.observation_digest,
            "execution_recipe_digest": sha256_digest(attempt.execution_recipe),
            "binding_digest": attempt.binding.digest,
            "route_trace_digest": (
                acquisition.route_trace.digest if acquisition is not None else None
            ),
            "failure_detail_digest": digest_or_none(attempt.failure_detail),
            "failure": (
                canonical_value(
                    {
                        key: attempt.failure_detail.get(key)
                        for key in (
                            "failure_class",
                            "stage",
                            "reason_code",
                            "message",
                        )
                        if attempt.failure_detail.get(key) is not None
                    }
                )
                if isinstance(attempt.failure_detail, Mapping)
                else None
            ),
            "implementation_fidelity_failure": (
                attempt.failure_detail
                if attempt.engineering_disposition
                == "IMPLEMENTATION_FIDELITY_REJECTED"
                else None
            ),
            "implementation_efficiency_repair_context": (
                _resource_efficiency_repair_context(candidate_run)
            ),
            "resource_prediction_digest": digest_or_none(attempt.resource_prediction),
            "live_health_decisions_digest": digest_or_none(
                attempt.live_health_decisions
            ),
            "diagnostic_feedback_digest": digest_or_none(attempt.diagnostic_feedback),
        }
    )


_PORTFOLIO_HISTORY_LIMIT = 64
_ROUND_ATTEMPT_HISTORY_LIMIT = 64
_ROUND_ATTEMPT_IDENTITY_HISTORY_LIMIT = 800
_METRIC_OBSERVATION_INDEX_LIMIT = 200
_PRODUCER_OPPORTUNITY_RECENT_LIMIT = 16
_PORTFOLIO_HISTORY_CONTAINER_KEYS = ("attempts", "records", "history", "observations")
_ROUND_ATTEMPT_IDENTITY_FIELDS = {
    "attempted_executable_identity_digests": "executable_identity_digest",
    "attempted_semantic_identity_digests": "candidate_semantic_digest",
    "attempted_effective_experiment_digests": "effective_experiment_digest",
    "attempted_effective_family_digests": "effective_family_digest",
}


def _compact_mechanism_experiences(
    scientific_memory: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Keep decision facts beyond the short attempt window, without snapshots."""

    global_memory = scientific_memory.get("global_memory", {})
    prior = scientific_memory.get("mechanism_experiences")
    if not isinstance(prior, (tuple, list)):
        prior = global_memory.get("mechanism_experiences", ()) if isinstance(global_memory, Mapping) else ()
    bootstrap = scientific_memory.get("round_attempts", ())
    fields = (
        "round_index", "attempt_index", "candidate_id", "producer_role",
        "compiler_candidate_id", "protocol_digest", "construction_parent",
        "effective_experiment_digest", "effective_family_digest", "primitive_ids",
        "physical_observation_digest", "mechanism_axis", "mechanism_axis_footprint",
        "core_mechanism_contrast",
        "development_metrics", "comparator_delta", "frontier_updated", "frontier_delta",
        "measured_execution_cost", "observation_seed", "outcome", "evidence_class",
        "unresolved_confounding", "failure", "engineering_disposition", "next_discriminative_task",
    )
    records: dict[str, Mapping[str, Any]] = {}
    for values in (prior, bootstrap, attempts):
        if not isinstance(values, (tuple, list)):
            continue
        for row in values:
            if not isinstance(row, Mapping):
                continue
            record = {key: row[key] for key in fields if row.get(key) is not None}
            if not record.get("candidate_id") and not record.get("effective_experiment_digest"):
                continue
            identity = record.get("physical_observation_digest") or sha256_digest(
                {key: record.get(key) for key in (
                    "round_index", "attempt_index", "candidate_id",
                    "effective_experiment_digest", "failure",
                )}
            )
            records[str(identity)] = canonical_value(record)
    return tuple(records.values())[-_ROUND_ATTEMPT_IDENTITY_HISTORY_LIMIT:]


def _compact_round_attempt_identity_history(
    scientific_memory: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> Mapping[str, tuple[str, ...]]:
    """Retain bounded identity-only history beyond the 64 payload summaries."""

    global_memory = scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    histories: dict[str, tuple[str, ...]] = {}
    for history_key, attempt_key in _ROUND_ATTEMPT_IDENTITY_FIELDS.items():
        values: list[str] = []
        seen: set[str] = set()
        for raw_values in (
            scientific_memory.get(history_key, ()),
            global_memory.get(history_key, ()),
        ):
            if not isinstance(raw_values, (tuple, list)):
                continue
            for value in raw_values:
                if isinstance(value, str) and value and value not in seen:
                    seen.add(value)
                    values.append(value)
        for attempt in attempts:
            if (
                attempt_key != "executable_identity_digest"
                and attempt.get("metric_bearing") is not True
            ):
                continue
            if (
                attempt_key == "executable_identity_digest"
                and not _attempt_retires_executable_identity(attempt)
            ):
                continue
            value = attempt.get(attempt_key)
            if isinstance(value, str) and value and value not in seen:
                seen.add(value)
                values.append(value)
        histories[history_key] = tuple(
            values[-_ROUND_ATTEMPT_IDENTITY_HISTORY_LIMIT:]
        )
    return canonical_value(histories)


def _compact_producer_opportunity_state(
    scientific_memory: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Advance compact, monotonic discovery-opportunity scheduling state."""

    global_memory = scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    raw_state = scientific_memory.get(
        "producer_opportunity_state",
        global_memory.get("producer_opportunity_state", {}),
    )
    state = dict(raw_state) if isinstance(raw_state, Mapping) else {}
    count = int(state.get("opportunity_count", 0))
    role_counts = {
        role: int(dict(state.get("role_counts", {})).get(role, 0))
        for role in DISCOVERY_PRODUCERS
    }
    recent_roles = list(state.get("recent_roles", ()))
    recent_ids = list(state.get("recent_selection_ids", ()))
    seen = set(str(item) for item in recent_ids)

    for attempt in attempts:
        if attempt.get("execution_lane") == "AUXILIARY_VERIFICATION":
            continue
        role = attempt.get("producer_role")
        if role not in DISCOVERY_PRODUCERS:
            continue
        acquisition = attempt.get("idea_acquisition")
        selected_spec = (
            acquisition.get("selected_spec_digest")
            if isinstance(acquisition, Mapping)
            else None
        )
        identity = next(
            (
                value
                for value in (
                    attempt.get("effective_experiment_digest"),
                    selected_spec,
                    attempt.get("spec_digest"),
                    attempt.get("candidate_id"),
                )
                if isinstance(value, str) and value
            ),
            None,
        )
        if identity is None:
            continue
        selection_id = f"{role}:{identity}"
        if selection_id in seen:
            continue
        seen.add(selection_id)
        recent_ids.append(selection_id)
        recent_roles.append(str(role))
        role_counts[str(role)] += 1
        count += 1

    return canonical_value(
        {
            "opportunity_count": count,
            "role_counts": role_counts,
            "recent_roles": tuple(recent_roles[-8:]),
            "recent_selection_ids": tuple(
                recent_ids[-_PRODUCER_OPPORTUNITY_RECENT_LIMIT:]
            ),
        }
    )


def _compact_metric_observation_index(
    scientific_memory: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Retain only the physical metric facts needed for zero-runner rebind."""

    global_memory = scientific_memory.get("global_memory")
    global_memory = global_memory if isinstance(global_memory, Mapping) else {}
    rows: list[Mapping[str, Any]] = []
    by_observation_digest: dict[str, int] = {}

    def append(row: Mapping[str, Any]) -> None:
        metrics = row.get("development_metrics")
        observation_digest = row.get("physical_observation_digest")
        if (
            row.get("metric_bearing") is not True
            or not isinstance(metrics, Mapping)
            or not isinstance(observation_digest, str)
            or not observation_digest
            or not isinstance(row.get("physical_observation_ref"), str)
        ):
            return
        compact = canonical_value(
            {
                "schema": "recclaw.research-line.metric-observation-index.v1",
                "round_index": row.get("round_index"),
                "candidate_id": row.get("candidate_id"),
                "candidate_semantic_digest": row.get(
                    "candidate_semantic_digest"
                ),
                "effective_experiment_digest": row.get(
                    "effective_experiment_digest"
                ),
                "effective_family_digest": row.get(
                    "effective_family_digest"
                ),
                "metric_bearing": True,
                "observation_seed": row.get("observation_seed"),
                "outcome": row.get("outcome"),
                "development_metrics": canonical_value(dict(metrics)),
                "physical_observation_ref": row.get(
                    "physical_observation_ref"
                ),
                "physical_observation_digest": observation_digest,
            }
        )
        previous = by_observation_digest.get(observation_digest)
        if previous is not None:
            rows[previous] = compact
            return
        by_observation_digest[observation_digest] = len(rows)
        rows.append(compact)

    for raw_rows in (
        scientific_memory.get("metric_observation_index", ()),
        global_memory.get("metric_observation_index", ()),
        attempts,
    ):
        if isinstance(raw_rows, (tuple, list)):
            for row in raw_rows:
                if isinstance(row, Mapping):
                    append(row)
    return tuple(rows[-_METRIC_OBSERVATION_INDEX_LIMIT:])


def _history_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        for key in _PORTFOLIO_HISTORY_CONTAINER_KEYS:
            nested = value.get(key)
            if isinstance(nested, (tuple, list)):
                return tuple(item for item in nested if isinstance(item, Mapping))
        return (value,) if "candidate_id" in value else ()
    if isinstance(value, (tuple, list)):
        return tuple(item for item in value if isinstance(item, Mapping))
    return ()


def _explicit_resource_compute_pattern(
    resource_prediction: Mapping[str, Any] | None,
) -> str | None:
    if not isinstance(resource_prediction, Mapping):
        return None
    prediction = resource_prediction.get("prediction")
    nested = prediction.get("compute_pattern") if isinstance(prediction, Mapping) else None
    top_level = resource_prediction.get("compute_pattern")
    values = [value for value in (nested, top_level) if value is not None]
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        return None
    if len(set(values)) != 1:
        return None
    return values[0].strip()


def _prepared_portfolio_candidate(
    result: ResearchRoundResult,
    attempt: RoundAttemptV1,
) -> tuple[PortfolioCandidateV2 | None, Any | None]:
    prepared = result.prepared
    if prepared is None:
        return None, None
    for handoff in tuple(getattr(prepared, "candidate_handoffs", ()) or ()):
        if getattr(handoff, "candidate_id", None) != attempt.candidate_id:
            continue
        candidate = getattr(handoff, "portfolio_candidate", None)
        if (
            isinstance(candidate, PortfolioCandidateV2)
            and candidate.candidate_id == attempt.candidate_id
            and candidate.semantic_digest == attempt.binding.mechanism_semantics_digest
        ):
            return candidate, handoff
        return None, None
    for candidate in tuple(getattr(prepared, "portfolio_candidates", ()) or ()):
        if (
            isinstance(candidate, PortfolioCandidateV2)
            and candidate.candidate_id == attempt.candidate_id
            and candidate.semantic_digest == attempt.binding.mechanism_semantics_digest
        ):
            return candidate, None
    return None, None


def _typed_resource_admission(attempt: RoundAttemptV1) -> bool | None:
    if attempt.metric_bearing:
        return True
    typed_resource_values = {
        "RESOURCE",
        "RESOURCE_CENSORED",
        "HEALTH_RESOURCE",
        "HEALTH_RESOURCE_CENSORED",
        "RESOURCE_HEALTH",
    }
    for evidence in (
        attempt.failure_detail,
        attempt.diagnostic_feedback,
        *attempt.live_health_decisions,
    ):
        if not isinstance(evidence, Mapping):
            continue
        for field_name in (
            "failure_class",
            "typed_blocker_class",
            "resource_failure_class",
            "resource_disposition",
            "health_resource_class",
            "censoring_trigger",
        ):
            value = evidence.get(field_name)
            normalized = str(value).strip().upper() if value is not None else ""
            if normalized in typed_resource_values:
                return False
    return None


def _portfolio_attempt_row(
    result: ResearchRoundResult,
    attempt: RoundAttemptV1,
    *,
    round_index: int,
) -> tuple[Mapping[str, Any], PortfolioCandidateV2 | None] | None:
    candidate, handoff = _prepared_portfolio_candidate(result, attempt)
    proposal = attempt.binding.proposal
    compute_pattern = (
        candidate.compute_pattern
        if candidate is not None
        else _explicit_resource_compute_pattern(attempt.resource_prediction)
    )
    family_id = (
        candidate.family_id
        if candidate is not None
        else getattr(proposal, "mechanism_axis", None)
    )
    parent_id = (
        candidate.parent_id
        if candidate is not None
        else getattr(proposal, "parent_candidate_id", None)
    )
    if (
        not isinstance(compute_pattern, str)
        or not compute_pattern.strip()
        or not isinstance(family_id, str)
        or not family_id.strip()
    ):
        return None
    semantic_digest = (
        candidate.semantic_digest
        if candidate is not None
        else attempt.binding.mechanism_semantics_digest
    )
    source_digest = sha256_digest(
        {
            "candidate_handoff_digest": (
                handoff.digest
                if handoff is not None
                else None
            ),
            "portfolio_candidate_digest": (
                sha256_digest(candidate.to_dict()) if candidate is not None else None
            ),
            "binding_digest": attempt.binding.digest,
            "resource_prediction_digest": (
                sha256_digest(attempt.resource_prediction)
                if attempt.resource_prediction is not None
                else None
            ),
        }
    )
    row: dict[str, Any] = {
        "round_index": round_index,
        "candidate_id": attempt.candidate_id,
        "candidate_semantic_digest": semantic_digest,
        "family_id": family_id.strip(),
        "compute_pattern": compute_pattern.strip(),
        "sealed": True,
        "sealed_valid_seal": bool(attempt.metric_bearing),
        "attempt_digest": attempt.digest,
        "source_digest": source_digest,
    }
    if parent_id is not None:
        if not isinstance(parent_id, str) or not parent_id.strip():
            return None
        row["parent_id"] = parent_id.strip()
    resource_admitted = _typed_resource_admission(attempt)
    if resource_admitted is not None:
        row["sealed_resource_admitted"] = resource_admitted
    return canonical_value(row), candidate


def _finite_comparator_delta(event: SearchUtilityEventV2) -> float | None:
    delta = event.comparator_delta
    if isinstance(delta, bool) or not isinstance(delta, (int, float)):
        return None
    delta = float(delta)
    return delta if math.isfinite(delta) else None


def _recorded_frontier_gain(
    frontier: Mapping[str, Any],
    *,
    event: SearchUtilityEventV2,
    round_index: int,
    legacy_comparator_delta: float,
) -> float:
    candidates: list[Any] = []
    trajectory = frontier.get("effect_trajectory")
    if isinstance(trajectory, (tuple, list)):
        candidates.extend(reversed(trajectory))
    global_bank = frontier.get("global")
    if isinstance(global_bank, Mapping):
        observations = global_bank.get("observations")
        if isinstance(observations, (tuple, list)):
            candidates.extend(reversed(observations))
        candidates.append(global_bank.get("last_observation"))
    for record in candidates:
        if not isinstance(record, Mapping):
            continue
        if (
            record.get("round_index") == round_index
            and record.get("candidate_id") == event.candidate_id
            and record.get("candidate_semantic_digest")
            == event.candidate_semantic_digest
            and isinstance(record.get("frontier_updated"), bool)
        ):
            if not record["frontier_updated"]:
                return 0.0
            if "frontier_delta" not in record:
                return max(0.0, legacy_comparator_delta)
            frontier_delta = record["frontier_delta"]
            if (
                isinstance(frontier_delta, bool)
                or not isinstance(frontier_delta, (int, float))
                or not math.isfinite(float(frontier_delta))
            ):
                return 0.0
            return max(0.0, float(frontier_delta))
    return 0.0


def _typed_parent_validation(interpretation: Any) -> str:
    allowed = {"INDEPENDENT", "VALIDATED", "UNVERIFIED", "FAILED", "UNKNOWN"}
    for owner in (
        interpretation,
        getattr(interpretation, "closure", None),
        getattr(interpretation, "episode", None),
    ):
        for field_name in (
            "stable_validation",
            "parent_validation",
            "validation_status",
            "development_validation_status",
        ):
            value = getattr(owner, field_name, None)
            value = getattr(value, "value", value)
            normalized = str(value).strip().upper() if value is not None else ""
            if normalized in allowed:
                return normalized
    return "UNKNOWN"


def _portfolio_history_projection(
    before: CampaignState,
    successor: ResearchContext,
    result: ResearchRoundResult,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    prior_rows: list[Mapping[str, Any]] = []
    row_candidates: dict[str, PortfolioCandidateV2 | None] = {}
    for attempt in result.attempts:
        projected = _portfolio_attempt_row(
            result,
            attempt,
            round_index=before.next_round_index,
        )
        if projected is None:
            continue
        row, candidate = projected
        prior_rows.append(row)
        row_candidates[attempt.candidate_id] = candidate

    family_rows: list[Mapping[str, Any]] = []
    parent_rows: list[Mapping[str, Any]] = []
    frontier_rows: list[Mapping[str, Any]] = []
    interpretation = result.interpretation
    metric_index = result.metric_bearing_attempt_index
    metric_attempt = (
        result.attempts[metric_index]
        if isinstance(metric_index, int)
        and not isinstance(metric_index, bool)
        and 0 <= metric_index < len(result.attempts)
        else None
    )
    event = getattr(interpretation, "search_utility_event", None)
    episode = getattr(interpretation, "episode", None)
    if (
        metric_attempt is not None
        and episode is not None
        and isinstance(event, SearchUtilityEventV2)
        and event.candidate_id == metric_attempt.candidate_id
        and event.candidate_semantic_digest
        == metric_attempt.binding.mechanism_semantics_digest
    ):
        delta = _finite_comparator_delta(event)
        candidate = row_candidates.get(metric_attempt.candidate_id)
        if delta is not None and candidate is not None:
            source_digest = event.digest
            family_rows.append(
                canonical_value(
                    {
                        "round_index": before.next_round_index,
                        "family_id": candidate.family_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_semantic_digest": candidate.semantic_digest,
                        "stable": True,
                        "stable_delta": delta,
                        "source_digest": source_digest,
                    }
                )
            )
            if candidate.parent_id is not None:
                parent_rows.append(
                    canonical_value(
                        {
                            "round_index": before.next_round_index,
                            "parent_id": candidate.parent_id,
                            "stable": True,
                            "stable_validation": _typed_parent_validation(interpretation),
                            "stable_delta": delta,
                            "source_digest": source_digest,
                        }
                    )
                )
            frontier_gain = _recorded_frontier_gain(
                successor.frontier,
                event=event,
                round_index=before.next_round_index,
                legacy_comparator_delta=delta,
            )
            frontier_rows.append(
                canonical_value(
                    {
                        "round_index": before.next_round_index,
                        "family_id": candidate.family_id,
                        "candidate_id": candidate.candidate_id,
                        "candidate_semantic_digest": candidate.semantic_digest,
                        "stable": True,
                        "stable_frontier_gain": frontier_gain,
                        "source_digest": source_digest,
                    }
                )
            )
    return {
        "portfolio_prior_attempts": tuple(prior_rows),
        "portfolio_family_history": tuple(family_rows),
        "portfolio_parent_history": tuple(parent_rows),
        "portfolio_frontier_history": tuple(frontier_rows),
    }


def _opportunity_ref(state: CampaignState) -> str:
    return "research-opportunity:" + sha256_digest(
        {
            "campaign_id": state.campaign_id,
            "round_index": state.next_round_index,
            "context_digest": state.context.digest,
            "profile_digest": state.active_profile.profile_digest,
            "incumbent_digest": sha256_digest(state.incumbent_observation),
        }
    )


def _fresh_profile_with_same_entries(
    predecessor: SearchExecutableProfileV1,
    *,
    fresh_campaign_id: str,
) -> SearchExecutableProfileV1:
    """Give a promoted policy a real fresh boundary when no capability changed."""

    payload = canonical_value(
        {
            "predecessor_profile_ref": predecessor.profile_ref,
            "predecessor_profile_digest": predecessor.profile_digest,
            "campaign_id": fresh_campaign_id,
            "protocol_ref": predecessor.protocol_ref,
            "protocol_digest": predecessor.protocol_digest,
            "entries": tuple(entry.canonical_dict() for entry in predecessor.entries),
        }
    )
    return SearchExecutableProfileV1(
        campaign_id=fresh_campaign_id,
        profile_ref=content_id(
            "recclaw-research-line-same-entry-next-round-profile-v1", payload
        ),
        profile_digest=sha256_digest(payload),
        protocol_ref=predecessor.protocol_ref,
        protocol_digest=predecessor.protocol_digest,
        activation=SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN,
        predecessor_campaign_id=predecessor.campaign_id,
        predecessor_profile_ref=predecessor.profile_ref,
        predecessor_profile_digest=predecessor.profile_digest,
        entries=predecessor.entries,
    )


def _promoted_incumbent(
    incumbent: Mapping[str, Any], frontier: Mapping[str, Any]
) -> Mapping[str, Any]:
    metric = frontier.get("incumbent_ndcg@10")
    prior = incumbent.get("frozen_ndcg@10")
    if (
        not isinstance(metric, (int, float))
        or isinstance(metric, bool)
        or not math.isfinite(float(metric))
        or not isinstance(prior, (int, float))
        or isinstance(prior, bool)
        or not math.isfinite(float(prior))
        or float(metric) <= float(prior)
        or not isinstance(frontier.get("incumbent_ref"), str)
        or not frontier["incumbent_ref"].strip()
    ):
        return incumbent
    try:
        digest = validate_sha256(
            frontier.get("incumbent_digest"), field_name="incumbent_digest"
        )
    except (TypeError, ValueError):
        return incumbent
    return canonical_value(
        {
            **dict(incumbent),
            "comparator_ref": frontier["incumbent_ref"],
            "comparator_digest": digest,
            "frozen_ndcg@10": float(metric),
        }
    )


def _incumbent_after_metric_round(
    context: ResearchContext,
    incumbent: Mapping[str, Any],
    frontier: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Keep the paired comparator fixed for a single-parent search."""

    baseline_context = context.knowledge_base.get("baseline_context")
    if is_single_parent_context(baseline_context):
        return incumbent
    return _promoted_incumbent(incumbent, frontier)


class ResearchCampaign:
    """Run successive rounds while preserving the arm's executable state."""

    state_filename = "CAMPAIGN_STATE.pkl"
    state_projection_filename = "CAMPAIGN_STATE.json"

    def __init__(
        self,
        *,
        root: Path,
        state: CampaignState,
        producer: ResearchProducer,
        runner: ExperimentRunner,
        round_inputs: CampaignRoundInputs | CampaignRoundInputFactory,
        implementer: ImplementerGateway | None = None,
        memory_writer: SearchMemoryWriterV1 | None = None,
        search_space_adapter: SearchSpaceAdapter | None = None,
        post_round_state_transition: Callable[
            [CampaignState, ResearchRoundResult, int, str], CampaignState
        ]
        | None = None,
        _persist_initial: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self._state = state
        self.producer = producer
        self.runner = runner
        self.round_inputs = round_inputs
        self.implementer = implementer
        self.memory_writer = memory_writer or SearchMemoryWriterV1(
            "DEVELOPMENT_ONLY/SEARCH_MEMORY"
        )
        self.search_space_adapter = search_space_adapter
        self.post_round_state_transition = post_round_state_transition
        self._restore_memory_head()
        if _persist_initial:
            self._initialize_checkpoint()

    @classmethod
    def start(cls, **kwargs: Any) -> "ResearchCampaign":
        return cls(**kwargs)

    @classmethod
    def resume(
        cls,
        *,
        root: Path,
        producer: ResearchProducer,
        runner: ExperimentRunner,
        round_inputs: CampaignRoundInputs | CampaignRoundInputFactory,
        implementer: ImplementerGateway | None = None,
        memory_writer: SearchMemoryWriterV1 | None = None,
        search_space_adapter: SearchSpaceAdapter | None = None,
        post_round_state_transition: Callable[
            [CampaignState, ResearchRoundResult, int, str], CampaignState
        ]
        | None = None,
    ) -> "ResearchCampaign":
        root = Path(root).resolve()
        state_path = root / cls.state_filename
        state = _read_pickle(state_path)
        if not isinstance(state, CampaignState):
            raise CampaignError("campaign checkpoint does not contain CampaignState")
        projection_path = root / cls.state_projection_filename
        if projection_path.is_file():
            projection = _read_json(projection_path)
            if projection.get("state_digest") != state.digest:
                raise CampaignError("campaign state projection digest drift")
        return cls(
            root=root,
            state=state,
            producer=producer,
            runner=runner,
            round_inputs=round_inputs,
            implementer=implementer,
            memory_writer=memory_writer,
            search_space_adapter=search_space_adapter,
            post_round_state_transition=post_round_state_transition,
            _persist_initial=False,
        )

    @classmethod
    def read_checkpoint_state(cls, root: Path) -> CampaignState:
        """Read and validate a campaign state for controller reconstruction."""

        root = Path(root).resolve()
        state = _read_pickle(root / cls.state_filename)
        if not isinstance(state, CampaignState):
            raise CampaignError("campaign checkpoint does not contain CampaignState")
        return state

    load = resume

    @property
    def state(self) -> CampaignState:
        return self._state

    @property
    def current_state(self) -> CampaignState:
        return self._state

    def round_checkpoint_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_CHECKPOINT.pkl"

    def round_trace_path(self, round_index: int) -> Path:
        return self.root / f"ROUND_{round_index:02d}_TRACE.json"

    def _round_generation_prefix(self, round_index: int) -> str:
        generation = _discovery_generation(self._state.context)
        suffix = "" if generation == 0 else f"_GENERATION_{generation:02d}"
        return f"ROUND_{round_index:02d}{suffix}"

    def _started_path(self, round_index: int) -> Path:
        return self.root / f"{self._round_generation_prefix(round_index)}_STARTED.json"

    def _physical_path(self, round_index: int) -> Path:
        return self.root / (
            f"{self._round_generation_prefix(round_index)}_PHYSICAL_OBSERVATION.json"
        )

    def _physical_started_path(self, round_index: int, attempt_index: int) -> Path:
        return self.root / (
            f"{self._round_generation_prefix(round_index)}_"
            f"ATTEMPT_{attempt_index:02d}_PHYSICAL_STARTED.json"
        )

    def _attempt_physical_path(self, round_index: int, attempt_index: int) -> Path:
        return self.root / (
            f"{self._round_generation_prefix(round_index)}_"
            f"ATTEMPT_{attempt_index:02d}_"
            "PHYSICAL_OBSERVATION.json"
        )

    def _attempt_manifest_path(self, round_index: int) -> Path:
        return self.root / (
            f"{self._round_generation_prefix(round_index)}_ATTEMPT_MANIFEST.json"
        )

    def _prepared_round_path(self, round_index: int) -> Path:
        return self.root / (
            f"{self._round_generation_prefix(round_index)}_PREPARED_CHECKPOINT.pkl"
        )

    def verification_checkpoint_path(self, verification_ref: str) -> Path:
        return self.root / f"VERIFICATION_{verification_ref}_CHECKPOINT.pkl"

    def verification_trace_path(self, verification_ref: str) -> Path:
        return self.root / f"VERIFICATION_{verification_ref}_TRACE.json"

    def _verification_started_path(self, verification_ref: str) -> Path:
        return self.root / f"VERIFICATION_{verification_ref}_STARTED.json"

    def _verification_physical_path(self, verification_ref: str) -> Path:
        return self.root / (
            f"VERIFICATION_{verification_ref}_PHYSICAL_OBSERVATION.json"
        )

    @staticmethod
    def _physical_manifest_row(
        payload: Mapping[str, Any],
        path: Path,
    ) -> dict[str, Any]:
        legacy = payload.get("schema") == "recclaw.research-line.physical-observation.v1"
        required = (
            "opportunity_ref",
            "execution_recipe_digest",
            "candidate_run",
        )
        if any(field_name not in payload for field_name in required):
            raise CampaignError(f"physical observation is incomplete: {path}")
        if not isinstance(payload["candidate_run"], Mapping):
            raise CampaignError(f"physical observation candidate_run is invalid: {path}")
        candidate_id = payload.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            candidate_id = "__LEGACY_SINGLE_ATTEMPT__"
        attempt_index = payload.get("attempt_index", 0)
        observation_core = canonical_value(
            {
                key: value
                for key, value in payload.items()
                if key not in {"observation_ref", "observation_digest"}
            }
        )
        return canonical_value(
            {
                "attempt_index": int(attempt_index),
                "candidate_id": candidate_id,
                "binding_digest": payload.get("binding_digest"),
                "execution_recipe_digest": str(payload["execution_recipe_digest"]),
                "candidate_run": dict(payload["candidate_run"]),
                "physical_run_id": payload.get("physical_run_id"),
                "physical_seed": payload.get("physical_seed"),
                "physical_context_digest": payload.get("physical_context_digest"),
                "research_context_digest": payload.get("research_context_digest"),
                "cuda_visible_devices": payload.get("cuda_visible_devices"),
                "reservation_digest": payload.get("reservation_digest"),
                "reservation_status": payload.get("reservation_status"),
                "final_worker_ceiling_seconds": payload.get(
                    "final_worker_ceiling_seconds"
                ),
                "physical_context_applied": payload.get(
                    "physical_context_applied", False
                ),
                "physical_observation_path": str(path),
                "observation_ref": str(
                    payload.get("observation_ref")
                    or content_id("recclaw-research-line-physical-observation-v1", observation_core)
                ),
                "observation_digest": str(
                    payload.get("observation_digest") or sha256_digest(observation_core)
                ),
                "legacy_single_attempt": legacy,
            }
        )

    def _write_attempt_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_digest: str,
        attempt_budget: int | None,
        attempts: Sequence[Mapping[str, Any]],
        status: str = "IN_PROGRESS",
        metric_bearing_attempt_index: int | None = None,
        incomplete_reason: str | None = None,
        evidence_pre_trace: Sequence[Mapping[str, Any]] = (),
        evidence_post_trace: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        payload = {
                "schema": "recclaw.research-line.round-attempt-manifest.v1",
                "round_index": round_index,
                "opportunity_ref": opportunity_ref,
                "state_digest": state_digest,
                "context_digest": context_digest,
                "profile_digest": profile_digest,
                "attempt_budget": attempt_budget,
                "status": status,
                "metric_bearing_attempt_index": metric_bearing_attempt_index,
                "incomplete_reason": incomplete_reason,
                "attempts": tuple(attempts),
            }
        if evidence_pre_trace:
            payload["evidence_pre_trace"] = tuple(evidence_pre_trace)
        if evidence_post_trace:
            payload["evidence_post_trace"] = tuple(evidence_post_trace)
        payload = canonical_value(payload)
        _atomic_write(
            self._attempt_manifest_path(round_index),
            canonical_json_bytes(payload) + b"\n",
        )
        return payload

    def _load_attempt_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_digest: str,
        attempt_budget: int | None,
    ) -> dict[str, Any]:
        manifest_path = self._attempt_manifest_path(round_index)
        if manifest_path.is_file():
            raw = dict(_read_json(manifest_path))
            if raw.get("opportunity_ref") != opportunity_ref:
                raise CampaignError("round attempt manifest opportunity identity drift")
            for field_name, expected in (
                ("state_digest", state_digest),
                ("context_digest", context_digest),
                ("profile_digest", profile_digest),
            ):
                if raw.get(field_name) != expected:
                    raise CampaignError(f"round attempt manifest {field_name} drift")
            if (
                attempt_budget is not None
                and raw.get("attempt_budget", attempt_budget) != attempt_budget
            ):
                raise CampaignError("round attempt manifest attempt_budget drift")
            raw_attempts = raw.get("attempts", ())
            if not isinstance(raw_attempts, (tuple, list)):
                raise CampaignError("round attempt manifest attempts are invalid")
            # This manifest is the idempotence authority for physical runner
            # work.  Older builds also placed prebinding diagnostics here;
            # those rows have no candidate_run and are not runnable attempts.
            attempts = [
                canonical_value(dict(item))
                for item in raw_attempts
                if isinstance(item, Mapping)
                and isinstance(item.get("candidate_run"), Mapping)
            ]
        else:
            attempts = []
            raw = {}

        known_by_index: dict[int, dict[str, Any]] = {}
        known_candidates: set[str] = set()
        for item in attempts:
            try:
                index = int(item["attempt_index"])
            except (KeyError, TypeError, ValueError) as error:
                raise CampaignError("round attempt manifest index is invalid") from error
            if index < 0 or index in known_by_index:
                raise CampaignError("round attempt manifest index is duplicated")
            candidate_id = item.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise CampaignError("round attempt manifest candidate identity is invalid")
            if candidate_id != "__LEGACY_SINGLE_ATTEMPT__":
                if candidate_id in known_candidates:
                    raise CampaignError("round attempt manifest candidate is duplicated")
                known_candidates.add(candidate_id)
            known_by_index[index] = item
        physical_paths = []
        legacy_path = self._physical_path(round_index)
        if legacy_path.is_file():
            physical_paths.append(legacy_path)
        generation_prefix = self._round_generation_prefix(round_index)
        physical_paths.extend(
            sorted(
                self.root.glob(
                    f"{generation_prefix}_ATTEMPT_*_PHYSICAL_OBSERVATION.json"
                )
            )
        )
        changed = not manifest_path.is_file() or "attempt_budget" not in raw
        for physical_path in physical_paths:
            physical = _read_json(physical_path)
            if physical.get("opportunity_ref") != opportunity_ref:
                raise CampaignError("physical observation opportunity identity drift")
            if physical.get("round_index") is not None and int(
                physical.get("round_index", -1)
            ) != round_index:
                raise CampaignError("physical observation round index drift")
            row = self._physical_manifest_row(physical, physical_path)
            index = int(row["attempt_index"])
            prior = known_by_index.get(index)
            if prior is None:
                if (
                    row["candidate_id"] != "__LEGACY_SINGLE_ATTEMPT__"
                    and row["candidate_id"] in known_candidates
                ):
                    raise CampaignError("round attempt manifest candidate is duplicated")
                known_by_index[index] = row
                changed = True
            elif (
                prior.get("observation_digest") != row["observation_digest"]
                or (
                    prior.get("candidate_id") != row["candidate_id"]
                    and not (
                        row.get("legacy_single_attempt") is True
                        and prior.get("candidate_id")
                        != "__LEGACY_SINGLE_ATTEMPT__"
                    )
                )
            ):
                raise CampaignError("round attempt manifest conflicts with physical observation")
            if row["candidate_id"] != "__LEGACY_SINGLE_ATTEMPT__":
                known_candidates.add(row["candidate_id"])
        attempts = [known_by_index[index] for index in sorted(known_by_index)]
        if [int(item["attempt_index"]) for item in attempts] != list(range(len(attempts))):
            raise CampaignError("round attempt manifest indices are not contiguous")
        result = {
            "schema": "recclaw.research-line.round-attempt-manifest.v1",
            "round_index": round_index,
            "opportunity_ref": opportunity_ref,
            "state_digest": state_digest,
            "context_digest": context_digest,
            "profile_digest": profile_digest,
            "attempt_budget": raw.get("attempt_budget", attempt_budget),
            "status": raw.get("status", "IN_PROGRESS"),
            "metric_bearing_attempt_index": raw.get("metric_bearing_attempt_index"),
            "incomplete_reason": raw.get("incomplete_reason"),
            "attempts": attempts,
        }
        for field_name in ("evidence_pre_trace", "evidence_post_trace"):
            if raw.get(field_name):
                result[field_name] = canonical_value(tuple(raw[field_name]))
        if changed:
            self._write_attempt_manifest(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=state_digest,
                context_digest=context_digest,
                profile_digest=profile_digest,
                attempt_budget=attempt_budget,
                attempts=attempts,
                status=str(result["status"]),
                metric_bearing_attempt_index=result["metric_bearing_attempt_index"],
                incomplete_reason=result["incomplete_reason"],
                evidence_pre_trace=result.get("evidence_pre_trace", ()),
                evidence_post_trace=result.get("evidence_post_trace", ()),
            )
        return result

    def _validate_prepared_round(
        self,
        *,
        prepared: Any,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
    ) -> PreparedResearchRoundV1:
        if not isinstance(prepared, PreparedResearchRoundV1):
            raise CampaignError("prepared round checkpoint has an invalid prepared round")
        if prepared.context_digest != state_before.context.digest:
            raise CampaignError("prepared round context identity drift")
        if (
            prepared.profile_ref != state_before.active_profile.profile_ref
            or prepared.profile_digest != state_before.active_profile.profile_digest
        ):
            raise CampaignError("prepared round profile identity drift")
        slate = prepared.search_slate
        if slate is not None:
            slate_profile_identity = (
                slate.campaign_id,
                slate.profile_ref,
                slate.profile_digest,
            )
            active_profile_identity = (
                state_before.active_profile.campaign_id,
                state_before.active_profile.profile_ref,
                state_before.active_profile.profile_digest,
            )
            if slate_profile_identity != active_profile_identity:
                innovation = prepared.innovation
                if innovation is None or not innovation.activation_ready:
                    raise CampaignError("prepared round search slate identity drift")
                try:
                    execution_profile = activate_next_fresh_search_profile(
                        predecessor=state_before.active_profile,
                        next_profile=innovation.next_profile,
                        registry=innovation.registry,
                        fresh_campaign_id=innovation.fresh_campaign_id,
                    )
                    expected_slate = freeze_experiment_slate(
                        profile=execution_profile,
                        bindings=prepared.search_bindings,
                        budget_snapshot=budget_snapshot,
                    )
                except (TypeError, ValueError) as exc:
                    raise CampaignError(
                        "prepared round search slate identity drift"
                    ) from exc
                if slate != expected_slate:
                    raise CampaignError("prepared round search slate identity drift")
        if slate is not None and sha256_digest(slate.budget_snapshot) != sha256_digest(
            budget_snapshot
        ):
            raise CampaignError("prepared round budget snapshot drift")
        if isinstance(round_index, bool) or round_index < 1:
            raise CampaignError("prepared round index is invalid")
        if not isinstance(opportunity_ref, str) or not opportunity_ref:
            raise CampaignError("prepared round opportunity identity is invalid")
        if round_index != state_before.next_round_index:
            raise CampaignError("prepared round index is not the current opportunity")
        if opportunity_ref != _opportunity_ref(state_before):
            raise CampaignError("prepared round opportunity identity drift")
        if (
            isinstance(attempt_budget, bool)
            or not isinstance(attempt_budget, int)
            or attempt_budget < 0
        ):
            raise CampaignError("prepared round attempt budget is invalid")
        return prepared

    def _persist_prepared_round(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
        prepared: PreparedResearchRoundV1,
    ) -> PreparedResearchRoundV1:
        """Atomically persist the scheduler's non-physical round boundary."""

        self._validate_prepared_round(
            prepared=prepared,
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_before=state_before,
            attempt_budget=attempt_budget,
            budget_snapshot=budget_snapshot,
        )
        payload = {
            "schema": "recclaw.research-line.prepared-round-checkpoint.v1",
            "round_index": round_index,
            "opportunity_ref": opportunity_ref,
            "state_digest": state_before.digest,
            "context_digest": state_before.context.digest,
            "profile_ref": state_before.active_profile.profile_ref,
            "profile_digest": state_before.active_profile.profile_digest,
            "attempt_budget": attempt_budget,
            "budget_snapshot_digest": sha256_digest(budget_snapshot),
            "research_context_projection_version": (
                RESEARCH_CONTEXT_PROJECTION_VERSION
            ),
            "original_active_parent_projection_version": (
                ORIGINAL_ACTIVE_PARENT_PROJECTION_VERSION
                if _is_original_matched_context(state_before.context)
                else None
            ),
            "effect_feedback_projection_digest": (
                _research_context_projection_digest(state_before.context)
            ),
            "prepared_digest": prepared.digest,
            "prepared": prepared,
        }
        path = self._prepared_round_path(round_index)
        if path.is_file():
            existing = self._load_prepared_round(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=state_before.digest,
                context_digest=state_before.context.digest,
                profile_ref=state_before.active_profile.profile_ref,
                profile_digest=state_before.active_profile.profile_digest,
                attempt_budget=attempt_budget,
                budget_snapshot=budget_snapshot,
            )
            if existing is None:
                raise CampaignError("prepared round checkpoint identity drift")
            if existing.digest == prepared.digest:
                return existing
            existing_retry = getattr(existing, "prebinding_retry", None)
            next_retry = getattr(prepared, "prebinding_retry", None)
            existing_status = (
                existing_retry.get("status")
                if isinstance(existing_retry, Mapping)
                else None
            )
            next_status = (
                next_retry.get("status")
                if isinstance(next_retry, Mapping)
                else None
            )
            existing_partial = (
                isinstance(existing_retry, Mapping)
                and not existing.resolutions
                and not existing.search_bindings
            )
            next_partial = (
                isinstance(next_retry, Mapping)
                and not prepared.resolutions
                and not prepared.search_bindings
            )
            existing_transient_provider_failure = (
                self._prepared_round_has_unfinished_provider_failure(
                    existing,
                    self.producer,
                )
            )
            existing_recoverable_resource_failure = (
                self._prepared_round_has_recoverable_resource_failure(existing)
                or _prepared_has_external_implementation_failure(existing)
            )
            monotonic_candidate_progression = (
                self._prepared_round_is_monotonic_candidate_progression(
                    existing,
                    prepared,
                    attempt_budget=attempt_budget,
                )
            )
            existing_resource_retry_digests = {
                tuple(item.digest for item in existing.producer_outcomes),
                (
                    existing.innovation.selected_outcome.digest,
                )
                if existing.innovation is not None
                else (),
            }
            allowed_transition = (
                existing_partial
                and existing_status == "PENDING"
                and next_status in {"SUCCEEDED", "EXHAUSTED"}
                and isinstance(next_retry, Mapping)
                and next_retry.get("ordinal") == existing_retry.get("ordinal")
            ) or (
                existing_partial
                and existing_status == "SUCCEEDED"
                and next_status == "SUCCEEDED"
                and not next_partial
            ) or (
                existing_transient_provider_failure
                and next_retry is None
            ) or (
                existing_recoverable_resource_failure
                and next_retry is None
                and tuple(item.digest for item in prepared.producer_outcomes)
                in existing_resource_retry_digests
                and (
                    not self._prepared_round_adds_attempts(existing, prepared)
                    or self._prepared_round_appends_same_candidate_retry(
                        existing, prepared
                    )
                )
            ) or (
                monotonic_candidate_progression
            )
            if not allowed_transition:
                raise CampaignError("prepared round checkpoint identity drift")
            _atomic_write(
                path,
                pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
            )
            return prepared
        _atomic_write(
            path,
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL),
        )
        return prepared

    @staticmethod
    def _prepared_round_has_terminal_provider_failure(
        prepared: PreparedResearchRoundV1,
    ) -> bool:
        return bool(
            not prepared.search_bindings
            and prepared.innovation is None
            and any(
                any(
                    isinstance(attempt, Mapping)
                    and (
                        attempt.get("failure_class") == "CLI_CONTRACT_ERROR"
                        or attempt.get("reason_code")
                        in {
                            "CONTENT_JSON_DECODE",
                            "ENVELOPE_JSON_DECODE",
                            "SCHEMA_VALIDATION",
                        }
                    )
                    for attempt in trace.get("attempts", ())
                )
                for trace in prepared.provider_traces
                if isinstance(trace, Mapping)
            )
        )

    @staticmethod
    def _prepared_round_has_unfinished_provider_failure(
        prepared: PreparedResearchRoundV1,
        producer: ResearchProducer | None = None,
    ) -> bool:
        """Whether preparation stopped at an unfinished Provider boundary."""
        transient = _prepared_has_unfinished_producer_failure(prepared)
        if not transient or prepared.provider_traces or producer is None:
            return transient
        return callable(
            getattr(producer, "configure_started_round_provider_replay", None)
        )

    @staticmethod
    def _prepared_round_has_recoverable_resource_failure(
        prepared: PreparedResearchRoundV1,
    ) -> bool:
        return _prepared_has_recoverable_resource_failure(prepared)

    @staticmethod
    def _prepared_round_is_monotonic_candidate_progression(
        existing: PreparedResearchRoundV1,
        prepared: PreparedResearchRoundV1,
        *,
        attempt_budget: int,
    ) -> bool:
        """Allow only same-slate growth after a candidate-local consumption.

        A recovery may durably reclassify one resource attempt and then consume
        another frozen candidate before the physical runner boundary.  The
        immutable round inputs and the complete prior attempt prefix must stay
        byte-equivalent; only the innovation result and its derived search
        projection may advance.
        """

        prior = existing.innovation
        current = prepared.innovation
        if (
            prior is None
            or current is None
            or prior.activation_ready
            or existing.search_bindings
            or existing.prebinding_retry is not None
            or prepared.prebinding_retry is not None
        ):
            return False
        prior_attempts = tuple(prior.attempts)
        current_attempts = tuple(current.attempts)
        prior_candidate_count = ResearchCampaign._prepared_candidate_attempt_count(
            prior_attempts
        )
        current_candidate_count = ResearchCampaign._prepared_candidate_attempt_count(
            current_attempts
        )
        prior_prefix = tuple(canonical_value(item) for item in prior_attempts)
        current_prefix = tuple(
            canonical_value(item)
            for item in current_attempts[: len(prior_attempts)]
        )
        if (
            not prior_attempts
            or len(current_attempts) <= len(prior_attempts)
            or current_prefix != prior_prefix
            or prior.candidate_attempt_count != prior_candidate_count
            or current.candidate_attempt_count != current_candidate_count
            or current_candidate_count <= prior_candidate_count
            or current_candidate_count > attempt_budget
        ):
            return False
        # Persistence must accept the same consumed-candidate transition as
        # runtime, including exhausted implementation repair without a scope.
        if not _prepared_has_untried_candidate_local_innovation(
            existing, attempt_budget=attempt_budget,
        ):
            return False
        frozen_fields = (
            "context_digest",
            "profile_ref",
            "profile_digest",
            "producer_outcomes",
            "carryover_outcomes",
            "resolutions",
            "deferred_search_outcomes",
            "metric_contract_digest",
            "observation_seed",
            "next_discriminative_test",
            "confirmation_seed",
        )
        if any(
            sha256_digest(getattr(existing, name))
            != sha256_digest(getattr(prepared, name))
            for name in frozen_fields
        ):
            return False
        prior_provider_traces = tuple(
            canonical_value(item) for item in existing.provider_traces
        )
        current_provider_trace_prefix = tuple(
            canonical_value(item)
            for item in prepared.provider_traces[: len(prior_provider_traces)]
        )
        if (
            len(prepared.provider_traces) < len(prior_provider_traces)
            or current_provider_trace_prefix != prior_provider_traces
        ):
            return False
        appended_spec_digests = {
            _innovation_attempt_proposal_digest(item)
            for item in current_attempts[len(prior_attempts) :]
            if isinstance(item, Mapping)
            and isinstance(_innovation_attempt_proposal_digest(item), str)
        }
        # The immutable producer slate owns candidate identity. The deferred
        # queue is only a derived execution projection: an older selected-only
        # resource retry could have persisted it empty without consuming the
        # remaining frozen candidates.
        prior_spec_digests = {
            _innovation_attempt_proposal_digest(item) for item in prior_attempts
        }
        prior_deferred = tuple(
            pair for pair in existing.resolutions
            if pair[0].spec is not None
            and pair[0].spec.digest not in prior_spec_digests
            and pair[1] is not None
            and pair[1].resolution
            is CapabilityResolutionResultV1.INNOVATION_REQUIRED
        )
        prior_deferred_spec_digests = {
            outcome.spec.digest
            for outcome, _resolution in prior_deferred
            if outcome.spec is not None
        }
        if (
            any(outcome.spec is None for outcome, _resolution in prior_deferred)
            or not appended_spec_digests
            or not appended_spec_digests.issubset(prior_deferred_spec_digests)
        ):
            return False
        expected_deferred = tuple(
            pair
            for pair in prior_deferred
            if pair[0].spec.digest not in appended_spec_digests
        )
        consumed_proposal_digests = {
            pair[0].spec.digest
            for pair in prior_deferred
            if pair[0].spec.digest in appended_spec_digests
        }
        if (
            # Runtime eligibility/ranking may omit already-explored frozen
            # outcomes. A derived queue may neither invent nor retain consumed
            # candidates, but it is not a second candidate-universe authority.
            not {sha256_digest(pair) for pair in prepared.deferred_innovation_outcomes}
            .issubset({sha256_digest(pair) for pair in expected_deferred})
            or current.idea_acquisition.selected_spec_digest not in consumed_proposal_digests
        ):
            return False
        producer_digests = {
            item.spec.digest for item in existing.producer_outcomes if item.spec is not None
        }
        return (
            prior.idea_acquisition.selected_spec_digest in producer_digests
            and current.idea_acquisition.selected_spec_digest in producer_digests
        )

    @staticmethod
    def _prepared_round_appends_same_candidate_retry(
        existing: PreparedResearchRoundV1,
        prepared: PreparedResearchRoundV1,
    ) -> bool:
        if existing.innovation is None or prepared.innovation is None:
            return False
        prior = tuple(existing.innovation.attempts)
        current = tuple(prepared.innovation.attempts)
        return (
            bool(prior)
            and len(current) > len(prior)
            and canonical_value(current[:len(prior)]) == canonical_value(prior)
            and all(
                _innovation_attempt_proposal_digest(item)
                == existing.innovation.idea_acquisition.selected_spec_digest
                for item in current[len(prior):]
            )
            and prepared.innovation.idea_acquisition.selected_spec_digest
            == existing.innovation.idea_acquisition.selected_spec_digest
        )

    @staticmethod
    def _prepared_candidate_attempt_count(
        attempts: Sequence[Mapping[str, Any]],
    ) -> int:
        def consumes_candidate_attempt(item: Mapping[str, Any]) -> bool:
            scope = item.get("failure_scope")
            failure = item.get("failure")
            if scope is None and isinstance(failure, Mapping):
                scope = failure.get("failure_scope")
            return str(scope or "").upper() not in {
                "WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE", "RECOVERY",
            }

        return len(
            {
                _innovation_attempt_proposal_digest(item)
                for item in attempts
                if isinstance(item, Mapping)
                and _innovation_attempt_proposal_digest(item) is not None
                and isinstance(item.get("candidate_root"), str)
                and bool(str(item["candidate_root"]).strip())
                and consumes_candidate_attempt(item)
            }
        )

    @staticmethod
    def _prepared_round_adds_attempts(
        existing: PreparedResearchRoundV1,
        prepared: PreparedResearchRoundV1,
    ) -> bool:
        if existing.innovation is None or prepared.innovation is None:
            return False
        return len(prepared.innovation.attempts) > len(
            existing.innovation.attempts
        )

    def _load_prepared_round(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_digest: str,
        context_digest: str,
        profile_ref: str,
        profile_digest: str,
        attempt_budget: int,
        budget_snapshot: Mapping[str, Any],
    ) -> PreparedResearchRoundV1 | None:
        path = self._prepared_round_path(round_index)
        if not path.is_file():
            return None
        payload = _read_pickle(path)
        if not isinstance(payload, Mapping):
            raise CampaignError("prepared round checkpoint root is invalid")
        if payload.get("schema") != "recclaw.research-line.prepared-round-checkpoint.v1":
            raise CampaignError("prepared round checkpoint schema drift")
        for field_name, expected in (
            ("round_index", round_index),
            ("opportunity_ref", opportunity_ref),
            ("state_digest", state_digest),
            ("context_digest", context_digest),
            ("profile_ref", profile_ref),
            ("profile_digest", profile_digest),
            ("attempt_budget", attempt_budget),
            ("budget_snapshot_digest", sha256_digest(budget_snapshot)),
        ):
            if payload.get(field_name) != expected:
                raise CampaignError(f"prepared round checkpoint {field_name} drift")
        prepared = self._validate_prepared_round(
            prepared=payload.get("prepared"),
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_before=self._state,
            attempt_budget=attempt_budget,
            budget_snapshot=budget_snapshot,
        )
        if payload.get("prepared_digest") != prepared.digest:
            raise CampaignError("prepared round checkpoint digest drift")
        return prepared

    def _prepared_context_projection_status(
        self,
        *,
        round_index: int,
        context: ResearchContext,
        prepared: PreparedResearchRoundV1,
    ) -> str:
        """Return CURRENT or LEGACY for an otherwise validated checkpoint."""

        payload = _read_pickle(self._prepared_round_path(round_index))
        if not isinstance(payload, Mapping):
            raise CampaignError("prepared round checkpoint root is invalid")
        if _stale_original_active_parent_projection(
            context=context,
            prepared=prepared,
            checkpoint_payload=payload,
        ):
            return "LEGACY"
        version = payload.get("research_context_projection_version")
        digest = payload.get("effect_feedback_projection_digest")
        if version is None and digest is None:
            return "LEGACY"
        if isinstance(version, bool) or not isinstance(version, int):
            raise CampaignError("prepared round projection version is invalid")
        if version < RESEARCH_CONTEXT_PROJECTION_VERSION:
            return "LEGACY"
        if version > RESEARCH_CONTEXT_PROJECTION_VERSION:
            raise CampaignError("prepared round projection version is newer than source")
        if digest != _research_context_projection_digest(context):
            # No returned research consumed this projection. Resume the same
            # unfinished request boundary without spending a discovery generation.
            if not prepared.provider_traces and _prepared_has_unfinished_producer_failure(prepared):
                return "CURRENT"
            raise CampaignError("prepared round effect-feedback projection digest drift")
        return "CURRENT"

    def _prepared_round_has_committed_main_execution(
        self,
        *,
        round_index: int,
        prepared: PreparedResearchRoundV1,
        manifest: Mapping[str, Any],
    ) -> bool:
        """Conservatively preserve a prepared slate once main execution started."""

        if manifest.get("attempts"):
            return True
        started_paths = tuple(
            self.root.glob(
                f"{self._round_generation_prefix(round_index)}_"
                "ATTEMPT_*_PHYSICAL_STARTED.json"
            )
        )
        for started_path in started_paths:
            marker = _read_json(started_path)
            binding_digests = {binding.digest for binding in prepared.search_bindings}
            if marker.get("binding_digest") not in binding_digests:
                raise CampaignError("physical start marker binding identity drift")
            return True
        binding_digests = {binding.digest for binding in prepared.search_bindings}
        mechanism_ids = {
            getattr(binding.proposal, "compiler_candidate_id", None)
            for binding in prepared.search_bindings
            if isinstance(
                getattr(binding.proposal, "compiler_candidate_id", None), str
            )
            and getattr(binding.proposal, "compiler_candidate_id", None)
        }
        if not binding_digests:
            return False
        execution_root = self.root / "execution" / "experiments"
        for confirmation_path in execution_root.glob(
            "*/worker/start_confirmation.json"
        ):
            confirmation = _read_json(confirmation_path)
            if confirmation.get("binding_digest") in binding_digests:
                return True
            recipe_path = confirmation_path.parents[1] / "execution_recipe.json"
            if recipe_path.is_file():
                recipe = _read_json(recipe_path)
                if recipe.get("mechanism_id") in mechanism_ids:
                    return True
        return False

    def _engineering_superseded_start_marker(
        self,
        *,
        round_index: int,
        manifest: Mapping[str, Any],
        engineering_source_identity: Mapping[str, Any] | None,
    ) -> tuple[Path, Mapping[str, Any]] | None:
        """Identify a source-stale start with neither observation nor worker result."""

        if engineering_source_identity is None or manifest.get("attempts"):
            return None
        started_paths = tuple(
            sorted(
                self.root.glob(
                    f"{self._round_generation_prefix(round_index)}_"
                    "ATTEMPT_*_PHYSICAL_STARTED.json"
                )
            )
        )
        if not started_paths:
            return None
        if len(started_paths) != 1:
            raise CampaignError("source migration found ambiguous physical starts")
        started_path = started_paths[0]
        marker = _read_json(started_path)
        if marker.get("engineering_source_identity") == canonical_value(
            dict(engineering_source_identity)
        ):
            return None
        attempt_index = marker.get("attempt_index")
        if isinstance(attempt_index, bool) or not isinstance(attempt_index, int):
            raise CampaignError("physical start marker attempt identity is invalid")
        if self._attempt_physical_path(round_index, attempt_index).is_file():
            return None
        checker = getattr(self.runner, "has_durable_worker_result", None)
        if not callable(checker):
            return None
        context_digest = marker.get("physical_context_digest")
        if not isinstance(context_digest, str) or checker(context_digest):
            return None
        return started_path, marker

    def _restore_memory_head(self) -> None:
        expected = self._state.search_memory_head
        actual = self.memory_writer.head
        if expected is None:
            if actual is not None:
                self._state = replace(self._state, search_memory_head=actual)
            return
        if actual is None:
            # SearchMemoryWriterV1 is intentionally a small in-process writer;
            # restore its head from the trusted local checkpoint on resume.
            self.memory_writer._head = expected
        elif actual.digest != expected.digest:
            raise CampaignError("injected Search Memory writer is ahead of checkpoint")

    def _round_memory_writer(self) -> SearchMemoryWriterV1:
        """Keep an incomplete round from mutating the campaign Memory head."""

        writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
        writer._head = self._state.search_memory_head
        return writer

    def _initialize_checkpoint(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        state_path = self.root / self.state_filename
        if state_path.exists():
            existing = _read_pickle(state_path)
            if not isinstance(existing, CampaignState) or existing.digest != self._state.digest:
                raise CampaignError("campaign root already contains different state")
            return
        self._persist_state()

    def _persist_state(self) -> None:
        state_payload = pickle.dumps(self._state, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(self.root / self.state_filename, state_payload)
        projection = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-state-projection.v1",
                "state_digest": self._state.digest,
                "state": self._state.to_dict(),
            }
        )
        _atomic_write(
            self.root / self.state_projection_filename,
            canonical_json_bytes(projection) + b"\n",
        )

    def _task_queue(self) -> ResearchTaskQueueV2:
        memory = self._state.context.scientific_memory
        global_memory = memory.get("global_memory")
        if not isinstance(global_memory, Mapping):
            global_memory = memory.get("global")
        if not isinstance(global_memory, Mapping):
            global_memory = memory
        return ResearchTaskQueueV2.from_dict(global_memory.get("task_queue"))

    @staticmethod
    def _verification_seed(
        task: ResearchTaskRecordV2,
        inputs: CampaignRoundInputs,
    ) -> str:
        executable_operations = {
            ResearchTaskOperationV2.NEW_SEED,
            ResearchTaskOperationV2.REPRODUCE,
            ResearchTaskOperationV2.MATCHED_CONTROL,
            ResearchTaskOperationV2.MECHANISM_OFF,
        }
        if task.operation not in executable_operations:
            raise CampaignError(
                f"queued task {task.task_id} is not an executable verification"
            )
        if task.operation in {
            ResearchTaskOperationV2.MATCHED_CONTROL,
            ResearchTaskOperationV2.MECHANISM_OFF,
        }:
            value = task.metadata.get("verification_seed")
            if not isinstance(value, str) or not value:
                raise CampaignError(
                    f"queued control {task.task_id} lacks its exact verification seed"
                )
            seed = value
        elif task.required_seed_or_control != "NEXT_DEVELOPMENT_SEED":
            seed = task.required_seed_or_control
        else:
            schedule = (
                inputs.verification_seed_schedule
                or inputs.observation_seed_schedule
            )
            if not schedule:
                raise CampaignError(
                    f"queued task {task.task_id} needs a frozen unseen seed schedule"
                )
            evidence = set(task.evidence_present)
            seed = next(
                (str(item) for item in schedule if str(item) not in evidence),
                "",
            )
            if not seed:
                raise CampaignError(
                    f"queued task {task.task_id} exhausted the frozen seed schedule"
                )
        schedule = (
            inputs.verification_seed_schedule
            or inputs.observation_seed_schedule
        )
        if schedule is not None and seed not in {str(item) for item in schedule}:
            raise CampaignError(
                f"verification seed {seed} is outside the frozen observation schedule"
            )
        return seed

    def _verification_inputs(
        self,
        task: ResearchTaskRecordV2,
        observation_seed: str,
    ) -> CampaignRoundInputs:
        base = self._round_inputs()
        budget = {
            **dict(base.budget_snapshot),
            "experiment_opportunities": 1,
            "max_attempts_per_round": 1,
            "round_attempt_budget": 1,
            "remaining_attempt_budget": 1,
            "attempt_budget": 1,
        }
        return replace(
            base,
            budget_snapshot=canonical_value(budget),
            observation_seed=observation_seed,
            confirmation_seed=None,
            next_discriminative_test=(
                f"Execute exact auxiliary verification task {task.task_id}."
            ),
            innovation_inputs=None,
            meta_research_inputs=None,
            attempt_scheduler=True,
            max_attempts_per_round=1,
            prebinding_token_ceiling_retry=False,
            close_exhausted_no_metric_slot=False,
            bootstrap_fixed_candidates=False,
            portfolio_candidates=(),
            observation_seed_schedule=(
                base.verification_seed_schedule
                or base.observation_seed_schedule
            ),
            round_role="VERIFICATION",
            search_space_adapter=(
                base.search_space_adapter or self.search_space_adapter
            ),
        )

    def _runner_for_verification(
        self,
        *,
        task: ResearchTaskRecordV2,
        verification_ref: str,
        observation_seed: str,
        state_before: CampaignState,
    ) -> ExperimentRunner:
        path = self._verification_physical_path(verification_ref)

        def guarded_runner(
            recipe: Mapping[str, Any], binding: Any
        ) -> Mapping[str, Any]:
            candidate_id = getattr(
                getattr(binding, "proposal", None), "candidate_id", None
            )
            if not isinstance(candidate_id, str) or not candidate_id:
                raise CampaignError("verification binding lacks candidate identity")
            if path.is_file():
                payload = _read_json(path)
                for field_name, expected in (
                    ("verification_ref", verification_ref),
                    ("task_id", task.task_id),
                    ("observation_seed", observation_seed),
                    ("candidate_id", candidate_id),
                    ("execution_recipe_digest", sha256_digest(recipe)),
                ):
                    if payload.get(field_name) != expected:
                        raise CampaignError(
                            f"verification physical observation {field_name} drift"
                        )
                value = payload.get("candidate_run")
                if not isinstance(value, Mapping):
                    raise CampaignError(
                        "verification physical observation lacks candidate_run"
                    )
                return value

            physical_context_value = canonical_value(
                {
                    "schema": PHYSICAL_CONTEXT_SCHEMA,
                    "campaign_id": state_before.campaign_id,
                    "round_index": state_before.next_round_index,
                    "opportunity_ref": _opportunity_ref(state_before),
                    "attempt_index": 0,
                    "execution_lane": "AUXILIARY_VERIFICATION",
                    "verification_ref": verification_ref,
                    "task_id": task.task_id,
                    "task_digest": task.digest,
                    "operation": task.operation.value,
                    "candidate_id": candidate_id,
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_semantic_digest": getattr(
                        binding, "mechanism_semantics_digest", None
                    ),
                    "research_context_digest": state_before.context.digest,
                    "profile_digest": state_before.active_profile.profile_digest,
                    "seed": observation_seed,
                }
            )
            context_digest = sha256_digest(physical_context_value)
            physical_context = _immutable_canonical_mapping(
                physical_context_value
            )
            context_runner = getattr(self.runner, "run_with_physical_context", None)
            if context_runner is not None and not callable(context_runner):
                raise CampaignError(
                    "runner run_with_physical_context is not callable"
                )
            if context_runner is None:
                value = self.runner(recipe, binding)
            else:
                value = context_runner(recipe, binding, physical_context)
            if not isinstance(value, Mapping):
                raise CampaignError("injected runner must return a mapping")
            candidate_run = canonical_value(dict(value))
            observation_core = canonical_value(
                {
                    "schema": (
                        "recclaw.research-line.verification-physical-observation.v1"
                    ),
                    "verification_ref": verification_ref,
                    "task_id": task.task_id,
                    "task_digest": task.digest,
                    "operation": task.operation.value,
                    "observation_seed": observation_seed,
                    "round_index": state_before.next_round_index,
                    "candidate_id": candidate_id,
                    "execution_recipe_digest": sha256_digest(recipe),
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_run": candidate_run,
                    **_physical_identity_fields(
                        candidate_run,
                        context_digest=context_digest,
                        context_applied=context_runner is not None,
                    ),
                }
            )
            payload = canonical_value(
                {
                    **dict(observation_core),
                    "observation_ref": content_id(
                        "recclaw-research-line-verification-observation-v1",
                        observation_core,
                    ),
                    "observation_digest": sha256_digest(observation_core),
                }
            )
            if not _write_once(path, canonical_json_bytes(payload) + b"\n"):
                existing = _read_json(path)
                if canonical_value(dict(existing)) != payload:
                    raise CampaignError(
                        "verification physical observation was sealed concurrently"
                    )
            return candidate_run

        return guarded_runner

    @staticmethod
    def _attach_verification_observation(
        result: ResearchRoundResult,
        payload: Mapping[str, Any],
    ) -> ResearchRoundResult:
        if len(result.attempts) != 1:
            return result
        attempt = replace(
            result.attempts[0],
            observation_ref=str(payload["observation_ref"]),
            observation_digest=str(payload["observation_digest"]),
        )
        return replace(result, attempts=(attempt,))

    @staticmethod
    def _advance_verification_state(
        before: CampaignState,
        task: ResearchTaskRecordV2,
        result: ResearchRoundResult,
    ) -> CampaignState:
        interpretation = result.interpretation
        if interpretation is None:
            raise CampaignError("verification did not produce typed feedback")
        successor = interpretation.successor_context
        policy = interpretation.policy_successor
        if successor.round_index != before.next_round_index:
            raise CampaignError("verification changed the discovery round index")
        if (
            successor.campaign_id != before.campaign_id
            or successor.active_profile_ref != before.active_profile.profile_ref
            or successor.active_profile_digest
            != before.active_profile.profile_digest
        ):
            raise CampaignError("verification changed the discovery profile identity")
        if policy.digest != before.policy.digest:
            raise CampaignError("verification changed the Research policy")
        queue_memory = successor.scientific_memory.get("global_memory")
        if not isinstance(queue_memory, Mapping):
            queue_memory = successor.scientific_memory
        queue = ResearchTaskQueueV2.from_dict(queue_memory.get("task_queue"))
        completed = queue.get(task.task_id)
        if result.has_metric_bearing_attempt and (
            completed is None
            or completed.status is not ResearchTaskStatusV2.SATISFIED
        ):
            raise CampaignError(
                "metric-bearing verification did not satisfy its exact queued task"
            )
        active_candidate_ids = {
            item.candidate_id
            for item in queue.tasks
            if item.status
            in {ResearchTaskStatusV2.PENDING, ResearchTaskStatusV2.ACTIVE}
        }
        lineage_binding = successor.frontier.get("lineage_parent_binding")
        if isinstance(lineage_binding, Mapping):
            lineage_candidate_id = lineage_binding.get("candidate_id")
            if isinstance(lineage_candidate_id, str) and lineage_candidate_id:
                active_candidate_ids.add(lineage_candidate_id)
        return CampaignState(
            campaign_id=before.campaign_id,
            next_round_index=before.next_round_index,
            context=successor,
            active_profile=before.active_profile,
            policy=before.policy,
            search_memory_head=interpretation.search_memory_snapshot,
            # Auxiliary verification closes its own work without retaining an
            # ever-growing executable backlog.  Keep only genuinely pending
            # discovery work and the measured construction parent.  Immutable
            # outcomes remain in scientific memory and campaign evidence.
            carryover_proposals=tuple(
                item
                for item in before.carryover_proposals
                if item.candidate_id in active_candidate_ids
            ),
            carryover_open_candidates=tuple(
                item
                for item in before.carryover_open_candidates
                if item.candidate_id in active_candidate_ids
                or item.compiler_candidate_id in active_candidate_ids
            ),
            qualified_execution_by_capability=(
                before.qualified_execution_by_capability
            ),
            candidate_root_by_capability=before.candidate_root_by_capability,
            resource_profile_by_capability=(
                before.resource_profile_by_capability or {}
            ),
            deferred_innovation_backlog=before.deferred_innovation_backlog,
            incumbent_observation=before.incumbent_observation,
            frontier=successor.frontier,
            last_round_result_digest=before.last_round_result_digest,
        )

    def _load_verification(
        self,
        verification_ref: str,
    ) -> CampaignVerificationRecord:
        record = _read_pickle(self.verification_checkpoint_path(verification_ref))
        if not isinstance(record, CampaignVerificationRecord):
            raise CampaignError(
                "verification checkpoint does not contain CampaignVerificationRecord"
            )
        if record.verification_ref != verification_ref:
            raise CampaignError("verification checkpoint identity drift")
        trace_path = self.verification_trace_path(verification_ref)
        if trace_path.is_file():
            trace = _read_json(trace_path)
            if trace.get("record_digest") != record.digest:
                raise CampaignError("verification trace digest drift")
        return record

    def _seal_verification_record(
        self,
        record: CampaignVerificationRecord,
    ) -> CampaignVerificationRecord:
        checkpoint_payload = pickle.dumps(
            record, protocol=pickle.HIGHEST_PROTOCOL
        )
        checkpoint_path = self.verification_checkpoint_path(
            record.verification_ref
        )
        if not _write_once(checkpoint_path, checkpoint_payload):
            prior = self._load_verification(record.verification_ref)
            if prior.digest != record.digest:
                raise CampaignError("verification checkpoint identity drift")
            record = prior
            checkpoint_payload = pickle.dumps(
                record, protocol=pickle.HIGHEST_PROTOCOL
            )
        trace = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-verification-trace.v1",
                "record_digest": record.digest,
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint_payload
                ).hexdigest(),
                "record": record.to_dict(),
            }
        )
        trace_path = self.verification_trace_path(record.verification_ref)
        trace_payload = canonical_json_bytes(trace) + b"\n"
        if not _write_once(trace_path, trace_payload):
            existing = _read_json(trace_path)
            if canonical_value(dict(existing)) != trace:
                raise CampaignError("verification trace identity drift")
        self._state = record.state_after
        self.memory_writer._head = record.state_after.search_memory_head
        self._persist_state()
        return record

    def run_next_verification(self) -> CampaignVerificationRecord | None:
        """Execute the next bound task in a separate evidence lane."""

        queue = self._task_queue()
        if queue.select_next() is None:
            return None
        base_inputs = self._round_inputs()
        remaining = queue
        executed_semantic_seed_pairs, _pending_task, _axis_effects = (
            _search_ranking_inputs(self._state.context)
        )
        task = None
        observation_seed = ""
        while True:
            candidate = remaining.select_next()
            if candidate is None:
                return None
            if base_inputs.evidence_port is not None and (
                candidate.metadata.get("guard_source") != "EVIDENCE_GUARD"
                or not isinstance(
                    candidate.metadata.get("helix_allocation_action_id"), str
                )
            ):
                remaining = ResearchTaskQueueV2(
                    tuple(item for item in remaining.tasks if item != candidate)
                )
                continue
            if candidate.operation not in {
                ResearchTaskOperationV2.NEW_SEED,
                ResearchTaskOperationV2.REPRODUCE,
                ResearchTaskOperationV2.MATCHED_CONTROL,
                ResearchTaskOperationV2.MECHANISM_OFF,
            }:
                remaining = ResearchTaskQueueV2(
                    tuple(item for item in remaining.tasks if item != candidate)
                )
                continue
            if (
                candidate.operation
                in {
                    ResearchTaskOperationV2.MATCHED_CONTROL,
                    ResearchTaskOperationV2.MECHANISM_OFF,
                }
                and candidate.metadata.get("execution_state")
                == "AWAITING_CANDIDATE_BINDING"
            ):
                remaining = ResearchTaskQueueV2(
                    tuple(item for item in remaining.tasks if item != candidate)
                )
                continue
            candidate_seed = self._verification_seed(candidate, base_inputs)
            selected = _verification_feedback_task(
                self._state.context,
                observation_seed=candidate_seed,
                executed_semantic_seed_pairs=executed_semantic_seed_pairs,
            )
            if selected is not None:
                task = queue.get(str(selected["task_id"]))
                observation_seed = candidate_seed
                break
            remaining = ResearchTaskQueueV2(
                tuple(item for item in remaining.tasks if item != candidate)
            )
        if task is None:
            return None
        verification_ref = _verification_identity(task, observation_seed)
        checkpoint_path = self.verification_checkpoint_path(verification_ref)
        if checkpoint_path.is_file():
            record = self._load_verification(verification_ref)
            if record.task_id != task.task_id:
                raise CampaignError("verification task identity drift")
            if self._state.digest == record.state_before.digest:
                self._state = record.state_after
                self.memory_writer._head = record.state_after.search_memory_head
                self._persist_state()
            elif self._state.digest != record.state_after.digest:
                raise CampaignError("verification checkpoint state drift")
            return record

        state_before = self._state
        inputs = self._verification_inputs(task, observation_seed)
        started = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-verification-started.v1",
                "verification_ref": verification_ref,
                "task_id": task.task_id,
                "task_digest": task.digest,
                "operation": task.operation.value,
                "observation_seed": observation_seed,
                "round_index": state_before.next_round_index,
                "state_digest": state_before.digest,
                "context_digest": state_before.context.digest,
                "profile_digest": state_before.active_profile.profile_digest,
            }
        )
        started_path = self._verification_started_path(verification_ref)
        if not _write_once(started_path, canonical_json_bytes(started) + b"\n"):
            existing = _read_json(started_path)
            if canonical_value(dict(existing)) != started:
                raise CampaignError("verification start marker identity drift")

        round_memory = self._round_memory_writer()
        result = run_research_round(
            context=state_before.context,
            active_profile=state_before.active_profile,
            producer=self.producer,
            producer_bindings=inputs.producer_bindings,
            resolver_environment=inputs.resolver_environment,
            carryover_proposals=state_before.carryover_proposals,
            carryover_open_candidates=state_before.carryover_open_candidates,
            budget_snapshot=inputs.budget_snapshot,
            router=inputs.router,
            policy=state_before.policy,
            memory_writer=round_memory,
            runner=self._runner_for_verification(
                task=task,
                verification_ref=verification_ref,
                observation_seed=observation_seed,
                state_before=state_before,
            ),
            incumbent_observation=state_before.incumbent_observation,
            metric_contract_digest=inputs.metric_contract_digest,
            observation_seed=observation_seed,
            next_discriminative_test=inputs.next_discriminative_test,
            confirmation_seed=None,
            qualified_execution_by_capability=(
                inputs.qualified_execution_by_capability
            ),
            research_profile_source=inputs.research_profile_source,
            candidate_handoff_factory=inputs.candidate_handoff_factory,
            candidate_root_by_capability=inputs.candidate_root_by_capability,
            resource_profile_by_capability=(
                inputs.resource_profile_by_capability
            ),
            innovation_inputs=None,
            meta_research_inputs=None,
            attempt_scheduler=True,
            max_attempts_per_round=1,
            prebinding_token_ceiling_retry=False,
            portfolio_candidates=None,
            evidence_port=inputs.evidence_port,
            observation_seed_schedule=inputs.observation_seed_schedule,
            evaluator=inputs.evaluator,
            split=inputs.split,
            frozen_profile_ref=inputs.frozen_profile_ref,
            round_role="VERIFICATION",
            search_space_adapter=(
                inputs.search_space_adapter or self.search_space_adapter
            ),
        )
        physical_path = self._verification_physical_path(verification_ref)
        if not physical_path.is_file():
            raise CampaignError(
                "verification returned without a durable physical observation"
            )
        physical = _read_json(physical_path)
        result = self._attach_verification_observation(result, physical)
        if result.provider_traces:
            raise CampaignError("verification unexpectedly called the Provider")
        if result.innovation is not None or result.meta_research is not None:
            raise CampaignError("verification entered a discovery or meta lane")
        if len(result.attempts) != 1:
            raise CampaignError("verification must execute exactly one attempt")
        state_after = self._advance_verification_state(
            state_before, task, result
        )
        status = (
            "TYPED_EPISODE"
            if result.has_metric_bearing_attempt
            and result.metric_bearing_attempt_index == 0
            and result.interpretation is not None
            and getattr(result.interpretation, "episode", None) is not None
            else "INCOMPLETE"
        )
        return self._seal_verification_record(
            CampaignVerificationRecord(
                verification_ref=verification_ref,
                task_id=task.task_id,
                operation=task.operation.value,
                observation_seed=observation_seed,
                status=status,
                state_before=state_before,
                result=result,
                state_after=state_after,
            )
        )

    def run_pending_verifications(
        self,
        *,
        max_tasks: int,
    ) -> tuple[CampaignVerificationRecord, ...]:
        """Drain exact queued evidence work without consuming discovery slots."""

        if (
            isinstance(max_tasks, bool)
            or not isinstance(max_tasks, int)
            or max_tasks < 1
        ):
            raise CampaignError("max_tasks must be a positive integer")
        records: list[CampaignVerificationRecord] = []
        for _ in range(max_tasks):
            record = self.run_next_verification()
            if record is None:
                return tuple(records)
            records.append(record)
            if record.status != "TYPED_EPISODE":
                # Auxiliary evidence failure is diagnostic; it cannot stop the
                # discovery campaign or consume another discovery round.
                return tuple(records)
        return tuple(records)

    def _round_inputs(self) -> CampaignRoundInputs:
        supplied = (
            self.round_inputs(self._state)
            if callable(self.round_inputs)
            else self.round_inputs
        )
        if not isinstance(supplied, CampaignRoundInputs):
            raise CampaignError("round_inputs must return CampaignRoundInputs")
        qualified = dict(self._state.qualified_execution_by_capability)
        qualified.update(dict(supplied.qualified_execution_by_capability))
        roots = dict(self._state.candidate_root_by_capability)
        roots.update(dict(supplied.candidate_root_by_capability))
        resource_profiles = dict(
            getattr(self._state, "resource_profile_by_capability", {}) or {}
        )
        resource_profiles.update(dict(supplied.resource_profile_by_capability))
        innovation_inputs = supplied.innovation_inputs
        if innovation_inputs is not None and self.implementer is not None:
            innovation_inputs = replace(innovation_inputs, implementer=self.implementer)
        return replace(
            supplied,
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resource_profiles,
            innovation_inputs=innovation_inputs,
        )

    def _runner_for_round(
        self,
        round_index: int,
        opportunity_ref: str,
        manifest: dict[str, Any] | None = None,
        observation_seed: str | int | None = None,
        engineering_source_identity: Mapping[str, Any] | None = None,
    ) -> ExperimentRunner:
        progress = manifest if manifest is not None else {"attempts": []}

        def guarded_runner(
            recipe: Mapping[str, Any], binding: Any
        ) -> Mapping[str, Any]:
            candidate_id = getattr(getattr(binding, "proposal", None), "candidate_id", None)
            if not isinstance(candidate_id, str) or not candidate_id:
                raise CampaignError("runner binding lacks a candidate identity")
            existing = next(
                (
                    item
                    for item in progress.get("attempts", ())
                    if isinstance(item, Mapping) and item.get("candidate_id") == candidate_id
                ),
                None,
            )
            if existing is None:
                existing = next(
                    (
                        item
                        for item in progress.get("attempts", ())
                        if isinstance(item, Mapping)
                        and item.get("legacy_single_attempt") is True
                    ),
                    None,
                )
            if existing is not None:
                observation_path = Path(str(existing["physical_observation_path"]))
                payload = _read_json(observation_path)
                if payload.get("opportunity_ref") != opportunity_ref:
                    raise CampaignError("physical observation opportunity identity drift")
                payload_candidate_id = payload.get("candidate_id")
                if payload_candidate_id not in {
                    None,
                    candidate_id,
                    "__LEGACY_SINGLE_ATTEMPT__",
                }:
                    raise CampaignError("physical observation candidate identity drift")
                if payload.get("attempt_index") is not None and int(
                    payload["attempt_index"]
                ) != int(existing["attempt_index"]):
                    raise CampaignError("physical observation attempt identity drift")
                if payload.get("execution_recipe_digest") != sha256_digest(recipe):
                    raise CampaignError("physical observation recipe identity drift")
                value = payload.get("candidate_run")
                if not isinstance(value, Mapping):
                    raise CampaignError("physical observation lacks candidate_run")
                return value

            attempt_index = max(
                (
                    int(item["attempt_index"])
                    for item in progress.get("attempts", ())
                    if isinstance(item, Mapping) and "attempt_index" in item
                ),
                default=-1,
            ) + 1
            context_value = canonical_value(
                {
                    "schema": PHYSICAL_CONTEXT_SCHEMA,
                    "campaign_id": self._state.campaign_id,
                    "round_index": round_index,
                    "opportunity_ref": opportunity_ref,
                    "attempt_index": attempt_index,
                    "candidate_id": candidate_id,
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_semantic_digest": getattr(
                        binding, "mechanism_semantics_digest", None
                    ),
                    "research_context_digest": self._state.context.digest,
                    "profile_digest": self._state.active_profile.profile_digest,
                    "seed": observation_seed,
                }
            )
            context_digest = sha256_digest(context_value)
            physical_context = _immutable_canonical_mapping(context_value)
            start_marker = canonical_value(
                {
                    "schema": "recclaw.research-line.physical-started.v1",
                    "round_index": round_index,
                    "opportunity_ref": opportunity_ref,
                    "attempt_index": attempt_index,
                    "candidate_id": candidate_id,
                    "binding_digest": getattr(binding, "digest", None),
                    "execution_recipe_digest": sha256_digest(recipe),
                    "physical_context_digest": context_digest,
                    **(
                        {
                            "engineering_source_identity": dict(
                                engineering_source_identity
                            )
                        }
                        if engineering_source_identity is not None
                        else {}
                    ),
                }
            )
            start_path = self._physical_started_path(round_index, attempt_index)
            if start_path.is_file():
                if _read_json(start_path) != start_marker:
                    raise CampaignError("physical start marker identity drift")
            else:
                _write_once(start_path, canonical_json_bytes(start_marker) + b"\n")
            context_runner = getattr(
                self.runner, "run_with_physical_context", None
            )
            if context_runner is not None and not callable(context_runner):
                raise CampaignError(
                    "runner run_with_physical_context is not callable"
                )
            if context_runner is None:
                value = self.runner(recipe, binding)
            else:
                value = context_runner(
                    recipe,
                    binding,
                    physical_context,
                )
            if not isinstance(value, Mapping):
                raise CampaignError("injected runner must return a mapping")
            candidate_run = canonical_value(dict(value))
            physical_identity = _physical_identity_fields(
                candidate_run,
                context_digest=context_digest,
                context_applied=context_runner is not None,
            )
            observation_core = canonical_value(
                {
                    "schema": "recclaw.research-line.physical-observation.v2",
                    "round_index": round_index,
                    "attempt_index": attempt_index,
                    "opportunity_ref": opportunity_ref,
                    "candidate_id": candidate_id,
                    "execution_recipe_digest": sha256_digest(recipe),
                    "binding_digest": getattr(binding, "digest", None),
                    "candidate_run": candidate_run,
                    **physical_identity,
                }
            )
            observation_digest = sha256_digest(observation_core)
            observation_ref = content_id(
                "recclaw-research-line-physical-observation-v2",
                observation_core,
            )
            payload = canonical_value(
                {
                    **dict(observation_core),
                    "observation_ref": observation_ref,
                    "observation_digest": observation_digest,
                }
            )
            observation_path = self._attempt_physical_path(round_index, attempt_index)
            _write_once(observation_path, canonical_json_bytes(payload) + b"\n")
            # The old single-attempt reader remains a compatibility projection
            # of the first physical attempt.
            if attempt_index == 0:
                _write_once(
                    self._physical_path(round_index),
                    canonical_json_bytes(payload) + b"\n",
                )
            row = self._physical_manifest_row(payload, observation_path)
            progress.setdefault("attempts", []).append(row)
            self._write_attempt_manifest(
                round_index=round_index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=(
                    int(progress["attempt_budget"])
                    if progress.get("attempt_budget") is not None
                    else None
                ),
                attempts=progress["attempts"],
                status="IN_PROGRESS",
            )
            return candidate_run

        return guarded_runner

    def _advance_state(
        self,
        before: CampaignState,
        result: ResearchRoundResult,
        *,
        consume_verification_slot: bool = False,
    ) -> CampaignState:
        if result.attempt_scheduler_enabled:
            metric_indices = tuple(
                index
                for index, attempt in enumerate(result.attempts)
                if attempt.metric_bearing
            )
            if (
                len(metric_indices) != 1
                or result.metric_bearing_attempt_index != metric_indices[0]
            ):
                raise CampaignError(
                    "formal round cannot advance without exactly one "
                    "metric-bearing attempt"
                )
        interpretation = result.interpretation
        if interpretation is None:
            raise CampaignError("Research round did not produce typed feedback")
        successor = interpretation.successor_context
        if consume_verification_slot:
            successor = replace(successor, round_index=before.next_round_index + 1)
        policy = interpretation.policy_successor
        active_profile = before.active_profile
        qualified = dict(before.qualified_execution_by_capability)
        roots = dict(before.candidate_root_by_capability)
        resource_profiles = dict(
            getattr(before, "resource_profile_by_capability", {}) or {}
        )

        proposal_items: list[CandidateProposalV4] = list(before.carryover_proposals)
        for outcome in (*result.producer_outcomes, *result.carryover_outcomes):
            if outcome.source_proposal is not None:
                proposal_items.append(outcome.source_proposal)
        open_items: list[QualifiedSearchCandidateProtocolV1] = list(
            before.carryover_open_candidates
        )
        attempted_candidate_ids = {
            attempt.candidate_id for attempt in result.attempts
        }
        legacy_selected_binding = (
            result.search_acquisition.selected_binding
            if not result.attempts
            and result.candidate_run is not None
            and result.search_acquisition is not None
            else None
        )
        legacy_candidate_id = (
            legacy_selected_binding.proposal.candidate_id
            if legacy_selected_binding is not None
            else None
        )
        if legacy_candidate_id is not None:
            attempted_candidate_ids.add(legacy_candidate_id)
        qualified_candidate_ids = {
            binding.proposal.candidate_id
            for binding in (
                result.prepared.search_bindings
                if isinstance(result.prepared, PreparedResearchRoundV1)
                else ()
            )
            if binding.capability_ref in qualified
        }
        queue_memory = successor.scientific_memory.get("global_memory", {})
        queue = ResearchTaskQueueV2.from_dict(
            queue_memory.get("task_queue")
            if isinstance(queue_memory, Mapping)
            else None
        )
        pending_confirmation_ids = {
            task.candidate_id
            for task in queue.tasks
            if task.status in {
                ResearchTaskStatusV2.PENDING,
                ResearchTaskStatusV2.ACTIVE,
            }
        }
        metric_candidate_id = (
            result.attempts[result.metric_bearing_attempt_index].candidate_id
            if result.metric_bearing_attempt_index is not None
            else legacy_candidate_id
            if result.has_metric_bearing_attempt
            else None
        )
        retained_attempt_ids = (
            {metric_candidate_id}
            if metric_candidate_id in pending_confirmation_ids
            else set()
        )
        retained_attempt_ids.update(qualified_candidate_ids)
        retired_attempt_ids = attempted_candidate_ids - retained_attempt_ids
        if attempted_candidate_ids:
            proposal_items = [
                proposal
                for proposal in proposal_items
                if proposal.candidate_id not in retired_attempt_ids
            ]
            open_items = [
                candidate
                for candidate in open_items
                if candidate.candidate_id not in retired_attempt_ids
            ]

        innovation = result.innovation
        if innovation is not None and innovation.activation_ready:
            active_profile, successor, candidate, execution = activate_staged_innovation(
                result
            )
            if innovation.registry is None or len(innovation.registry.capabilities) != 1:
                raise CampaignError(
                    "one metric round must activate exactly one qualified capability"
                )
            capability = innovation.capability
            if capability is None:
                raise CampaignError(
                    "activated innovation lacks its qualified capability"
                )
            if len(active_profile.entries) != len(before.active_profile.entries) + 1:
                raise CampaignError(
                    "activated profile must add exactly one qualified capability"
                )
            predecessor_by_ref = {
                entry.capability_ref: entry
                for entry in before.active_profile.entries
            }
            active_by_ref = {
                entry.capability_ref: entry
                for entry in active_profile.entries
            }
            if any(
                active_by_ref.get(capability_ref) != predecessor_entry
                for capability_ref, predecessor_entry in predecessor_by_ref.items()
            ):
                raise CampaignError(
                    "activated profile must preserve every predecessor entry"
                )
            added_refs = set(active_by_ref) - set(predecessor_by_ref)
            if added_refs != {capability.capability_id}:
                raise CampaignError(
                    "activated profile must add only the qualified capability"
                )
            added_entry = active_by_ref[capability.capability_id]
            if (
                added_entry.capability_ref != capability.capability_id
                or added_entry.semantic_identity_digest
                != candidate.semantic_identity_digest
            ):
                raise CampaignError(
                    "activated profile added entry differs from the qualified innovation"
                )
            capability_ref = capability.capability_id
            qualified[capability_ref] = canonical_value(dict(execution))
            root = innovation.candidate_root or str(
                execution.get("candidate_root_ref", "")
            )
            if root:
                roots[capability_ref] = root
            if innovation.resource_profile is not None:
                resource_profiles[capability_ref] = canonical_value(
                    dict(innovation.resource_profile)
                )
            if isinstance(candidate, CandidateProposalV4):
                proposal_items.append(candidate)
            else:
                open_items.append(candidate)

        meta = result.meta_research
        if meta is not None and meta.activated_policy is not None:
            if meta.activation_receipt is None:
                raise CampaignError("promoted Meta policy lacks an activation receipt")
            if (
                active_profile.campaign_id == before.active_profile.campaign_id
                and active_profile.profile_digest == before.active_profile.profile_digest
            ):
                active_profile = _fresh_profile_with_same_entries(
                    before.active_profile,
                    fresh_campaign_id=meta.activation_receipt.campaign_id,
                )
            if active_profile.campaign_id != meta.activation_receipt.campaign_id:
                raise CampaignError("Meta activation is not bound to the next-round profile")
            successor, policy = activate_promoted_meta_strategy(
                result,
                next_profile=active_profile,
            )

        if result.attempts:
            # The interpreter's successor is outcome-focused and may rebuild
            # scientific memory.  Same-round failed generations live on the
            # actual predecessor Context and must not disappear when the first
            # successful metric finally advances the round.
            prior_attempts = before.context.scientific_memory.get(
                "round_attempts", ()
            )
            prior_attempts = (
                tuple(prior_attempts)
                if isinstance(prior_attempts, (tuple, list))
                else ()
            )
            metric_effect_observation: Mapping[str, Any] | None = None
            search_utility_event = getattr(
                result.interpretation, "search_utility_event", None
            )
            if search_utility_event is not None and hasattr(
                search_utility_event, "to_dict"
            ):
                metric_effect_observation = search_utility_event.to_dict()
                for row in reversed(successor.frontier.get("effect_trajectory", ())):
                    if row.get("round_index") == before.next_round_index and row.get("candidate_id") == search_utility_event.candidate_id:
                        metric_effect_observation = {**metric_effect_observation, **dict(row)}
                        break
            compact_attempts = tuple(
                _compact_round_attempt_summary(
                    attempt,
                    round_index=before.next_round_index,
                    context=before.context,
                    effect_observation=(
                        metric_effect_observation
                        if attempt.attempt_index
                        == result.metric_bearing_attempt_index
                        else None
                    ),
                )
                for attempt in result.attempts
            )
            combined_attempts = (*prior_attempts, *compact_attempts)[
                -_ROUND_ATTEMPT_HISTORY_LIMIT:
            ]
            innovation_attempts = (
                tuple(innovation.attempts)
                if innovation is not None
                else ()
            )
            identity_history = _compact_round_attempt_identity_history(
                before.context.scientific_memory,
                (*compact_attempts, *innovation_attempts),
            )
            mechanism_experiences = _compact_mechanism_experiences(
                before.context.scientific_memory,
                (*compact_attempts, *(
                    {**dict(item), "round_index": before.next_round_index, "attempt_index": index}
                    for index, item in enumerate(innovation_attempts) if item.get("failure")
                )),
            )
            producer_opportunity_state = _compact_producer_opportunity_state(
                before.context.scientific_memory,
                (
                    innovation_attempts
                    if innovation_attempts
                    else compact_attempts
                ),
            )
            metric_observation_index = _compact_metric_observation_index(
                before.context.scientific_memory,
                compact_attempts,
            )
            scientific_memory = {
                **successor.scientific_memory,
                "round_attempts": combined_attempts,
                "mechanism_experiences": mechanism_experiences,
                **identity_history,
                "producer_opportunity_state": producer_opportunity_state,
                "metric_observation_index": metric_observation_index,
            }
            existing_global = successor.scientific_memory.get("global_memory")
            if isinstance(existing_global, Mapping):
                scientific_memory["global_memory"] = {
                    **existing_global,
                    "round_attempts": combined_attempts,
                    "mechanism_experiences": mechanism_experiences,
                    **identity_history,
                    "producer_opportunity_state": producer_opportunity_state,
                    "metric_observation_index": metric_observation_index,
                }
            successor = replace(
                successor,
                scientific_memory=scientific_memory,
            )

        # Fixed-profile proposals are materialized afresh for each ordered
        # seed.  Keeping unattempted materializations forever caused the pool
        # to deplete and then let later rounds advance without an experiment.
        proposal_items = [
            proposal
            for proposal in proposal_items
            if not proposal.producer_id.startswith("producer:bootstrap:")
            or proposal.candidate_id in pending_confirmation_ids
        ]

        portfolio_history = _portfolio_history_projection(
            before,
            successor,
            result,
        )
        if any(portfolio_history.values()) or any(
            alias in successor.scientific_memory.get("global_memory", {})
            if isinstance(successor.scientific_memory.get("global_memory"), Mapping)
            else False
            for alias in portfolio_history
        ):
            existing_global_memory = successor.scientific_memory.get("global_memory", {})
            global_memory = (
                dict(existing_global_memory)
                if isinstance(existing_global_memory, Mapping)
                else {}
            )
            for alias, rows in portfolio_history.items():
                existing_rows = _history_rows(global_memory.get(alias))
                if rows or existing_rows:
                    global_memory[alias] = canonical_value(
                        (*existing_rows, *rows)[-_PORTFOLIO_HISTORY_LIMIT:]
                    )
            successor = replace(
                successor,
                scientific_memory={
                    **successor.scientific_memory,
                    "global_memory": global_memory,
                },
            )

        successor = replace(
            successor,
            campaign_id=active_profile.campaign_id,
            active_profile_ref=active_profile.profile_ref,
            active_profile_digest=active_profile.profile_digest,
            policy=policy.to_dict(),
            scientific_memory={
                **successor.scientific_memory,
                "discovery_generation": 0,
            },
        )
        if successor.round_index != before.next_round_index + 1:
            raise CampaignError("Research successor did not advance exactly one round")
        deferred_innovation_backlog = _next_deferred_innovation_backlog(
            before,
            result,
        )
        return CampaignState(
            campaign_id=active_profile.campaign_id,
            next_round_index=successor.round_index,
            context=successor,
            active_profile=active_profile,
            policy=policy,
            search_memory_head=interpretation.search_memory_snapshot,
            carryover_proposals=tuple(
                item
                for item in _unique_by_id(proposal_items)
                if isinstance(item, CandidateProposalV4)
            ),
            carryover_open_candidates=tuple(
                item
                for item in _unique_by_id(open_items)
                if is_qualified_open_spec_candidate(item)
            ),
            qualified_execution_by_capability=qualified,
            candidate_root_by_capability=roots,
            resource_profile_by_capability=resource_profiles,
            deferred_innovation_backlog=deferred_innovation_backlog,
            incumbent_observation=_incumbent_after_metric_round(
                before.context,
                before.incumbent_observation,
                successor.frontier,
            ),
            frontier=successor.frontier,
            last_round_result_digest=sha256_digest(result.to_dict()),
        )

    @staticmethod
    def _attach_observation_identities(
        result: ResearchRoundResult,
        manifest: Mapping[str, Any],
    ) -> ResearchRoundResult:
        rows = {
            str(item.get("candidate_id")): item
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
        }
        legacy_rows = tuple(
            item
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
            and item.get("legacy_single_attempt") is True
        )

        def row_for(attempt: RoundAttemptV1) -> Mapping[str, Any]:
            row = rows.get(attempt.candidate_id)
            if row is not None:
                return row
            return next(
                (
                    item
                    for item in legacy_rows
                    if int(item.get("attempt_index", -1)) == attempt.attempt_index
                ),
                {},
            )

        attempts = tuple(
            replace(
                attempt,
                observation_ref=(
                    str(row_for(attempt)["observation_ref"])
                    if row_for(attempt).get("observation_ref") is not None
                    else attempt.observation_ref
                ),
                observation_digest=(
                    str(row_for(attempt)["observation_digest"])
                    if row_for(attempt).get("observation_digest") is not None
                    else attempt.observation_digest
                ),
            )
            for attempt in result.attempts
        )
        return replace(result, attempts=attempts)

    def _persist_attempt_result_manifest(
        self,
        *,
        round_index: int,
        opportunity_ref: str,
        state_before: CampaignState,
        manifest: Mapping[str, Any],
        result: ResearchRoundResult,
        status: str,
        prebinding_summaries: tuple[Mapping[str, Any], ...] | None = None,
    ) -> Mapping[str, Any]:
        rows: list[dict[str, Any]] = [
            dict(item)
            for item in manifest.get("attempts", ())
            if isinstance(item, Mapping)
        ]
        by_candidate = {
            str(item.get("candidate_id")): item
            for item in rows
            if item.get("candidate_id") is not None
        }
        for attempt in result.attempts:
            row = by_candidate.get(attempt.candidate_id)
            if row is None:
                row = next(
                    (
                        item
                        for item in rows
                        if item.get("legacy_single_attempt") is True
                        and int(item.get("attempt_index", -1)) == attempt.attempt_index
                    ),
                    None,
                )
                if row is not None:
                    old_candidate_id = row.get("candidate_id")
                    row["candidate_id"] = attempt.candidate_id
                    row["legacy_single_attempt"] = False
                    if old_candidate_id is not None:
                        by_candidate.pop(str(old_candidate_id), None)
                    by_candidate[attempt.candidate_id] = row
            if row is None:
                row = {
                    "attempt_index": attempt.attempt_index,
                    "candidate_id": attempt.candidate_id,
                    "candidate_run": attempt.candidate_run,
                    "execution_recipe_digest": sha256_digest(attempt.execution_recipe),
                }
                rows.append(row)
                by_candidate[attempt.candidate_id] = row
            row.update(
                {
                    "attempt_digest": attempt.digest,
                    "engineering_disposition": attempt.engineering_disposition,
                    "failure_scope": attempt.failure_scope,
                    "failure_detail": attempt.failure_detail,
                    "diagnostic_feedback": attempt.diagnostic_feedback,
                    "route_trace_digest": (
                        attempt.acquisition.route_trace.digest
                        if attempt.acquisition is not None
                        else None
                    ),
                }
            )
            if attempt.evidence_pre_adjudication is not None:
                row["evidence_pre_adjudication"] = attempt.evidence_pre_adjudication
            if attempt.evidence_post_adjudication is not None:
                row["evidence_post_adjudication"] = attempt.evidence_post_adjudication
        candidate_prebinding_summaries = (
            prebinding_summaries
            if prebinding_summaries is not None
            else self._prebinding_candidate_failure_summaries(result)
            if not result.attempts
            else ()
        )

        def recoverable_prebinding(summary: Mapping[str, Any]) -> bool:
            failure = summary.get("failure")
            if not isinstance(failure, Mapping):
                return False
            scope = failure.get("failure_scope")
            scope = getattr(scope, "value", scope)
            return str(scope or "").upper() in {
                "WORKER_TRANSIENT",
                "SHARED_INFRASTRUCTURE",
                "RECOVERY",
            }

        durable_prebinding_summaries = tuple(
            summary
            for summary in candidate_prebinding_summaries
            if not recoverable_prebinding(summary)
        )
        for index, summary in enumerate(durable_prebinding_summaries):
            candidate_id = str(summary["candidate_id"])
            row = by_candidate.get(candidate_id)
            if row is not None and row.get("candidate_run") is not None:
                continue
            if row is None:
                used_indices = {
                    int(item["attempt_index"])
                    for item in rows
                    if isinstance(item.get("attempt_index"), int)
                    and not isinstance(item.get("attempt_index"), bool)
                }
                attempt_index = index
                while attempt_index in used_indices:
                    attempt_index += 1
                row = {
                    "attempt_index": attempt_index,
                    "candidate_id": candidate_id,
                }
                rows.append(row)
                by_candidate[candidate_id] = row
            row.update(
                {
                    "attempt_digest": sha256_digest(summary),
                    "engineering_disposition": "ENGINEERING_FAILURE",
                    "failure_detail": summary.get("failure"),
                    "diagnostic_feedback": summary,
                    "spec_digest": summary.get("spec_digest"),
                    "candidate_semantic_digest": summary.get(
                        "candidate_semantic_digest"
                    ),
                    "effective_experiment_digest": summary.get(
                        "effective_experiment_digest"
                    ),
                    "effective_family_digest": summary.get(
                        "effective_family_digest"
                    ),
                    "revision_count": summary.get("revision_count"),
                }
            )
        return self._write_attempt_manifest(
            round_index=round_index,
            opportunity_ref=opportunity_ref,
            state_digest=state_before.digest,
            context_digest=state_before.context.digest,
            profile_digest=state_before.active_profile.profile_digest,
            attempt_budget=(
                int(manifest["attempt_budget"])
                if manifest.get("attempt_budget") is not None
                else None
            ),
            attempts=tuple(sorted(rows, key=lambda item: int(item["attempt_index"]))),
            status=status,
            metric_bearing_attempt_index=result.metric_bearing_attempt_index,
            incomplete_reason=result.incomplete_reason,
            evidence_pre_trace=result.evidence_pre_trace,
            evidence_post_trace=result.evidence_post_trace,
        )

    @staticmethod
    def _hold_state_after_incomplete_attempts(
        before: CampaignState,
        result: ResearchRoundResult,
        *,
        prebinding_summaries: tuple[Mapping[str, Any], ...] | None = None,
        generation_negative: Mapping[str, Any] | None = None,
    ) -> CampaignState:
        """Start a fresh same-round discovery generation after local failures."""

        if (
            _innovation_resource_failure_scope(result.innovation)
            in {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE", "RECOVERY"}
        ):
            # A resource probe infrastructure stop is not a candidate outcome
            # or a producer opportunity.  Keep the exact generation recoverable.
            return before

        if (
            generation_negative is None
            and not result.attempts
            and result.innovation is None
            and not prebinding_summaries
            and result.incomplete_reason
            == "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
        ):
            return before

        if (
            result.incomplete_reason == "ROUND_EXISTING_OBSERVATION_REBOUND"
            and result.interpretation is not None
        ):
            successor_memory = dict(
                result.interpretation.successor_context.scientific_memory
            )
            next_generation = _discovery_generation(before.context) + 1
            successor_memory["discovery_generation"] = next_generation
            # Reuse the existing one-generation prepared-retry allowance so
            # the newly satisfied task can yield immediately to fresh
            # discovery even when its task-bound proposal generation was the
            # configured final generation.  The marker is generation-scoped.
            successor_memory[
                "stale_prepared_feedback_retry_generation"
            ] = next_generation
            return replace(
                before,
                context=replace(
                    before.context,
                    scientific_memory=canonical_value(successor_memory),
                ),
                search_memory_head=result.interpretation.search_memory_snapshot,
                last_round_result_digest=sha256_digest(result.to_dict()),
            )

        retryable_scopes = {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
        continued_generation_duplicate = (
            _discovery_generation(before.context) > 0
            and not result.attempts
            and result.incomplete_reason
            in {
                "ROUND_ALL_EFFECTIVE_EXPERIMENTS_DUPLICATE",
                "ROUND_ALL_EFFECTIVE_FAMILIES_DUPLICATE",
            }
        )
        terminal_nonretryable_attempt = bool(
            generation_negative is None
            and result.attempts
            and any(
                attempt.failure_scope not in retryable_scopes
                for attempt in result.attempts
            )
        )
        if terminal_nonretryable_attempt and not prebinding_summaries:
            return before
        attempted_ids = {attempt.candidate_id for attempt in result.attempts}
        prepared_ids = {
            binding.proposal.candidate_id
            for binding in (
                result.prepared.search_bindings
                if isinstance(result.prepared, PreparedResearchRoundV1)
                else ()
            )
        }
        if (
            prepared_ids - attempted_ids
            and not continued_generation_duplicate
            and generation_negative is None
            and not prebinding_summaries
        ):
            return before

        innovation_attempts = tuple(
            item
            for item in (
                result.innovation.attempts
                if result.innovation is not None
                and not result.innovation.activation_ready
                else ()
            )
            if isinstance(item, Mapping)
        )
        feedback_proposal_retry = (
            result.incomplete_reason
            == "ROUND_FEEDBACK_PROPOSAL_RETRY_REQUIRED"
        )
        feedback_rejections: tuple[Mapping[str, Any], ...] = ()
        if feedback_proposal_retry and result.interpretation is not None:
            feedback_projection = result.interpretation.feedback_projection
            diagnostic_detail = (
                feedback_projection.get("diagnostic_detail")
                if isinstance(feedback_projection, Mapping)
                else None
            )
            raw_rejections = (
                diagnostic_detail.get("feedback_confirmation_rejections", ())
                if isinstance(diagnostic_detail, Mapping)
                else ()
            )
            if isinstance(raw_rejections, (tuple, list)):
                feedback_rejections = tuple(
                    item for item in raw_rejections if isinstance(item, Mapping)
                )
        if (
            not result.attempts
            and not innovation_attempts
            and not prebinding_summaries
            and generation_negative is None
            and not feedback_proposal_retry
            and not continued_generation_duplicate
        ):
            return before

        compact_attempts = tuple(
            _compact_round_attempt_summary(
                attempt,
                round_index=before.next_round_index,
                context=before.context,
            )
            for attempt in result.attempts
            if attempt.failure_scope in retryable_scopes
        )
        feedback_rejection_by_spec = {
            str(item["spec_digest"]): item
            for item in feedback_rejections
            if isinstance(item.get("spec_digest"), str)
        }

        def innovation_failure(item: Mapping[str, Any]) -> Any:
            failure = item.get("failure")
            if failure is not None:
                return failure
            rejection = feedback_rejection_by_spec.get(str(item.get("spec_digest")))
            if rejection is None:
                return None
            return canonical_value(
                {
                    "failure_class": "SELECTION",
                    "stage": "ADAPTER_CONFIRMATION",
                    "reason_code": rejection.get("reason"),
                    "message": (
                        "candidate did not resolve the retained feedback task "
                        "to an exact executable binding"
                    ),
                }
            )

        compact_innovation = (
            tuple(
                canonical_value(
                    {
                        **dict(item),
                        "round_index": before.next_round_index,
                        "discovery_generation": _discovery_generation(
                            before.context
                        ),
                        "attempt_index": index,
                    }
                )
                for index, item in enumerate(prebinding_summaries)
            )
            if prebinding_summaries is not None
            else tuple(
                canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line."
                            "discovery-prebinding-attempt-summary.v1"
                        ),
                        "round_index": before.next_round_index,
                        "discovery_generation": _discovery_generation(
                            before.context
                        ),
                        "attempt_index": index,
                        "candidate_id": item.get("candidate_id"),
                        "candidate_semantic_digest": item.get(
                            "mechanism_semantics_digest"
                        ),
                        "effective_experiment_digest": item.get(
                            "effective_experiment_digest"
                        ),
                        "effective_family_digest": item.get(
                            "effective_family_digest"
                        ),
                        "primitive_ids": item.get("primitive_ids"),
                        "spec_digest": item.get("spec_digest"),
                        "core_mechanism_contrast": item.get("core_mechanism_contrast"),
                        "next_discriminative_task": item.get("next_discriminative_task"),
                        "producer_role": item.get("producer_role"),
                        "repair_attempt": item.get("repair_attempt"),
                        "candidate_root": item.get("candidate_root"),
                        "failure": innovation_failure(item),
                        "quality_admission": item.get("quality_admission"),
                        "feedback_rejection": feedback_rejection_by_spec.get(
                            str(item.get("spec_digest"))
                        ),
                    }
                )
                for index, item in enumerate(innovation_attempts)
            )
        )
        innovation_spec_digests = {
            str(item["spec_digest"])
            for item in innovation_attempts
            if isinstance(item.get("spec_digest"), str)
        }
        compact_feedback_rejections = tuple(
            canonical_value(
                {
                    "schema": (
                        "recclaw.research-line.discovery-prebinding-"
                        "feedback-rejection-summary.v1"
                    ),
                    "round_index": before.next_round_index,
                    "discovery_generation": _discovery_generation(before.context),
                    "candidate_id": item.get("candidate_id"),
                    "candidate_semantic_digest": item.get(
                        "candidate_semantic_digest"
                    ),
                    "effective_experiment_digest": item.get(
                        "effective_experiment_digest"
                    ),
                    "effective_family_digest": item.get(
                        "effective_family_digest"
                    ),
                    "primitive_ids": item.get("primitive_ids"),
                    "spec_digest": item.get("spec_digest"),
                    "producer_role": item.get("producer_role"),
                    "failure": {
                        "failure_class": "SELECTION",
                        "stage": "ADAPTER_CONFIRMATION",
                        "reason_code": item.get("reason"),
                        "message": (
                            "candidate did not resolve the retained feedback task "
                            "to an exact executable binding"
                        ),
                    },
                    "feedback_rejection": item,
                }
            )
            for item in feedback_rejections
            if str(item.get("spec_digest")) not in innovation_spec_digests
        )
        prior_attempts = before.context.scientific_memory.get("round_attempts", ())
        prior_attempts = (
            tuple(prior_attempts)
            if isinstance(prior_attempts, (tuple, list))
            else ()
        )
        combined_attempts = (
            *prior_attempts,
            *compact_attempts,
            *compact_innovation,
            *compact_feedback_rejections,
        )[-_ROUND_ATTEMPT_HISTORY_LIMIT:]
        identity_history = _compact_round_attempt_identity_history(
            before.context.scientific_memory,
            (*compact_attempts, *compact_innovation, *compact_feedback_rejections),
        )
        mechanism_experiences = _compact_mechanism_experiences(
            before.context.scientific_memory,
            (*compact_attempts, *compact_innovation, *compact_feedback_rejections),
        )
        producer_opportunity_state = _compact_producer_opportunity_state(
            before.context.scientific_memory,
            (
                compact_innovation
                if compact_innovation
                else compact_attempts
            ),
        )
        metric_observation_index = _compact_metric_observation_index(
            before.context.scientific_memory,
            compact_attempts,
        )
        existing_global = before.context.scientific_memory.get("global_memory")
        global_memory = (
            dict(existing_global) if isinstance(existing_global, Mapping) else None
        )
        if global_memory is not None:
            global_memory["round_attempts"] = combined_attempts
            global_memory["mechanism_experiences"] = mechanism_experiences
            global_memory.update(identity_history)
            global_memory["producer_opportunity_state"] = (
                producer_opportunity_state
            )
            global_memory["metric_observation_index"] = metric_observation_index
        scientific_memory = {
            **before.context.scientific_memory,
            "round_attempts": combined_attempts,
            "mechanism_experiences": mechanism_experiences,
            **identity_history,
            "producer_opportunity_state": producer_opportunity_state,
            "metric_observation_index": metric_observation_index,
            "discovery_generation": _discovery_generation(before.context)
            + (0 if terminal_nonretryable_attempt else 1),
        }
        if generation_negative is not None:
            prior_generation_negatives = scientific_memory.get(
                "generation_negatives", ()
            )
            prior_generation_negatives = (
                tuple(prior_generation_negatives)
                if isinstance(prior_generation_negatives, (tuple, list))
                else ()
            )
            scientific_memory["generation_negatives"] = (
                *prior_generation_negatives,
                canonical_value(
                    {
                        **dict(generation_negative),
                        "round_index": before.next_round_index,
                        "discovery_generation": _discovery_generation(before.context),
                        "counts_as_candidate_attempt": False,
                        "counts_as_metric_worker": False,
                    }
                ),
            )[-_ROUND_ATTEMPT_HISTORY_LIMIT:]
        if any(
            attempt.engineering_disposition
            == "IMPLEMENTATION_FIDELITY_REJECTED"
            for attempt in result.attempts
        ):
            scientific_memory["stale_prepared_feedback_retry_generation"] = (
                _discovery_generation(before.context) + 1
            )
        if feedback_proposal_retry and result.prepared is None:
            scientific_memory["stale_prepared_feedback_retry_generation"] = (
                _discovery_generation(before.context) + 1
            )
        if global_memory is not None:
            scientific_memory["global_memory"] = global_memory
        next_context = replace(
            before.context,
            scientific_memory=scientific_memory,
        )
        retryable_ids = {
            attempt.candidate_id
            for attempt in result.attempts
            if attempt.failure_scope in retryable_scopes
        }
        retryable_ids.update(
            str(item["candidate_id"])
            for item in innovation_attempts
            if isinstance(item.get("candidate_id"), str)
            and item.get("candidate_id")
        )
        retryable_ids.update(
            str(item["candidate_id"])
            for item in (prebinding_summaries or ())
            if isinstance(item.get("candidate_id"), str)
            and item.get("candidate_id")
        )
        retryable_ids.update(
            str(item["candidate_id"])
            for item in feedback_rejections
            if isinstance(item.get("candidate_id"), str)
            and item.get("candidate_id")
        )
        return CampaignState(
            campaign_id=before.campaign_id,
            next_round_index=before.next_round_index,
            context=next_context,
            active_profile=before.active_profile,
            policy=before.policy,
            search_memory_head=before.search_memory_head,
            carryover_proposals=tuple(
                item
                for item in before.carryover_proposals
                if item.candidate_id not in retryable_ids
            ),
            carryover_open_candidates=tuple(
                item
                for item in before.carryover_open_candidates
                if item.candidate_id not in retryable_ids
            ),
            qualified_execution_by_capability=before.qualified_execution_by_capability,
            candidate_root_by_capability=before.candidate_root_by_capability,
            resource_profile_by_capability=getattr(
                before, "resource_profile_by_capability", {}
            ),
            deferred_innovation_backlog=getattr(
                before, "deferred_innovation_backlog", ()
            ),
            incumbent_observation=before.incumbent_observation,
            frontier=before.frontier,
            last_round_result_digest=sha256_digest(result.to_dict()),
        )

    @staticmethod
    def _prebinding_candidate_failure_summaries(
        result: ResearchRoundResult,
    ) -> tuple[Mapping[str, Any], ...]:
        """Collapse implementation revisions into one diagnostic per candidate."""

        if result.innovation is None:
            return ()
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
        for item in result.innovation.attempts:
            if not isinstance(item, Mapping):
                continue
            candidate_id = item.get("candidate_id")
            spec_digest = item.get("spec_digest")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            if not isinstance(spec_digest, str) or not spec_digest:
                continue
            grouped.setdefault((spec_digest, candidate_id), []).append(item)
        summaries: list[Mapping[str, Any]] = []
        for revisions in grouped.values():
            final = revisions[-1]
            # Repair failures may omit the unchanged candidate/spec mechanism.
            mechanism_context = {
                key: revision[key]
                for revision in revisions
                for key in ("core_mechanism_contrast", "next_discriminative_task")
                if revision.get(key) is not None
            }
            summaries.append(
                canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line.discovery-prebinding-"
                            "attempt-summary.v1"
                        ),
                        "candidate_id": final.get("candidate_id"),
                        "candidate_semantic_digest": final.get(
                            "mechanism_semantics_digest"
                        ),
                        "effective_experiment_digest": final.get(
                            "effective_experiment_digest"
                        ),
                        "effective_family_digest": final.get(
                            "effective_family_digest"
                        ),
                        "primitive_ids": final.get("primitive_ids"),
                        "spec_digest": final.get("spec_digest"),
                        **mechanism_context,
                        "producer_role": final.get("producer_role"),
                        "candidate_root": final.get("candidate_root"),
                        "repair_attempt": final.get("repair_attempt"),
                        "failure": final.get("failure"),
                        "revision_count": len(revisions),
                    }
                )
            )
        return tuple(summaries)

    @staticmethod
    def _successful_projection_generation_negative(
        result: ResearchRoundResult,
    ) -> Mapping[str, Any] | None:
        """Type a complete Provider generation rejected before OpenSpec binding."""

        if result.attempts or result.innovation is not None:
            return None
        prepared = result.prepared
        if (
            isinstance(prepared, PreparedResearchRoundV1)
            and _prepared_has_unfinished_producer_failure(prepared)
        ):
            # A missing call is not a rejected research generation. In
            # particular, retain completed peers while the final call resumes.
            return None
        if (
            isinstance(prepared, PreparedResearchRoundV1)
            and ResearchCampaign._prepared_round_has_terminal_provider_failure(
                prepared
            )
        ):
            failures = tuple(
                canonical_value(
                    {
                        "logical_call_id": trace.get("logical_call_id"),
                        "provider_role": (
                            trace.get("receipt", {}).get("provider_role")
                            if isinstance(trace.get("receipt"), Mapping)
                            else None
                        ),
                        "request_digest": (
                            trace.get("receipt", {}).get("request_digest")
                            if isinstance(trace.get("receipt"), Mapping)
                            else None
                        ),
                        "failure": trace.get("failure"),
                    }
                )
                for trace in prepared.provider_traces
                if isinstance(trace, Mapping)
                and any(
                    isinstance(attempt, Mapping)
                    and (
                        attempt.get("failure_class") == "CLI_CONTRACT_ERROR"
                        or attempt.get("reason_code")
                        in {
                            "CONTENT_JSON_DECODE",
                            "ENVELOPE_JSON_DECODE",
                            "SCHEMA_VALIDATION",
                        }
                    )
                    for attempt in trace.get("attempts", ())
                )
            )
            return canonical_value(
                {
                    "schema": "recclaw.research-line.discovery-generation-negative.v1",
                    "reason_code": "PROVIDER_GENERATION_TERMINAL_RESPONSE_FAILURE",
                    "terminal_failures": failures,
                    "provider_traces_digest": sha256_digest(
                        result.provider_traces
                    ),
                }
            )
        traces_by_role: dict[str, list[Mapping[str, Any]]] = {}
        for trace in result.provider_traces:
            if not isinstance(trace, Mapping):
                continue
            receipt = trace.get("receipt")
            if not isinstance(receipt, Mapping):
                continue
            role = receipt.get("provider_role")
            if isinstance(role, str):
                traces_by_role.setdefault(role, []).append(receipt)

        summaries: list[Mapping[str, Any]] = []
        seen_roles: set[str] = set()
        for outcome in result.producer_outcomes:
            role = outcome.producer_role
            receipts = traces_by_role.get(role, ())
            if not receipts and role in DISCOVERY_PRODUCERS:
                receipts = traces_by_role.get("original_matched_proposal", ())
            final_receipt = receipts[-1] if receipts else {}
            if (
                outcome.spec is not None
                or role in seen_roles
                or final_receipt.get("status") != "SUCCESS"
            ):
                continue
            response_digests = tuple(
                dict.fromkeys(
                    str(receipt["response_digest"])
                    for receipt in receipts
                    if receipt.get("status") == "SUCCESS"
                    and isinstance(receipt.get("response_digest"), str)
                    and receipt["response_digest"]
                )
            )
            if not response_digests:
                continue
            seen_roles.add(role)
            identity = sha256_digest(
                {
                    "producer_outcome_digest": outcome.digest,
                    "provider_response_digests": response_digests,
                }
            )
            summaries.append(
                canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line."
                            "discovery-prebinding-attempt-summary.v1"
                        ),
                        "candidate_id": f"producer-projection:{identity[:24]}",
                        "spec_digest": identity,
                        "producer_role": role,
                        "producer_outcome_digest": outcome.digest,
                        "provider_response_digests": response_digests,
                        "metric_bearing": False,
                        "failure": {
                            "failure_class": "PROJECTION",
                            "failure_scope": "ENGINEERING_DIAGNOSTIC",
                            "stage": "PROVIDER_RESPONSE_PROJECTION",
                            "reason_code": "PROVIDER_RESPONSE_PROJECTION_FAILED",
                            "message": outcome.failure_detail
                            or "successful Provider response did not project",
                            "producer_failure_code": outcome.failure_code,
                        },
                    }
                )
            )
        if not summaries:
            prepared = result.prepared
            complete_generation_without_binding = bool(
                result.incomplete_reason == "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
                and isinstance(prepared, PreparedResearchRoundV1)
                and not prepared.search_bindings
                and result.producer_outcomes
                and all(
                    outcome.spec is not None and outcome.failure_code is None
                    for outcome in result.producer_outcomes
                )
            )
            partial_generation_without_binding = bool(
                result.incomplete_reason == "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
                and isinstance(prepared, PreparedResearchRoundV1)
                and not prepared.search_bindings
                and result.producer_outcomes
                and any(
                    outcome.spec is not None
                    for outcome in result.producer_outcomes
                )
                and any(
                    outcome.failure_code is not None
                    and bool(str(outcome.failure_detail or "").strip())
                    for outcome in result.producer_outcomes
                )
            )
            if partial_generation_without_binding:
                return canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line."
                            "discovery-generation-negative.v1"
                        ),
                        "reason_code": (
                            "PROVIDER_GENERATION_PARTIAL_NO_LEGAL_BINDING"
                        ),
                        "producer_outcomes": tuple(
                            canonical_value(
                                {
                                    "producer_role": outcome.producer_role,
                                    "spec_digest": (
                                        outcome.spec.digest
                                        if outcome.spec is not None
                                        else None
                                    ),
                                    "failure_code": outcome.failure_code,
                                    "failure_detail_digest": (
                                        sha256_digest(outcome.failure_detail)
                                        if outcome.failure_detail
                                        else None
                                    ),
                                }
                            )
                            for outcome in result.producer_outcomes
                        ),
                        "provider_traces_digest": sha256_digest(
                            result.provider_traces
                        ),
                    }
                )
            if not complete_generation_without_binding:
                return None
            binding_rejections = tuple(
                canonical_value(
                    {
                        "producer_role": outcome.producer_role,
                        "spec_digest": outcome.spec.digest,
                        "resolution": getattr(
                            resolution.resolution, "value", str(resolution.resolution)
                        ),
                        "reason_codes": tuple(resolution.reason_codes),
                    }
                )
                for outcome, resolution in prepared.resolutions
                if outcome.spec is not None and resolution is not None
            )
            return canonical_value(
                {
                    "schema": "recclaw.research-line.discovery-generation-negative.v1",
                    "reason_code": "PROVIDER_GENERATION_NO_LEGAL_SEARCH_BINDING",
                    "binding_rejections": binding_rejections,
                    "provider_traces_digest": sha256_digest(result.provider_traces),
                }
            )
        return canonical_value(
            {
                "schema": "recclaw.research-line.discovery-generation-negative.v1",
                "reason_code": "PROVIDER_GENERATION_UNPROJECTABLE",
                "projection_failures": summaries,
                "provider_traces_digest": sha256_digest(result.provider_traces),
            }
        )

    @staticmethod
    def _unreplayable_started_generation_negative(
        result: ResearchRoundResult,
    ) -> Mapping[str, Any] | None:
        """Close a half-started Provider generation whose sealed calls are absent."""

        prepared = result.prepared
        if (
            result.attempts
            or result.innovation is not None
            or result.provider_traces
            or not isinstance(prepared, PreparedResearchRoundV1)
            or prepared.search_bindings
        ):
            return None
        outcomes = tuple(result.producer_outcomes)
        scheduled_roles = _prepared_producer_roles(prepared)
        if (
            len(outcomes) != len(scheduled_roles)
            or {outcome.producer_role for outcome in outcomes}
            != set(scheduled_roles)
        ):
            return None
        missing_by_role: list[Mapping[str, Any]] = []
        for outcome in outcomes:
            detail = str(outcome.failure_detail or "")
            if outcome.failure_code != "SEALED_PROVIDER_REQUEST_UNAVAILABLE":
                return None
            missing_by_role.append(
                canonical_value(
                    {
                        "producer_role": outcome.producer_role,
                        "failure_detail_digest": sha256_digest(detail),
                    }
                )
            )
        return canonical_value(
            {
                "schema": "recclaw.research-line.discovery-generation-negative.v1",
                "reason_code": "STARTED_GENERATION_SEALED_REQUESTS_UNAVAILABLE",
                "missing_provider_requests": tuple(missing_by_role),
            }
        )

    @staticmethod
    def _candidate_failure_key(summary: Mapping[str, Any]) -> tuple[str, str]:
        return (
            str(summary.get("candidate_id") or ""),
            str(
                summary.get("spec_digest")
                or summary.get("candidate_semantic_digest")
                or ""
            ),
        )

    @classmethod
    def _current_round_candidate_failure_summaries(
        cls,
        state: CampaignState,
    ) -> tuple[Mapping[str, Any], ...]:
        rows = state.context.scientific_memory.get("round_attempts", ())
        if not isinstance(rows, (tuple, list)):
            return ()
        by_candidate: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in rows:
            if (
                not isinstance(item, Mapping)
                or item.get("round_index") != state.next_round_index
                or item.get("metric_bearing") is True
            ):
                continue
            schema = item.get("schema")
            if schema == (
                "recclaw.research-line.discovery-prebinding-attempt-summary.v1"
            ):
                if not cls._is_candidate_local_prebinding_failure(item):
                    continue
            elif schema == "recclaw.research-line.round-attempt-summary.v1":
                if item.get("failure_scope") not in {
                    "CANDIDATE_LOCAL",
                    "LINEAGE_COMPUTE_PATTERN",
                } or item.get("failure_detail_digest") is None:
                    continue
            else:
                continue
            key = cls._candidate_failure_key(item)
            if all(key):
                by_candidate[key] = item
        return tuple(by_candidate.values())

    @classmethod
    def _current_round_budget_attempt_summaries(
        cls,
        state: CampaignState,
    ) -> tuple[Mapping[str, Any], ...]:
        """Return durable distinct candidate/spec attempts for slot budgeting.

        Only candidates that reached implementation or qualification consume
        the fixed slot.  Proposal-semantic and effective-family duplicates are
        generation diagnostics and can trigger a bounded fresh generation.
        """

        rows = state.context.scientific_memory.get("round_attempts", ())
        if not isinstance(rows, (tuple, list)):
            return ()
        by_candidate: dict[tuple[str, str], Mapping[str, Any]] = {}
        for item in rows:
            if (
                not isinstance(item, Mapping)
                or item.get("round_index") != state.next_round_index
                or item.get("metric_bearing") is True
                or item.get("schema")
                not in {
                    "recclaw.research-line.discovery-prebinding-attempt-summary.v1",
                    "recclaw.research-line.round-attempt-summary.v1",
                }
            ):
                continue
            key = cls._candidate_failure_key(item)
            if all(key) and cls._budget_attempt_consumed(item):
                by_candidate[key] = item
        return tuple(by_candidate.values())

    @classmethod
    def _budget_attempt_consumed(cls, summary: Mapping[str, Any]) -> bool:
        """Separate fixed candidate-attempt accounting from evidence blame."""

        if not all(cls._candidate_failure_key(summary)):
            return False
        schema = summary.get("schema")
        if schema == "recclaw.research-line.round-attempt-summary.v1":
            return summary.get("failure_scope") in {
                "CANDIDATE_LOCAL",
                "LINEAGE_COMPUTE_PATTERN",
            }
        if schema != (
            "recclaw.research-line.discovery-prebinding-attempt-summary.v1"
        ):
            return False
        return cls._is_candidate_local_prebinding_failure(summary)

    @staticmethod
    def _is_candidate_local_prebinding_failure(
        summary: Mapping[str, Any],
    ) -> bool:
        """Accept only real implementation/qualification candidate outcomes.

        Proposal semantics and effective-identity rejection happen before a
        candidate implementation exists.  They are generation diagnostics, not
        fixed-slot candidate outcomes, even when the runtime preserves them in
        the same compact prebinding evidence envelope.
        """

        failure = summary.get("failure")
        if not isinstance(failure, Mapping) or not failure:
            return False
        raw_scope = failure.get("failure_scope") or failure.get("engineering_scope")
        if raw_scope is not None and str(
            getattr(raw_scope, "value", raw_scope) or ""
        ).upper() != "CANDIDATE_LOCAL":
            return False
        raw_class = failure.get("failure_class")
        failure_class = str(getattr(raw_class, "value", raw_class) or "").upper()
        stage = str(failure.get("stage") or "").upper()
        if failure_class in {
            "PROJECTION",
            "PROVIDER",
            "SELECTION",
            "SEMANTIC_IDENTITY",
            "SHARED_INFRASTRUCTURE",
        } or stage in {
            "POST_PROPOSAL_PRE_IMPLEMENTATION",
            "PROVIDER_RESPONSE_PROJECTION",
        }:
            return False
        return bool(str(failure.get("reason_code") or "").strip()) and bool(
            str(failure.get("message") or "").strip()
        )

    @staticmethod
    def _terminal_physical_generation_negative(
        result: ResearchRoundResult,
        *,
        allow_untried_prepared_bindings: bool = False,
    ) -> Mapping[str, Any] | None:
        """Consume terminal non-candidate physical work once per generation.

        The physical observation remains the immutable idempotence authority,
        while this compact projection advances discovery without claiming a
        candidate outcome, metric, or explored mechanism identity.
        """

        attempted_ids = {attempt.candidate_id for attempt in result.attempts}
        prepared_ids = {
            binding.proposal.candidate_id
            for binding in (
                result.prepared.search_bindings
                if isinstance(result.prepared, PreparedResearchRoundV1)
                else ()
            )
        }
        if prepared_ids - attempted_ids and not allow_untried_prepared_bindings:
            return None

        terminal = tuple(
            canonical_value(
                {
                    "candidate_id": attempt.candidate_id,
                    "failure_scope": attempt.failure_scope,
                    "engineering_disposition": attempt.engineering_disposition,
                    "exit_status": attempt.candidate_run.get("exit_status"),
                    "physical_observation_ref": attempt.observation_ref,
                    "physical_observation_digest": attempt.observation_digest,
                }
            )
            for attempt in result.attempts
            if not attempt.metric_bearing
            and attempt.failure_scope in {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE"}
            and isinstance(attempt.observation_digest, str)
            and attempt.observation_digest
        )
        if not terminal:
            return None
        return canonical_value(
            {
                "schema": "recclaw.research-line.discovery-generation-negative.v1",
                "reason_code": "TERMINAL_PHYSICAL_ENGINEERING_FAILURE",
                "terminal_physical_observations": terminal,
                "terminal_observations_digest": sha256_digest(terminal),
            }
        )

    @classmethod
    def _preimplementation_generation_negative(
        cls,
        result: ResearchRoundResult,
    ) -> Mapping[str, Any] | None:
        """Project an all-preimplementation generation into one typed negative."""

        if result.attempts or result.innovation is None:
            return None
        if (
            _innovation_resource_failure_scope(result.innovation)
            in {"WORKER_TRANSIENT", "SHARED_INFRASTRUCTURE", "RECOVERY"}
        ):
            return None
        summaries = cls._prebinding_candidate_failure_summaries(result)
        if not summaries or any(
            cls._is_candidate_local_prebinding_failure(item)
            for item in summaries
        ):
            return None
        failures = tuple(
            canonical_value(
                {
                    "candidate_id": item.get("candidate_id"),
                    "spec_digest": item.get("spec_digest"),
                    "failure_class": (
                        item.get("failure", {}).get("failure_class")
                        if isinstance(item.get("failure"), Mapping)
                        else None
                    ),
                    "stage": (
                        item.get("failure", {}).get("stage")
                        if isinstance(item.get("failure"), Mapping)
                        else None
                    ),
                    "reason_code": (
                        item.get("failure", {}).get("reason_code")
                        if isinstance(item.get("failure"), Mapping)
                        else None
                    ),
                }
            )
            for item in summaries
        )
        return canonical_value(
            {
                "schema": "recclaw.research-line.discovery-generation-negative.v1",
                "reason_code": "PROVIDER_GENERATION_PREIMPLEMENTATION_REJECTED",
                "preimplementation_failures": failures,
                "provider_traces_digest": sha256_digest(result.provider_traces),
            }
        )

    @staticmethod
    def _is_slot_training_consumed_summary(summary: Mapping[str, Any]) -> bool:
        return bool(
            summary.get("schema")
            == "recclaw.research-line.round-attempt-summary.v1"
            and summary.get("metric_bearing") is False
            and summary.get("failure_scope") != "SHARED_INFRASTRUCTURE"
            and str(summary.get("outcome", "")).upper() == "RESOURCE_CENSORED"
            and isinstance(summary.get("physical_observation_digest"), str)
            and summary.get("physical_observation_digest")
        )

    @staticmethod
    def _can_close_exhausted_no_metric_slot(
        inputs: CampaignRoundInputs,
        result: ResearchRoundResult,
        attempt_budget: int | None,
        *,
        slot_failure_summaries: tuple[Mapping[str, Any], ...] | None = None,
        budget_attempt_summaries: tuple[Mapping[str, Any], ...] | None = None,
    ) -> bool:
        if result.has_metric_bearing_attempt:
            return False
        if result.incomplete_reason == "ROUND_SLOT_TRAINING_CONSUMED_NO_METRIC":
            if slot_failure_summaries is not None:
                return any(
                    ResearchCampaign._is_slot_training_consumed_summary(summary)
                    for summary in slot_failure_summaries
                )
            return any(
                not attempt.metric_bearing
                and attempt.failure_scope != "SHARED_INFRASTRUCTURE"
                and str(attempt.candidate_run.get("exit_status", "")).upper()
                == "RESOURCE_CENSORED"
                and isinstance(attempt.observation_digest, str)
                and attempt.observation_digest
                for attempt in result.attempts
            )
        if (
            attempt_budget is None
            or attempt_budget < 1
            or result.incomplete_reason != "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
        ):
            return False
        if budget_attempt_summaries is not None:
            keys = tuple(
                ResearchCampaign._candidate_failure_key(summary)
                for summary in budget_attempt_summaries
            )
            return bool(
                len(keys) == attempt_budget
                and all(all(key) for key in keys)
                and len(set(keys)) == attempt_budget
            )
        if slot_failure_summaries is not None:
            if len(slot_failure_summaries) != attempt_budget:
                return False
            keys = tuple(
                ResearchCampaign._candidate_failure_key(summary)
                for summary in slot_failure_summaries
            )
            if any(not all(key) for key in keys) or len(set(keys)) != attempt_budget:
                return False
            return all(
                (
                    ResearchCampaign._is_candidate_local_prebinding_failure(
                        summary
                    )
                    if summary.get("schema")
                    == (
                        "recclaw.research-line."
                        "discovery-prebinding-attempt-summary.v1"
                    )
                    else summary.get("schema")
                    == "recclaw.research-line.round-attempt-summary.v1"
                    and summary.get("metric_bearing") is False
                    and summary.get("failure_scope")
                    in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
                    and summary.get("failure_detail_digest") is not None
                )
                for summary in slot_failure_summaries
            )
        if result.attempts:
            if len(result.attempts) != attempt_budget:
                return False
            return all(
                not attempt.metric_bearing
                and attempt.failure_scope
                in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
                and attempt.failure_detail is not None
                for attempt in result.attempts
            )
        summaries = (
            slot_failure_summaries
            if slot_failure_summaries is not None
            else ResearchCampaign._prebinding_candidate_failure_summaries(result)
        )
        if len(summaries) != attempt_budget:
            return False
        if len({summary["candidate_id"] for summary in summaries}) != attempt_budget:
            return False
        if len({summary["spec_digest"] for summary in summaries}) != attempt_budget:
            return False
        return all(
            ResearchCampaign._is_candidate_local_prebinding_failure(summary)
            for summary in summaries
        )

    @classmethod
    def _close_exhausted_no_metric_state(
        cls,
        before: CampaignState,
        result: ResearchRoundResult,
        *,
        prebinding_summaries: tuple[Mapping[str, Any], ...] | None = None,
        slot_failure_summaries: tuple[Mapping[str, Any], ...] | None = None,
        budget_attempt_summaries: tuple[Mapping[str, Any], ...] | None = None,
        generation_negative: Mapping[str, Any] | None = None,
    ) -> CampaignState:
        held = cls._hold_state_after_incomplete_attempts(
            before,
            replace(result, prepared=None),
            prebinding_summaries=prebinding_summaries,
            generation_negative=generation_negative,
        )
        if held.frontier != before.frontier:
            raise CampaignError("no-metric slot closure must not change the frontier")
        scientific_memory = dict(held.context.scientific_memory)
        prior_failures = scientific_memory.get("no_metric_slot_failures", ())
        prior_failures = (
            tuple(prior_failures)
            if isinstance(prior_failures, (tuple, list))
            else ()
        )
        closure_summaries = (
            slot_failure_summaries
            if slot_failure_summaries is not None
            else cls._prebinding_candidate_failure_summaries(result)
            if not result.attempts
            else ()
        )
        slot_attempt_summaries = (
            budget_attempt_summaries
            if budget_attempt_summaries is not None
            else tuple(
                _compact_round_attempt_summary(
                    attempt,
                    round_index=before.next_round_index,
                )
                for attempt in result.attempts
            )
            if result.attempts
            else closure_summaries
        )
        candidate_negative_interpretation = result.interpretation
        if (
            closure_summaries
            and candidate_negative_interpretation is not None
            and (
                candidate_negative_interpretation.failure_taxonomy
                == "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
                or (
                    budget_attempt_summaries is not None
                    and len(budget_attempt_summaries) > len(result.attempts)
                )
            )
        ):
            candidate_negative_interpretation = None
        diagnostic_context = (
            candidate_negative_interpretation.successor_context
            if candidate_negative_interpretation is not None
            else None
        )
        diagnostic_policy = (
            candidate_negative_interpretation.policy_successor
            if candidate_negative_interpretation is not None
            else None
        )
        diagnostic_search_memory = (
            candidate_negative_interpretation.search_memory_snapshot
            if candidate_negative_interpretation is not None
            else None
        )
        synthetic_candidate_feedback = None
        if diagnostic_context is None and closure_summaries:
            synthetic_candidate_feedback = canonical_value(
                {
                    "schema": (
                        "recclaw.research-line."
                        "candidate-no-metric-feedback.v1"
                    ),
                    "round_index": before.next_round_index,
                    "failure_class": "CANDIDATE_ATTRIBUTED_NO_METRIC",
                    "candidate_failures": closure_summaries,
                    "slot_attempt_count": len(slot_attempt_summaries),
                    "mechanism_effect_update_allowed": False,
                }
            )
            predecessor_digest = (
                before.search_memory_head.digest
                if before.search_memory_head is not None
                else None
            )
            diagnostic_search_memory = SearchMemorySnapshotV1(
                namespace="DEVELOPMENT_ONLY/SEARCH_MEMORY",
                round_index=before.next_round_index,
                predecessor_digest=predecessor_digest,
                beliefs=(
                    before.search_memory_head.beliefs
                    if before.search_memory_head is not None
                    else ()
                ),
                route_trace_digest=sha256_digest(closure_summaries),
                feedback_projection_digest=sha256_digest(
                    synthetic_candidate_feedback
                ),
            )
            diagnostic_context = held.context
            diagnostic_policy = before.policy

        if diagnostic_policy is not None:
            acquisition_parameters = dict(diagnostic_policy.acquisition_parameters)
            latest_event = None
            if result.attempts:
                feedback = result.attempts[-1].diagnostic_feedback
                if isinstance(feedback, Mapping):
                    candidate_event = feedback.get("common_search_utility_slot")
                    if isinstance(candidate_event, Mapping):
                        latest_event = candidate_event
            if latest_event is not None:
                for target, source in (
                    ("core_mechanism_contrast", "core_mechanism_contrast"),
                    ("unresolved_confounding", "unresolved_confounding"),
                    ("last_evidence_class", "evidence_class"),
                    ("last_failure_class", "failure_class"),
                    ("last_axis_footprint", "mechanism_axis_footprint"),
                ):
                    if latest_event.get(source) is not None:
                        acquisition_parameters[target] = canonical_value(
                            latest_event[source]
                        )
            elif closure_summaries:
                failure = closure_summaries[-1].get("failure")
                failure = failure if isinstance(failure, Mapping) else {}
                acquisition_parameters["last_evidence_class"] = (
                    "CANDIDATE_ATTRIBUTED_PREBINDING_NEGATIVE"
                )
                acquisition_parameters["last_failure_class"] = str(
                    failure.get("failure_class") or "CANDIDATE_LOCAL"
                )
            for target, source in (
                ("attempted_family_digests", "effective_family_digest"),
                ("attempted_experiment_digests", "effective_experiment_digest"),
            ):
                values = list(acquisition_parameters.get(target, ()))
                for summary in closure_summaries:
                    value = summary.get(source)
                    if isinstance(value, str) and value and value not in values:
                        values.append(value)
                if values:
                    acquisition_parameters[target] = tuple(values[-800:])
            diagnostic_policy = replace(
                diagnostic_policy,
                acquisition_parameters=tuple(acquisition_parameters.items()),
            )

        typed_negative_evidence = tuple(
            canonical_value(
                {
                    "schema": (
                        "recclaw.research-line."
                        "candidate-no-metric-negative.v1"
                    ),
                    "candidate_id": summary.get("candidate_id"),
                    "candidate_semantic_digest": summary.get(
                        "candidate_semantic_digest"
                    ),
                    "effective_experiment_digest": summary.get(
                        "effective_experiment_digest"
                    ),
                    "effective_family_digest": summary.get(
                        "effective_family_digest"
                    ),
                    "outcome": summary.get("outcome"),
                    "failure_scope": summary.get("failure_scope")
                    or (
                        summary.get("failure", {}).get("failure_scope")
                        if isinstance(summary.get("failure"), Mapping)
                        else None
                    ),
                    "failure": summary.get("failure"),
                    "implementation_efficiency_repair_context": summary.get(
                        "implementation_efficiency_repair_context"
                    ),
                    "physical_observation_ref": summary.get(
                        "physical_observation_ref"
                    ),
                    "physical_observation_digest": summary.get(
                        "physical_observation_digest"
                    ),
                    "mechanism_effect_update_allowed": False,
                }
            )
            for summary in closure_summaries
        )
        generation_negatives = tuple(
            item
            for item in scientific_memory.get("generation_negatives", ())
            if isinstance(item, Mapping)
            and item.get("round_index") == before.next_round_index
        )
        closure = canonical_value(
            {
                "schema": "recclaw.research-line.no-metric-slot-failure.v1",
                "round_index": before.next_round_index,
                "failure_code": (
                    "SLOT_TRAINING_CONSUMED_NO_METRIC"
                    if result.incomplete_reason
                    == "ROUND_SLOT_TRAINING_CONSUMED_NO_METRIC"
                    else (
                        "DISCOVERY_GENERATION_BUDGET_EXHAUSTED_"
                        "NO_LEGAL_BINDING"
                    )
                    if result.incomplete_reason
                    == (
                        "ROUND_DISCOVERY_GENERATION_BUDGET_EXHAUSTED_"
                        "NO_LEGAL_BINDING"
                    )
                    else "FIXED_ATTEMPT_BUDGET_EXHAUSTED_NO_METRIC"
                ),
                "attempt_count": len(slot_attempt_summaries),
                "attempt_digests": tuple(
                    sha256_digest(summary) for summary in slot_attempt_summaries
                ),
                "incomplete_reason": result.incomplete_reason,
                "frontier_updated": False,
                "typed_negative_evidence": typed_negative_evidence,
                "generation_negative_count": len(generation_negatives),
                "generation_negative_digests": tuple(
                    sha256_digest(item) for item in generation_negatives
                ),
            }
        )
        if diagnostic_context is not None:
            diagnostic_memory = dict(diagnostic_context.scientific_memory)
            bookkeeping_keys = (
                "round_attempts",
                *_ROUND_ATTEMPT_IDENTITY_FIELDS,
                "producer_opportunity_state",
                "metric_observation_index",
                "generation_negatives",
                "stale_prepared_feedback_retry_generation",
            )
            for key in bookkeeping_keys:
                if key in scientific_memory:
                    diagnostic_memory[key] = scientific_memory[key]
            diagnostic_global = diagnostic_memory.get("global_memory")
            held_global = scientific_memory.get("global_memory")
            if isinstance(diagnostic_global, Mapping) and isinstance(
                held_global, Mapping
            ):
                merged_global = dict(diagnostic_global)
                for key in bookkeeping_keys:
                    if key in held_global:
                        merged_global[key] = held_global[key]
                diagnostic_memory["global_memory"] = canonical_value(
                    merged_global
                )
            if synthetic_candidate_feedback is not None:
                diagnostic_memory["search_memory_head"] = (
                    diagnostic_search_memory.to_dict()
                )
                diagnostic_memory["latest_feedback"] = (
                    synthetic_candidate_feedback
                )
                global_memory = diagnostic_memory.get("global_memory")
                if isinstance(global_memory, Mapping):
                    global_memory = dict(global_memory)
                    global_memory["search_memory_head"] = (
                        diagnostic_search_memory.to_dict()
                    )
                    global_memory["latest_feedback"] = (
                        synthetic_candidate_feedback
                    )
                    diagnostic_memory["global_memory"] = canonical_value(
                        global_memory
                    )
            scientific_memory = diagnostic_memory
            prior_failures = scientific_memory.get(
                "no_metric_slot_failures", prior_failures
            )
            prior_failures = (
                tuple(prior_failures)
                if isinstance(prior_failures, (tuple, list))
                else ()
            )
        scientific_memory["no_metric_slot_failures"] = (
            *prior_failures,
            closure,
        )[-_ROUND_ATTEMPT_HISTORY_LIMIT:]
        scientific_memory["discovery_generation"] = 0
        next_context = replace(
            diagnostic_context or held.context,
            round_index=before.next_round_index + 1,
            frontier=before.context.frontier,
            scientific_memory=canonical_value(scientific_memory),
            policy=(
                diagnostic_policy.to_dict()
                if diagnostic_policy is not None
                else held.context.policy
            ),
        )
        return replace(
            held,
            next_round_index=before.next_round_index + 1,
            context=next_context,
            incumbent_observation=before.incumbent_observation,
            frontier=before.frontier,
            policy=diagnostic_policy or held.policy,
            search_memory_head=(
                diagnostic_search_memory
                if diagnostic_search_memory is not None
                else held.search_memory_head
            ),
            last_round_result_digest=sha256_digest(result.to_dict()),
        )

    @staticmethod
    def _attach_candidate_no_metric_interpretation(
        before: CampaignState,
        result: ResearchRoundResult,
        after: CampaignState,
        slot_failure_summaries: tuple[Mapping[str, Any], ...],
    ) -> tuple[ResearchRoundResult, CampaignState]:
        """Bind one truthful typed interpretation to an aggregated slot closure."""

        feedback = after.context.scientific_memory.get("latest_feedback")
        if not isinstance(feedback, Mapping) or feedback.get("schema") != (
            "recclaw.research-line.candidate-no-metric-feedback.v1"
        ):
            if (
                result.interpretation is not None
                and result.interpretation.failure_taxonomy
                != "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
            ):
                return result, replace(
                    after,
                    last_round_result_digest=sha256_digest(result.to_dict()),
                )
            raise CampaignError(
                "candidate no-metric closure lacks typed successor feedback"
            )
        if after.search_memory_head is None:
            raise CampaignError(
                "candidate no-metric closure lacks a search-memory successor"
            )
        negative_evidence = tuple(
            "/".join(
                str(value)
                for value in (
                    failure.get("failure_class"),
                    failure.get("stage"),
                    failure.get("reason_code"),
                )
                if value is not None
            )
            for summary in slot_failure_summaries
            for failure in (
                summary.get("failure")
                if isinstance(summary.get("failure"), Mapping)
                else {},
            )
        )
        interpretation = MissingSearchInterpretation(
            failure_taxonomy="CANDIDATE_ATTRIBUTED_NO_METRIC",
            mechanism_attribution="NOT_APPLICABLE",
            negative_evidence=negative_evidence,
            feedback_projection=canonical_value(dict(feedback)),
            search_memory_snapshot=after.search_memory_head,
            policy_successor=after.policy,
            successor_context=after.context,
            behavior_before=_behavior(before.context, before.policy),
            behavior_after=_behavior(after.context, after.policy),
            route_trace_digest=sha256_digest(slot_failure_summaries),
        )
        normalized_result = replace(result, interpretation=interpretation)
        normalized_after = replace(
            after,
            last_round_result_digest=sha256_digest(normalized_result.to_dict()),
        )
        return normalized_result, normalized_after

    @staticmethod
    def _status(result: ResearchRoundResult) -> str:
        if not result.has_metric_bearing_attempt:
            return "INCOMPLETE"
        if result.selected_outcome is None or result.candidate_run is None:
            return "OUTCOME_MISSING"
        if result.interpretation is None or getattr(result.interpretation, "episode", None) is None:
            return "TYPED_FAILURE"
        return "TYPED_EPISODE"

    def _load_round(self, round_index: int) -> CampaignRoundRecord:
        record = _read_pickle(self.round_checkpoint_path(round_index))
        if not isinstance(record, CampaignRoundRecord):
            raise CampaignError("round checkpoint does not contain CampaignRoundRecord")
        if record.round_index != round_index:
            raise CampaignError("round checkpoint index drift")
        trace_path = self.round_trace_path(round_index)
        if trace_path.is_file():
            trace = _read_json(trace_path)
            if trace.get("record_digest") != record.digest:
                raise CampaignError("round trace digest drift")
        return record

    def _prior_admission_preparation(
        self, record: CampaignRoundRecord,
    ) -> tuple[CampaignState, PreparedResearchRoundV1] | None:
        """Recover the original slate if an old admission stop advanced to an unsent generation."""
        current = record.result.prepared
        if (
            record.status != "INCOMPLETE"
            or not record.result.attempt_scheduler_enabled
            or record.result.attempts
            or record.round_index <= 1
            or _discovery_generation(record.state_before.context) != 1
            or record.state_before.digest != record.state_after.digest
            or not isinstance(current, PreparedResearchRoundV1)
            or current.provider_traces
            or current.innovation is not None
            or current.search_bindings
            or any(outcome.spec is not None for outcome in current.producer_outcomes)
        ):
            return None
        generation_stops = tuple(
            item for item in record.state_before.context.scientific_memory.get("generation_negatives", ())
            if item.get("round_index") == record.round_index and item.get("discovery_generation") == 0
        )
        if (
            len(generation_stops) != 1
            or generation_stops[0].get("reason_code") != "PROVIDER_GENERATION_PREIMPLEMENTATION_REJECTED"
            or not generation_stops[0].get("preimplementation_failures")
            or not all(_is_implementation_admission_failure(failure)
                       for failure in generation_stops[0]["preimplementation_failures"])
        ):
            return None
        path = self.root / f"ROUND_{record.round_index:02d}_PREPARED_CHECKPOINT.pkl"
        if not path.is_file():
            return None
        payload = _read_pickle(path)
        prepared = payload.get("prepared")
        if (
            not isinstance(prepared, PreparedResearchRoundV1)
            or prepared.innovation is None
            or not any(
                isinstance(failure := attempt.get("failure"), Mapping)
                and _is_implementation_admission_failure(failure)
                for attempt in prepared.innovation.attempts
            )
        ):
            return None
        original = self._load_round(record.round_index - 1).state_after
        if (
            original.next_round_index != record.round_index
            or _discovery_generation(original.context) != 0
            or original.digest != payload.get("state_digest")
            or original.context.digest != prepared.context_digest
        ):
            return None
        return original, prepared

    def _seal_record(self, record: CampaignRoundRecord) -> CampaignRoundRecord:
        checkpoint_payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_path = self.round_checkpoint_path(record.round_index)
        replace_incomplete = False
        if checkpoint_path.is_file():
            prior = self._load_round(record.round_index)
            if not (
                prior.status == "INCOMPLETE"
                and prior.result.attempt_scheduler_enabled
            ):
                raise CampaignError("cannot overwrite a sealed completed round")
            replaying_generation = (
                prior.opportunity_ref == record.opportunity_ref
                and prior.state_before.digest == record.state_before.digest
            )
            advancing_generation = (
                prior.state_after.digest == record.state_before.digest
                and _discovery_generation(record.state_before.context)
                == _discovery_generation(prior.state_before.context) + 1
            )
            admission_preparation = self._prior_admission_preparation(prior)
            resuming_admission = (
                admission_preparation is not None
                and admission_preparation[0].digest == record.state_before.digest
            )
            if not replaying_generation and not advancing_generation and not resuming_admission:
                raise CampaignError(
                    "incomplete round is neither a replay nor its next discovery generation"
                )
            if resuming_admission:
                # Keep the erroneous, unsent generation as history while the
                # ordinary checkpoint resumes the original scientific state.
                prefix = f"ROUND_{record.round_index:02d}_GENERATION_01"
                _write_once(self.root / f"{prefix}_CHECKPOINT.pkl", checkpoint_path.read_bytes())
                _write_once(self.root / f"{prefix}_TRACE.json", self.round_trace_path(record.round_index).read_bytes())
                _write_once(self.root / f"{prefix}_ADMISSION_RESUME.json", canonical_json_bytes({
                    "reason": "unsent generation followed a pre-HTTP implementation admission stop",
                    "retained_record_digest": prior.digest,
                    "resumed_state_digest": record.state_before.digest,
                }) + b"\n")
            replace_incomplete = True
        if replace_incomplete:
            _atomic_write(checkpoint_path, checkpoint_payload)
        elif not _write_once(checkpoint_path, checkpoint_payload):
            raise CampaignError("round checkpoint was sealed concurrently")
        trace = compact_campaign_round_trace(
            record,
            checkpoint_sha256=hashlib.sha256(checkpoint_payload).hexdigest(),
            checkpoint_ref=checkpoint_path.name,
        )
        trace_payload = canonical_json_bytes(trace) + b"\n"
        trace_path = self.round_trace_path(record.round_index)
        if replace_incomplete:
            _atomic_write(trace_path, trace_payload)
        elif not _write_once(trace_path, trace_payload):
            raise CampaignError("round trace was sealed concurrently")
        self._state = record.state_after
        self.memory_writer._head = record.state_after.search_memory_head
        self._persist_state()
        return record

    def record_missing_round(
        self,
        round_index: int | None = None,
        *,
        reason: str = "interrupted before a durable physical observation",
        failure_code: str = "INTERRUPTED_BEFORE_DURABLE_PHYSICAL_OBSERVATION",
    ) -> CampaignRoundRecord:
        """Consume one interrupted opportunity without replaying external work."""

        expected = self._state.next_round_index
        index = expected if round_index is None else int(round_index)
        checkpoint_path = self.round_checkpoint_path(index)
        if checkpoint_path.is_file():
            record = self._load_round(index)
            if record.status == "INCOMPLETE" and record.result.attempt_scheduler_enabled:
                return self.run_round(index)
            return record
        if index != expected:
            raise CampaignError(
                f"campaign is at round {expected}; cannot record round {index}"
            )
        inputs = self._round_inputs()
        if inputs.attempt_scheduler and (
            self._started_path(index).is_file()
            or self._prepared_round_path(index).is_file()
        ):
            # A scheduler round with a start marker is recoverable only from
            # its prepared boundary; never turn it into a synthetic missing
            # round or re-run preparation here.
            return self.run_round(index)
        if self._physical_path(index).is_file():
            raise CampaignError(
                "durable physical observation exists; recover the ordinary round"
            )
        opportunity_ref = _opportunity_ref(self._state)
        _write_once(
            self._started_path(index),
            canonical_json_bytes(
                {
                    "schema": "recclaw.research-line.campaign-round-started.v1",
                    "round_index": index,
                    "opportunity_ref": opportunity_ref,
                    "state_digest": self._state.digest,
                    "context_digest": self._state.context.digest,
                    "profile_digest": self._state.active_profile.profile_digest,
                }
            )
            + b"\n",
        )
        round_memory = self._round_memory_writer()

        def interrupted_producer(_role: str, _view: Mapping[str, Any]) -> Any:
            raise RuntimeError(reason)

        producer_outcomes = produce_research_specs(
            self._state.context,
            interrupted_producer,
            inputs.producer_bindings,
        )
        interpretation = interpret_missing_search_opportunity(
            context=self._state.context,
            producer_outcomes=producer_outcomes,
            diagnostic_detail={
                "reason": reason,
                "failure_code": failure_code,
                "opportunity_ref": opportunity_ref,
            },
            next_discriminative_test=inputs.next_discriminative_test,
            policy=self._state.policy,
            memory_writer=round_memory,
        )
        result = ResearchRoundResult(
            context=self._state.context,
            active_profile=self._state.active_profile,
            producer_outcomes=producer_outcomes,
            carryover_outcomes=(),
            resolutions=(),
            deferred_innovation_outcomes=(),
            deferred_search_outcomes=(),
            search_acquisition=None,
            innovation=None,
            selected_outcome=None,
            execution_recipe=None,
            candidate_run=None,
            interpretation=interpretation,
            provider_traces=(),
            meta_research=None,
            attempt_scheduler_enabled=inputs.attempt_scheduler,
            incomplete_reason=(
                "ROUND_INTERRUPTED_BEFORE_PHYSICAL_OBSERVATION"
                if inputs.attempt_scheduler
                else None
            ),
        )
        before = self._state
        after = before if inputs.attempt_scheduler else self._advance_state(before, result)
        return self._seal_record(
            CampaignRoundRecord(
                round_index=index,
                opportunity_ref=opportunity_ref,
                status=("INCOMPLETE" if inputs.attempt_scheduler else "OUTCOME_MISSING"),
                state_before=before,
                result=result,
                state_after=after,
            )
        )

    def run_round(self, round_index: int | None = None) -> CampaignRoundRecord:
        # A durable STARTED marker owns the current round identity.  Finish that
        # round with its original Context; apply utility normalization only at
        # the next clean opportunity boundary.
        if not self._started_path(self._state.next_round_index).is_file():
            normalized_context, normalized_policy, utility_history_changed = (
                normalize_fidelity_utility_state(
                    self._state.context,
                    self._state.policy,
                )
            )
            if utility_history_changed:
                self._state = replace(
                    self._state,
                    context=normalized_context,
                    policy=normalized_policy,
                )
                self._persist_state()
        expected = self._state.next_round_index
        index = expected if round_index is None else round_index
        if isinstance(index, bool) or index < 1:
            raise CampaignError("round_index must be positive")

        checkpoint_path = self.round_checkpoint_path(index)
        resuming_incomplete = False
        continuing_discovery_generation = False
        continuing_generation_record: CampaignRoundRecord | None = None
        incomplete_resume_record: CampaignRoundRecord | None = None
        resume_existing_observation = False
        if checkpoint_path.is_file():
            record = self._load_round(index)
            admission_preparation = self._prior_admission_preparation(record)
            if admission_preparation is not None:
                original, prepared = admission_preparation
                if self._state.digest != record.state_after.digest:
                    raise CampaignError("admission resume current state drift")
                self._state = original
                self.memory_writer._head = original.search_memory_head
                record = replace(record, opportunity_ref=_opportunity_ref(original),
                    state_before=original, state_after=original, result=replace(
                        record.result, context=original.context, active_profile=original.active_profile,
                        producer_outcomes=prepared.producer_outcomes, carryover_outcomes=prepared.carryover_outcomes,
                        resolutions=prepared.resolutions, deferred_innovation_outcomes=prepared.deferred_innovation_outcomes,
                        deferred_search_outcomes=prepared.deferred_search_outcomes, search_acquisition=prepared.search_acquisition,
                        innovation=prepared.innovation, provider_traces=prepared.provider_traces,
                        interpretation=None, incomplete_reason="ROUND_IMPLEMENTATION_ADMISSION_PAUSED", prepared=prepared,
                    ))
            if record.status == "INCOMPLETE" and record.result.attempt_scheduler_enabled:
                incomplete_resume_record = record
                if index != self._state.next_round_index:
                    raise CampaignError(
                        "incomplete round is not the campaign's current opportunity"
                    )
                if self._state.digest not in {
                    record.state_before.digest,
                    record.state_after.digest,
                }:
                    raise CampaignError("incomplete round checkpoint state drift")
                before_generation = _discovery_generation(
                    record.state_before.context
                )
                after_generation = _discovery_generation(
                    record.state_after.context
                )
                if after_generation not in {
                    before_generation,
                    before_generation + 1,
                }:
                    raise CampaignError(
                        "incomplete round discovery generation drift"
                    )
                resume_existing_observation = (
                    self._can_resume_existing_observation_rebind(record)
                )
                before_negatives = record.state_before.context.scientific_memory.get(
                    "generation_negatives", ()
                )
                after_negatives = record.state_after.context.scientific_memory.get(
                    "generation_negatives", ()
                )
                generation_negative_advanced = (
                    isinstance(before_negatives, (tuple, list))
                    and isinstance(after_negatives, (tuple, list))
                    and len(after_negatives) > len(before_negatives)
                )
                before_generation_suffix = (
                    ""
                    if before_generation == 0
                    else f"_GENERATION_{before_generation:02d}"
                )
                prior_manifest_path = self.root / (
                    f"ROUND_{index:02d}{before_generation_suffix}_ATTEMPT_MANIFEST.json"
                )
                prior_attempt_budget = None
                if prior_manifest_path.is_file():
                    raw_attempt_budget = _read_json(prior_manifest_path).get(
                        "attempt_budget"
                    )
                    if isinstance(raw_attempt_budget, int) and not isinstance(
                        raw_attempt_budget, bool
                    ):
                        prior_attempt_budget = raw_attempt_budget
                untried_candidate_local_innovation = (
                    isinstance(record.result.prepared, PreparedResearchRoundV1)
                    and _prepared_has_untried_candidate_local_innovation(
                        record.result.prepared,
                        attempt_budget=prior_attempt_budget,
                    )
                )
                if (
                    self._state.digest == record.state_after.digest
                    and after_generation == before_generation + 1
                    and (
                        generation_negative_advanced
                        or not resume_existing_observation
                    )
                    and not untried_candidate_local_innovation
                ):
                    # Candidate-local exhaustion is scientific input to a new
                    # Provider generation in the same logical round.  Keep the
                    # updated Context instead of replaying the exhausted slate.
                    self._state = record.state_after
                    self.memory_writer._head = record.state_after.search_memory_head
                    continuing_discovery_generation = True
                    continuing_generation_record = record
                else:
                    # A partial generation or shared-infrastructure stop must
                    # resume its existing prepared slate and physical manifest.
                    self._state = record.state_before
                    self.memory_writer._head = record.state_before.search_memory_head
                    resuming_incomplete = True
            else:
                if index == self._state.next_round_index:
                    if record.state_before.digest != self._state.digest:
                        raise CampaignError("round checkpoint predecessor state drift")
                    self._state = record.state_after
                    self.memory_writer._head = record.state_after.search_memory_head
                    self._persist_state()
                return record
        if index != expected:
            raise CampaignError(
                f"campaign is at round {expected}; cannot run round {index}"
            )
        if (
            self.round_trace_path(index).is_file()
            and not resuming_incomplete
            and not continuing_discovery_generation
        ):
            raise CampaignError("round trace exists without its checkpoint")

        inputs = self._round_inputs()
        configured_attempt_budget = (
            _configured_attempt_budget(inputs) if inputs.attempt_scheduler else None
        )
        if resuming_incomplete and not inputs.attempt_scheduler:
            raise CampaignError(
                "an incomplete scheduler round must resume with attempt_scheduler=True"
            )
        opportunity_ref = _opportunity_ref(self._state)
        state_before = self._state
        if (
            continuing_discovery_generation
            and _discovery_generation(self._state.context)
            >= MAX_DISCOVERY_GENERATIONS_PER_ROUND
        ):
            prior = continuing_generation_record
            if (
                prior is None
                or not inputs.feedback_proposal_generation_exhausted
                or prior.result.incomplete_reason
                != "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
                or prior.result.attempts
                or prior.result.has_metric_bearing_attempt
            ):
                raise CampaignError(
                    "exhausted discovery generation is not a sealed no-binding "
                    "checkpoint"
                )
            current_generation = _discovery_generation(self._state.context)
            generation_negatives = tuple(
                item
                for item in self._state.context.scientific_memory.get(
                    "generation_negatives", ()
                )
                if isinstance(item, Mapping)
                and item.get("round_index") == index
            )
            if (
                tuple(item.get("discovery_generation") for item in generation_negatives)
                != tuple(range(current_generation))
                or any(
                    item.get("counts_as_candidate_attempt") is not False
                    or item.get("counts_as_metric_worker") is not False
                    for item in generation_negatives
                )
            ):
                raise CampaignError(
                    "exhausted discovery generation evidence is not contiguous"
                )
            generation_negative = (
                self._successful_projection_generation_negative(prior.result)
                or self._preimplementation_generation_negative(prior.result)
                or self._terminal_physical_generation_negative(prior.result)
            )
            if generation_negative is None:
                raise CampaignError(
                    "exhausted no-binding checkpoint lacks typed generation evidence"
                )
            replayed_held = self._hold_state_after_incomplete_attempts(
                prior.state_before,
                prior.result,
                generation_negative=generation_negative,
            )
            if replayed_held.digest != self._state.digest:
                raise CampaignError(
                    "exhausted no-binding checkpoint cannot be replayed exactly"
                )
            result = replace(
                prior.result,
                incomplete_reason=(
                    "ROUND_DISCOVERY_GENERATION_BUDGET_EXHAUSTED_"
                    "NO_LEGAL_BINDING"
                ),
            )
            state_after = self._close_exhausted_no_metric_state(
                prior.state_before,
                result,
                generation_negative=generation_negative,
            )
            if self.post_round_state_transition is not None:
                state_after = self.post_round_state_transition(
                    state_after,
                    result,
                    index,
                    "TYPED_FAILURE_NO_METRIC",
                )
                if not isinstance(state_after, CampaignState):
                    raise CampaignError(
                        "post-round state transition must return CampaignState"
                    )
            return self._seal_record(
                CampaignRoundRecord(
                    round_index=index,
                    opportunity_ref=prior.opportunity_ref,
                    status="TYPED_FAILURE_NO_METRIC",
                    state_before=prior.state_before,
                    result=result,
                    state_after=state_after,
                )
            )
        prior_slot_failures = (
            self._current_round_candidate_failure_summaries(state_before)
            if inputs.attempt_scheduler
            else ()
        )
        prior_budget_attempts = (
            self._current_round_budget_attempt_summaries(state_before)
            if inputs.attempt_scheduler
            else ()
        )
        prior_slot_training_consumed = any(
            self._is_slot_training_consumed_summary(summary)
            for summary in prior_slot_failures
        )
        remaining_attempt_budget = (
            0
            if prior_slot_training_consumed
            else max(0, int(configured_attempt_budget) - len(prior_budget_attempts))
            if configured_attempt_budget is not None
            else None
        )
        if inputs.attempt_scheduler and remaining_attempt_budget == 0:
            # Campaign is the fixed-slot authority. Once every candidate outcome
            # is durably accounted, a later supervisor call must observe the
            # sealed no-progress boundary rather than starting physical work with
            # an invalid zero-sized runtime budget.
            accounted_slot_failures = tuple(
                prior_slot_failures[: int(configured_attempt_budget or 0)]
            )
            terminal_no_metric_reason = (
                "ROUND_SLOT_TRAINING_CONSUMED_NO_METRIC"
                if prior_slot_training_consumed
                else "ROUND_ATTEMPT_BUDGET_EXHAUSTED"
            )
            if checkpoint_path.is_file():
                exhausted_record = self._load_round(index)
                if (
                    exhausted_record.status == "INCOMPLETE"
                    and exhausted_record.result.attempt_scheduler_enabled
                    and exhausted_record.state_after.digest == self._state.digest
                    and configured_attempt_budget is not None
                    and self._can_close_exhausted_no_metric_slot(
                        inputs,
                        replace(
                            exhausted_record.result,
                            attempts=(),
                            incomplete_reason=terminal_no_metric_reason,
                        ),
                        configured_attempt_budget,
                        slot_failure_summaries=accounted_slot_failures,
                        budget_attempt_summaries=tuple(
                            prior_budget_attempts[
                                : int(configured_attempt_budget or 0)
                            ]
                        ),
                    )
                ):
                    result = replace(
                        exhausted_record.result,
                        attempts=(),
                        incomplete_reason=terminal_no_metric_reason,
                    )
                    manifest = self._load_attempt_manifest(
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        state_digest=state_before.digest,
                        context_digest=state_before.context.digest,
                        profile_digest=state_before.active_profile.profile_digest,
                        attempt_budget=configured_attempt_budget,
                    )
                    self._write_attempt_manifest(
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        state_digest=state_before.digest,
                        context_digest=state_before.context.digest,
                        profile_digest=state_before.active_profile.profile_digest,
                        attempt_budget=configured_attempt_budget,
                        attempts=tuple(manifest.get("attempts", ())),
                        status="TYPED_FAILURE_NO_METRIC",
                        metric_bearing_attempt_index=None,
                        incomplete_reason=terminal_no_metric_reason,
                        evidence_pre_trace=result.evidence_pre_trace,
                        evidence_post_trace=result.evidence_post_trace,
                    )
                    state_after = self._close_exhausted_no_metric_state(
                        state_before,
                        result,
                        slot_failure_summaries=accounted_slot_failures,
                        budget_attempt_summaries=tuple(
                            prior_budget_attempts[
                                : int(configured_attempt_budget or 0)
                            ]
                        ),
                    )
                    result, state_after = (
                        self._attach_candidate_no_metric_interpretation(
                            state_before,
                            result,
                            state_after,
                            accounted_slot_failures,
                        )
                    )
                    if self.post_round_state_transition is not None:
                        state_after = self.post_round_state_transition(
                            state_after,
                            result,
                            index,
                            "TYPED_FAILURE_NO_METRIC",
                        )
                        if not isinstance(state_after, CampaignState):
                            raise CampaignError(
                                "post-round state transition must return CampaignState"
                            )
                        result, state_after = (
                            self._attach_candidate_no_metric_interpretation(
                                state_before,
                                result,
                                state_after,
                                accounted_slot_failures,
                            )
                        )
                    return self._seal_record(
                        CampaignRoundRecord(
                            round_index=index,
                            opportunity_ref=opportunity_ref,
                            status="TYPED_FAILURE_NO_METRIC",
                            state_before=state_before,
                            result=result,
                            state_after=state_after,
                        )
                    )
            raise CampaignError(
                "candidate slot budget is exhausted without a durable "
                "incomplete checkpoint"
            )
        started = canonical_value(
            {
                "schema": "recclaw.research-line.campaign-round-started.v1",
                "round_index": index,
                **({"research_window": inputs.budget_snapshot["research_window"]}
                   if "research_window" in inputs.budget_snapshot else {}),
                "opportunity_ref": opportunity_ref,
                "state_digest": self._state.digest,
                "context_digest": self._state.context.digest,
                "profile_digest": self._state.active_profile.profile_digest,
            }
        )
        started_path = self._started_path(index)
        started_exists = started_path.exists()
        if started_path.exists():
            started_value = _read_json(started_path)
            # Resource facts belong to the already-started Research input.
            # A retry keeps that snapshot; a legacy start keeps its absence.
            frozen_budget = dict(inputs.budget_snapshot)
            frozen_budget.pop("research_window", None)
            if "research_window" in started_value:
                frozen_budget["research_window"] = started_value["research_window"]
            inputs = replace(inputs, budget_snapshot=canonical_value(frozen_budget))
            if started_value.get("opportunity_ref") != opportunity_ref:
                raise CampaignError(
                    "round start marker has a different opportunity identity"
                )
            manifest = self._load_attempt_manifest(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=configured_attempt_budget,
            )
            if (
                not manifest["attempts"]
                and not resuming_incomplete
                and not inputs.attempt_scheduler
            ):
                return self.record_missing_round(
                    index,
                    reason="round was interrupted before a durable physical observation",
                    failure_code="INTERRUPTED_BEFORE_DURABLE_PHYSICAL_OBSERVATION",
                )
        else:
            _write_once(started_path, canonical_json_bytes(started) + b"\n")
            manifest = self._load_attempt_manifest(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=self._state.digest,
                context_digest=self._state.context.digest,
                profile_digest=self._state.active_profile.profile_digest,
                attempt_budget=configured_attempt_budget,
            )

        prepared_round: PreparedResearchRoundV1 | None = None
        if inputs.attempt_scheduler:
            prepared_round = self._load_prepared_round(
                round_index=index,
                opportunity_ref=opportunity_ref,
                state_digest=state_before.digest,
                context_digest=state_before.context.digest,
                profile_ref=state_before.active_profile.profile_ref,
                profile_digest=state_before.active_profile.profile_digest,
                attempt_budget=int(manifest["attempt_budget"]),
                budget_snapshot=inputs.budget_snapshot,
            )
            if resuming_incomplete:
                checkpoint_prepared = record.result.prepared
                if resume_existing_observation or prepared_round is None:
                    prepared_round = self._validate_prepared_round(
                        prepared=checkpoint_prepared,
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        state_before=state_before,
                        attempt_budget=int(manifest["attempt_budget"]),
                        budget_snapshot=inputs.budget_snapshot,
                    )
                elif isinstance(checkpoint_prepared, PreparedResearchRoundV1):
                    if (
                        checkpoint_prepared.digest != prepared_round.digest
                        and not self._prepared_round_has_unfinished_provider_failure(
                            checkpoint_prepared,
                            self.producer,
                        )
                        and not self._prepared_round_has_recoverable_resource_failure(
                            checkpoint_prepared
                        )
                        and not _prepared_has_external_implementation_failure(checkpoint_prepared)
                        and not self._prepared_round_is_monotonic_candidate_progression(
                            checkpoint_prepared, prepared_round,
                            attempt_budget=int(manifest["attempt_budget"]),
                        )
                    ):
                        raise CampaignError(
                            "incomplete round prepared checkpoint digest drift"
                        )
            if (
                resuming_incomplete
                and inputs.close_exhausted_no_metric_slot
                and incomplete_resume_record is not None
            ):
                recovered_result = self._attach_observation_identities(
                    replace(
                        incomplete_resume_record.result,
                        prepared=prepared_round,
                    ),
                    manifest,
                )
                terminal_negative = self._terminal_physical_generation_negative(
                    recovered_result,
                    allow_untried_prepared_bindings=True,
                )
                if terminal_negative is not None:
                    result = recovered_result
                    state_after = self._hold_state_after_incomplete_attempts(
                        state_before,
                        result,
                        generation_negative=terminal_negative,
                    )
                    if state_after.digest == state_before.digest:
                        raise CampaignError(
                            "terminal physical recovery did not advance generation"
                        )
                    if self.post_round_state_transition is not None:
                        state_after = self.post_round_state_transition(
                            state_after,
                            result,
                            index,
                            "INCOMPLETE",
                        )
                        if not isinstance(state_after, CampaignState):
                            raise CampaignError(
                                "post-round state transition must return CampaignState"
                            )
                    return self._seal_record(
                        CampaignRoundRecord(
                            round_index=index,
                            opportunity_ref=opportunity_ref,
                            status="INCOMPLETE",
                            state_before=state_before,
                            result=result,
                            state_after=state_after,
                        )
                    )
            if prepared_round is None and manifest["attempts"]:
                raise CampaignError(
                    "scheduler round has no prepared checkpoint for safe recovery"
                )
            superseded_start = (
                self._engineering_superseded_start_marker(
                    round_index=index,
                    manifest=manifest,
                    engineering_source_identity=(
                        inputs.engineering_source_identity
                    ),
                )
                if prepared_round is not None
                else None
            )
            if prepared_round is not None and superseded_start is not None:
                started_path, start_marker = superseded_start
                generation_negative = canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line."
                            "discovery-generation-negative.v1"
                        ),
                        "reason_code": "ENGINEERING_SOURCE_ATTEMPT_SUPERSEDED",
                        "prepared_checkpoint_ref": str(
                            self._prepared_round_path(index)
                        ),
                        "prepared_digest": prepared_round.digest,
                        "physical_started_ref": str(started_path),
                        "physical_started_digest": sha256_digest(start_marker),
                        "prior_engineering_source_identity": start_marker.get(
                            "engineering_source_identity"
                        ),
                        "required_engineering_source_identity": (
                            inputs.engineering_source_identity
                        ),
                    }
                )
                result = ResearchRoundResult(
                    context=state_before.context,
                    active_profile=state_before.active_profile,
                    producer_outcomes=prepared_round.producer_outcomes,
                    carryover_outcomes=prepared_round.carryover_outcomes,
                    resolutions=prepared_round.resolutions,
                    deferred_innovation_outcomes=(
                        prepared_round.deferred_innovation_outcomes
                    ),
                    deferred_search_outcomes=(
                        prepared_round.deferred_search_outcomes
                    ),
                    search_acquisition=None,
                    innovation=None,
                    selected_outcome=None,
                    execution_recipe=None,
                    candidate_run=None,
                    interpretation=None,
                    provider_traces=prepared_round.provider_traces,
                    attempts=(),
                    metric_bearing_attempt_index=None,
                    attempt_scheduler_enabled=True,
                    incomplete_reason=(
                        "ROUND_ENGINEERING_SOURCE_ATTEMPT_SUPERSEDED"
                    ),
                    prepared=None,
                )
                state_after = self._hold_state_after_incomplete_attempts(
                    state_before,
                    result,
                    generation_negative=generation_negative,
                )
                if state_after.digest == state_before.digest:
                    raise CampaignError(
                        "engineering source supersession did not advance generation"
                    )
                return self._seal_record(
                    CampaignRoundRecord(
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        status="INCOMPLETE",
                        state_before=state_before,
                        result=result,
                        state_after=state_after,
                    )
                )
            if (
                prepared_round is not None
                and self._prepared_context_projection_status(
                    round_index=index,
                    context=state_before.context,
                    prepared=prepared_round,
                )
                == "LEGACY"
                and not self._prepared_round_has_committed_main_execution(
                    round_index=index,
                    prepared=prepared_round,
                    manifest=manifest,
                )
            ):
                generation_negative = canonical_value(
                    {
                        "schema": (
                            "recclaw.research-line."
                            "discovery-generation-negative.v1"
                        ),
                        "reason_code": "PRE_REPAIR_CONTEXT_SUPERSEDED",
                        "prepared_checkpoint_ref": str(
                            self._prepared_round_path(index)
                        ),
                        "prepared_digest": prepared_round.digest,
                        "provider_traces_digest": sha256_digest(
                            prepared_round.provider_traces
                        ),
                        "producer_outcomes_digest": sha256_digest(
                            tuple(
                                outcome.to_dict()
                                for outcome in prepared_round.producer_outcomes
                            )
                        ),
                        "prior_projection_version": None,
                        "required_projection_version": (
                            RESEARCH_CONTEXT_PROJECTION_VERSION
                        ),
                        "required_effect_feedback_projection_digest": (
                            _research_context_projection_digest(
                                state_before.context
                            )
                        ),
                    }
                )
                result = ResearchRoundResult(
                    context=state_before.context,
                    active_profile=state_before.active_profile,
                    producer_outcomes=prepared_round.producer_outcomes,
                    carryover_outcomes=prepared_round.carryover_outcomes,
                    resolutions=prepared_round.resolutions,
                    deferred_innovation_outcomes=(
                        prepared_round.deferred_innovation_outcomes
                    ),
                    deferred_search_outcomes=(
                        prepared_round.deferred_search_outcomes
                    ),
                    search_acquisition=None,
                    innovation=None,
                    selected_outcome=None,
                    execution_recipe=None,
                    candidate_run=None,
                    interpretation=None,
                    provider_traces=prepared_round.provider_traces,
                    attempts=(),
                    metric_bearing_attempt_index=None,
                    attempt_scheduler_enabled=True,
                    incomplete_reason="ROUND_PRE_REPAIR_CONTEXT_SUPERSEDED",
                    # The immutable generation-specific prepared checkpoint is
                    # retained on disk and referenced above.  Excluding it here
                    # prevents its unexecuted bindings from blocking the fresh
                    # generation transition.
                    prepared=None,
                )
                self._write_attempt_manifest(
                    round_index=index,
                    opportunity_ref=opportunity_ref,
                    state_digest=state_before.digest,
                    context_digest=state_before.context.digest,
                    profile_digest=state_before.active_profile.profile_digest,
                    attempt_budget=int(manifest["attempt_budget"]),
                    attempts=tuple(manifest.get("attempts", ())),
                    status="INCOMPLETE",
                    metric_bearing_attempt_index=None,
                    incomplete_reason=result.incomplete_reason,
                )
                state_after = self._hold_state_after_incomplete_attempts(
                    state_before,
                    result,
                    generation_negative=generation_negative,
                )
                if state_after.digest == state_before.digest:
                    raise CampaignError(
                        "superseded prepared context did not advance generation"
                    )
                if self.post_round_state_transition is not None:
                    state_after = self.post_round_state_transition(
                        state_after,
                        result,
                        index,
                        "INCOMPLETE",
                    )
                    if not isinstance(state_after, CampaignState):
                        raise CampaignError(
                            "post-round state transition must return CampaignState"
                        )
                return self._seal_record(
                    CampaignRoundRecord(
                        round_index=index,
                        opportunity_ref=opportunity_ref,
                        status="INCOMPLETE",
                        state_before=state_before,
                        result=result,
                        state_after=state_after,
                    )
                )
        partial_prebinding_checkpoint = (
            prepared_round is not None
            and (
                (
                    isinstance(
                        getattr(prepared_round, "prebinding_retry", None), Mapping
                    )
                    and not prepared_round.resolutions
                    and not prepared_round.search_bindings
                )
                or self._prepared_round_has_unfinished_provider_failure(
                    prepared_round,
                    self.producer,
                )
                or self._prepared_round_has_recoverable_resource_failure(
                    prepared_round
                )
                or _prepared_has_external_implementation_failure(prepared_round)
                or _prepared_has_untried_candidate_local_innovation(
                    prepared_round,
                    attempt_budget=int(remaining_attempt_budget),
                )
            )
        )
        transient_prepared_checkpoint = bool(
            prepared_round is not None
            and self._prepared_round_has_unfinished_provider_failure(
                prepared_round,
                self.producer,
            )
        )
        if inputs.attempt_scheduler and (
            (
                prepared_round is None
                and not manifest["attempts"]
            )
            or partial_prebinding_checkpoint
        ):
            # The callback below is the only path that may create this
            # scheduler-only checkpoint, and it runs before the first runner.
            prepare_callback: Callable[[PreparedResearchRoundV1], None] | None = (
                lambda prepared: self._persist_prepared_round(
                    round_index=index,
                    opportunity_ref=opportunity_ref,
                    state_before=state_before,
                    attempt_budget=int(manifest["attempt_budget"]),
                    budget_snapshot=inputs.budget_snapshot,
                    prepared=prepared,
                )
            )
        else:
            prepare_callback = None
        round_memory = self._round_memory_writer()
        round_carryover_proposals = tuple(self._state.carryover_proposals)
        if (
            inputs.attempt_scheduler
            and inputs.bootstrap_fixed_candidates
            and not inputs.portfolio_candidates
        ):
            existing_mechanisms = tuple(
                proposal.mechanism_id for proposal in round_carryover_proposals
            )
            remaining_slots = max(
                0,
                int(manifest["attempt_budget"])
                - len(round_carryover_proposals),
            )
            materialized = (
                bootstrap_search_pool(
                    self._state.context,
                    self._state.active_profile,
                    self._state.policy,
                    max_proposals=remaining_slots,
                    excluded_mechanism_ids=existing_mechanisms,
                )
                if remaining_slots
                else ()
            )
            round_carryover_proposals = tuple(
                item
                for item in _unique_by_id(
                    (*round_carryover_proposals, *materialized)
                )
                if isinstance(item, CandidateProposalV4)
            )
        deferred_backlog = tuple(
            getattr(self._state, "deferred_innovation_backlog", ())
        )
        round_producer: ResearchProducer = self.producer
        if prepared_round is None or transient_prepared_checkpoint:
            round_producer = _producer_for_round(
                self.producer,
                deferred_backlog,
                round_role=inputs.round_role,
            )
        if (
            started_exists
            and inputs.attempt_scheduler
            and prepared_round is None
            and not manifest["attempts"]
        ):
            configure_started_replay = getattr(
                round_producer,
                "configure_started_round_provider_replay",
                None,
            )
            if callable(configure_started_replay):
                configure_started_replay(state_before.context.context_ref)
        result = run_research_round(
            context=self._state.context,
            active_profile=self._state.active_profile,
            producer=round_producer,
            producer_bindings=inputs.producer_bindings,
            resolver_environment=inputs.resolver_environment,
            carryover_proposals=round_carryover_proposals,
            carryover_open_candidates=self._state.carryover_open_candidates,
            budget_snapshot=inputs.budget_snapshot,
            router=inputs.router,
            policy=self._state.policy,
            memory_writer=round_memory,
            runner=self._runner_for_round(
                index,
                opportunity_ref,
                manifest,
                observation_seed=inputs.observation_seed,
                engineering_source_identity=inputs.engineering_source_identity,
            ),
            incumbent_observation=self._state.incumbent_observation,
            metric_contract_digest=inputs.metric_contract_digest,
            observation_seed=inputs.observation_seed,
            next_discriminative_test=inputs.next_discriminative_test,
            confirmation_seed=inputs.confirmation_seed,
            qualified_execution_by_capability=inputs.qualified_execution_by_capability,
            research_profile_source=inputs.research_profile_source,
            candidate_handoff_factory=inputs.candidate_handoff_factory,
            candidate_root_by_capability=inputs.candidate_root_by_capability,
            resource_profile_by_capability=inputs.resource_profile_by_capability,
            innovation_inputs=inputs.innovation_inputs,
            meta_research_inputs=inputs.meta_research_inputs,
            attempt_scheduler=inputs.attempt_scheduler,
            max_attempts_per_round=(
                int(remaining_attempt_budget)
                if inputs.attempt_scheduler
                else inputs.max_attempts_per_round
            ),
            prebinding_token_ceiling_retry=(
                inputs.prebinding_token_ceiling_retry
            ),
            recovered_attempts=(
                tuple(manifest["attempts"]) if inputs.attempt_scheduler else ()
            ),
            recovered_evidence_pre_trace=(
                tuple(manifest.get("evidence_pre_trace", ()))
                if inputs.attempt_scheduler
                else ()
            ),
            prepared_round=prepared_round,
            on_prepared=prepare_callback,
            portfolio_candidates=(
                inputs.portfolio_candidates if inputs.portfolio_candidates else None
            ),
            evidence_port=inputs.evidence_port,
            observation_seed_schedule=inputs.observation_seed_schedule,
            evaluator=inputs.evaluator,
            split=inputs.split,
            frozen_profile_ref=inputs.frozen_profile_ref,
            round_role=inputs.round_role,
            search_space_adapter=(
                inputs.search_space_adapter or self.search_space_adapter
            ),
            feedback_proposal_generation_exhausted=(
                inputs.feedback_proposal_generation_exhausted
            ),
        )
        if inputs.attempt_scheduler:
            result = self._attach_observation_identities(result, manifest)
        if result.incomplete_reason in {
            "ROUND_IMPLEMENTATION_PROVIDER_UNAVAILABLE",
            "ROUND_IMPLEMENTATION_ADMISSION_PAUSED",
        }:
            # Preserve the exact opportunity, context and budget. Infrastructure
            # failure is not a candidate attempt, scientific negative or result.
            self._persist_attempt_result_manifest(
                round_index=index, opportunity_ref=opportunity_ref,
                state_before=state_before, manifest=manifest, result=result,
                status="INCOMPLETE", prebinding_summaries=(),
            )
            return self._seal_record(CampaignRoundRecord(
                round_index=index, opportunity_ref=opportunity_ref,
                status="INCOMPLETE", state_before=state_before,
                result=result, state_after=state_before,
            ))
        projection_generation_negative = (
            self._successful_projection_generation_negative(result)
        )
        physical_budget_attempts = tuple(
            _compact_round_attempt_summary(
                attempt,
                round_index=state_before.next_round_index,
            )
            for attempt in result.attempts
            if not attempt.metric_bearing
            and attempt.failure_scope
            in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
            and attempt.failure_detail is not None
        )
        current_prebinding_attempts = (
            self._prebinding_candidate_failure_summaries(result)
        )
        current_budget_attempts = (
            *current_prebinding_attempts,
            *physical_budget_attempts,
        )
        current_slot_failures = tuple(
            summary
            for summary in current_budget_attempts
            if (
                self._is_candidate_local_prebinding_failure(summary)
                if summary.get("schema")
                == (
                    "recclaw.research-line."
                    "discovery-prebinding-attempt-summary.v1"
                )
                else summary.get("failure_scope")
                in {"CANDIDATE_LOCAL", "LINEAGE_COMPUTE_PATTERN"}
                and summary.get("failure_detail_digest") is not None
            )
        )
        prior_keys = {
            self._candidate_failure_key(summary)
            for summary in prior_slot_failures
        }
        counted_current_failures: list[Mapping[str, Any]] = []
        for summary in current_slot_failures:
            key = self._candidate_failure_key(summary)
            if (
                not all(key)
                or key in prior_keys
                or any(
                    self._candidate_failure_key(item) == key
                    for item in counted_current_failures
                )
            ):
                continue
            if (
                remaining_attempt_budget is not None
                and len(counted_current_failures) >= remaining_attempt_budget
            ):
                break
            counted_current_failures.append(summary)
        counted_current = tuple(counted_current_failures)
        slot_failures = (*prior_slot_failures, *counted_current)
        prior_budget_keys = {
            self._candidate_failure_key(summary)
            for summary in prior_budget_attempts
        }
        counted_current_budget: list[Mapping[str, Any]] = []
        for summary in current_budget_attempts:
            key = self._candidate_failure_key(summary)
            if (
                not all(key)
                or not self._budget_attempt_consumed(summary)
                or key in prior_budget_keys
                or any(
                    self._candidate_failure_key(item) == key
                    for item in counted_current_budget
                )
            ):
                continue
            if (
                remaining_attempt_budget is not None
                and len(counted_current_budget) >= remaining_attempt_budget
            ):
                break
            counted_current_budget.append(summary)
        slot_budget_attempts = (
            *prior_budget_attempts,
            *tuple(counted_current_budget),
        )
        if (
            inputs.attempt_scheduler
            and configured_attempt_budget is not None
            and len(slot_budget_attempts) == configured_attempt_budget
            and not result.has_metric_bearing_attempt
        ):
            result = replace(
                result,
                incomplete_reason="ROUND_ATTEMPT_BUDGET_EXHAUSTED",
            )
        generation_negative = (
            projection_generation_negative
            or self._preimplementation_generation_negative(result)
            or self._terminal_physical_generation_negative(result)
            or (
                self._unreplayable_started_generation_negative(result)
                if started_exists and not manifest["attempts"]
                else None
            )
        )
        discovery_generation_exhausted = bool(
            inputs.feedback_proposal_generation_exhausted
            and result.incomplete_reason
            == "ROUND_ATTEMPT_NO_LEGAL_SEARCH_BINDING"
            and not result.attempts
            and not result.has_metric_bearing_attempt
            and generation_negative is not None
        )
        if discovery_generation_exhausted:
            result = replace(
                result,
                incomplete_reason=(
                    "ROUND_DISCOVERY_GENERATION_BUDGET_EXHAUSTED_"
                    "NO_LEGAL_BINDING"
                ),
            )
        if (
            not result.has_metric_bearing_attempt
            and result.incomplete_reason is None
        ):
            result = replace(
                result,
                incomplete_reason="ROUND_NO_UNIQUE_METRIC_BEARING_RESULT",
            )
        close_no_metric_slot = (
            discovery_generation_exhausted
            or self._can_close_exhausted_no_metric_slot(
                inputs,
                result,
                configured_attempt_budget,
                slot_failure_summaries=(
                    tuple(slot_failures) if slot_failures else None
                ),
                budget_attempt_summaries=(
                    tuple(slot_budget_attempts)
                    if slot_budget_attempts
                    else None
                ),
            )
        )
        status = (
            "TYPED_FAILURE_NO_METRIC"
            if close_no_metric_slot
            else self._status(result)
        )
        self._persist_attempt_result_manifest(
            round_index=index,
            opportunity_ref=opportunity_ref,
            state_before=state_before,
            manifest=manifest,
            result=result,
            status=status,
            prebinding_summaries=current_prebinding_attempts,
        )
        state_after = (
            self._close_exhausted_no_metric_state(
                state_before,
                result,
                prebinding_summaries=current_prebinding_attempts,
                slot_failure_summaries=(
                    tuple(slot_failures) if slot_failures else None
                ),
                budget_attempt_summaries=(
                    tuple(slot_budget_attempts)
                    if slot_budget_attempts
                    else None
                ),
                generation_negative=(
                    generation_negative if discovery_generation_exhausted else None
                ),
            )
            if close_no_metric_slot
            else (
                self._hold_state_after_incomplete_attempts(
                    state_before,
                    result,
                    prebinding_summaries=current_prebinding_attempts,
                    generation_negative=generation_negative,
                )
                if not result.has_metric_bearing_attempt
                else self._advance_state(
                    state_before, result,
                    consume_verification_slot=inputs.round_role == "VERIFICATION",
                )
            )
        )
        if close_no_metric_slot and slot_failures:
            result, state_after = self._attach_candidate_no_metric_interpretation(
                state_before,
                result,
                state_after,
                tuple(slot_failures),
            )
        if self.post_round_state_transition is not None:
            state_after = self.post_round_state_transition(
                state_after,
                result,
                index,
                status,
            )
            if not isinstance(state_after, CampaignState):
                raise CampaignError(
                    "post-round state transition must return CampaignState"
                )
            if close_no_metric_slot and slot_failures:
                result, state_after = self._attach_candidate_no_metric_interpretation(
                    state_before,
                    result,
                    state_after,
                    tuple(slot_failures),
                )
        record = CampaignRoundRecord(
            round_index=index,
            opportunity_ref=opportunity_ref,
            status=status,
            state_before=state_before,
            result=result,
            state_after=state_after,
        )
        return self._seal_record(record)


    def can_resume_existing_observation_rebind(self) -> bool:
        """Whether the exhausted current generation can close an exact task."""

        checkpoint_path = self.round_checkpoint_path(self._state.next_round_index)
        if not checkpoint_path.is_file():
            return False
        record = self._load_round(self._state.next_round_index)
        return bool(
            record.status == "INCOMPLETE"
            and record.result.attempt_scheduler_enabled
            and self._state.digest == record.state_after.digest
            and self._can_resume_existing_observation_rebind(record)
        )

    def _can_resume_existing_observation_rebind(
        self,
        record: CampaignRoundRecord,
    ) -> bool:
        if record.result.incomplete_reason == "ROUND_EXISTING_OBSERVATION_REBOUND":
            return False
        prepared = record.result.prepared
        innovation = record.result.innovation
        if not isinstance(prepared, PreparedResearchRoundV1):
            return False
        pending_task = _discovery_feedback_task(
            record.state_before.context,
            observation_seed=prepared.observation_seed,
        )
        if pending_task is None:
            return False
        task_record = pending_task.get("task_record")
        if (
            not isinstance(task_record, Mapping)
            or getattr(
                task_record.get("operation"),
                "value",
                task_record.get("operation"),
            )
            not in {"MATCHED_CONTROL", "MECHANISM_OFF"}
        ):
            return False
        inputs = None
        if innovation is None:
            adapter = self.search_space_adapter or _default_search_space_adapter()
        else:
            inputs = self._round_inputs()
            adapter = (
                inputs.search_space_adapter
                or self.search_space_adapter
                or _default_search_space_adapter()
            )
        exact_outcomes = tuple(
            outcome.digest
            for outcome, resolution in prepared.resolutions
            if resolution is not None
            and getattr(resolution.resolution, "value", resolution.resolution)
            == "INNOVATION_REQUIRED"
            and getattr(
                _resolve_confirmation(
                    adapter,
                    pending_task=pending_task,
                    primary_binding=outcome,
                    next_discriminative_test=prepared.next_discriminative_test,
                ).kind,
                "value",
                None,
            )
            == "EXACT_BINDING"
        )
        if innovation is None:
            # A prepared slate persisted before an adapter repair can become
            # exact when re-evaluated even though the stale result never
            # reached acquisition.  Let the normal round path acquire it;
            # otherwise the generation ceiling blocks a now-valid binding.
            return bool(exact_outcomes)
        if exact_outcomes != (innovation.selected_outcome.digest,):
            return False
        assert inputs is not None
        evaluator = inputs.evaluator or COMMON_EVALUATOR
        split = inputs.split or COMMON_SPLIT
        return (
            _existing_exact_feedback_observation_rebind(
                context=record.state_before.context,
                pending_task=pending_task,
                innovation=innovation,
                exact_confirmation=True,
                evaluator=evaluator,
                split=split,
                observation_seed=prepared.observation_seed,
                metric_contract_digest=prepared.metric_contract_digest,
            )
            is not None
        )


__all__ = [
    "CampaignError",
    "CampaignRoundInputFactory",
    "CampaignRoundInputs",
    "CampaignRoundRecord",
    "CampaignVerificationRecord",
    "CampaignState",
    "MAX_DISCOVERY_GENERATIONS_PER_ROUND",
    "ResearchCampaign",
    "ResearchCampaignState",
]
