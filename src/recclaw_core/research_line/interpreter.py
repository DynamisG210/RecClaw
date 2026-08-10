"""Episode and diagnostic feedback bridge for the Research Line."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import Any, Mapping, Sequence

from recclaw_core.helix.scientific_attribution import (
    PromptFeedbackProjectionV2,
    ResearchTaskStatusV1,
    ResearchTaskTypeV1,
    ResearchTaskV1,
    SearchUtilityEventV2,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    DevelopmentalMechanismBeliefV1,
    SearchMemorySnapshotV1,
    SearchMemoryWriterV1,
    VersionedMetaPolicyUpdaterV1,
    VersionedResearchPolicyV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (
    EpisodeMemoryLaneV1,
    FrozenComparisonIdentityV1,
    ScientificEpisodeClosureV1,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode_adapter import (
    project_episode_to_mechanism_belief,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    EpisodeEvidenceClassV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)
from recclaw_core.research_line.interfaces import (
    BehaviorProjection,
    ProducerOutcome,
    ResearchContext,
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
    ResearchTaskStatusV2,
)


class ScientificInterpreterError(ValueError):
    """Raised when closed feedback cannot be bound to the next round."""


_SCIENTIFIC_FAILURES = frozenset(
    {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
        ResearchFailureClassV1.INCONCLUSIVE,
    }
)
_FAILURE_TAXONOMY = {
    ResearchFailureClassV1.NONE: "SCIENTIFIC_SUPPORTING_COMPARISON",
    ResearchFailureClassV1.MECHANISM: "SCIENTIFIC_MECHANISM_NEGATIVE",
    ResearchFailureClassV1.INCONCLUSIVE: "SCIENTIFIC_INCONCLUSIVE",
}
_MECHANISM_ATTRIBUTION = {
    ResearchFailureClassV1.NONE: "SUPPORTING_COMPARISON",
    ResearchFailureClassV1.MECHANISM: "MECHANISM_NEGATIVE",
    ResearchFailureClassV1.INCONCLUSIVE: "NOT_ADJUDICATED",
}
_NEXT_DEVELOPMENT_SEED = "NEXT_DEVELOPMENT_SEED"
_MEMORY_HISTORY_LIMIT = 64
_COMPARISON_IDENTITY_FIELDS = (
    "campaign_id",
    "context_ref",
    "context_digest",
    "executable_capability_ref",
    "executable_capability_digest",
    "executable_profile_ref",
    "executable_profile_digest",
    "experiment_binding_ref",
    "experiment_binding_digest",
    "comparator_ref",
    "comparator_digest",
    "protocol_ref",
    "protocol_digest",
)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ScientificInterpreterError(f"{field} must be normalized and non-empty")
    return value


def _digest(value: Any, *, field: str) -> str:
    try:
        return validate_sha256(value, field_name=field)
    except (TypeError, ValueError) as error:
        raise ScientificInterpreterError(str(error)) from error


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ScientificInterpreterError(f"{field} must be a mapping")
    return value


def _unit(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ScientificInterpreterError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ScientificInterpreterError(f"{field} must be numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ScientificInterpreterError(f"{field} must be finite and in [0,1]")
    return number


def _validate_context(
    *,
    episode: TypedResearchEpisodeV1 | None,
    identity: FrozenComparisonIdentityV1,
    context: ResearchContext,
    policy: VersionedResearchPolicyV1,
    policy_context: ResearchContext | None = None,
) -> None:
    if episode is not None and not isinstance(episode, TypedResearchEpisodeV1):
        raise ScientificInterpreterError("episode must be TypedResearchEpisodeV1")
    if not isinstance(identity, FrozenComparisonIdentityV1):
        raise ScientificInterpreterError(
            "comparison_identity must be FrozenComparisonIdentityV1"
        )
    if not isinstance(context, ResearchContext):
        raise ScientificInterpreterError("context must be ResearchContext")
    if not isinstance(policy, VersionedResearchPolicyV1):
        raise ScientificInterpreterError(
            "policy must be VersionedResearchPolicyV1"
        )
    context_identity = (
        context.campaign_id,
        context.context_ref,
        context.digest,
    )
    if (
        identity.campaign_id,
        identity.context_ref,
        identity.context_digest,
    ) != context_identity:
        raise ScientificInterpreterError("comparison identity is not bound to context")
    if episode is not None and (
        episode.campaign_id,
        episode.context_ref,
        episode.context_digest,
    ) != context_identity:
        raise ScientificInterpreterError("Episode context identity drift")
    profile = context.active_profile_ref, context.active_profile_digest
    if (identity.executable_profile_ref, identity.executable_profile_digest) != profile:
        raise ScientificInterpreterError("comparison profile identity drift")
    if episode is not None and (
        episode.executable_profile_ref,
        episode.executable_profile_digest,
    ) != profile:
        raise ScientificInterpreterError("Episode profile identity drift")
    protocol = context.protocol_ref, context.protocol_digest
    if (identity.protocol_ref, identity.protocol_digest) != protocol:
        raise ScientificInterpreterError("comparison protocol identity drift")
    if episode is not None and (episode.protocol_ref, episode.protocol_digest) != protocol:
        raise ScientificInterpreterError("Episode protocol identity drift")
    effective_policy_context = policy_context or context
    if canonical_value(effective_policy_context.policy) != canonical_value(
        policy.to_dict()
    ):
        raise ScientificInterpreterError(
            "context policy and supplied versioned policy are not the same input"
        )


def _validate_same_round_working_context(
    frozen_context: ResearchContext,
    working_context: ResearchContext,
) -> None:
    """Keep physical identity frozen while carrying in-round diagnostics.

    Engineering attempts may update resource/task/role memory and policy before
    the metric-bearing attempt, but they may not alter the scientific round,
    profile, protocol, goal, data context, or budget that bound the prepared
    Producer/Resolver/slate state.
    """

    if not isinstance(working_context, ResearchContext):
        raise ScientificInterpreterError("working context must be ResearchContext")
    frozen_fields = (
        "campaign_id",
        "round_index",
        "context_ref",
        "active_profile_ref",
        "active_profile_digest",
        "protocol_ref",
        "protocol_digest",
        "knowledge_base",
        "frozen_goal",
        "budget",
    )
    if any(
        canonical_value(getattr(frozen_context, field_name))
        != canonical_value(getattr(working_context, field_name))
        for field_name in frozen_fields
    ):
        raise ScientificInterpreterError(
            "in-round diagnostic working context changed a frozen round input"
        )


def _validate_closure(
    *,
    episode: TypedResearchEpisodeV1 | None,
    identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
) -> None:
    if not isinstance(closure, ScientificEpisodeClosureV1):
        raise ScientificInterpreterError(
            "closure must be ScientificEpisodeClosureV1"
        )
    if (
        closure.comparison_identity_ref != identity.identity_id
        or closure.comparison_identity_digest != identity.digest
    ):
        raise ScientificInterpreterError("closure comparison identity drift")
    if episode is None:
        diagnostic = (
            closure.failure_class not in _SCIENTIFIC_FAILURES
            and closure.engineering_diagnostic_allowed
            and closure.memory_lane is EpisodeMemoryLaneV1.ENGINEERING_DIAGNOSTIC
            and closure.evidence_class is EpisodeEvidenceClassV1.ENGINEERING_ONLY
            and closure.episode_ref is None
            and closure.episode_digest is None
            and closure.outcome_ref is None
            and closure.outcome_digest is None
            and closure.failure_detail_ref is not None
            and closure.failure_detail_digest is not None
        )
        if not diagnostic:
            raise ScientificInterpreterError(
                "diagnostic closure is not an engineering-only D0 closure"
            )
        return
    if episode.failure_class not in _SCIENTIFIC_FAILURES:
        raise ScientificInterpreterError(
            "engineering, resource, runtime, and provider failures are not Episodes"
        )
    if closure.failure_class is not episode.failure_class or (
        closure.episode_ref,
        closure.episode_digest,
        closure.outcome_ref,
        closure.outcome_digest,
        closure.evidence_class,
    ) != (
        episode.episode_id,
        episode.digest,
        episode.outcome_ref,
        episode.outcome_digest,
        episode.evidence_class,
    ):
        raise ScientificInterpreterError("closure does not bind the supplied Episode")
    if any(
        getattr(identity, field) != getattr(episode, field)
        for field in _COMPARISON_IDENTITY_FIELDS
    ):
        raise ScientificInterpreterError("comparison identity drift")
    if episode.failure_class is ResearchFailureClassV1.INCONCLUSIVE and (
        closure.memory_lane is not EpisodeMemoryLaneV1.NONE
        or closure.mechanism_memory_allowed
    ):
        raise ScientificInterpreterError(
            "INCONCLUSIVE Episodes cannot enter mechanism memory"
        )


def _producer_map(
    outcomes: Sequence[ProducerOutcome], context: ResearchContext
) -> dict[str, ProducerOutcome]:
    if isinstance(outcomes, (str, bytes)) or len(outcomes) != len(DISCOVERY_PRODUCERS):
        raise ScientificInterpreterError(
            "exactly one outcome is required for each of the four Producers"
        )
    result: dict[str, ProducerOutcome] = {}
    for outcome in outcomes:
        if not isinstance(outcome, ProducerOutcome):
            raise ScientificInterpreterError("producer_outcomes contain an invalid type")
        if outcome.producer_role in result:
            raise ScientificInterpreterError("producer_outcomes repeat a Producer role")
        if (outcome.context_ref, outcome.context_digest) != (
            context.context_ref,
            context.digest,
        ):
            raise ScientificInterpreterError(
                f"Producer outcome is not bound to {context.context_ref}"
            )
        result[outcome.producer_role] = outcome
    if set(result) != set(DISCOVERY_PRODUCERS):
        raise ScientificInterpreterError(
            "producer_outcomes must cover the frozen four-role portfolio"
        )
    return result


def _selected_has_delayed_provenance(
    selected: ProducerOutcome,
    outcomes: Mapping[str, ProducerOutcome],
) -> bool:
    """Recognize carryover lineage without crediting an unrelated fixture spec."""

    current = outcomes.get(selected.producer_role)
    if (
        current is not None
        and current.spec is not None
        and selected.spec is not None
        and current.spec.digest == selected.spec.digest
    ):
        return True
    if selected.source_proposal is not None:
        return True
    if selected.provenance:
        return selected.provenance.get("status") in {
            "PRODUCED",
            "CARRYOVER",
        }
    # Existing runtime carryover OpenSpecs have no source proposal yet, but
    # their resolver facts carry this explicit lineage marker.
    return "requested_current_semantics_digest" in selected.resolution_facts


def _route(
    metadata: Mapping[str, Any],
    outcomes: Mapping[str, ProducerOutcome],
    event: SearchUtilityEventV2,
    selected_outcome: ProducerOutcome | None,
) -> tuple[str, str, float, ProducerOutcome]:
    route = _mapping(metadata, field="route_metadata")
    trace = _digest(route.get("route_trace_digest"), field="route_trace_digest")
    role = _text(route.get("selected_producer_role"), field="selected_producer_role")
    if role not in outcomes:
        raise ScientificInterpreterError("route selected an unknown Producer role")
    candidate = _text(route.get("selected_candidate_id"), field="selected_candidate_id")
    semantic_digest = _digest(
        route.get("selected_candidate_semantic_digest"),
        field="selected_candidate_semantic_digest",
    )
    if candidate != event.candidate_id or semantic_digest != event.candidate_semantic_digest:
        raise ScientificInterpreterError("route and evaluator candidate identities differ")
    axis = route.get("selected_mechanism_axis")
    if axis is not None and axis != event.mechanism_axis:
        raise ScientificInterpreterError("route and evaluator mechanism axes differ")
    selected = selected_outcome or outcomes[role]
    if (
        selected.producer_role != role
        or selected.context_ref != outcomes[role].context_ref
        or selected.context_digest != outcomes[role].context_digest
        or selected.spec is None
    ):
        raise ScientificInterpreterError(
            "selected pool outcome is not bound to its Producer/context"
        )
    source = selected.source_proposal
    if source is not None and source.candidate_id != candidate:
        raise ScientificInterpreterError(
            "route candidate differs from the selected Producer proposal"
        )
    probability = _unit(
        route.get("required_selected_runnable_probability"),
        field="required_selected_runnable_probability",
    )
    return trace, role, probability, selected


def _runnable_probability(event: SearchUtilityEventV2) -> float:
    if event.runnable_observation == "RUNNABLE":
        return 1.0
    if event.runnable_observation == "NOT_RUNNABLE":
        return 0.0
    raise ScientificInterpreterError(
        "runnable_observation must be RUNNABLE or NOT_RUNNABLE"
    )


def _meta_aggregate(
    outcomes: Mapping[str, ProducerOutcome],
    selected_role: str,
    selected: ProducerOutcome,
    event: SearchUtilityEventV2,
    required_probability: float,
    policy: VersionedResearchPolicyV1,
    *,
    scientific_episode: bool,
) -> dict[str, Any]:
    useful_rates = dict(policy.producer_token_allocation)
    current_spec = outcomes[selected_role].spec
    selected_is_current_spec = (
        current_spec is not None
        and selected.spec is not None
        and current_spec.digest == selected.spec.digest
    )
    delayed_provenance = _selected_has_delayed_provenance(selected, outcomes)
    if (
        (selected_is_current_spec or delayed_provenance)
        and scientific_episode
        and event.comparator_delta != "NOT_AVAILABLE"
    ):
        effect = float(event.comparator_delta)
        useful_rates[selected_role] = max(
            0.0,
            min(1.0, useful_rates[selected_role] + effect),
        )
    return {
        "producer_useful_rates": useful_rates,
        "mechanism_axis_gaps": (event.mechanism_axis,) if scientific_episode else (),
        "calibration_error": abs(required_probability - _runnable_probability(event)),
    }


def _task_operation(
    *,
    route: Mapping[str, Any],
    event: SearchUtilityEventV2,
    episode: TypedResearchEpisodeV1 | None,
    required_seed_or_control: str,
) -> ResearchTaskOperationV2:
    raw_operation = route.get("task_operation", route.get("next_task_operation"))
    if raw_operation is not None:
        try:
            operation = ResearchTaskOperationV2(
                raw_operation.value if isinstance(raw_operation, Enum) else str(raw_operation)
            )
        except ValueError as error:
            raise ScientificInterpreterError(
                "task_operation is not a ResearchTaskOperationV2"
            ) from error
        if (
            operation is ResearchTaskOperationV2.NEW_SEED
            and required_seed_or_control == event.observation_seed
        ):
            return (
                ResearchTaskOperationV2.REPRODUCE
                if route.get("allow_same_seed_reproduce", False)
                else ResearchTaskOperationV2.MOVE_ON
            )
        return operation

    raw_type = route.get("next_task_type")
    if raw_type is None:
        # Resource/implementation failures need a repair task, not a repeat.
        if episode is None:
            return ResearchTaskOperationV2.REPAIR
        if required_seed_or_control == event.observation_seed:
            return (
                ResearchTaskOperationV2.REPRODUCE
                if route.get("allow_same_seed_reproduce", False)
                else ResearchTaskOperationV2.MOVE_ON
            )
        return ResearchTaskOperationV2.NEW_SEED
    raw_type = raw_type.value if isinstance(raw_type, Enum) else str(raw_type)
    mapped = {
        "VALIDATE_SAME_CANDIDATE": ResearchTaskOperationV2.NEW_SEED,
        "RUN_MATCHED_CONTROL": ResearchTaskOperationV2.MATCHED_CONTROL,
        "RUN_ABLATION": ResearchTaskOperationV2.MECHANISM_OFF,
        "REPAIR_IMPLEMENTATION": ResearchTaskOperationV2.REPAIR,
        "PROTOCOL_BRANCH_DIAGNOSTIC": ResearchTaskOperationV2.MOVE_ON,
    }
    try:
        operation = mapped[raw_type]
    except KeyError as error:
        raise ScientificInterpreterError(
            "next_task_type is not a supported legacy Research task type"
        ) from error
    if (
        operation is ResearchTaskOperationV2.NEW_SEED
        and required_seed_or_control == event.observation_seed
        and not route.get("allow_same_seed_reproduce", False)
    ):
        return (
            ResearchTaskOperationV2.REPRODUCE
            if route.get("allow_same_seed_reproduce", False)
            else ResearchTaskOperationV2.MOVE_ON
        )
    return operation


def _task_record(
    *,
    episode: TypedResearchEpisodeV1 | None,
    identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
    event: SearchUtilityEventV2,
    metadata: Mapping[str, Any],
    selected: ProducerOutcome,
    context: ResearchContext,
    next_test: str,
) -> ResearchTaskRecordV2:
    route = _mapping(metadata, field="route_metadata")
    source = selected.source_proposal
    supplied_program = route.get("mechanism_program")
    if source is not None:
        program = source.mechanism_program
        if supplied_program is not None and canonical_value(supplied_program) != canonical_value(program):
            raise ScientificInterpreterError("route mechanism program differs from proposal")
        parent = source.parent_candidate_id
    else:
        program = _mapping(supplied_program, field="mechanism_program")
        parent = route.get("parent_candidate_id")
    program_digest = sha256_digest(program)
    supplied_digest = route.get("mechanism_program_digest")
    if supplied_digest is not None and _digest(
        supplied_digest, field="mechanism_program_digest"
    ) != program_digest:
        raise ScientificInterpreterError("mechanism program digest does not match payload")
    if parent is not None:
        parent = _text(parent, field="parent_candidate_id")
    comparator = _text(
        route.get(
            "comparator_identity",
            episode.comparator_ref if episode is not None else identity.comparator_ref,
        ),
        field="comparator_identity",
    )
    required_seed = _text(
        route.get("required_seed_or_control", event.observation_seed),
        field="required_seed_or_control",
    )
    missing = route.get("missing_seed_count", 1)
    if isinstance(missing, bool) or not isinstance(missing, int) or missing < 0:
        raise ScientificInterpreterError("missing_seed_count must be non-negative")
    owner = route.get("owner_arm_instance_id")
    if owner is not None:
        owner = _text(owner, field="owner_arm_instance_id")
    priority = route.get("task_utility_priority")
    priority = (
        _unit(priority, field="task_utility_priority")
        if priority is not None
        else 0.5
        if event.comparator_delta == "NOT_AVAILABLE"
        else min(1.0, 0.5 + abs(float(event.comparator_delta)))
    )
    protocol_digest = (
        episode.protocol_digest if episode is not None else identity.protocol_digest
    )
    operation = _task_operation(
        route=route,
        event=event,
        episode=episode,
        required_seed_or_control=required_seed,
    )
    raw_evidence = route.get("evidence_present", ())
    if isinstance(raw_evidence, str):
        raw_evidence = (raw_evidence,)
    if not isinstance(raw_evidence, (tuple, list)):
        raise ScientificInterpreterError("evidence_present must be a sequence")
    evidence = tuple(str(item) for item in raw_evidence)
    if episode is not None and not evidence:
        evidence = (event.observation_seed,)
    deadline = route.get("task_deadline_round")
    if deadline is not None and (
        isinstance(deadline, bool) or not isinstance(deadline, int)
    ):
        raise ScientificInterpreterError("task_deadline_round must be an integer")
    provenance = selected.provenance
    provenance_digest = None
    if selected.source_proposal is not None:
        provenance_digest = selected.source_proposal.digest
    elif isinstance(provenance, Mapping):
        candidate_provenance_digest = provenance.get("spec_digest")
        if isinstance(candidate_provenance_digest, str):
            provenance_digest = candidate_provenance_digest
    metadata_payload = {
        "next_discriminative_test": next_test,
        "mechanism_axis": event.mechanism_axis,
        "capability_family": route.get(
            "capability_family",
            route.get("candidate_family", route.get("family", event.mechanism_axis)),
        ),
        "owner_arm_instance_id": owner,
        "legacy_task_type": route.get("next_task_type"),
    }
    task_id = sha256_digest(
        {
            "operation": operation.value,
            "candidate_id": event.candidate_id,
            "candidate_semantic_digest": event.candidate_semantic_digest,
            "mechanism_program_digest": program_digest,
            "parent_candidate_id": parent,
            "required_seed_or_control": required_seed,
            "comparator_identity": comparator,
            "protocol_digest": protocol_digest,
        }
    )
    return ResearchTaskRecordV2(
        task_id=task_id,
        operation=operation,
        candidate_id=event.candidate_id,
        candidate_semantic_digest=event.candidate_semantic_digest,
        mechanism_program_digest=program_digest,
        parent_candidate_id=parent,
        comparator_identity=comparator,
        protocol_digest=protocol_digest,
        required_seed_or_control=required_seed,
        priority=priority,
        created_round=context.round_index,
        evidence_present=evidence,
        missing_seed_count=missing,
        mechanism_program=program,
        producer_role=selected.producer_role,
        provenance_digest=provenance_digest,
        deadline_round=deadline,
        metadata=metadata_payload,
    )


def _task(
    **kwargs: Any,
) -> ResearchTaskV1:
    """Retain the private V1 task helper for existing callers/tests."""

    return _task_record(**kwargs).to_legacy_task()


def _mapping_copy(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _bounded_append(
    value: Any,
    record: Mapping[str, Any],
    *,
    limit: int = _MEMORY_HISTORY_LIMIT,
) -> list[Mapping[str, Any]]:
    entries = list(value) if isinstance(value, (tuple, list)) else []
    normalized = canonical_value(record)
    if normalized not in entries:
        entries.append(normalized)
    return entries[-limit:]


def _selected_parent_id(
    selected: ProducerOutcome,
    route: Mapping[str, Any],
) -> str | None:
    source = selected.source_proposal
    parent = (
        source.parent_candidate_id
        if source is not None
        else route.get("parent_candidate_id")
    )
    if parent is None:
        parent = route.get("parent_candidate_id")
    return str(parent) if parent is not None else None


def _family_key(event: SearchUtilityEventV2, route: Mapping[str, Any]) -> str:
    value = route.get(
        "capability_family",
        route.get("candidate_family", route.get("family", event.mechanism_axis)),
    )
    return _text(value, field="capability_family")


def _frontier_update(
    *,
    context: ResearchContext,
    event: SearchUtilityEventV2,
    episode: TypedResearchEpisodeV1 | None,
    route: Mapping[str, Any],
    selected: ProducerOutcome,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    """Update global/family/parent/control/confirmation frontier banks.

    Engineering closures deliberately take the structural path only: their
    resource/search record is handled by ``_successor`` and no frontier or
    mechanism effect entry is created here.
    """

    if episode is None:
        return dict(context.frontier), False, {}

    frontier = dict(context.frontier)
    current_key = (
        "incumbent_ndcg@10"
        if "incumbent_ndcg@10" in frontier
        else "value"
        if "value" in frontier
        else None
    )
    current = frontier.get(current_key) if current_key is not None else None
    current_value = (
        float(current)
        if isinstance(current, (int, float)) and not isinstance(current, bool)
        else None
    )
    global_bank = _mapping_copy(frontier.get("global"))
    family_bank = _mapping_copy(frontier.get("family"))
    parent_bank = _mapping_copy(frontier.get("parent"))
    control_bank = _mapping_copy(frontier.get("control"))
    confirmation_bank = _mapping_copy(frontier.get("confirmation"))
    if current_value is not None:
        global_bank.setdefault("best_value", current_value)
    global_bank.setdefault("observations", [])
    changed = False
    event_record: dict[str, Any] = {}

    if event.comparator_delta != "NOT_AVAILABLE":
        delta = float(event.comparator_delta)
        candidate_value = (
            current_value + delta if current_value is not None else None
        )
        changed = (
            candidate_value is not None
            and current_value is not None
            and candidate_value > current_value
        )
        event_record = canonical_value(
            {
                "round_index": context.round_index,
                "candidate_id": event.candidate_id,
                "candidate_semantic_digest": event.candidate_semantic_digest,
                "mechanism_axis": event.mechanism_axis,
                "comparator_delta": delta,
                "candidate_value": candidate_value,
                "frontier_updated": changed,
                "evidence_class": episode.evidence_class.value,
            }
        )
        trajectory = list(frontier.get("effect_trajectory", ()))
        trajectory.append(event_record)
        frontier["effect_trajectory"] = tuple(trajectory)
        global_observations = list(global_bank.get("observations", ()))
        global_observations.append(event_record)
        global_bank["observations"] = tuple(global_observations)
        global_bank["last_observation"] = event_record
        if changed:
            global_bank["best_value"] = candidate_value
            global_bank["best_candidate_id"] = event.candidate_id
            global_bank["best_round_index"] = context.round_index
            if current_key is not None:
                frontier[current_key] = candidate_value
                if episode.outcome_ref is not None:
                    frontier["incumbent_ref"] = episode.outcome_ref
                if episode.outcome_digest is not None:
                    frontier["incumbent_digest"] = episode.outcome_digest
                frontier["incumbent_candidate_id"] = event.candidate_id
                frontier["incumbent_round_index"] = context.round_index

        family = _family_key(event, route)
        family_entry = _mapping_copy(family_bank.get(family))
        family_current = family_entry.get("best_value", current_value)
        family_value = (
            float(family_current) + delta
            if isinstance(family_current, (int, float))
            and not isinstance(family_current, bool)
            else candidate_value
        )
        family_entry.setdefault("observations", [])
        family_observations = list(family_entry["observations"])
        family_observations.append(event_record)
        family_entry["observations"] = tuple(family_observations)
        family_entry["last_delta"] = delta
        family_entry["last_candidate_id"] = event.candidate_id
        if family_value is not None:
            family_entry["best_value"] = max(
                float(family_entry.get("best_value", family_value)),
                float(family_value),
            )
        family_bank[family] = family_entry

        parent = _selected_parent_id(selected, route)
        if parent is not None:
            parent_entry = _mapping_copy(parent_bank.get(parent))
            parent_entry.setdefault("observations", [])
            parent_observations = list(parent_entry["observations"])
            parent_observations.append(event_record)
            parent_entry["observations"] = tuple(parent_observations)
            parent_entry["last_delta"] = delta
            parent_entry["last_child_candidate_id"] = event.candidate_id
            parent_bank[parent] = parent_entry

        control = route.get(
            "control_id",
            route.get("matched_control_id", route.get("comparator_identity")),
        )
        if control is not None:
            control = _text(control, field="control_id")
            control_entry = _mapping_copy(control_bank.get(control))
            control_entry.setdefault("observations", [])
            control_observations = list(control_entry["observations"])
            control_observations.append(event_record)
            control_entry["observations"] = tuple(control_observations)
            control_entry["last_candidate_id"] = event.candidate_id
            control_entry["last_delta"] = delta
            control_entry["executed"] = bool(
                route.get("matched_control_executed", False)
            )
            control_bank[control] = control_entry

        confirmation = _mapping_copy(confirmation_bank.get(event.candidate_id))
        seeds = list(confirmation.get("observed_seeds", ()))
        if event.observation_seed not in seeds:
            seeds.append(event.observation_seed)
        confirmation["observed_seeds"] = tuple(seeds)
        confirmation["observed_seed_count"] = len(seeds)
        confirmation["status"] = "PENDING" if changed else confirmation.get(
            "status", "UNCONFIRMED"
        )
        confirmation["required_seed_count"] = int(
            route.get("required_seed_count", 2)
        )
        confirmation["matched_control_status"] = confirmation.get(
            "matched_control_status", "PENDING"
        )
        confirmation["mechanism_off_status"] = confirmation.get(
            "mechanism_off_status", "PENDING"
        )
        confirmation_bank[event.candidate_id] = confirmation

    frontier["global"] = canonical_value(global_bank)
    frontier["family"] = canonical_value(family_bank)
    frontier["parent"] = canonical_value(parent_bank)
    frontier["control"] = canonical_value(control_bank)
    frontier["confirmation"] = canonical_value(confirmation_bank)
    return frontier, changed, event_record


def _task_queue_from_memory(memory: Mapping[str, Any]) -> ResearchTaskQueueV2:
    global_memory = memory.get("global_memory")
    raw_queue = (
        global_memory.get("task_queue")
        if isinstance(global_memory, Mapping)
        else None
    )
    if raw_queue is None:
        raw_queue = memory.get("task_queue")
    if isinstance(raw_queue, ResearchTaskQueueV2):
        return raw_queue
    return ResearchTaskQueueV2.from_dict(raw_queue)


def _confirmation_task(
    base: ResearchTaskRecordV2,
    *,
    operation: ResearchTaskOperationV2,
    required_seed_or_control: str,
    priority: float,
    metadata: Mapping[str, Any],
    candidate_id: str | None = None,
    candidate_semantic_digest: str | None = None,
    mechanism_program_digest: str | None = None,
    mechanism_program: Mapping[str, Any] | None = None,
) -> ResearchTaskRecordV2:
    task_candidate_id = candidate_id or base.candidate_id
    task_semantic_digest = (
        candidate_semantic_digest or base.candidate_semantic_digest
    )
    task_program_digest = mechanism_program_digest or base.mechanism_program_digest
    task_program = mechanism_program or base.mechanism_program
    task_id = sha256_digest(
        {
            "operation": operation.value,
            "candidate_id": task_candidate_id,
            "candidate_semantic_digest": task_semantic_digest,
            "required_seed_or_control": required_seed_or_control,
            "comparator_identity": base.comparator_identity,
            "protocol_digest": base.protocol_digest,
        }
    )
    return replace(
        base,
        task_id=task_id,
        operation=operation,
        candidate_id=task_candidate_id,
        candidate_semantic_digest=task_semantic_digest,
        mechanism_program_digest=task_program_digest,
        mechanism_program=task_program,
        required_seed_or_control=required_seed_or_control,
        priority=priority,
        status=ResearchTaskStatusV2.PENDING,
        close_reason=None,
        missing_seed_count=1,
        metadata={**base.metadata, **dict(metadata)},
    )


def _task_queue_transition(
    *,
    context: ResearchContext,
    queue: ResearchTaskQueueV2,
    task_record: ResearchTaskRecordV2,
    event: SearchUtilityEventV2,
    episode: TypedResearchEpisodeV1 | None,
    route: Mapping[str, Any],
    frontier_updated: bool,
) -> tuple[ResearchTaskQueueV2, dict[str, Any]]:
    """Satisfy prior work, enqueue follow-ups, and retain every task identity."""

    satisfied: list[str] = []
    closed: list[str] = []
    created: list[str] = []
    # Merely routing with a queue head active is not evidence that the
    # selected experiment satisfied it.  Runtime emits ``satisfies_task_id``
    # only when the candidate/seed (or repair) identity actually matches; old
    # callers may still use the explicit ``completed_task_id`` alias.
    explicit_satisfied = route.get(
        "satisfies_task_id",
        route.get("completed_task_id"),
    )
    observed_control = route.get(
        "observed_control_id",
        route.get("control_id", route.get("matched_control_id")),
    )
    observed_mechanism_off = route.get("observed_mechanism_off_id")

    for existing in queue.tasks:
        if existing.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            continue
        matches_identity = (
            existing.candidate_semantic_digest == event.candidate_semantic_digest
            and existing.candidate_id == event.candidate_id
        )
        matches_required = False
        if explicit_satisfied is not None:
            matches_required = (
                existing.task_id == str(explicit_satisfied)
                and matches_identity
            )
        elif matches_identity:
            if existing.operation in {
                ResearchTaskOperationV2.NEW_SEED,
                ResearchTaskOperationV2.REPRODUCE,
            }:
                matches_required = (
                    existing.required_seed_or_control == event.observation_seed
                    or (
                        existing.required_seed_or_control
                        == _NEXT_DEVELOPMENT_SEED
                        and event.observation_seed not in existing.evidence_present
                    )
                )
            elif existing.operation is ResearchTaskOperationV2.MATCHED_CONTROL:
                matches_required = (
                    observed_control is not None
                    and existing.required_seed_or_control == str(observed_control)
                )
            elif existing.operation is ResearchTaskOperationV2.MECHANISM_OFF:
                matches_required = (
                    observed_mechanism_off is not None
                    and existing.required_seed_or_control
                    == str(observed_mechanism_off)
                )
            elif existing.operation is ResearchTaskOperationV2.REPAIR:
                matches_required = bool(episode is None)
        if not matches_required:
            continue
        queue = queue.satisfy(
            existing.task_id,
            evidence=(event.observation_seed,),
            reason="REQUIRED_TASK_EVIDENCE_OBSERVED",
        )
        satisfied.append(existing.task_id)

    if task_record.operation is ResearchTaskOperationV2.MOVE_ON:
        # A same-seed default is recorded for auditability but cannot remain a
        # pending executable task and re-trigger the old repeat loop.
        closed_record = replace(
            task_record,
            status=ResearchTaskStatusV2.CLOSED,
            close_reason="NO_ACTIONABLE_FOLLOW_UP",
            missing_seed_count=0,
        )
        prior = queue.get(closed_record.task_id)
        if prior is None:
            queue = queue.enqueue(closed_record)
            created.append(closed_record.task_id)
        elif prior.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.close(
                prior.task_id,
                reason="NO_ACTIONABLE_FOLLOW_UP",
            )
        for existing in queue.tasks:
            if (
                existing.status in {
                    ResearchTaskStatusV2.PENDING,
                    ResearchTaskStatusV2.ACTIVE,
                }
                and existing.candidate_semantic_digest
                == task_record.candidate_semantic_digest
                and existing.operation is not ResearchTaskOperationV2.REPAIR
            ):
                queue = queue.close(
                    existing.task_id,
                    reason="MOVE_ON_AFTER_NON_CONFIRMING_RESULT",
                )
                closed.append(existing.task_id)
    else:
        prior = queue.get(task_record.task_id)
        queue = queue.enqueue(task_record)
        if prior is None:
            created.append(task_record.task_id)

    if frontier_updated and episode is not None:
        next_seed = route.get(
            "confirmation_seed",
            route.get("next_seed_or_control", route.get("required_seed_or_control")),
        )
        next_seed = (
            _NEXT_DEVELOPMENT_SEED
            if next_seed is None or str(next_seed) == event.observation_seed
            else str(next_seed)
        )
        followups = [
            _confirmation_task(
                task_record,
                operation=ResearchTaskOperationV2.NEW_SEED,
                required_seed_or_control=next_seed,
                priority=1.0,
                metadata={"confirmation_target": "NEW_SEED"},
            ),
        ]
        deferred_requirements: list[str] = []
        for prefix, operation, required, priority in (
            (
                "matched_control",
                ResearchTaskOperationV2.MATCHED_CONTROL,
                str(route.get("matched_control_seed", event.observation_seed)),
                0.99,
            ),
            (
                "mechanism_off",
                ResearchTaskOperationV2.MECHANISM_OFF,
                str(route.get("mechanism_off_seed", event.observation_seed)),
                0.98,
            ),
        ):
            candidate_id = route.get(f"{prefix}_candidate_id")
            semantic_digest = route.get(f"{prefix}_semantic_digest")
            program = route.get(f"{prefix}_program")
            program_digest = route.get(f"{prefix}_program_digest")
            supplied = tuple(
                item is not None
                for item in (
                    candidate_id,
                    semantic_digest,
                    program,
                    program_digest,
                )
            )
            if not any(supplied):
                deferred_requirements.append(prefix.upper())
                continue
            if not all(supplied) or not isinstance(program, Mapping):
                raise ScientificInterpreterError(
                    f"{prefix} task requires a complete executable candidate identity"
                )
            validated_semantic = validate_sha256(
                semantic_digest,
                field_name=f"{prefix}_semantic_digest",
            )
            validated_program = validate_sha256(
                program_digest,
                field_name=f"{prefix}_program_digest",
            )
            if sha256_digest(program) != validated_program:
                raise ScientificInterpreterError(
                    f"{prefix} task program digest mismatch"
                )
            followups.append(
                _confirmation_task(
                    task_record,
                    operation=operation,
                    required_seed_or_control=required,
                    priority=priority,
                    candidate_id=_text(
                        candidate_id,
                        field=f"{prefix}_candidate_id",
                    ),
                    candidate_semantic_digest=validated_semantic,
                    mechanism_program_digest=validated_program,
                    mechanism_program=program,
                    metadata={
                        "confirmation_target": prefix.upper(),
                        "frontier_candidate_id": task_record.candidate_id,
                        "frontier_candidate_semantic_digest": (
                            task_record.candidate_semantic_digest
                        ),
                    },
                )
            )
        for followup in followups:
            prior = queue.get(followup.task_id)
            queue = queue.enqueue(followup)
            if prior is None:
                created.append(followup.task_id)
        transition_deferred = tuple(deferred_requirements)
    else:
        transition_deferred = ()
    head = queue.select_next()
    return queue, {
        "satisfied_task_ids": tuple(satisfied),
        "closed_task_ids": tuple(closed),
        "created_task_ids": tuple(created),
        "deferred_requirements": transition_deferred,
        "head_task_id": head.task_id if head is not None else None,
    }


def _behavior(context: ResearchContext, policy: VersionedResearchPolicyV1) -> BehaviorProjection:
    risk = context.frontier.get("implementation_risk", {})
    return BehaviorProjection(
        round_index=context.round_index,
        context_ref=context.context_ref,
        context_digest=context.digest,
        profile_ref=context.active_profile_ref,
        profile_digest=context.active_profile_digest,
        policy_digest=policy.digest,
        producer_inputs_digest=context.producer_inputs_digest,
        producer_allocation=policy.producer_token_allocation,
        axis_priorities=policy.mechanism_axis_targeting,
        memory_retrieval_policy=policy.memory_retrieval_policy,
        acquisition_parameters=dict(policy.acquisition_parameters),
        implementation_risk=risk if isinstance(risk, Mapping) else {},
    )


def _successor(
    context: ResearchContext,
    policy: VersionedResearchPolicyV1,
    snapshot: SearchMemorySnapshotV1,
    latest_feedback: Mapping[str, Any],
    *,
    event: SearchUtilityEventV2 | None = None,
    episode: TypedResearchEpisodeV1 | None = None,
    task_record: ResearchTaskRecordV2 | None = None,
    selected_outcome: ProducerOutcome | None = None,
    producer_outcomes: Sequence[ProducerOutcome] = (),
    route_metadata: Mapping[str, Any] | None = None,
    advance_round: bool = True,
) -> ResearchContext:
    route = dict(route_metadata) if isinstance(route_metadata, Mapping) else {}
    frontier = dict(context.frontier)
    frontier_updated = False
    event_record: dict[str, Any] = {}
    if event is not None and selected_outcome is not None:
        frontier, frontier_updated, event_record = _frontier_update(
            context=context,
            event=event,
            episode=episode,
            route=route,
            selected=selected_outcome,
        )

    queue = _task_queue_from_memory(context.scientific_memory)
    transition: dict[str, Any] = {}
    if task_record is not None and event is not None:
        queue, transition = _task_queue_transition(
            context=context,
            queue=queue,
            task_record=task_record,
            event=event,
            episode=episode,
            route=route,
            frontier_updated=frontier_updated,
        )
    next_record = queue.select_next()

    feedback = canonical_value(dict(latest_feedback))
    feedback["research_task_slot"] = (
        next_record.prompt_projection() if next_record is not None else None
    )
    feedback["task_queue_head"] = (
        next_record.to_dict() if next_record is not None else None
    )
    feedback["task_queue_transition"] = canonical_value(transition)
    feedback["task_queue_digest"] = queue.digest

    memory = dict(context.scientific_memory)
    existing_global = memory.get("global_memory")
    if not isinstance(existing_global, Mapping):
        existing_global = memory.get("global")
    if isinstance(existing_global, Mapping):
        global_memory = dict(existing_global)
    else:
        global_memory = {
            key: value
            for key, value in memory.items()
            if key not in {"by_role", "global_memory", "global"}
        }
    head = snapshot.to_dict()
    global_memory["search_memory_head"] = head
    global_memory["latest_feedback"] = feedback
    global_memory["task_queue"] = queue.to_dict()
    global_memory["frontier"] = frontier

    executed_observations = memory.get(
        "executed_observations", global_memory.get("executed_observations", ())
    )
    executed_observations = (
        list(executed_observations)
        if isinstance(executed_observations, (tuple, list))
        else []
    )
    search_observations = global_memory.get("search_observations", ())
    search_observations = (
        list(search_observations)
        if isinstance(search_observations, (tuple, list))
        else []
    )
    if event is not None:
        resource_cost_projection = canonical_value(
            dict(event.resource_cost_projection)
        )
        cost_memory_fields = (
            {
                "resource_cost_projection": resource_cost_projection,
                "resource_cost_projection_digest": sha256_digest(
                    resource_cost_projection
                ),
            }
            if resource_cost_projection
            else {}
        )
        executed_observation = canonical_value(
            {
                "round_index": context.round_index,
                "candidate_semantic_digest": event.candidate_semantic_digest,
                "observation_seed": event.observation_seed,
                "common_outcome_class": event.common_outcome_class,
                "mechanism_axis": event.mechanism_axis,
                "comparator_delta": event.comparator_delta,
            }
        )
        executed_observations = _bounded_append(
            executed_observations,
            executed_observation,
        )
        search_observation = canonical_value(
            {
                "round_index": context.round_index,
                "event_id": sha256_digest(executed_observation),
                "candidate_id": event.candidate_id,
                "candidate_semantic_digest": event.candidate_semantic_digest,
                "observation_seed": event.observation_seed,
                "common_outcome_class": event.common_outcome_class,
                "mechanism_axis": event.mechanism_axis,
                "comparator_delta": event.comparator_delta,
                "evidence_domain": (
                    "SCIENTIFIC_EPISODE" if episode is not None else "RESOURCE_SEARCH"
                ),
                "mechanism_effect_update_allowed": episode is not None,
                **cost_memory_fields,
            }
        )
        search_observations = _bounded_append(
            search_observations,
            search_observation,
        )
        if episode is None:
            resource_memory = _mapping_copy(global_memory.get("resource_memory"))
            resource_observations = resource_memory.get("observations", ())
            resource_observations = (
                list(resource_observations)
                if isinstance(resource_observations, (tuple, list))
                else []
            )
            resource_observation = canonical_value(
                {
                    "round_index": context.round_index,
                    "event_id": sha256_digest(executed_observation),
                    "failure_class": event.common_outcome_class,
                    "typed_blocker_class": event.typed_blocker_class,
                    "candidate_id": event.candidate_id,
                    "observation_seed": event.observation_seed,
                    "resource_or_search_only": True,
                    **cost_memory_fields,
                }
            )
            resource_observations = _bounded_append(
                resource_observations,
                resource_observation,
            )
            resource_memory["observations"] = tuple(resource_observations)
            resource_memory["last_observation"] = resource_observation
            global_memory["resource_memory"] = resource_memory
        elif event_record:
            scientific_observations = global_memory.get(
                "scientific_observations", ()
            )
            scientific_observations = (
                list(scientific_observations)
                if isinstance(scientific_observations, (tuple, list))
                else []
            )
            scientific_observation = canonical_value(
                {
                    **event_record,
                    "failure_class": episode.failure_class.value,
                    "mechanism_effect_update_allowed": True,
                }
            )
            scientific_observations = _bounded_append(
                scientific_observations,
                scientific_observation,
            )
            global_memory["scientific_observations"] = tuple(scientific_observations)
    global_memory["executed_observations"] = executed_observations
    global_memory["search_observations"] = tuple(search_observations)

    raw_by_role = memory.get("by_role")
    raw_by_role = raw_by_role if isinstance(raw_by_role, Mapping) else {}
    role_memories: dict[str, dict[str, Any]] = {}
    lineage_index = _mapping_copy(global_memory.get("producer_lineage_index"))

    def append_history(
        role_memory: dict[str, Any],
        key: str,
        record: Mapping[str, Any],
        *,
        limit: int | None = None,
    ) -> None:
        entries = role_memory.get(key, ())
        entries = list(entries) if isinstance(entries, (tuple, list)) else []
        normalized = canonical_value(record)
        if normalized not in entries:
            entries.append(normalized)
        if limit is not None:
            entries = entries[-limit:]
        role_memory[key] = tuple(entries)

    for role in DISCOVERY_PRODUCERS:
        role_memory = (
            dict(raw_by_role.get(role, {}))
            if isinstance(raw_by_role.get(role, {}), Mapping)
            else {}
        )
        # These fields were the replicated tail in the pre-V2 layout.  They
        # now live once in global memory; selected execution feedback is added
        # below to the originating role only.
        for shared_key in (
            "search_memory_head",
            "latest_feedback",
            "task_queue",
            "task_queue_head",
            "executed_observations",
            "search_observations",
        ):
            role_memory.pop(shared_key, None)
        role_memory["producer_role"] = role
        provenance = _mapping_copy(role_memory.get("provenance"))
        provenance["producer_role"] = role
        role_memory["provenance"] = provenance
        role_memories[role] = role_memory

    outcome_map = {
        outcome.producer_role: outcome
        for outcome in producer_outcomes
        if isinstance(outcome, ProducerOutcome)
    }
    for outcome in producer_outcomes:
        if not isinstance(outcome, ProducerOutcome):
            continue
        role_memory = role_memories[outcome.producer_role]
        if outcome.spec is not None:
            source = outcome.source_proposal
            proposal_record = {
                "round_index": context.round_index,
                "producer_role": outcome.producer_role,
                "outcome_digest": outcome.digest,
                "spec_id": outcome.spec.spec_id,
                "spec_digest": outcome.spec.digest,
                "source_proposal_digest": source.digest if source is not None else None,
                "source_candidate_id": source.candidate_id if source is not None else None,
                "provenance": outcome.provenance,
            }
            append_history(role_memory, "proposal_history", proposal_record)
            for digest in (
                outcome.digest,
                outcome.spec.digest,
                source.digest if source is not None else None,
            ):
                if digest is not None:
                    lineage_index[digest] = outcome.producer_role
        else:
            append_history(
                role_memory,
                "failure_history",
                {
                    "round_index": context.round_index,
                    "producer_role": outcome.producer_role,
                    "outcome_digest": outcome.digest,
                    "failure_code": outcome.failure_code,
                    "failure_detail": outcome.failure_detail,
                    "provenance": outcome.provenance,
                    "resource_or_search_only": True,
                },
            )
            credit = _mapping_copy(role_memory.get("credit"))
            credit["producer_failure_count"] = int(
                credit.get("producer_failure_count", 0)
            ) + 1
            role_memory["credit"] = credit

    if selected_outcome is not None:
        selected_role = selected_outcome.producer_role
        role_memory = role_memories[selected_role]
        for digest in (
            selected_outcome.digest,
            selected_outcome.spec.digest if selected_outcome.spec is not None else None,
            (
                selected_outcome.source_proposal.digest
                if selected_outcome.source_proposal is not None
                else None
            ),
        ):
            if digest is not None:
                lineage_index[digest] = selected_role
        current = outcome_map.get(selected_role)
        origin = (
            "CURRENT_ROUND"
            if current is not None and current.digest == selected_outcome.digest
            else "CARRYOVER"
        )
        execution_record = {
            "round_index": context.round_index,
            "producer_role": selected_role,
            "origin": origin,
            "outcome_digest": selected_outcome.digest,
            "spec_digest": (
                selected_outcome.spec.digest
                if selected_outcome.spec is not None
                else None
            ),
            "source_proposal_digest": (
                selected_outcome.source_proposal.digest
                if selected_outcome.source_proposal is not None
                else None
            ),
            "candidate_id": event.candidate_id if event is not None else None,
            "candidate_semantic_digest": (
                event.candidate_semantic_digest if event is not None else None
            ),
            "observation_seed": event.observation_seed if event is not None else None,
            "common_outcome_class": (
                event.common_outcome_class if event is not None else None
            ),
            "comparator_delta": event.comparator_delta if event is not None else None,
            "frontier_updated": frontier_updated,
            "task_satisfied_count": len(transition.get("satisfied_task_ids", ())),
            "task_closed_count": len(transition.get("closed_task_ids", ())),
            "provenance": selected_outcome.provenance,
        }
        if event is not None and event.resource_cost_projection:
            execution_record.update(
                {
                    "resource_cost_projection": canonical_value(
                        dict(event.resource_cost_projection)
                    ),
                    "resource_cost_projection_digest": sha256_digest(
                        event.resource_cost_projection
                    ),
                }
            )
        compute_pattern = route.get("selected_compute_pattern")
        if compute_pattern is not None:
            execution_record["compute_pattern"] = _text(
                compute_pattern,
                field="selected_compute_pattern",
            )
        resource_admission = route.get("selected_resource_admission_state")
        if resource_admission is not None:
            if isinstance(resource_admission, Enum):
                resource_admission = resource_admission.value
            execution_record["resource_admission_state"] = _text(
                resource_admission,
                field="selected_resource_admission_state",
            )
        resource_evidence_digest = route.get("selected_resource_evidence_digest")
        if resource_evidence_digest is not None:
            execution_record["resource_evidence_digest"] = _digest(
                resource_evidence_digest,
                field="selected_resource_evidence_digest",
            )
        profile_digest = route.get("selected_portfolio_profile_digest")
        if profile_digest is not None:
            execution_record["portfolio_profile_digest"] = _digest(
                profile_digest,
                field="selected_portfolio_profile_digest",
            )
        append_history(
            role_memory,
            "execution_history",
            execution_record,
            limit=_MEMORY_HISTORY_LIMIT,
        )
        role_memory["latest_feedback"] = feedback
        credit = _mapping_copy(role_memory.get("credit"))
        credit["selected_execution_count"] = int(
            credit.get("selected_execution_count", 0)
        ) + 1
        if episode is None:
            credit["resource_search_execution_count"] = int(
                credit.get("resource_search_execution_count", 0)
            ) + 1
        else:
            credit["scientific_execution_count"] = int(
                credit.get("scientific_execution_count", 0)
            ) + 1
            if frontier_updated:
                credit["frontier_gain_count"] = int(
                    credit.get("frontier_gain_count", 0)
                ) + 1
            if episode.failure_class is ResearchFailureClassV1.MECHANISM:
                credit["mechanism_negative_count"] = int(
                    credit.get("mechanism_negative_count", 0)
                ) + 1
        credit["task_satisfied_count"] = int(
            credit.get("task_satisfied_count", 0)
        ) + len(transition.get("satisfied_task_ids", ()))
        credit["task_closed_count"] = int(
            credit.get("task_closed_count", 0)
        ) + len(transition.get("closed_task_ids", ()))
        role_memory["credit"] = credit
        role_memory["last_execution_round"] = context.round_index
        if next_record is not None:
            role_memory["last_task_id"] = next_record.task_id

    global_memory["producer_lineage_index"] = lineage_index
    memory["global_memory"] = canonical_value(global_memory)
    memory["by_role"] = canonical_value(role_memories)
    # Retain the old top-level aliases for runtime/search consumers that have
    # not yet adopted the V2 global queue projection.
    memory["search_memory_head"] = head
    memory["latest_feedback"] = feedback
    memory["task_queue"] = queue.to_dict()
    memory["executed_observations"] = executed_observations
    memory["search_observations"] = tuple(search_observations)
    return replace(
        context,
        round_index=context.round_index + (1 if advance_round else 0),
        policy=policy.to_dict(),
        scientific_memory=memory,
        frontier=frontier,
    )


@dataclass(frozen=True, slots=True)
class EpisodeInterpretation:
    """One shared result for scientific Episodes and diagnostic closures."""

    episode: TypedResearchEpisodeV1 | None
    closure: ScientificEpisodeClosureV1
    failure_taxonomy: str
    mechanism_attribution: str
    negative_evidence: tuple[str, ...]
    search_utility_event: SearchUtilityEventV2
    next_discriminative_task: ResearchTaskV1
    mechanism_belief: DevelopmentalMechanismBeliefV1 | None
    feedback_projection: PromptFeedbackProjectionV2
    search_memory_snapshot: SearchMemorySnapshotV1
    policy_successor: VersionedResearchPolicyV1
    successor_context: ResearchContext
    behavior_before: BehaviorProjection
    behavior_after: BehaviorProjection
    route_trace_digest: str


@dataclass(frozen=True, slots=True)
class MissingSearchInterpretation:
    """Fail-soft interpretation when no legal experiment binding exists."""

    failure_taxonomy: str
    mechanism_attribution: str
    negative_evidence: tuple[str, ...]
    feedback_projection: Mapping[str, Any]
    search_memory_snapshot: SearchMemorySnapshotV1
    policy_successor: VersionedResearchPolicyV1
    successor_context: ResearchContext
    behavior_before: BehaviorProjection
    behavior_after: BehaviorProjection
    route_trace_digest: str


def _interpret(
    *,
    episode: TypedResearchEpisodeV1 | None,
    comparison_identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
    context: ResearchContext,
    producer_outcomes: Sequence[ProducerOutcome],
    route_metadata: Mapping[str, Any],
    evaluator_projection: SearchUtilityEventV2,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    selected_outcome: ProducerOutcome | None = None,
    frozen_context: ResearchContext | None = None,
    advance_round: bool = True,
) -> EpisodeInterpretation:
    if not isinstance(evaluator_projection, SearchUtilityEventV2):
        raise ScientificInterpreterError(
            "evaluator_projection must be SearchUtilityEventV2"
        )
    if not isinstance(memory_writer, SearchMemoryWriterV1):
        raise ScientificInterpreterError("memory_writer must be SearchMemoryWriterV1")
    identity_context = context if frozen_context is None else frozen_context
    _validate_same_round_working_context(identity_context, context)
    _validate_context(
        episode=episode,
        identity=comparison_identity,
        context=identity_context,
        policy=policy,
        policy_context=context,
    )
    _validate_closure(
        episode=episode,
        identity=comparison_identity,
        closure=closure,
    )
    outcomes = _producer_map(producer_outcomes, identity_context)
    trace, selected_role, required_probability, selected = _route(
        route_metadata, outcomes, evaluator_projection, selected_outcome
    )
    metadata = _mapping(route_metadata, field="route_metadata")
    aggregate = _meta_aggregate(
        outcomes,
        selected_role,
        selected,
        evaluator_projection,
        required_probability,
        policy,
        scientific_episode=episode is not None,
    )
    belief: DevelopmentalMechanismBeliefV1 | None = None
    if episode is not None and episode.failure_class in {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
    }:
        belief = project_episode_to_mechanism_belief(
            comparison_identity=comparison_identity,
            closure=closure,
            episode=episode,
            mechanism_axis=evaluator_projection.mechanism_axis,
        )
    next_test = (
        episode.next_discriminative_test
        if episode is not None
        else _text(metadata.get("next_discriminative_test"), field="next_discriminative_test")
    )
    task_record = _task_record(
        episode=episode,
        identity=comparison_identity,
        closure=closure,
        event=evaluator_projection,
        metadata=metadata,
        selected=selected,
        context=context,
        next_test=next_test,
    )
    task = task_record.to_legacy_task()
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=task.prompt_projection(),
    )
    successor_policy = VersionedMetaPolicyUpdaterV1().update(
        policy,
        completed_round_index=context.round_index,
        aggregate=aggregate,
    )
    before = _behavior(context, policy)
    predecessor = memory_writer.head.digest if memory_writer.head else None
    memory_feedback = feedback.to_dict()
    if episode is None:
        memory_feedback["engineering_diagnostic"] = {
            "closure_id": closure.closure_id,
            "failure_class": closure.failure_class.value,
            "failure_detail_ref": closure.failure_detail_ref,
            "failure_detail_digest": closure.failure_detail_digest,
        }
    snapshot = memory_writer.commit(
        round_index=context.round_index,
        expected_predecessor_digest=predecessor,
        beliefs=(belief,) if belief is not None else (),
        route_trace_digest=trace,
        feedback_projection=memory_feedback,
    )
    successor_context = _successor(
        context,
        successor_policy,
        snapshot,
        memory_feedback,
        event=evaluator_projection,
        episode=episode,
        task_record=task_record,
        selected_outcome=selected,
        producer_outcomes=producer_outcomes,
        route_metadata=metadata,
        advance_round=advance_round,
    )
    queued_task = _task_queue_from_memory(
        successor_context.scientific_memory
    ).select_next()
    next_task = queued_task.to_legacy_task() if queued_task is not None else task
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=next_task.prompt_projection(),
    )
    after = _behavior(successor_context, successor_policy)
    behavior_fields = {
        "producer_inputs_digest",
        "producer_allocation",
        "axis_priorities",
        "memory_retrieval_policy",
        "acquisition_parameters",
        "implementation_risk",
    }
    if advance_round:
        changed_behavior_fields = set(before.changed_fields(after))
    else:
        changed_behavior_fields = {
            field_name
            for field_name in behavior_fields
            if canonical_value(getattr(before, field_name))
            != canonical_value(getattr(after, field_name))
        }
    if not changed_behavior_fields & behavior_fields:
        raise ScientificInterpreterError(
            "policy successor does not change a subsequent decision input"
        )
    taxonomy = (
        _FAILURE_TAXONOMY[episode.failure_class]
        if episode is not None
        else f"ENGINEERING_DIAGNOSTIC_{closure.failure_class.value}"
    )
    attribution = (
        _MECHANISM_ATTRIBUTION[episode.failure_class]
        if episode is not None
        else "NOT_APPLICABLE"
    )
    return EpisodeInterpretation(
        episode=episode,
        closure=closure,
        failure_taxonomy=taxonomy,
        mechanism_attribution=attribution,
        negative_evidence=tuple(belief.evidence_against) if belief else (),
        search_utility_event=evaluator_projection,
        next_discriminative_task=next_task,
        mechanism_belief=belief,
        feedback_projection=feedback,
        search_memory_snapshot=snapshot,
        policy_successor=successor_policy,
        successor_context=successor_context,
        behavior_before=before,
        behavior_after=after,
        route_trace_digest=trace,
    )


def interpret_missing_search_opportunity(
    *,
    context: ResearchContext,
    producer_outcomes: Sequence[ProducerOutcome],
    diagnostic_detail: Mapping[str, Any],
    next_discriminative_test: str,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    route_trace_digest: str | None = None,
) -> MissingSearchInterpretation:
    """Persist one missing ordinary experiment opportunity without inventing an Episode."""

    if not isinstance(memory_writer, SearchMemoryWriterV1):
        raise ScientificInterpreterError("memory_writer must be SearchMemoryWriterV1")
    if canonical_value(context.policy) != canonical_value(policy.to_dict()):
        raise ScientificInterpreterError(
            "context policy and supplied versioned policy are not the same input"
        )
    _producer_map(producer_outcomes, context)
    detail = canonical_value(_mapping(diagnostic_detail, field="diagnostic_detail"))
    next_test = _text(next_discriminative_test, field="next_discriminative_test")
    trace = route_trace_digest or sha256_digest(
        {
            "context_digest": context.digest,
            "diagnostic_detail": detail,
        }
    )
    _digest(trace, field="route_trace_digest")
    feedback = {
        "schema": "recclaw.research-line.missing-search-feedback.v1",
        "failure_class": ResearchFailureClassV1.OUTCOME_MISSING.value,
        "diagnostic_detail": detail,
        "next_discriminative_test": next_test,
    }
    successor_policy = VersionedMetaPolicyUpdaterV1().update(
        policy,
        completed_round_index=context.round_index,
        aggregate={
            "producer_useful_rates": dict(policy.producer_token_allocation),
            "mechanism_axis_gaps": (),
            "calibration_error": 0.0,
        },
    )
    before = _behavior(context, policy)
    predecessor = memory_writer.head.digest if memory_writer.head else None
    snapshot = memory_writer.commit(
        round_index=context.round_index,
        expected_predecessor_digest=predecessor,
        beliefs=(),
        route_trace_digest=trace,
        feedback_projection=feedback,
    )
    successor_context = _successor(
        context,
        successor_policy,
        snapshot,
        feedback,
        producer_outcomes=producer_outcomes,
    )
    after = _behavior(successor_context, successor_policy)
    return MissingSearchInterpretation(
        failure_taxonomy="ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING",
        mechanism_attribution="NOT_APPLICABLE",
        negative_evidence=(),
        feedback_projection=feedback,
        search_memory_snapshot=snapshot,
        policy_successor=successor_policy,
        successor_context=successor_context,
        behavior_before=before,
        behavior_after=after,
        route_trace_digest=trace,
    )


def interpret_typed_research_episode(
    *,
    episode: TypedResearchEpisodeV1,
    comparison_identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
    context: ResearchContext,
    producer_outcomes: Sequence[ProducerOutcome],
    route_metadata: Mapping[str, Any],
    evaluator_projection: SearchUtilityEventV2,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    selected_outcome: ProducerOutcome | None = None,
    frozen_context: ResearchContext | None = None,
) -> EpisodeInterpretation:
    return _interpret(
        episode=episode,
        comparison_identity=comparison_identity,
        closure=closure,
        context=context,
        producer_outcomes=producer_outcomes,
        route_metadata=route_metadata,
        evaluator_projection=evaluator_projection,
        policy=policy,
        memory_writer=memory_writer,
        selected_outcome=selected_outcome,
        frozen_context=frozen_context,
    )


def interpret_scientific_diagnostic(
    *,
    closure: ScientificEpisodeClosureV1,
    comparison_identity: FrozenComparisonIdentityV1,
    context: ResearchContext,
    producer_outcomes: Sequence[ProducerOutcome],
    route_metadata: Mapping[str, Any],
    evaluator_projection: SearchUtilityEventV2,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    selected_outcome: ProducerOutcome | None = None,
    frozen_context: ResearchContext | None = None,
    advance_round: bool = True,
) -> EpisodeInterpretation:
    return _interpret(
        episode=None,
        comparison_identity=comparison_identity,
        closure=closure,
        context=context,
        producer_outcomes=producer_outcomes,
        route_metadata=route_metadata,
        evaluator_projection=evaluator_projection,
        policy=policy,
        memory_writer=memory_writer,
        selected_outcome=selected_outcome,
        frozen_context=frozen_context,
        advance_round=advance_round,
    )


__all__ = [
    "EpisodeInterpretation",
    "MissingSearchInterpretation",
    "ScientificInterpreterError",
    "interpret_missing_search_opportunity",
    "interpret_scientific_diagnostic",
    "interpret_typed_research_episode",
]
