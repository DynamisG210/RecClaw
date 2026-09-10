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
    VersionedResearchPolicyV1,
    _next_acquisition_parameters,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    MECHANISM_AXIS_UNIVERSE_V1,
    canonical_mechanism_axis,
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
    observation_updates_search_utility,
    research_producer_roles,
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
    scheduled_roles = research_producer_roles(context.budget)
    if isinstance(outcomes, (str, bytes)) or len(outcomes) != len(scheduled_roles):
        raise ScientificInterpreterError(
            "exactly one outcome is required for each scheduled Producer"
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
    if set(result) != set(scheduled_roles):
        raise ScientificInterpreterError(
            "producer_outcomes must cover the frozen research schedule"
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


def _axis_values(
    value: Any,
    *,
    axis_universe: tuple[str, ...] = MECHANISM_AXIS_UNIVERSE_V1,
) -> tuple[str, ...]:
    """Project mechanism labels into the active policy's axis universe."""

    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, (tuple, list)):
        return ()
    allowed = tuple(axis_universe)
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            continue
        text = item.strip()
        axis = text if text in allowed else canonical_mechanism_axis(text)
        if axis in allowed and axis not in result:
            result.append(axis)
    return tuple(result)


def _complete_mechanism_values(value: Any) -> tuple[str, ...]:
    """Read raw slot/dimension identities without applying policy aliases."""

    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Mapping):
        result: list[str] = []
        for key in (
            "slot_id",
            "dimension",
            "mechanism_dimension",
            "changed_axis",
            "axis",
            "mechanism_axis",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                text = item.strip()
                if text not in result:
                    result.append(text)
        return tuple(result)
    if not isinstance(value, (tuple, list)):
        return ()
    result: list[str] = []
    for item in value:
        for text in _complete_mechanism_values(item):
            if text not in result:
                result.append(text)
    return tuple(result)


def _program_mechanism_values(
    program: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    """Prefer raw changed dimensions over a policy-axis projection."""

    if not isinstance(program, Mapping):
        return ()
    payload = program.get("program_payload", program)
    if not isinstance(payload, Mapping):
        return ()
    declared: list[str] = []
    raw: list[str] = []
    for key in (
        "mechanism_axis_footprint",
        "changed_axis_footprint",
        "changed_axes",
        "changed_dimensions",
        "high_change_dimensions",
        "changed_slots",
    ):
        identities = _complete_mechanism_values(payload.get(key, ()))
        for identity in identities:
            if identity not in declared:
                declared.append(identity)
            if identity not in MECHANISM_AXIS_UNIVERSE_V1 and identity not in raw:
                raw.append(identity)
    # A program's raw BL-ICF slot/dimension identity is authoritative.  A
    # policy scalar is retained only when no raw identity was declared.
    return tuple(raw or declared)


def _non_policy_mechanism_values(value: Any) -> tuple[str, ...]:
    return tuple(
        identity
        for identity in _complete_mechanism_values(value)
        if identity not in MECHANISM_AXIS_UNIVERSE_V1
    )


def _mechanism_axis_footprint(
    *,
    event: SearchUtilityEventV2,
    selected: ProducerOutcome,
    route: Mapping[str, Any],
) -> tuple[str, ...]:
    """Recover the complete changed-axis footprint from existing program facts."""

    # Only this event's evaluator axis, actual evaluator footprint, resolved
    # mechanism facts, and the compiled/executed program can establish what
    # changed in the current experiment.  A future task axis is not a current
    # mechanism footprint.
    source = selected.resolution_facts
    programs = (
        selected.source_mechanism_program,
        (
            selected.source_proposal.mechanism_program
            if selected.source_proposal is not None
            else None
        ),
        route.get("mechanism_program"),
    )
    program_values: list[str] = []
    for program in programs:
        for identity in _program_mechanism_values(program):
            if identity not in program_values:
                program_values.append(identity)

    event_values = _complete_mechanism_values(event.mechanism_axis_footprint)
    resolution_values: list[str] = []
    if isinstance(source, Mapping):
        for key in (
            "mechanism_axis_footprint",
            "changed_axis_footprint",
            "changed_axes",
            "changed_dimensions",
        ):
            for identity in _complete_mechanism_values(source.get(key, ())):
                if identity not in resolution_values:
                    resolution_values.append(identity)

    program_raw_values = tuple(
        identity
        for identity in program_values
        if identity not in MECHANISM_AXIS_UNIVERSE_V1
    )
    partial_raw_values = tuple(
        dict.fromkeys(
            (
                *_non_policy_mechanism_values(event_values),
                *_non_policy_mechanism_values(resolution_values),
            )
        )
    )
    if program_values:
        # The compiled/executed program wins over partial policy projections.
        # Preserve additional raw event/resolution identities, but never add
        # a seven-axis alias such as ``architecture`` to the raw footprint.
        values: list[Any] = list(program_raw_values or partial_raw_values or program_values)
        for identity in partial_raw_values:
            if identity not in values:
                values.append(identity)
    else:
        values = list(event_values)
        for identity in resolution_values:
            if identity not in values:
                values.append(identity)
        if isinstance(source, Mapping):
            values.extend(
                _complete_mechanism_values(source.get("high_change_dimensions", ()))
            )
    # mechanism_axis is a policy scalar in the normal executable path.  It is
    # retained as an actual footprint only when no more specific evaluator,
    # resolution, or program identity supplied one; otherwise it would add a
    # second alias for the same raw mechanism change.
    if not values:
        values.extend(_complete_mechanism_values(event.mechanism_axis))
    result: list[str] = []
    for value in values:
        for identity in _complete_mechanism_values(value):
            if identity not in result:
                result.append(identity)
    return tuple(result)


def _has_independent_design_evidence(
    *,
    context: ResearchContext | None,
    event: SearchUtilityEventV2,
    route: Mapping[str, Any],
) -> bool:
    """Recognize only a bound, already executed independent evidence task.

    Plan fields, operation names, and Provider/route booleans describe an
    intention.  Causal credit requires the durable task identity to have been
    satisfied by this metric-bearing event, with its candidate/program,
    comparator, parent, and seed bindings intact.  A prior confirmation entry
    is accepted only when it records the task identity as SATISFIED.
    """

    task_record = route.get("active_task_record")
    if isinstance(task_record, Mapping):
        operation = task_record.get("operation")
        operation = getattr(operation, "value", operation)
        operation_text = str(operation or "")
        task_id = task_record.get("task_id")
        task_metadata = task_record.get("metadata")
        task_metadata = task_metadata if isinstance(task_metadata, Mapping) else {}
        if (
            operation_text in {"MATCHED_CONTROL", "MECHANISM_OFF", "REPRODUCE"}
            and isinstance(task_id, str)
            and route.get("satisfies_task_id") == task_id
            and route.get("active_task_record_digest")
            == sha256_digest(task_record)
            and task_record.get("candidate_semantic_digest")
            == event.candidate_semantic_digest
            and (
                task_record.get("candidate_id") in (None, event.candidate_id)
            )
            and (
                task_record.get("comparator_identity") in (None, "")
                or task_record.get("comparator_identity")
                == route.get("comparator_identity")
            )
            and event.comparator_delta != "NOT_AVAILABLE"
        ):
            program = route.get("mechanism_program")
            effective_identity = route.get("search_space_execution_binding")
            if isinstance(program, Mapping) and (
                task_record.get("mechanism_program_digest")
                == sha256_digest(program)
                and isinstance(effective_identity, Mapping)
            ):
                parent = task_record.get("parent_candidate_id")
                selected_parent = route.get(
                    "selected_candidate_parent_id",
                    route.get("parent_candidate_id"),
                )
                if (
                    effective_identity.get("effective_experiment_digest")
                    and effective_identity.get("effective_family_digest")
                    and task_metadata.get("effective_experiment_digest")
                    == effective_identity.get("effective_experiment_digest")
                    and task_metadata.get("effective_family_digest")
                    == effective_identity.get("effective_family_digest")
                    and parent in (None, selected_parent)
                ):
                    return True

    if context is None:
        return False
    confirmation_bank = context.frontier.get("confirmation", {})
    if not isinstance(confirmation_bank, Mapping):
        return False
    entry = confirmation_bank.get(event.candidate_id)
    if not isinstance(entry, Mapping):
        return False
    current_program = route.get("mechanism_program")
    current_identity = route.get("search_space_execution_binding")
    if not isinstance(current_program, Mapping) or not isinstance(
        current_identity, Mapping
    ):
        return False
    current_parent = route.get(
        "selected_candidate_parent_id",
        route.get("parent_candidate_id"),
    )

    def exact_satisfied_evidence(value: Any, *, replication: bool) -> bool:
        if not isinstance(value, Mapping):
            return False
        task_id = value.get("task_id")
        task_digest = value.get("task_record_digest")
        if not isinstance(task_id, str) or not isinstance(task_digest, str):
            return False
        try:
            queue = _task_queue_from_memory(context.scientific_memory)
        except (TypeError, ValueError):
            return False
        task = queue.get(task_id)
        if (
            task is None
            or task.status is not ResearchTaskStatusV2.SATISFIED
            or task.digest != task_digest
            or task.candidate_semantic_digest != value.get(
                "candidate_semantic_digest"
            )
            or task.mechanism_program_digest != value.get(
                "mechanism_program_digest"
            )
            or task.comparator_identity != value.get("comparator_identity")
            or task.parent_candidate_id != value.get("parent_candidate_id")
        ):
            return False
        if (
            task.metadata.get("effective_experiment_digest")
            != value.get("effective_experiment_digest")
            or task.metadata.get("effective_family_digest")
            != value.get("effective_family_digest")
        ):
            return False
        if (
            value.get("frontier_candidate_id") != event.candidate_id
            or value.get("frontier_candidate_semantic_digest")
            != event.candidate_semantic_digest
            or value.get("frontier_candidate_program_digest")
            != sha256_digest(current_program)
            or value.get("comparator_identity")
            != route.get("comparator_identity")
            or (
                value.get("parent_candidate_id") is not None
                and value.get("parent_candidate_id") != current_parent
            )
        ):
            return False
        if (
            value.get("frontier_effective_experiment_digest")
            != current_identity.get("effective_experiment_digest")
            or value.get("frontier_effective_family_digest")
            != current_identity.get("effective_family_digest")
        ):
            return False
        if replication:
            evidence_seeds = value.get("evidence_present", ())
            return isinstance(evidence_seeds, (tuple, list)) and any(
                str(seed) != event.observation_seed for seed in evidence_seeds
            )
        return True

    # These statuses are written only after the exact queued control/ablation
    # task is satisfied; the stored evidence must bind back to that task and
    # to the current primary candidate identity.
    bound_control = (
        entry.get("matched_control_status") == "SATISFIED"
        and exact_satisfied_evidence(
            entry.get("matched_control_evidence"), replication=False
        )
    ) or (
        entry.get("mechanism_off_status") == "SATISFIED"
        and exact_satisfied_evidence(
            entry.get("mechanism_off_evidence"), replication=False
        )
    )
    replicated = exact_satisfied_evidence(
        entry.get("replication_evidence"), replication=True
    )
    return bool(bound_control or replicated)


def _enrich_search_utility_event(
    *,
    event: SearchUtilityEventV2,
    selected: ProducerOutcome,
    route: Mapping[str, Any],
    context: ResearchContext | None,
    episode: TypedResearchEpisodeV1 | None,
    axis_universe: tuple[str, ...] = MECHANISM_AXIS_UNIVERSE_V1,
) -> SearchUtilityEventV2:
    footprint = _mechanism_axis_footprint(
        event=event,
        selected=selected,
        route=route,
    )
    design_evidence = _has_independent_design_evidence(
        context=context,
        event=event,
        route=route,
    )
    failure_class = (
        episode.failure_class.value
        if episode is not None
        else event.common_outcome_class
    )
    confounds = list(event.unresolved_confounding)
    if selected.spec is not None:
        competing = selected.spec.competing_explanation
        if competing and competing not in confounds:
            confounds.append(competing)
    if len(footprint) > 1 and not design_evidence:
        confounds.append("MULTI_AXIS_SINGLE_SEED_CONFOUNDING")
    if episode is None:
        evidence_class = "ENGINEERING_FAILURE"
        causal_credit_allowed = False
    elif episode.failure_class is ResearchFailureClassV1.INCONCLUSIVE:
        evidence_class = "INCONCLUSIVE"
        causal_credit_allowed = False
        confounds.append("INCONCLUSIVE_EPISODE")
    elif not footprint:
        evidence_class = "INCONCLUSIVE"
        causal_credit_allowed = False
        confounds.append("MISSING_ACTUAL_MECHANISM_FOOTPRINT")
    elif design_evidence:
        # A composite program remains composite even when its verification
        # lane has run.  Only an actually isolated one-axis contrast can
        # become causal evidence for that axis.
        if len(footprint) == 1:
            evidence_class = (
                "CAUSAL_SUPPORT"
                if episode.failure_class is ResearchFailureClassV1.NONE
                else "CAUSAL_REFUTATION"
            )
            causal_credit_allowed = True
        else:
            evidence_class = (
                "DESCRIPTIVE_SUPPORT"
                if episode.failure_class is ResearchFailureClassV1.NONE
                else "DESCRIPTIVE_REFUTATION"
            )
            causal_credit_allowed = False
            confounds.append("COMPOSITE_FOOTPRINT_REQUIRES_ISOLATION")
    else:
        evidence_class = (
            "DESCRIPTIVE_SUPPORT"
            if episode.failure_class is ResearchFailureClassV1.NONE
            else "DESCRIPTIVE_REFUTATION"
        )
        causal_credit_allowed = False
    contrast = route.get("core_mechanism_contrast")
    if not isinstance(contrast, str) or not contrast.strip():
        contrast = route.get("mechanism_contrast")
    if (not isinstance(contrast, str) or not contrast.strip()) and selected.spec is not None:
        contrast = selected.spec.mechanism_change
    projected_axis = _axis_values(
        event.mechanism_axis,
        axis_universe=axis_universe,
    )
    normalized_axis = projected_axis[0] if projected_axis else event.mechanism_axis
    return replace(
        event,
        mechanism_axis=normalized_axis or event.mechanism_axis,
        mechanism_axis_footprint=footprint,
        evidence_class=evidence_class,
        failure_class=failure_class,
        unresolved_confounding=tuple(dict.fromkeys(confounds)),
        core_mechanism_contrast=(
            str(contrast).strip()
            if isinstance(contrast, str) and contrast.strip()
            else None
        ),
        causal_credit_allowed=causal_credit_allowed,
    )


def _event_updates_search_utility(
    route: Mapping[str, Any],
    event: SearchUtilityEventV2,
    episode: TypedResearchEpisodeV1 | None,
) -> bool:
    if episode is None:
        return False
    return observation_updates_search_utility(
        {
            **dict(route),
            "common_outcome_class": event.common_outcome_class,
            "unresolved_confounding": event.unresolved_confounding,
        }
    )


def _historical_axis_state(
    *,
    context: ResearchContext | None,
    event: SearchUtilityEventV2,
    episode: TypedResearchEpisodeV1 | None,
    route: Mapping[str, Any],
    task_record: ResearchTaskRecordV2 | None = None,
    axis_universe: tuple[str, ...] = MECHANISM_AXIS_UNIVERSE_V1,
) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...]]:
    scores = {axis: 0.0 for axis in axis_universe}
    measured: set[str] = set()
    rows: list[Mapping[str, Any]] = []
    utility_neutralizations: tuple[Mapping[str, Any], ...] = ()
    if context is not None:
        global_memory = context.scientific_memory.get("global_memory", {})
        if isinstance(global_memory, Mapping):
            raw_rows = global_memory.get("search_observations", ())
            if not isinstance(raw_rows, (tuple, list)) or not raw_rows:
                raw_rows = global_memory.get("scientific_observations", ())
            if isinstance(raw_rows, (tuple, list)):
                rows.extend(row for row in raw_rows if isinstance(row, Mapping))
            raw_neutralizations = global_memory.get("utility_neutralizations", ())
            if isinstance(raw_neutralizations, (tuple, list)):
                utility_neutralizations = tuple(
                    row for row in raw_neutralizations if isinstance(row, Mapping)
                )
    current_round_index = (
        context.round_index - 1
        if context is not None and episode is not None
        else None
    )
    current_event_recorded = False
    for row in rows:
        if not observation_updates_search_utility(
            row, neutralizations=utility_neutralizations
        ):
            continue
        is_current_event = (
            current_round_index is not None
            and row.get("round_index") == current_round_index
            and row.get("candidate_semantic_digest")
            == event.candidate_semantic_digest
        )
        if is_current_event:
            current_event_recorded = True
        footprint = _axis_values(
            row.get("mechanism_axis_footprint", ()),
            axis_universe=axis_universe,
        )
        if not footprint:
            footprint = _axis_values(
                row.get("mechanism_axis"),
                axis_universe=axis_universe,
            )
        if (
            is_current_event
            and task_record is not None
            and row.get("guard_claim_state")
            in {"PRELIMINARY_NONPOSITIVE", "REFUTED"}
        ):
            task_footprint = _axis_values(
                _program_mechanism_values(task_record.mechanism_program),
                axis_universe=axis_universe,
            )
            if task_footprint:
                footprint = task_footprint
        evidence = str(row.get("evidence_class", ""))
        causal = bool(row.get("causal_credit_allowed"))
        if evidence in {"CAUSAL_SUPPORT", "CAUSAL_REFUTATION"} and causal and len(footprint) == 1:
            increment = 1.0 if evidence == "CAUSAL_SUPPORT" else -1.0
        elif (
            isinstance(row.get("comparator_delta"), (int, float))
            and not isinstance(row.get("comparator_delta"), bool)
            and math.isfinite(float(row["comparator_delta"]))
            and footprint
        ):
            # A real dev delta is descriptive quality evidence even when the
            # episode cannot support causal attribution. Split composite
            # observations across their declared footprint so magnitude, not
            # prose classification, guides the next discovery slate.
            increment = float(row["comparator_delta"]) / len(footprint)
        elif (
            row.get("guard_claim_state")
            in {"PRELIMINARY_NONPOSITIVE", "REFUTED"}
            and isinstance(row.get("guard_confidence_weight"), (int, float))
            and not isinstance(row.get("guard_confidence_weight"), bool)
            and math.isfinite(float(row["guard_confidence_weight"]))
            and isinstance(row.get("comparator_delta"), (int, float))
            and not isinstance(row.get("comparator_delta"), bool)
            and float(row["comparator_delta"]) < 0.0
            and footprint
        ):
            increment = -float(row["guard_confidence_weight"]) / len(footprint)
        else:
            increment = 0.0
        for axis in footprint:
            measured.add(axis)
            scores[axis] += increment
    event_policy_footprint = _axis_values(
        event.mechanism_axis_footprint,
        axis_universe=axis_universe,
    )
    if episode is not None and not current_event_recorded:
        for axis in event_policy_footprint:
            measured.add(axis)
            if (
                event.evidence_class in {"CAUSAL_SUPPORT", "CAUSAL_REFUTATION"}
                and event.causal_credit_allowed
                and len(event.mechanism_axis_footprint) == 1
            ):
                scores[axis] += (
                    1.0
                    if event.evidence_class == "CAUSAL_SUPPORT"
                    else -1.0
                )
            elif event.comparator_delta != "NOT_AVAILABLE" and event_policy_footprint:
                scores[axis] += float(event.comparator_delta) / len(
                    event_policy_footprint
                )
    if context is None or episode is not None:
        measured.update(event_policy_footprint if episode is not None else ())
    uncovered = tuple(
        axis for axis in axis_universe if axis not in measured
    )
    explicit_followups = _axis_values(
        route.get("next_discriminative_axes", ()),
        axis_universe=axis_universe,
    )
    # A measured footprint is historical context, not an automatic follow-up
    # request.  Only an explicit next-axis task may receive follow-up priority.
    followups = explicit_followups
    return scores, uncovered, tuple(dict.fromkeys(followups))


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
    context: ResearchContext | None = None,
    route: Mapping[str, Any] | None = None,
    episode: TypedResearchEpisodeV1 | None = None,
    task_record: ResearchTaskRecordV2 | None = None,
) -> dict[str, Any]:
    route = route if isinstance(route, Mapping) else {}
    parameters = dict(policy.acquisition_parameters)
    prior_rates = parameters.get("producer_useful_rates")
    useful_rates = (
        {role: float(prior_rates[role]) for role in DISCOVERY_PRODUCERS}
        if isinstance(prior_rates, Mapping)
        and set(prior_rates) == set(DISCOVERY_PRODUCERS)
        else dict(policy.producer_token_allocation)
    )
    current_spec = outcomes[selected_role].spec
    selected_is_current_spec = (
        current_spec is not None
        and selected.spec is not None
        and current_spec.digest == selected.spec.digest
    )
    delayed_provenance = _selected_has_delayed_provenance(selected, outcomes)
    causal_credit_allowed = bool(event.causal_credit_allowed) and (
        episode is None
        or episode.failure_class is not ResearchFailureClassV1.INCONCLUSIVE
    )
    current_utility_allowed = observation_updates_search_utility(
        {
            **dict(route),
            "common_outcome_class": event.common_outcome_class,
            "unresolved_confounding": event.unresolved_confounding,
        }
    )
    if (
        (selected_is_current_spec or delayed_provenance)
        and scientific_episode
        and event.comparator_delta != "NOT_AVAILABLE"
        and current_utility_allowed
    ):
        effect = float(event.comparator_delta)
        useful_rates[selected_role] = max(
            0.0,
            min(1.0, useful_rates[selected_role] + effect),
        )
    axis_universe = tuple(policy.mechanism_axis_targeting)
    focused_axes: tuple[str, ...] | None = None
    if set(axis_universe).isdisjoint(MECHANISM_AXIS_UNIVERSE_V1):
        focused_axes = axis_universe
    if context is not None and focused_axes is not None:
        question_axes = _axis_values(
            tuple(
                question.get("mechanism_axis")
                for question in context.unresolved_questions
                if isinstance(question, Mapping)
            ),
            axis_universe=axis_universe,
        )
        if question_axes:
            focused_axes = question_axes
    axis_scores, uncovered_axes, causal_followup_axes = _historical_axis_state(
        context=context,
        event=event,
        episode=episode,
        route=route,
        task_record=task_record,
        axis_universe=axis_universe,
    )
    quality_rows: list[Mapping[str, Any]] = []
    utility_neutralizations: tuple[Mapping[str, Any], ...] = ()
    if context is not None:
        global_memory = context.scientific_memory.get("global_memory", {})
        raw_quality_rows = (
            global_memory.get("scientific_observations", ())
            if isinstance(global_memory, Mapping)
            else ()
        )
        if isinstance(raw_quality_rows, (tuple, list)):
            quality_rows = [
                row for row in raw_quality_rows if isinstance(row, Mapping)
            ]
        raw_neutralizations = (
            global_memory.get("utility_neutralizations", ())
            if isinstance(global_memory, Mapping)
            else ()
        )
        if isinstance(raw_neutralizations, (tuple, list)):
            utility_neutralizations = tuple(
                row for row in raw_neutralizations if isinstance(row, Mapping)
            )

    if utility_neutralizations:
        # Rebuild the clamped selected-role aggregate from retained observations;
        # subtracting a newly neutralized historical effect is not reversible.
        useful_rates = dict(policy.producer_token_allocation)
        for row in quality_rows:
            role = row.get("producer_role")
            delta = row.get("comparator_delta")
            if (
                isinstance(role, str)
                and role in useful_rates
                and isinstance(delta, (int, float))
                and not isinstance(delta, bool)
                and math.isfinite(float(delta))
                and observation_updates_search_utility(
                    row, neutralizations=utility_neutralizations
                )
            ):
                useful_rates[role] = max(
                    0.0, min(1.0, useful_rates[role] + float(delta))
                )

    def mean_quality(identity_field: str) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for row in quality_rows:
            if not observation_updates_search_utility(
                row, neutralizations=utility_neutralizations
            ):
                continue
            identity = row.get(identity_field)
            delta = row.get("comparator_delta")
            if (
                not isinstance(identity, str)
                or not identity
                or isinstance(delta, bool)
                or not isinstance(delta, (int, float))
                or not math.isfinite(float(delta))
            ):
                continue
            totals[identity] = totals.get(identity, 0.0) + float(delta)
            counts[identity] = counts.get(identity, 0) + 1
        return {
            identity: totals[identity] / counts[identity]
            for identity in sorted(totals)
        }
    attempted_family_digests: list[str] = []
    attempted_experiment_digests: list[str] = []
    if context is not None:
        global_memory = context.scientific_memory.get("global_memory", {})
        attempts = (
            global_memory.get("round_attempts", ())
            if isinstance(global_memory, Mapping)
            else ()
        )
        if isinstance(attempts, (tuple, list)):
            for attempt in attempts:
                if not isinstance(attempt, Mapping):
                    continue
                family = attempt.get("effective_family_digest")
                experiment = attempt.get("effective_experiment_digest")
                if isinstance(family, str) and family not in attempted_family_digests:
                    attempted_family_digests.append(family)
                if isinstance(experiment, str) and experiment not in attempted_experiment_digests:
                    attempted_experiment_digests.append(experiment)
        for raw_values in (
            context.scientific_memory.get(
                "attempted_effective_family_digests", ()
            ),
            (
                global_memory.get("attempted_effective_family_digests", ())
                if isinstance(global_memory, Mapping)
                else ()
            ),
        ):
            if isinstance(raw_values, (tuple, list)):
                for family in raw_values:
                    if (
                        isinstance(family, str)
                        and family not in attempted_family_digests
                    ):
                        attempted_family_digests.append(family)
        for raw_values in (
            context.scientific_memory.get(
                "attempted_effective_experiment_digests", ()
            ),
            (
                global_memory.get("attempted_effective_experiment_digests", ())
                if isinstance(global_memory, Mapping)
                else ()
            ),
        ):
            if isinstance(raw_values, (tuple, list)):
                for experiment in raw_values:
                    if (
                        isinstance(experiment, str)
                        and experiment not in attempted_experiment_digests
                    ):
                        attempted_experiment_digests.append(experiment)
    next_task = (
        episode.next_discriminative_test
        if episode is not None
        else route.get("next_discriminative_test")
    )
    engineering_failure = None
    if not scientific_episode:
        engineering_failure = {
            "failure_class": event.common_outcome_class,
            "typed_blocker_class": event.typed_blocker_class,
            "capability_family": _family_key(event, route),
        }
    return {
        "producer_useful_rates": useful_rates,
        "producer_quality_scores": mean_quality("producer_role"),
        "family_quality_scores": mean_quality("effective_family_digest"),
        "experiment_quality_scores": mean_quality(
            "effective_experiment_digest"
        ),
        "measured_axes": (
            _axis_values(
                event.mechanism_axis_footprint,
                axis_universe=axis_universe,
            )
            if scientific_episode and current_utility_allowed
            else ()
        ),
        **({"focused_axes": focused_axes} if focused_axes is not None else {}),
        "uncovered_axes": uncovered_axes,
        "causal_followup_axes": causal_followup_axes,
        "axis_scores": axis_scores,
        "calibration_error": abs(required_probability - _runnable_probability(event)),
        "next_discriminative_task": next_task,
        "core_mechanism_contrast": event.core_mechanism_contrast,
        "unresolved_confounding": event.unresolved_confounding,
        "last_evidence_class": event.evidence_class,
        "last_failure_class": event.failure_class,
        "last_axis_footprint": event.mechanism_axis_footprint,
        "research_task_record": (
            task_record.to_dict() if task_record is not None else None
        ),
        "attempted_family_digests": tuple(attempted_family_digests),
        "attempted_experiment_digests": tuple(attempted_experiment_digests),
        "engineering_failure": engineering_failure,
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
    elif selected.source_mechanism_program is not None:
        program = _mapping(
            selected.source_mechanism_program,
            field="source_mechanism_program",
        )
        parent_binding = route.get("lineage_parent_binding")
        parent = (
            parent_binding.get("candidate_id")
            if isinstance(parent_binding, Mapping)
            else route.get("parent_candidate_id")
        )
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
        "mechanism_axis_footprint": event.mechanism_axis_footprint,
        "evidence_class": event.evidence_class,
        "failure_class": event.failure_class,
        "unresolved_confounding": event.unresolved_confounding,
        "core_mechanism_contrast": event.core_mechanism_contrast,
        "causal_credit_allowed": event.causal_credit_allowed,
        "capability_family": route.get(
            "capability_family",
            route.get("candidate_family", route.get("family", event.mechanism_axis)),
        ),
        "owner_arm_instance_id": owner,
        "legacy_task_type": route.get("next_task_type"),
        # Preserve the existing OpenSpec fields needed to make the next task
        # executable and falsifiable; the task record remains the single
        # structured task identity consumed downstream.
        "expected_observable": selected.spec.expected_evidence,
        "falsifier": selected.spec.falsifier,
        "matched_control_requirement": selected.spec.matched_control_requirement,
        "parent_candidate_id": parent,
        "comparator_identity": comparator,
        "task_operation": operation.value,
    }
    effective_identity = route.get("search_space_execution_binding")
    effective_identity = (
        effective_identity if isinstance(effective_identity, Mapping) else {}
    )
    metadata_payload.update(
        {
            "adapter_id": effective_identity.get("adapter_id"),
            "binding_ref": effective_identity.get("binding_ref"),
            "binding_digest": effective_identity.get("binding_digest"),
            "effective_experiment_digest": effective_identity.get(
                "effective_experiment_digest"
            ),
            "effective_family_digest": effective_identity.get(
                "effective_family_digest"
            ),
        }
    )
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
        if event.candidate_value is not None:
            candidate_value = float(event.candidate_value)
        elif current_value is not None:
            candidate_value = current_value + delta
        else:
            candidate_value = None
        frontier_delta = (
            candidate_value - current_value
            if candidate_value is not None and current_value is not None
            else None
        )
        causal_credit_allowed = (
            bool(event.causal_credit_allowed)
            if event.evidence_class != "UNCLASSIFIED"
            else (
                episode is not None
                and episode.failure_class
                in {
                    ResearchFailureClassV1.NONE,
                    ResearchFailureClassV1.MECHANISM,
                }
                and len(event.mechanism_axis_footprint) == 1
            )
        ) and (
            episode is None
            or episode.failure_class is not ResearchFailureClassV1.INCONCLUSIVE
        )
        # Search-frontier promotion is a performance fact, not a causal claim.
        # A valid metric may improve the incumbent even when a single-seed or
        # multi-axis episode is only descriptive. Causal credit continues to
        # control mechanism/family attribution below.
        changed = (
            frontier_delta is not None
            and frontier_delta > 0.0
        )
        event_record = canonical_value(
            {
                "round_index": context.round_index,
                "candidate_id": event.candidate_id,
                "candidate_semantic_digest": event.candidate_semantic_digest,
                "mechanism_axis": event.mechanism_axis,
                "comparator_delta": delta,
                "candidate_value": candidate_value,
                "frontier_delta": frontier_delta,
                "frontier_updated": changed,
                "mechanism_axis_footprint": event.mechanism_axis_footprint,
                "evidence_class": event.evidence_class,
                "failure_class": event.failure_class or episode.failure_class.value,
                "unresolved_confounding": event.unresolved_confounding,
                "core_mechanism_contrast": event.core_mechanism_contrast,
                "causal_credit_allowed": causal_credit_allowed,
                "mechanism_effect_update_allowed": causal_credit_allowed,
                # Search utility is an observed development-direction fact,
                # not a causal or publication claim.  Keep this authority
                # separate so single-seed INCONCLUSIVE wording cannot erase
                # real positive/negative feedback from the next search round.
                "search_utility_update_allowed": _event_updates_search_utility(
                    route, event, episode
                ),
                "directional_evidence_class": (
                    "OBSERVED_IMPROVEMENT"
                    if delta > 0.0
                    else (
                        "OBSERVED_REGRESSION"
                        if delta < 0.0
                        else "OBSERVED_NEUTRAL"
                    )
                ),
                "producer_role": selected.producer_role,
                "effective_family_digest": route.get(
                    "effective_family_digest"
                ),
                "effective_experiment_digest": route.get(
                    "effective_experiment_digest"
                ),
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
            lineage_candidate_id = route.get("lineage_candidate_id")
            lineage_program_digest = route.get("lineage_program_digest")
            if (
                isinstance(lineage_candidate_id, str)
                and lineage_candidate_id
                and isinstance(lineage_program_digest, str)
            ):
                lineage_binding = canonical_value(
                    {
                        "candidate_id": lineage_candidate_id,
                        "program_digest": _digest(
                            lineage_program_digest,
                            field="lineage_program_digest",
                        ),
                    }
                )
                frontier["lineage_parent_binding"] = lineage_binding
                global_bank["best_observed_candidate_value"] = candidate_value
                global_bank["best_observed_candidate_id"] = event.candidate_id
                global_bank["lineage_parent_binding"] = lineage_binding
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
        if not causal_credit_allowed:
            # Descriptive evidence still moves the performance frontier, but
            # it does not update mechanism-effect banks. Only a genuinely
            # improved candidate is worth a later confirmation action.
            if changed:
                confirmation = _mapping_copy(
                    confirmation_bank.get(event.candidate_id)
                )
                seeds = list(confirmation.get("observed_seeds", ()))
                if event.observation_seed not in seeds:
                    seeds.append(event.observation_seed)
                confirmation["observed_seeds"] = tuple(seeds)
                confirmation["observed_seed_count"] = len(seeds)
                confirmation["status"] = "PENDING"
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

        family = _family_key(event, route)
        family_entry = _mapping_copy(family_bank.get(family))
        family_entry.setdefault("observations", [])
        family_observations = list(family_entry["observations"])
        family_observations.append(event_record)
        family_entry["observations"] = tuple(family_observations)
        family_entry["last_delta"] = delta
        family_entry["last_candidate_id"] = event.candidate_id
        if candidate_value is not None:
            family_entry["best_value"] = max(
                float(family_entry.get("best_value", candidate_value)),
                candidate_value,
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

        if changed:
            confirmation = _mapping_copy(confirmation_bank.get(event.candidate_id))
            seeds = list(confirmation.get("observed_seeds", ()))
            if event.observation_seed not in seeds:
                seeds.append(event.observation_seed)
            confirmation["observed_seeds"] = tuple(seeds)
            confirmation["observed_seed_count"] = len(seeds)
            confirmation["status"] = "PENDING"
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


def _retain_routed_active_task(
    queue: ResearchTaskQueueV2,
    route: Mapping[str, Any],
) -> ResearchTaskQueueV2:
    """Restore the active policy task before applying round evidence.

    Full Helix may project an exact task into discovery policy while its
    auxiliary Evidence Port owns execution.  A NOOP allocation can therefore
    leave the persisted queue empty even though the task remains active in the
    route.  Rehydrate that exact record so an unrelated discovery result cannot
    silently replace it; a matching result can still satisfy it normally.
    """

    raw_task = route.get("active_task_record")
    if not isinstance(raw_task, Mapping):
        return queue
    task = ResearchTaskRecordV2.from_dict(raw_task)
    active_task_id = route.get("active_task_id")
    if active_task_id is not None and str(active_task_id) != task.task_id:
        raise ScientificInterpreterError(
            "active task id differs from the routed active task record"
        )
    return queue.enqueue(task)


def _confirmation_binding(
    *,
    prefix: str,
    route: Mapping[str, Any],
    task_record: ResearchTaskRecordV2,
) -> Mapping[str, Any] | None:
    candidate_id = route.get(f"{prefix}_candidate_id")
    semantic_digest = route.get(f"{prefix}_semantic_digest")
    program = route.get(f"{prefix}_program")
    program_digest = route.get(f"{prefix}_program_digest")
    supplied = tuple(
        item is not None
        for item in (candidate_id, semantic_digest, program, program_digest)
    )
    if any(supplied):
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
        return canonical_value(
            {
                "candidate_id": _text(
                    candidate_id,
                    field=f"{prefix}_candidate_id",
                ),
                "candidate_semantic_digest": validated_semantic,
                "mechanism_program": program,
                "mechanism_program_digest": validated_program,
                "binding_origin": "ROUTE_EXECUTABLE_PAYLOAD",
            }
        )

    declared = task_record.metadata.get(f"{prefix}_binding")
    if not isinstance(declared, Mapping):
        return None
    required_fields = (
        "candidate_id",
        "candidate_semantic_digest",
        "mechanism_program",
        "mechanism_program_digest",
        "binding_origin",
    )
    if any(declared.get(field_name) is None for field_name in required_fields):
        raise ScientificInterpreterError(
            f"declared {prefix} binding is incomplete"
        )
    declared_program = declared["mechanism_program"]
    if not isinstance(declared_program, Mapping):
        raise ScientificInterpreterError(
            f"declared {prefix} mechanism program is invalid"
        )
    declared_program_digest = validate_sha256(
        declared["mechanism_program_digest"],
        field_name=f"declared_{prefix}_program_digest",
    )
    if sha256_digest(declared_program) != declared_program_digest:
        raise ScientificInterpreterError(
            f"declared {prefix} executable program digest mismatch"
        )
    return canonical_value(dict(declared))


def _task_matches_metric_event(
    *,
    task: ResearchTaskRecordV2,
    event: SearchUtilityEventV2,
    route: Mapping[str, Any],
) -> bool:
    execution_state = task.metadata.get("execution_state")
    if (
        task.operation
        in {
            ResearchTaskOperationV2.MATCHED_CONTROL,
            ResearchTaskOperationV2.MECHANISM_OFF,
        }
        and execution_state == "AWAITING_CANDIDATE_BINDING"
    ):
        # A durable intent task is allowed to bind only when the next route
        # names that exact task and carries the newly executed program.  The
        # frontier candidate stored on the task is an anchor, never the
        # predicted identity of the future control/ablation.
        if route.get("satisfies_task_id") != task.task_id:
            return False
        frontier_candidate_id = task.metadata.get("frontier_candidate_id")
        frontier_semantic_digest = task.metadata.get(
            "frontier_candidate_semantic_digest"
        )
        if (
            event.candidate_id == frontier_candidate_id
            and event.candidate_semantic_digest == frontier_semantic_digest
        ):
            return False
        program = route.get("mechanism_program")
        identity = route.get("search_space_execution_binding")
        if not isinstance(program, Mapping) or not isinstance(identity, Mapping):
            return False
        if not identity.get("effective_experiment_digest") or not identity.get(
            "effective_family_digest"
        ):
            return False
        parent = route.get(
            "selected_candidate_parent_id",
            route.get("parent_candidate_id"),
        )
        if task.parent_candidate_id not in (None, parent):
            return False
        if route.get("comparator_identity") != task.comparator_identity:
            return False
        operation = route.get("task_operation")
        if operation is not None:
            operation = getattr(operation, "value", operation)
            if str(operation) != task.operation.value:
                return False
        return True

    binding = task.metadata.get("execution_binding")
    if (
        isinstance(binding, Mapping)
        and binding.get("binding_origin") in {
            "ACTIVE_PROFILE",
            "PACKAGE_CONTROL_CATALOG",
        }
    ):
        expected_program_digest = binding.get("bound_program_digest")
        program = route.get("mechanism_program")
        return (
            task.candidate_semantic_digest == event.candidate_semantic_digest
            and isinstance(expected_program_digest, str)
            and isinstance(program, Mapping)
            and expected_program_digest == task.mechanism_program_digest
            and canonical_value(program) == canonical_value(task.mechanism_program)
        )
    return (
        task.candidate_id == event.candidate_id
        and task.candidate_semantic_digest == event.candidate_semantic_digest
    )


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


def _unbound_confirmation_task(
    base: ResearchTaskRecordV2,
    *,
    prefix: str,
    operation: ResearchTaskOperationV2,
    priority: float,
) -> ResearchTaskRecordV2:
    """Keep a control/ablation intent executable before its candidate exists."""

    return _confirmation_task(
        base,
        operation=operation,
        required_seed_or_control="AWAITING_CONTROL_PROPOSAL",
        priority=priority,
        metadata={
            "confirmation_target": prefix.upper(),
            "execution_state": "AWAITING_CANDIDATE_BINDING",
            "binding_requirement": "NEEDS_PROPOSAL",
            "frontier_candidate_id": base.candidate_id,
            "frontier_candidate_semantic_digest": (
                base.candidate_semantic_digest
            ),
            "frontier_candidate_program_digest": base.mechanism_program_digest,
            "frontier_parent_candidate_id": base.parent_candidate_id,
            "frontier_comparator_identity": base.comparator_identity,
            "core_mechanism_contrast": base.metadata.get(
                "core_mechanism_contrast"
            ),
            "next_discriminative_test": base.metadata.get(
                "next_discriminative_test"
            ),
            "task_operation": operation.value,
        },
    )


def _bind_unbound_confirmation_task(
    task: ResearchTaskRecordV2,
    *,
    event: SearchUtilityEventV2,
    route: Mapping[str, Any],
) -> ResearchTaskRecordV2:
    """Bind one durable intent to the exact candidate that actually ran."""

    if task.metadata.get("execution_state") != "AWAITING_CANDIDATE_BINDING":
        return task
    program = route.get("mechanism_program")
    if not isinstance(program, Mapping):
        raise ScientificInterpreterError(
            "candidate-bound confirmation lacks an executable program"
        )
    identity = route.get("search_space_execution_binding")
    if not isinstance(identity, Mapping):
        raise ScientificInterpreterError(
            "candidate-bound confirmation lacks adapter attestation"
        )
    program_digest = sha256_digest(program)
    execution_binding = canonical_value(
        {
            "binding_origin": "ROUTE_EXECUTABLE_PAYLOAD",
            "candidate_id": event.candidate_id,
            "candidate_semantic_digest": event.candidate_semantic_digest,
            "mechanism_program": program,
            "mechanism_program_digest": program_digest,
            "effective_experiment_digest": identity[
                "effective_experiment_digest"
            ],
            "effective_family_digest": identity["effective_family_digest"],
            "adapter_id": identity.get("adapter_id"),
            "binding_ref": identity.get("binding_ref"),
            "binding_digest": identity.get("binding_digest"),
            "opaque_binding": identity.get("opaque_binding"),
            "parent_candidate_id": route.get(
                "selected_candidate_parent_id",
                route.get("parent_candidate_id"),
            ),
            "comparator_identity": route.get("comparator_identity"),
        }
    )
    metadata = dict(task.metadata)
    metadata.update(
        {
            "execution_state": "BOUND",
            "execution_binding": execution_binding,
            "bound_candidate_id": event.candidate_id,
            "bound_candidate_semantic_digest": event.candidate_semantic_digest,
            "bound_program_digest": program_digest,
            "effective_experiment_digest": identity[
                "effective_experiment_digest"
            ],
            "effective_family_digest": identity["effective_family_digest"],
            "adapter_id": identity.get("adapter_id"),
            "binding_ref": identity.get("binding_ref"),
            "binding_digest": identity.get("binding_digest"),
        }
    )
    return replace(
        task,
        candidate_id=event.candidate_id,
        candidate_semantic_digest=event.candidate_semantic_digest,
        mechanism_program_digest=program_digest,
        mechanism_program=program,
        metadata=metadata,
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
    active_task_id = route.get("active_task_id")
    observed_control = route.get(
        "observed_control_id",
        route.get("control_id", route.get("matched_control_id")),
    )
    observed_mechanism_off = route.get("observed_mechanism_off_id")
    metric_bearing = (
        episode is not None and event.comparator_delta != "NOT_AVAILABLE"
    )
    close_task_id = route.get("close_task_id")
    if close_task_id is not None:
        close_task_id = str(close_task_id)
        existing = queue.get(close_task_id)
        if existing is not None and existing.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.close(
                close_task_id,
                reason=str(
                    route.get(
                        "close_task_reason",
                        "CLOSED_BY_ROUND_TASK_RESOLUTION",
                    )
                ),
            )
            closed.append(close_task_id)

    confirmation_bank = context.frontier.get("confirmation", {})
    confirmation = (
        confirmation_bank.get(event.candidate_id, {})
        if isinstance(confirmation_bank, Mapping)
        else {}
    )
    confirmation = confirmation if isinstance(confirmation, Mapping) else {}
    observed_seeds = set(confirmation.get("observed_seeds", ()))
    observed_seeds.add(event.observation_seed)
    required_seed_count = int(
        route.get(
            "required_seed_count",
            confirmation.get("required_seed_count", 2),
        )
    )
    seed_threshold_reached = (
        metric_bearing and len(observed_seeds) >= required_seed_count
    )
    seed_lane_operations = {
        ResearchTaskOperationV2.NEW_SEED,
        ResearchTaskOperationV2.REPRODUCE,
        ResearchTaskOperationV2.MOVE_ON,
    }
    satisfied_confirmation_task = False

    for existing in queue.tasks:
        if existing.status not in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            continue
        matches_identity = _task_matches_metric_event(
            task=existing,
            event=event,
            route=route,
        )
        matches_required = False
        if explicit_satisfied is not None:
            matches_required = (
                existing.task_id == str(explicit_satisfied)
                and matches_identity
                and (
                    metric_bearing
                    or existing.operation is ResearchTaskOperationV2.REPAIR
                )
            )
        elif matches_identity:
            if existing.operation in {
                ResearchTaskOperationV2.NEW_SEED,
                ResearchTaskOperationV2.REPRODUCE,
            }:
                matches_required = metric_bearing and (
                    existing.required_seed_or_control == event.observation_seed
                    or (
                        existing.required_seed_or_control
                        == _NEXT_DEVELOPMENT_SEED
                        and event.observation_seed not in existing.evidence_present
                    )
                )
            elif existing.operation is ResearchTaskOperationV2.MATCHED_CONTROL:
                matches_required = (
                    metric_bearing
                    and (
                        (
                            active_task_id is not None
                            and existing.task_id == str(active_task_id)
                        )
                        or (
                            observed_control is not None
                            and existing.required_seed_or_control
                            == str(observed_control)
                        )
                    )
                )
            elif existing.operation is ResearchTaskOperationV2.MECHANISM_OFF:
                matches_required = (
                    metric_bearing
                    and (
                        (
                            active_task_id is not None
                            and existing.task_id == str(active_task_id)
                        )
                        or (
                            observed_mechanism_off is not None
                            and existing.required_seed_or_control
                            == str(observed_mechanism_off)
                        )
                    )
                )
            elif existing.operation is ResearchTaskOperationV2.REPAIR:
                matches_required = bool(episode is None)
        if not matches_required:
            continue
        if existing.operation in {
            ResearchTaskOperationV2.MATCHED_CONTROL,
            ResearchTaskOperationV2.MECHANISM_OFF,
        }:
            satisfied_confirmation_task = True
        if existing.metadata.get("execution_state") == "AWAITING_CANDIDATE_BINDING":
            bound = _bind_unbound_confirmation_task(
                existing,
                event=event,
                route=route,
            )
            queue = queue._replace(bound)
        queue = queue.satisfy(
            existing.task_id,
            evidence=(event.observation_seed,),
            reason="REQUIRED_TASK_EVIDENCE_OBSERVED",
        )
        satisfied.append(existing.task_id)

    if metric_bearing and active_task_id is None:
        # Ordinary discovery remains the primary schedule.  A generic
        # follow-up suggestion that was not selected before a later fresh
        # candidate ran is retained as closed search history, not an immortal
        # obligation or an extra physical worker.  Evidence-Guard allocations
        # remain owned by their explicit auxiliary ledger.
        for existing in queue.tasks:
            if (
                existing.status
                in {
                    ResearchTaskStatusV2.PENDING,
                    ResearchTaskStatusV2.ACTIVE,
                }
                and existing.created_round < context.round_index
                and existing.candidate_semantic_digest
                != event.candidate_semantic_digest
                and existing.metadata.get("helix_allocation_action_id") is None
                and existing.operation
                in {
                    ResearchTaskOperationV2.NEW_SEED,
                    ResearchTaskOperationV2.REPRODUCE,
                    ResearchTaskOperationV2.MATCHED_CONTROL,
                    ResearchTaskOperationV2.MECHANISM_OFF,
                }
            ):
                queue = queue.close(
                    existing.task_id,
                    reason="SUPERSEDED_BY_FRESH_DISCOVERY",
                )
                closed.append(existing.task_id)

    if episode is not None and not frontier_updated:
        # A non-improving metric is search feedback, not a reason to spend the
        # next discovery round reproducing or dissecting the failed candidate.
        # Preserve the task identity as closed evidence and retire any open
        # same-candidate confirmation work.  The full episode remains in
        # scientific memory for the Providers and meta learner.
        closed_record = replace(
            task_record,
            status=ResearchTaskStatusV2.CLOSED,
            close_reason="NON_FRONTIER_RESULT",
            missing_seed_count=0,
        )
        prior = queue.get(closed_record.task_id)
        if prior is None:
            queue = queue.enqueue(closed_record)
            created.append(closed_record.task_id)
            closed.append(closed_record.task_id)
        elif prior.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.close(prior.task_id, reason="NON_FRONTIER_RESULT")
            closed.append(prior.task_id)
        for existing in queue.tasks:
            if (
                existing.status
                in {
                    ResearchTaskStatusV2.PENDING,
                    ResearchTaskStatusV2.ACTIVE,
                }
                and existing.candidate_semantic_digest
                == task_record.candidate_semantic_digest
                and existing.operation is not ResearchTaskOperationV2.REPAIR
            ):
                queue = queue.close(
                    existing.task_id,
                    reason="NON_FRONTIER_RESULT",
                )
                closed.append(existing.task_id)
    elif seed_threshold_reached and task_record.operation in seed_lane_operations:
        # Once the declared independent-seed requirement is met, preserve the
        # next-task identity as closed and retire any stale same-candidate seed
        # work.  Confirmation remains pending until its declared controls are
        # observed; missing control bindings must not turn into unbounded seed
        # repetition.
        prior = queue.get(task_record.task_id)
        if prior is None:
            queue = queue.enqueue(
                replace(
                    task_record,
                    status=ResearchTaskStatusV2.CLOSED,
                    close_reason="CANDIDATE_SEED_THRESHOLD_REACHED",
                    missing_seed_count=0,
                )
            )
            created.append(task_record.task_id)
            closed.append(task_record.task_id)
        elif prior.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.close(
                prior.task_id,
                reason="CANDIDATE_SEED_THRESHOLD_REACHED",
            )
            closed.append(prior.task_id)
        for existing in queue.tasks:
            if (
                existing.status in {
                    ResearchTaskStatusV2.PENDING,
                    ResearchTaskStatusV2.ACTIVE,
                }
                and existing.candidate_id == event.candidate_id
                and existing.candidate_semantic_digest
                == event.candidate_semantic_digest
                and existing.operation in seed_lane_operations
            ):
                queue = queue.close(
                    existing.task_id,
                    reason="CANDIDATE_SEED_THRESHOLD_REACHED",
                )
                closed.append(existing.task_id)
    elif task_record.operation is ResearchTaskOperationV2.MOVE_ON:
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
        if prior is None:
            queue = queue.enqueue(task_record)
            created.append(task_record.task_id)
        elif prior.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.enqueue(task_record)

    # Confirmation is follow-up work for a promising frontier result, not the
    # default successor of every single-seed metric.  A non-frontier result is
    # already useful negative search feedback; forcing matched-control and
    # mechanism-off tasks here would make discovery refine a failed candidate
    # instead of changing mechanism or direction.
    if episode is not None and frontier_updated:
        followup_base = queue.get(task_record.task_id) or task_record
        next_seed = route.get(
            "confirmation_seed",
            route.get("next_seed_or_control", route.get("required_seed_or_control")),
        )
        next_seed = str(next_seed) if next_seed is not None else None
        if next_seed in {event.observation_seed, _NEXT_DEVELOPMENT_SEED}:
            next_seed = None
        followups = []
        if frontier_updated and not seed_threshold_reached and next_seed is not None:
            followups.append(
                _confirmation_task(
                    followup_base,
                    operation=ResearchTaskOperationV2.NEW_SEED,
                    required_seed_or_control=next_seed,
                    priority=1.0,
                    metadata={
                        "confirmation_target": "NEW_SEED",
                        "frontier_candidate_id": task_record.candidate_id,
                        "frontier_candidate_semantic_digest": (
                            task_record.candidate_semantic_digest
                        ),
                        "frontier_candidate_program_digest": (
                            task_record.mechanism_program_digest
                        ),
                    },
                )
            )
        deferred_requirements: list[str] = []
        followup_availability = route.get(
            "followup_confirmation_availability", {}
        )
        followup_availability = (
            followup_availability
            if isinstance(followup_availability, Mapping)
            else {}
        )
        for prefix, operation, priority in (
            (
                "matched_control",
                ResearchTaskOperationV2.MATCHED_CONTROL,
                0.99,
            ),
            (
                "mechanism_off",
                ResearchTaskOperationV2.MECHANISM_OFF,
                0.98,
            ),
        ):
            adapter_status = followup_availability.get(operation.value)
            if adapter_status == "UNSUPPORTED":
                continue
            binding = _confirmation_binding(
                prefix=prefix,
                route=route,
                task_record=followup_base,
            )
            if binding is None:
                deferred_requirements.append(prefix.upper())
                continue
            if prefix == "matched_control":
                required = str(
                    route.get(
                        "matched_control_id",
                        route.get("matched_control_seed", binding["candidate_id"]),
                    )
                )
            else:
                ablation = binding.get("ablation")
                ablation_id = (
                    ablation.get("ablation_id")
                    if isinstance(ablation, Mapping)
                    else None
                )
                required = str(
                    route.get(
                        "mechanism_off_id",
                        route.get(
                            "mechanism_off_seed",
                            ablation_id or binding["candidate_id"],
                        ),
                    )
                )
            followups.append(
                _confirmation_task(
                    followup_base,
                    operation=operation,
                    required_seed_or_control=required,
                    priority=priority,
                    candidate_id=binding["candidate_id"],
                    candidate_semantic_digest=binding[
                        "candidate_semantic_digest"
                    ],
                    mechanism_program_digest=binding[
                        "mechanism_program_digest"
                    ],
                    mechanism_program=binding["mechanism_program"],
                    metadata={
                        "confirmation_target": prefix.upper(),
                        "execution_state": "OPEN",
                        "execution_binding": binding,
                        "verification_seed": event.observation_seed,
                        "frontier_candidate_id": task_record.candidate_id,
                        "frontier_candidate_semantic_digest": (
                            task_record.candidate_semantic_digest
                        ),
                        "frontier_candidate_program_digest": (
                            task_record.mechanism_program_digest
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
        "closed_task_ids": tuple(dict.fromkeys(closed)),
        "created_task_ids": tuple(created),
        "deferred_requirements": transition_deferred,
        "metric_bearing_evidence": metric_bearing,
        "observed_seed_count": len(observed_seeds),
        "required_seed_count": required_seed_count,
        "seed_threshold_reached": seed_threshold_reached,
        "head_task_id": head.task_id if head is not None else None,
    }


def _apply_confirmation_task_evidence(
    *,
    frontier: Mapping[str, Any],
    queue: ResearchTaskQueueV2,
    transition: Mapping[str, Any],
) -> dict[str, Any]:
    updated = dict(frontier)
    confirmation_bank = _mapping_copy(updated.get("confirmation"))
    changed = False
    for task_id in transition.get("satisfied_task_ids", ()):
        task = queue.get(str(task_id))
        if task is None or task.status is not ResearchTaskStatusV2.SATISFIED:
            continue
        frontier_candidate_id = task.metadata.get("frontier_candidate_id")
        if not isinstance(frontier_candidate_id, str):
            continue
        confirmation = _mapping_copy(
            confirmation_bank.get(frontier_candidate_id)
        )
        exact_task_evidence = {
            "task_id": task.task_id,
            "task_record_digest": task.digest,
            "candidate_id": task.candidate_id,
            "candidate_semantic_digest": task.candidate_semantic_digest,
            "mechanism_program_digest": task.mechanism_program_digest,
            "effective_experiment_digest": task.metadata.get(
                "effective_experiment_digest"
            ),
            "effective_family_digest": task.metadata.get(
                "effective_family_digest"
            ),
            "parent_candidate_id": task.parent_candidate_id,
            "comparator_identity": task.comparator_identity,
            "required_seed_or_control": task.required_seed_or_control,
            "frontier_candidate_id": frontier_candidate_id,
            "frontier_candidate_semantic_digest": task.metadata.get(
                "frontier_candidate_semantic_digest"
            ),
            "frontier_candidate_program_digest": task.metadata.get(
                "frontier_candidate_program_digest"
            ),
            "frontier_effective_experiment_digest": task.metadata.get(
                "effective_experiment_digest"
            ),
            "frontier_effective_family_digest": task.metadata.get(
                "effective_family_digest"
            ),
            "evidence_present": task.evidence_present,
        }
        if task.operation is ResearchTaskOperationV2.MATCHED_CONTROL:
            confirmation["matched_control_status"] = "SATISFIED"
            confirmation["matched_control_task_id"] = task.task_id
            confirmation["matched_control_evidence"] = exact_task_evidence
        elif task.operation is ResearchTaskOperationV2.MECHANISM_OFF:
            confirmation["mechanism_off_status"] = "SATISFIED"
            confirmation["mechanism_off_task_id"] = task.task_id
            confirmation["mechanism_off_evidence"] = exact_task_evidence
        elif task.operation in {
            ResearchTaskOperationV2.NEW_SEED,
            ResearchTaskOperationV2.REPRODUCE,
        }:
            confirmation["replication_evidence"] = exact_task_evidence
        else:
            continue
        changed = True
        seed_count = int(confirmation.get("observed_seed_count", 0))
        required_seed_count = int(confirmation.get("required_seed_count", 2))
        if (
            seed_count >= required_seed_count
            and confirmation.get("matched_control_status") == "SATISFIED"
            and confirmation.get("mechanism_off_status") == "SATISFIED"
        ):
            confirmation["status"] = "CONFIRMED"
        else:
            confirmation["status"] = "PENDING"
        confirmation_bank[frontier_candidate_id] = canonical_value(
            confirmation
        )
    if changed:
        updated["confirmation"] = canonical_value(confirmation_bank)
    return updated


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
    frontier_update_allowed: bool = True,
) -> ResearchContext:
    route = dict(route_metadata) if isinstance(route_metadata, Mapping) else {}
    execution_lane = str(route.get("execution_lane", "DISCOVERY"))
    auxiliary_verification = execution_lane == "AUXILIARY_VERIFICATION"
    frontier = dict(context.frontier)
    frontier_updated = False
    event_record: dict[str, Any] = {}
    if (
        frontier_update_allowed
        and event is not None
        and selected_outcome is not None
    ):
        frontier, frontier_updated, event_record = _frontier_update(
            context=context,
            event=event,
            episode=episode,
            route=route,
            selected=selected_outcome,
        )

    queue = _retain_routed_active_task(
        _task_queue_from_memory(context.scientific_memory),
        route,
    )
    transition: dict[str, Any] = {}
    if task_record is None or event is None:
        close_task_id = route.get("close_task_id")
        existing = queue.get(str(close_task_id)) if close_task_id is not None else None
        if existing is not None and existing.status in {
            ResearchTaskStatusV2.PENDING,
            ResearchTaskStatusV2.ACTIVE,
        }:
            queue = queue.close(
                existing.task_id,
                reason=str(
                    route.get(
                        "close_task_reason",
                        "CLOSED_BY_NON_METRIC_TASK_RESOLUTION",
                    )
                ),
            )
            transition = {
                "satisfied_task_ids": (),
                "closed_task_ids": (existing.task_id,),
                "created_task_ids": (),
                "deferred_requirements": (),
                "metric_bearing_evidence": False,
            }
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
        frontier = _apply_confirmation_task_evidence(
            frontier=frontier,
            queue=queue,
            transition=transition,
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
                "execution_lane": execution_lane,
                "candidate_id": event.candidate_id,
                "candidate_semantic_digest": event.candidate_semantic_digest,
                "producer_role": (
                    selected_outcome.producer_role
                    if selected_outcome is not None
                    else None
                ),
                "effective_family_digest": route.get(
                    "effective_family_digest"
                ),
                "effective_experiment_digest": route.get(
                    "effective_experiment_digest"
                ),
                "observation_seed": event.observation_seed,
                "common_outcome_class": event.common_outcome_class,
                "mechanism_axis": event.mechanism_axis,
                "mechanism_axis_footprint": event.mechanism_axis_footprint,
                "comparator_delta": event.comparator_delta,
                "evidence_class": event.evidence_class,
                "failure_class": event.failure_class,
                "unresolved_confounding": event.unresolved_confounding,
                "core_mechanism_contrast": event.core_mechanism_contrast,
                "causal_credit_allowed": event.causal_credit_allowed,
                "search_utility_update_allowed": _event_updates_search_utility(
                    route, event, episode
                ),
                "mechanism_program": route.get("mechanism_program"),
                "fidelity_classification": route.get("fidelity_classification"),
                "directional_evidence_class": (
                    "OBSERVED_IMPROVEMENT"
                    if isinstance(event.comparator_delta, (int, float))
                    and not isinstance(event.comparator_delta, bool)
                    and float(event.comparator_delta) > 0.0
                    else (
                        "OBSERVED_REGRESSION"
                        if isinstance(event.comparator_delta, (int, float))
                        and not isinstance(event.comparator_delta, bool)
                        and float(event.comparator_delta) < 0.0
                        else "OBSERVED_NEUTRAL"
                    )
                )
                if episode is not None
                else None,
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
                "mechanism_axis_footprint": event.mechanism_axis_footprint,
                "comparator_delta": event.comparator_delta,
                "evidence_class": event.evidence_class,
                "failure_class": event.failure_class,
                "unresolved_confounding": event.unresolved_confounding,
                "core_mechanism_contrast": event.core_mechanism_contrast,
                "evidence_domain": (
                    "SCIENTIFIC_EPISODE" if episode is not None else "RESOURCE_SEARCH"
                ),
                "mechanism_effect_update_allowed": bool(
                    episode is not None and event.causal_credit_allowed
                ),
                "search_utility_update_allowed": executed_observation.get(
                    "search_utility_update_allowed"
                ),
                "directional_evidence_class": executed_observation.get(
                    "directional_evidence_class"
                ),
                "causal_credit_allowed": event.causal_credit_allowed,
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
                    "failure_code": event.typed_blocker_class,
                    "typed_blocker_class": event.typed_blocker_class,
                    "candidate_id": event.candidate_id,
                    "observation_seed": event.observation_seed,
                    "capability_family": _family_key(event, route),
                    "failure_fingerprint": sha256_digest(
                        {
                            "common_outcome_class": event.common_outcome_class,
                            "typed_blocker_class": event.typed_blocker_class,
                            "capability_family": _family_key(event, route),
                        }
                    ),
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
                    "mechanism_effect_update_allowed": bool(
                        event.causal_credit_allowed
                    ),
                    "search_utility_update_allowed": event_record.get(
                        "search_utility_update_allowed"
                    ),
                    "mechanism_program": route.get("mechanism_program"),
                    "fidelity_classification": route.get(
                        "fidelity_classification"
                    ),
                    "directional_evidence_class": event_record.get(
                        "directional_evidence_class"
                    ),
                }
            )
            scientific_observations = _bounded_append(
                scientific_observations,
                scientific_observation,
            )
            global_memory["scientific_observations"] = tuple(scientific_observations)
    global_memory["executed_observations"] = executed_observations
    global_memory["search_observations"] = tuple(search_observations)
    if (
        event is not None
        and event.evidence_class != "UNCLASSIFIED"
        and route.get("execution_lane") != "AUXILIARY_VERIFICATION"
    ):
        global_memory["latest_mechanism_attribution"] = canonical_value(
            {
                "mechanism_axis": event.mechanism_axis,
                "mechanism_axis_footprint": event.mechanism_axis_footprint,
                "evidence_class": event.evidence_class,
                "failure_class": event.failure_class,
                "unresolved_confounding": event.unresolved_confounding,
                "core_mechanism_contrast": event.core_mechanism_contrast,
                "causal_credit_allowed": event.causal_credit_allowed,
                "next_discriminative_task": route.get(
                    "next_discriminative_test"
                ),
            }
        )
    if (
        event is not None
        and episode is None
        and route.get("execution_lane") != "AUXILIARY_VERIFICATION"
    ):
        global_memory["last_engineering_failure"] = canonical_value(
            {
                "round_index": context.round_index,
                "candidate_id": event.candidate_id,
                "failure_class": event.common_outcome_class,
                "typed_blocker_class": event.typed_blocker_class,
                "capability_family": _family_key(event, route),
                "failure_fingerprint": sha256_digest(
                    {
                        "common_outcome_class": event.common_outcome_class,
                        "typed_blocker_class": event.typed_blocker_class,
                        "capability_family": _family_key(event, route),
                    }
                ),
                "mechanism_effect_update_allowed": False,
            }
        )

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
            "execution_lane": execution_lane,
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
            (
                "verification_history"
                if auxiliary_verification
                else "execution_history"
            ),
            execution_record,
            limit=_MEMORY_HISTORY_LIMIT,
        )
        role_memory["latest_feedback"] = feedback
        credit = _mapping_copy(role_memory.get("credit"))
        if auxiliary_verification:
            credit["auxiliary_verification_count"] = int(
                credit.get("auxiliary_verification_count", 0)
            ) + 1
        else:
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
        role_memory[
            (
                "last_verification_round"
                if auxiliary_verification
                else "last_execution_round"
            )
        ] = context.round_index
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
    evaluator_projection = _enrich_search_utility_event(
        event=evaluator_projection,
        selected=selected,
        route=metadata,
        context=context,
        episode=episode,
        axis_universe=tuple(policy.mechanism_axis_targeting),
    )
    next_test = (
        episode.next_discriminative_test
        if episode is not None
        else _text(
            metadata.get("next_discriminative_test"),
            field="next_discriminative_test",
        )
    )
    # Interpretation records the next discriminative task in scientific
    # memory.  It is evidence for later research, not a per-round policy
    # command that may seize the discovery lane.
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
    belief: DevelopmentalMechanismBeliefV1 | None = None
    belief_policy_footprint = _axis_values(
        evaluator_projection.mechanism_axis_footprint,
        axis_universe=tuple(policy.mechanism_axis_targeting),
    )
    if episode is not None and episode.failure_class in {
        ResearchFailureClassV1.NONE,
        ResearchFailureClassV1.MECHANISM,
    } and evaluator_projection.causal_credit_allowed and len(
        evaluator_projection.mechanism_axis_footprint
    ) == 1 and len(belief_policy_footprint) == 1:
        belief = project_episode_to_mechanism_belief(
            comparison_identity=comparison_identity,
            closure=closure,
            episode=episode,
            mechanism_axis=belief_policy_footprint[0],
        )
    task = task_record.to_legacy_task()
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=task.prompt_projection(),
    )
    # Keep strategy and allocation changes behind the explicit meta-policy
    # promotion path, while projecting closed scientific memory into the
    # next round's existing acquisition inputs.
    successor_policy = policy
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
    successor_queue = _task_queue_from_memory(successor_context.scientific_memory)
    queued_task = successor_queue.select_next()
    resolved_task = (
        queued_task
        if queued_task is not None
        else successor_queue.get(task_record.task_id) or task_record
    )
    next_task = resolved_task.to_legacy_task()
    if episode is not None:
        aggregate = _meta_aggregate(
            outcomes,
            selected_role,
            selected,
            evaluator_projection,
            required_probability,
            policy,
            scientific_episode=True,
            context=successor_context,
            route=metadata,
            episode=episode,
            task_record=queued_task if queued_task is not None else task_record,
        )
        successor_policy = replace(
            policy,
            acquisition_parameters=_next_acquisition_parameters(policy, aggregate),
        )
        successor_context = replace(
            successor_context,
            policy=successor_policy.to_dict(),
        )
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=next_task.prompt_projection(),
    )
    after = _behavior(successor_context, successor_policy)
    taxonomy = (
        _FAILURE_TAXONOMY[episode.failure_class]
        if episode is not None
        else f"ENGINEERING_DIAGNOSTIC_{closure.failure_class.value}"
    )
    if episode is None:
        attribution = "NOT_APPLICABLE"
    elif evaluator_projection.evidence_class in {
        "CAUSAL_SUPPORT",
        "CAUSAL_REFUTATION",
        "DESCRIPTIVE_SUPPORT",
        "DESCRIPTIVE_REFUTATION",
    } and len(evaluator_projection.mechanism_axis_footprint) > 1:
        attribution = evaluator_projection.evidence_class
    else:
        # Preserve the existing V1 public interpretation for the ordinary
        # one-axis development path; the richer event fields carry the
        # descriptive/causal distinction for every persisted observation.
        attribution = _MECHANISM_ATTRIBUTION[episode.failure_class]
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
    route_metadata: Mapping[str, Any] | None = None,
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
    successor_policy = policy
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
        route_metadata=route_metadata,
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


def interpret_verification_episode(
    *,
    episode: TypedResearchEpisodeV1,
    comparison_identity: FrozenComparisonIdentityV1,
    closure: ScientificEpisodeClosureV1,
    context: ResearchContext,
    route_metadata: Mapping[str, Any],
    evaluator_projection: SearchUtilityEventV2,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    selected_outcome: ProducerOutcome,
    frozen_context: ResearchContext | None = None,
) -> EpisodeInterpretation:
    """Interpret one exact queued verification without discovery credit.

    Verification reuses a previously bound candidate or package-owned control,
    so it has no current four-Producer portfolio and must not update Producer
    allocation, discovery frontier credit, or the logical discovery index.
    It does persist the physical evidence, satisfy the exact queued task, and
    expose the observation to Evidence Guard for replication/control
    aggregation.
    """

    if not isinstance(evaluator_projection, SearchUtilityEventV2):
        raise ScientificInterpreterError(
            "evaluator_projection must be SearchUtilityEventV2"
        )
    if not isinstance(memory_writer, SearchMemoryWriterV1):
        raise ScientificInterpreterError("memory_writer must be SearchMemoryWriterV1")
    if not isinstance(selected_outcome, ProducerOutcome) or selected_outcome.spec is None:
        raise ScientificInterpreterError(
            "verification requires one real persisted Producer outcome"
        )
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
    metadata = _mapping(route_metadata, field="route_metadata")
    trace = _digest(
        metadata.get("route_trace_digest"),
        field="route_trace_digest",
    )
    role = _text(
        metadata.get("selected_producer_role"),
        field="selected_producer_role",
    )
    candidate = _text(
        metadata.get("selected_candidate_id"),
        field="selected_candidate_id",
    )
    semantic_digest = _digest(
        metadata.get("selected_candidate_semantic_digest"),
        field="selected_candidate_semantic_digest",
    )
    if (
        role != selected_outcome.producer_role
        or selected_outcome.context_ref != identity_context.context_ref
        or selected_outcome.context_digest != identity_context.digest
    ):
        raise ScientificInterpreterError(
            "verification outcome is not bound to its Producer/context"
        )
    if (
        candidate != evaluator_projection.candidate_id
        or semantic_digest != evaluator_projection.candidate_semantic_digest
    ):
        raise ScientificInterpreterError(
            "verification route and evaluator candidate identities differ"
        )
    axis = metadata.get("selected_mechanism_axis")
    if axis is not None and axis != evaluator_projection.mechanism_axis:
        raise ScientificInterpreterError(
            "verification route and evaluator mechanism axes differ"
        )
    next_test = episode.next_discriminative_test
    task_record = _task_record(
        episode=episode,
        identity=comparison_identity,
        closure=closure,
        event=evaluator_projection,
        metadata=metadata,
        selected=selected_outcome,
        context=context,
        next_test=next_test,
    )
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=task_record.to_legacy_task().prompt_projection(),
    )
    before = _behavior(context, policy)
    predecessor = memory_writer.head.digest if memory_writer.head else None
    memory_feedback = feedback.to_dict()
    memory_feedback["execution_lane"] = "AUXILIARY_VERIFICATION"
    snapshot = memory_writer.commit(
        round_index=context.round_index,
        expected_predecessor_digest=predecessor,
        beliefs=(),
        route_trace_digest=trace,
        feedback_projection=memory_feedback,
    )
    successor_context = _successor(
        context,
        policy,
        snapshot,
        memory_feedback,
        event=evaluator_projection,
        episode=episode,
        task_record=task_record,
        selected_outcome=selected_outcome,
        producer_outcomes=(),
        route_metadata={
            **dict(metadata),
            "execution_lane": "AUXILIARY_VERIFICATION",
        },
        advance_round=False,
        frontier_update_allowed=False,
    )
    queued_task = _task_queue_from_memory(
        successor_context.scientific_memory
    ).select_next()
    next_task = queued_task.to_legacy_task() if queued_task is not None else task_record.to_legacy_task()
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=next_task.prompt_projection(),
    )
    after = _behavior(successor_context, policy)
    if before.producer_inputs_digest == after.producer_inputs_digest:
        raise ScientificInterpreterError(
            "verification did not change task/evidence inputs"
        )
    return EpisodeInterpretation(
        episode=episode,
        closure=closure,
        failure_taxonomy=_FAILURE_TAXONOMY[episode.failure_class],
        mechanism_attribution="AUXILIARY_VERIFICATION_ONLY",
        negative_evidence=(),
        search_utility_event=evaluator_projection,
        next_discriminative_task=next_task,
        mechanism_belief=None,
        feedback_projection=feedback,
        search_memory_snapshot=snapshot,
        policy_successor=policy,
        successor_context=successor_context,
        behavior_before=before,
        behavior_after=after,
        route_trace_digest=trace,
    )


def interpret_verification_diagnostic(
    *,
    closure: ScientificEpisodeClosureV1,
    comparison_identity: FrozenComparisonIdentityV1,
    context: ResearchContext,
    route_metadata: Mapping[str, Any],
    evaluator_projection: SearchUtilityEventV2,
    policy: VersionedResearchPolicyV1,
    memory_writer: SearchMemoryWriterV1,
    selected_outcome: ProducerOutcome,
    frozen_context: ResearchContext | None = None,
) -> EpisodeInterpretation:
    """Persist a failed auxiliary attempt while leaving its task pending."""

    if not isinstance(evaluator_projection, SearchUtilityEventV2):
        raise ScientificInterpreterError(
            "evaluator_projection must be SearchUtilityEventV2"
        )
    if not isinstance(memory_writer, SearchMemoryWriterV1):
        raise ScientificInterpreterError("memory_writer must be SearchMemoryWriterV1")
    if not isinstance(selected_outcome, ProducerOutcome) or selected_outcome.spec is None:
        raise ScientificInterpreterError(
            "verification diagnostic requires one persisted Producer outcome"
        )
    identity_context = context if frozen_context is None else frozen_context
    _validate_same_round_working_context(identity_context, context)
    _validate_context(
        episode=None,
        identity=comparison_identity,
        context=identity_context,
        policy=policy,
        policy_context=context,
    )
    _validate_closure(
        episode=None,
        identity=comparison_identity,
        closure=closure,
    )
    metadata = _mapping(route_metadata, field="route_metadata")
    trace = _digest(
        metadata.get("route_trace_digest"),
        field="route_trace_digest",
    )
    queue = _task_queue_from_memory(context.scientific_memory)
    queued = queue.select_next()
    if queued is None:
        raise ScientificInterpreterError(
            "verification diagnostic has no pending exact task"
        )
    next_task = queued.to_legacy_task()
    feedback = PromptFeedbackProjectionV2(
        common_search_utility_slot=evaluator_projection,
        research_task_slot=next_task.prompt_projection(),
    )
    memory_feedback = feedback.to_dict()
    memory_feedback.update(
        {
            "execution_lane": "AUXILIARY_VERIFICATION",
            "engineering_diagnostic": {
                "closure_id": closure.closure_id,
                "failure_class": closure.failure_class.value,
                "failure_detail_ref": closure.failure_detail_ref,
                "failure_detail_digest": closure.failure_detail_digest,
            },
        }
    )
    predecessor = memory_writer.head.digest if memory_writer.head else None
    snapshot = memory_writer.commit(
        round_index=context.round_index,
        expected_predecessor_digest=predecessor,
        beliefs=(),
        route_trace_digest=trace,
        feedback_projection=memory_feedback,
    )
    successor_context = _successor(
        context,
        policy,
        snapshot,
        memory_feedback,
        event=evaluator_projection,
        episode=None,
        task_record=None,
        selected_outcome=selected_outcome,
        producer_outcomes=(),
        route_metadata={
            **dict(metadata),
            "execution_lane": "AUXILIARY_VERIFICATION",
        },
        advance_round=False,
        frontier_update_allowed=False,
    )
    return EpisodeInterpretation(
        episode=None,
        closure=closure,
        failure_taxonomy=f"ENGINEERING_DIAGNOSTIC_{closure.failure_class.value}",
        mechanism_attribution="NOT_APPLICABLE",
        negative_evidence=(),
        search_utility_event=evaluator_projection,
        next_discriminative_task=next_task,
        mechanism_belief=None,
        feedback_projection=feedback,
        search_memory_snapshot=snapshot,
        policy_successor=policy,
        successor_context=successor_context,
        behavior_before=_behavior(context, policy),
        behavior_after=_behavior(successor_context, policy),
        route_trace_digest=trace,
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
    "interpret_verification_diagnostic",
    "interpret_verification_episode",
]
