from __future__ import annotations

from dataclasses import replace
from typing import Any

from recclaw_core.helix.scientific_attribution import SearchUtilityEventV2
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.scientific_episode import (
    FrozenComparisonIdentityV1,
    close_scientific_episode,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    EpisodeEvidenceClassV1,
    OpenResearchSpecV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)
from recclaw_core.research_line.interfaces import (
    ProducerOutcome,
    ResearchContext,
)
from recclaw_core.research_line.interpreter import (
    interpret_scientific_diagnostic,
    interpret_typed_research_episode,
)


def _digest(label: str) -> str:
    return sha256_digest({"label": label})


def _policy_and_context() -> tuple[Any, ResearchContext]:
    policy = initial_research_policy()
    context = ResearchContext(
        campaign_id="campaign-1",
        round_index=1,
        knowledge_base={"mechanisms": ["graph", "ssl"]},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={
            "value": 0.12,
            "mechanism": "LIGHTGCN",
            "implementation_risk": {"new_family": 0.5},
        },
        scientific_memory={
            "by_role": {
                role: {"prior": [role]} for role in DISCOVERY_PRODUCERS
            }
        },
        unresolved_questions=({"question": "Does propagation improve sparse users?"},),
        policy=policy.to_dict(),
        budget={"proposal_calls": 4, "ordinary_experiments": 1},
        active_profile_ref="profile:66",
        active_profile_digest=_digest("profile"),
        protocol_ref="protocol:pilot",
        protocol_digest=_digest("protocol"),
    )
    return policy, context


def _episode(
    context: ResearchContext,
    failure_class: ResearchFailureClassV1,
) -> TypedResearchEpisodeV1:
    evidence_class = (
        EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT
        if failure_class is ResearchFailureClassV1.INCONCLUSIVE
        else EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT
    )
    interpretation = (
        "NOT_ADJUDICATED"
        if failure_class is ResearchFailureClassV1.INCONCLUSIVE
        else (
            "MECHANISM_NEGATIVE"
            if failure_class is ResearchFailureClassV1.MECHANISM
            else "SUPPORTING_COMPARISON"
        )
    )
    return TypedResearchEpisodeV1(
        campaign_id=context.campaign_id,
        context_ref=context.context_ref,
        context_digest=context.digest,
        hypothesis="Propagation changes the long-tail ranking signal.",
        executable_capability_ref="capability:fixture",
        executable_capability_digest=_digest("capability"),
        executable_profile_ref=context.active_profile_ref,
        executable_profile_digest=context.active_profile_digest,
        experiment_binding_ref="binding:round-1",
        experiment_binding_digest=_digest("binding"),
        comparator_ref="comparator:matched-incumbent",
        comparator_digest=_digest("comparator"),
        outcome_ref="outcome:round-1",
        outcome_digest=_digest("outcome"),
        cost_ref="cost:round-1",
        cost_digest=_digest("cost"),
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
        evidence_class=evidence_class,
        experiment_executed=True,
        mechanism_interpretation=interpretation,
        competing_explanation="The apparent effect may be optimization noise.",
        failure_class=failure_class,
        mechanism_negative_evidence=(
            failure_class is ResearchFailureClassV1.MECHANISM
        ),
        next_discriminative_test="Run a matched sparse-user control.",
        qualification_receipt_ref=None,
        qualification_receipt_digest=None,
        qualification_evidence_used_as_scientific=False,
    )


def _comparison_identity(
    value: TypedResearchEpisodeV1 | ResearchContext,
) -> FrozenComparisonIdentityV1:
    return FrozenComparisonIdentityV1(
        campaign_id=value.campaign_id,
        context_ref=value.context_ref,
        context_digest=getattr(value, "context_digest", value.digest),
        executable_capability_ref=getattr(value, "executable_capability_ref", "capability:fixture"),
        executable_capability_digest=_digest("capability"),
        executable_profile_ref=value.executable_profile_ref
        if isinstance(value, TypedResearchEpisodeV1)
        else value.active_profile_ref,
        executable_profile_digest=value.executable_profile_digest
        if isinstance(value, TypedResearchEpisodeV1)
        else value.active_profile_digest,
        experiment_binding_ref=getattr(value, "experiment_binding_ref", "binding:round-1"),
        experiment_binding_digest=_digest("binding"),
        comparator_ref=getattr(value, "comparator_ref", "comparator:matched-incumbent"),
        comparator_digest=_digest("comparator"),
        protocol_ref=value.protocol_ref,
        protocol_digest=value.protocol_digest,
    )


def _producer_outcomes(context: ResearchContext) -> tuple[ProducerOutcome, ...]:
    selected_role = DISCOVERY_PRODUCERS[0]
    spec = OpenResearchSpecV1(
        hypothesis="Propagation changes the long-tail ranking signal.",
        mechanism_change="Change propagation while preserving the evaluator.",
        competing_explanation="The effect may be optimization noise.",
        matched_control_requirement="Use the incumbent under the same protocol.",
        implementation_requirements=("fixture package",),
        expected_evidence=("full-sort NDCG",),
        falsifier="No improvement under the frozen evaluator.",
        compatibility_requirements=("RecBole",),
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
        context_ref=context.context_ref,
        context_digest=context.digest,
        current_profile_ref=context.active_profile_ref,
        current_profile_digest=context.active_profile_digest,
        producer_role=selected_role,
        high_change_justification="The current profile lacks this intervention.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )
    outcomes = []
    for role in DISCOVERY_PRODUCERS:
        if role == selected_role:
            outcomes.append(
                ProducerOutcome(
                    producer_role=role,
                    context_ref=context.context_ref,
                    context_digest=context.digest,
                    spec=spec,
                    resolution_facts={"fixture": "selected"},
                )
            )
        else:
            outcomes.append(
                ProducerOutcome(
                    producer_role=role,
                    context_ref=context.context_ref,
                    context_digest=context.digest,
                    spec=None,
                    resolution_facts={"fixture": "non-selected"},
                    failure_code="FIXTURE_NOT_SELECTED",
                    failure_detail="The focused test supplies one selected OpenSpec.",
                )
            )
    return tuple(outcomes)


def _event(failure_class: ResearchFailureClassV1) -> SearchUtilityEventV2:
    diagnostic = failure_class in {
        ResearchFailureClassV1.RESOURCE,
        ResearchFailureClassV1.OUTCOME_MISSING,
    }
    return SearchUtilityEventV2(
        candidate_semantic_digest=_digest("candidate-semantic"),
        candidate_id="cand-fixture",
        mechanism_axis="propagation",
        common_outcome_class=failure_class.value if diagnostic else "COMPARED_OUTCOME",
        runnable_observation="NOT_RUNNABLE" if diagnostic else "RUNNABLE",
        comparator_delta=(
            0.2
            if failure_class is ResearchFailureClassV1.NONE
            else -0.1
            if failure_class is ResearchFailureClassV1.MECHANISM
            else "NOT_AVAILABLE"
        ),
        metric_contract_digest=_digest("metric-contract"),
        resource_cost_projection={"wall_time_ms": 1000, "provider_cost": 0.01},
        typed_blocker_class=(
            "RESOURCE_EXHAUSTED"
            if failure_class is ResearchFailureClassV1.RESOURCE
            else "OUTCOME_MISSING"
            if failure_class is ResearchFailureClassV1.OUTCOME_MISSING
            else "NONE"
        ),
        observation_seed="seed-1",
    )


def _interpret(
    failure_class: ResearchFailureClassV1,
    *,
    event_override: SearchUtilityEventV2 | None = None,
) -> tuple[Any, Any, ResearchContext, Any, SearchMemoryWriterV1]:
    policy, context = _policy_and_context()
    episode = _episode(context, failure_class)
    comparison_identity = _comparison_identity(episode)
    closure = close_scientific_episode(
        comparison_identity=comparison_identity,
        failure_class=failure_class,
        episode=episode,
        observed_outcome_ref=episode.outcome_ref,
        observed_outcome_digest=episode.outcome_digest,
    )
    outcomes = _producer_outcomes(context)
    event = event_override or _event(failure_class)
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    route_metadata = {
        "route_trace_digest": _digest("route"),
        "selected_producer_role": DISCOVERY_PRODUCERS[0],
        "selected_candidate_id": event.candidate_id,
        "selected_candidate_semantic_digest": event.candidate_semantic_digest,
        "selected_mechanism_axis": event.mechanism_axis,
        "required_selected_runnable_probability": 0.8,
        "mechanism_program": program,
        "mechanism_program_digest": sha256_digest(program),
        "comparator_identity": episode.comparator_ref,
        "required_seed_or_control": "seed-2",
        "next_task_type": "VALIDATE_SAME_CANDIDATE",
        "task_utility_priority": 0.7,
        "missing_seed_count": 1,
    }
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    result = interpret_typed_research_episode(
        episode=episode,
        comparison_identity=comparison_identity,
        closure=closure,
        context=context,
        producer_outcomes=outcomes,
        route_metadata=route_metadata,
        evaluator_projection=event,
        policy=policy,
        memory_writer=writer,
    )
    return result, policy, context, outcomes, writer


def test_supporting_episode_enters_d1_memory_and_keeps_negative_empty() -> None:
    result, policy, context, _outcomes, writer = _interpret(
        ResearchFailureClassV1.NONE
    )

    assert result.failure_taxonomy == "SCIENTIFIC_SUPPORTING_COMPARISON"
    assert result.mechanism_attribution == "SUPPORTING_COMPARISON"
    assert result.mechanism_belief is not None
    assert result.mechanism_belief.evidence_for
    assert result.mechanism_belief.evidence_against == ()
    assert result.negative_evidence == ()
    assert result.search_memory_snapshot.beliefs == (result.mechanism_belief,)
    assert writer.head is result.search_memory_snapshot
    assert result.policy_successor.version == policy.version + 1
    assert result.policy_successor.predecessor_digest == policy.digest
    changed = result.behavior_before.changed_fields(result.behavior_after)
    assert result.behavior_before.producer_inputs_digest == context.producer_inputs_digest
    assert result.behavior_after.producer_inputs_digest == result.successor_context.producer_inputs_digest
    assert {"producer_allocation", "axis_priorities"} <= set(changed)
    assert result.successor_context.round_index == context.round_index + 1
    assert result.successor_context.frontier["value"] == 0.32
    assert result.successor_context.frontier["incumbent_ref"] == result.episode.outcome_ref
    assert result.successor_context.frontier["effect_trajectory"][-1][
        "frontier_updated"
    ] is True
    assert result.successor_context.producer_view(DISCOVERY_PRODUCERS[0])[
        "producer_token_fraction"
    ] != context.producer_view(DISCOVERY_PRODUCERS[0])["producer_token_fraction"]
    allocation = dict(result.policy_successor.producer_token_allocation)
    assert allocation[DISCOVERY_PRODUCERS[0]] > allocation[DISCOVERY_PRODUCERS[1]]


def test_mechanism_negative_episode_uses_d1_negative_evidence_only() -> None:
    result, _policy, context, _outcomes, _writer = _interpret(
        ResearchFailureClassV1.MECHANISM
    )

    assert result.failure_taxonomy == "SCIENTIFIC_MECHANISM_NEGATIVE"
    assert result.mechanism_attribution == "MECHANISM_NEGATIVE"
    assert result.mechanism_belief is not None
    assert result.mechanism_belief.evidence_for == ()
    assert result.mechanism_belief.evidence_against
    assert result.negative_evidence == result.mechanism_belief.evidence_against
    allocation = dict(result.policy_successor.producer_token_allocation)
    assert allocation[DISCOVERY_PRODUCERS[0]] < allocation[DISCOVERY_PRODUCERS[1]]
    assert result.successor_context.frontier["value"] == context.frontier["value"]
    assert result.successor_context.frontier["effect_trajectory"][-1][
        "frontier_updated"
    ] is False


def test_inconclusive_episode_writes_search_feedback_without_mechanism_negative() -> None:
    result, policy, _context, _outcomes, _writer = _interpret(
        ResearchFailureClassV1.INCONCLUSIVE
    )

    assert result.failure_taxonomy == "SCIENTIFIC_INCONCLUSIVE"
    assert result.mechanism_attribution == "NOT_ADJUDICATED"
    assert result.mechanism_belief is None
    assert result.negative_evidence == ()
    assert result.search_memory_snapshot.beliefs == ()
    assert result.next_discriminative_task.task_status.value == "PENDING"
    assert result.policy_successor.producer_token_allocation == policy.producer_token_allocation


def test_inconclusive_positive_effect_updates_development_frontier() -> None:
    event = replace(
        _event(ResearchFailureClassV1.INCONCLUSIVE),
        comparator_delta=0.03,
    )
    result, _policy, context, _outcomes, _writer = _interpret(
        ResearchFailureClassV1.INCONCLUSIVE,
        event_override=event,
    )

    assert result.mechanism_attribution == "NOT_ADJUDICATED"
    assert result.successor_context.frontier["value"] == 0.15
    assert result.successor_context.frontier["incumbent_candidate_id"] == event.candidate_id
    assert result.successor_context.frontier["effect_trajectory"][-1][
        "comparator_delta"
    ] == 0.03


def test_carryover_episode_does_not_credit_failed_current_producer() -> None:
    policy, context = _policy_and_context()
    episode = _episode(context, ResearchFailureClassV1.NONE)
    comparison_identity = _comparison_identity(episode)
    closure = close_scientific_episode(
        comparison_identity=comparison_identity,
        failure_class=ResearchFailureClassV1.NONE,
        episode=episode,
        observed_outcome_ref=episode.outcome_ref,
        observed_outcome_digest=episode.outcome_digest,
    )
    selected = _producer_outcomes(context)[0]
    failed_outcomes = tuple(
        ProducerOutcome(
            producer_role=role,
            context_ref=context.context_ref,
            context_digest=context.digest,
            spec=None,
            resolution_facts={"fixture": "current-provider-failure"},
            failure_code="PROVIDER_CALL_FAILED",
            failure_detail="The current Producer call failed before yielding an OpenSpec.",
        )
        for role in DISCOVERY_PRODUCERS
    )
    event = _event(ResearchFailureClassV1.NONE)
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    result = interpret_typed_research_episode(
        episode=episode,
        comparison_identity=comparison_identity,
        closure=closure,
        context=context,
        producer_outcomes=failed_outcomes,
        route_metadata={
            "route_trace_digest": _digest("carryover-route"),
            "selected_producer_role": selected.producer_role,
            "selected_candidate_id": event.candidate_id,
            "selected_candidate_semantic_digest": event.candidate_semantic_digest,
            "selected_mechanism_axis": event.mechanism_axis,
            "required_selected_runnable_probability": 0.8,
            "mechanism_program": program,
            "mechanism_program_digest": sha256_digest(program),
            "comparator_identity": episode.comparator_ref,
            "required_seed_or_control": "seed-2",
            "next_task_type": "VALIDATE_SAME_CANDIDATE",
            "task_utility_priority": 0.7,
            "missing_seed_count": 1,
        },
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        selected_outcome=selected,
    )

    assert result.episode is episode
    assert result.search_memory_snapshot.feedback_projection_digest
    assert result.policy_successor.producer_token_allocation == policy.producer_token_allocation
    assert result.behavior_after.axis_priorities != result.behavior_before.axis_priorities


def _diagnostic_interpret(
    failure_class: ResearchFailureClassV1,
    *,
    event_override: SearchUtilityEventV2 | None = None,
) -> tuple[Any, Any, ResearchContext]:
    policy, context = _policy_and_context()
    identity = _comparison_identity(context)
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=failure_class,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref="failure-detail:round-1",
        failure_detail_digest=_digest("failure-detail"),
    )
    outcomes = _producer_outcomes(context)
    event = event_override or _event(failure_class)
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    route = {
        "route_trace_digest": _digest("route"),
        "selected_producer_role": DISCOVERY_PRODUCERS[0],
        "selected_candidate_id": event.candidate_id,
        "selected_candidate_semantic_digest": event.candidate_semantic_digest,
        "required_selected_runnable_probability": 0.2,
        "mechanism_program": program,
        "comparator_identity": identity.comparator_ref,
        "required_seed_or_control": "repair-seed-1",
        "next_task_type": "REPAIR_IMPLEMENTATION",
        "next_discriminative_test": "Repair the diagnosed path and rerun the matched comparison.",
        "task_utility_priority": 0.8,
        "missing_seed_count": 1,
    }
    result = interpret_scientific_diagnostic(
        closure=closure,
        comparison_identity=identity,
        context=context,
        producer_outcomes=outcomes,
        route_metadata=route,
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )
    return result, closure, context


def test_resource_diagnostic_persists_blocker_and_changes_round_two_without_mechanism_evidence() -> None:
    event = replace(
        _event(ResearchFailureClassV1.RESOURCE),
        common_outcome_class="RESOURCE_CENSORED",
    )
    result, closure, context = _diagnostic_interpret(
        ResearchFailureClassV1.RESOURCE,
        event_override=event,
    )

    assert result.episode is None
    assert result.closure is closure
    assert closure.failure_detail_ref == "failure-detail:round-1"
    assert closure.failure_detail_digest == _digest("failure-detail")
    assert result.failure_taxonomy == "ENGINEERING_DIAGNOSTIC_RESOURCE"
    assert result.mechanism_attribution == "NOT_APPLICABLE"
    assert result.mechanism_belief is None
    assert result.negative_evidence == ()
    assert result.search_memory_snapshot.beliefs == ()
    assert (
        result.feedback_projection.common_search_utility_slot.typed_blocker_class
        == "RESOURCE_EXHAUSTED"
    )
    assert result.next_discriminative_task.task_type.value == "REPAIR_IMPLEMENTATION"
    assert result.behavior_after.round_index == context.round_index + 1
    assert (
        result.behavior_after.producer_allocation
        == result.behavior_before.producer_allocation
    )
    assert result.behavior_after.axis_priorities == result.behavior_before.axis_priorities
    assert (
        result.behavior_before.producer_inputs_digest
        != result.behavior_after.producer_inputs_digest
    )
    assert result.successor_context.active_profile_ref == context.active_profile_ref
    diagnostic = result.successor_context.scientific_memory["latest_feedback"][
        "engineering_diagnostic"
    ]
    assert diagnostic["failure_class"] == ResearchFailureClassV1.RESOURCE.value
    assert diagnostic["failure_detail_ref"] == closure.failure_detail_ref
    assert diagnostic["failure_detail_digest"] == closure.failure_detail_digest
    assert result.successor_context.scientific_memory["executed_observations"] == [
        {
            "round_index": context.round_index,
            "candidate_semantic_digest": event.candidate_semantic_digest,
            "observation_seed": event.observation_seed,
            "common_outcome_class": "RESOURCE_CENSORED",
            "mechanism_axis": event.mechanism_axis,
            "comparator_delta": "NOT_AVAILABLE",
        }
    ]
    role_memory = result.successor_context.producer_view(DISCOVERY_PRODUCERS[0])[
        "scientific_memory"
    ]
    assert role_memory["latest_feedback"]["engineering_diagnostic"] == diagnostic
