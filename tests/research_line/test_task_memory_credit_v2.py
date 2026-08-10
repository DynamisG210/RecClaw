from __future__ import annotations

from dataclasses import replace

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
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
    ResearchTaskStatusV2,
)
from recclaw_core.research_line.interpreter import (
    _task_operation,
    _task_queue_transition,
    interpret_scientific_diagnostic,
    interpret_typed_research_episode,
)
from recclaw_core.research_line.runtime import _search_ranking_inputs


def _digest(label: str) -> str:
    return sha256_digest({"label": label})


def _context() -> tuple[object, ResearchContext]:
    policy = initial_research_policy()
    context = ResearchContext(
        campaign_id="task-memory-campaign",
        round_index=1,
        knowledge_base={"mechanisms": ["graph", "ssl"]},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={"value": 0.12, "mechanism": "LIGHTGCN"},
        scientific_memory={
            "global_memory": {
                "latest_feedback": {"shared": "one-copy"},
                "search_observations": (),
            },
            "by_role": {
                role: {"lineage": {"role": role}}
                for role in DISCOVERY_PRODUCERS
            },
        },
        unresolved_questions=({"question": "Which mechanism explains the gap?"},),
        policy=policy.to_dict(),
        budget={"proposal_calls": 4, "ordinary_experiments": 1},
        active_profile_ref="profile:task-memory",
        active_profile_digest=_digest("profile"),
        protocol_ref="protocol:task-memory",
        protocol_digest=_digest("protocol"),
    )
    return policy, context


def _task(
    operation: ResearchTaskOperationV2,
    *,
    label: str,
    required: str,
    priority: float = 0.5,
    status: ResearchTaskStatusV2 = ResearchTaskStatusV2.PENDING,
) -> ResearchTaskRecordV2:
    program = {"operator": "fixture", "label": label}
    return ResearchTaskRecordV2(
        task_id=_digest(f"task:{label}"),
        operation=operation,
        candidate_id="candidate:task-memory",
        candidate_semantic_digest=_digest("candidate:task-memory"),
        mechanism_program_digest=sha256_digest(program),
        parent_candidate_id="candidate:parent",
        comparator_identity="comparator:task-memory",
        protocol_digest=_digest("protocol:task-memory"),
        required_seed_or_control=required,
        priority=priority,
        created_round=1,
        mechanism_program=program,
        status=status,
    )


def test_durable_queue_keeps_all_operations_and_lifecycle_state() -> None:
    queue = ResearchTaskQueueV2()
    tasks = tuple(
        _task(operation, label=operation.value, required=f"required:{operation.value}")
        for operation in ResearchTaskOperationV2
    )
    for task in tasks:
        queue = queue.enqueue(task)

    assert {task.operation for task in queue.tasks} == set(ResearchTaskOperationV2)
    repair = queue.get(_digest(f"task:{ResearchTaskOperationV2.REPAIR.value}"))
    assert repair is not None
    queue = queue.activate(repair.task_id)
    queue = queue.satisfy(
        repair.task_id,
        evidence=("repair-receipt",),
        reason="REPAIR_EXECUTED",
    )
    assert queue.get(repair.task_id).status is ResearchTaskStatusV2.SATISFIED

    control = queue.select_next()
    assert control is not None
    assert control.operation is ResearchTaskOperationV2.MATCHED_CONTROL
    queue = queue.close(control.task_id, reason="CONTROL_SUPERSEDED")
    assert queue.get(control.task_id).status is ResearchTaskStatusV2.CLOSED

    # Re-enqueuing the same identity merges priority/evidence instead of
    # deleting the older task or replacing the queue with a latest-only slot.
    same_new_seed = _task(
        ResearchTaskOperationV2.NEW_SEED,
        label=ResearchTaskOperationV2.NEW_SEED.value,
        required="required:NEW_SEED",
        priority=0.99,
    )
    queue = queue.enqueue(
        replace(same_new_seed, evidence_present=("second-observation",))
    )
    new_seed = queue.get(same_new_seed.task_id)
    assert new_seed.priority == 0.99
    assert "second-observation" in new_seed.evidence_present
    assert len(queue.tasks) == len(ResearchTaskOperationV2)


def test_producer_view_has_one_shared_global_and_role_provenance_memory() -> None:
    _policy, context = _context()
    views = {
        role: context.producer_view(role)
        for role in DISCOVERY_PRODUCERS
    }

    assert len({sha256_digest(view["global_memory"]) for view in views.values()}) == 1
    assert all(
        view["global_memory"]["latest_feedback"] == {"shared": "one-copy"}
        for view in views.values()
    )
    assert len({sha256_digest(view["memory"]) for view in views.values()}) == 4
    assert all(view["memory"]["lineage"]["role"] == role for role, view in views.items())
    assert all("latest_feedback" not in view["memory"] for view in views.values())


def test_same_seed_default_moves_on_but_explicit_reproduce_is_available() -> None:
    _policy, context = _context()
    event = _event(delta=0.0)
    episode = _episode(context)
    assert (
        _task_operation(
            route={},
            event=event,
            episode=episode,
            required_seed_or_control=event.observation_seed,
        )
        is ResearchTaskOperationV2.MOVE_ON
    )
    assert (
        _task_operation(
            route={"allow_same_seed_reproduce": True},
            event=event,
            episode=episode,
            required_seed_or_control=event.observation_seed,
        )
        is ResearchTaskOperationV2.REPRODUCE
    )


def test_active_queue_head_is_not_satisfied_without_matching_evidence() -> None:
    _policy, context = _context()
    event = _event(delta=0.0)
    pending = replace(
        _task(
            ResearchTaskOperationV2.NEW_SEED,
            label="pending-other-candidate",
            required="seed:required",
        ),
        candidate_id="candidate:other",
        candidate_semantic_digest=_digest("candidate:other"),
    )
    current = replace(
        _task(
            ResearchTaskOperationV2.MOVE_ON,
            label="current-move-on",
            required=event.observation_seed,
        ),
        candidate_id=event.candidate_id,
        candidate_semantic_digest=event.candidate_semantic_digest,
    )

    queue, transition = _task_queue_transition(
        queue=ResearchTaskQueueV2().enqueue(pending),
        task_record=current,
        event=event,
        episode=_episode(context),
        context=context,
        route={"active_task_id": pending.task_id},
        frontier_updated=False,
    )

    assert queue.get(pending.task_id).status is ResearchTaskStatusV2.PENDING
    assert transition["satisfied_task_ids"] == ()


def _spec(context: ResearchContext, role: str) -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
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
        producer_role=role,
        high_change_justification="The current profile lacks this intervention.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def _outcomes(
    context: ResearchContext,
    *,
    selected_spec: OpenResearchSpecV1,
    selected_provenance: dict[str, object] | None = None,
    all_failures: bool = False,
) -> tuple[ProducerOutcome, ...]:
    selected_role = selected_spec.producer_role
    result = []
    for role in DISCOVERY_PRODUCERS:
        if role == selected_role and not all_failures:
            result.append(
                ProducerOutcome(
                    producer_role=role,
                    context_ref=context.context_ref,
                    context_digest=context.digest,
                    spec=selected_spec,
                    resolution_facts={"fixture": "selected"},
                    provenance=selected_provenance or {},
                )
            )
        else:
            result.append(
                ProducerOutcome(
                    producer_role=role,
                    context_ref=context.context_ref,
                    context_digest=context.digest,
                    spec=None,
                    resolution_facts={"fixture": "failure"},
                    failure_code="PRODUCER_CALL_FAILED",
                    failure_detail="fixture failure before a spec was produced",
                    provenance={
                        "producer_role": role,
                        "status": "CALL_OR_PROJECTION_FAILURE",
                    },
                )
            )
    return tuple(result)


def _episode(context: ResearchContext) -> TypedResearchEpisodeV1:
    return TypedResearchEpisodeV1(
        campaign_id=context.campaign_id,
        context_ref=context.context_ref,
        context_digest=context.digest,
        hypothesis="Propagation changes the long-tail ranking signal.",
        executable_capability_ref="capability:fixture",
        executable_capability_digest=_digest("capability"),
        executable_profile_ref=context.active_profile_ref,
        executable_profile_digest=context.active_profile_digest,
        experiment_binding_ref="binding:task-memory",
        experiment_binding_digest=_digest("binding"),
        comparator_ref="comparator:matched-incumbent",
        comparator_digest=_digest("comparator"),
        outcome_ref="outcome:task-memory",
        outcome_digest=_digest("outcome"),
        cost_ref="cost:task-memory",
        cost_digest=_digest("cost"),
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
        evidence_class=EpisodeEvidenceClassV1.DEVELOPMENT_EXPERIMENT,
        experiment_executed=True,
        mechanism_interpretation="SUPPORTING_COMPARISON",
        competing_explanation="The apparent effect may be optimization noise.",
        failure_class=ResearchFailureClassV1.NONE,
        mechanism_negative_evidence=False,
        next_discriminative_test="Run a matched control and a fresh seed.",
        qualification_receipt_ref=None,
        qualification_receipt_digest=None,
        qualification_evidence_used_as_scientific=False,
    )


def _event(*, delta: float | str, outcome: str = "COMPARED_OUTCOME") -> SearchUtilityEventV2:
    return SearchUtilityEventV2(
        candidate_semantic_digest=_digest("candidate-semantic"),
        candidate_id="candidate:delayed",
        mechanism_axis="propagation",
        common_outcome_class=outcome,
        runnable_observation="RUNNABLE",
        comparator_delta=delta,
        metric_contract_digest=_digest("metric-contract"),
        resource_cost_projection={"wall_time_ms": 1000, "provider_cost": 0.01},
        typed_blocker_class="NONE" if delta != "NOT_AVAILABLE" else "RESOURCE_EXHAUSTED",
        observation_seed="seed:observed",
    )


def _identity(context: ResearchContext, episode: TypedResearchEpisodeV1 | None) -> FrozenComparisonIdentityV1:
    return FrozenComparisonIdentityV1(
        campaign_id=context.campaign_id,
        context_ref=context.context_ref,
        context_digest=context.digest,
        executable_capability_ref=(
            episode.executable_capability_ref if episode is not None else "capability:fixture"
        ),
        executable_capability_digest=_digest("capability"),
        executable_profile_ref=context.active_profile_ref,
        executable_profile_digest=context.active_profile_digest,
        experiment_binding_ref=(
            episode.experiment_binding_ref if episode is not None else "binding:task-memory"
        ),
        experiment_binding_digest=_digest("binding"),
        comparator_ref=(
            episode.comparator_ref if episode is not None else "comparator:task-memory"
        ),
        comparator_digest=_digest("comparator"),
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
    )


def test_delayed_execution_credits_original_producer_and_populates_frontier_banks() -> None:
    policy, context = _context()
    selected_role = DISCOVERY_PRODUCERS[0]
    selected_spec = _spec(context, selected_role)
    selected = ProducerOutcome(
        producer_role=selected_role,
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=selected_spec,
        resolution_facts={"requested_current_semantics_digest": _digest("old-round")},
        provenance={
            "producer_role": selected_role,
            "spec_digest": selected_spec.digest,
            "status": "CARRYOVER",
        },
    )
    current_outcomes = _outcomes(
        context,
        selected_spec=selected_spec,
        all_failures=True,
    )
    episode = _episode(context)
    identity = _identity(context, episode)
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=ResearchFailureClassV1.NONE,
        episode=episode,
        observed_outcome_ref=episode.outcome_ref,
        observed_outcome_digest=episode.outcome_digest,
    )
    event = replace(
        _event(delta=0.2),
        resource_cost_projection={
            "resource_telemetry_sha256": _digest("telemetry:metric"),
            "gpu_reservation_status": "RESERVED",
            "reserved_gpu_worker_seconds": 12.5,
            "reserved_gpu_worker_seconds_semantics": "FIXED_RESERVATION",
            "training_device_evidence": {
                "device": "cuda:0",
                "validated": True,
            },
        },
    )
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    control_program = {"operator": "matched-control"}
    mechanism_off_program = {"operator": "mechanism-off"}
    route = {
        "route_trace_digest": _digest("delayed-route"),
        "selected_producer_role": selected_role,
        "selected_candidate_id": event.candidate_id,
        "selected_candidate_semantic_digest": event.candidate_semantic_digest,
        "selected_mechanism_axis": event.mechanism_axis,
        "required_selected_runnable_probability": 0.8,
        "mechanism_program": program,
        "mechanism_program_digest": sha256_digest(program),
        "comparator_identity": episode.comparator_ref,
        "required_seed_or_control": "seed:next",
        "next_task_type": "VALIDATE_SAME_CANDIDATE",
        "capability_family": "propagation-family",
        "parent_candidate_id": "candidate:parent",
        "matched_control_candidate_id": "candidate:matched-control",
        "matched_control_semantic_digest": _digest("matched-control-semantic"),
        "matched_control_program": control_program,
        "matched_control_program_digest": sha256_digest(control_program),
        "matched_control_seed": "seed:control",
        "mechanism_off_candidate_id": "candidate:mechanism-off",
        "mechanism_off_semantic_digest": _digest("mechanism-off-semantic"),
        "mechanism_off_program": mechanism_off_program,
        "mechanism_off_program_digest": sha256_digest(mechanism_off_program),
        "mechanism_off_seed": "seed:ablation",
        "selected_compute_pattern": "pattern:delayed",
        "selected_resource_admission_state": "RESOURCE_ADMITTED",
        "selected_resource_evidence_digest": _digest("resource:delayed"),
        "selected_portfolio_profile_digest": _digest("profile:delayed"),
    }

    result = interpret_typed_research_episode(
        episode=episode,
        comparison_identity=identity,
        closure=closure,
        context=context,
        producer_outcomes=current_outcomes,
        route_metadata=route,
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        selected_outcome=selected,
    )

    memory = result.successor_context.scientific_memory
    global_memory = memory["global_memory"]
    queue = ResearchTaskQueueV2.from_dict(global_memory["task_queue"])
    assert {
        ResearchTaskOperationV2.NEW_SEED,
        ResearchTaskOperationV2.MATCHED_CONTROL,
        ResearchTaskOperationV2.MECHANISM_OFF,
    } <= {task.operation for task in queue.tasks}
    assert result.next_discriminative_task.task_type.value == "RUN_MATCHED_CONTROL"
    assert not global_memory["latest_feedback"]["task_queue_transition"][
        "deferred_requirements"
    ]
    head = queue.select_next()
    assert head is not None
    assert (
        global_memory["latest_feedback"]["research_task_slot"]
        == head.prompt_projection()
    )
    assert set(global_memory["latest_feedback"]["research_task_slot"]) == {
        "task_type",
        "candidate_semantic_digest",
        "required_seed_or_control",
        "task_status",
    }
    route_without_control_bindings = {
        key: value
        for key, value in route.items()
        if not key.startswith("matched_control_")
        and not key.startswith("mechanism_off_")
    }
    unbound = interpret_typed_research_episode(
        episode=episode,
        comparison_identity=identity,
        closure=closure,
        context=context,
        producer_outcomes=current_outcomes,
        route_metadata=route_without_control_bindings,
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        selected_outcome=selected,
    )
    unbound_global = unbound.successor_context.scientific_memory["global_memory"]
    unbound_queue = ResearchTaskQueueV2.from_dict(unbound_global["task_queue"])
    assert {
        item.operation for item in unbound_queue.tasks
        if item.status is ResearchTaskStatusV2.PENDING
    } == {ResearchTaskOperationV2.NEW_SEED}
    assert tuple(
        unbound_global["latest_feedback"]["task_queue_transition"][
            "deferred_requirements"
        ]
    ) == ("MATCHED_CONTROL", "MECHANISM_OFF")
    observations, pending_task, effect_by_axis = _search_ranking_inputs(
        result.successor_context
    )
    assert (event.candidate_semantic_digest, event.observation_seed) in observations
    assert pending_task is not None
    assert {
        key: pending_task[key]
        for key in head.prompt_projection()
    } == head.prompt_projection()
    assert pending_task["task_id"] == head.task_id
    assert pending_task["candidate_id"] == head.candidate_id
    assert pending_task["operation"] == head.operation.value
    assert pending_task["priority"] == head.priority
    assert effect_by_axis["propagation"] == 0.2
    assert all(
        ResearchTaskQueueV2.from_dict(
            result.successor_context.producer_view(role)["global_memory"]["task_queue"]
        ).digest
        == queue.digest
        for role in DISCOVERY_PRODUCERS
    )
    assert {
        "global",
        "family",
        "parent",
        "control",
        "confirmation",
    } <= set(result.successor_context.frontier)

    role_memory = memory["by_role"][selected_role]
    assert role_memory["execution_history"][-1]["origin"] == "CARRYOVER"
    assert role_memory["execution_history"][-1]["producer_role"] == selected_role
    execution_record = role_memory["execution_history"][-1]
    assert execution_record["compute_pattern"] == "pattern:delayed"
    assert execution_record["resource_admission_state"] == "RESOURCE_ADMITTED"
    assert execution_record["resource_evidence_digest"] == _digest(
        "resource:delayed"
    )
    assert execution_record["resource_cost_projection"] == dict(
        event.resource_cost_projection
    )
    assert execution_record["resource_cost_projection_digest"] == sha256_digest(
        event.resource_cost_projection
    )
    assert "candidate_run" not in execution_record
    assert "resource_telemetry" not in execution_record
    assert role_memory["credit"]["frontier_gain_count"] == 1
    assert global_memory["producer_lineage_index"][selected.digest] == selected_role
    assert all(
        "execution_history" not in memory["by_role"][role]
        for role in DISCOVERY_PRODUCERS[1:]
    )
    latest_search_observation = global_memory["search_observations"][-1]
    assert latest_search_observation["resource_cost_projection"] == dict(
        event.resource_cost_projection
    )
    assert latest_search_observation["resource_cost_projection_digest"] == sha256_digest(
        event.resource_cost_projection
    )
    assert "candidate_run" not in latest_search_observation
    assert "resource_telemetry" not in latest_search_observation
    assert "resource_cost_projection" not in global_memory["executed_observations"][-1]
    assert "candidate_run" not in global_memory["executed_observations"][-1]


def test_engineering_failure_updates_resource_search_only_and_creates_repair_task() -> None:
    policy, context = _context()
    selected_role = DISCOVERY_PRODUCERS[0]
    selected_spec = _spec(context, selected_role)
    outcomes = _outcomes(context, selected_spec=selected_spec)
    identity = _identity(context, None)
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=ResearchFailureClassV1.RESOURCE,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref="failure-detail:task-memory",
        failure_detail_digest=_digest("failure-detail"),
    )
    event = replace(
        _event(delta="NOT_AVAILABLE", outcome="RESOURCE_CENSORED"),
        resource_cost_projection={
            "resource_telemetry_sha256": _digest("telemetry:engineering"),
            "gpu_reservation_status": "CENSORED",
            "reserved_gpu_worker_seconds": 7.0,
            "training_device_evidence": {
                "device": "cuda:0",
                "validated": False,
            },
        },
    )
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    result = interpret_scientific_diagnostic(
        closure=closure,
        comparison_identity=identity,
        context=context,
        producer_outcomes=outcomes,
        route_metadata={
            "route_trace_digest": _digest("engineering-route"),
            "selected_producer_role": selected_role,
            "selected_candidate_id": event.candidate_id,
            "selected_candidate_semantic_digest": event.candidate_semantic_digest,
            "selected_mechanism_axis": event.mechanism_axis,
            "required_selected_runnable_probability": 0.2,
            "mechanism_program": program,
            "mechanism_program_digest": sha256_digest(program),
            "comparator_identity": identity.comparator_ref,
            "required_seed_or_control": "repair:next",
            "next_task_type": "REPAIR_IMPLEMENTATION",
            "next_discriminative_test": "Repair and rerun the blocked path.",
            "selected_compute_pattern": "pattern:engineering",
            "selected_resource_admission_state": "RESOURCE_ADMITTED",
            "selected_resource_evidence_digest": _digest("resource:engineering"),
        },
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )

    successor = result.successor_context
    assert successor.frontier == context.frontier
    global_memory = successor.scientific_memory["global_memory"]
    assert global_memory["resource_memory"]["last_observation"][
        "resource_or_search_only"
    ] is True
    resource_observation = global_memory["resource_memory"]["last_observation"]
    assert resource_observation["resource_cost_projection"] == dict(
        event.resource_cost_projection
    )
    assert resource_observation["resource_cost_projection_digest"] == sha256_digest(
        event.resource_cost_projection
    )
    search_observation = global_memory["search_observations"][-1]
    assert search_observation["resource_cost_projection"] == dict(
        event.resource_cost_projection
    )
    assert search_observation["resource_cost_projection_digest"] == sha256_digest(
        event.resource_cost_projection
    )
    assert "scientific_observations" not in global_memory
    assert result.negative_evidence == ()
    assert "mechanism_negative_count" not in successor.scientific_memory["by_role"][
        selected_role
    ].get("credit", {})
    engineering_record = successor.scientific_memory["by_role"][selected_role][
        "execution_history"
    ][-1]
    assert engineering_record["compute_pattern"] == "pattern:engineering"
    assert engineering_record["resource_cost_projection"] == dict(
        event.resource_cost_projection
    )
    assert "candidate_run" not in engineering_record
    queue = ResearchTaskQueueV2.from_dict(global_memory["task_queue"])
    repair = [
        task
        for task in queue.tasks
        if task.operation is ResearchTaskOperationV2.REPAIR
    ]
    assert len(repair) == 1
    assert repair[0].status is ResearchTaskStatusV2.PENDING


def test_cost_observation_and_execution_histories_are_capped_at_64() -> None:
    policy, context = _context()
    old_rows = tuple({"row": index} for index in range(70))
    memory = dict(context.scientific_memory)
    global_memory = dict(memory["global_memory"])
    global_memory["executed_observations"] = old_rows
    global_memory["search_observations"] = old_rows
    memory["global_memory"] = global_memory
    memory["by_role"] = {
        role: {
            **dict(memory["by_role"][role]),
            "execution_history": old_rows,
        }
        for role in DISCOVERY_PRODUCERS
    }
    context = replace(context, scientific_memory=memory)
    selected_role = DISCOVERY_PRODUCERS[0]
    selected_spec = _spec(context, selected_role)
    outcomes = _outcomes(context, selected_spec=selected_spec)
    identity = _identity(context, None)
    closure = close_scientific_episode(
        comparison_identity=identity,
        failure_class=ResearchFailureClassV1.RESOURCE,
        episode=None,
        observed_outcome_ref=None,
        observed_outcome_digest=None,
        failure_detail_ref="failure-detail:history-cap",
        failure_detail_digest=_digest("history-cap-failure"),
    )
    event = replace(
        _event(delta="NOT_AVAILABLE", outcome="RESOURCE_CENSORED"),
        resource_cost_projection={
            "gpu_reservation_status": "CENSORED",
            "reserved_gpu_worker_seconds": 3.0,
        },
    )
    program = {"operator": "fixture", "axis": event.mechanism_axis}
    result = interpret_scientific_diagnostic(
        closure=closure,
        comparison_identity=identity,
        context=context,
        producer_outcomes=outcomes,
        route_metadata={
            "route_trace_digest": _digest("history-cap-route"),
            "selected_producer_role": selected_role,
            "selected_candidate_id": event.candidate_id,
            "selected_candidate_semantic_digest": event.candidate_semantic_digest,
            "selected_mechanism_axis": event.mechanism_axis,
            "required_selected_runnable_probability": 0.2,
            "mechanism_program": program,
            "mechanism_program_digest": sha256_digest(program),
            "comparator_identity": identity.comparator_ref,
            "required_seed_or_control": "repair:history-cap",
            "next_task_type": "REPAIR_IMPLEMENTATION",
            "next_discriminative_test": "Repair the bounded history path.",
            "selected_compute_pattern": "pattern:history-cap",
            "selected_resource_admission_state": "RESOURCE_ADMITTED",
            "selected_resource_evidence_digest": _digest("resource:history-cap"),
        },
        evaluator_projection=event,
        policy=policy,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )

    successor_memory = result.successor_context.scientific_memory
    successor_global = successor_memory["global_memory"]
    assert len(successor_global["executed_observations"]) == 64
    assert len(successor_global["search_observations"]) == 64
    assert len(
        successor_global["resource_memory"]["observations"]
    ) == 1
    assert len(
        successor_memory["by_role"][selected_role]["execution_history"]
    ) == 64
