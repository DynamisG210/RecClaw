from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_EVALUATOR,
    ExperimentBindingV1,
)
from recclaw_core.experiments.helix_abc_v1.innovation_spine import (
    SharedImplementerPolicy,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    meta_v17_static_producer_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchProfileEntryOriginV1,
    adapt_current_search_profile,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CapabilityKindV1,
    ResearchFailureClassV1,
)
from recclaw_core.research_line.interfaces import ResearchContext
from recclaw_core.research_line.replay import OfflineProducerReplayV1
from recclaw_core.research_line.runtime import (
    InnovationRuntimeInputs,
    MetaResearchInputs,
    _search_ranking_inputs,
    activate_promoted_meta_strategy,
    activate_staged_innovation,
    bindings_for_context,
    resolver_environment_for_profile,
    run_research_round,
)


_HELIX_TEST_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "helix_abc_v1"
sys.path.insert(0, str(_HELIX_TEST_ROOT))

from test_e0_search_adapter import _outside_66_program, _proposal  # noqa: E402
from test_vnext_local_orchestration import (  # noqa: E402
    _fixture as _qualification_fixture,
    _open_draft,
    _policy as _fixture_policy,
    _positive_response,
    _unit_check,
)


def _digest(label: str) -> str:
    return sha256_digest({"research_line_runtime": label})


def _router() -> StrongStaticRouterV1:
    return StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=4,
    )


def _context(
    profile: Any,
    *,
    round_index: int = 1,
    policy: Any | None = None,
) -> ResearchContext:
    policy = policy or initial_research_policy()
    return ResearchContext(
        campaign_id=profile.campaign_id,
        round_index=round_index,
        knowledge_base={"search_space": "BL-ICF", "executable_entries": len(profile.entries)},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={"incumbent_ndcg@10": 0.40},
        scientific_memory={
            "by_role": {role: {"prior": role} for role in DISCOVERY_PRODUCERS}
        },
        unresolved_questions=({"question": "which mechanism moves the frontier?"},),
        policy=policy.to_dict(),
        budget={"producer_calls": 4, "experiment_opportunities": 1},
        active_profile_ref=profile.profile_ref,
        active_profile_digest=profile.profile_digest,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
    )


def _bindings(context: ResearchContext, profile: Any) -> dict[str, Any]:
    return bindings_for_context(
        context,
        active_profile=profile,
        implementation_requirements=(
            "RecBole general recommender interface",
            "candidate-local package",
        ),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )


def _environment(profile: Any) -> dict[str, Any]:
    return resolver_environment_for_profile(
        profile,
        available_dependencies=("recbole-runtime",),
        budget_limits={"implementation_tokens": 5000, "implementation_units": 2},
        protocol_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )


def _fixed_proposals(profile: Any, *, select_first: bool = True) -> dict[str, Any]:
    mechanisms = executable_mechanisms()[:4]
    return {
        role: _proposal(
            candidate_id=f"cand-runtime-{role.replace('_', '-')}",
            mechanism_id=mechanism.mechanism_id,
            mechanism_axis=mechanism.mechanism_axis,
            mechanism_program=mechanism.mechanism_program,
            protocol_digest=profile.protocol_digest,
            selected=select_first and index == 0,
            role=role,
        )
        for index, (role, mechanism) in enumerate(
            zip(DISCOVERY_PRODUCERS, mechanisms, strict=True)
        )
    }


def _runner(
    calls: list[dict[str, Any]],
    *,
    status: str = "SUCCESS",
    seed: int = 54304,
    binding_mutation: dict[str, Any] | None = None,
):
    def run(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        calls.append({"recipe": recipe, "binding": binding})
        experiment_binding = ExperimentBindingV1.from_execution_recipe(
            recipe,
            candidate_root=None,
            dataset_manifest_digest=_digest("dataset-manifest"),
            seed=seed,
            epochs=1,
            timeout_seconds=60,
            execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
            resource_telemetry=False,
            watchdog_seconds=None,
            prefix_contract_digest=None,
            run_id=f"run-{binding.proposal.candidate_id}",
            round_id="round-fixture",
            claim_id="claim-fixture",
            permit_digest=_digest("permit"),
            runtime_binding_digest=_digest("runtime-binding"),
            runtime_release_digest=_digest("runtime-release"),
            runner_abi="fixture-runner-v1",
            filesystem_capability_digest=_digest("filesystem-capability"),
        )
        if binding_mutation:
            experiment_binding = replace(experiment_binding, **binding_mutation)
        return {
            "exit_status": status,
            "metrics": {"ndcg@10": 0.45} if status == "SUCCESS" else {},
            "experiment_binding": experiment_binding.canonical_dict(),
            "experiment_binding_ref": experiment_binding.ref,
            "experiment_binding_digest": experiment_binding.digest,
            "binding_digest": experiment_binding.digest,
            "wall_time_ms": 17,
            "seed": seed,
            "execution_recipe_digest": sha256_digest(recipe),
        }

    return run


def _incumbent() -> dict[str, Any]:
    return {
        "comparator_ref": "incumbent:round-start",
        "comparator_digest": _digest("incumbent"),
        "frozen_ndcg@10": 0.40,
    }


def _innovation_inputs(
    tmp_path: Path,
    current: Any,
    *,
    resource_admission_required: bool = False,
) -> InnovationRuntimeInputs:
    base_policy = _fixture_policy()
    implementer_policy = SharedImplementerPolicy(
        allowed_files=base_policy.allowed_files,
        dependency_identity_ref=base_policy.dependency_identity_ref,
        dependency_identity_digest=base_policy.dependency_identity_digest,
        runtime_identity_ref=base_policy.runtime_identity_ref,
        runtime_identity_digest=base_policy.runtime_identity_digest,
        prompt_digest=base_policy.prompt_digest,
        tool_policy_digest=base_policy.tool_policy_digest,
        implementation_token_ceiling=base_policy.implementation_token_ceiling,
        execution_contract={
            "capability_family": "INTERACTION_GATE",
            "model": "GWaveOneInteractionGate",
            "base_model_config": "BPR",
            "config": {"embedding_size": 8},
        },
    )
    def resource_probe(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["execution_recipe"]["entrypoint"] == kwargs["entrypoint"]
        assert kwargs["source_path"].is_file()
        profile_identity = {
            "candidate_ref": kwargs["candidate_ref"],
            "candidate_package_digest": kwargs["candidate_package_digest"],
            "candidate_source_sha256": kwargs["source_sha256"],
        }
        prediction = {
            "completion_probability": 0.9,
            "estimated_total_wall_time_seconds": 30.0,
            "model": "FIXED_BATCH_THROUGHPUT_LINEAR_EXTRAPOLATION_V3",
            "peak_memory_prediction_mib": 128.0,
            "prediction_interval_seconds": [20.0, 40.0],
        }
        return {
            **profile_identity,
            "completion_probability": 0.9,
            "effect_fields_consumed": [],
            "full_run_budget_after_probes_seconds": 3600.0,
            "held_out_reads": 0,
            "outcome_fields_consumed": [],
            "prediction": prediction,
            "prediction_interval_seconds": [20.0, 40.0],
            "probe_process": {
                "exit_code": 0,
                "process_isolated": True,
                "start_method": "spawn",
                "status": "RESULT",
            },
            "profile_digest": sha256_digest(profile_identity),
            "status": "RESOURCE_ADMITTED",
        }

    return InnovationRuntimeInputs(
        implementer=lambda _request: _positive_response(),
        policy=implementer_policy,
        candidate_parent=tmp_path / "innovation",
        fixture_factory=lambda _policy, attempt, _root: _qualification_fixture(
            tmp_path, label=f"runtime-{attempt}"
        ),
        unit_check_factory=lambda _policy: _unit_check,
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="runtime-capability-v1",
        registry_version="runtime-registry-v1",
        predecessor_registry_ref="registry:fixed-66",
        predecessor_registry_digest=_digest("fixed-registry"),
        profile_version="runtime-profile-v2",
        fresh_campaign_id=f"{current.campaign_id}:fresh",
        resource_admission_required=resource_admission_required,
        resource_probe=(resource_probe if resource_admission_required else None),
        resource_probe_parent=(
            tmp_path / "resource-probes"
            if resource_admission_required
            else None
        ),
    )


def test_runner_binding_must_match_complete_selected_recipe() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:binding-check")
    context = _context(profile)
    proposals = _fixed_proposals(profile)
    calls: list[dict[str, Any]] = []

    with pytest.raises(ValueError, match="mechanism_id"):
        run_research_round(
            context=context,
            active_profile=profile,
            producer=lambda role, _view: proposals[role],
            producer_bindings=_bindings(context, profile),
            resolver_environment=_environment(profile),
            carryover_proposals=(),
            budget_snapshot={"experiment_opportunities": 1},
            router=_router(),
            policy=initial_research_policy(),
            memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
            runner=_runner(calls, binding_mutation={"mechanism_id": "forged"}),
            incumbent_observation=_incumbent(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed="54304",
            next_discriminative_test="reject a mismatched complete binding",
        )


def test_round_trace_consumes_current_provider_receipts() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:provider-trace")
    context = _context(profile)
    proposals = _fixed_proposals(profile)
    calls: list[dict[str, Any]] = []

    class TracedProducer:
        def __init__(self) -> None:
            self.call_traces: tuple[dict[str, Any], ...] = ()

        def __call__(self, role: str, _view: dict[str, Any]) -> Any:
            self.call_traces = (
                *self.call_traces,
                {
                    "kind": "research_producer",
                    "logical_call_id": f"trace:{role}",
                    "usage": {"billed_tokens": 11},
                },
            )
            return proposals[role]

    producer = TracedProducer()
    result = run_research_round(
        context=context,
        active_profile=profile,
        producer=producer,
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=initial_research_policy(),
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="consume Provider receipt trace",
    )

    assert len(result.provider_traces) == 4
    assert sum(trace["usage"]["billed_tokens"] for trace in result.provider_traces) == 44
    assert len(result.to_dict()["provider_traces"]) == 4


def test_complete_two_round_spine_activates_and_consumes_innovation(
    tmp_path: Path,
) -> None:
    current = adapt_current_search_profile(campaign_id="campaign:runtime-round-1")
    context = _context(current)
    policy = initial_research_policy()
    proposals = _fixed_proposals(current)
    innovation_proposal = _proposal(
        candidate_id="cand-runtime-qualified-interaction-gate",
        mechanism_id="INTERACTION_GATE_RUNTIME",
        mechanism_axis="message_transform",
        mechanism_program=_outside_66_program(),
        protocol_digest=current.protocol_digest,
        selected=True,
        role="mechanism_composer",
    )
    proposals["mechanism_composer"] = innovation_proposal

    def producer(role: str, _view: dict[str, Any]) -> Any:
        return proposals[role]

    innovation_inputs = _innovation_inputs(tmp_path, current)
    runner_calls: list[dict[str, Any]] = []
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    round_one = run_research_round(
        context=context,
        active_profile=current,
        producer=producer,
        producer_bindings=_bindings(context, current),
        resolver_environment=_environment(current),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=policy,
        memory_writer=writer,
        runner=_runner(runner_calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="rerun the matched control",
        innovation_inputs=innovation_inputs,
    )

    assert len(round_one.producer_outcomes) == 4
    assert len(runner_calls) == 1
    assert round_one.interpretation is not None
    assert round_one.interpretation.episode is not None
    assert (
        round_one.interpretation.episode.failure_class
        is ResearchFailureClassV1.INCONCLUSIVE
    )
    assert round_one.innovation is not None and round_one.innovation.admitted
    assert len(round_one.innovation.next_profile.executable_entries) == 67
    assert round_one.innovation.profile_receipt.current_profile_unchanged is True
    assert current.profile_ref == context.active_profile_ref
    trace = round_one.to_dict()
    assert trace["context"]["context_ref"] == context.context_ref
    assert trace["innovation"]["capability"] is not None
    assert trace["interpretation"]["episode"]["failure_class"] == "INCONCLUSIVE"

    active, context_two, admitted_proposal, qualified_execution = (
        activate_staged_innovation(round_one)
    )
    assert context_two.round_index == 2
    assert context_two.active_profile_ref == active.profile_ref
    assert "activated_capability" in context_two.producer_view("frontier_architect")[
        "scientific_memory"
    ]
    assert context_two.producer_inputs_digest != context.producer_inputs_digest

    round_two_fixed = _fixed_proposals(active, select_first=False)

    def round_two_producer(role: str, view: dict[str, Any]) -> Any:
        assert "activated_capability" in view["scientific_memory"]
        return round_two_fixed[role]

    round_two = run_research_round(
        context=context_two,
        active_profile=active,
        producer=round_two_producer,
        producer_bindings=_bindings(context_two, active),
        resolver_environment=_environment(active),
        carryover_proposals=(admitted_proposal,),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=round_one.interpretation.policy_successor,
        memory_writer=writer,
        runner=_runner(runner_calls, seed=54305),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54305",
        next_discriminative_test="test the interaction gate mechanism-off control",
        qualified_execution_by_capability={
            round_one.innovation.capability.capability_id: qualified_execution
        },
    )

    assert len(runner_calls) == 2
    assert round_two.selected_outcome is not None
    assert round_two.selected_outcome.source_proposal == admitted_proposal
    assert round_two.search_acquisition.selected_binding.entry_origin is (
        SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
    )
    assert round_two.execution_recipe["model"] == "GWaveOneInteractionGate"
    assert round_two.execution_recipe["model"] != "BPR"
    assert round_two.interpretation.behavior_before.changed_fields(
        round_two.interpretation.behavior_after
    )


def test_innovation_without_search_binding_still_builds_next_fresh_profile(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:innovation-only")
    context = _context(profile)
    proposals = {
        role: _proposal(
            candidate_id=f"cand-innovation-only-{index}",
            mechanism_id=f"INTERACTION_GATE_ONLY_{index}",
            mechanism_axis="message_transform",
            mechanism_program=_outside_66_program(),
            protocol_digest=profile.protocol_digest,
            selected=index == 0,
            role=role,
        )
        for index, role in enumerate(DISCOVERY_PRODUCERS)
    }

    result = run_research_round(
        context=context,
        active_profile=profile,
        producer=lambda role, _view: proposals[role],
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=initial_research_policy(),
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner([]),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="activate the admitted capability then search it",
        innovation_inputs=_innovation_inputs(
            tmp_path,
            profile,
            resource_admission_required=True,
        ),
    )

    assert result.search_acquisition is None
    assert result.interpretation.failure_taxonomy == (
        "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
    )
    assert result.innovation is not None and result.innovation.activation_ready
    assert result.innovation.mechanically_qualified is True
    assert result.innovation.resource_admitted is True
    assert result.innovation.resource_profile is not None
    assert result.innovation.quality_admission is not None
    assert result.innovation.quality_admission["runnable_probability"] == 0.9
    assert (
        result.innovation.search_candidate.utility_features.runnable_probability
        == 0.9
    )
    assert (
        result.innovation.qualification.stage_observations[
            "DISPOSABLE_PROCESS"
        ]["start_method"]
        == "spawn"
    )
    assert len(result.innovation.next_profile.executable_entries) == 67
    active, successor, _proposal_value, _execution = activate_staged_innovation(result)
    assert successor.campaign_id == active.campaign_id
    assert successor.active_profile_ref == active.profile_ref


def test_runtime_failure_is_consumed_as_diagnostic_and_still_advances_memory() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:runtime-diagnostic")
    context = _context(profile)
    proposals = _fixed_proposals(profile)
    current_semantics = _environment(profile)["current_capabilities"][0][
        "semantics_digest"
    ]
    search_open_draft = _open_draft(producer_role="falsification_designer")
    search_open_draft.update(
        high_change_justification=(
            "The open draft names current semantics but has no executable program identity."
        ),
        current_profile_expressibility_claim="EXPRESSIBLE",
        resolution_facts={
            "requested_current_semantics_digest": current_semantics,
            "capability_diff": (),
            "high_change_dimensions": (),
            "required_dependencies": (),
            "required_budget": {},
        },
    )
    calls: list[dict[str, Any]] = []

    def mixed_producer(role: str, _view: dict[str, Any]) -> Any:
        if role == "lineage_refiner":
            return _open_draft(producer_role=role)
        if role == "falsification_designer":
            return search_open_draft
        return proposals[role]

    result = run_research_round(
        context=context,
        active_profile=profile,
        producer=mixed_producer,
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=initial_research_policy(),
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        runner=_runner(calls, status="RUNTIME_FAILURE"),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="repair the runtime path",
    )

    assert len(calls) == 1
    assert result.interpretation is not None
    assert result.interpretation.episode is None
    assert result.interpretation.failure_taxonomy == "ENGINEERING_DIAGNOSTIC_RUNTIME"
    assert result.interpretation.mechanism_belief is None
    assert result.interpretation.successor_context.round_index == 2
    assert len(result.deferred_innovation_outcomes) == 1
    assert result.deferred_innovation_outcomes[0][0].producer_role == "lineage_refiner"
    assert len(result.deferred_search_outcomes) == 1
    assert result.deferred_search_outcomes[0][0].producer_role == (
        "falsification_designer"
    )


def test_successor_context_history_moves_same_seed_repeat_to_new_search_candidate() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:runtime-repeat-history")
    context = _context(profile)
    proposals = _fixed_proposals(profile, select_first=False)
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    calls: list[dict[str, Any]] = []

    def producer(_role: str, view: dict[str, Any]) -> Any:
        return proposals[_role]

    round_one = run_research_round(
        context=context,
        active_profile=profile,
        producer=producer,
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=initial_research_policy(),
        memory_writer=writer,
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="avoid an unrequested same-seed repeat",
    )

    assert round_one.search_acquisition is not None
    first_candidate_id = round_one.search_acquisition.selected_binding.proposal.candidate_id
    assert round_one.successor_context.scientific_memory["executed_observations"]
    _pairs, _pending, axis_effects = _search_ranking_inputs(
        round_one.successor_context
    )
    selected_axis = round_one.search_acquisition.selected_binding.proposal.mechanism_axis
    observation = round_one.successor_context.scientific_memory[
        "executed_observations"
    ][-1]
    assert observation["mechanism_axis"] == selected_axis
    assert observation["comparator_delta"] != "NOT_AVAILABLE"
    assert axis_effects[selected_axis] == float(observation["comparator_delta"])

    round_two = run_research_round(
        context=round_one.successor_context,
        active_profile=profile,
        producer=producer,
        producer_bindings=_bindings(round_one.successor_context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=round_one.interpretation.policy_successor,
        memory_writer=writer,
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="avoid an unrequested same-seed repeat",
    )

    assert round_two.search_acquisition is not None
    assert round_two.search_acquisition.selected_binding is not None
    assert round_two.search_acquisition.selected_binding.proposal.candidate_id != (
        first_candidate_id
    )
    assert round_two.context.scientific_memory["executed_observations"]
    assert round_two.search_acquisition.route_trace.policy_digest != (
        round_one.search_acquisition.route_trace.policy_digest
    )


def test_missing_search_opportunity_is_typed_memory_and_round_two_recovers() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:runtime-missing")
    context = _context(profile)
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    calls: list[dict[str, Any]] = []

    def failed_producer(role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError(f"fixture producer failure: {role}")

    round_one = run_research_round(
        context=context,
        active_profile=profile,
        producer=failed_producer,
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=initial_research_policy(),
        memory_writer=writer,
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="restore one legal BL-ICF Search candidate",
    )

    assert calls == []
    assert round_one.interpretation is not None
    assert round_one.interpretation.failure_taxonomy == (
        "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
    )
    assert round_one.interpretation.negative_evidence == ()
    latest = round_one.successor_context.scientific_memory["latest_feedback"]
    assert latest["failure_class"] == ResearchFailureClassV1.OUTCOME_MISSING.value
    assert latest["diagnostic_detail"]["reason"] == "NO_LEGAL_SEARCH_BINDING"
    assert latest["diagnostic_detail"]["producer_failures"] == {
        role: "PRODUCER_CALL_FAILED" for role in DISCOVERY_PRODUCERS
    }

    proposals = _fixed_proposals(profile)

    def recovered_producer(role: str, view: dict[str, Any]) -> Any:
        assert view["scientific_memory"]["latest_feedback"]["failure_class"] == (
            ResearchFailureClassV1.OUTCOME_MISSING.value
        )
        return proposals[role]

    round_two = run_research_round(
        context=round_one.successor_context,
        active_profile=profile,
        producer=recovered_producer,
        producer_bindings=_bindings(round_one.successor_context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=round_one.interpretation.policy_successor,
        memory_writer=writer,
        runner=_runner(calls, seed=54305),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54305",
        next_discriminative_test="compare the recovered candidate",
    )

    assert len(calls) == 1
    assert round_two.interpretation is not None
    assert round_two.interpretation.episode is not None


def test_offline_replay_shadow_promotion_changes_next_campaign_policy() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:meta-round-1")
    policy = meta_v17_static_producer_policy()
    context = _context(profile, policy=policy)
    proposals = _fixed_proposals(profile)
    calls: list[dict[str, Any]] = []
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    replay_calls: list[dict[str, Any]] = []

    def producer(role: str, _view: dict[str, Any]) -> Any:
        return proposals[role]

    def replay_producer(role: str, view: dict[str, Any]) -> Any:
        replay_calls.append(view)
        is_challenger = (
            view["memory_retrieval_policy"] == "ROLE_SCOPED_GAP_AWARE_V1"
        )
        draft = _open_draft(producer_role=role)
        if is_challenger:
            draft["mechanism_change"] = (
                f"{draft['mechanism_change']} Role-specific replay wedge: {role}."
            )
        return draft

    offline_replay = OfflineProducerReplayV1(
        producer=replay_producer,
        producer_bindings=_bindings(context, profile),
        equal_replay_token_charge=1000,
        deterministic_directive_replay=True,
    )

    round_one = run_research_round(
        context=context,
        active_profile=profile,
        producer=producer,
        producer_bindings=_bindings(context, profile),
        resolver_environment=_environment(profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=policy,
        memory_writer=writer,
        runner=_runner(calls),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54304",
        next_discriminative_test="replay the next strategy against the same contexts",
        meta_research_inputs=MetaResearchInputs(
            offline_replay=offline_replay,
            next_campaign_id="campaign:meta-round-2",
        ),
    )

    assert len(replay_calls) == 8
    assert all(view["scientific_memory"].get("search_memory_head") for view in replay_calls)
    assert round_one.meta_research is not None
    assert round_one.meta_research.shadow_evaluation.verdict == "PASS", (
        round_one.meta_research.shadow_evaluation.reason_codes,
        round_one.meta_research.shadow_evaluation.champion,
        round_one.meta_research.shadow_evaluation.challenger,
    )
    assert round_one.meta_research.promotion_decision.verdict == "PROMOTE"
    assert round_one.meta_research.activated_policy is not None
    assert round_one.meta_research.activation_receipt is not None
    assert round_one.meta_research.activated_policy.digest != (
        round_one.interpretation.policy_successor.digest
    )

    next_profile = adapt_current_search_profile(campaign_id="campaign:meta-round-2")
    context_two, policy_two = activate_promoted_meta_strategy(
        round_one,
        next_profile=next_profile,
    )
    assert context_two.policy == policy_two.to_dict()
    assert context_two.scientific_memory["meta_strategy"][
        "promotion_decision_digest"
    ] == round_one.meta_research.promotion_decision.digest
    assert context_two.producer_inputs_digest != round_one.successor_context.producer_inputs_digest

    round_two_proposals = _fixed_proposals(next_profile)

    def round_two_producer(role: str, view: dict[str, Any]) -> Any:
        assert view["policy"]["memory_retrieval_policy"] == (
            "ROLE_SCOPED_GAP_AWARE_V1"
        )
        assert "meta_strategy" in view["scientific_memory"]
        return round_two_proposals[role]

    round_two = run_research_round(
        context=context_two,
        active_profile=next_profile,
        producer=round_two_producer,
        producer_bindings=_bindings(context_two, next_profile),
        resolver_environment=_environment(next_profile),
        carryover_proposals=(),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        policy=policy_two,
        memory_writer=writer,
        runner=_runner(calls, seed=54305),
        incumbent_observation=_incumbent(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed="54305",
        next_discriminative_test="consume the promoted strategy",
    )

    assert len(calls) == 2
    assert round_two.interpretation is not None
