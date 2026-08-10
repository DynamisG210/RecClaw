from __future__ import annotations

import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import COMMON_EVALUATOR
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.research_line.bootstrap import bootstrap_search_pool
from recclaw_core.research_line.campaign import (
    CampaignRoundInputs,
    CampaignState,
    ResearchCampaign,
)
from recclaw_core.research_line.interfaces import (
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
    ResearchTaskStatusV2,
)
from recclaw_core.research_line.portfolio import (
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)
TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_open_candidate import _provider_open_draft  # noqa: E402
from test_dynamic_handoff_v2 import _profile_source  # noqa: E402
from test_runtime import (  # noqa: E402
    _bindings,
    _context,
    _environment,
    _fixed_proposals,
    _incumbent,
    _innovation_inputs,
    _router,
    _runner,
)


def _inputs(state: CampaignState, *, innovation: Any = None, seed: int = 54304):
    return CampaignRoundInputs(
        producer_bindings=_bindings(state.context, state.active_profile),
        resolver_environment=_environment(state.active_profile),
        budget_snapshot={"experiment_opportunities": 1},
        router=_router(),
        metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
        observation_seed=str(seed),
        next_discriminative_test="use the typed round feedback for the next decision",
        innovation_inputs=innovation,
    )


def _state(profile: Any, policy: Any, *, proposals=()):
    context = _context(profile, policy=policy)
    return CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=proposals,
    )


def _state_with_pending_candidate_task(
    profile: Any,
    policy: Any,
    proposal: Any,
    *,
    required_seed: str,
    evidence_present: tuple[str, ...] = (),
) -> CampaignState:
    context = _context(profile, policy=policy)
    entry = next(
        item
        for item in profile.entries
        if item.semantic_identity_ref
        == f"bl-icf-mechanism:{proposal.mechanism_id}"
    )
    task = ResearchTaskRecordV2(
        task_id=f"task:legacy-confirm:{proposal.candidate_id}",
        operation=ResearchTaskOperationV2.NEW_SEED,
        candidate_id=proposal.candidate_id,
        candidate_semantic_digest=entry.semantic_identity_digest,
        mechanism_program_digest=sha256_digest(proposal.mechanism_program),
        parent_candidate_id=proposal.parent_candidate_id,
        comparator_identity=_incumbent()["comparator_ref"],
        protocol_digest=context.protocol_digest,
        required_seed_or_control=required_seed,
        priority=1.0,
        created_round=context.round_index,
        evidence_present=evidence_present,
        mechanism_program=proposal.mechanism_program,
        status=ResearchTaskStatusV2.PENDING,
    )
    memory = dict(context.scientific_memory)
    global_memory = dict(memory.get("global_memory", {}))
    global_memory["task_queue"] = ResearchTaskQueueV2((task,)).to_dict()
    if evidence_present:
        global_memory["executed_observations"] = tuple(
            {
                "candidate_semantic_digest": entry.semantic_identity_digest,
                "observation_seed": seed,
            }
            for seed in evidence_present
        )
    memory["global_memory"] = global_memory
    context = replace(context, scientific_memory=memory)
    return CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=(proposal,),
    )


def test_missing_opportunity_is_typed_and_feedback_changes_round_two_inputs(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:campaign-missing")
    policy = initial_research_policy()
    initial = _state(profile, policy)
    producer_views: list[dict[str, Any]] = []

    def failed_producer(_role: str, view: dict[str, Any]) -> Any:
        producer_views.append(view)
        raise RuntimeError("fixture provider unavailable")

    def must_not_run(_recipe: Any, _binding: Any) -> Any:
        pytest.fail("missing opportunity must not invoke the runner")

    campaign = ResearchCampaign(
        root=tmp_path / "missing-campaign",
        state=initial,
        producer=failed_producer,
        runner=must_not_run,
        round_inputs=lambda state: _inputs(state, seed=55000 + state.round_index),
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )

    round_one = campaign.run_round()
    round_two = campaign.run_round()

    assert round_one.status == "OUTCOME_MISSING"
    assert round_two.status == "OUTCOME_MISSING"
    assert round_one.result.interpretation is not None
    assert round_one.result.interpretation.failure_taxonomy == (
        "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
    )
    assert campaign.state.next_round_index == 3
    assert len(producer_views) == 8
    assert producer_views[0]["context_digest"] != producer_views[4]["context_digest"]
    assert producer_views[4]["round_index"] == 2
    assert campaign.state.search_memory_head is not None


def test_resume_returns_sealed_round_without_another_physical_call(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:campaign-resume")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    initial = _state(
        profile,
        policy,
        proposals=bootstrap_search_pool(context, profile, policy),
    )
    calls: list[dict[str, Any]] = []

    def failed_producer(_role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError("no current Producer proposal")

    campaign = ResearchCampaign(
        root=tmp_path / "resume-campaign",
        state=initial,
        producer=failed_producer,
        runner=_runner(calls, seed=54304),
        round_inputs=lambda state: _inputs(state, seed=54304),
    )
    first = campaign.run_round()
    assert first.result.candidate_run is not None
    assert len(calls) == 1

    def should_not_produce(_role: str, _view: dict[str, Any]) -> Any:
        pytest.fail("a sealed round must not call the Producer on resume")

    resumed = ResearchCampaign.resume(
        root=tmp_path / "resume-campaign",
        producer=should_not_produce,
        runner=lambda _recipe, _binding: pytest.fail("sealed round must not run"),
        round_inputs=lambda state: _inputs(state, seed=54305),
    )
    replayed = resumed.run_round(round_index=1)

    assert replayed.digest == first.digest
    assert replayed.result.candidate_run == first.result.candidate_run
    assert len(calls) == 1
    assert not any(
        alias in first.state_after.context.scientific_memory.get("global_memory", {})
        for alias in (
            "portfolio_prior_attempts",
            "portfolio_family_history",
            "portfolio_parent_history",
            "portfolio_frontier_history",
        )
    )


def test_campaign_projects_bounded_portfolio_history_and_next_round_source_uses_it(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:portfolio-history")
    policy = initial_research_policy()
    proposals = _fixed_proposals(profile)
    proposal_items = tuple(proposals.values())
    initial = _state(profile, policy, proposals=proposal_items)
    portfolio: list[PortfolioCandidateV2] = []
    for index, proposal in enumerate(proposal_items, start=1):
        entry = next(
            item
            for item in profile.entries
            if item.semantic_identity_ref
            == f"bl-icf-mechanism:{proposal.mechanism_id}"
        )
        portfolio.append(
            PortfolioCandidateV2(
                candidate_id=proposal.candidate_id,
                semantic_digest=entry.semantic_identity_digest,
                family_id=proposal.mechanism_axis,
                parent_id=(
                    proposal_items[0].candidate_id
                    if index == 2
                    else None
                ),
                valid_seal_probability=0.9,
                family_delta=0.0,
                parent_delta=0.0,
                information_value=1.0,
                predicted_gpu_seconds=float(index),
                age_rounds=0,
                repeat_count=0,
                lineage_risk=0.0,
                compute_pattern=f"handoff-pattern-{index}",
                resource_admission_state=ResourceAdmissionStateV2.ACTIVE,
                frontier_gain=0.5,
            )
        )
    calls: list[dict[str, Any]] = []

    def producer(_role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError("portfolio history test uses frozen carryover proposals")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        status = "RESOURCE_CENSORED" if not calls else "SUCCESS"
        return _runner(calls, status=status, seed=54304)(
            recipe,
            binding,
        )

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        return replace(
            _inputs(state, seed=54304),
            budget_snapshot={
                "experiment_opportunities": 1,
                "round_attempt_budget": 2,
            },
            attempt_scheduler=True,
            max_attempts_per_round=2,
            portfolio_candidates=tuple(portfolio),
        )

    campaign = ResearchCampaign(
        root=tmp_path / "portfolio-history",
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    record = campaign.run_round()

    assert record.status == "TYPED_EPISODE"
    assert record.result.metric_bearing_attempt_index == 1
    assert record.result.attempts[0].failure_scope == "CANDIDATE_LOCAL"
    assert record.result.attempts[1].metric_bearing
    global_memory = record.state_after.context.scientific_memory["global_memory"]
    prior_rows = tuple(global_memory["portfolio_prior_attempts"])
    family_rows = tuple(global_memory["portfolio_family_history"])
    parent_rows = tuple(global_memory["portfolio_parent_history"])
    frontier_rows = tuple(global_memory["portfolio_frontier_history"])
    assert len(prior_rows) == 2
    assert len(family_rows) == len(parent_rows) == len(frontier_rows) == 1

    failed_id = record.result.attempts[0].candidate_id
    metric_id = record.result.attempts[1].candidate_id
    failed_row = next(row for row in prior_rows if row["candidate_id"] == failed_id)
    metric_row = next(row for row in prior_rows if row["candidate_id"] == metric_id)
    assert failed_row["sealed"] is True
    assert failed_row["sealed_valid_seal"] is False
    assert failed_row["sealed_resource_admitted"] is False
    assert metric_row["sealed_valid_seal"] is True
    assert metric_row["sealed_resource_admitted"] is True
    assert metric_row["compute_pattern"] == "handoff-pattern-2"
    assert failed_row["compute_pattern"] == "handoff-pattern-1"
    assert all(
        field_name not in row
        for row in (*prior_rows, *family_rows, *parent_rows, *frontier_rows)
        for field_name in ("candidate_run", "metrics", "outcome")
    )
    assert family_rows[0]["candidate_id"] == metric_id
    assert family_rows[0]["stable"] is True
    assert family_rows[0]["stable_delta"] == pytest.approx(0.05)
    assert parent_rows[0]["parent_id"] == proposal_items[0].candidate_id
    assert parent_rows[0]["stable_validation"] == "UNKNOWN"
    assert parent_rows[0]["stable_delta"] == pytest.approx(0.05)
    assert frontier_rows[0]["candidate_id"] == metric_id
    assert frontier_rows[0]["stable_frontier_gain"] == pytest.approx(0.05)
    assert all(
        len(global_memory[alias]) <= 64
        for alias in (
            "portfolio_prior_attempts",
            "portfolio_family_history",
            "portfolio_parent_history",
            "portfolio_frontier_history",
        )
    )

    assert record.result.prepared is not None
    source = _profile_source(initial.context, profile)
    next_round_profiles = source.build_profiles(
        context=record.state_after.context,
        active_profile=record.state_after.active_profile,
        resolutions=record.result.resolutions,
        search_bindings=record.result.prepared.search_bindings,
    )
    metric_profile = next(
        item
        for item in next_round_profiles
        if item.portfolio_profile.candidate.candidate_id == metric_id
    )
    assert metric_profile.portfolio_profile.candidate.repeat_count == 1
    assert metric_profile.portfolio_profile.candidate.family_delta > 0.0
    assert metric_profile.portfolio_profile.candidate.frontier_gain > 0.0

    state = record.state_after
    metric_attempt = record.result.attempts[1]
    for _ in range(65):
        successor_context = replace(
            state.context,
            round_index=state.round_index + 1,
            policy=state.policy.to_dict(),
            frontier=state.frontier,
            scientific_memory=state.context.scientific_memory,
        )
        interpretation = replace(
            record.result.interpretation,
            policy_successor=state.policy,
            successor_context=successor_context,
        )
        synthetic_result = replace(
            record.result,
            context=state.context,
            active_profile=state.active_profile,
            attempts=(metric_attempt,),
            metric_bearing_attempt_index=0,
            interpretation=interpretation,
        )
        state = campaign._advance_state(state, synthetic_result)

    final_global_memory = state.context.scientific_memory["global_memory"]
    assert len(final_global_memory["portfolio_prior_attempts"]) == 64
    assert len(final_global_memory["portfolio_family_history"]) == 64
    assert len(final_global_memory["portfolio_parent_history"]) == 64
    assert len(final_global_memory["portfolio_frontier_history"]) == 64
    assert len(state.context.scientific_memory["round_attempts"]) <= 64


def test_legacy_candidate_is_retained_for_pending_confirmation_then_retires(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:legacy-confirmation")
    policy = initial_research_policy()
    proposal = _fixed_proposals(profile)["mechanism_composer"]
    initial = _state(profile, policy, proposals=(proposal,))
    producer_calls: list[str] = []
    runner_calls: list[dict[str, Any]] = []

    def producer(role: str, _view: dict[str, Any]) -> Any:
        producer_calls.append(role)
        raise RuntimeError("legacy test uses the frozen carryover candidate")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        return _runner(
            runner_calls,
            seed=54304 + len(runner_calls),
        )(recipe, binding)

    campaign = ResearchCampaign(
        root=tmp_path / "legacy-confirmation",
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=lambda state: _inputs(
            state,
            seed=54304 + state.round_index - 1,
        ),
    )

    first = campaign.run_round()
    candidate_id = proposal.candidate_id
    assert first.result.search_acquisition is not None
    assert first.result.search_acquisition.selected_binding.proposal.candidate_id == (
        candidate_id
    )
    assert not first.result.attempt_scheduler_enabled
    assert first.result.attempts == ()
    first_queue = ResearchTaskQueueV2.from_dict(
        first.state_after.context.scientific_memory["global_memory"]["task_queue"]
    )
    first_task = next(
        item
        for item in first_queue.tasks
        if item.candidate_id == candidate_id
        and item.required_seed_or_control == "NEXT_DEVELOPMENT_SEED"
    )
    assert first_task.status in {
        ResearchTaskStatusV2.PENDING,
        ResearchTaskStatusV2.ACTIVE,
    }
    assert any(item.candidate_id == candidate_id for item in campaign.state.carryover_proposals)

    second = campaign.run_round()
    second_queue = ResearchTaskQueueV2.from_dict(
        second.state_after.context.scientific_memory["global_memory"]["task_queue"]
    )
    second_task = next(
        item for item in second_queue.tasks if item.task_id == first_task.task_id
    )
    assert second_task.status is ResearchTaskStatusV2.SATISFIED
    assert second.result.search_acquisition is not None
    assert second.result.search_acquisition.selected_binding.proposal.candidate_id == (
        candidate_id
    )
    assert not any(item.candidate_id == candidate_id for item in campaign.state.carryover_proposals)
    assert len(runner_calls) == 2
    assert len(producer_calls) == 8
    runner_calls_after_confirmation = len(runner_calls)

    third = campaign.run_round()
    assert third.status == "OUTCOME_MISSING"
    assert third.result.candidate_run is None
    assert len(runner_calls) == runner_calls_after_confirmation
    assert len(producer_calls) == 12


def test_legacy_generic_next_seed_task_satisfies_on_first_unseen_seed(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:legacy-next-seed")
    policy = initial_research_policy()
    proposal = _fixed_proposals(profile)["mechanism_composer"]
    initial = _state_with_pending_candidate_task(
        profile,
        policy,
        proposal,
        required_seed="NEXT_DEVELOPMENT_SEED",
        evidence_present=("54303",),
    )
    calls: list[dict[str, Any]] = []

    def producer(_role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError("legacy test uses the frozen carryover candidate")

    campaign = ResearchCampaign(
        root=tmp_path / "legacy-next-seed",
        state=initial,
        producer=producer,
        runner=lambda recipe, binding: _runner(calls, seed=54304)(recipe, binding),
        round_inputs=lambda state: _inputs(state, seed=54304),
    )

    first = campaign.run_round()
    queue = ResearchTaskQueueV2.from_dict(
        first.state_after.context.scientific_memory["global_memory"]["task_queue"]
    )
    task = next(
        item
        for item in queue.tasks
        if item.task_id == f"task:legacy-confirm:{proposal.candidate_id}"
    )
    assert task.status is ResearchTaskStatusV2.SATISFIED
    assert "54304" in task.evidence_present
    assert len(calls) == 1


def test_interrupted_unsealed_round_becomes_typed_missing_without_reexecution(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:campaign-interrupted")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    initial = _state(
        profile,
        policy,
        proposals=bootstrap_search_pool(context, profile, policy),
    )
    producer_calls: list[str] = []
    runner_calls: list[int] = []

    def failed_producer(role: str, _view: dict[str, Any]) -> Any:
        producer_calls.append(role)
        raise RuntimeError("no current Producer proposal")

    def interrupted_runner(_recipe: Any, _binding: Any) -> Any:
        runner_calls.append(1)
        raise RuntimeError("simulated interruption before durable observation")

    root = tmp_path / "interrupted-campaign"
    campaign = ResearchCampaign(
        root=root,
        state=initial,
        producer=failed_producer,
        runner=interrupted_runner,
        round_inputs=lambda state: _inputs(state, seed=54304),
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        campaign.run_round()

    def must_not_call(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("interrupted opportunity was executed again")

    resumed = ResearchCampaign.resume(
        root=root,
        producer=must_not_call,
        runner=must_not_call,
        round_inputs=lambda state: _inputs(state, seed=54304),
    )
    record = resumed.run_round()

    assert record.status == "OUTCOME_MISSING"
    assert record.result.candidate_run is None
    assert record.result.interpretation is not None
    assert record.result.interpretation.failure_taxonomy == (
        "ENGINEERING_DIAGNOSTIC_OUTCOME_MISSING"
    )
    assert resumed.state.next_round_index == 2
    assert resumed.state.search_memory_head is not None
    assert len(producer_calls) == 4
    assert runner_calls == [1]

def test_admitted_capability_is_carried_into_next_fresh_round(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:campaign-admit")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    initial = _state(
        profile,
        policy,
        proposals=bootstrap_search_pool(context, profile, policy),
    )
    innovation = _innovation_inputs(tmp_path, profile)
    implementer_calls: list[dict[str, Any]] = []
    original_implementer = innovation.implementer

    def implementer(request: dict[str, Any]) -> Any:
        implementer_calls.append(request)
        return original_implementer(request)

    innovation = replace(innovation, implementer=implementer)
    calls: list[dict[str, Any]] = []

    def producer(role: str, _view: dict[str, Any]) -> Any:
        return _provider_open_draft(
            role,
            capability_family="OPEN_INTERACTION_CUSTOM",
            base_model_config="BPR",
            model="GWaveOneInteractionGate",
        )

    def round_inputs(state: CampaignState) -> CampaignRoundInputs:
        return _inputs(
            state,
            innovation=innovation if state.round_index == 1 else None,
            seed=54304 + state.round_index - 1,
        )

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        return _runner(calls, seed=54304 + len(calls))(recipe, binding)

    campaign = ResearchCampaign(
        root=tmp_path / "admission-campaign",
        state=initial,
        producer=producer,
        implementer=implementer,
        runner=runner,
        round_inputs=round_inputs,
    )
    first = campaign.run_round()

    assert first.result.innovation is not None
    assert first.result.innovation.activation_ready
    capability = first.result.innovation.capability
    assert capability is not None
    assert campaign.state.active_profile.campaign_id == innovation.fresh_campaign_id
    assert any(
        entry.capability_ref == capability.capability_id
        for entry in campaign.state.active_profile.entries
    )
    assert campaign.state.candidate_root_by_capability[capability.capability_id]
    assert any(
        candidate.capability_ref == capability.capability_id
        for candidate in campaign.state.carryover_open_candidates
    )
    assert implementer_calls
    episode = first.result.interpretation.episode
    assert episode is not None
    assert campaign.state.incumbent_observation == {
        "comparator_ref": episode.outcome_ref,
        "comparator_digest": episode.outcome_digest,
        "frozen_ndcg@10": 0.45,
    }

    second = campaign.run_round()
    assert second.state_before.active_profile.campaign_id == innovation.fresh_campaign_id
    assert second.result.search_acquisition is not None
    assert (
        second.result.search_acquisition.selected_binding.entry_origin.value
        == "QUALIFIED_REGISTRY"
    )
    assert (
        second.result.search_acquisition.selected_binding.capability_ref
        == capability.capability_id
    )
    assert not any(
        candidate.capability_ref == capability.capability_id
        for candidate in campaign.state.carryover_open_candidates
    )
    assert len(calls) == 2


def test_fifty_round_campaign_scheduler_persists_and_closes_tasks(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:persisted-fifty-rounds")
    policy = initial_research_policy()
    fixed_proposals = _fixed_proposals(profile)
    valid_roles = tuple(fixed_proposals)[:2]
    valid_proposals = {
        role: fixed_proposals[role]
        for role in valid_roles
    }
    initial_with_task = _state_with_pending_candidate_task(
        profile,
        policy,
        valid_proposals[valid_roles[0]],
        required_seed="61001",
    )
    initial = replace(initial_with_task, carryover_proposals=())
    calls: list[dict[str, Any]] = []
    producer_calls: list[str] = []
    physical_pairs: list[tuple[str, str]] = []
    runner_invocations_by_round: dict[int, int] = {}
    active_round = 0
    active_seed = ""
    crash_injected = False
    task_directed_rounds = 0
    role_memory_divergence_rounds = 0
    state_sizes: list[int] = []
    round_attempt_memory_sizes: list[int] = []
    trace_sizes: list[int] = []

    def producer(role: str, view: dict[str, Any]) -> Any:
        producer_calls.append(role)
        round_index = int(view["round_index"])
        if round_index != 2 and role in valid_proposals:
            return valid_proposals[role]
        raise RuntimeError("the campaign reuses its frozen executable pool")

    def portfolio_for(state: CampaignState) -> tuple[PortfolioCandidateV2, ...]:
        global_memory = state.context.scientific_memory.get("global_memory", {})
        queue = ResearchTaskQueueV2.from_dict(global_memory.get("task_queue"))
        head = queue.select_next()
        task_semantic = (
            head.candidate_semantic_digest
            if head is not None
            and head.operation is ResearchTaskOperationV2.NEW_SEED
            else None
        )
        proposals = {
            item.candidate_id: item for item in state.carryover_proposals
        }
        if not proposals:
            proposals.update(
                {
                    item.candidate_id: item
                    for item in valid_proposals.values()
                }
            )
        candidates: list[PortfolioCandidateV2] = []
        for index, proposal in enumerate(proposals.values(), start=1):
            entry = next(
                item
                for item in state.active_profile.entries
                if item.semantic_identity_ref
                == f"bl-icf-mechanism:{proposal.mechanism_id}"
            )
            candidates.append(
                PortfolioCandidateV2(
                    candidate_id=proposal.candidate_id,
                    semantic_digest=entry.semantic_identity_digest,
                    family_id=proposal.mechanism_axis,
                    parent_id=None,
                    valid_seal_probability=0.95,
                    family_delta=0.0,
                    parent_delta=0.0,
                    information_value=0.8,
                    predicted_gpu_seconds=(
                        500.0
                        if entry.semantic_identity_digest == task_semantic
                        else float(10 + index)
                    ),
                    age_rounds=0,
                    repeat_count=0,
                    lineage_risk=0.0,
                    compute_pattern=f"fake-pattern-{index}",
                    resource_admission_state=ResourceAdmissionStateV2.ACTIVE,
                    frontier_gain=0.5,
                )
            )
        return tuple(candidates)

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        nonlocal active_round, active_seed
        active_round = state.round_index
        seed = 61000 + active_round
        active_seed = str(seed)
        return CampaignRoundInputs(
            producer_bindings=_bindings(state.context, state.active_profile),
            resolver_environment=_environment(state.active_profile),
            budget_snapshot={
                "experiment_opportunities": 50,
                "round_attempt_budget": 4,
            },
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed=active_seed,
            confirmation_seed=(
                str(seed + 1) if state.round_index < 50 else None
            ),
            next_discriminative_test="close the durable confirmation task",
            attempt_scheduler=True,
            max_attempts_per_round=4,
            portfolio_candidates=portfolio_for(state),
        )

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        nonlocal crash_injected
        invocation = runner_invocations_by_round.get(active_round, 0)
        runner_invocations_by_round[active_round] = invocation + 1
        if active_round == 10 and invocation == 1 and not crash_injected:
            crash_injected = True
            raise RuntimeError("simulated crash before second physical observation")
        multi_attempt_round = active_round != 2
        status = (
            "RESOURCE_CENSORED"
            if multi_attempt_round and invocation == 0
            else "SUCCESS"
        )
        result = dict(
            _runner(
                calls,
                status=status,
                seed=int(active_seed),
            )(recipe, binding)
        )
        if status == "SUCCESS":
            result["metrics"] = {"ndcg@10": 0.41}
        pair = (binding.proposal.candidate_id, str(result["seed"]))
        assert pair not in physical_pairs
        physical_pairs.append(pair)
        return result

    root = tmp_path / "persisted-fifty-rounds"
    campaign = ResearchCampaign(
        root=root,
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    records: list[Any] = []
    for round_index in range(1, 51):
        if round_index == 26:
            prior_digest = campaign.state.digest
            campaign = ResearchCampaign.resume(
                root=root,
                producer=producer,
                runner=runner,
                round_inputs=inputs,
            )
            assert campaign.state.digest == prior_digest

        before_queue = ResearchTaskQueueV2.from_dict(
            campaign.state.context.scientific_memory.get("global_memory", {}).get(
                "task_queue"
            )
        )
        before_head = before_queue.select_next()
        if round_index == 10:
            with pytest.raises(
                RuntimeError,
                match="simulated crash before second physical observation",
            ):
                campaign.run_round()
            manifest = json.loads(
                (root / "ROUND_10_ATTEMPT_MANIFEST.json").read_text(
                    encoding="utf-8"
                )
            )
            assert len(manifest["attempts"]) == 1
            assert not (root / "ROUND_10_CHECKPOINT.pkl").exists()
            assert not (root / "ROUND_10_TRACE.json").exists()
            assert len(producer_calls) == 40
            assert len(calls) == 18
            campaign = ResearchCampaign.resume(
                root=root,
                producer=producer,
                runner=runner,
                round_inputs=inputs,
            )

        record = campaign.run_round()
        records.append(record)
        assert record.round_index == round_index
        assert record.status == "TYPED_EPISODE"
        assert record.result.has_metric_bearing_attempt
        expected_attempt_count = 1 if round_index == 2 else 2
        assert record.result.metric_bearing_attempt_index == expected_attempt_count - 1
        assert len(record.result.attempts) == expected_attempt_count
        if expected_attempt_count == 2:
            assert record.result.attempts[0].engineering_disposition == (
                "ENGINEERING_FAILURE"
            )
            assert record.result.attempts[1].metric_bearing
            assert record.result.attempts[0].diagnostic_successor_context is not None
            assert (
                record.result.attempts[0].diagnostic_search_memory_snapshot
                is not None
            )
        else:
            assert record.result.attempts[0].metric_bearing
        assert not record.result.provider_traces
        attempt_history = record.state_after.context.scientific_memory.get(
            "round_attempts", ()
        )
        assert len(attempt_history) <= 64
        assert all(
            item.get("schema")
            == "recclaw.research-line.round-attempt-summary.v1"
            for item in attempt_history
        )
        assert all(
            field_name not in item
            for item in attempt_history
            for field_name in (
                "diagnostic_successor_context",
                "diagnostic_policy_successor",
                "diagnostic_search_memory_snapshot",
                "candidate_run",
                "execution_recipe",
            )
        )
        state_sizes.append(
            len(pickle.dumps(record.state_after, protocol=pickle.HIGHEST_PROTOCOL))
        )
        round_attempt_memory_sizes.append(
            len(pickle.dumps(attempt_history, protocol=pickle.HIGHEST_PROTOCOL))
        )
        if (
            before_head is not None
            and before_head.operation is ResearchTaskOperationV2.NEW_SEED
            and before_head.required_seed_or_control == str(61000 + round_index)
        ):
            task_directed_rounds += 1
            assert (
                record.result.attempts[0].binding.mechanism_semantics_digest
                == before_head.candidate_semantic_digest
            )
        before_roles = record.state_before.context.scientific_memory["by_role"]
        after_roles = record.state_after.context.scientific_memory["by_role"]
        assert after_roles != before_roles
        assert any(
            isinstance(role_memory, dict)
            and role_memory.get("execution_history")
            for role_memory in after_roles.values()
        )
        assert len({sha256_digest(item) for item in after_roles.values()}) >= 2
        role_memory_divergence_rounds += 1

    assert campaign.state.next_round_index == 51
    assert len(calls) == 99
    assert len(physical_pairs) == 99
    assert len(set(physical_pairs)) == len(physical_pairs)
    assert len(producer_calls) == 200
    assert all(producer_calls.count(role) == 50 for role in fixed_proposals)
    assert runner_invocations_by_round[1] == 2
    assert runner_invocations_by_round[2] == 1
    assert runner_invocations_by_round[10] == 3
    assert all(
        runner_invocations_by_round[round_index] == 2
        for round_index in range(1, 51)
        if round_index not in {2, 10}
    )
    assert crash_injected
    assert task_directed_rounds >= 1
    assert role_memory_divergence_rounds == 50
    assert len(round_attempt_memory_sizes) == 50
    assert len(round_attempt_memory_sizes[-1:]) == 1
    assert len(
        campaign.state.context.scientific_memory["round_attempts"]
    ) == 64
    assert max(state_sizes) < 8_000_000
    assert max(round_attempt_memory_sizes) < 1_000_000
    global_memory = campaign.state.context.scientific_memory["global_memory"]
    observations = global_memory["executed_observations"]
    semantic_seed_pairs = {
        (item["candidate_semantic_digest"], item["observation_seed"])
        for item in observations
    }
    # Scientific memory is the bounded routing working set, not the immutable
    # campaign evidence store.  All 99 attempts remain sealed below while the
    # successor context carries only the latest 64 observations.
    assert len(observations) == 64
    assert len(semantic_seed_pairs) == 64
    queue = ResearchTaskQueueV2.from_dict(global_memory["task_queue"])
    assert sum(
        item.status is ResearchTaskStatusV2.SATISFIED for item in queue.tasks
    ) >= 1

    for round_index, record in enumerate(records, start=1):
        checkpoint = root / f"ROUND_{round_index:02d}_CHECKPOINT.pkl"
        trace = root / f"ROUND_{round_index:02d}_TRACE.json"
        manifest_path = root / f"ROUND_{round_index:02d}_ATTEMPT_MANIFEST.json"
        assert checkpoint.is_file()
        assert trace.is_file()
        assert manifest_path.is_file()
        trace_sizes.append(trace.stat().st_size)
        trace_payload = json.loads(trace.read_text(encoding="utf-8"))
        assert trace_payload["record_digest"] == record.digest
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "TYPED_EPISODE"
        expected_attempt_count = 1 if round_index == 2 else 2
        assert manifest["metric_bearing_attempt_index"] == expected_attempt_count - 1
        assert [item["attempt_index"] for item in manifest["attempts"]] == list(
            range(expected_attempt_count)
        )
        assert len(
            list(
                root.glob(
                    f"ROUND_{round_index:02d}_ATTEMPT_*_PHYSICAL_OBSERVATION.json"
                )
            )
        ) == expected_attempt_count

    assert len(trace_sizes) == 50
    assert max(trace_sizes) < 50_000_000
    assert len(
        list(root.glob("ROUND_*_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))
    ) == 99

    final_digest = campaign.state.digest

    def must_not_produce(_role: str, _view: dict[str, Any]) -> Any:
        pytest.fail("final sealed resume must not call the Producer")

    def must_not_run(_recipe: Any, _binding: Any) -> Any:
        pytest.fail("final sealed resume must not call the runner")

    resumed = ResearchCampaign.resume(
        root=root,
        producer=must_not_produce,
        runner=must_not_run,
        round_inputs=inputs,
    )
    assert resumed.state.digest == final_digest
    replayed = resumed.run_round(round_index=50)
    assert replayed.digest == records[-1].digest
    assert resumed.state.digest == final_digest
