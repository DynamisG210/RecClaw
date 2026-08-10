from __future__ import annotations

from dataclasses import replace

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.research_line.portfolio import (
    AttemptFailureScopeV2,
    ParentValidationStateV2,
    PortfolioAttemptFailureV2,
    PortfolioCandidateV2,
    PortfolioControlStateV2,
    PortfolioEligibilityV2,
    PortfolioError,
    ResourceAdmissionStateV2,
    assess_resource_reserve_v2,
    rank_candidate_portfolio_v2,
    rerank_after_attempt_failure_v2,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DiscoveryCreditV1,
    MatchedControlPlanV1,
    ProposalIntentV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchAdapterError,
    SearchExecutableProfileV1,
    adapt_current_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    route_frozen_experiment_slate,
)


def _digest(label: str) -> str:
    return sha256_digest({"router-portfolio-v2": label})


def _candidate(
    index: int,
    *,
    pattern: str | None = None,
    p: float = 0.9,
    family_delta: float = 0.0,
    parent_delta: float = 0.0,
    info: float = 0.6,
    gpu_seconds: float = 100.0,
    age: int = 0,
    repeats: int = 0,
    lineage_risk: float = 0.0,
    resource_state: ResourceAdmissionStateV2 = (
        ResourceAdmissionStateV2.RESOURCE_ADMITTED
    ),
    parent_state: ParentValidationStateV2 = ParentValidationStateV2.INDEPENDENT,
    parent_id: str | None = None,
    frontier_gain: float = 0.5,
    dominated_by: str | None = None,
) -> PortfolioCandidateV2:
    return PortfolioCandidateV2(
        candidate_id=f"candidate-{index}",
        semantic_digest=_digest(f"semantic-{index}"),
        family_id=f"family-{index}",
        parent_id=parent_id,
        valid_seal_probability=p,
        family_delta=family_delta,
        parent_delta=parent_delta,
        information_value=info,
        predicted_gpu_seconds=gpu_seconds,
        age_rounds=age,
        repeat_count=repeats,
        lineage_risk=lineage_risk,
        compute_pattern=pattern or f"pattern-{index}",
        resource_admission_state=resource_state,
        parent_state=parent_state,
        frontier_gain=frontier_gain,
        dominated_by=dominated_by,
    )


def test_separate_heads_use_gpu_cost_and_retain_family_parent_delta() -> None:
    expensive = _candidate(
        1,
        gpu_seconds=400.0,
        family_delta=0.20,
        parent_delta=0.10,
    )
    cheap = _candidate(
        2,
        gpu_seconds=100.0,
        family_delta=0.20,
        parent_delta=0.10,
    )

    ranking = rank_candidate_portfolio_v2((expensive, cheap))
    assert ranking.selected_candidate_id == cheap.candidate_id
    row = next(
        item
        for item in ranking.records
        if item.candidate.candidate_id == cheap.candidate_id
    )
    assert row.candidate.family_delta == 0.20
    assert row.candidate.parent_delta == 0.10
    assert row.candidate.predicted_gpu_seconds == 100.0
    assert row.effective_valid_seal_probability == 0.9

    altered_parent = replace(cheap, parent_delta=-0.90)
    altered_expensive = replace(expensive, predicted_gpu_seconds=100.0)
    altered = rank_candidate_portfolio_v2((altered_expensive, altered_parent))
    assert altered.selected_candidate_id == expensive.candidate_id


def test_lineage_and_resource_states_do_not_enter_formal_pool() -> None:
    safe = _candidate(1)
    unverified = _candidate(
        2,
        parent_id="parent-open",
        parent_state=ParentValidationStateV2.UNVERIFIED,
    )
    failed_parent = _candidate(
        3,
        parent_id="parent-failed",
        parent_state=ParentValidationStateV2.FAILED,
    )
    stale = _candidate(4, age=7)
    dominated = _candidate(5, dominated_by=safe.candidate_id)
    qualified_only = _candidate(
        6,
        resource_state=ResourceAdmissionStateV2.QUALIFIED,
    )
    state_stale = _candidate(
        7,
        resource_state=ResourceAdmissionStateV2.STALE,
    )
    ranking = rank_candidate_portfolio_v2(
        (
            safe,
            unverified,
            failed_parent,
            stale,
            dominated,
            qualified_only,
            state_stale,
        ),
        control=PortfolioControlStateV2(
            stale_after_rounds=7,
            dominated_candidate_ids=(dominated.candidate_id,),
        ),
    )
    by_id = {item.candidate.candidate_id: item for item in ranking.records}
    assert ranking.selected_candidate_id == safe.candidate_id
    assert by_id[unverified.candidate_id].eligibility is (
        PortfolioEligibilityV2.UNVERIFIED_PARENT
    )
    assert by_id[failed_parent.candidate_id].eligibility is (
        PortfolioEligibilityV2.FAILED_PARENT
    )
    assert by_id[stale.candidate_id].eligibility is PortfolioEligibilityV2.STALE
    assert by_id[dominated.candidate_id].eligibility is (
        PortfolioEligibilityV2.DOMINATED
    )
    assert by_id[qualified_only.candidate_id].eligibility is (
        PortfolioEligibilityV2.NOT_RESOURCE_ADMITTED
    )
    assert by_id[state_stale.candidate_id].eligibility is (
        PortfolioEligibilityV2.STALE
    )


def test_top4_is_only_first_window_and_failover_reaches_full_pool() -> None:
    candidates = tuple(_candidate(index, gpu_seconds=100.0 + index) for index in range(1, 7))
    initial = rank_candidate_portfolio_v2(
        candidates,
        control=PortfolioControlStateV2(first_window_size=4),
    )
    assert len(initial.first_window_candidate_ids) == 4
    assert len(initial.ranked_candidate_ids) == 6

    failed = PortfolioAttemptFailureV2(
        candidate_id=initial.selected_candidate_id,
        scope=AttemptFailureScopeV2.CANDIDATE_LOCAL,
        reason="resource_censored",
    )
    reranked = rerank_after_attempt_failure_v2(
        candidates,
        failed,
        control=PortfolioControlStateV2(first_window_size=4),
    )
    assert reranked.selected_candidate_id != failed.candidate_id
    assert failed.candidate_id not in reranked.ranked_candidate_ids
    assert len(reranked.ranked_candidate_ids) == 5
    assert any(
        candidate_id not in initial.first_window_candidate_ids
        for candidate_id in reranked.first_window_candidate_ids
    )


def test_correlated_compute_patterns_are_not_multiplied_as_independent() -> None:
    first = _candidate(1, pattern="slow-pattern", p=0.8)
    second = _candidate(2, pattern="slow-pattern", p=0.8)
    independent = _candidate(3, pattern="independent-pattern", p=0.6)
    reserve = assess_resource_reserve_v2((first, second, independent))
    assert reserve.probability_at_least_one_valid_seal == pytest.approx(0.92)
    assert reserve.correlated_compute_groups[1][0] == "slow-pattern" or (
        reserve.correlated_compute_groups[0][0] == "slow-pattern"
    )

    blocked = rank_candidate_portfolio_v2(
        (first, second, independent),
        control=PortfolioControlStateV2(
            compute_pattern_failures={"slow-pattern": 2},
            compute_pattern_attempts={"slow-pattern": 2},
        ),
    )
    by_id = {item.candidate.candidate_id: item for item in blocked.records}
    assert by_id[first.candidate_id].eligibility is (
        PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN
    )
    assert by_id[second.candidate_id].eligibility is (
        PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN
    )
    assert blocked.selected_candidate_id == independent.candidate_id

    automatic_circuit = PortfolioControlStateV2()
    automatic_circuit = automatic_circuit.with_attempt_failure(
        PortfolioAttemptFailureV2(
            candidate_id="failed-a",
            scope=AttemptFailureScopeV2.LINEAGE_COMPUTE_PATTERN,
            compute_pattern="slow-pattern",
        )
    )
    automatic_circuit = automatic_circuit.with_attempt_failure(
        PortfolioAttemptFailureV2(
            candidate_id="failed-b",
            scope=AttemptFailureScopeV2.LINEAGE_COMPUTE_PATTERN,
            compute_pattern="slow-pattern",
        )
    )
    automatic = rank_candidate_portfolio_v2(
        (first, second, independent), control=automatic_circuit
    )
    automatic_by_id = {
        item.candidate.candidate_id: item for item in automatic.records
    }
    assert automatic_by_id[first.candidate_id].eligibility is (
        PortfolioEligibilityV2.CORRELATED_COMPUTE_PATTERN
    )


def test_candidate_mapping_rejects_current_outcome_leak() -> None:
    payload = {
        "candidate_id": "candidate-leak",
        "semantic_digest": _digest("leak"),
        "family_id": "family",
        "parent_id": None,
        "valid_seal_probability": 0.8,
        "family_delta": 0.0,
        "parent_delta": 0.0,
        "information_value": 0.5,
        "predicted_gpu_seconds": 100.0,
        "age_rounds": 0,
        "repeat_count": 0,
        "lineage_risk": 0.0,
        "compute_pattern": "pattern",
        "outcome": 0.99,
    }
    with pytest.raises(PortfolioError, match="outcome-bearing"):
        PortfolioCandidateV2.from_mapping(payload)


def _legacy_proposal(
    profile: SearchExecutableProfileV1,
    index: int,
    candidate_id: str,
) -> CandidateProposalV4:
    entry = profile.entries[index]
    mechanism_id = entry.capability_ref.rsplit(":", 1)[-1]
    mechanism = next(
        item for item in executable_mechanisms() if item.mechanism_id == mechanism_id
    )
    utility = SearchUtilityFeaturesV1(
        runnable_probability=0.8,
        useful_signal=0.7,
        frontier_potential=0.6,
        information_gain=0.6,
        cost=0.2,
        blocker_risk=0.05,
    )
    control = MatchedControlPlanV1(
        mechanism_question_digest=_digest(f"question-{index}"),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest=profile.protocol_digest,
        changed_axis=mechanism.mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id="producer:router-portfolio-v2",
        producer_role="mechanism_composer",
        proposal_intent=ProposalIntentV1.DISCOVERY,
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        candidate_label="portfolio integration fixture",
        mechanism_hypothesis="the candidate is executable",
        competing_hypothesis="the candidate is not useful",
        predicted_outcome_signature="predeclared fixture",
        failure_mode="the fixture fails",
        utility_features=utility,
        feature_evidence=RouterFeatureEvidenceV1(
            compile_valid=True,
            handler_available=True,
            materializer_available=True,
            blocker_rate=utility.blocker_risk,
            semantic_duplicate=False,
            parent_available=True,
            mechanism_depth=1,
            estimated_cost=utility.cost,
            llm_diagnostic=utility,
        ),
        matched_control_plan=control,
        discriminative_plan=None,
        parent_candidate_id=None,
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


def _portfolio_route_fixture():
    current = adapt_current_search_profile(
        campaign_id="campaign:router-portfolio-v2"
    )
    profile = replace(
        current,
        entries=current.entries[:6],
        campaign_id=current.campaign_id,
    )
    proposals = tuple(
        _legacy_proposal(profile, index, f"cand-portfolio-{index + 1}")
        for index in range(6)
    )
    bindings = tuple(
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=profile.entries[index].capability_ref,
        )
        for index, proposal in enumerate(proposals)
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot={"training_runs": 1},
    )
    portfolio = tuple(
        replace(
            _candidate(
                index + 1,
                p=0.9,
                info=0.8,
                gpu_seconds=100.0 + index,
            ),
            candidate_id=bindings[index].proposal.candidate_id,
            semantic_digest=bindings[index].mechanism_semantics_digest,
        )
        for index in range(6)
    )
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=1,
    )
    return profile, slate, portfolio, router


def test_search_adapter_portfolio_path_preserves_router_compatibility() -> None:
    profile, slate, portfolio, router = _portfolio_route_fixture()
    first = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        portfolio_candidates=portfolio,
        first_window_size=4,
    )
    assert len(first.route_trace.ranked_candidate_ids) == 4
    assert len(first.route_trace.decisions) == 6
    assert all(item.allowed for item in first.route_trace.decisions)

    failed = PortfolioAttemptFailureV2(
        candidate_id=first.route_trace.selected_candidate_id,
        reason="worker_resource_censored",
    )
    second = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        portfolio_candidates=portfolio,
        attempt_failures=(failed,),
        first_window_size=4,
    )
    assert second.selected_binding is not None
    assert second.selected_binding.proposal.candidate_id != failed.candidate_id
    failed_decision = next(
        item
        for item in second.route_trace.decisions
        if item.candidate_id == failed.candidate_id
    )
    assert not failed_decision.allowed


def test_pending_new_seed_task_preempts_unrelated_portfolio_exploration() -> None:
    profile, slate, portfolio, router = _portfolio_route_fixture()
    target = portfolio[-1]
    seed = "seed:confirmation"

    routed = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        portfolio_candidates=portfolio,
        current_observation_seed=seed,
        pending_task={
            "task_id": _digest("confirmation-task"),
            "task_type": "VALIDATE_SAME_CANDIDATE",
            "operation": "NEW_SEED",
            "candidate_semantic_digest": target.semantic_digest,
            "required_seed_or_control": seed,
            "task_status": "PENDING",
            "priority": 1.0,
        },
        first_window_size=4,
    )

    assert routed.selected_binding is not None
    assert routed.selected_binding.proposal.candidate_id == target.candidate_id
    assert routed.route_trace.ranked_candidate_ids[0] == target.candidate_id


def test_portfolio_failure_without_profiles_is_rejected() -> None:
    profile, slate, _portfolio, router = _portfolio_route_fixture()
    failure = PortfolioAttemptFailureV2(
        candidate_id=slate.bindings[0].proposal.candidate_id,
    )
    with pytest.raises(SearchAdapterError, match="explicit PortfolioCandidateV2"):
        route_frozen_experiment_slate(
            profile=profile,
            slate=slate,
            router=router,
            attempt_failures=(failure,),
        )


def test_portfolio_route_rejects_a_missing_candidate_profile() -> None:
    profile, slate, portfolio, router = _portfolio_route_fixture()
    with pytest.raises(SearchAdapterError, match="missing"):
        route_frozen_experiment_slate(
            profile=profile,
            slate=slate,
            router=router,
            portfolio_candidates=portfolio[:-1],
        )


def test_legacy_route_without_portfolio_arguments_is_unchanged() -> None:
    profile, slate, _portfolio, router = _portfolio_route_fixture()
    first = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
    )
    second = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
    )
    assert first.route_trace == second.route_trace
    assert len(first.route_trace.ranked_candidate_ids) == 1
