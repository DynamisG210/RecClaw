from __future__ import annotations

import json
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
from recclaw_core.research_line.campaign import (
    CampaignError,
    CampaignRoundInputs,
    CampaignState,
    ResearchCampaign,
)
from recclaw_core.research_line.portfolio import (
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)
from recclaw_core.research_line.runtime import run_research_round

from test_runtime import (
    _bindings,
    _context,
    _environment,
    _fixed_proposals,
    _incumbent,
    _router,
    _runner,
)


def _round_kwargs(profile: Any, context: Any, *, runner: Any) -> dict[str, Any]:
    return {
        "context": context,
        "active_profile": profile,
        "producer": lambda role, _view: _fixed_proposals(profile)[role],
        "producer_bindings": _bindings(context, profile),
        "resolver_environment": _environment(profile),
        "carryover_proposals": (),
        "budget_snapshot": {"experiment_opportunities": 1},
        "router": _router(),
        "policy": initial_research_policy(),
        "memory_writer": SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
        "runner": runner,
        "incumbent_observation": _incumbent(),
        "metric_contract_digest": sha256_digest(COMMON_EVALUATOR),
        "observation_seed": "54304",
        "next_discriminative_test": "confirm the metric-bearing fallback",
        "attempt_scheduler": True,
        "max_attempts_per_round": 4,
    }


@pytest.mark.parametrize("failure_count", (1, 3))
def test_candidate_local_failure_reroutes_full_pool_until_metric(
    failure_count: int,
) -> None:
    profile = adapt_current_search_profile(
        campaign_id=f"campaign:attempt-failover-{failure_count}"
    )
    context = _context(profile)
    calls: list[dict[str, Any]] = []
    failure_runner = _runner(calls, status="RESOURCE_CENSORED")
    success_runner = _runner(calls, status="SUCCESS")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        return (
            failure_runner(recipe, binding)
            if len(calls) < failure_count
            else success_runner(recipe, binding)
        )

    result = run_research_round(**_round_kwargs(profile, context, runner=runner))

    assert len(calls) == failure_count + 1
    assert len(result.attempts) == failure_count + 1
    assert result.attempts[0].engineering_disposition == "ENGINEERING_FAILURE"
    assert result.attempts[0].failure_scope == "CANDIDATE_LOCAL"
    assert all(
        attempt.engineering_disposition == "ENGINEERING_FAILURE"
        for attempt in result.attempts[:-1]
    )
    assert result.attempts[-1].metric_bearing
    assert result.metric_bearing_attempt_index == failure_count
    assert result.interpretation is not None
    assert result.interpretation.episode is not None
    assert result.interpretation.successor_context.round_index == context.round_index + 1
    assert (
        len({attempt.candidate_id for attempt in result.attempts})
        == failure_count + 1
    )
    assert result.search_acquisition is result.attempts[-1].acquisition
    for attempt in result.attempts[:-1]:
        assert attempt.diagnostic_feedback is not None
        assert attempt.diagnostic_policy_successor is not None
        assert attempt.diagnostic_search_memory_snapshot is not None
        assert attempt.diagnostic_successor_context is not None
        assert (
            attempt.diagnostic_successor_context["round_index"]
            == context.round_index
        )
    successor_memory = result.interpretation.successor_context.scientific_memory
    global_memory = successor_memory["global_memory"]
    resource_observations = global_memory["resource_memory"]["observations"]
    assert len(resource_observations) == failure_count
    assert all(
        observation["resource_or_search_only"] is True
        for observation in resource_observations
    )
    assert len(global_memory["search_observations"]) >= failure_count + 1
    assert len(
        {
            sha256_digest(role_memory)
            for role_memory in successor_memory["by_role"].values()
        }
    ) >= 2


def test_shared_infrastructure_failure_holds_without_draining_pool() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:attempt-shared-stop")
    context = _context(profile)
    calls: list[dict[str, Any]] = []
    base_runner = _runner(calls, status="RESOURCE_CENSORED")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        result = dict(base_runner(recipe, binding))
        result["failure_scope"] = "SHARED_INFRASTRUCTURE"
        return result

    result = run_research_round(**_round_kwargs(profile, context, runner=runner))

    assert len(calls) == 1
    assert len(result.attempts) == 1
    assert result.attempts[0].failure_scope == "SHARED_INFRASTRUCTURE"
    assert result.metric_bearing_attempt_index is None
    assert result.interpretation is None
    assert result.incomplete_reason == "ROUND_ATTEMPT_SHARED_INFRASTRUCTURE_STOP"


def test_scheduler_requires_explicit_frozen_attempt_budget() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:attempt-explicit-cap")
    context = _context(profile)
    calls: list[dict[str, Any]] = []
    failure_runner = _runner(calls, status="RESOURCE_CENSORED")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        return failure_runner(recipe, binding)

    kwargs = _round_kwargs(profile, context, runner=runner)
    kwargs.pop("max_attempts_per_round")
    with pytest.raises(ValueError, match="requires an explicit frozen"):
        run_research_round(**kwargs)
    assert calls == []


def test_scheduler_consumes_complete_portfolio_and_reranks_after_failure() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:portfolio-integration")
    context = _context(profile)
    proposals = _fixed_proposals(profile)
    portfolio: list[PortfolioCandidateV2] = []
    expected_order: list[str] = []
    for index, proposal in enumerate(proposals.values(), start=1):
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
                parent_id=None,
                valid_seal_probability=0.9,
                family_delta=0.0,
                parent_delta=0.0,
                information_value=1.0,
                predicted_gpu_seconds=float(index),
                age_rounds=0,
                repeat_count=0,
                lineage_risk=0.0,
                compute_pattern=f"fixture-pattern-{index}",
                resource_admission_state=ResourceAdmissionStateV2.ACTIVE,
                frontier_gain=0.5,
            )
        )
        expected_order.append(proposal.candidate_id)

    calls: list[dict[str, Any]] = []
    failure_runner = _runner(calls, status="RESOURCE_CENSORED")
    success_runner = _runner(calls, status="SUCCESS")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        return (
            failure_runner(recipe, binding)
            if not calls
            else success_runner(recipe, binding)
        )

    kwargs = _round_kwargs(profile, context, runner=runner)
    kwargs["portfolio_candidates"] = tuple(portfolio)
    result = run_research_round(**kwargs)

    assert [item["binding"].proposal.candidate_id for item in calls] == expected_order[:2]
    assert result.metric_bearing_attempt_index == 1
    assert result.attempts[0].candidate_id != result.attempts[1].candidate_id
    assert result.prepared is not None
    assert result.prepared.portfolio_candidates == tuple(portfolio)


def test_prepared_checkpoint_recovers_crash_before_first_runner_and_rejects_drift(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:prepared-crash")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    from recclaw_core.research_line.bootstrap import bootstrap_search_pool

    initial = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
    )
    initial = replace(
        initial,
        resource_profile_by_capability={
            "capability:resource-profile-fixture": {"predicted_gpu_seconds": 12.0}
        },
    )
    producer_calls: list[str] = []

    class CountedProducer:
        def __init__(self) -> None:
            self.call_traces: list[dict[str, Any]] = []

        def __call__(self, role: str, _view: dict[str, Any]) -> Any:
            producer_calls.append(role)
            self.call_traces.append({"role": role})
            return _fixed_proposals(profile)[role]

    runner_calls: list[dict[str, Any]] = []
    runner = _runner(runner_calls, status="SUCCESS")

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        return CampaignRoundInputs(
            producer_bindings=_bindings(state.context, state.active_profile),
            resolver_environment=_environment(state.active_profile),
            budget_snapshot={"experiment_opportunities": 1},
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed="54304",
            next_discriminative_test="resume after prepared checkpoint crash",
            attempt_scheduler=True,
            max_attempts_per_round=4,
        )

    root = tmp_path / "prepared-crash"
    producer = CountedProducer()
    campaign = ResearchCampaign(
        root=root,
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    persist_prepared = campaign._persist_prepared_round

    def crash_after_prepared(**kwargs: Any) -> Any:
        value = persist_prepared(**kwargs)
        raise RuntimeError("simulated crash after prepared callback")

    campaign._persist_prepared_round = crash_after_prepared  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="after prepared callback"):
        campaign.run_round()

    prepared_path = root / "ROUND_01_PREPARED_CHECKPOINT.pkl"
    manifest_path = root / "ROUND_01_ATTEMPT_MANIFEST.json"
    assert prepared_path.is_file()
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["attempts"] == []
    assert runner_calls == []
    assert len(producer_calls) == 4
    assert set(producer_calls) == set(_fixed_proposals(profile))
    assert len(producer.call_traces) == 4

    resumed = ResearchCampaign.resume(
        root=root,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    opportunity_ref = json.loads(
        (root / "ROUND_01_STARTED.json").read_text(encoding="utf-8")
    )["opportunity_ref"]
    common = {
        "round_index": 1,
        "opportunity_ref": opportunity_ref,
        "state_digest": resumed.state.digest,
        "context_digest": resumed.state.context.digest,
        "profile_ref": resumed.state.active_profile.profile_ref,
        "profile_digest": resumed.state.active_profile.profile_digest,
        "attempt_budget": 4,
        "budget_snapshot": {"experiment_opportunities": 1},
    }
    for field_name, bad_value in (
        ("opportunity_ref", "research-opportunity:other"),
        ("state_digest", "0" * 64),
        ("context_digest", "1" * 64),
        ("profile_ref", "profile:other"),
        ("profile_digest", "2" * 64),
        ("attempt_budget", 3),
        ("budget_snapshot", {"experiment_opportunities": 2}),
    ):
        bad = dict(common)
        bad[field_name] = bad_value
        with pytest.raises(CampaignError):
            resumed._load_prepared_round(**bad)

    record = resumed.run_round()
    assert record.status == "TYPED_EPISODE"
    assert len(producer_calls) == 4
    assert len(producer.call_traces) == 4
    assert len(runner_calls) == 1
    assert record.result.has_metric_bearing_attempt
    assert record.result.metric_bearing_attempt_index == 0
    assert len(record.result.provider_traces) == 4
    assert resumed.state.next_round_index == 2
    assert len(list(root.glob("ROUND_01_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))) == 1


def test_sealed_incomplete_round_resumes_remaining_pool_without_replaying_attempt(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:attempt-sealed-resume")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    from recclaw_core.research_line.bootstrap import bootstrap_search_pool

    initial = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
    )
    initial = replace(
        initial,
        resource_profile_by_capability={
            "capability:resource-profile-fixture": {"predicted_gpu_seconds": 12.0}
        },
    )
    calls: list[dict[str, Any]] = []
    first_failure = _runner(calls, status="RESOURCE_CENSORED")
    success = _runner(calls, status="SUCCESS")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        if len(calls) == 0:
            value = dict(first_failure(recipe, binding))
            value["failure_scope"] = "SHARED_INFRASTRUCTURE"
            return value
        return success(recipe, binding)

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        return CampaignRoundInputs(
            producer_bindings=_bindings(state.context, state.active_profile),
            resolver_environment=_environment(state.active_profile),
            budget_snapshot={"experiment_opportunities": 1},
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed="54304",
            next_discriminative_test="continue after the sealed shared failure",
            attempt_scheduler=True,
            max_attempts_per_round=4,
        )

    producer_calls: list[str] = []

    class CountedProducer:
        def __init__(self) -> None:
            self.call_traces: list[dict[str, Any]] = []

        def __call__(self, role: str, _view: dict[str, Any]) -> Any:
            producer_calls.append(role)
            self.call_traces.append({"role": role})
            return _fixed_proposals(profile)[role]

    producer = CountedProducer()

    root = tmp_path / "sealed-attempt-resume"
    campaign = ResearchCampaign(
        root=root,
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    incomplete = campaign.run_round()
    assert incomplete.status == "INCOMPLETE"
    assert incomplete.result.metric_bearing_attempt_index is None
    assert campaign.state.next_round_index == 1
    assert campaign.state.resource_profile_by_capability == (
        initial.resource_profile_by_capability
    )
    assert len(calls) == 1
    assert (root / "ROUND_01_CHECKPOINT.pkl").is_file()

    resumed = ResearchCampaign.resume(
        root=root,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
    )
    record = resumed.run_round()

    assert record.status == "TYPED_EPISODE"
    assert len(calls) == 2
    assert len(producer_calls) == 4
    assert set(producer_calls) == set(_fixed_proposals(profile))
    assert len(producer.call_traces) == 4
    assert record.result.provider_traces == incomplete.result.provider_traces
    assert len(record.result.provider_traces) == 4
    physical_candidate_ids = [
        item["binding"].proposal.candidate_id for item in calls
    ]
    assert len(set(physical_candidate_ids)) == 2
    assert [item.attempt_index for item in record.result.attempts] == [0, 1]
    assert len({item.candidate_id for item in record.result.attempts}) == 2
    assert record.result.metric_bearing_attempt_index == 1
    assert resumed.state.next_round_index == 2
    assert resumed.state.resource_profile_by_capability == (
        initial.resource_profile_by_capability
    )
    assert len(list(root.glob("ROUND_01_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))) == 2


def test_campaign_resume_replays_manifested_failure_without_physical_replay(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:attempt-recovery")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    from recclaw_core.research_line.bootstrap import bootstrap_search_pool

    initial = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=bootstrap_search_pool(context, profile, policy),
    )
    calls: list[dict[str, Any]] = []
    first_failure = _runner(calls, status="RESOURCE_CENSORED")
    success = _runner(calls, status="SUCCESS")

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        return CampaignRoundInputs(
            producer_bindings=_bindings(state.context, state.active_profile),
            resolver_environment=_environment(state.active_profile),
            budget_snapshot={"experiment_opportunities": 1},
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed="54304",
            next_discriminative_test="resume the same round after a sealed failure",
            attempt_scheduler=True,
            max_attempts_per_round=4,
        )

    def no_producer(_role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError("no new Producer proposal is needed for recovery")

    def crash_after_first(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        if len(calls) == 0:
            return first_failure(recipe, binding)
        raise RuntimeError("simulated crash before second observation")

    campaign_root = tmp_path / "attempt-recovery"
    campaign = ResearchCampaign(
        root=campaign_root,
        state=initial,
        producer=no_producer,
        runner=crash_after_first,
        round_inputs=inputs,
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        campaign.run_round()

    manifest_path = campaign_root / "ROUND_01_ATTEMPT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["attempts"]) == 1
    assert len(list(campaign_root.glob("ROUND_01_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))) == 1

    resumed = ResearchCampaign.resume(
        root=campaign_root,
        producer=no_producer,
        runner=success,
        round_inputs=inputs,
    )
    record = resumed.run_round()

    assert record.status == "TYPED_EPISODE"
    assert len(calls) == 2
    assert [item.attempt_index for item in record.result.attempts] == [0, 1]
    assert record.result.metric_bearing_attempt_index == 1
    assert resumed.state.next_round_index == 2
    sealed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(sealed_manifest["attempts"]) == 2
    assert len(list(campaign_root.glob("ROUND_01_ATTEMPT_*_PHYSICAL_OBSERVATION.json"))) == 2
