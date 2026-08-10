from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
import pickle
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
    bind_search_candidate,
)
from recclaw_core.research_line.portfolio import (
    PortfolioCandidateV2,
    ResourceAdmissionStateV2,
)
from recclaw_core.research_line.interfaces import (
    ResearchTaskOperationV2,
    ResearchTaskQueueV2,
    ResearchTaskRecordV2,
)
from recclaw_core.research_line.profile_source import (
    ResearchProfileSourceError,
    ResearchProfileSourceV1,
)
from recclaw_core.research_line.runtime import (
    RoundCandidateHandoffV1,
    run_research_round,
)
import recclaw_core.research_line.runtime as runtime_module
from recclaw_core.research_line.fresh_runner import make_fresh_runner
from recclaw_core.research_line.standalone import (
    StandaloneCampaignError,
    compose_standalone_campaign,
    load_research_profile_source,
)


_TEST_ROOT = Path(__file__).resolve().parent
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))
_HELIX_TEST_ROOT = _TEST_ROOT.parent / "experiments" / "helix_abc_v1"
if str(_HELIX_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_HELIX_TEST_ROOT))

from test_fresh_runner import _fake_launch, _qualified_custom  # noqa: E402
from test_open_spec_search_binding import (  # noqa: E402
    _active_profile as _qualified_active_profile,
    _qualified_candidate,
    _spec as _qualified_spec,
)
from test_standalone_campaign import _config as _standalone_config  # noqa: E402
from test_standalone_campaign import _forbidden_boundary  # noqa: E402
from test_runtime import (  # noqa: E402
    _bindings,
    _context,
    _environment,
    _fixed_proposals,
    _incumbent,
    _router,
    _runner,
)


def _portfolio(binding: Any, index: int) -> PortfolioCandidateV2:
    return PortfolioCandidateV2(
        candidate_id=binding.proposal.candidate_id,
        semantic_digest=binding.mechanism_semantics_digest,
        family_id=binding.proposal.mechanism_axis,
        parent_id=None,
        valid_seal_probability=0.9,
        family_delta=0.0,
        parent_delta=0.0,
        information_value=1.0,
        predicted_gpu_seconds=float(index + 1),
        age_rounds=0,
        repeat_count=0,
        lineage_risk=0.0,
        compute_pattern=f"handoff-pattern-{index}",
        resource_admission_state=ResourceAdmissionStateV2.RESOURCE_ADMITTED,
        frontier_gain=0.5,
    )


def _resource_profile(binding: Any) -> dict[str, Any]:
    return {
        "schema": "test-resource-profile.v1",
        "candidate_ref": binding.capability_ref,
        "candidate_deadline_seconds": 31,
        "effect_fields_consumed": [],
        "outcome_fields_consumed": [],
        "held_out_reads": 0,
        "mechanism_effect_update_allowed": False,
    }


def _e_resource_profile(
    binding: Any,
    *,
    package_digest: str,
    source_digest: str,
    semantic_digest: str,
    pattern: str,
) -> dict[str, Any]:
    prediction = {
        "identity": {
            "candidate_id": binding.proposal.candidate_id,
            "candidate_ref": binding.capability_ref,
            "candidate_package_digest": package_digest,
            "candidate_source_sha256": source_digest,
            "semantic_digest": semantic_digest,
        },
        "model": "TEST_IDENTITY_BOUND_RESOURCE_EVIDENCE_V1",
        "completion_probability": 0.8,
        "prediction_interval_seconds": [8.0, 12.0],
        "estimated_total_wall_time_seconds": 10.0,
        "peak_memory_prediction_mib": 1000.0,
        "predicted_gpu_worker_seconds": 9.0,
        "compute_pattern": pattern,
    }
    return {
        "schema": "test-resource-profile.v2",
        "profile_digest": sha256_digest(
            {"candidate": binding.proposal.candidate_id, "pattern": pattern}
        ),
        "candidate_ref": binding.capability_ref,
        "candidate_package_digest": package_digest,
        "candidate_source_sha256": source_digest,
        "completion_probability": 0.8,
        "prediction_interval_seconds": [8.0, 12.0],
        "full_run_budget_after_probes_seconds": 100.0,
        "status": "RESOURCE_ADMITTED",
        "effect_fields_consumed": [],
        "outcome_fields_consumed": [],
        "held_out_reads": 0,
        "mechanism_effect_update_allowed": False,
        "probe_process": {
            "process_isolated": True,
            "start_method": "spawn",
            "status": "RESULT",
            "exit_code": 0,
        },
        "schedule": [{"arm": binding.capability_ref, "probability": 0.8}],
        "prediction": prediction,
    }


def _profile_source(
    context: Any,
    profile: Any,
    *,
    mandate: bool = True,
    pattern: str = "source-pattern-shared",
    task_candidate_id: str | None = None,
) -> ResearchProfileSourceV1:
    bindings = _source_bindings(context, profile)
    records: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        candidate_id = binding.proposal.candidate_id
        package_digest = sha256_digest({"package": candidate_id})
        candidate_source = sha256_digest({"source": candidate_id})
        record: dict[str, Any] = {
            "candidate_identity": {
                "candidate_id": candidate_id,
                "semantic_digest": binding.mechanism_semantics_digest,
                "family_id": binding.proposal.mechanism_axis,
                "parent_id": getattr(binding.proposal, "parent_candidate_id", None),
                "compute_pattern": pattern,
                "resource_candidate_ref": binding.capability_ref,
                "candidate_package_digest": package_digest,
                "candidate_source_sha256": candidate_source,
            },
            "calibration_prior": {
                "alpha": 2.0,
                "beta": 3.0,
                "frozen": True,
                "source_ref": "test-calibration:v1",
                "source_digest": sha256_digest({"calibration": candidate_id}),
            },
            "resource_profile": _e_resource_profile(
                binding,
                package_digest=package_digest,
                source_digest=candidate_source,
                semantic_digest=binding.mechanism_semantics_digest,
                pattern=pattern,
            ),
        }
        if mandate and candidate_id != task_candidate_id:
            record["exploration_mandate"] = {
                "kind": "EXPLORATION",
                "frozen": True,
                "source_ref": "test-exploration:v1",
                "source_digest": sha256_digest({"mandate": candidate_id}),
                "created_round": context.round_index,
                "priority": 0.61,
                "producer_role": binding.proposal.producer_role,
                "exploration_floor_authorized": True,
            }
        records[candidate_id] = record
    return ResearchProfileSourceV1.from_records(
        source_ref="test-profile-source:v1",
        records=records,
    )


def _pre_round_profile_source(
    profile: Any,
    *,
    pattern: str = "pre-round-pattern-shared",
) -> ResearchProfileSourceV1:
    """Build only stable role/capability policies; no current candidate IDs."""

    policies: list[dict[str, Any]] = []
    for binding in _source_bindings(None, profile):
        role = binding.proposal.producer_role
        stable_ref = binding.capability_ref
        package_digest = sha256_digest({"package-policy": stable_ref})
        source_digest = sha256_digest({"source-policy": stable_ref})
        resource = _e_resource_profile(
            binding,
            package_digest=package_digest,
            source_digest=source_digest,
            semantic_digest=binding.mechanism_semantics_digest,
            pattern=pattern,
        )
        resource["profile_digest"] = sha256_digest(
            {"policy-resource": stable_ref, "pattern": pattern}
        )
        resource["prediction"]["identity"].pop("candidate_id")
        policies.append(
            {
                "producer_role": role,
                "capability_ref": stable_ref,
                "capability_digest": binding.capability_digest,
                "family_id": binding.proposal.mechanism_axis,
                "compute_pattern": pattern,
                "calibration_prior": {
                    "alpha": 2.0,
                    "beta": 3.0,
                    "frozen": True,
                    "source_ref": "test-calibration:pre-round",
                    "source_digest": sha256_digest(
                        {"calibration-policy": stable_ref}
                    ),
                },
                "exploration_mandate": {
                    "kind": "EXPLORATION",
                    "frozen": True,
                    "source_ref": "test-exploration:pre-round",
                    "source_digest": sha256_digest(
                        {"mandate-policy": stable_ref}
                    ),
                    "created_round": 1,
                    "priority": 0.61,
                    "producer_role": role,
                    "exploration_floor_authorized": True,
                },
                "qualified_execution": None,
                "candidate_root_path": None,
                "resource_profile": resource,
            }
        )
    return ResearchProfileSourceV1.from_pre_round_policies(
        source_ref="test-profile-source:pre-round-v1",
        policies=policies,
    )


def _role_only_pre_round_profile_source(profile: Any) -> ResearchProfileSourceV1:
    """Build reusable role policies with no future capability or pattern data."""

    roles = tuple(
        dict.fromkeys(
            binding.proposal.producer_role
            for binding in _source_bindings(None, profile)
        )
    )
    policies = [
        {
            "producer_role": role,
            "calibration_prior": {
                "alpha": 2.0,
                "beta": 3.0,
                "frozen": True,
                "source_ref": "test-calibration:role-only",
                "source_digest": sha256_digest(
                    {"calibration-role": role}
                ),
            },
            "exploration_mandate": {
                "kind": "EXPLORATION",
                "frozen": True,
                "source_ref": "test-exploration:role-only",
                "source_digest": sha256_digest({"mandate-role": role}),
                "created_round": 1,
                "priority": 0.61,
                "producer_role": role,
                "exploration_floor_authorized": True,
            },
        }
        for role in roles
    ]
    return ResearchProfileSourceV1.from_pre_round_policies(
        source_ref="test-profile-source:role-only-v1",
        policies=policies,
    )


def _role_only_resource_map(
    bindings: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    resources: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        capability_ref = binding.capability_ref
        pattern = f"resource-pattern-{capability_ref}"
        package_digest = sha256_digest({"package-map": capability_ref})
        source_digest = sha256_digest({"source-map": capability_ref})
        resource = _e_resource_profile(
            binding,
            package_digest=package_digest,
            source_digest=source_digest,
            semantic_digest=binding.mechanism_semantics_digest,
            pattern=pattern,
        )
        resource["profile_digest"] = sha256_digest(
            {"durable-resource": capability_ref, "pattern": pattern}
        )
        resource["prediction"]["identity"].pop("candidate_id")
        resources[capability_ref] = resource
    return resources


def _dynamic_proposals(profile: Any) -> dict[str, Any]:
    return {
        role: replace(
            proposal,
            candidate_id=f"cand-live-{role.replace('_', '-')}",
            mechanism_program=proposal.to_dict()["mechanism_program"],
        )
        for role, proposal in _fixed_proposals(profile).items()
    }


def _source_bindings(context: Any, profile: Any) -> tuple[Any, ...]:
    del context
    proposals = _fixed_proposals(profile)
    bindings = []
    for proposal in proposals.values():
        entry = next(
            item
            for item in profile.entries
            if item.semantic_identity_ref
            == f"bl-icf-mechanism:{proposal.mechanism_id}"
        )
        bindings.append(
            bind_search_candidate(
                profile=profile,
                proposal=proposal,
                capability_ref=entry.capability_ref,
            )
        )
    return tuple(bindings)


def _task_context(context: Any, binding: Any, *, priority: float = 0.93) -> Any:
    task = ResearchTaskRecordV2(
        task_id="task-exact-profile-source",
        operation=ResearchTaskOperationV2.NEW_SEED,
        candidate_id=binding.proposal.candidate_id,
        candidate_semantic_digest=binding.mechanism_semantics_digest,
        mechanism_program_digest=sha256_digest({"mechanism": binding.proposal.mechanism_id}),
        parent_candidate_id=getattr(binding.proposal, "parent_candidate_id", None),
        comparator_identity="test-comparator",
        protocol_digest=context.protocol_digest,
        required_seed_or_control="54304",
        priority=priority,
        created_round=context.round_index,
        producer_role=binding.proposal.producer_role,
    )
    queue = ResearchTaskQueueV2((task,))
    return replace(
        context,
        scientific_memory={
            **context.scientific_memory,
            "global_memory": {"task_queue": queue.to_dict()},
        },
    )


def _handoff_factory(calls: list[tuple[str, ...]]):
    def factory(*, search_bindings: Any, **_kwargs: Any) -> tuple[RoundCandidateHandoffV1, ...]:
        bindings = tuple(search_bindings)
        calls.append(tuple(binding.proposal.candidate_id for binding in bindings))
        return tuple(
            RoundCandidateHandoffV1(
                candidate_id=binding.proposal.candidate_id,
                binding_digest=binding.digest,
                portfolio_candidate=_portfolio(binding, index),
                qualified_execution=None,
                candidate_root_path=None,
                resource_profile=_resource_profile(binding),
            )
            for index, binding in enumerate(bindings)
        )

    return factory


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
        "next_discriminative_test": "reuse the prepared candidate handoff",
        "attempt_scheduler": True,
        "max_attempts_per_round": 4,
    }


def test_handoff_is_before_prepared_and_reused_for_failover() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:dynamic-handoff-order")
    context = _context(profile)
    events: list[tuple[str, str | None]] = []
    factory_calls: list[tuple[str, ...]] = []
    prepared_values: list[Any] = []
    runner_calls: list[dict[str, Any]] = []
    failing = _runner(runner_calls, status="RESOURCE_CENSORED")
    succeeding = _runner(runner_calls, status="SUCCESS")

    def runner(recipe: dict[str, Any], binding: Any) -> dict[str, Any]:
        events.append(("runner", binding.proposal.candidate_id))
        return (
            failing(recipe, binding)
            if len(runner_calls) == 0
            else succeeding(recipe, binding)
        )

    def factory(*, search_bindings: Any, **kwargs: Any):
        events.append(("factory", None))
        return _handoff_factory(factory_calls)(
            search_bindings=search_bindings,
            **kwargs,
        )

    result = run_research_round(
        **_round_kwargs(profile, context, runner=runner),
        candidate_handoff_factory=factory,
        on_prepared=lambda prepared: (
            prepared_values.append(prepared),
            events.append(("prepared", None)),
        ),
    )

    assert [item[0] for item in events[:4]] == [
        "factory",
        "prepared",
        "runner",
        "runner",
    ]
    assert len(factory_calls) == 1
    assert len(runner_calls) == 2
    assert result.prepared is not None
    assert len(prepared_values) == 1
    assert tuple(
        item.candidate_id for item in result.prepared.candidate_handoffs
    ) == factory_calls[0]
    assert len({item.candidate_id for item in result.attempts}) == 2
    assert result.prepared.to_dict()["candidate_handoffs"][0][
        "resource_profile"
    ]["candidate_ref"]


def test_prepared_resume_does_not_replay_handoff_factory() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:dynamic-handoff-resume")
    context = _context(profile)
    factory_calls: list[tuple[str, ...]] = []
    prepared_values: list[Any] = []
    first_kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("the zero-budget preparation must not run a runner")
        ),
    )
    first_kwargs["max_attempts_per_round"] = 0
    first_kwargs.update(
        candidate_handoff_factory=_handoff_factory(factory_calls),
        on_prepared=prepared_values.append,
    )
    first = run_research_round(**first_kwargs)
    assert first.prepared is not None
    assert len(factory_calls) == 1
    persisted_prepared = pickle.loads(pickle.dumps(first.prepared))
    assert persisted_prepared.candidate_handoffs == first.prepared.candidate_handoffs

    def forbidden_factory(**_kwargs: Any) -> Any:
        raise AssertionError("prepared resume must not replay the handoff factory")

    resumed_kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prepared zero-budget resume must not run a runner")
        ),
    )
    resumed_kwargs["max_attempts_per_round"] = 0
    resumed_kwargs.update(
        prepared_round=persisted_prepared,
        candidate_handoff_factory=forbidden_factory,
    )
    resumed = run_research_round(**resumed_kwargs)
    assert resumed.prepared is persisted_prepared
    assert resumed.attempts == ()
    assert len(factory_calls) == 1


@pytest.mark.parametrize("failure", ("missing", "binding"))
def test_handoff_coverage_and_identity_fail_before_runner(failure: str) -> None:
    profile = adapt_current_search_profile(campaign_id=f"campaign:handoff-fail-{failure}")
    context = _context(profile)
    runner_calls: list[dict[str, Any]] = []

    def bad_factory(*, search_bindings: Any, **kwargs: Any):
        valid = _handoff_factory([])(search_bindings=search_bindings, **kwargs)
        if failure == "missing":
            return valid[:-1]
        return (
            *valid[:-1],
            replace(valid[-1], binding_digest="0" * 64),
        )

    with pytest.raises(ValueError, match="coverage|binding digest"):
        run_research_round(
            **_round_kwargs(profile, context, runner=_runner(runner_calls)),
            candidate_handoff_factory=bad_factory,
        )
    assert runner_calls == []


def test_fresh_runner_uses_per_call_root_and_resource_profile(
    tmp_path: Path,
) -> None:
    recipe, binding = _qualified_custom()
    candidate_root = tmp_path / "dynamic-root"
    (candidate_root / "recclaw_ext").mkdir(parents=True)
    resource_profile = {
        "candidate_ref": binding.capability_ref,
        "candidate_deadline_seconds": 17,
        "profile_digest": sha256_digest({"dynamic": binding.capability_ref}),
        "effect_fields_consumed": [],
        "outcome_fields_consumed": [],
        "held_out_reads": 0,
        "mechanism_effect_update_allowed": False,
    }
    dynamic_recipe = {
        **recipe,
        "candidate_root_path": str(candidate_root),
        "resource_prediction": resource_profile,
    }
    launch_calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="dynamic-handoff-run",
        seed=2026,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        launch=_fake_launch(launch_calls),
    )

    result = runner(dynamic_recipe, binding)

    assert launch_calls[0]["candidate_root"] == candidate_root.resolve()
    assert launch_calls[0]["resource_prediction"] == resource_profile
    assert result["experiment_binding"]["candidate_root_path"] == str(
        candidate_root.resolve()
    )


def test_standalone_carries_real_handoff_factory_without_static_profiles(
    tmp_path: Path,
) -> None:
    def factory(**_kwargs: Any) -> tuple[RoundCandidateHandoffV1, ...]:
        return ()

    config = _standalone_config(tmp_path / "standalone-dynamic")
    composition = compose_standalone_campaign(
        config,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
        candidate_handoff_factory=factory,
    )

    assert composition.config.candidate_handoff_factory is factory
    assert composition.manifest["portfolio"]["mode"] == "DYNAMIC_HANDOFF_FACTORY"
    assert composition.campaign._round_inputs().candidate_handoff_factory is factory


def test_profile_source_empty_queue_requires_frozen_exploration_mandate() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-empty")
    context = _context(profile)
    source = _profile_source(context, profile)
    prepared_values: list[Any] = []
    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("zero-budget profile preparation must not run a runner")
        ),
    )
    kwargs["max_attempts_per_round"] = 0

    result = run_research_round(
        **kwargs,
        research_profile_source=source,
        on_prepared=prepared_values.append,
    )

    assert result.prepared is not None
    assert prepared_values == [result.prepared]
    assert len(result.prepared.candidate_handoffs) == 4
    for handoff in result.prepared.candidate_handoffs:
        assert handoff.portfolio_profile is not None
        assert handoff.portfolio_profile.evidence["selection_kind"] == (
            "EXPLORATION_MANDATE"
        )
        assert handoff.portfolio_profile.evidence["task"] is None
        assert handoff.source_digest == source.source_digest
        assert handoff.lineage_identity_digest


def test_pre_round_source_materializes_unseen_candidate_ids_and_resumes_without_recall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:pre-round-source")
    context = _context(profile)
    source = _pre_round_profile_source(profile)
    source_path = tmp_path / "pre-round-profile-source.json"
    source_path.write_text(json.dumps(source.to_dict()), encoding="utf-8")
    loaded_source = load_research_profile_source(source_path)
    source_payload = json.dumps(loaded_source.to_dict(), sort_keys=True)
    dynamic = _dynamic_proposals(profile)
    dynamic_ids = {proposal.candidate_id for proposal in dynamic.values()}
    assert dynamic_ids.isdisjoint(
        set(
            item.candidate_id
            for item in _fixed_proposals(profile).values()
        )
    )
    assert not any(candidate_id in source_payload for candidate_id in dynamic_ids)

    source_calls: list[str] = []
    original_build_profiles = ResearchProfileSourceV1.build_profiles

    def counted_build_profiles(source_instance: Any, **kwargs: Any) -> Any:
        source_calls.append(source_instance.source_digest)
        return original_build_profiles(source_instance, **kwargs)

    monkeypatch.setattr(
        ResearchProfileSourceV1,
        "build_profiles",
        counted_build_profiles,
    )
    producer_calls: list[str] = []

    def producer(role: str, _view: Any) -> Any:
        producer_calls.append(role)
        return dynamic[role]

    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("zero-budget pre-round preparation must not run a runner")
        ),
    )
    kwargs["producer"] = producer
    kwargs["max_attempts_per_round"] = 0
    first = run_research_round(
        **kwargs,
        research_profile_source=loaded_source,
    )

    assert first.prepared is not None
    assert len(producer_calls) == 4
    assert source_calls == [loaded_source.source_digest]
    assert {
        item.candidate_id for item in first.prepared.candidate_handoffs
    } == dynamic_ids
    assert all(
        item.portfolio_profile is not None
        and item.portfolio_profile.candidate.candidate_id in dynamic_ids
        for item in first.prepared.candidate_handoffs
    )
    assert (
        first.prepared.search_acquisition is not None
        and first.prepared.search_acquisition.selected_binding is not None
        and first.prepared.search_acquisition.selected_binding.proposal.candidate_id
        in dynamic_ids
    )

    persisted = pickle.loads(pickle.dumps(first.prepared))
    resumed_kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prepared resume must not run a runner")
        ),
    )
    resumed_kwargs["producer"] = lambda *_args: (_ for _ in ()).throw(
        AssertionError("prepared resume must not call a Producer")
    )
    resumed_kwargs["max_attempts_per_round"] = 0
    resumed = run_research_round(
        **resumed_kwargs,
        prepared_round=persisted,
        research_profile_source=None,
    )
    assert resumed.prepared is persisted
    assert source_calls == [loaded_source.source_digest]
    assert resumed.prepared.candidate_handoffs == persisted.candidate_handoffs


def test_role_only_policy_rebinds_same_role_candidates_to_distinct_resource_patterns() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:role-only-patterns")
    context = _context(profile)
    source = _role_only_pre_round_profile_source(profile)
    assert all(
        "compute_pattern" not in policy
        and "capability_ref" not in policy
        and "family_id" not in policy
        for policy in source.pre_round_policies.values()
    )
    assert "candidate_id" not in json.dumps(source.to_dict())
    base = _fixed_proposals(profile)
    extra = replace(
        base["lineage_refiner"],
        candidate_id="cand-role-only-extra",
        producer_role="mechanism_composer",
        mechanism_program=base["lineage_refiner"].to_dict()["mechanism_program"],
    )
    bindings = _source_bindings(None, profile)
    resource_map = _role_only_resource_map(bindings)
    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("role-only preparation must not run a runner")
        ),
    )
    kwargs["carryover_proposals"] = (extra,)
    kwargs["max_attempts_per_round"] = 0
    capability_by_role = {
        binding.proposal.producer_role: binding.capability_ref
        for binding in bindings
    }
    lineage_capability_ref = next(
        binding.capability_ref
        for binding in bindings
        if binding.proposal.producer_role == "lineage_refiner"
    )

    result = run_research_round(
        **kwargs,
        research_profile_source=source,
        qualified_execution_by_capability={},
        candidate_root_by_capability={},
        resource_profile_by_capability=resource_map,
    )

    assert result.prepared is not None
    mechanism_composer_handoffs = tuple(
        handoff
        for handoff in result.prepared.candidate_handoffs
        if handoff.portfolio_candidate is not None
        and handoff.portfolio_candidate.candidate_id
        in {
            base["mechanism_composer"].candidate_id,
            extra.candidate_id,
        }
    )
    assert len(mechanism_composer_handoffs) == 2
    assert {
        handoff.portfolio_candidate.compute_pattern
        for handoff in mechanism_composer_handoffs
    } == {
        f"resource-pattern-{capability_by_role['mechanism_composer']}",
        f"resource-pattern-{lineage_capability_ref}",
    }
    assert all(
        handoff.portfolio_profile is not None
        and handoff.portfolio_profile.candidate.compute_pattern
        == handoff.portfolio_candidate.compute_pattern
        for handoff in mechanism_composer_handoffs
    )


def test_static_role_policy_materializes_unseen_qualified_profile_expansion(
    tmp_path: Path,
) -> None:
    base_profile = adapt_current_search_profile(
        campaign_id="campaign:qualified-expansion-base"
    )
    source = _role_only_pre_round_profile_source(base_profile)
    source_identity = source.identity
    base_context = _context(base_profile, round_index=1)
    base_kwargs = _round_kwargs(
        base_profile,
        base_context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("round-one preparation must not run a runner")
        ),
    )
    base_kwargs["max_attempts_per_round"] = 0
    first = run_research_round(
        **base_kwargs,
        research_profile_source=source,
        qualified_execution_by_capability={},
        candidate_root_by_capability={},
        resource_profile_by_capability=_role_only_resource_map(
            _source_bindings(None, base_profile)
        ),
    )
    assert first.prepared is not None

    candidate, capability = _qualified_candidate(_qualified_spec())
    expanded_profile = _qualified_active_profile(
        base_profile,
        capability,
        include_fixed=False,
    )
    expanded_context = _context(expanded_profile, round_index=2)
    expanded_binding = bind_search_candidate(
        profile=expanded_profile,
        proposal=candidate,
        capability_ref=candidate.capability_ref,
    )
    execution_contract = dict(candidate.execution_contract)
    qualified_execution = {
        "capability_family": execution_contract["capability_family"],
        "model": execution_contract["model"],
        "base_model_config": execution_contract["base_model_config"],
        "config": execution_contract["config"],
        "entrypoint_source_sha256": candidate.source_tree_digest,
        "candidate_package_ref": candidate.candidate_package_ref,
        "candidate_package_digest": candidate.candidate_package_digest,
        "candidate_root_ref": candidate.candidate_root_ref,
        "candidate_root_digest": candidate.candidate_root_digest,
        "candidate_source_tree_digest": candidate.source_tree_digest,
    }
    candidate_root = tmp_path / "qualified-expansion-root"
    (candidate_root / "recclaw_ext").mkdir(parents=True)
    resource_profile = _e_resource_profile(
        expanded_binding,
        package_digest=candidate.candidate_package_digest,
        source_digest=candidate.source_tree_digest,
        semantic_digest=expanded_binding.mechanism_semantics_digest,
        pattern="qualified-expansion-pattern",
    )
    source_payload = json.dumps(source.to_dict(), sort_keys=True)
    assert candidate.candidate_id not in source_payload
    assert candidate.capability_ref not in source_payload

    expanded_kwargs = _round_kwargs(
        expanded_profile,
        expanded_context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified expansion preparation must not run a runner")
        ),
    )
    expanded_kwargs["producer"] = lambda role, _view: {
        "producer_role": role
    }
    expanded_kwargs["carryover_proposals"] = ()
    expanded_kwargs["carryover_open_candidates"] = (candidate,)
    expanded_kwargs["max_attempts_per_round"] = 0
    second = run_research_round(
        **expanded_kwargs,
        research_profile_source=source,
        qualified_execution_by_capability={
            candidate.capability_ref: qualified_execution
        },
        candidate_root_by_capability={candidate.capability_ref: candidate_root},
        resource_profile_by_capability={
            candidate.capability_ref: resource_profile
        },
    )

    assert second.prepared is not None
    assert source.identity == source_identity
    assert len(second.prepared.candidate_handoffs) == 1
    handoff = second.prepared.candidate_handoffs[0]
    assert handoff.candidate_id == candidate.candidate_id
    assert handoff.qualified_execution == qualified_execution
    assert handoff.candidate_root_path == str(candidate_root.resolve())
    assert handoff.resource_profile is not None
    assert (
        handoff.portfolio_candidate.compute_pattern
        == "qualified-expansion-pattern"
    )

    missing_resource_kwargs = dict(expanded_kwargs)
    with pytest.raises(ValueError, match="durable capability-keyed resource"):
        run_research_round(
            **missing_resource_kwargs,
            research_profile_source=source,
            qualified_execution_by_capability={
                candidate.capability_ref: qualified_execution
            },
            candidate_root_by_capability={candidate.capability_ref: candidate_root},
            resource_profile_by_capability={},
        )

    drifted_execution = dict(qualified_execution)
    drifted_execution["candidate_package_digest"] = sha256_digest(
        {"drifted": candidate.capability_ref}
    )
    with pytest.raises(
        ValueError,
        match="qualified execution candidate_package_digest drift",
    ):
        run_research_round(
            **expanded_kwargs,
            research_profile_source=source,
            qualified_execution_by_capability={
                candidate.capability_ref: drifted_execution
            },
            candidate_root_by_capability={candidate.capability_ref: candidate_root},
            resource_profile_by_capability={
                candidate.capability_ref: resource_profile
            },
        )

    persisted = pickle.loads(pickle.dumps(second.prepared))
    resumed_kwargs = _round_kwargs(
        expanded_profile,
        expanded_context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("qualified prepared resume must not run a runner")
        ),
    )
    resumed_kwargs["producer"] = lambda *_args: (_ for _ in ()).throw(
        AssertionError("qualified prepared resume must not call a Producer")
    )
    resumed_kwargs["max_attempts_per_round"] = 0
    resumed = run_research_round(
        **resumed_kwargs,
        prepared_round=persisted,
        research_profile_source=None,
        qualified_execution_by_capability={},
        candidate_root_by_capability={},
        resource_profile_by_capability={},
    )
    assert resumed.prepared is persisted
    assert resumed.prepared.candidate_handoffs == persisted.candidate_handoffs


def test_pre_round_source_missing_ambiguous_and_identity_drift_fail_closed() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:pre-round-source-fail")
    context = _context(profile)
    source = _pre_round_profile_source(profile)
    policies = list(source.pre_round_policies.values())

    with pytest.raises(
        ResearchProfileSourceError,
        match="duplicate stable selector|ambiguous",
    ):
        ResearchProfileSourceV1.from_pre_round_policies(
            source_ref="test-profile-source:ambiguous",
            policies=(policies[0], policies[0]),
        )

    missing_source = ResearchProfileSourceV1.from_pre_round_policies(
        source_ref="test-profile-source:missing",
        policies=policies[:-1],
    )
    runner_calls: list[dict[str, Any]] = []
    kwargs = _round_kwargs(
        profile,
        context,
        runner=_runner(runner_calls),
    )
    kwargs["producer"] = lambda role, _view: _dynamic_proposals(profile)[role]
    kwargs["max_attempts_per_round"] = 0
    with pytest.raises(ValueError, match="coverage is missing or ambiguous"):
        run_research_round(
            **kwargs,
            research_profile_source=missing_source,
        )
    assert runner_calls == []

    drifted_policy = dict(policies[0])
    drifted_policy["capability_digest"] = sha256_digest(
        {"drifted-capability": policies[0]["capability_ref"]}
    )
    drifted_source = ResearchProfileSourceV1.from_pre_round_policies(
        source_ref="test-profile-source:identity-drift",
        policies=(drifted_policy, *policies[1:]),
    )
    with pytest.raises(ValueError, match="coverage is missing or ambiguous"):
        run_research_round(
            **kwargs,
            research_profile_source=drifted_source,
        )
    assert runner_calls == []

    bad_resource_policy = dict(policies[0])
    bad_resource = dict(bad_resource_policy["resource_profile"])
    bad_resource["candidate_ref"] = "capability:drift"
    bad_resource_policy["resource_profile"] = bad_resource
    with pytest.raises(ValueError, match="candidate_ref"):
        ResearchProfileSourceV1.from_pre_round_policies(
            source_ref="test-profile-source:resource-drift",
            policies=(bad_resource_policy, *policies[1:]),
        )


def test_profile_source_preserves_exact_task_priority_over_exploration() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-task")
    context = _context(profile)
    first_binding = _source_bindings(context, profile)[0]
    task_context = _task_context(context, first_binding, priority=0.93)
    source = _profile_source(
        task_context,
        profile,
        task_candidate_id=first_binding.proposal.candidate_id,
    )
    kwargs = _round_kwargs(
        profile,
        task_context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("zero-budget task preparation must not run a runner")
        ),
    )
    kwargs["max_attempts_per_round"] = 0

    result = run_research_round(**kwargs, research_profile_source=source)

    selected = next(
        item
        for item in result.prepared.candidate_handoffs
        if item.candidate_id == first_binding.proposal.candidate_id
    )
    assert selected.portfolio_profile is not None
    assert selected.portfolio_profile.evidence["selection_kind"] == "TASK"
    assert selected.portfolio_profile.evidence["task"]["priority"] == 0.93
    assert selected.portfolio_candidate.task_priority == 0.93


def test_profile_source_drops_current_round_outcome_fields_from_prior_projection() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-cutoff")
    context = _context(profile)
    context = replace(
        context,
        scientific_memory={
            "global_memory": {
                "profile_family_history": [
                    {
                        "round_index": context.round_index,
                        "family_id": "architecture",
                        "stable": True,
                        "stable_delta": 0.4,
                        "outcome": "CURRENT_ROUND_RESULT_MUST_NOT_ENTER_PROFILE",
                    }
                ]
            }
        },
    )
    source = _profile_source(context, profile)
    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("zero-budget cutoff test must not run a runner")
        ),
    )
    kwargs["max_attempts_per_round"] = 0

    result = run_research_round(**kwargs, research_profile_source=source)

    assert result.prepared is not None
    for handoff in result.prepared.candidate_handoffs:
        assert "outcome" not in handoff.portfolio_profile.evidence
        assert "CURRENT_ROUND_RESULT_MUST_NOT_ENTER_PROFILE" not in str(
            handoff.portfolio_profile.evidence
        )


def test_profile_source_requires_exact_coverage_before_runner() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-coverage")
    context = _context(profile)
    source = _profile_source(context, profile)
    records = dict(source.records)
    records.pop(next(iter(records)))
    incomplete = ResearchProfileSourceV1.from_records(
        source_ref="test-profile-source:incomplete",
        records=records,
    )
    runner_calls: list[dict[str, Any]] = []

    with pytest.raises(ValueError, match="coverage"):
        run_research_round(
            **_round_kwargs(profile, context, runner=_runner(runner_calls)),
            research_profile_source=incomplete,
        )
    assert runner_calls == []


def test_profile_source_and_lineage_identity_drift_fail_at_handoff_boundary() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-identity")
    context = _context(profile)
    source = _profile_source(context, profile)
    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("identity drift must fail before runner")
        ),
    )
    kwargs["max_attempts_per_round"] = 0
    prepared: list[Any] = []
    result = run_research_round(
        **kwargs,
        research_profile_source=source,
        on_prepared=prepared.append,
    )
    assert result.prepared is not None
    handoff = result.prepared.candidate_handoffs[0]
    with pytest.raises(ValueError, match="profile_digest"):
        replace(handoff, profile_digest="0" * 64)
    with pytest.raises(ValueError, match="lineage_identity_digest"):
        replace(handoff, lineage_identity_digest="0" * 64)


def test_profile_source_nested_mutation_cannot_change_sealed_identity() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-sealed")
    context = _context(profile)
    source = _profile_source(context, profile)
    candidate_id = next(iter(source.records))

    source.records[candidate_id]["resource_profile"]["candidate_ref"] = "drifted"

    with pytest.raises(ValueError, match="mutated after sealing"):
        _ = source.identity


def test_prepared_profile_source_resume_does_not_recall_source() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-resume")
    context = _context(profile)
    source = _profile_source(context, profile)
    first_kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prepared zero-budget round must not run a runner")
        ),
    )
    first_kwargs["max_attempts_per_round"] = 0
    first = run_research_round(
        **first_kwargs,
        research_profile_source=source,
    )
    persisted = pickle.loads(pickle.dumps(first.prepared))

    resumed_kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("prepared zero-budget resume must not run a runner")
        ),
    )
    resumed_kwargs["max_attempts_per_round"] = 0
    resumed = run_research_round(
        **resumed_kwargs,
        prepared_round=persisted,
        # A prepared checkpoint is self-contained; no source is recalled.
        research_profile_source=None,
    )

    assert resumed.prepared is persisted
    assert resumed.prepared.candidate_handoffs == persisted.candidate_handoffs
    assert all(item.portfolio_profile is not None for item in persisted.candidate_handoffs)


def test_correlated_failover_uses_frozen_handoff_compute_pattern(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-correlation")
    context = _context(profile)
    source = _profile_source(context, profile, pattern="shared-compute-pattern")
    observed_failures: list[tuple[str | None, ...]] = []
    original_route = runtime_module.route_frozen_experiment_slate

    def capture_route(**route_kwargs: Any) -> Any:
        observed_failures.append(
            tuple(
                getattr(item, "compute_pattern", None)
                for item in route_kwargs.get("attempt_failures", ())
            )
        )
        return original_route(**route_kwargs)

    monkeypatch.setattr(runtime_module, "route_frozen_experiment_slate", capture_route)
    runner_calls: list[dict[str, Any]] = []
    failure_runner = _runner(runner_calls, status="RESOURCE_CENSORED")

    result = run_research_round(
        **_round_kwargs(profile, context, runner=failure_runner),
        research_profile_source=source,
    )

    assert len(result.attempts) == 2
    assert len(runner_calls) == 2
    assert any(
        failures and all(item == "shared-compute-pattern" for item in failures)
        for failures in observed_failures
    )
    assert result.incomplete_reason == "ROUND_ATTEMPT_NO_ELIGIBLE_REMAINING_BINDING"


def test_standalone_manifest_binds_profile_source_identity_across_resume(
    tmp_path: Path,
) -> None:
    config = _standalone_config(tmp_path / "profile-source-manifest")
    profile = adapt_current_search_profile(campaign_id="campaign:profile-source-manifest")
    source = _profile_source(_context(profile), profile)
    source_path = tmp_path / "profile-source.json"
    source_path.write_text(json.dumps(source.to_dict()), encoding="utf-8")
    loaded_source = load_research_profile_source(source_path)
    assert loaded_source.identity == source.identity
    source = loaded_source
    config = replace(config, research_profile_source=source)
    composition = compose_standalone_campaign(
        config,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
    )
    assert composition.manifest["portfolio"]["mode"] == "PROFILE_SOURCE"
    assert composition.manifest["execution"]["profile_source_identity"] == {
        "schema": source.schema,
        "schema_version": source.schema_version,
        "source_ref": source.source_ref,
        "source_digest": source.source_digest,
    }

    drifted_source = ResearchProfileSourceV1.from_records(
        source_ref="test-profile-source:drifted",
        records=source.records,
    )
    with pytest.raises(StandaloneCampaignError, match="execution inputs|portfolio profile"):
        compose_standalone_campaign(
            replace(config, research_profile_source=drifted_source),
            resume=True,
            provider_call=_forbidden_boundary,
            launch=_forbidden_boundary,
        )


def test_standalone_manifest_binds_pre_round_source_identity_across_resume(
    tmp_path: Path,
) -> None:
    config = _standalone_config(tmp_path / "pre-round-source-manifest")
    profile = adapt_current_search_profile(campaign_id="campaign:pre-round-manifest")
    source = _pre_round_profile_source(profile)
    config = replace(config, research_profile_source=source)
    composition = compose_standalone_campaign(
        config,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
    )
    assert composition.manifest["portfolio"]["mode"] == "PROFILE_SOURCE"
    assert composition.manifest["portfolio"]["profile_source_identity"] == {
        "schema": source.schema,
        "schema_version": source.schema_version,
        "source_ref": source.source_ref,
        "source_digest": source.source_digest,
    }

    drifted_source = ResearchProfileSourceV1.from_pre_round_policies(
        source_ref="test-profile-source:pre-round-drift",
        policies=tuple(source.pre_round_policies.values()),
    )
    with pytest.raises(StandaloneCampaignError, match="execution inputs|portfolio profile"):
        compose_standalone_campaign(
            replace(config, research_profile_source=drifted_source),
            resume=True,
            provider_call=_forbidden_boundary,
            launch=_forbidden_boundary,
        )


def test_legacy_no_portfolio_handoff_and_prepared_digest_shape_is_unchanged() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:legacy-handoff-shape")
    context = _context(profile)
    kwargs = _round_kwargs(
        profile,
        context,
        runner=lambda *_args: (_ for _ in ()).throw(
            AssertionError("legacy zero-budget route must not run a runner")
        ),
    )
    kwargs["max_attempts_per_round"] = 0
    result = run_research_round(**kwargs)
    assert result.prepared is not None
    assert "candidate_handoffs" not in result.prepared.to_dict()
    assert "portfolio_profile" not in result.prepared.to_dict()
