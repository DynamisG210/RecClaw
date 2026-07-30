from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.open_meta import (  # noqa: E402
    EXPERIMENT_FALLBACK_SCOPE_V1,
    IDEA_FALLBACK_SCOPE_V1,
    NEXT_ROUND,
    RESEARCH_STATIC_VNEXT,
    ExperimentBudgetV1,
    OpenMetaContractError,
    OpenMetaReplayDatasetV1,
    PolicyBudgetAllocationV1,
    PolicySupportStatusV1,
    TypedEpisodePolicySummaryV1,
    build_open_meta_replay_record,
    read_open_meta_replay_dataset,
    run_static_experiment_policy,
    run_static_idea_policy,
    schedule_static_policy_activation,
    write_open_meta_replay_dataset,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    AcquisitionDispositionV1,
    AcquisitionStageV1,
    NEXT_FRESH_CAMPAIGN,
)

from open_meta_fixtures import (  # noqa: E402
    current_experiment_policy,
    current_idea_policy,
    digest,
    experiment_policy_input,
    idea_candidates,
    idea_policy_input,
    out_of_support_experiment_input,
    replay_dataset,
)
from scientific_episode_fixtures import (  # noqa: E402
    comparison_identity,
    research_episode,
)


def _keys(value: Any) -> tuple[str, ...]:
    if isinstance(value, dict):
        return tuple(str(key) for key in value) + tuple(
            key for item in value.values() for key in _keys(item)
        )
    if isinstance(value, list):
        return tuple(key for item in value for key in _keys(item))
    return ()


def test_static_policies_are_in_support_distinct_and_explainable() -> None:
    idea = run_static_idea_policy(idea_policy_input())
    experiment = run_static_experiment_policy(experiment_policy_input())

    assert idea.support_status is PolicySupportStatusV1.IN_SUPPORT
    assert experiment.support_status is PolicySupportStatusV1.IN_SUPPORT
    assert idea.policy_mode == experiment.policy_mode == RESEARCH_STATIC_VNEXT
    assert idea.fallback_scope == IDEA_FALLBACK_SCOPE_V1
    assert experiment.fallback_scope == EXPERIMENT_FALLBACK_SCOPE_V1
    assert idea.fallback_reason_codes == (
        "PRE_LEARNING_NO_PROMOTED_IDEA_POLICY",
    )
    assert experiment.fallback_reason_codes == (
        "PRE_LEARNING_NO_PROMOTED_EXPERIMENT_POLICY",
    )
    assert {
        allocation.unit_name for allocation in idea.budget_allocations
    } == {"IDEATION_SLOT", "IMPLEMENTATION_SLOT", "QUALIFICATION_SLOT"}
    assert {
        allocation.unit_name for allocation in experiment.budget_allocations
    } == {"EXPERIMENT_SLOT"}
    assert all(
        item.subject_kind == "OPEN_RESEARCH_SPEC"
        for item in idea.acquisition_decisions
    )
    assert all(
        item.subject_kind == "EXECUTABLE_CAPABILITY"
        for item in experiment.acquisition_decisions
    )
    assert (
        idea.acquisition_decisions[0].feature_schema_digest
        != experiment.acquisition_decisions[0].feature_schema_digest
    )
    assert "catalog fallback" in idea.explanation
    assert "outcome feature" in experiment.explanation
    assert (
        "FIXED_66_CATALOG_FALLBACK"
        not in idea.canonical_bytes().decode("utf-8")
    )


def test_out_of_support_is_explicit_and_allocates_nothing() -> None:
    decision = run_static_experiment_policy(
        out_of_support_experiment_input()
    )

    assert decision.support_status is PolicySupportStatusV1.OUT_OF_SUPPORT
    assert decision.support_reason_codes == ("SUBJECT_OUTSIDE_FROZEN_SLATE",)
    assert decision.policy_mode == RESEARCH_STATIC_VNEXT
    assert decision.fallback_scope == EXPERIMENT_FALLBACK_SCOPE_V1
    assert decision.budget_allocations == ()
    assert decision.acquisition_decisions
    assert all(
        item.disposition is AcquisitionDispositionV1.DEFER
        for item in decision.acquisition_decisions
    )
    assert all(
        "SUBJECT_OUTSIDE_FROZEN_SLATE" in item.reason_codes
        for item in decision.acquisition_decisions
    )


def test_idea_and_experiment_budgets_cannot_cross_domains() -> None:
    with pytest.raises(OpenMetaContractError, match="IdeaBudgetV1"):
        replace(
            idea_policy_input(),
            budget=ExperimentBudgetV1(experiment_slots=1),
        )

    with pytest.raises(OpenMetaContractError, match="cannot use"):
        PolicyBudgetAllocationV1(
            stage=AcquisitionStageV1.IDEA,
            target_kind="OPEN_RESEARCH_SPEC",
            target_ref="open-spec:wrong-budget:v1",
            units=1,
            unit_name="EXPERIMENT_SLOT",
            reason_code="CROSS_DOMAIN_NOT_ALLOWED",
        )


def test_policy_input_and_static_decision_are_origin_outcome_blind() -> None:
    episode = research_episode(comparison_identity())
    changed_outcome = replace(
        episode,
        outcome_ref="outcome:historical-alternative",
        outcome_digest=digest("historical-alternative-outcome"),
        mechanism_interpretation="Different historical interpretation.",
    )
    first_input = idea_policy_input(
        history=(TypedEpisodePolicySummaryV1.from_episode(episode),)
    )
    second_input = idea_policy_input(
        history=(TypedEpisodePolicySummaryV1.from_episode(changed_outcome),)
    )
    first = run_static_idea_policy(first_input)
    second = run_static_idea_policy(second_input)

    forbidden = ("origin", "outcome", "held_out", "source_label", "producer")
    assert not any(
        token in key.lower()
        for key in _keys(first_input.canonical_dict())
        for token in forbidden
    )
    assert first_input.digest != second_input.digest
    assert first.budget_allocations == second.budget_allocations
    assert first.acquisition_decisions == second.acquisition_decisions
    assert first.support_status is second.support_status
    assert first.support_reason_codes == second.support_reason_codes


def test_input_and_decision_digests_are_order_stable() -> None:
    candidates = idea_candidates()
    forward = idea_policy_input(candidates=candidates)
    reverse = idea_policy_input(candidates=tuple(reversed(candidates)))

    assert forward.digest == reverse.digest
    assert run_static_idea_policy(forward).digest == run_static_idea_policy(
        reverse
    ).digest


def test_activation_is_next_round_for_experiment_and_next_campaign_for_idea() -> None:
    idea_decision = run_static_idea_policy(idea_policy_input())
    experiment_decision = run_static_experiment_policy(
        experiment_policy_input()
    )
    idea = schedule_static_policy_activation(
        idea_decision,
        current_policy=current_idea_policy(),
    )
    experiment = schedule_static_policy_activation(
        experiment_decision,
        current_policy=current_experiment_policy(),
    )

    assert idea.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert experiment.activation_boundary == NEXT_ROUND
    assert not idea.promotion_authorized
    assert not experiment.promotion_authorized
    assert not idea.replaces_current_policy
    assert not experiment.replaces_current_policy

    with pytest.raises(OpenMetaContractError, match="NEXT_ROUND"):
        replace(experiment, activation_boundary=NEXT_FRESH_CAMPAIGN)
    with pytest.raises(OpenMetaContractError, match="cannot replace"):
        replace(experiment, replaces_current_policy=True)


def test_replay_round_trip_and_digest_are_byte_stable(tmp_path: Path) -> None:
    dataset = replay_dataset()
    reversed_dataset = OpenMetaReplayDatasetV1(
        dataset_version="1.0.0",
        records=tuple(reversed(dataset.records)),
    )
    assert dataset.digest == reversed_dataset.digest

    first_path = tmp_path / "open_meta_replay_v1.json"
    second_path = tmp_path / "open_meta_replay_v1_second.json"
    first_file_digest = write_open_meta_replay_dataset(first_path, dataset)
    loaded = read_open_meta_replay_dataset(first_path)
    second_file_digest = write_open_meta_replay_dataset(second_path, loaded)

    assert loaded == dataset
    assert loaded.digest == dataset.digest
    assert first_file_digest == second_file_digest
    assert first_path.read_bytes() == second_path.read_bytes()
    assert {record.activation_boundary for record in loaded.records} == {
        NEXT_ROUND,
        NEXT_FRESH_CAMPAIGN,
    }
    assert all(record.policy_version for record in loaded.records)
    assert all(record.policy_input_digest for record in loaded.records)
    assert all(record.budget_allocations for record in loaded.records)
    assert all(record.acquisition_decisions for record in loaded.records)


def test_replay_builder_rejects_cross_bound_activation() -> None:
    idea_decision = run_static_idea_policy(idea_policy_input())
    experiment_decision = run_static_experiment_policy(
        experiment_policy_input()
    )
    experiment_activation = schedule_static_policy_activation(
        experiment_decision,
        current_policy=current_experiment_policy(),
    )

    with pytest.raises(OpenMetaContractError, match="does not bind"):
        build_open_meta_replay_record(
            decision=idea_decision,
            activation=experiment_activation,
        )
