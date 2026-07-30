from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import recclaw_core.experiments.helix_abc_v1 as public_api
from recclaw_core.experiments.helix_abc_v1 import (
    NEXT_FRESH_CAMPAIGN,
    NEXT_ROUND,
    RESEARCH_STATIC_VNEXT,
    OpenMetaReplayDatasetV1,
    PolicySupportStatusV1,
    ResearchFailureClassV1,
    ScientificEpisodeAdapterError,
    build_open_meta_replay_record,
    project_episode_to_mechanism_belief,
    read_open_meta_replay_dataset,
    run_static_experiment_policy,
    run_static_idea_policy,
    schedule_static_policy_activation,
    write_open_meta_replay_dataset,
)
from recclaw_core.experiments.helix_abc_v1.open_meta import (
    __all__ as OPEN_META_PUBLIC_NAMES,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
)

from open_meta_fixtures import (
    current_experiment_policy,
    current_idea_policy,
    experiment_policy_input,
    idea_candidates,
    idea_policy_input,
    out_of_support_experiment_input,
)
from scientific_episode_fixtures import (
    canonical_closure_fixtures,
    comparison_identity,
    research_episode,
)


def _episode_for(fixture_name: str):
    identity = comparison_identity()
    if fixture_name == "mechanism_negative":
        return research_episode(
            identity,
            failure_class=ResearchFailureClassV1.MECHANISM,
        )
    if fixture_name == "inconclusive":
        return research_episode(
            identity,
            failure_class=ResearchFailureClassV1.INCONCLUSIVE,
            evidence_class=public_api.EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        )
    return research_episode(identity)


def test_d1_projects_only_closed_mechanism_memory_into_existing_writer() -> None:
    identity = comparison_identity()
    closures = canonical_closure_fixtures()
    accepted = {}
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")

    for fixture_name in ("success", "mechanism_negative"):
        accepted[fixture_name] = project_episode_to_mechanism_belief(
            comparison_identity=identity,
            closure=closures[fixture_name],
            episode=_episode_for(fixture_name),
            mechanism_axis="propagation",
        )

    snapshot = writer.commit(
        round_index=1,
        expected_predecessor_digest=None,
        beliefs=tuple(accepted.values()),
        route_trace_digest=sha256_digest(
            {"wave2_d1_f0": "accepted-beliefs"}
        ),
        feedback_projection={
            "typed_episode_belief_refs": tuple(
                belief.hypothesis_id for belief in accepted.values()
            ),
        },
    )

    assert snapshot.beliefs == tuple(accepted.values())
    assert accepted["success"].evidence_for
    assert not accepted["success"].evidence_against
    assert accepted["mechanism_negative"].evidence_against
    assert not accepted["mechanism_negative"].evidence_for
    assert set(accepted["success"].to_dict()) == {
        "competing_hypotheses",
        "evidence_against",
        "evidence_for",
        "hypothesis_id",
        "mechanism_axis",
        "next_discriminative_test",
        "predicted_outcome_signature",
        "unresolved_confounds",
    }


def test_d1_rejects_every_non_mechanism_memory_fixture_without_a_write() -> None:
    identity = comparison_identity()
    closures = canonical_closure_fixtures()
    writer = SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY")
    rejected = {
        name
        for name in closures
        if name not in {"success", "mechanism_negative"}
    }

    for fixture_name in sorted(rejected):
        with pytest.raises(ScientificEpisodeAdapterError):
            project_episode_to_mechanism_belief(
                comparison_identity=identity,
                closure=closures[fixture_name],
                episode=_episode_for(fixture_name),
                mechanism_axis="propagation",
            )

    assert rejected == {
        "identity_drift",
        "implementation_failure",
        "inconclusive",
        "interface_failure",
        "missing_outcome",
        "package_failure",
        "protocol_failure",
        "provider_failure",
        "resource_failure",
        "runtime_failure",
    }
    assert writer.head is None


def test_f0_keeps_idea_and_experiment_static_support_domains_separate() -> None:
    idea = run_static_idea_policy(idea_policy_input())
    experiment = run_static_experiment_policy(experiment_policy_input())
    low_change = replace(idea_candidates()[0], high_change=False)
    unsupported_idea = run_static_idea_policy(
        idea_policy_input(candidates=(low_change,))
    )
    unsupported_experiment = run_static_experiment_policy(
        out_of_support_experiment_input()
    )

    assert idea.stage is public_api.AcquisitionStageV1.IDEA
    assert experiment.stage is public_api.AcquisitionStageV1.EXPERIMENT
    assert idea.policy_ref != experiment.policy_ref
    assert idea.policy_digest != experiment.policy_digest
    assert idea.policy_mode == experiment.policy_mode == RESEARCH_STATIC_VNEXT
    assert idea.support_status is PolicySupportStatusV1.IN_SUPPORT
    assert experiment.support_status is PolicySupportStatusV1.IN_SUPPORT
    assert unsupported_idea.support_reason_codes == (
        "PARAMETER_OR_CONFIG_ONLY_OUT_OF_SUPPORT",
    )
    assert unsupported_experiment.support_reason_codes == (
        "SUBJECT_OUTSIDE_FROZEN_SLATE",
    )
    for decision in (unsupported_idea, unsupported_experiment):
        assert decision.support_status is PolicySupportStatusV1.OUT_OF_SUPPORT
        assert decision.budget_allocations == ()
        assert decision.support_reason_codes
        assert all(
            item.disposition is public_api.AcquisitionDispositionV1.DEFER
            for item in decision.acquisition_decisions
        )


def test_f0_replay_digest_and_future_activation_are_deterministic(
    tmp_path: Path,
) -> None:
    idea = run_static_idea_policy(idea_policy_input())
    experiment = run_static_experiment_policy(experiment_policy_input())
    idea_activation = schedule_static_policy_activation(
        idea,
        current_policy=current_idea_policy(),
    )
    experiment_activation = schedule_static_policy_activation(
        experiment,
        current_policy=current_experiment_policy(),
    )
    dataset = OpenMetaReplayDatasetV1(
        dataset_version="1.0.0",
        records=(
            build_open_meta_replay_record(
                decision=idea,
                activation=idea_activation,
            ),
            build_open_meta_replay_record(
                decision=experiment,
                activation=experiment_activation,
            ),
        ),
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first_digest = write_open_meta_replay_dataset(first_path, dataset)
    second_digest = write_open_meta_replay_dataset(second_path, dataset)

    assert first_digest == second_digest
    assert first_path.read_bytes() == second_path.read_bytes()
    assert read_open_meta_replay_dataset(first_path) == dataset
    assert idea_activation.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert experiment_activation.activation_boundary == NEXT_ROUND
    for activation in (idea_activation, experiment_activation):
        assert activation.policy_frozen is True
        assert activation.promotion_authorized is False
        assert activation.replaces_current_policy is False


def test_g_public_exports_are_the_owner_declared_f0_and_d1_surfaces() -> None:
    assert set(OPEN_META_PUBLIC_NAMES) <= set(public_api.__all__)
    assert public_api.project_episode_to_mechanism_belief is (
        project_episode_to_mechanism_belief
    )
