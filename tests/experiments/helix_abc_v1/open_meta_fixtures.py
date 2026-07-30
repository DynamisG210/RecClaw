from __future__ import annotations

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.open_meta import (
    CapabilityIdentityV1,
    ExperimentBudgetV1,
    ExperimentPolicyInputV1,
    IdeaBudgetV1,
    IdeaCandidateIdentityV1,
    IdeaPolicyInputV1,
    OpenMetaReplayDatasetV1,
    PolicyIdentityV1,
    TypedEpisodePolicySummaryV1,
    build_open_meta_replay_record,
    run_static_experiment_policy,
    run_static_idea_policy,
    schedule_static_policy_activation,
)

from scientific_episode_fixtures import comparison_identity, research_episode


def digest(label: str) -> str:
    return sha256_digest({"open_meta_fixture": label})


def episode_summary() -> TypedEpisodePolicySummaryV1:
    return TypedEpisodePolicySummaryV1.from_episode(
        research_episode(comparison_identity())
    )


def idea_candidates() -> tuple[IdeaCandidateIdentityV1, ...]:
    return (
        IdeaCandidateIdentityV1(
            research_spec_ref="open-spec:interaction-gate:v1",
            research_spec_digest=digest("open-spec-interaction-gate"),
            direction_ref="direction:interaction-head",
            direction_digest=digest("direction-interaction-head"),
            high_change=True,
            current_profile_expressible=False,
        ),
        IdeaCandidateIdentityV1(
            research_spec_ref="open-spec:propagation-gate:v1",
            research_spec_digest=digest("open-spec-propagation-gate"),
            direction_ref="direction:propagation",
            direction_digest=digest("direction-propagation"),
            high_change=True,
            current_profile_expressible=False,
        ),
    )


def capability_candidates() -> tuple[CapabilityIdentityV1, ...]:
    return (
        CapabilityIdentityV1(
            capability_ref="capability:interaction-gate:v1",
            capability_digest=digest("capability-interaction-gate"),
        ),
        CapabilityIdentityV1(
            capability_ref="capability:propagation-gate:v1",
            capability_digest=digest("capability-propagation-gate"),
        ),
    )


def idea_policy_input(
    *,
    history: tuple[TypedEpisodePolicySummaryV1, ...] | None = None,
    candidates: tuple[IdeaCandidateIdentityV1, ...] | None = None,
) -> IdeaPolicyInputV1:
    return IdeaPolicyInputV1(
        research_context_ref="research-context:fresh-campaign:r0",
        research_context_digest=digest("research-context"),
        protocol_ref="protocol:ml1m-general-cf:v1",
        protocol_digest=digest("protocol"),
        current_profile_ref="profile:wave1-next-fresh:v1",
        current_profile_digest=digest("profile"),
        candidates=candidates or idea_candidates(),
        budget=IdeaBudgetV1(
            ideation_slots=2,
            implementation_slots=1,
            qualification_slots=1,
        ),
        historical_episode_summaries=(
            (episode_summary(),) if history is None else history
        ),
    )


def experiment_policy_input(
    *,
    acquisition_subjects: tuple[CapabilityIdentityV1, ...] | None = None,
) -> ExperimentPolicyInputV1:
    capabilities = capability_candidates()
    return ExperimentPolicyInputV1(
        research_context_ref="research-context:fresh-campaign:r2-round-1",
        research_context_digest=digest("research-context-r2"),
        protocol_ref="protocol:ml1m-general-cf:v1",
        protocol_digest=digest("protocol"),
        current_profile_ref="profile:wave1-next-fresh:v1",
        current_profile_digest=digest("profile"),
        frozen_slate_ref="frozen-slate:fresh-campaign:r2-round-1",
        frozen_slate_digest=digest("frozen-slate"),
        frozen_slate_capabilities=capabilities,
        acquisition_subjects=acquisition_subjects or capabilities,
        budget=ExperimentBudgetV1(experiment_slots=1),
        historical_episode_summaries=(episode_summary(),),
    )


def out_of_support_experiment_input() -> ExperimentPolicyInputV1:
    outsider = CapabilityIdentityV1(
        capability_ref="capability:not-in-frozen-slate:v1",
        capability_digest=digest("capability-outside-slate"),
    )
    return experiment_policy_input(acquisition_subjects=(outsider,))


def current_idea_policy() -> PolicyIdentityV1:
    return PolicyIdentityV1(
        ref="policy:current-open-idea:v0",
        digest_value=digest("current-idea-policy"),
    )


def current_experiment_policy() -> PolicyIdentityV1:
    return PolicyIdentityV1(
        ref="policy:current-open-experiment:v0",
        digest_value=digest("current-experiment-policy"),
    )


def replay_dataset() -> OpenMetaReplayDatasetV1:
    idea_decision = run_static_idea_policy(idea_policy_input())
    experiment_decision = run_static_experiment_policy(
        experiment_policy_input()
    )
    idea_activation = schedule_static_policy_activation(
        idea_decision,
        current_policy=current_idea_policy(),
    )
    experiment_activation = schedule_static_policy_activation(
        experiment_decision,
        current_policy=current_experiment_policy(),
    )
    return OpenMetaReplayDatasetV1(
        dataset_version="1.0.0",
        records=(
            build_open_meta_replay_record(
                decision=idea_decision,
                activation=idea_activation,
            ),
            build_open_meta_replay_record(
                decision=experiment_decision,
                activation=experiment_activation,
            ),
        ),
    )
