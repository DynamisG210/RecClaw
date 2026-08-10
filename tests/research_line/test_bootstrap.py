from __future__ import annotations

from dataclasses import replace
from typing import Any

from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    project_candidate_proposal_v4,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import CapabilityResolutionResultV1
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    bind_search_candidate,
    freeze_experiment_slate,
    route_frozen_experiment_slate,
    adapt_current_search_profile,
)
from recclaw_core.research_line.bootstrap import bootstrap_search_pool
from recclaw_core.research_line.interfaces import ResearchContext
from recclaw_core.research_line.runtime import (
    bindings_for_context,
    resolver_environment_for_profile,
)


def _context(
    profile: Any,
    policy: Any,
    *,
    unresolved_questions: tuple[dict[str, Any], ...] = (),
    frontier: dict[str, Any] | None = None,
    scientific_memory: dict[str, Any] | None = None,
) -> ResearchContext:
    return ResearchContext(
        campaign_id=profile.campaign_id,
        round_index=1,
        knowledge_base={"search_space": "BL-ICF"},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier=frontier or {"unresolved_axes": ("architecture",)},
        scientific_memory=scientific_memory
        or {"uncovered_axes": ("self_supervision",)},
        unresolved_questions=unresolved_questions,
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


def test_bootstrap_is_deterministic_axis_diverse_and_not_bpr_only() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:bootstrap")
    policy = replace(
        initial_research_policy(),
        mechanism_axis_targeting=("objective",),
    )
    context = _context(
        profile,
        policy,
        unresolved_questions=(
            {"question": "Which propagation mechanism moves the frontier?", "mechanism_axis": "propagation"},
        ),
        frontier={"unresolved_axes": ("architecture",)},
        scientific_memory={"uncovered_axes": ("message_transform",)},
    )

    first = bootstrap_search_pool(context, profile, policy)
    second = bootstrap_search_pool(context, profile, policy)

    assert len(first) == 4
    assert tuple(item.digest for item in first) == tuple(item.digest for item in second)
    assert len({item.mechanism_axis for item in first}) == 4
    assert tuple(item.mechanism_axis for item in first) == (
        "objective",
        "propagation",
        "architecture",
        "message_transform",
    )
    assert any("LIGHTGCN" in item.mechanism_id for item in first)
    for proposal in first:
        assert proposal.feature_evidence.compile_valid is True
        assert proposal.utility_features.runnable_probability == 1.0
        assert proposal.matched_control_plan.primary_candidate_id == proposal.candidate_id
        assert proposal.matched_control_plan.comparator_program_digest is not None
        assert proposal.matched_control_plan.plan_status == "QUEUE_MATCHED_CONTROL"


def test_policy_axis_priority_changes_bootstrap_pool_order() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:bootstrap-priority")
    first_policy = replace(
        initial_research_policy(),
        mechanism_axis_targeting=("objective", "propagation", "geometry"),
    )
    second_policy = replace(
        first_policy,
        mechanism_axis_targeting=("propagation", "objective", "geometry"),
    )
    first_context = _context(profile, first_policy)
    first = bootstrap_search_pool(first_context, profile, first_policy)
    second = bootstrap_search_pool(_context(profile, second_policy), profile, second_policy)

    assert first[0].mechanism_axis == "objective"
    assert second[0].mechanism_axis == "propagation"
    assert tuple(item.mechanism_axis for item in first) != tuple(
        item.mechanism_axis for item in second
    )
    assert tuple(item.digest for item in first) != tuple(item.digest for item in second)
    assert first_context.policy == first_policy.to_dict()


def test_bootstrap_proposals_resolve_bind_and_route_through_search() -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:bootstrap-bind")
    policy = initial_research_policy()
    context = _context(profile, policy)
    bindings = _bindings(context, profile)
    environment = _environment(profile)
    pool = bootstrap_search_pool(context, profile, policy)
    search_bindings = []
    for proposal in pool:
        spec, facts = project_candidate_proposal_v4(
            proposal,
            bindings=bindings,
        )
        resolution = resolve_capability(
            spec,
            resolution_facts=facts,
            environment=environment,
        )
        assert resolution.resolution is CapabilityResolutionResultV1.SEARCH_READY
        entry = next(
            item
            for item in profile.entries
            if item.semantic_identity_digest
            == facts["requested_current_semantics_digest"]
        )
        search_bindings.append(
            bind_search_candidate(
                profile=profile,
                proposal=proposal,
                capability_ref=entry.capability_ref,
            )
        )

    slate = freeze_experiment_slate(
        profile=profile,
        bindings=tuple(search_bindings),
        budget_snapshot={"experiment_opportunities": 1},
    )
    acquisition = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=StrongStaticRouterV1(slate_ceiling=3),
        policy_projection=policy.to_dict(),
    )

    assert acquisition.selected_binding is not None
    assert acquisition.selected_binding.proposal in pool
