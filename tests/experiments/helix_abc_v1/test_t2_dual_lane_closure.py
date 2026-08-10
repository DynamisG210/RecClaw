from __future__ import annotations

import copy
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    VersionedCapabilityRegistry,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
    project_candidate_proposal_v4,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchAdapterError,
    SearchProfileActivationV1,
    SearchProfileEntryOriginV1,
    adapt_current_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    predecessor_executable_entries,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    AcquisitionDispositionV1,
    CapabilityResolutionResultV1,
)
from recclaw_core.experiments.helix_abc_v1.vnext_orchestration import (
    build_local_next_fresh_profile,
    resolve_producer_outcomes,
    route_current_search_outcomes,
    route_next_fresh_search,
)
from recclaw_core.research_line.interfaces import ProducerOutcome

from test_e0_search_adapter import (
    A0_FIXTURE_PATH,
    E0_FIXTURE_PATH,
    _fixture,
    _outside_66_program,
    _proposal,
    _qualified_capability,
)


def _digest(label: str) -> str:
    return sha256_digest({"fixture": f"t2:{label}"})


def _environment() -> tuple[dict[str, Any], dict[str, Any]]:
    bindings = frozen_search_bindings(
        context_ref="context:t2-dual-lane",
        context_digest=_digest("context"),
    )
    environment = frozen_search_resolver_environment(
        available_dependencies=("recbole-runtime",),
        budget_limits={
            "implementation_tokens": 5000,
            "implementation_units": 2,
        },
    )
    return bindings, environment


def _open_outcome(
    draft: dict[str, Any],
    *,
    bindings: dict[str, Any],
) -> ProducerOutcome:
    spec, facts = project_open_producer_draft(draft, bindings=bindings)
    return ProducerOutcome(
        producer_role=spec.producer_role,
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        spec=spec,
        resolution_facts=facts,
    )


def _search_outcome(
    *,
    proposal: Any,
    bindings: dict[str, Any],
) -> ProducerOutcome:
    spec, facts = project_candidate_proposal_v4(
        proposal,
        bindings=bindings,
        required_budget={"implementation_tokens": 100},
    )
    return ProducerOutcome(
        producer_role=spec.producer_role,
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        spec=spec,
        resolution_facts=facts,
        source_proposal=proposal,
    )


def _router() -> StrongStaticRouterV1:
    return StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=2,
    )


def test_producer_outcomes_preserve_failures_and_all_five_dispositions() -> None:
    bindings, environment = _environment()
    draft = _fixture(A0_FIXTURE_PATH)["producer_drafts"][0]

    unsupported = copy.deepcopy(draft)
    unsupported["resolution_facts"]["required_dependencies"] = [
        "missing-t2-dependency"
    ]
    deferred = copy.deepcopy(draft)
    deferred["compatibility_requirements"].append("online feedback")
    invalid = copy.deepcopy(draft)
    invalid["current_profile_expressibility_claim"] = "UNRESOLVED"

    profile = adapt_current_search_profile(campaign_id="campaign:t2-current")
    mechanism = executable_mechanisms()[0]
    search_proposal = _proposal(
        candidate_id="cand-t2-search-ready",
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        protocol_digest=profile.protocol_digest,
        selected=True,
    )
    search = _search_outcome(proposal=search_proposal, bindings=bindings)
    failure = ProducerOutcome(
        producer_role="mechanism_composer",
        context_ref=bindings["context_ref"],
        context_digest=bindings["context_digest"],
        spec=None,
        resolution_facts={},
        failure_code="PRODUCER_TIMEOUT",
        failure_detail="typed test failure",
    )

    outcomes = (
        search,
        _open_outcome(draft, bindings=bindings),
        _open_outcome(unsupported, bindings=bindings),
        _open_outcome(deferred, bindings=bindings),
        _open_outcome(invalid, bindings=bindings),
        failure,
    )
    resolved = resolve_producer_outcomes(outcomes, environment=environment)

    assert tuple(item[1].resolution for item in resolved[:-1]) == (
        CapabilityResolutionResultV1.SEARCH_READY,
        CapabilityResolutionResultV1.INNOVATION_REQUIRED,
        CapabilityResolutionResultV1.UNSUPPORTED,
        CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE,
        CapabilityResolutionResultV1.INVALID_SPEC,
    )
    assert resolved[-1] == (failure, None)
    assert all(
        item[1] is None or item[1].no_silent_fallback
        for item in resolved
    )


def test_dual_lane_routes_current_and_next_fresh_without_same_round_or_fallback() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    bindings, environment = _environment()
    current = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    first_mechanism = executable_mechanisms()[0]
    second_mechanism = executable_mechanisms()[1]
    first_proposal = _proposal(
        candidate_id="cand-t2-current-search-first",
        mechanism_id=first_mechanism.mechanism_id,
        mechanism_axis=first_mechanism.mechanism_axis,
        mechanism_program=first_mechanism.mechanism_program,
        protocol_digest=current.protocol_digest,
        selected=True,
    )
    second_proposal = _proposal(
        candidate_id="cand-t2-current-search-second",
        mechanism_id=second_mechanism.mechanism_id,
        mechanism_axis=second_mechanism.mechanism_axis,
        mechanism_program=second_mechanism.mechanism_program,
        protocol_digest=current.protocol_digest,
        selected=False,
        role="frontier_architect",
    )
    first_outcome = _search_outcome(
        proposal=first_proposal,
        bindings=bindings,
    )
    second_outcome = _search_outcome(
        proposal=second_proposal,
        bindings=bindings,
    )
    current_resolved = resolve_producer_outcomes(
        (first_outcome, second_outcome),
        environment=environment,
    )
    assert all(resolution is not None for _outcome, resolution in current_resolved)
    current_pairs = tuple(
        (outcome, resolution)
        for outcome, resolution in current_resolved
        if resolution is not None
    )
    current_before = current.canonical_bytes()
    current_result = route_current_search_outcomes(
        current_pairs,
        active_profile=current,
        budget_snapshot=fixture["experiment_budget"],
        router=_router(),
    )
    assert current_result.selected_binding is not None
    assert len(current_result.decisions) == 2
    assert len(set(current_result.route_trace.ranked_candidate_ids)) == 2
    assert set(current_result.route_trace.ranked_candidate_ids) == {
        first_proposal.candidate_id,
        second_proposal.candidate_id,
    }
    assert len({decision.context_ref for decision in current_result.decisions}) == 1
    assert sum(
        decision.disposition is AcquisitionDispositionV1.SELECT
        for decision in current_result.decisions
    ) == 1
    assert current_result.selected_binding.proposal.candidate_id in {
        first_proposal.candidate_id,
        second_proposal.candidate_id,
    }
    assert current_result.selected_binding.entry_origin is (
        SearchProfileEntryOriginV1.FIXED_66
    )
    assert current.canonical_bytes() == current_before

    first_resolution = current_resolved[0][1]
    second_resolution = current_resolved[1][1]
    assert first_resolution is not None
    assert second_resolution is not None
    first_binding = bind_search_candidate(
        profile=current,
        proposal=first_proposal,
        capability_ref=first_resolution.resolved_current_capability_ref,
    )
    second_binding = bind_search_candidate(
        profile=current,
        proposal=second_proposal,
        capability_ref=second_resolution.resolved_current_capability_ref,
    )
    current_slate = freeze_experiment_slate(
        profile=current,
        bindings=(first_binding, second_binding),
        budget_snapshot=fixture["experiment_budget"],
    )
    current_slate_before = current_slate.canonical_bytes()
    outside_program = _outside_66_program()
    capability = _qualified_capability(
        profile=current,
        program=outside_program,
    )
    registry = VersionedCapabilityRegistry.build(
        registry_version="t2-registry-v1",
        predecessor_registry_ref="registry:t2-fixed",
        predecessor_registry_digest=_digest("predecessor-registry"),
        protocol_ref=current.protocol_ref,
        protocol_digest=current.protocol_digest,
        capabilities=(capability,),
    )
    _manifest, next_profile, receipt = build_local_next_fresh_profile(
        registry,
        profile_version="t2-next-profile-v1",
        predecessor_profile_ref=current.profile_ref,
        predecessor_profile_digest=current.profile_digest,
        current_campaign_slate_ref=current_slate.slate_id,
        current_campaign_slate_digest=current_slate.digest,
        predecessor_executable_entries=predecessor_executable_entries(current),
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )
    assert receipt.current_profile_unchanged is True

    qualified_proposal = _proposal(
        candidate_id="cand-t2-qualified-next",
        mechanism_id="NGCF_T2_QUALIFIED",
        mechanism_axis="message_transform",
        mechanism_program=outside_program,
        protocol_digest=current.protocol_digest,
        selected=True,
    )
    active, next_result = route_next_fresh_search(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=fixture["campaigns"]["next_fresh"],
        proposals=(qualified_proposal, first_proposal),
        capability_refs=(
            capability.capability_id,
            first_resolution.resolved_current_capability_ref,
        ),
        budget_snapshot=fixture["experiment_budget"],
        router=_router(),
    )
    assert active.activation is SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN
    assert next_result.selected_binding is not None
    assert next_result.selected_binding.entry_origin is (
        SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
    )
    assert next_result.selected_binding.capability_ref == capability.capability_id
    assert current.canonical_bytes() == current_before
    assert current_slate.canonical_bytes() == current_slate_before

    with pytest.raises(SearchAdapterError, match="cannot activate in the current campaign"):
        route_next_fresh_search(
            predecessor=current,
            next_profile=next_profile,
            registry=registry,
            fresh_campaign_id=current.campaign_id,
            proposals=(qualified_proposal,),
            capability_refs=(capability.capability_id,),
            budget_snapshot=fixture["experiment_budget"],
            router=_router(),
        )

    disguised_fallback = _proposal(
        candidate_id="cand-t2-qualified-fixed-fallback",
        mechanism_id=first_mechanism.mechanism_id,
        mechanism_axis=first_mechanism.mechanism_axis,
        mechanism_program=first_mechanism.mechanism_program,
        protocol_digest=current.protocol_digest,
    )
    with pytest.raises(
        SearchAdapterError,
        match="qualified capability cannot silently fall back to the fixed 66",
    ):
        route_next_fresh_search(
            predecessor=current,
            next_profile=next_profile,
            registry=registry,
            fresh_campaign_id="campaign:t2-fallback-check",
            proposals=(disguised_fallback,),
            capability_refs=(capability.capability_id,),
            budget_snapshot=fixture["experiment_budget"],
            router=_router(),
        )
