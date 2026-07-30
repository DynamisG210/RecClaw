from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    CampaignRuntimeError,
    bl_icf_executable_profile_v2,
    executable_mechanisms,
    execution_recipe_for_program,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.capability_admission import (
    VersionedCapabilityRegistry,
)
from recclaw_core.experiments.helix_abc_v1.compilation_cache import (
    compile_campaign_program,
)
from recclaw_core.experiments.helix_abc_v1.next_fresh_profile import (
    NextFreshProfileBuildManifest,
    build_next_fresh_profile,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DiscriminativeExperimentPlanV1,
    DiscoveryCreditV1,
    MatchedControlPlanV1,
    ProposalIntentV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    IdeaRouteV1,
    SearchAdapterError,
    SearchProfileActivationV1,
    SearchProfileEntryOriginV1,
    acquire_candidate_idea,
    acquire_open_idea,
    activate_next_fresh_search_profile,
    adapt_current_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    predecessor_executable_entries,
    route_frozen_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    AcquisitionDispositionV1,
    AcquisitionStageV1,
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)


TEST_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"
E0_FIXTURE_PATH = FIXTURE_ROOT / "e0_search_adapter_cases_v1.json"
A0_FIXTURE_PATH = FIXTURE_ROOT / "a0_open_spec_cases_v1.json"
BL_ICF_FIXTURE_PATH = (
    TEST_ROOT / "fixtures" / "bl_icf_anchor_programs_v1.json"
)


def _fixture(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(label: str) -> str:
    return sha256_digest({"fixture": f"e0:{label}"})


def _utility(*, selected: bool) -> SearchUtilityFeaturesV1:
    return SearchUtilityFeaturesV1(
        runnable_probability=0.99 if selected else 0.80,
        useful_signal=0.99 if selected else 0.70,
        frontier_potential=0.99 if selected else 0.60,
        information_gain=0.99 if selected else 0.70,
        cost=0.05 if selected else 0.20,
        blocker_risk=0.01 if selected else 0.10,
    )


def _proposal(
    *,
    candidate_id: str,
    mechanism_id: str,
    mechanism_axis: str,
    mechanism_program: dict[str, Any] | Any,
    protocol_digest: str,
    selected: bool = False,
    role: str = "mechanism_composer",
) -> CandidateProposalV4:
    control = MatchedControlPlanV1(
        mechanism_question_digest=_digest(
            f"question:{candidate_id}"
        ),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest=protocol_digest,
        changed_axis=mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    discriminative = (
        DiscriminativeExperimentPlanV1(
            competing_hypotheses=(
                "The candidate mechanism causes the signature.",
                "The matched control explains the signature.",
            ),
            predicted_outcome_signature=(
                "A deterministic E0 fixture signature."
            ),
            primary_candidate=candidate_id,
            matched_control_plan=control,
            falsifier="The matched control reproduces the signature.",
            next_decision_rule=(
                "Prefer the explanation with the matched signature."
            ),
        )
        if role == "falsification_designer"
        else None
    )
    utility = _utility(selected=selected)
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id=f"producer:e0:{role}",
        producer_role=role,
        proposal_intent=(
            ProposalIntentV1.FALSIFICATION
            if role == "falsification_designer"
            else ProposalIntentV1.DISCOVERY
        ),
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism_id,
        mechanism_axis=mechanism_axis,
        mechanism_program=mechanism_program,
        candidate_label=f"E0 Search projection for {mechanism_id}",
        mechanism_hypothesis=(
            f"{mechanism_id} may change the frozen development signature."
        ),
        competing_hypothesis=(
            "The matched control may explain the same signature."
        ),
        predicted_outcome_signature=(
            "A deterministic E0 fixture signature."
        ),
        failure_mode="The predicted signature is absent.",
        utility_features=utility,
        feature_evidence=RouterFeatureEvidenceV1(
            compile_valid=True,
            handler_available=True,
            materializer_available=True,
            blocker_rate=0.0,
            semantic_duplicate=False,
            parent_available=True,
            mechanism_depth=1,
            estimated_cost=utility.cost,
            llm_diagnostic=utility,
        ),
        matched_control_plan=control,
        discriminative_plan=discriminative,
        parent_candidate_id=None,
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


def _fixed_proposals(
    *, protocol_digest: str
) -> tuple[CandidateProposalV4, ...]:
    roles = (
        "mechanism_composer",
        "lineage_refiner",
        "falsification_designer",
        "frontier_architect",
    )
    return tuple(
        _proposal(
            candidate_id=f"cand-e0-fixed-{index:03d}",
            mechanism_id=mechanism.mechanism_id,
            mechanism_axis=mechanism.mechanism_axis,
            mechanism_program=mechanism.mechanism_program,
            protocol_digest=protocol_digest,
            role=roles[index % len(roles)],
        )
        for index, mechanism in enumerate(executable_mechanisms())
    )


def _outside_66_program() -> dict[str, Any]:
    document = _fixture(BL_ICF_FIXTURE_PATH)
    program = next(
        copy.deepcopy(row["program"])
        for row in document["fixtures"]
        if row["anchor_name"] == "NGCF"
    )
    report = compile_campaign_program(program)
    assert report.is_valid
    with pytest.raises(CampaignRuntimeError, match="outside the exact"):
        execution_recipe_for_program(program)
    return program


def _qualified_capability(
    *,
    profile: Any,
    program: dict[str, Any],
) -> QualifiedCapabilityV1:
    report = compile_campaign_program(program)
    assert report.is_valid
    return QualifiedCapabilityV1(
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="1.0.0",
        semantic_identity_ref="semantics:e0-qualified-ngcf",
        semantic_identity_digest=str(report.mechanism_semantics_digest),
        executable_entrypoint=(
            "recclaw_ext.generated.e0_qualified:E0QualifiedModel"
        ),
        candidate_package_ref="candidate-package:e0-qualified-ngcf",
        candidate_package_digest=_digest("candidate-package"),
        source_tree_digest=_digest("source-tree"),
        qualification_receipt_ref=(
            "qualification-receipt:e0-qualified-ngcf"
        ),
        qualification_receipt_digest=_digest("qualification-receipt"),
        qualification_stage=QualificationStageV1.ONE_EPOCH_SMOKE,
        qualification_status=QualificationStatusV1.PASS,
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
        predecessor_capability_ref=None,
        predecessor_capability_digest=None,
        current_campaign_ineligible=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )


def _next_profile(
    *,
    current_profile: Any,
    current_slate: Any,
    capability: QualifiedCapabilityV1,
) -> tuple[VersionedCapabilityRegistry, Any, Any]:
    registry = VersionedCapabilityRegistry.build(
        registry_version="e0-qualified-registry-v1",
        predecessor_registry_ref="registry:e0-fixed-66",
        predecessor_registry_digest=_digest("fixed-registry"),
        protocol_ref=current_profile.protocol_ref,
        protocol_digest=current_profile.protocol_digest,
        capabilities=(capability,),
    )
    manifest = NextFreshProfileBuildManifest(
        profile_version="e0-next-fresh-v1",
        predecessor_profile_ref=current_profile.profile_ref,
        predecessor_profile_digest=current_profile.profile_digest,
        current_campaign_profile_ref=current_profile.profile_ref,
        current_campaign_profile_digest=current_profile.profile_digest,
        current_campaign_slate_ref=current_slate.slate_id,
        current_campaign_slate_digest=current_slate.digest,
        predecessor_executable_entries=predecessor_executable_entries(
            current_profile
        ),
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        registry_version=registry.registry_version,
        protocol_ref=current_profile.protocol_ref,
        protocol_digest=current_profile.protocol_digest,
        compatibility_requirements=(
            "general collaborative filtering",
            "pairwise input",
        ),
    )
    profile, receipt = build_next_fresh_profile(manifest, registry)
    return registry, profile, receipt


def test_current_66_profile_is_mechanical_and_source_bytes_unchanged() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    expected = fixture["current_profile"]
    source = bl_icf_executable_profile_v2()
    before = canonical_json_bytes(source)

    first = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    second = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    after = canonical_json_bytes(bl_icf_executable_profile_v2())

    assert first == second
    assert first.profile_ref == expected["profile_ref"]
    assert first.profile_digest == expected["profile_digest"]
    assert len(first.entries) == expected["executable_count"] == 66
    assert first.activation is (
        SearchProfileActivationV1.CURRENT_FROZEN_CAMPAIGN
    )
    assert bytes_sha256(before) == expected["canonical_bytes_sha256"]
    assert len(before) == expected["canonical_byte_length"]
    assert before == after
    assert (
        sum(
            entry.semantic_identity_ref.startswith(
                "bl-icf-mechanism:BPR_MF"
            )
            for entry in first.entries
        )
        == expected["bpr_count"]
    )
    assert (
        sum(
            entry.semantic_identity_ref.startswith(
                "bl-icf-mechanism:LIGHTGCN"
            )
            for entry in first.entries
        )
        == expected["lightgcn_count"]
    )
    assert predecessor_executable_entries(first) == tuple(
        sorted(
            predecessor_executable_entries(first),
            key=lambda item: item[0],
        )
    )


def test_all_current_66_preserve_the_existing_router_trace() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    profile = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    proposals = _fixed_proposals(
        protocol_digest=profile.protocol_digest
    )
    entry_by_semantics = {
        entry.semantic_identity_digest: entry
        for entry in profile.entries
    }
    bindings = tuple(
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=entry_by_semantics[
                executable_mechanisms()[index].mechanism_semantics_digest
            ].capability_ref,
        )
        for index, proposal in enumerate(proposals)
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot=fixture["experiment_budget"],
    )
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=66,
    )

    direct = router.route(proposals)
    adapted = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
    )

    assert adapted.route_trace == direct
    assert adapted.route_trace.digest == direct.digest
    assert len(adapted.route_trace.ranked_candidate_ids) == 66
    assert len(adapted.decisions) == 66
    assert sum(
        decision.disposition is AcquisitionDispositionV1.SELECT
        for decision in adapted.decisions
    ) == 1
    assert all(
        decision.stage is AcquisitionStageV1.EXPERIMENT
        for decision in adapted.decisions
    )
    assert all(
        decision.budget_snapshot_digest
        == sha256_digest(fixture["experiment_budget"])
        for decision in adapted.decisions
    )


def test_idea_acquisition_routes_search_innovation_and_blocked() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    a0_fixture = _fixture(A0_FIXTURE_PATH)
    profile = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    bindings = frozen_search_bindings(
        context_ref="context:e0-idea-acquisition",
        context_digest=_digest("idea-context"),
    )
    environment = frozen_search_resolver_environment(
        available_dependencies=("recbole-runtime",),
        budget_limits={
            "implementation_tokens": 5000,
            "implementation_units": 2,
        },
    )
    current_proposal = _fixed_proposals(
        protocol_digest=profile.protocol_digest
    )[0]

    search = acquire_candidate_idea(
        current_proposal,
        bindings=bindings,
        environment=environment,
        required_budget={"implementation_tokens": 100},
    )
    innovation_draft = copy.deepcopy(a0_fixture["producer_drafts"][0])
    innovation = acquire_open_idea(
        innovation_draft,
        bindings=bindings,
        environment=environment,
    )
    unsupported_draft = copy.deepcopy(innovation_draft)
    unsupported_draft["resolution_facts"][
        "required_dependencies"
    ] = ["missing-e0-dependency"]
    unsupported = acquire_open_idea(
        unsupported_draft,
        bindings=bindings,
        environment=environment,
    )
    deferred_draft = copy.deepcopy(innovation_draft)
    deferred_draft["compatibility_requirements"].append(
        "online feedback"
    )
    deferred = acquire_open_idea(
        deferred_draft,
        bindings=bindings,
        environment=environment,
    )
    invalid_draft = copy.deepcopy(innovation_draft)
    invalid_draft["current_profile_expressibility_claim"] = "UNRESOLVED"
    invalid = acquire_open_idea(
        invalid_draft,
        bindings=bindings,
        environment=environment,
    )

    assert search.route is IdeaRouteV1.SEARCH
    assert search.search_proposal is current_proposal
    assert search.resolution.resolution is (
        CapabilityResolutionResultV1.SEARCH_READY
    )
    assert search.decision.disposition is (
        AcquisitionDispositionV1.SELECT
    )
    assert innovation.route is IdeaRouteV1.INNOVATION
    assert innovation.search_proposal is None
    assert innovation.resolution.resolution is (
        CapabilityResolutionResultV1.INNOVATION_REQUIRED
    )
    assert innovation.resolution.no_silent_fallback is True
    assert innovation.resolution.catalog_fallback_used is False
    assert unsupported.route is IdeaRouteV1.BLOCKED
    assert unsupported.search_proposal is None
    assert unsupported.resolution.resolution is (
        CapabilityResolutionResultV1.UNSUPPORTED
    )
    assert unsupported.decision.disposition is (
        AcquisitionDispositionV1.REJECT
    )
    assert deferred.route is IdeaRouteV1.BLOCKED
    assert deferred.search_proposal is None
    assert deferred.resolution.resolution is (
        CapabilityResolutionResultV1.DEFERRED_PROTOCOL_CHANGE
    )
    assert deferred.decision.disposition is (
        AcquisitionDispositionV1.DEFER
    )
    assert invalid.route is IdeaRouteV1.BLOCKED
    assert invalid.search_proposal is None
    assert invalid.resolution.resolution is (
        CapabilityResolutionResultV1.INVALID_SPEC
    )
    assert invalid.decision.disposition is (
        AcquisitionDispositionV1.REJECT
    )
    assert (
        search.decision.feature_schema_digest
        != search.decision.budget_schema_digest
        != search.decision.eligibility_schema_digest
    )


def test_unqualified_open_program_cannot_enter_current_slate() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    profile = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    program = _outside_66_program()
    proposal = _proposal(
        candidate_id="cand-e0-unqualified-open",
        mechanism_id="NGCF_UNQUALIFIED_OPEN",
        mechanism_axis="message_transform",
        mechanism_program=program,
        protocol_digest=profile.protocol_digest,
        selected=True,
    )

    with pytest.raises(
        SearchAdapterError,
        match="fixed-profile proposal is outside the exact 66",
    ):
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=profile.entries[0].capability_ref,
        )
    with pytest.raises(
        SearchAdapterError,
        match="not in the active executable profile",
    ):
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref="unqualified-open-spec:no-capability",
        )


def test_qualified_capability_activates_and_is_selected_only_next_fresh() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    current = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    current_fixed_proposal = _fixed_proposals(
        protocol_digest=current.protocol_digest
    )[0]
    fixed_entry = next(
        entry
        for entry in current.entries
        if entry.semantic_identity_digest
        == executable_mechanisms()[0].mechanism_semantics_digest
    )
    current_binding = bind_search_candidate(
        profile=current,
        proposal=current_fixed_proposal,
        capability_ref=fixed_entry.capability_ref,
    )
    current_slate = freeze_experiment_slate(
        profile=current,
        bindings=(current_binding,),
        budget_snapshot=fixture["experiment_budget"],
    )
    current_profile_before = current.canonical_bytes()
    current_slate_before = current_slate.canonical_bytes()
    outside_program = _outside_66_program()
    capability = _qualified_capability(
        profile=current,
        program=outside_program,
    )
    registry, next_profile, receipt = _next_profile(
        current_profile=current,
        current_slate=current_slate,
        capability=capability,
    )

    with pytest.raises(
        SearchAdapterError,
        match="not in the active executable profile",
    ):
        bind_search_candidate(
            profile=current,
            proposal=_proposal(
                candidate_id="cand-e0-qualified-current-blocked",
                mechanism_id="NGCF_QUALIFIED_NEXT",
                mechanism_axis="message_transform",
                mechanism_program=outside_program,
                protocol_digest=current.protocol_digest,
                selected=True,
            ),
            capability_ref=capability.capability_id,
        )
    with pytest.raises(
        SearchAdapterError,
        match="cannot activate in the current campaign",
    ):
        activate_next_fresh_search_profile(
            predecessor=current,
            next_profile=next_profile,
            registry=registry,
            fresh_campaign_id=current.campaign_id,
        )

    activated = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=fixture["campaigns"]["next_fresh"],
    )
    qualified_proposal = _proposal(
        candidate_id="cand-e0-qualified-next-selected",
        mechanism_id="NGCF_QUALIFIED_NEXT",
        mechanism_axis="message_transform",
        mechanism_program=outside_program,
        protocol_digest=activated.protocol_digest,
        selected=True,
    )
    qualified_binding = bind_search_candidate(
        profile=activated,
        proposal=qualified_proposal,
        capability_ref=capability.capability_id,
    )
    carried_fixed_binding = bind_search_candidate(
        profile=activated,
        proposal=current_fixed_proposal,
        capability_ref=fixed_entry.capability_ref,
    )
    next_slate = freeze_experiment_slate(
        profile=activated,
        bindings=(carried_fixed_binding, qualified_binding),
        budget_snapshot=fixture["experiment_budget"],
    )
    result = route_frozen_experiment_slate(
        profile=activated,
        slate=next_slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=2,
        ),
    )

    assert receipt.current_profile_unchanged is True
    assert receipt.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert activated.activation is (
        SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN
    )
    assert len(activated.entries) == 67
    assert qualified_binding.entry_origin is (
        SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
    )
    assert qualified_binding.executable_entrypoint == (
        capability.executable_entrypoint
    )
    assert result.selected_binding == qualified_binding
    selected_decision = next(
        decision
        for decision in result.decisions
        if decision.disposition is AcquisitionDispositionV1.SELECT
    )
    assert selected_decision.subject_ref == capability.capability_id
    assert selected_decision.subject_digest == capability.digest
    assert current.canonical_bytes() == current_profile_before
    assert current_slate.canonical_bytes() == current_slate_before


def test_qualified_exact_66_program_is_not_a_fallback_path() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    current = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    current_proposal = _fixed_proposals(
        protocol_digest=current.protocol_digest
    )[0]
    fixed_entry = next(
        entry
        for entry in current.entries
        if entry.semantic_identity_digest
        == executable_mechanisms()[0].mechanism_semantics_digest
    )
    current_slate = freeze_experiment_slate(
        profile=current,
        bindings=(
            bind_search_candidate(
                profile=current,
                proposal=current_proposal,
                capability_ref=fixed_entry.capability_ref,
            ),
        ),
        budget_snapshot=fixture["experiment_budget"],
    )
    outside_program = _outside_66_program()
    capability = _qualified_capability(
        profile=current,
        program=outside_program,
    )
    registry, next_profile, _receipt = _next_profile(
        current_profile=current,
        current_slate=current_slate,
        capability=capability,
    )
    activated = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=fixture["campaigns"]["next_fresh"],
    )
    mechanism = executable_mechanisms()[0]
    disguised_fallback = _proposal(
        candidate_id="cand-e0-qualified-fixed66-fallback",
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        protocol_digest=current.protocol_digest,
    )

    with pytest.raises(
        SearchAdapterError,
        match="qualified capability cannot silently fall back to the fixed 66",
    ):
        bind_search_candidate(
            profile=activated,
            proposal=disguised_fallback,
            capability_ref=capability.capability_id,
        )


def test_slate_profile_identity_and_budget_are_auditable() -> None:
    fixture = _fixture(E0_FIXTURE_PATH)
    profile = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    proposal = _fixed_proposals(
        protocol_digest=profile.protocol_digest
    )[0]
    entry = next(
        item
        for item in profile.entries
        if item.semantic_identity_digest
        == executable_mechanisms()[0].mechanism_semantics_digest
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=(
            bind_search_candidate(
                profile=profile,
                proposal=proposal,
                capability_ref=entry.capability_ref,
            ),
        ),
        budget_snapshot=fixture["experiment_budget"],
    )
    wrong_campaign_profile = replace(
        profile,
        campaign_id="campaign:e0-unrelated",
    )

    with pytest.raises(
        SearchAdapterError,
        match="not bound to the active campaign profile",
    ):
        route_frozen_experiment_slate(
            profile=wrong_campaign_profile,
            slate=slate,
            router=StrongStaticRouterV1(),
        )
    result = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=StrongStaticRouterV1(),
    )
    assert result.slate_ref == slate.slate_id
    assert result.slate_digest == slate.digest
    assert result.decisions[0].context_ref == slate.slate_id
    assert result.decisions[0].context_digest == slate.digest
    assert result.decisions[0].budget_schema_ref == (
        slate.budget_schema_ref
    )
    assert result.decisions[0].budget_snapshot_digest == sha256_digest(
        fixture["experiment_budget"]
    )
