from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1 import search_adapter as adapter
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    project_open_producer_draft,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DiscoveryCreditV1,
    MatchedControlPlanV1,
    ProposalIntentV1,
    RouterHardGateReasonV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    OpenSpecSearchCandidateV1,
    SearchExecutableEntryV1,
    SearchExecutableProfileV1,
    SearchProfileActivationV1,
    SearchProfileEntryOriginV1,
    adapt_current_search_profile,
    bind_search_candidate,
    freeze_experiment_slate,
    open_spec_realization_identity,
    route_frozen_experiment_slate,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    NEXT_FRESH_CAMPAIGN,
    CapabilityKindV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanisms,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures"


def _digest(label: str) -> str:
    return sha256_digest({"open-spec-search-binding": label})


def _spec() -> Any:
    fixture = json.loads(
        (FIXTURE_ROOT / "a0_open_spec_cases_v1.json").read_text(
            encoding="utf-8"
        )
    )
    draft = dict(fixture["producer_drafts"][0])
    draft.update(
        {
            "idea_mode": "FRONTIER_HYPOTHESIS",
            "research_question": "Does the qualified interaction gate change ranking?",
            "observed_failure_mode": "NOT_OBSERVED",
            "closest_parent": "fixed-profile interaction model",
            "minimal_testable_wedge": "add one candidate-local interaction gate",
            "causal_chain": (
                "interaction gate",
                "pairwise score",
                "ranking signal",
            ),
            "discriminative_predictions": (
                "gate ablation removes the signature",
                "capacity control does not remove the signature",
            ),
            "mechanism_off_definition": "without the learned interaction gate",
            "resource_hypothesis": "one candidate-local model and one smoke run",
            "realization_mode": "NON_NESTED",
            "execution_contract": {
                "capability_family": "OPEN_INTERACTION_CUSTOM",
                "model": "FreshCandidateModel",
                "base_model_config": "LightGCN",
                "config": {"embedding_size": 8},
            },
        }
    )
    bindings = frozen_search_bindings(
        context_ref="context:open-spec-search-binding",
        context_digest=_digest("context"),
    )
    spec, _facts = project_open_producer_draft(draft, bindings=bindings)
    return spec


def _qualified_candidate(
    spec: Any,
) -> tuple[OpenSpecSearchCandidateV1, QualifiedCapabilityV1]:
    package_ref = "candidate-package:open-spec-search-binding"
    package_digest = _digest("candidate-package")
    root_ref = "candidate-root:open-spec-search-binding"
    root_digest = _digest("candidate-root")
    source_digest = _digest("source-tree")
    entrypoint = "recclaw_ext.open_binding:FreshCandidateModel"
    contract = dict(spec.execution_contract)
    semantic_ref, semantic_digest = open_spec_realization_identity(
        spec,
        candidate_package_ref=package_ref,
        candidate_package_digest=package_digest,
        candidate_root_ref=root_ref,
        candidate_root_digest=root_digest,
        source_tree_digest=source_digest,
        executable_entrypoint=entrypoint,
        execution_contract=contract,
    )
    receipt_ref = "qualification-receipt:open-spec-search-binding"
    receipt_digest = _digest("qualification-receipt")
    capability = QualifiedCapabilityV1(
        capability_kind=CapabilityKindV1.COMPLETE_MODEL,
        capability_version="open-spec-search-binding-v1",
        semantic_identity_ref=semantic_ref,
        semantic_identity_digest=semantic_digest,
        executable_entrypoint=entrypoint,
        candidate_package_ref=package_ref,
        candidate_package_digest=package_digest,
        source_tree_digest=source_digest,
        qualification_receipt_ref=receipt_ref,
        qualification_receipt_digest=receipt_digest,
        qualification_stage=QualificationStageV1.ONE_EPOCH_SMOKE,
        qualification_status=QualificationStatusV1.PASS,
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        compatibility_requirements=spec.compatibility_requirements,
        predecessor_capability_ref=None,
        predecessor_capability_digest=None,
        current_campaign_ineligible=True,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )
    utility = SearchUtilityFeaturesV1(
        runnable_probability=0.99,
        useful_signal=0.95,
        frontier_potential=0.99,
        information_gain=0.95,
        cost=0.10,
        blocker_risk=0.01,
    )
    candidate = OpenSpecSearchCandidateV1(
        spec=spec,
        capability_ref=capability.capability_id,
        capability_digest=capability.digest,
        candidate_package_ref=package_ref,
        candidate_package_digest=package_digest,
        candidate_root_ref=root_ref,
        candidate_root_digest=root_digest,
        source_tree_digest=source_digest,
        executable_entrypoint=entrypoint,
        execution_contract=contract,
        semantic_identity_ref=semantic_ref,
        semantic_identity_digest=semantic_digest,
        qualification_receipt_ref=receipt_ref,
        qualification_receipt_digest=receipt_digest,
        mechanism_axis="interaction_structure",
        utility_features=utility,
        feature_evidence=RouterFeatureEvidenceV1(
            compile_valid=False,
            handler_available=True,
            materializer_available=True,
            blocker_rate=0.01,
            semantic_duplicate=False,
            parent_available=True,
            mechanism_depth=1,
            estimated_cost=utility.cost,
            llm_diagnostic=utility,
        ),
    )
    return candidate, capability


def _active_profile(
    current: Any,
    capability: QualifiedCapabilityV1,
    *,
    include_fixed: bool,
) -> SearchExecutableProfileV1:
    qualified_entry = SearchExecutableEntryV1(
        capability_ref=capability.capability_id,
        capability_digest=capability.digest,
        executable_entrypoint=capability.executable_entrypoint,
        semantic_identity_ref=capability.semantic_identity_ref,
        semantic_identity_digest=capability.semantic_identity_digest,
        origin=SearchProfileEntryOriginV1.QUALIFIED_REGISTRY,
        qualification_receipt_ref=capability.qualification_receipt_ref,
        qualification_receipt_digest=capability.qualification_receipt_digest,
        activation_boundary=NEXT_FRESH_CAMPAIGN,
    )
    entries = (qualified_entry,)
    if include_fixed:
        entries = (current.entries[0], qualified_entry)
    return SearchExecutableProfileV1(
        campaign_id="campaign:open-spec-search-binding:fresh",
        profile_ref="profile:open-spec-search-binding:fresh",
        profile_digest=_digest("active-profile"),
        protocol_ref=current.protocol_ref,
        protocol_digest=current.protocol_digest,
        activation=SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN,
        predecessor_campaign_id=current.campaign_id,
        predecessor_profile_ref=current.profile_ref,
        predecessor_profile_digest=current.profile_digest,
        entries=entries,
    )


def _legacy_proposal(
    profile: Any,
    *,
    mechanism_index: int = 0,
    candidate_id: str | None = None,
) -> CandidateProposalV4:
    entry = profile.entries[mechanism_index]
    mechanism_id = entry.capability_ref.rsplit(":", 1)[-1]
    mechanism = next(
        item
        for item in executable_mechanisms()
        if item.mechanism_id == mechanism_id
    )
    utility = SearchUtilityFeaturesV1(
        runnable_probability=0.60,
        useful_signal=0.60,
        frontier_potential=0.40,
        information_gain=0.40,
        cost=0.40,
        blocker_risk=0.10,
    )
    candidate_id = candidate_id or "cand-open-spec-search-binding-legacy"
    control = MatchedControlPlanV1(
        mechanism_question_digest=_digest("legacy-question"),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest=profile.protocol_digest,
        changed_axis=mechanism.mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id="producer:open-spec-search-binding",
        producer_role="mechanism_composer",
        proposal_intent=ProposalIntentV1.DISCOVERY,
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        candidate_label="legacy mixed-slate candidate",
        mechanism_hypothesis="The fixed mechanism remains a useful control.",
        competing_hypothesis="The OpenSpec realization explains the signal.",
        predicted_outcome_signature="mixed-slate fixture signature",
        failure_mode="The fixed signature is absent.",
        utility_features=utility,
        feature_evidence=RouterFeatureEvidenceV1(
            compile_valid=True,
            handler_available=True,
            materializer_available=True,
            blocker_rate=0.10,
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


def test_open_candidate_identity_and_route_do_not_call_bl_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:current"
    )
    candidate, capability = _qualified_candidate(_spec())
    profile = _active_profile(current, capability, include_fixed=False)
    semantic_ref, semantic_digest = open_spec_realization_identity(
        candidate.spec,
        candidate_package_ref=candidate.candidate_package_ref,
        candidate_package_digest=candidate.candidate_package_digest,
        candidate_root_ref=candidate.candidate_root_ref,
        candidate_root_digest=candidate.candidate_root_digest,
        source_tree_digest=candidate.source_tree_digest,
        executable_entrypoint=candidate.executable_entrypoint,
        execution_contract=candidate.execution_contract,
    )

    assert candidate.semantic_identity_ref == semantic_ref
    assert candidate.semantic_identity_digest == semantic_digest
    assert candidate.semantic_identity_digest == candidate.realization_semantics_digest
    assert candidate.mechanism_id == (
        f"OPEN_{candidate.realization_semantics_digest[:16].upper()}"
    )

    def fail_compile(_program: Any) -> Any:
        pytest.fail("OpenSpec route must not invoke the BL-ICF compiler")

    monkeypatch.setattr(adapter, "compile_program", fail_compile)
    binding = bind_search_candidate(
        profile=profile,
        proposal=candidate,
        capability_ref=capability.capability_id,
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=(binding,),
        budget_snapshot={"training_runs": 1},
    )
    result = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=1,
        ),
    )

    assert result.selected_binding == binding
    assert result.route_trace.ranked_candidate_ids == (candidate.candidate_id,)
    decision = result.route_trace.decisions[0]
    assert decision.compile_report_digest is None
    assert decision.mechanism_semantics_digest == candidate.realization_semantics_digest


def test_qualified_open_candidate_survives_later_profile_expansion() -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:lineage-current"
    )
    candidate, capability = _qualified_candidate(_spec())
    admitted = _active_profile(current, capability, include_fixed=True)
    expanded = replace(
        admitted,
        campaign_id="campaign:open-spec-search-binding:lineage-expanded",
        profile_ref="profile:open-spec-search-binding:lineage-expanded",
        profile_digest=_digest("lineage-expanded-profile"),
        predecessor_campaign_id=admitted.campaign_id,
        predecessor_profile_ref=admitted.profile_ref,
        predecessor_profile_digest=admitted.profile_digest,
    )

    binding = bind_search_candidate(
        profile=expanded,
        proposal=candidate,
        capability_ref=capability.capability_id,
    )

    assert binding.capability_ref == capability.capability_id
    assert binding.entry_origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY


def test_mixed_slate_keeps_v4_compile_and_open_realization_in_one_trace() -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:mixed-current"
    )
    candidate, capability = _qualified_candidate(_spec())
    profile = _active_profile(current, capability, include_fixed=True)
    legacy = _legacy_proposal(profile)
    bindings = (
        bind_search_candidate(
            profile=profile,
            proposal=legacy,
            capability_ref=current.entries[0].capability_ref,
        ),
        bind_search_candidate(
            profile=profile,
            proposal=candidate,
            capability_ref=capability.capability_id,
        ),
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot={"training_runs": 1},
    )
    legacy_semantics = bindings[0].mechanism_semantics_digest
    result = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=2,
        ),
        executed_semantic_seed_pairs=((legacy_semantics, "seed-1"),),
        current_observation_seed="seed-1",
    )

    assert result.route_trace.ordered_candidate_ids == (
        legacy.candidate_id,
        candidate.candidate_id,
    )
    assert set(result.route_trace.ranked_candidate_ids) == {
        legacy.candidate_id,
        candidate.candidate_id,
    }
    decision_by_id = {
        decision.candidate_id: decision
        for decision in result.route_trace.decisions
    }
    assert decision_by_id[legacy.candidate_id].compile_report_digest is not None
    assert decision_by_id[candidate.candidate_id].compile_report_digest is None
    assert (
        decision_by_id[candidate.candidate_id].mechanism_semantics_digest
        == candidate.realization_semantics_digest
    )
    assert decision_by_id[legacy.candidate_id].allowed is True
    assert decision_by_id[legacy.candidate_id].reason is RouterHardGateReasonV1.ALLOW
    assert result.selected_binding == bindings[1]


def test_legacy_repeat_is_soft_penalized_but_remains_eligible() -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:legacy-repeat"
    )
    profile = replace(current, entries=(current.entries[0], current.entries[1]))
    repeat = _legacy_proposal(
        profile,
        mechanism_index=0,
        candidate_id="cand-open-spec-search-binding-repeat",
    )
    new = _legacy_proposal(
        profile,
        mechanism_index=1,
        candidate_id="cand-open-spec-search-binding-new",
    )
    bindings = (
        bind_search_candidate(
            profile=profile,
            proposal=repeat,
            capability_ref=profile.entries[0].capability_ref,
        ),
        bind_search_candidate(
            profile=profile,
            proposal=new,
            capability_ref=profile.entries[1].capability_ref,
        ),
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot={"training_runs": 1},
    )
    repeat_semantics = bindings[0].mechanism_semantics_digest
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=2,
    )
    result = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        executed_semantic_seed_pairs=((repeat_semantics, "seed-1"),),
        current_observation_seed="seed-1",
    )
    decision = {
        item.candidate_id: item for item in result.route_trace.decisions
    }[repeat.candidate_id]

    assert result.selected_binding == bindings[1]
    assert decision.allowed is True
    assert decision.reason is RouterHardGateReasonV1.ALLOW
    assert decision.reason is not RouterHardGateReasonV1.SEMANTIC_DUPLICATE
    without_history = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
    )
    assert result.route_trace.policy_digest != without_history.route_trace.policy_digest


def test_all_repeat_legacy_slate_still_selects_a_candidate() -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:all-repeats"
    )
    profile = replace(current, entries=(current.entries[0], current.entries[1]))
    first = _legacy_proposal(
        profile,
        mechanism_index=0,
        candidate_id="cand-open-spec-search-binding-all-repeat-1",
    )
    second = _legacy_proposal(
        profile,
        mechanism_index=1,
        candidate_id="cand-open-spec-search-binding-all-repeat-2",
    )
    bindings = tuple(
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=profile.entries[index].capability_ref,
        )
        for index, proposal in enumerate((first, second))
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot={"training_runs": 1},
    )
    history = tuple(
        (binding.mechanism_semantics_digest, "seed-1")
        for binding in bindings
    )
    result = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=1,
        ),
        executed_semantic_seed_pairs=history,
        current_observation_seed="seed-1",
    )

    assert result.selected_binding is not None
    assert result.route_trace.selected_candidate_id == first.candidate_id
    assert all(
        decision.reason is not RouterHardGateReasonV1.SEMANTIC_DUPLICATE
        for decision in result.route_trace.decisions
    )


def test_explicit_new_seed_validation_task_neutralizes_repeat_penalty() -> None:
    profile = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:replication-task"
    )
    proposal = _legacy_proposal(
        profile,
        candidate_id="cand-open-spec-search-binding-replication",
    )
    binding = bind_search_candidate(
        profile=profile,
        proposal=proposal,
        capability_ref=profile.entries[0].capability_ref,
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=(binding,),
        budget_snapshot={"training_runs": 1},
    )
    semantics = binding.mechanism_semantics_digest
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=1,
    )
    base_score = router.score(proposal.utility_features)
    replication = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        executed_semantic_seed_pairs=((semantics, "seed-1"),),
        current_observation_seed="seed-2",
        pending_task={
            "task_status": "PENDING",
            "task_type": "VALIDATE_SAME_CANDIDATE",
            "candidate_semantic_digest": semantics,
            "required_seed_or_control": "seed-2",
        },
    )
    same_seed_default = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        executed_semantic_seed_pairs=((semantics, "seed-1"),),
        current_observation_seed="seed-1",
        pending_task={
            "task_status": "PENDING",
            "task_type": "VALIDATE_SAME_CANDIDATE",
            "candidate_semantic_digest": semantics,
            "required_seed_or_control": "seed-1",
        },
    )

    assert replication.route_trace.selection_score == base_score
    assert same_seed_default.route_trace.selection_score == base_score - 0.25


def test_observed_axis_effects_change_the_existing_acquisition_ranking() -> None:
    current = adapt_current_search_profile(
        campaign_id="campaign:open-spec-search-binding:axis-effects"
    )
    first_axis = _legacy_proposal(current, mechanism_index=0).mechanism_axis
    second_index = next(
        index
        for index in range(1, len(current.entries))
        if _legacy_proposal(current, mechanism_index=index).mechanism_axis
        != first_axis
    )
    profile = replace(
        current,
        entries=(current.entries[0], current.entries[second_index]),
    )
    first = _legacy_proposal(
        profile,
        mechanism_index=0,
        candidate_id="cand-axis-effect-negative",
    )
    second = _legacy_proposal(
        profile,
        mechanism_index=1,
        candidate_id="cand-axis-effect-positive",
    )
    assert first.mechanism_axis != second.mechanism_axis
    bindings = tuple(
        bind_search_candidate(
            profile=profile,
            proposal=proposal,
            capability_ref=profile.entries[index].capability_ref,
        )
        for index, proposal in enumerate((first, second))
    )
    slate = freeze_experiment_slate(
        profile=profile,
        bindings=bindings,
        budget_snapshot={"training_runs": 1},
    )
    router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=2,
    )

    without_history = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
    )
    with_history = route_frozen_experiment_slate(
        profile=profile,
        slate=slate,
        router=router,
        mechanism_axis_effects={
            first.mechanism_axis: -0.10,
            second.mechanism_axis: 0.05,
        },
    )

    assert without_history.selected_binding == bindings[0]
    assert with_history.selected_binding == bindings[1]
    assert (
        with_history.route_trace.policy_digest
        != without_history.route_trace.policy_digest
    )
