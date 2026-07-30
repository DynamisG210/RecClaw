from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

import recclaw_core.experiments.helix_abc_v1 as public_api
from recclaw_core.experiments.helix_abc_v1 import (
    NEXT_FRESH_CAMPAIGN,
    NEXT_ROUND,
    RESEARCH_STATIC_VNEXT,
    CapabilityIdentityV1,
    ExperimentBudgetV1,
    ExperimentPolicyInputV1,
    IdeaBudgetV1,
    IdeaCandidateIdentityV1,
    IdeaPolicyInputV1,
    IdeaRouteV1,
    OpenMetaReplayDatasetV1,
    PolicySupportStatusV1,
    ResearchFailureClassV1,
    SearchAdapterError,
    SearchProfileActivationV1,
    acquire_candidate_idea,
    acquire_open_idea,
    activate_next_fresh_search_profile,
    adapt_current_search_profile,
    bind_search_candidate,
    build_open_meta_replay_record,
    freeze_experiment_slate,
    project_episode_to_mechanism_belief,
    route_frozen_experiment_slate,
    run_static_experiment_policy,
    run_static_idea_policy,
    schedule_static_policy_activation,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    bl_icf_executable_profile_v2,
    executable_mechanisms,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.open_meta import (
    __all__ as OPEN_META_PUBLIC_NAMES,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (
    frozen_search_bindings,
    frozen_search_resolver_environment,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    __all__ as SEARCH_ADAPTER_PUBLIC_NAMES,
)

from open_meta_fixtures import (
    current_experiment_policy,
    current_idea_policy,
)
from scientific_episode_fixtures import (
    canonical_closure_fixtures,
    comparison_identity,
    research_episode,
)
from test_e0_search_adapter import (
    A0_FIXTURE_PATH,
    E0_FIXTURE_PATH,
    _digest,
    _fixed_proposals,
    _fixture,
    _next_profile,
    _outside_66_program,
    _proposal,
    _qualified_capability,
)


ROOT = Path(__file__).resolve().parents[3]
GATE_RECEIPT = (
    ROOT
    / "docs"
    / "research_line"
    / "vnext"
    / "WAVE2_INTEGRATED_GATE_RECEIPT.json"
)


def _experiment_policy_input(
    *,
    profile: Any,
    slate: Any,
) -> ExperimentPolicyInputV1:
    capabilities = tuple(
        CapabilityIdentityV1(
            capability_ref=binding.capability_ref,
            capability_digest=binding.capability_digest,
        )
        for binding in slate.bindings
    )
    return ExperimentPolicyInputV1(
        research_context_ref=f"context:{profile.campaign_id}",
        research_context_digest=_digest(f"context:{profile.campaign_id}"),
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
        current_profile_ref=profile.profile_ref,
        current_profile_digest=profile.profile_digest,
        frozen_slate_ref=slate.slate_id,
        frozen_slate_digest=slate.digest,
        frozen_slate_capabilities=capabilities,
        acquisition_subjects=capabilities,
        budget=ExperimentBudgetV1(
            experiment_slots=len(capabilities),
        ),
        historical_episode_summaries=(),
    )


def _idea_policy_input(
    *,
    profile: Any,
    acquisition: Any,
    high_change: bool,
    current_profile_expressible: bool,
) -> IdeaPolicyInputV1:
    candidate = IdeaCandidateIdentityV1(
        research_spec_ref=acquisition.spec.spec_id,
        research_spec_digest=acquisition.spec.digest,
        direction_ref=f"direction:{acquisition.spec.spec_id}",
        direction_digest=sha256_digest(
            {"direction": acquisition.spec.spec_id}
        ),
        high_change=high_change,
        current_profile_expressible=current_profile_expressible,
    )
    return IdeaPolicyInputV1(
        research_context_ref=f"context:{profile.campaign_id}",
        research_context_digest=_digest(f"context:{profile.campaign_id}"),
        protocol_ref=profile.protocol_ref,
        protocol_digest=profile.protocol_digest,
        current_profile_ref=profile.profile_ref,
        current_profile_digest=profile.profile_digest,
        candidates=(candidate,),
        budget=IdeaBudgetV1(
            ideation_slots=1,
            implementation_slots=1,
            qualification_slots=1,
        ),
        historical_episode_summaries=(),
    )


def _build_wave2_evidence() -> tuple[dict[str, Any], OpenMetaReplayDatasetV1]:
    fixture = _fixture(E0_FIXTURE_PATH)
    source_profile_before = canonical_json_bytes(
        bl_icf_executable_profile_v2()
    )
    current = adapt_current_search_profile(
        campaign_id=fixture["campaigns"]["current"]
    )
    current_before = current.canonical_bytes()
    current_proposal = _fixed_proposals(
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
        proposal=current_proposal,
        capability_ref=fixed_entry.capability_ref,
    )
    current_slate = freeze_experiment_slate(
        profile=current,
        bindings=(current_binding,),
        budget_snapshot=fixture["experiment_budget"],
    )
    current_slate_before = current_slate.canonical_bytes()

    bindings = frozen_search_bindings(
        context_ref="context:wave2-complete-idea",
        context_digest=_digest("wave2-complete-idea"),
    )
    environment = frozen_search_resolver_environment(
        available_dependencies=("recbole-runtime",),
        budget_limits={
            "implementation_tokens": 5000,
            "implementation_units": 2,
        },
    )
    current_acquisition = acquire_candidate_idea(
        current_proposal,
        bindings=bindings,
        environment=environment,
        required_budget={"implementation_tokens": 100},
    )
    innovation_draft = copy.deepcopy(
        _fixture(A0_FIXTURE_PATH)["producer_drafts"][0]
    )
    innovation_acquisition = acquire_open_idea(
        innovation_draft,
        bindings=bindings,
        environment=environment,
    )
    f0_innovation = run_static_idea_policy(
        _idea_policy_input(
            profile=current,
            acquisition=innovation_acquisition,
            high_change=True,
            current_profile_expressible=False,
        )
    )
    f0_current_search = run_static_idea_policy(
        _idea_policy_input(
            profile=current,
            acquisition=current_acquisition,
            high_change=False,
            current_profile_expressible=True,
        )
    )

    current_experiment = run_static_experiment_policy(
        _experiment_policy_input(
            profile=current,
            slate=current_slate,
        )
    )
    current_route = route_frozen_experiment_slate(
        profile=current,
        slate=current_slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=1,
        ),
        policy_projection={
            "f0_decision_ref": current_experiment.decision_id,
            "f0_decision_digest": current_experiment.digest,
        },
    )

    outside_program = _outside_66_program()
    capability = _qualified_capability(
        profile=current,
        program=outside_program,
    )
    registry, next_profile, profile_receipt = _next_profile(
        current_profile=current,
        current_slate=current_slate,
        capability=capability,
    )
    unqualified_proposal = _proposal(
        candidate_id="cand-wave2-unqualified-current",
        mechanism_id="NGCF_UNQUALIFIED_OPEN",
        mechanism_axis="message_transform",
        mechanism_program=outside_program,
        protocol_digest=current.protocol_digest,
        selected=True,
    )
    with pytest.raises(SearchAdapterError):
        bind_search_candidate(
            profile=current,
            proposal=unqualified_proposal,
            capability_ref=capability.capability_id,
        )

    activated = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=fixture["campaigns"]["next_fresh"],
    )
    qualified_proposal = _proposal(
        candidate_id="cand-wave2-qualified-next",
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
    next_slate = freeze_experiment_slate(
        profile=activated,
        bindings=(qualified_binding,),
        budget_snapshot=fixture["experiment_budget"],
    )
    next_experiment = run_static_experiment_policy(
        _experiment_policy_input(
            profile=activated,
            slate=next_slate,
        )
    )
    next_route = route_frozen_experiment_slate(
        profile=activated,
        slate=next_slate,
        router=StrongStaticRouterV1(
            runnable_floor=0.0,
            utility_floor=0.0,
            blocker_ceiling=1.0,
            cost_ceiling=1.0,
            slate_ceiling=1,
        ),
        policy_projection={
            "f0_decision_ref": next_experiment.decision_id,
            "f0_decision_digest": next_experiment.digest,
        },
    )

    idea_activation = schedule_static_policy_activation(
        f0_innovation,
        current_policy=current_idea_policy(),
    )
    experiment_activation = schedule_static_policy_activation(
        next_experiment,
        current_policy=current_experiment_policy(),
    )
    replay = OpenMetaReplayDatasetV1(
        dataset_version="1.0.0",
        records=(
            build_open_meta_replay_record(
                decision=f0_innovation,
                activation=idea_activation,
            ),
            build_open_meta_replay_record(
                decision=next_experiment,
                activation=experiment_activation,
            ),
        ),
    )

    identity = comparison_identity()
    d1_belief = project_episode_to_mechanism_belief(
        comparison_identity=identity,
        closure=canonical_closure_fixtures()["success"],
        episode=research_episode(identity),
        mechanism_axis="propagation",
    )
    rejected_closures = tuple(
        sorted(
            name
            for name in canonical_closure_fixtures()
            if name not in {"success", "mechanism_negative"}
        )
    )

    assert current_acquisition.route is IdeaRouteV1.SEARCH
    assert innovation_acquisition.route is IdeaRouteV1.INNOVATION
    assert current_acquisition.search_proposal is current_proposal
    assert innovation_acquisition.search_proposal is None
    assert f0_innovation.support_status is PolicySupportStatusV1.IN_SUPPORT
    assert (
        f0_current_search.support_status
        is PolicySupportStatusV1.OUT_OF_SUPPORT
    )
    assert f0_current_search.budget_allocations == ()
    assert current_route.selected_binding is current_binding
    assert next_route.selected_binding is qualified_binding
    assert qualified_binding.executable_entrypoint == (
        capability.executable_entrypoint
    )
    assert len(current.entries) == 66
    assert len(activated.entries) == 67
    assert current.canonical_bytes() == current_before
    assert current_slate.canonical_bytes() == current_slate_before
    assert (
        canonical_json_bytes(bl_icf_executable_profile_v2())
        == source_profile_before
    )
    assert profile_receipt.current_profile_unchanged is True
    assert (
        activated.activation
        is SearchProfileActivationV1.NEXT_FRESH_CAMPAIGN
    )
    assert f0_innovation.policy_mode == RESEARCH_STATIC_VNEXT
    assert next_experiment.policy_mode == RESEARCH_STATIC_VNEXT
    assert idea_activation.activation_boundary == NEXT_FRESH_CAMPAIGN
    assert experiment_activation.activation_boundary == NEXT_ROUND
    assert idea_activation.promotion_authorized is False
    assert experiment_activation.promotion_authorized is False

    evidence = {
        "current_profile": {
            "canonical_bytes_sha256": sha256_digest(
                json.loads(current_before)
            ),
            "entry_count": len(current.entries),
            "profile_digest": current.profile_digest,
            "profile_ref": current.profile_ref,
            "source_canonical_bytes_sha256": bytes_sha256(
                source_profile_before
            ),
        },
        "current_route": {
            "experiment_decision_digest": current_experiment.digest,
            "route_trace_digest": current_route.route_trace.digest,
            "selected_binding_digest": current_binding.digest,
            "slate_canonical_bytes_sha256": bytes_sha256(
                current_slate_before
            ),
            "slate_digest": current_slate.digest,
        },
        "d1": {
            "accepted_belief_digest": sha256_digest(d1_belief.to_dict()),
            "rejected_closure_matrix_digest": sha256_digest(
                rejected_closures
            ),
        },
        "f0": {
            "experiment_activation_digest": experiment_activation.digest,
            "idea_activation_digest": idea_activation.digest,
            "idea_decision_digest": f0_innovation.digest,
            "idea_oos_decision_digest": f0_current_search.digest,
            "policy_mode": RESEARCH_STATIC_VNEXT,
            "replay_dataset_digest": replay.digest,
        },
        "next_fresh_profile": {
            "entry_count": len(activated.entries),
            "experiment_decision_digest": next_experiment.digest,
            "profile_digest": activated.profile_digest,
            "profile_ref": activated.profile_ref,
            "qualified_entrypoint": qualified_binding.executable_entrypoint,
            "route_trace_digest": next_route.route_trace.digest,
            "selected_binding_digest": qualified_binding.digest,
        },
        "isolation": {
            "current_profile_bytes_unchanged": True,
            "current_slate_bytes_unchanged": True,
            "idea_experiment_schema_separate": True,
            "unqualified_current_binding_blocked": True,
        },
    }
    return evidence, replay


def test_complete_wave2_evidence_matches_canonical_gate_receipt(
    tmp_path: Path,
) -> None:
    evidence, replay = _build_wave2_evidence()
    receipt_bytes = GATE_RECEIPT.read_bytes()
    receipt = json.loads(receipt_bytes)
    replay_path = tmp_path / "wave2-replay.json"
    public_api.write_open_meta_replay_dataset(replay_path, replay)

    assert receipt_bytes == canonical_json_bytes(receipt) + b"\n"
    assert receipt["evidence"] == evidence
    assert public_api.read_open_meta_replay_dataset(replay_path) == replay
    assert receipt["authority"] == {
        "evidence_class": "LOCAL_ENGINEERING_ONLY",
        "promotion_authorized": False,
        "scientific_gate_authorized": False,
    }
    assert receipt["execution"] == {
        "held_out_accesses": 0,
        "outcomes_consumed": 0,
        "provider_calls": 0,
        "r1_r2_started": False,
        "wave1_one_epoch_repeated": False,
    }


def test_complete_wave2_public_surface_is_owner_declared() -> None:
    assert set(SEARCH_ADAPTER_PUBLIC_NAMES) <= set(public_api.__all__)
    assert set(OPEN_META_PUBLIC_NAMES) <= set(public_api.__all__)
