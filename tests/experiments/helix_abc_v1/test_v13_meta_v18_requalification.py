from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    executable_mechanism,
    executable_mechanisms,
    root_parent_mechanism_id,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ProducerExecutionModeV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext import (
    ResearchContextV1,
    materialize_candidate_pool,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext.v18_support import (
    load_feature_support,
    route_support_aware,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    CHECKPOINT_SHA256_V18,
    POLICY_BUNDLE_DIGEST_V18,
    PROMOTION_DECISION_DIGEST_V18,
    MetaV18CampaignRuntimeV1,
    meta_v18_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    CandidateProposalV4,
    DiscriminativeExperimentPlanV1,
    DiscoveryCreditV1,
    MatchedControlPlanV1,
    ProducerCallRecordV1,
    ProducerSessionResultV1,
    ProposalIntentV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
    / "meta_vnext_policy_checkpoint_v18.json"
)
PROMOTION = (
    ROOT
    / "docs/research_line/v13_requalification"
    / "META_VNEXT_PROMOTION_DECISION_V18.json"
)
ROLES = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)


def _proposal(
    mechanism_id: str,
    role: str,
    index: int,
    utility_override: SearchUtilityFeaturesV1 | None = None,
) -> CandidateProposalV4:
    mechanism = executable_mechanism(mechanism_id)
    candidate_id = f"cand-{index:024x}"
    utility = utility_override or SearchUtilityFeaturesV1(
        runnable_probability=1.0,
        useful_signal=0.62 + 0.01 * (index % 4),
        frontier_potential=0.64 + 0.01 * (index % 3),
        information_gain=0.66 + 0.01 * (index % 2),
        cost=0.6 if len(mechanism.operator_ids) == 2 else 0.3,
        blocker_risk=0.0,
    )
    control = MatchedControlPlanV1(
        mechanism_question_digest=sha256_digest(
            {"mechanism_id": mechanism.mechanism_id}
        ),
        primary_candidate_id=candidate_id,
        comparator_candidate_id=None,
        comparator_program_digest=None,
        protocol_digest="a" * 64,
        changed_axis=mechanism.mechanism_axis,
        plan_status="QUEUE_MATCHED_CONTROL",
    )
    discriminative = (
        DiscriminativeExperimentPlanV1(
            competing_hypotheses=(
                "the mechanism changes ranking quality",
                "the apparent change is a matched-control fluctuation",
            ),
            predicted_outcome_signature="positive matched comparator delta",
            primary_candidate=candidate_id,
            matched_control_plan=control,
            falsifier="non-positive matched comparator delta",
            next_decision_rule="queue the exact root control",
        )
        if role == "falsification_designer"
        else None
    )
    return CandidateProposalV4(
        candidate_id=candidate_id,
        producer_id=f"producer-{role}",
        producer_role=role,
        proposal_intent=(
            ProposalIntentV1.FALSIFICATION
            if role == "falsification_designer"
            else ProposalIntentV1.DISCOVERY
        ),
        discovery_credit=DiscoveryCreditV1.DISCOVERY,
        mechanism_id=mechanism.mechanism_id,
        mechanism_axis=mechanism.mechanism_axis,
        mechanism_program=mechanism.mechanism_program,
        candidate_label=mechanism.mechanism_id,
        mechanism_hypothesis="the typed operator causes the outcome signature",
        competing_hypothesis="the matched root explains the outcome",
        predicted_outcome_signature="positive matched comparator delta",
        failure_mode="the signature is absent",
        utility_features=utility,
        feature_evidence=RouterFeatureEvidenceV1(
            compile_valid=True,
            handler_available=True,
            materializer_available=True,
            blocker_rate=0.0,
            semantic_duplicate=False,
            parent_available=True,
            mechanism_depth=len(mechanism.operator_ids),
            estimated_cost=utility.cost,
            llm_diagnostic=utility,
        ),
        matched_control_plan=control,
        discriminative_plan=discriminative,
        parent_candidate_id=None,
        assigned_before_call=True,
        post_hoc_relabel=False,
    )


def _pool(proposals: tuple[CandidateProposalV4, ...]):
    parents = {
        item.candidate_id: executable_mechanism(
            root_parent_mechanism_id(item.mechanism_id)
        ).mechanism_program
        for item in proposals
    }
    return materialize_candidate_pool(
        pool_id="v18-requalification-pool",
        proposals=proposals,
        parent_programs=parents,
        research_context=ResearchContextV1(
            round_fraction=0.0,
            remaining_execution_fraction=1.0,
            remaining_token_fraction=1.0,
            remaining_gpu_fraction=1.0,
            starting_frontier=0.0,
            recent_frontier_gain=0.0,
            stagnation_fraction=0.0,
            axis_coverage=tuple(
                (axis, 0.0)
                for axis in (
                    "architecture",
                    "geometry",
                    "message_transform",
                    "objective",
                    "propagation",
                    "sampling",
                    "self_supervision",
                )
            ),
            exact_duplicate_count=0,
            near_duplicate_count=0,
            blocker_count=0,
            lineage_depth=0,
            task_scale=1.0,
            task_density=0.2843119865332499,
        ),
        producer_invocation_digests=tuple(
            hashlib.sha256(f"call-{index}".encode()).hexdigest()
            for index in range(4)
        ),
        pre_round_state_digest="b" * 64,
        candidate_order_policy_digest="c" * 64,
        static_router=StrongStaticRouterV1(slate_ceiling=len(proposals)),
        policy_projection=meta_v18_research_control_policy().to_dict(),
    )


def _session(
    proposals: tuple[CandidateProposalV4, ...],
    round_index: int,
) -> ProducerSessionResultV1:
    calls = tuple(
        ProducerCallRecordV1(
            session_id=f"session-{round_index}",
            mode=(
                ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
            ),
            physical_call_id=f"call-{round_index}-{index}",
            producer_id=item.producer_id,
            producer_role=item.producer_role,
            request_digest=sha256_digest(
                {"round": round_index, "index": index, "kind": "request"}
            ),
            response_digest=sha256_digest(
                {"round": round_index, "index": index, "kind": "response"}
            ),
            context_digest=sha256_digest({"round": round_index}),
            memory_digest=sha256_digest({"memory": round_index}),
            prompt_digest=sha256_digest({"prompt": index}),
            rng_digest=sha256_digest({"rng": index}),
            candidate_ids=(item.candidate_id,),
            input_tokens=100,
            output_tokens=50,
            billed_tokens=150,
            latency_ms=10,
        )
        for index, item in enumerate(proposals)
    )
    return ProducerSessionResultV1(
        session_id=f"session-{round_index}",
        mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
        calls=calls,
        proposals=proposals,
        total_resource_envelope_digest="d" * 64,
        base_model_ref="fixture",
        bl_projection_digest="e" * 64,
        candidate_schema_ref="CandidateProposalV4",
        proposal_count=len(proposals),
        physical_call_count=len(calls),
        input_tokens=400,
        output_tokens=200,
        billed_tokens=600,
        session_latency_ms=10,
    )


def test_v18_checkpoint_binds_parent_profile_schema_and_no_pilot_outcomes() -> None:
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == (
        CHECKPOINT_SHA256_V18
    )
    checkpoint = json.loads(CHECKPOINT.read_text())
    assert checkpoint["policy_bundle_digest"] == POLICY_BUNDLE_DIGEST_V18
    assert checkpoint["candidate_contract"] == "CandidateProposalV4"
    assert checkpoint["executable_profile_id"] == "BL_ICF_EXECUTABLE_PROFILE_V2"
    assert checkpoint["pilot_outcomes_used"] is False
    assert checkpoint["shadow_only"] is False
    forbidden = (
        "claim_ceiling",
        "evidence_admission",
        "protocol_branch",
        "evidence_use",
        "guard_event",
    )
    serialized = CHECKPOINT.read_text().lower()
    assert all(name not in serialized for name in forbidden)
    assert sha256_digest(json.loads(PROMOTION.read_text())) == (
        PROMOTION_DECISION_DIGEST_V18
    )


def test_all_64_discovery_candidates_are_scored_without_feature_extrapolation() -> None:
    profile = executable_mechanisms()
    assert len(profile) == 66
    mechanisms = tuple(
        item for item in profile if item.parent_mechanism_id is not None
    )
    proposals = tuple(
        _proposal(item.mechanism_id, ROLES[index % 4], index + 1)
        for index, item in enumerate(mechanisms)
    )
    pool = _pool(proposals)
    runtime = MetaV18CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id="v18-requalification",
        search_seed=9301,
        scheduled_rounds=5,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    runtime.bind_instances(
        {ArmCode.A: "a", ArmCode.B: "b", ArmCode.C: "c"}
    )
    decision, support = route_support_aware(
        pool=pool,
        policy=runtime.policy,
        router=runtime.router,
        fast_state=runtime._states[ArmCode.B].fast_state,
        actual_task_scale=1.0,
        actual_task_density=0.2843119865332499,
    )
    assert len(pool.eligible_candidates) == 64
    assert len(decision.scored_candidates) == 64
    assert support.task_context_supported is False
    assert support.fast_supported is False
    assert support.fast_support_reason == "ACTUAL_TASK_CONTEXT_OUT_OF_SUPPORT"
    assert set(support.generic_transfer_axes) == {
        "architecture",
        "geometry",
        "sampling",
    }
    assert support.clipped_feature_count > 0
    assert all(
        math.isfinite(item.final_score)
        for item in decision.scored_candidates
    )
    assert decision.selected_candidate_id in {
        item.candidate_id for item in proposals
    }


def test_v18_b_c_share_slow_checkpoint_but_fast_state_is_private() -> None:
    runtime = MetaV18CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id="v18-bc-identity",
        search_seed=9301,
        scheduled_rounds=5,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    runtime.bind_instances(
        {ArmCode.A: "a", ArmCode.B: "b", ArmCode.C: "c"}
    )
    assert runtime.initial_semantic_state_projection(
        ArmCode.B
    ) == runtime.initial_semantic_state_projection(ArmCode.C)
    assert runtime._states[ArmCode.B].fast_state.digest != (
        runtime._states[ArmCode.C].fast_state.digest
    )
    assert runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V18


def test_v18_directives_keep_four_discovery_roles_and_rotate_uncovered_axes() -> None:
    runtime = MetaV18CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id="v18-directives",
        search_seed=9301,
        scheduled_rounds=5,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    runtime.bind_instances(
        {ArmCode.A: "a", ArmCode.B: "b", ArmCode.C: "c"}
    )
    proposals = tuple(
        _proposal(
            mechanism_id,
            role,
            index + 1,
        )
        for index, (mechanism_id, role) in enumerate(
            (
                ("BPR_MF__BPR_MARGIN", "mechanism_composer"),
                ("BPR_MF__BPR_NORM_CONSTRAINT", "lineage_refiner"),
                ("LIGHTGCN__LGCN_RANK_AWARE", "falsification_designer"),
                ("LIGHTGCN__LGCN_SHALLOW", "frontier_architect"),
            )
        )
    )
    parents = {
        item.candidate_id: executable_mechanism(
            root_parent_mechanism_id(item.mechanism_id)
        ).mechanism_program
        for item in proposals
    }
    directed_axis_sets = []
    for round_index in range(1, 4):
        directives = runtime.producer_directives(
            arm=ArmCode.B,
            round_index=round_index,
            memory_summary={},
        )
        assert {item.producer_role for item in directives} == set(ROLES)
        falsification = next(
            item
            for item in directives
            if item.producer_role == "falsification_designer"
        )
        assert falsification.proposal_intent == "FALSIFICATION"
        assert falsification.required_mechanism_id is None
        directed_axis_sets.append(
            {item.primary_axis for item in directives}
        )
        route = runtime.route_session(
            arm=ArmCode.B,
            round_index=round_index,
            session=_session(proposals, round_index),
            exact_parent_programs=parents,
            static_router=StrongStaticRouterV1(slate_ceiling=4),
            research_policy=meta_v18_research_control_policy(),
        )
        runtime.record_observation(
            arm=ArmCode.B,
            round_index=round_index,
            candidate_id=route.selected_candidate_id,
            run_status="SUCCESS",
            ndcg=0.10 + 0.01 * round_index,
            wall_time_ms=100,
            source_search_utility_event_digest=sha256_digest(
                {"round": round_index, "feedback": "search-only"}
            ),
        )
    assert all(len(item) >= 3 for item in directed_axis_sets)
    assert len(set().union(*directed_axis_sets)) >= 4
    assert len(runtime._states[ArmCode.B].fast_state.observed_responses) == 0
    assert runtime._states[ArmCode.B].fast_state.round_boundary == 3


def test_v18_support_resource_uses_only_search_side_features() -> None:
    support = load_feature_support()
    names = {str(row[0]) for row in support["feature_bounds"]}
    forbidden_tokens = {
        "claim",
        "admission",
        "protocol_branch",
        "evidence",
        "guard",
    }
    assert not {
        name
        for name in names
        if any(token in name.lower() for token in forbidden_tokens)
    }


def test_v18_router_responds_to_real_utility_without_axis_or_producer_collapse() -> None:
    representatives = {
        "architecture": "LIGHTGCN__LGCN_DUAL_PATH",
        "geometry": "BPR_MF__BPR_NORM_CONSTRAINT",
        "message_transform": "LIGHTGCN__LGCN_RESIDUAL",
        "objective": "BPR_MF__BPR_MARGIN",
        "propagation": "LIGHTGCN__LGCN_SHALLOW",
        "sampling": "BPR_MF__BPR_MIXED_NEGATIVE",
        "self_supervision": "LIGHTGCN__LGCN_AUX_ALIGNMENT",
    }
    runtime = MetaV18CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id="v18-non-collapse",
        search_seed=9301,
        scheduled_rounds=5,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    runtime.bind_instances(
        {ArmCode.A: "a", ArmCode.B: "b", ArmCode.C: "c"}
    )
    selected_axes = []
    selected_roles = []
    axis_items = tuple(representatives.items())
    for target_index, (target_axis, _) in enumerate(axis_items):
        ordered = (
            axis_items[target_index],
            axis_items[(target_index + 1) % len(axis_items)],
            axis_items[(target_index + 2) % len(axis_items)],
            axis_items[(target_index + 3) % len(axis_items)],
        )
        proposals = []
        for index, (axis, mechanism_id) in enumerate(ordered):
            high = axis == target_axis
            utility = SearchUtilityFeaturesV1(
                runnable_probability=0.98 if high else 0.87,
                useful_signal=0.75 if high else 0.50,
                frontier_potential=0.78 if high else 0.45,
                information_gain=0.72 if high else 0.40,
                cost=0.20 if high else 0.65,
                blocker_risk=0.03 if high else 0.10,
            )
            proposals.append(
                _proposal(
                    mechanism_id,
                    ROLES[(index + target_index) % len(ROLES)],
                    index + 1,
                    utility_override=utility,
                )
            )
        pool = _pool(tuple(proposals))
        decision, _ = route_support_aware(
            pool=pool,
            policy=runtime.policy,
            router=runtime.router,
            fast_state=runtime._states[ArmCode.B].fast_state,
            actual_task_scale=1.0,
            actual_task_density=0.2843119865332499,
        )
        selected = next(
            item
            for item in proposals
            if item.candidate_id == decision.selected_candidate_id
        )
        selected_axes.append(selected.mechanism_axis)
        selected_roles.append(selected.producer_role)
    assert len(set(selected_axes)) >= 5
    assert max(selected_roles.count(role) for role in ROLES) <= 4
