from __future__ import annotations

from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.meta_control import (
    MetaControlError,
    ProposalOnlyShadowMetricsV1,
    activate_promoted_control_policy,
    build_meta_update_proposal,
    decide_meta_control_promotion,
    evaluate_proposal_only_shadow,
    materialize_proposed_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    MetaV17CampaignRuntimeV1,
    meta_v17_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemorySnapshotV1,
)


ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources/"
    "meta_vnext_policy_checkpoint_v17.json"
)


def _memory() -> SearchMemorySnapshotV1:
    return SearchMemorySnapshotV1(
        namespace="DEVELOPMENT_ONLY/SEARCH_MEMORY",
        round_index=5,
        predecessor_digest=sha256_digest({"round": 4}),
        beliefs=(),
        route_trace_digest=sha256_digest({"route": 5}),
        feedback_projection_digest=sha256_digest({"feedback": 5}),
    )


def _runtime() -> MetaV17CampaignRuntimeV1:
    runtime = MetaV17CampaignRuntimeV1(
        checkpoint_path=CHECKPOINT,
        experiment_id="META-CONTROL-TEST",
        search_seed=9214,
        scheduled_rounds=5,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )
    runtime.bind_instances(
        {
            ArmCode.A: "opaque-a",
            ArmCode.B: "opaque-b",
            ArmCode.C: "opaque-c",
        }
    )
    return runtime


def test_v17_control_policy_is_single_content_bound_identity() -> None:
    policy = meta_v17_research_control_policy()
    assert policy.meta_router_policy_digest is not None
    assert policy.meta_router_promotion_decision_digest is not None
    assert policy.promotion_decision_digest is None
    assert policy.activation_boundary == "NEXT_CAMPAIGN"
    assert policy.control_mode == "PROMOTED_META_CONTROL_V1"
    assert dict(policy.producer_token_allocation) == {
        "mechanism_composer": 0.25,
        "lineage_refiner": 0.25,
        "falsification_designer": 0.25,
        "frontier_architect": 0.25,
    }
    assert "regularization" not in policy.mechanism_axis_targeting


def test_producer_directives_are_paired_diverse_and_deterministic() -> None:
    runtime = _runtime()
    b = runtime.producer_directives(
        arm=ArmCode.B,
        round_index=1,
        memory_summary={},
    )
    c = runtime.producer_directives(
        arm=ArmCode.C,
        round_index=1,
        memory_summary={},
    )
    assert [item.to_dict() for item in b] == [
        item.to_dict() for item in c
    ]
    assert len(b) == 4
    assert len({item.primary_axis for item in b}) == 4
    assert len({item.lineage_root for item in b}) == 2
    assert sum(item.proposal_intent == "CONTROL" for item in b) == 1
    assert next(
        item for item in b if item.proposal_intent == "CONTROL"
    ).required_mechanism_id is not None
    assert runtime.producer_directives(
        arm=ArmCode.B,
        round_index=1,
        memory_summary={},
    ) == b


def test_meta_control_lifecycle_promotes_only_paired_improvement() -> None:
    parent = meta_v17_research_control_policy()
    proposal = build_meta_update_proposal(
        policy=parent,
        search_memory=_memory(),
    )
    challenger_policy = materialize_proposed_control_policy(
        parent=parent,
        proposal=proposal,
    )
    champion = ProposalOnlyShadowMetricsV1(
        proposal_count=4,
        common_eligible_count=4,
        unique_semantics_count=2,
        mechanism_axis_count=2,
        lineage_root_count=1,
        control_count=1,
        semantic_collision_count=2,
        billed_tokens=80_000,
    )
    challenger = ProposalOnlyShadowMetricsV1(
        proposal_count=4,
        common_eligible_count=4,
        unique_semantics_count=4,
        mechanism_axis_count=3,
        lineage_root_count=2,
        control_count=1,
        semantic_collision_count=0,
        billed_tokens=80_000,
    )
    evaluation = evaluate_proposal_only_shadow(
        proposal=proposal,
        champion_policy_digest=parent.digest,
        challenger_policy_digest=challenger_policy.digest,
        champion=champion,
        challenger=challenger,
        same_model_prompt_schema_and_contexts=True,
        deterministic_directive_replay=True,
    )
    assert evaluation.verdict == "PASS"
    decision = decide_meta_control_promotion(
        proposal=proposal,
        evaluation=evaluation,
    )
    assert decision.verdict == "PROMOTE"
    activated, receipt = activate_promoted_control_policy(
        parent=parent,
        proposal=proposal,
        decision=decision,
        campaign_id="NEXT-CAMPAIGN-V12",
    )
    assert activated.digest == challenger_policy.digest
    assert receipt.activated_policy_digest == activated.digest
    assert receipt.predecessor_policy_digest == parent.digest


def test_hold_cannot_activate_and_does_not_change_current_policy() -> None:
    parent = meta_v17_research_control_policy()
    proposal = build_meta_update_proposal(
        policy=parent,
        search_memory=_memory(),
    )
    challenger_policy = materialize_proposed_control_policy(
        parent=parent,
        proposal=proposal,
    )
    same = ProposalOnlyShadowMetricsV1(
        proposal_count=4,
        common_eligible_count=4,
        unique_semantics_count=3,
        mechanism_axis_count=3,
        lineage_root_count=2,
        control_count=1,
        semantic_collision_count=1,
        billed_tokens=80_000,
    )
    evaluation = evaluate_proposal_only_shadow(
        proposal=proposal,
        champion_policy_digest=parent.digest,
        challenger_policy_digest=challenger_policy.digest,
        champion=same,
        challenger=same,
        same_model_prompt_schema_and_contexts=True,
        deterministic_directive_replay=True,
    )
    decision = decide_meta_control_promotion(
        proposal=proposal,
        evaluation=evaluation,
    )
    assert decision.verdict == "HOLD"
    assert evaluation.reason_codes == ("NO_UPSTREAM_QUALITY_IMPROVEMENT",)
    with pytest.raises(MetaControlError):
        activate_promoted_control_policy(
            parent=parent,
            proposal=proposal,
            decision=decision,
            campaign_id="FORBIDDEN-CAMPAIGN",
        )
    assert parent == meta_v17_research_control_policy()
