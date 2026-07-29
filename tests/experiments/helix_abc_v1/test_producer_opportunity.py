from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import json

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.contracts import (
    ArmCode,
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    MetaV20CampaignRuntimeV1,
    POLICY_BUNDLE_DIGEST_V20,
    meta_v20_research_control_policy,
)
from recclaw_core.experiments.helix_abc_v1.producer_opportunity import (
    PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1,
    ProducerOpportunityError,
    acquire_producer_opportunity,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    DISCOVERY_PRODUCERS,
    FixtureProducerBrokerV1,
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    ProposalIntentV1,
)


ROLE_BY_CANDIDATE = {
    f"candidate-{role}": role for role in DISCOVERY_PRODUCERS
}
PARENT_SCORE_ORDER = (
    "candidate-falsification_designer",
    "candidate-mechanism_composer",
    "candidate-frontier_architect",
    "candidate-lineage_refiner",
)
V20_CHECKPOINT = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
    / "meta_vnext_policy_checkpoint_v20.json"
)
ANCHOR_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "bl_icf_anchor_programs_v1.json"
)


def _producer_session(round_index: int):
    fixtures = json.loads(
        ANCHOR_FIXTURES.read_text(encoding="utf-8")
    )["fixtures"]
    by_name = {
        item["anchor_name"]: item["program"] for item in fixtures
    }
    drafts = []
    for role, anchor, axis in zip(
        DISCOVERY_PRODUCERS,
        ("BPR_MF", "LIGHTGCN", "SGL", "NGCF"),
        ("objective", "propagation", "self_supervision", "architecture"),
        strict=True,
    ):
        drafts.append(
            {
                "mechanism_program": copy.deepcopy(by_name[anchor]),
                "mechanism_axis": axis,
                "proposal_intent": (
                    ProposalIntentV1.FALSIFICATION.value
                    if role == "falsification_designer"
                    else ProposalIntentV1.DISCOVERY.value
                ),
                "utility_features": {
                    "runnable_probability": 0.9,
                    "useful_signal": 0.8,
                    "frontier_potential": 0.8,
                    "information_gain": 0.8,
                    "cost": 0.3,
                    "blocker_risk": 0.1,
                },
            }
        )
    return FixtureProducerBrokerV1().dispatch(
        session_id=f"meta-v20-opportunity-{round_index}",
        mode=(
            ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
        ),
        drafts=drafts,
        context={"round_index": round_index, "search_seed": 9400},
        role_memory={
            role: {"prior": sha256_digest({"role": role})}
            for role in DISCOVERY_PRODUCERS
        },
        seed=9400,
        ceilings=ResourceCeilingsV1(
            total_input_tokens=60_000,
            total_output_tokens=20_000,
            total_billed_token_debit=80_000,
            total_proposal_count=4,
            wall_time_ms=1_500_000,
            retry_debit=0,
            proposal_attempt_debit=4,
            ordinary_executions=1,
            common_validation_count=4,
            gpu_device_time_ms=1_200_000,
            gpu_cost_microunits=340_000,
        ),
    )


def test_first_four_opportunities_cover_every_role_then_exploit_score() -> None:
    history: list[str] = []
    decisions = []
    for _ in range(8):
        decision = acquire_producer_opportunity(
            parent_ranked_candidate_ids=PARENT_SCORE_ORDER,
            parent_decision_digest="parent-decision",
            producer_role_by_candidate_id=ROLE_BY_CANDIDATE,
            prior_selected_roles=history,
        )
        decisions.append(decision)
        history.append(decision.selected_producer_role)
    assert set(history[:4]) == set(DISCOVERY_PRODUCERS)
    assert history[4:] == ["falsification_designer"] * 4
    assert all(
        item.policy_digest == PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
        for item in decisions
    )
    assert decisions[0].selection_reason == "BLOCK_ROLE_COVERAGE"
    assert decisions[4].selection_reason == "PARENT_META_SCORE"


def test_each_eight_opportunity_block_reopens_role_coverage() -> None:
    history: list[str] = []
    for _ in range(16):
        decision = acquire_producer_opportunity(
            parent_ranked_candidate_ids=PARENT_SCORE_ORDER,
            parent_decision_digest="parent-decision",
            producer_role_by_candidate_id=ROLE_BY_CANDIDATE,
            prior_selected_roles=history,
        )
        history.append(decision.selected_producer_role)
    assert set(history[:4]) == set(DISCOVERY_PRODUCERS)
    assert set(history[8:12]) == set(DISCOVERY_PRODUCERS)
    assert history.count("falsification_designer") == 10
    assert all(
        history.count(role) == 2
        for role in DISCOVERY_PRODUCERS
        if role != "falsification_designer"
    )


def test_missing_role_is_not_invented_and_score_order_is_stable() -> None:
    roles = {
        candidate_id: role
        for candidate_id, role in ROLE_BY_CANDIDATE.items()
        if role != "lineage_refiner"
    }
    ranked = tuple(
        candidate_id
        for candidate_id in PARENT_SCORE_ORDER
        if candidate_id in roles
    )
    history: list[str] = []
    for _ in range(4):
        decision = acquire_producer_opportunity(
            parent_ranked_candidate_ids=ranked,
            parent_decision_digest="parent-decision",
            producer_role_by_candidate_id=roles,
            prior_selected_roles=history,
        )
        history.append(decision.selected_producer_role)
    assert "lineage_refiner" not in history
    assert set(history[:3]) == set(roles.values())
    assert history[3] == "falsification_designer"


def test_acquisition_rejects_partial_or_unknown_role_identity() -> None:
    with pytest.raises(ProducerOpportunityError):
        acquire_producer_opportunity(
            parent_ranked_candidate_ids=PARENT_SCORE_ORDER,
            parent_decision_digest="parent-decision",
            producer_role_by_candidate_id={
                PARENT_SCORE_ORDER[0]: "unknown"
            },
            prior_selected_roles=(),
        )


def test_v20_checkpoint_and_arm_private_acquisition_are_bound() -> None:
    runtime = MetaV20CampaignRuntimeV1(
        checkpoint_path=V20_CHECKPOINT,
        experiment_id="META-V20-OPPORTUNITY-UNIT",
        search_seed=9400,
        scheduled_rounds=50,
        task_scale=1.0,
        task_density=1.0,
    )
    runtime.bind_instances(
        {
            ArmCode.A: "opaque-a",
            ArmCode.B: "opaque-b",
            ArmCode.C: "opaque-c",
        }
    )
    proposals = {
        candidate_id: SimpleNamespace(producer_role=role)
        for candidate_id, role in ROLE_BY_CANDIDATE.items()
    }
    for round_index in range(1, 9):
        runtime._acquire_candidate_order(
            arm=ArmCode.B,
            round_index=round_index,
            parent_ranked_candidate_ids=PARENT_SCORE_ORDER,
            proposal_by_id=proposals,
            parent_decision_digest=f"parent-{round_index}",
        )
    audit = runtime.audit_projection()
    selected = [
        item["decision"]["selected_producer_role"]
        for item in audit["producer_opportunity_decisions"]
    ]
    assert set(selected[:4]) == set(DISCOVERY_PRODUCERS)
    assert selected[4:] == ["falsification_designer"] * 4
    assert runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V20
    assert audit["development_activation_not_promotion"] is True
    assert runtime.arm_private_context_digest(ArmCode.B) != (
        runtime.arm_private_context_digest(ArmCode.C)
    )


def test_v20_full_route_session_uses_acquisition_before_boundary() -> None:
    runtime = MetaV20CampaignRuntimeV1(
        checkpoint_path=V20_CHECKPOINT,
        experiment_id="META-V20-FULL-ROUTE-UNIT",
        search_seed=9400,
        scheduled_rounds=8,
        task_scale=1.0,
        task_density=1.0,
    )
    runtime.bind_instances(
        {
            ArmCode.A: "opaque-a",
            ArmCode.B: "opaque-b",
            ArmCode.C: "opaque-c",
        }
    )
    selected_roles = []
    for round_index in range(1, 9):
        session = _producer_session(round_index)
        route = runtime.route_session(
            arm=ArmCode.B,
            round_index=round_index,
            session=session,
            static_router=StrongStaticRouterV1(),
            research_policy=meta_v20_research_control_policy(),
        )
        proposal = next(
            item
            for item in session.proposals
            if item.candidate_id == route.selected_candidate_id
        )
        selected_roles.append(proposal.producer_role)
        runtime.record_round_boundary(
            arm=ArmCode.B,
            round_index=round_index,
            proposal_source="NORMAL_ROUTED_PROPOSAL",
            observation_path="NO_OBSERVATION",
            candidate_id=route.selected_candidate_id,
            runtime_candidate_id=None,
            run_status="COMPLETED",
            ndcg=None,
            wall_time_ms=0,
            source_search_utility_event_digest=sha256_digest(
                {"round_index": round_index}
            ),
        )
    assert set(selected_roles[:4]) == set(DISCOVERY_PRODUCERS)
    assert len(runtime.audit_projection()[
        "producer_opportunity_decisions"
    ]) == 8
