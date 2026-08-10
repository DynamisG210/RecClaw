from __future__ import annotations

from dataclasses import replace

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (
    CurrentProfileExpressibilityV1,
    OpenResearchSpecV1,
)
from recclaw_core.research_line.interfaces import (
    BehaviorProjection,
    ProducerOutcome,
    ResearchContext,
    ResearchLineInterfaceError,
)


def _digest(label: str) -> str:
    return sha256_digest({"label": label})


def _context() -> ResearchContext:
    return ResearchContext(
        campaign_id="campaign-1",
        round_index=1,
        knowledge_base={"mechanisms": ["graph", "ssl"]},
        frozen_goal={"metric": "NDCG@10", "direction": "maximize"},
        frontier={"value": 0.12, "mechanism": "LIGHTGCN"},
        scientific_memory={"by_role": {"frontier_architect": {"avoid": ["dup"]}}},
        unresolved_questions=({"question": "Does depth improve sparse users?"},),
        policy={
            "producer_token_allocation": tuple(
                (role, 0.25) for role in DISCOVERY_PRODUCERS
            ),
            "mechanism_axis_targeting": ("propagation", "objective"),
            "memory_retrieval_policy": "ROLE_SCOPED_PRIOR_ROUND_V1",
        },
        budget={"proposal_calls": 4, "ordinary_experiments": 1},
        active_profile_ref="profile:66",
        active_profile_digest=_digest("profile"),
        protocol_ref="protocol:pilot",
        protocol_digest=_digest("protocol"),
    )


def _spec(context: ResearchContext, role: str) -> OpenResearchSpecV1:
    return OpenResearchSpecV1(
        hypothesis="A mechanism intervention can improve ranking quality.",
        mechanism_change="Change propagation while preserving the evaluator.",
        competing_explanation="The effect may be optimization noise.",
        matched_control_requirement="Use the incumbent under the same protocol when scheduled.",
        implementation_requirements=("candidate-local package",),
        expected_evidence=("full-sort NDCG",),
        falsifier="No improvement under the frozen evaluator.",
        compatibility_requirements=("RecBole",),
        protocol_ref=context.protocol_ref,
        protocol_digest=context.protocol_digest,
        context_ref=context.context_ref,
        context_digest=context.digest,
        current_profile_ref=context.active_profile_ref,
        current_profile_digest=context.active_profile_digest,
        producer_role=role,
        high_change_justification="The current profile lacks this intervention.",
        current_profile_expressibility_claim=(
            CurrentProfileExpressibilityV1.NOT_EXPRESSIBLE
        ),
    )


def test_context_is_shared_but_role_and_policy_change_producer_behavior() -> None:
    context = _context()
    frontier = context.producer_view("frontier_architect")
    composer = context.producer_view("mechanism_composer")

    assert frontier["context_digest"] == composer["context_digest"] == context.digest
    assert frontier["producer_role"] != composer["producer_role"]
    assert frontier["memory"] == {"avoid": ["dup"]}
    assert composer["memory"] == context.to_dict()["scientific_memory"]
    assert frontier["scientific_memory"] == context.to_dict()["scientific_memory"]
    assert frontier["policy"] == context.to_dict()["policy"]
    assert frontier["campaign_id"] == context.campaign_id
    assert frontier["round_index"] == context.round_index
    assert frontier["knowledge_base"] == context.to_dict()["knowledge_base"]
    assert frontier["frozen_goal"] == context.to_dict()["frozen_goal"]

    changed = replace(
        context,
        policy={
            **context.policy,
            "mechanism_axis_targeting": ("sampling",),
        },
    )
    assert changed.digest != context.digest
    assert changed.producer_view("frontier_architect")[
        "mechanism_axis_targeting"
    ] == ["sampling"]


def test_producer_outcome_binds_open_spec_to_context() -> None:
    context = _context()
    spec = _spec(context, "mechanism_composer")
    outcome = ProducerOutcome(
        producer_role="mechanism_composer",
        context_ref=context.context_ref,
        context_digest=context.digest,
        spec=spec,
        resolution_facts={"capability_diff": ["new_model_family"]},
    )
    assert outcome.to_dict()["spec"]["context_digest"] == context.digest

    with pytest.raises(ResearchLineInterfaceError):
        ProducerOutcome(
            producer_role="frontier_architect",
            context_ref=context.context_ref,
            context_digest=context.digest,
            spec=spec,
            resolution_facts={},
        )


def test_behavior_projection_reports_only_real_next_round_input_changes() -> None:
    allocation = tuple((role, 0.25) for role in DISCOVERY_PRODUCERS)
    before = BehaviorProjection(
        round_index=1,
        context_ref="context:r1",
        context_digest=_digest("context-r1"),
        profile_ref="profile:66",
        profile_digest=_digest("profile-66"),
        policy_digest=_digest("policy-1"),
        producer_inputs_digest=_digest("producer-inputs-1"),
        producer_allocation=allocation,
        axis_priorities=("propagation",),
        memory_retrieval_policy="ROLE_SCOPED_PRIOR_ROUND_V1",
        acquisition_parameters={"exploration_weight": 0.5},
        implementation_risk={"new_family": 0.5},
    )
    after = BehaviorProjection(
        round_index=2,
        context_ref="context:r2",
        context_digest=_digest("context-r2"),
        profile_ref="profile:67",
        profile_digest=_digest("profile-67"),
        policy_digest=_digest("policy-2"),
        producer_inputs_digest=_digest("producer-inputs-2"),
        producer_allocation=allocation,
        axis_priorities=("sampling",),
        memory_retrieval_policy="ROLE_SCOPED_NEGATIVE_EVIDENCE_V2",
        acquisition_parameters={"exploration_weight": 0.7},
        implementation_risk={"new_family": 0.2},
    )

    assert before.changed_fields(after) == (
        "acquisition_parameters",
        "axis_priorities",
        "implementation_risk",
        "memory_retrieval_policy",
        "policy_digest",
        "producer_inputs_digest",
        "profile_digest",
        "profile_ref",
    )


def test_behavior_projection_detects_memory_only_producer_input_change() -> None:
    before_context = _context()
    after_context = replace(
        before_context,
        round_index=2,
        scientific_memory={
            **before_context.scientific_memory,
            "global_negative_evidence": {"mechanism": "oversmoothing"},
        },
    )
    allocation = tuple((role, 0.25) for role in DISCOVERY_PRODUCERS)
    common = {
        "profile_ref": "profile:66",
        "profile_digest": _digest("profile-66"),
        "policy_digest": _digest("policy-1"),
        "producer_allocation": allocation,
        "axis_priorities": ("propagation",),
        "memory_retrieval_policy": "ROLE_SCOPED_PRIOR_ROUND_V1",
        "acquisition_parameters": {"exploration_weight": 0.5},
        "implementation_risk": {"new_family": 0.5},
    }
    before = BehaviorProjection(
        round_index=1,
        context_ref=before_context.context_ref,
        context_digest=before_context.digest,
        producer_inputs_digest=before_context.producer_inputs_digest,
        **common,
    )
    after = BehaviorProjection(
        round_index=2,
        context_ref=after_context.context_ref,
        context_digest=after_context.digest,
        producer_inputs_digest=after_context.producer_inputs_digest,
        **common,
    )

    assert before.changed_fields(after) == ("producer_inputs_digest",)
