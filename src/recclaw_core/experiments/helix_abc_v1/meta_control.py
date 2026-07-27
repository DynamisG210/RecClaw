"""Minimal, content-bound lifecycle for the Research Meta control plane.

The lifecycle owns search-policy proposals and their development qualification.
It has no Evidence Authority and cannot activate a policy inside the campaign
that produced its Search Memory input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .canonical import canonical_value, sha256_digest, validate_sha256
from .contracts import validate_no_research_evidence_authority_fields
from .research_capability import (
    SearchMemorySnapshotV1,
    VersionedResearchPolicyV1,
)
from .research_contracts import DISCOVERY_PRODUCERS


CANONICAL_CONTROL_AXES_V1 = (
    "architecture",
    "geometry",
    "message_transform",
    "objective",
    "propagation",
    "sampling",
    "self_supervision",
)


class MetaControlError(ValueError):
    """Raised when a Meta control proposal crosses its frozen boundary."""


@dataclass(frozen=True, slots=True)
class MetaControlUpdateProposalV1:
    proposal_id: str
    parent_policy_digest: str
    source_search_memory_digest: str
    source_round_index: int
    proposed_axis_priority: tuple[str, ...]
    proposed_memory_retrieval_policy: str
    proposed_producer_token_allocation: tuple[tuple[str, float], ...]
    meta_router_policy_digest: str
    evaluation_mode: str = "PAIRED_PROPOSAL_ONLY_SHADOW_V1"
    activation_boundary: str = "NEXT_CAMPAIGN"

    def __post_init__(self) -> None:
        for name in (
            "parent_policy_digest",
            "source_search_memory_digest",
            "meta_router_policy_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.source_round_index < 1:
            raise MetaControlError("Meta proposal requires a closed SearchRound")
        axes = tuple(str(item) for item in self.proposed_axis_priority)
        if (
            len(axes) != len(CANONICAL_CONTROL_AXES_V1)
            or set(axes) != set(CANONICAL_CONTROL_AXES_V1)
        ):
            raise MetaControlError(
                "Meta proposal must order the exact canonical mechanism axes"
            )
        allocations = dict(self.proposed_producer_token_allocation)
        if (
            set(allocations) != set(DISCOVERY_PRODUCERS)
            or any(value <= 0.0 for value in allocations.values())
            or abs(sum(allocations.values()) - 1.0) > 1e-12
        ):
            raise MetaControlError(
                "Meta proposal must preserve four positive Producer shares"
            )
        if self.activation_boundary != "NEXT_CAMPAIGN":
            raise MetaControlError("Meta proposals cannot activate mid-campaign")
        validate_no_research_evidence_authority_fields(self)

    @property
    def digest(self) -> str:
        return sha256_digest(
            {
                "schema": "recclaw.meta-control-update-proposal.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ProposalOnlyShadowMetricsV1:
    proposal_count: int
    common_eligible_count: int
    unique_semantics_count: int
    mechanism_axis_count: int
    lineage_root_count: int
    control_count: int
    semantic_collision_count: int
    billed_tokens: int

    def __post_init__(self) -> None:
        if any(
            int(value) < 0
            for value in (
                self.proposal_count,
                self.common_eligible_count,
                self.unique_semantics_count,
                self.mechanism_axis_count,
                self.lineage_root_count,
                self.control_count,
                self.semantic_collision_count,
                self.billed_tokens,
            )
        ):
            raise MetaControlError("Shadow metrics must be non-negative")
        if self.common_eligible_count > self.proposal_count:
            raise MetaControlError("eligible count exceeds proposal count")
        if self.unique_semantics_count > self.common_eligible_count:
            raise MetaControlError("unique semantics exceed eligible proposals")

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaControlShadowEvaluationV1:
    proposal_digest: str
    champion_policy_digest: str
    challenger_policy_digest: str
    champion: ProposalOnlyShadowMetricsV1
    challenger: ProposalOnlyShadowMetricsV1
    same_model_prompt_schema_and_contexts: bool
    deterministic_directive_replay: bool
    verdict: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "proposal_digest",
            "champion_policy_digest",
            "challenger_policy_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.verdict not in {"PASS", "HOLD"}:
            raise MetaControlError("Shadow evaluation has an invalid verdict")
        validate_no_research_evidence_authority_fields(self)

    @property
    def digest(self) -> str:
        return sha256_digest(
            {
                "schema": "recclaw.meta-control-shadow-evaluation.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaControlPromotionDecisionV1:
    proposal_digest: str
    evaluation_digest: str
    parent_policy_digest: str
    challenger_policy_digest: str
    verdict: str
    activation_boundary: str

    def __post_init__(self) -> None:
        for name in (
            "proposal_digest",
            "evaluation_digest",
            "parent_policy_digest",
            "challenger_policy_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.verdict not in {"PROMOTE", "HOLD"}:
            raise MetaControlError("Meta control decision has an invalid verdict")
        if self.activation_boundary != "NEXT_CAMPAIGN":
            raise MetaControlError("Meta control decision changed activation boundary")
        validate_no_research_evidence_authority_fields(self)

    @property
    def digest(self) -> str:
        return sha256_digest(
            {
                "schema": "recclaw.meta-control-promotion-decision.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaControlActivationReceiptV1:
    decision_digest: str
    campaign_id: str
    predecessor_policy_digest: str
    activated_policy_digest: str
    activation_boundary: str = "NEXT_CAMPAIGN"

    def __post_init__(self) -> None:
        for name in (
            "decision_digest",
            "predecessor_policy_digest",
            "activated_policy_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if not self.campaign_id:
            raise MetaControlError("Activation receipt requires a campaign ID")
        if self.activation_boundary != "NEXT_CAMPAIGN":
            raise MetaControlError("Activation receipt changed its boundary")
        validate_no_research_evidence_authority_fields(self)

    @property
    def digest(self) -> str:
        return sha256_digest(
            {
                "schema": "recclaw.meta-control-activation-receipt.v1",
                **self.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


def build_meta_update_proposal(
    *,
    policy: VersionedResearchPolicyV1,
    search_memory: SearchMemorySnapshotV1,
) -> MetaControlUpdateProposalV1:
    """Build a challenger from a closed development Search Memory snapshot."""

    if search_memory.round_index < 1:
        raise MetaControlError("Search Memory is not closed")
    counts = {axis: 0 for axis in CANONICAL_CONTROL_AXES_V1}
    unresolved = {axis: 0 for axis in CANONICAL_CONTROL_AXES_V1}
    for belief in search_memory.beliefs:
        axis = str(belief.mechanism_axis)
        if axis not in counts:
            continue
        counts[axis] += 1
        unresolved[axis] += int(bool(belief.unresolved_confounds))
    axis_priority = tuple(
        sorted(
            CANONICAL_CONTROL_AXES_V1,
            key=lambda axis: (
                counts[axis],
                -unresolved[axis],
                CANONICAL_CONTROL_AXES_V1.index(axis),
            ),
        )
    )
    allocations = tuple(
        (role, float(dict(policy.producer_token_allocation)[role]))
        for role in DISCOVERY_PRODUCERS
    )
    router_digest = policy.meta_router_policy_digest
    if router_digest is None:
        raise MetaControlError(
            "Only a content-bound Meta router policy may propose a successor"
        )
    payload = {
        "parent_policy_digest": policy.digest,
        "source_search_memory_digest": search_memory.digest,
        "source_round_index": search_memory.round_index,
        "proposed_axis_priority": axis_priority,
        "proposed_memory_retrieval_policy": (
            "ROLE_SCOPED_GAP_AWARE_V1"
        ),
        "proposed_producer_token_allocation": allocations,
        "meta_router_policy_digest": router_digest,
    }
    return MetaControlUpdateProposalV1(
        proposal_id=f"meta-control-{sha256_digest(payload)[:20]}",
        **payload,
    )


def evaluate_proposal_only_shadow(
    *,
    proposal: MetaControlUpdateProposalV1,
    champion_policy_digest: str,
    challenger_policy_digest: str,
    champion: ProposalOnlyShadowMetricsV1,
    challenger: ProposalOnlyShadowMetricsV1,
    same_model_prompt_schema_and_contexts: bool,
    deterministic_directive_replay: bool,
) -> MetaControlShadowEvaluationV1:
    """Evaluate upstream control without pretending it is outcome replay."""

    reasons: list[str] = []
    if not same_model_prompt_schema_and_contexts:
        reasons.append("SHADOW_INPUTS_NOT_PAIRED")
    if not deterministic_directive_replay:
        reasons.append("DIRECTIVE_REPLAY_NOT_DETERMINISTIC")
    if (
        champion.proposal_count != challenger.proposal_count
        or champion.proposal_count < 4
        or champion.proposal_count % 4 != 0
    ):
        reasons.append("FOUR_PRODUCER_SCHEDULE_NOT_PRESERVED")
    if champion.billed_tokens != challenger.billed_tokens:
        reasons.append("TOKEN_BUDGET_NOT_EQUAL")
    if challenger.common_eligible_count < champion.common_eligible_count:
        reasons.append("COMMON_ELIGIBILITY_REGRESSION")
    if challenger.unique_semantics_count < champion.unique_semantics_count:
        reasons.append("SEMANTIC_DIVERSITY_REGRESSION")
    if challenger.mechanism_axis_count < champion.mechanism_axis_count:
        reasons.append("AXIS_COVERAGE_REGRESSION")
    if challenger.lineage_root_count < champion.lineage_root_count:
        reasons.append("LINEAGE_COVERAGE_REGRESSION")
    if challenger.control_count < 1:
        reasons.append("MATCHED_CONTROL_MISSING")
    if (
        challenger.semantic_collision_count
        > champion.semantic_collision_count
    ):
        reasons.append("SEMANTIC_COLLISION_REGRESSION")
    strict_improvement = any(
        (
            challenger.unique_semantics_count
            > champion.unique_semantics_count,
            challenger.mechanism_axis_count
            > champion.mechanism_axis_count,
            challenger.lineage_root_count
            > champion.lineage_root_count,
            challenger.semantic_collision_count
            < champion.semantic_collision_count,
        )
    )
    if not strict_improvement:
        reasons.append("NO_UPSTREAM_QUALITY_IMPROVEMENT")
    return MetaControlShadowEvaluationV1(
        proposal_digest=proposal.digest,
        champion_policy_digest=champion_policy_digest,
        challenger_policy_digest=challenger_policy_digest,
        champion=champion,
        challenger=challenger,
        same_model_prompt_schema_and_contexts=(
            same_model_prompt_schema_and_contexts
        ),
        deterministic_directive_replay=deterministic_directive_replay,
        verdict="PASS" if not reasons else "HOLD",
        reason_codes=tuple(reasons),
    )


def decide_meta_control_promotion(
    *,
    proposal: MetaControlUpdateProposalV1,
    evaluation: MetaControlShadowEvaluationV1,
) -> MetaControlPromotionDecisionV1:
    if evaluation.proposal_digest != proposal.digest:
        raise MetaControlError("Evaluation is not bound to the proposal")
    return MetaControlPromotionDecisionV1(
        proposal_digest=proposal.digest,
        evaluation_digest=evaluation.digest,
        parent_policy_digest=proposal.parent_policy_digest,
        challenger_policy_digest=evaluation.challenger_policy_digest,
        verdict="PROMOTE" if evaluation.verdict == "PASS" else "HOLD",
        activation_boundary="NEXT_CAMPAIGN",
    )


def materialize_proposed_control_policy(
    *,
    parent: VersionedResearchPolicyV1,
    proposal: MetaControlUpdateProposalV1,
) -> VersionedResearchPolicyV1:
    if parent.digest != proposal.parent_policy_digest:
        raise MetaControlError("Meta proposal parent does not match")
    return VersionedResearchPolicyV1(
        version=parent.version + 1,
        producer_token_allocation=(
            proposal.proposed_producer_token_allocation
        ),
        mechanism_axis_targeting=proposal.proposed_axis_priority,
        memory_retrieval_policy=(
            proposal.proposed_memory_retrieval_policy
        ),
        router_priors=parent.router_priors,
        acquisition_parameters=parent.acquisition_parameters,
        predecessor_digest=parent.digest,
        meta_router_policy_digest=proposal.meta_router_policy_digest,
        meta_router_promotion_decision_digest=(
            parent.meta_router_promotion_decision_digest
        ),
        promotion_decision_digest=None,
        activation_boundary="NEXT_CAMPAIGN",
        control_mode="PROMOTED_META_CONTROL_V1",
    )


def activate_promoted_control_policy(
    *,
    parent: VersionedResearchPolicyV1,
    proposal: MetaControlUpdateProposalV1,
    decision: MetaControlPromotionDecisionV1,
    campaign_id: str,
) -> tuple[VersionedResearchPolicyV1, MetaControlActivationReceiptV1]:
    if parent.digest != proposal.parent_policy_digest:
        raise MetaControlError("Meta proposal parent does not match")
    if (
        decision.verdict != "PROMOTE"
        or decision.proposal_digest != proposal.digest
        or decision.parent_policy_digest != parent.digest
    ):
        raise MetaControlError("Only the exact promoted proposal may activate")
    activated = materialize_proposed_control_policy(
        parent=parent,
        proposal=proposal,
    )
    if activated.digest != decision.challenger_policy_digest:
        raise MetaControlError(
            "Promotion decision does not identify the activated policy"
        )
    receipt = MetaControlActivationReceiptV1(
        decision_digest=decision.digest,
        campaign_id=campaign_id,
        predecessor_policy_digest=parent.digest,
        activated_policy_digest=activated.digest,
    )
    return activated, receipt


__all__ = [
    "CANONICAL_CONTROL_AXES_V1",
    "MetaControlActivationReceiptV1",
    "MetaControlError",
    "MetaControlPromotionDecisionV1",
    "MetaControlShadowEvaluationV1",
    "MetaControlUpdateProposalV1",
    "ProposalOnlyShadowMetricsV1",
    "activate_promoted_control_policy",
    "build_meta_update_proposal",
    "decide_meta_control_promotion",
    "evaluate_proposal_only_shadow",
    "materialize_proposed_control_policy",
]
