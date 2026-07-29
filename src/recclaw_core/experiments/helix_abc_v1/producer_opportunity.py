"""Deterministic Producer opportunity acquisition for Meta campaign routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .canonical import canonical_value, sha256_digest
from .research_capability import DISCOVERY_PRODUCERS


PRODUCER_OPPORTUNITY_POLICY_ID_V1 = (
    "META_PRODUCER_BLOCK8_COVER4_THEN_SCORE_V1"
)
PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1 = sha256_digest(
    {
        "block_size": 8,
        "coverage_roles": DISCOVERY_PRODUCERS,
        "parent_decision_binding": True,
        "policy_id": PRODUCER_OPPORTUNITY_POLICY_ID_V1,
        "selection": (
            "FORCE_EACH_AVAILABLE_ROLE_ONCE_PER_BLOCK_BEFORE_SCORE_ONLY"
        ),
        "tie_break": "PARENT_META_SCORE_ORDER",
    }
)


class ProducerOpportunityError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ProducerOpportunityDecisionV1:
    policy_id: str
    policy_digest: str
    opportunity_index: int
    block_index: int
    block_position: int
    parent_decision_digest: str
    parent_ranked_candidate_ids: tuple[str, ...]
    ranked_candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    selected_producer_role: str
    selection_reason: str
    roles_selected_in_block_before: tuple[str, ...]
    role_counts_before: tuple[tuple[str, int], ...]
    role_counts_after: tuple[tuple[str, int], ...]

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return canonical_value(self)


def acquire_producer_opportunity(
    *,
    parent_ranked_candidate_ids: Sequence[str],
    parent_decision_digest: str,
    producer_role_by_candidate_id: Mapping[str, str],
    prior_selected_roles: Sequence[str],
) -> ProducerOpportunityDecisionV1:
    """Combine fixed role coverage with the frozen parent Meta score order."""

    ranked = tuple(str(item) for item in parent_ranked_candidate_ids)
    if not ranked or len(set(ranked)) != len(ranked):
        raise ProducerOpportunityError(
            "Producer opportunity acquisition requires a unique non-empty rank"
        )
    if set(ranked) != set(producer_role_by_candidate_id):
        raise ProducerOpportunityError(
            "Producer role mapping must cover the exact ranked candidates"
        )
    roles = tuple(str(item) for item in DISCOVERY_PRODUCERS)
    role_set = set(roles)
    candidate_roles = {
        candidate_id: str(producer_role_by_candidate_id[candidate_id])
        for candidate_id in ranked
    }
    if any(role not in role_set for role in candidate_roles.values()):
        raise ProducerOpportunityError("Candidate has an unknown Producer role")
    history = tuple(str(item) for item in prior_selected_roles)
    if any(role not in role_set for role in history):
        raise ProducerOpportunityError("Opportunity history has an unknown role")

    block_size = 8
    opportunity_index = len(history)
    block_position = opportunity_index % block_size
    block_index = opportunity_index // block_size
    selected_in_block = history[
        opportunity_index - block_position : opportunity_index
    ]
    missing_roles = tuple(
        role for role in roles if role not in selected_in_block
    )
    missing_candidates = tuple(
        candidate_id
        for candidate_id in ranked
        if candidate_roles[candidate_id] in missing_roles
    )
    remaining_slots = block_size - block_position
    coverage_required = bool(missing_candidates) and (
        block_position < len(roles)
        or remaining_slots <= len(missing_roles)
    )
    if coverage_required:
        ordered = (
            *missing_candidates,
            *(
                candidate_id
                for candidate_id in ranked
                if candidate_id not in set(missing_candidates)
            ),
        )
        reason = "BLOCK_ROLE_COVERAGE"
    else:
        ordered = ranked
        reason = "PARENT_META_SCORE"

    selected_candidate_id = ordered[0]
    selected_role = candidate_roles[selected_candidate_id]
    counts_before = tuple(
        (role, history.count(role)) for role in roles
    )
    counts_after = tuple(
        (
            role,
            history.count(role) + int(role == selected_role),
        )
        for role in roles
    )
    return ProducerOpportunityDecisionV1(
        policy_id=PRODUCER_OPPORTUNITY_POLICY_ID_V1,
        policy_digest=PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1,
        opportunity_index=opportunity_index,
        block_index=block_index,
        block_position=block_position,
        parent_decision_digest=str(parent_decision_digest),
        parent_ranked_candidate_ids=ranked,
        ranked_candidate_ids=tuple(ordered),
        selected_candidate_id=selected_candidate_id,
        selected_producer_role=selected_role,
        selection_reason=reason,
        roles_selected_in_block_before=selected_in_block,
        role_counts_before=counts_before,
        role_counts_after=counts_after,
    )


__all__ = [
    "PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1",
    "PRODUCER_OPPORTUNITY_POLICY_ID_V1",
    "ProducerOpportunityDecisionV1",
    "ProducerOpportunityError",
    "acquire_producer_opportunity",
]
