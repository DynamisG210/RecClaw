"""Pure deterministic M0 Fusion skeleton.

The module deliberately has no dependency on an Evidence Guard package.
"""

from __future__ import annotations

from typing import Any, Mapping

from .canonical import sha256_digest
from .contracts import (
    EvidenceAdjudicationStatus,
    EvidenceAdjudicationV1,
    FusedSearchFeedbackV1,
    SearchFeedbackClass,
)

FUSION_POLICY_DIGEST = sha256_digest(
    {"contract_label": "recclaw.deterministic-fusion.m0.v1"}
)


class DeterministicFusionV1:
    def fuse(
        self,
        *,
        candidate_id: str,
        raw_outcome_projection: Mapping[str, Any] | None,
        adjudication: EvidenceAdjudicationV1,
    ) -> FusedSearchFeedbackV1:
        if adjudication.candidate_id != candidate_id:
            raise ValueError("candidate/adjudication identity mismatch")
        if adjudication.status is not EvidenceAdjudicationStatus.NOT_ADJUDICATED:
            raise ValueError("M0 Fusion accepts only Null-port NOT_ADJUDICATED")
        if raw_outcome_projection is None:
            return FusedSearchFeedbackV1(
                candidate_id=candidate_id,
                search_feedback_class=SearchFeedbackClass.NO_SEARCH_UPDATE,
                adjudication_status=adjudication.status,
                raw_outcome_projection_digest=None,
                frontier_eligibility="EXCLUDED",
                fusion_policy_digest=FUSION_POLICY_DIGEST,
            )
        return FusedSearchFeedbackV1(
            candidate_id=candidate_id,
            search_feedback_class=SearchFeedbackClass.BASELINE_RESULT,
            adjudication_status=adjudication.status,
            raw_outcome_projection_digest=sha256_digest(raw_outcome_projection),
            frontier_eligibility="CURRENT_FRONTIER",
            fusion_policy_digest=FUSION_POLICY_DIGEST,
        )

