"""Common deterministic Fusion and narrow Search-Memory bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest

from .contracts import CompactFeedback, PortAdjudication, PortStage, PortStatus


@dataclass(frozen=True, slots=True)
class FusionDispositionV1:
    candidate_id: str
    selection_action: str
    compact_feedback: CompactFeedback | None
    policy_digest: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class DeterministicHelixFusionV1:
    policy_digest = sha256_digest(
        {
            "policy": "DeterministicHelixFusionV1",
            "pre_block": "NEXT_FROM_SAME_SLATE",
            "null": "NO_COMPACT_FEEDBACK",
            "post": "EXACT_EIGHT_FIELD_COMPACT_FEEDBACK",
        }
    )

    def fuse(self, adjudication: PortAdjudication) -> FusionDispositionV1:
        compact = None
        if adjudication.stage is PortStage.PRE:
            if adjudication.status is PortStatus.BLOCK:
                selection = "NEXT_FROM_SAME_SLATE"
            elif adjudication.status is PortStatus.NOT_ADJUDICATED:
                selection = "SELECT_CURRENT_NOT_ADJUDICATED"
            else:
                selection = "SELECT_CURRENT"
        elif adjudication.status is PortStatus.NOT_ADJUDICATED:
            selection = "NO_SEARCH_UPDATE"
        else:
            selection = "POST_RESULT_FEEDBACK"
            compact = CompactFeedback(
                candidate_id=adjudication.candidate_id,
                protocol_status=adjudication.protocol_status,
                outcome_class=adjudication.outcome_class,
                claim_ceiling=adjudication.claim_ceiling,
                reason_codes=adjudication.reason_codes,
                comparator_delta=adjudication.comparator_delta,
                evidence_use=adjudication.evidence_use,
                recommended_validation=adjudication.recommended_validation,
            )
        return FusionDispositionV1(
            candidate_id=adjudication.candidate_id,
            selection_action=selection,
            compact_feedback=compact,
            policy_digest=self.policy_digest,
        )


@dataclass(frozen=True, slots=True)
class SearchMemoryFusionInstructionV1:
    candidate_id: str
    destination: str
    compact_feedback_digest: str
    engineering_retention: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class HelixFusionBridgeV1:
    def map(self, feedback: CompactFeedback) -> SearchMemoryFusionInstructionV1:
        if feedback.protocol_status == "PROTOCOL_BRANCH":
            destination = "EXCLUDE_FROM_CURRENT_FRONTIER"
        elif feedback.recommended_validation == "REQUIRES_CONFIRMATION":
            destination = "VALIDATION_ROUTER"
        elif feedback.evidence_use in {
            "RECORD_EXECUTABILITY_ONLY",
            "RECORD_RUNTIME_BLOCKER_ONLY",
            "QUARANTINE_PROVENANCE_INCOMPLETE",
        }:
            destination = "DIAGNOSTIC_MEMORY"
        elif feedback.evidence_use == "EXCLUDE_FROM_CURRENT_CLAIM":
            destination = "NO_ACCEPTED_SIGNAL"
        else:
            destination = "CURRENT_FRONTIER"
        return SearchMemoryFusionInstructionV1(
            candidate_id=feedback.candidate_id,
            destination=destination,
            compact_feedback_digest=feedback.digest,
            engineering_retention="DEVELOPMENT_ONLY",
        )
