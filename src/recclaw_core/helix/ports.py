"""Shared Null EvidencePort for Arms A and B."""

from __future__ import annotations

from .contracts import (
    CandidateEnvelope,
    PortAdjudication,
    PortStage,
    PortStatus,
    RawResultEnvelope,
)


class NullEvidencePortV1:
    """Return NOT_ADJUDICATED; never pretend to allow or admit evidence."""

    def pre_run(self, candidate: CandidateEnvelope) -> PortAdjudication:
        return PortAdjudication(
            candidate_id=candidate.candidate_id,
            stage=PortStage.PRE,
            status=PortStatus.NOT_ADJUDICATED,
            protocol_status="NOT_ADJUDICATED",
            outcome_class="NOT_ADJUDICATED",
            claim_ceiling="NOT_ADJUDICATED",
            reason_codes=("NULL_PORT",),
            comparator_delta=None,
            evidence_use="NOT_ADJUDICATED",
            recommended_validation="NONE",
        )

    def post_run(self, raw_result: RawResultEnvelope) -> PortAdjudication:
        return PortAdjudication(
            candidate_id=raw_result.candidate_id,
            stage=PortStage.POST,
            status=PortStatus.NOT_ADJUDICATED,
            protocol_status="NOT_ADJUDICATED",
            outcome_class="NOT_ADJUDICATED",
            claim_ceiling="NOT_ADJUDICATED",
            reason_codes=("NULL_PORT",),
            comparator_delta=None,
            evidence_use="NOT_ADJUDICATED",
            recommended_validation="NONE",
        )
