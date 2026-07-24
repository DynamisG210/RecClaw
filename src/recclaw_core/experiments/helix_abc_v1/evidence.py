"""Shared EvidencePort seam and the non-authoritative M0 Null port."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import (
    EvidenceAdjudicationStatus,
    EvidenceAdjudicationV1,
    EvidenceStage,
)


@runtime_checkable
class EvidencePortV1(Protocol):
    def pre_run(self, candidate: Mapping[str, Any]) -> EvidenceAdjudicationV1: ...

    def post_run(self, raw_result: Mapping[str, Any]) -> EvidenceAdjudicationV1: ...


def _candidate_id(payload: Mapping[str, Any]) -> str:
    value = str(payload.get("candidate_id") or "")
    if not value:
        raise ValueError("candidate_id is required")
    return value


class NullEvidencePortV1:
    """Return only NOT_ADJUDICATED; this object grants no permission."""

    def pre_run(self, candidate: Mapping[str, Any]) -> EvidenceAdjudicationV1:
        return EvidenceAdjudicationV1(
            candidate_id=_candidate_id(candidate),
            stage=EvidenceStage.PRE,
            status=EvidenceAdjudicationStatus.NOT_ADJUDICATED,
            reason_codes=("NULL_PORT", "NO_EVIDENCE_AUTHORITY"),
        )

    def post_run(self, raw_result: Mapping[str, Any]) -> EvidenceAdjudicationV1:
        return EvidenceAdjudicationV1(
            candidate_id=_candidate_id(raw_result),
            stage=EvidenceStage.POST,
            status=EvidenceAdjudicationStatus.NOT_ADJUDICATED,
            reason_codes=("NULL_PORT", "NO_EVIDENCE_AUTHORITY"),
        )

