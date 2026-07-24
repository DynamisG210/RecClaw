"""Same-slate synthetic Helix composition; no proposal refresh or extra budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .contracts import CandidateEnvelope, PortAdjudication
from .fusion import DeterministicHelixFusionV1


class PreRunPort(Protocol):
    def pre_run(self, candidate: CandidateEnvelope) -> PortAdjudication: ...


@dataclass(frozen=True, slots=True)
class SameSlateSelectionV1:
    selected_candidate: CandidateEnvelope | None
    inspected_candidate_ids: tuple[str, ...]
    terminal_status: str
    producer_refresh_count: int
    extra_proposal_count: int


class SameSlateHelixSelectorV1:
    def __init__(self, fusion: DeterministicHelixFusionV1) -> None:
        self._fusion = fusion

    def select(
        self, slate: Sequence[CandidateEnvelope], port: PreRunPort
    ) -> SameSlateSelectionV1:
        inspected: list[str] = []
        for candidate in tuple(slate):
            inspected.append(candidate.candidate_id)
            disposition = self._fusion.fuse(port.pre_run(candidate))
            if disposition.selection_action == "SELECT_CURRENT":
                return SameSlateSelectionV1(
                    selected_candidate=candidate,
                    inspected_candidate_ids=tuple(inspected),
                    terminal_status="SELECTED",
                    producer_refresh_count=0,
                    extra_proposal_count=0,
                )
            if disposition.selection_action == "NEXT_FROM_SAME_SLATE":
                continue
            if disposition.selection_action == "SELECT_CURRENT_NOT_ADJUDICATED":
                return SameSlateSelectionV1(
                    selected_candidate=candidate,
                    inspected_candidate_ids=tuple(inspected),
                    terminal_status="SELECTED_NOT_ADJUDICATED",
                    producer_refresh_count=0,
                    extra_proposal_count=0,
                )
            raise ValueError("unknown deterministic Fusion selection action")
        return SameSlateSelectionV1(
            selected_candidate=None,
            inspected_candidate_ids=tuple(inspected),
            terminal_status="SLATE_EXHAUSTED",
            producer_refresh_count=0,
            extra_proposal_count=0,
        )
