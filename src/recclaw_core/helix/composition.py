"""Same-slate synthetic Helix composition; no proposal refresh or extra budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import CandidateEnvelope, PortAdjudication
from .fusion import DeterministicHelixFusionV1
from .scientific_attribution import DeterministicHelixAdmissionV13


class PreRunPort(Protocol):
    def pre_run(self, candidate: CandidateEnvelope) -> PortAdjudication: ...


@dataclass(frozen=True, slots=True)
class SameSlateSelectionV1:
    selected_candidate: CandidateEnvelope | None
    inspected_candidate_ids: tuple[str, ...]
    terminal_status: str
    producer_refresh_count: int
    extra_proposal_count: int
    last_adjudication: PortAdjudication | None = None


class SameSlateHelixSelectorV1:
    def __init__(
        self,
        admission: (
            DeterministicHelixAdmissionV13
            | DeterministicHelixFusionV1
            | None
        ) = None,
    ) -> None:
        selected = admission or DeterministicHelixAdmissionV13()
        self._admission = selected

    def select(
        self, slate: Sequence[CandidateEnvelope], port: PreRunPort
    ) -> SameSlateSelectionV1:
        inspected: list[str] = []
        last_adjudication: PortAdjudication | None = None
        for candidate in tuple(slate):
            inspected.append(candidate.candidate_id)
            last_adjudication = port.pre_run(candidate)
            if isinstance(
                self._admission, DeterministicHelixAdmissionV13
            ):
                action = self._admission.admit_pre(last_adjudication)
            else:
                action = self._admission.fuse(
                    last_adjudication
                ).selection_action
            if action == "SELECT_CURRENT":
                return SameSlateSelectionV1(
                    selected_candidate=candidate,
                    inspected_candidate_ids=tuple(inspected),
                    terminal_status="SELECTED",
                    producer_refresh_count=0,
                    extra_proposal_count=0,
                    last_adjudication=last_adjudication,
                )
            if action == "NEXT_FROM_SAME_SLATE":
                continue
            if action == "SELECT_CURRENT_NOT_ADJUDICATED":
                return SameSlateSelectionV1(
                    selected_candidate=candidate,
                    inspected_candidate_ids=tuple(inspected),
                    terminal_status="SELECTED_NOT_ADJUDICATED",
                    producer_refresh_count=0,
                    extra_proposal_count=0,
                    last_adjudication=last_adjudication,
                )
            raise ValueError("unknown deterministic V13 PRE admission action")
        return SameSlateSelectionV1(
            selected_candidate=None,
            inspected_candidate_ids=tuple(inspected),
            terminal_status="SLATE_EXHAUSTED",
            producer_refresh_count=0,
            extra_proposal_count=0,
            last_adjudication=last_adjudication,
        )
