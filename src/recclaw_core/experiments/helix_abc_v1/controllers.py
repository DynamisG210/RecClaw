"""Thin, fixture-only M0 controller seams.

These adapters contain no LLM, Router, Meta, materializer, or Runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .canonical import sha256_digest
from .contracts import validate_no_research_evidence_authority_fields

ORIGINAL_SOURCE_COMMIT = "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
ORIGINAL_AGENT_BLOB_SHA1 = "c40334b72dbb9557eced2bd081915b5333156fdf"
RESEARCH_LINE_INTRODUCTION_COMMIT = "a96db67d98ccddde58e4fa641d8ccea92f10c33a"


class ControllerContractError(ValueError):
    pass


@runtime_checkable
class ProposalControllerV1(Protocol):
    def propose(
        self,
        context: Mapping[str, Any],
        space_projection: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]: ...

    def select(
        self,
        common_eligible_actions: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

    def close_round(
        self,
        round_feedback: Mapping[str, Any],
        search_memory_commit_or_no_write: Mapping[str, Any] | str,
    ) -> Mapping[str, Any]: ...


def _require_fixture_mode(context: Mapping[str, Any]) -> None:
    if context.get("execution_mode") != "M0_FIXTURE_ONLY":
        raise ControllerContractError(
            "M0 controller adapters accept fixture input only; runtime proposal is unavailable"
        )
    validate_no_research_evidence_authority_fields(context)


@dataclass(frozen=True, slots=True)
class OriginalControllerV1:
    source_commit: str = ORIGINAL_SOURCE_COMMIT
    source_blob_sha1: str = ORIGINAL_AGENT_BLOB_SHA1

    @staticmethod
    def proposal_refresh_required(
        *,
        round_index: int,
        proposal_every: int,
        force_refresh: bool,
        proposal_artifact_exists: bool,
    ) -> bool:
        if round_index < 1:
            raise ControllerContractError("round_index must be >= 1")
        every = max(1, int(proposal_every))
        return bool(
            force_refresh
            or round_index == 1
            or (round_index - 1) % every == 0
            or not proposal_artifact_exists
        )

    def propose(
        self,
        context: Mapping[str, Any],
        space_projection: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        _require_fixture_mode(context)
        validate_no_research_evidence_authority_fields(space_projection)
        validate_no_research_evidence_authority_fields(budget)
        proposals = context.get("fixture_proposals", ())
        if not isinstance(proposals, (list, tuple)):
            raise ControllerContractError("fixture_proposals must be a sequence")
        return tuple(dict(item) for item in proposals)

    def select(
        self,
        common_eligible_actions: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _require_fixture_mode(context)
        validate_no_research_evidence_authority_fields(budget)
        if not common_eligible_actions:
            raise ControllerContractError("Original selection requires an eligible action")
        indexed = tuple(enumerate(common_eligible_actions))
        ordered = sorted(
            indexed,
            key=lambda item: (-float(item[1].get("original_score", 0.0)), item[0]),
        )
        return dict(ordered[0][1])

    def close_round(
        self,
        round_feedback: Mapping[str, Any],
        search_memory_commit_or_no_write: Mapping[str, Any] | str,
    ) -> Mapping[str, Any]:
        validate_no_research_evidence_authority_fields(round_feedback)
        if search_memory_commit_or_no_write != "NO_WRITE":
            raise ControllerContractError("Arm A must not write Research Search Memory")
        feedback_digest = sha256_digest(round_feedback)
        transition = {
            "applied_transition_class": "ORIGINAL_FEEDBACK_CONSUMED",
            "feedback_consumption_count": 1,
            "round_feedback_digest": feedback_digest,
            "search_memory_commit": "NO_WRITE",
            "source_blob_sha1": self.source_blob_sha1,
            "source_commit": self.source_commit,
        }
        return {**transition, "transition_digest": sha256_digest(transition)}

    def replay_golden_fixture(self, fixture: Mapping[str, Any]) -> dict[str, Any]:
        fixed = fixture["fixed_input"]
        proposal_schedule = [
            {
                "round_index": round_index,
                "refresh_required": self.proposal_refresh_required(
                    round_index=round_index,
                    proposal_every=int(fixed["proposal_every"]),
                    force_refresh=False,
                    proposal_artifact_exists=True,
                ),
            }
            for round_index in range(
                int(fixed["start_round"]), int(fixed["final_round"]) + 1
            )
        ]
        actions = tuple(dict(item) for item in fixed["common_eligible_actions"])
        selection_order = [
            item["candidate_id"]
            for _, item in sorted(
                enumerate(actions),
                key=lambda pair: (
                    -float(pair[1].get("original_score", 0.0)),
                    pair[0],
                ),
            )
        ]
        selected = self.select(
            actions,
            {"execution_mode": "M0_FIXTURE_ONLY"},
            fixed["budget_snapshot"],
        )
        transition = self.close_round(fixed["round_feedback"], "NO_WRITE")
        return {
            "budget_debits": fixed["expected_budget_debits"],
            "feedback_consumption_count": transition["feedback_consumption_count"],
            "proposal_schedule": proposal_schedule,
            "selected_candidate_id": selected["candidate_id"],
            "selection_order": selection_order,
            "stop": {
                "after_round": int(fixed["final_round"]),
                "next_round_opened": False,
                "reason": "CONFIGURED_FINAL_ROUND",
            },
        }


@dataclass(frozen=True, slots=True)
class ResearchLineControllerV1:
    """A seam only; Producer, Router, Meta, and Search Memory arrive in M2."""

    interface_version: str = "recclaw.research-line-controller.m0.v1"

    def propose(
        self,
        context: Mapping[str, Any],
        space_projection: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]:
        _require_fixture_mode(context)
        validate_no_research_evidence_authority_fields(space_projection)
        validate_no_research_evidence_authority_fields(budget)
        proposals = context.get("fixture_proposals", ())
        if not isinstance(proposals, (list, tuple)):
            raise ControllerContractError("fixture_proposals must be a sequence")
        return tuple(dict(item) for item in proposals)

    def select(
        self,
        common_eligible_actions: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        budget: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        _require_fixture_mode(context)
        validate_no_research_evidence_authority_fields(budget)
        if not common_eligible_actions:
            raise ControllerContractError(
                "M0 has no Router hard-gate implementation for an empty slate"
            )
        return dict(common_eligible_actions[0])

    def close_round(
        self,
        round_feedback: Mapping[str, Any],
        search_memory_commit_or_no_write: Mapping[str, Any] | str,
    ) -> Mapping[str, Any]:
        validate_no_research_evidence_authority_fields(round_feedback)
        validate_no_research_evidence_authority_fields(
            search_memory_commit_or_no_write
        )
        transition = {
            "applied_transition_class": "M0_FIXTURE_FEEDBACK_CONSUMED",
            "feedback_consumption_count": 1,
            "round_feedback_digest": sha256_digest(round_feedback),
            "search_memory_commit": search_memory_commit_or_no_write,
        }
        return {**transition, "transition_digest": sha256_digest(transition)}

