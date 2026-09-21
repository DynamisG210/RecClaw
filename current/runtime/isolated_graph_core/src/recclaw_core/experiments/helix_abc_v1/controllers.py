"""Thin, fixture-only M0 controller seams.

These adapters contain no LLM, Router, Meta, materializer, or Runner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .canonical import canonical_value, sha256_digest
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


@dataclass(slots=True)
class OriginalRuntimeAdapterV1:
    """Faithful BL projection of the pre-Research-Line planner behavior.

    It preserves the Original refresh cadence, novelty/revisit/crash scoring,
    family outcome credit, execution-signature de-duplication and one feedback
    transition per SearchRound.  It intentionally has no Research Router,
    Search Memory, Meta, or Evidence authority dependency.
    """

    proposal_every: int = 3
    novelty_weight: float = 0.45
    priority_weight: float = 0.25
    status_weight: float = 0.20
    revisit_penalty: float = 0.10
    crash_penalty: float = 0.20
    recent_schedule_penalty: float = 0.15
    history: list[dict[str, Any]] = field(default_factory=list)
    cached_proposals: tuple[Mapping[str, Any], ...] = ()
    last_refresh_round: int | None = None

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "adapter": "OriginalRuntimeAdapterV1",
                "proposal_every": self.proposal_every,
                "source_blob_sha1": ORIGINAL_AGENT_BLOB_SHA1,
                "source_commit": ORIGINAL_SOURCE_COMMIT,
                "weights": {
                    "crash_penalty": self.crash_penalty,
                    "novelty_weight": self.novelty_weight,
                    "priority_weight": self.priority_weight,
                    "recent_schedule_penalty": self.recent_schedule_penalty,
                    "revisit_penalty": self.revisit_penalty,
                    "status_weight": self.status_weight,
                },
            }
        )

    def refresh_required(self, round_index: int) -> bool:
        return OriginalControllerV1.proposal_refresh_required(
            round_index=round_index,
            proposal_every=self.proposal_every,
            force_refresh=False,
            proposal_artifact_exists=bool(self.cached_proposals),
        )

    def install_proposals(
        self, *, round_index: int, proposals: Sequence[Mapping[str, Any]]
    ) -> None:
        if not proposals:
            raise ControllerContractError("Original proposal refresh is empty")
        self.cached_proposals = tuple(dict(item) for item in proposals)
        self.last_refresh_round = round_index

    def _family_credit(self, family_id: str) -> float:
        keeps = revises = discards = crashes = collapses = 0
        for row in self.history:
            if str(row.get("family_id")) != family_id:
                continue
            decision = str(row.get("decision"))
            keeps += int(decision == "keep")
            revises += int(decision == "revise")
            discards += int(decision == "discard")
            crashes += int(decision == "crash")
            collapses += int(bool(row.get("quality_collapse")))
        return min(0.45, 0.2 * keeps + 0.04 * revises) - min(
            0.45, 0.15 * crashes + 0.04 * discards + 0.18 * collapses
        )

    def _score(self, action: Mapping[str, Any]) -> float:
        candidate_id = str(action["candidate_id"])
        family_id = str(action.get("family_id") or action.get("mechanism_id"))
        runs = [
            row
            for row in self.history
            if str(row.get("candidate_id")) == candidate_id
        ]
        explored = len(runs)
        crashes = sum(
            int(str(row.get("decision")) == "crash") for row in runs
        )
        recently_scheduled = bool(self.history) and str(
            self.history[-1].get("candidate_id")
        ) == candidate_id
        priority = {"high": 1.0, "medium": 0.5, "low": 0.2}.get(
            str(action.get("priority", "high")).lower(), 0.0
        )
        status = {
            "implemented": 1.0,
            "implement-ready": 0.7,
            "spec-ready": 0.3,
            "idea": 0.1,
        }.get(str(action.get("status", "implemented")).lower(), 0.0)
        novelty = 1.0 / (1.0 + explored)
        return (
            self.novelty_weight * novelty
            + self.priority_weight * priority
            + self.status_weight * status
            + self._family_credit(family_id)
            - self.revisit_penalty * explored
            - self.crash_penalty * crashes
            - (
                self.recent_schedule_penalty
                if recently_scheduled
                else 0.0
            )
        )

    def rank(
        self, common_eligible_actions: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        if not common_eligible_actions:
            return ()
        used_semantics = {
            str(row["mechanism_semantics_digest"])
            for row in self.history
            if row.get("mechanism_semantics_digest")
        }
        indexed = list(enumerate(common_eligible_actions))
        indexed.sort(
            key=lambda item: (
                1
                if str(item[1].get("mechanism_semantics_digest"))
                in used_semantics
                else 0,
                -self._score(item[1]),
                item[0],
            )
        )
        return tuple(dict(item) for _index, item in indexed)

    def close_round(self, feedback: Mapping[str, Any]) -> Mapping[str, Any]:
        validate_no_research_evidence_authority_fields(feedback)
        outcome = dict(feedback["search_outcome"])
        metrics = dict(outcome.get("normalized_metrics", {}))
        metric = metrics.get("ndcg@10", metrics.get("ndcg"))
        prior_values = [
            float(row["metric"])
            for row in self.history
            if isinstance(row.get("metric"), (int, float))
        ]
        prior_best = max(prior_values) if prior_values else None
        if str(outcome.get("run_status")) != "SUCCESS" or metric is None:
            decision = "crash"
        elif prior_best is None or float(metric) > prior_best + 1e-5:
            decision = "keep"
        elif float(metric) >= prior_best - 0.002:
            decision = "revise"
        else:
            decision = "discard"
        row = {
            "candidate_id": str(feedback["candidate_id"]),
            "decision": decision,
            "family_id": str(feedback.get("mechanism_id") or ""),
            "mechanism_semantics_digest": feedback.get(
                "mechanism_semantics_digest"
            ),
            "metric": float(metric) if metric is not None else None,
            "quality_collapse": (
                prior_best is not None
                and metric is not None
                and float(metric) < prior_best - 0.05
            ),
            "round_index": int(feedback["round_index"]),
        }
        self.history.append(row)
        transition = {
            "applied_transition_class": "ORIGINAL_RUNTIME_FEEDBACK_CONSUMED",
            "decision": decision,
            "feedback_consumption_count": 1,
            "history_digest": sha256_digest(self.history),
            "round_feedback_digest": sha256_digest(feedback),
            "search_memory_commit": "NO_WRITE",
            "source_blob_sha1": ORIGINAL_AGENT_BLOB_SHA1,
            "source_commit": ORIGINAL_SOURCE_COMMIT,
        }
        return {**transition, "transition_digest": sha256_digest(transition)}

    def state_projection(self) -> Mapping[str, Any]:
        return canonical_value(
            {
                "best_metric": max(
                    (
                        float(row["metric"])
                        for row in self.history
                        if isinstance(row.get("metric"), (int, float))
                    ),
                    default=None,
                ),
                "executed": [
                    {
                        "candidate_id": row["candidate_id"],
                        "decision": row["decision"],
                        "family_id": row["family_id"],
                        "metric": row["metric"],
                    }
                    for row in self.history[-12:]
                ],
                "last_refresh_round": self.last_refresh_round,
                "proposal_every": self.proposal_every,
            }
        )

    def to_state(self) -> Mapping[str, Any]:
        """Return the complete resumable Original proposal/selection state."""

        return canonical_value(
            {
                "schema": "recclaw.original-runtime-adapter-state.v1",
                "identity_digest": self.identity_digest,
                "history": tuple(dict(item) for item in self.history),
                "cached_proposals": tuple(
                    dict(item) for item in self.cached_proposals
                ),
                "last_refresh_round": self.last_refresh_round,
            }
        )

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> "OriginalRuntimeAdapterV1":
        """Restore state only for the exact sealed Original policy identity."""

        if not isinstance(value, Mapping):
            raise ControllerContractError(
                "Original controller state must be a mapping"
            )
        controller = cls()
        if (
            value.get("schema")
            != "recclaw.original-runtime-adapter-state.v1"
            or value.get("identity_digest") != controller.identity_digest
        ):
            raise ControllerContractError(
                "Original controller state identity drift"
            )
        history = value.get("history", ())
        cached = value.get("cached_proposals", ())
        if not isinstance(history, (tuple, list)) or not isinstance(
            cached, (tuple, list)
        ):
            raise ControllerContractError(
                "Original controller state payload is invalid"
            )
        controller.history = [dict(item) for item in history]
        controller.cached_proposals = tuple(dict(item) for item in cached)
        last_refresh = value.get("last_refresh_round")
        controller.last_refresh_round = (
            None if last_refresh is None else int(last_refresh)
        )
        return controller


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
