"""Budgeted allocation for explicitly requested Helix evidence work.

The Research Line owns candidate generation, execution, search learning, and
the development frontier.  Helix must not shadow those responsibilities.
This module binds an explicitly requested replication or declared control to
its existing budget. The standalone discovery loop does not automatically
request these actions or turn them into discovery-slot replacements.

The deterministic information score is descriptive metadata, not a calibrated
estimate of improvement and not authority to preempt candidate exploration.
Research feedback is useful independently of any physical evidence action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Protocol

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)


class AllocationActionV31(str, Enum):
    NOOP = "NOOP"
    REPLICATE = "REPLICATE"
    CONTROL = "CONTROL"


class AllocationClosureV31(str, Enum):
    SATISFIED = "SATISFIED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AllocationLedgerPort(Protocol):
    def allocation_budget_snapshot(
        self, *, max_actions: int
    ) -> Mapping[str, Any]: ...

    def open_allocation_for_identity(
        self, *, control_identity_digest: str
    ) -> Mapping[str, Any] | None: ...

    def reserve_allocation_action(
        self,
        *,
        action_id: str,
        control_identity_digest: str,
        action: str,
        target: str,
        max_actions: int,
        decision: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bool]: ...

    def close_allocation_action(
        self,
        *,
        action_id: str,
        closure_status: str,
        closure: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], bool]: ...


@dataclass(frozen=True, slots=True)
class HelixAllocationPolicyV31:
    """Frozen cap and decision rules for marginal evidence spending.

    ``max_actions`` is a subset of the ordinary metric-round budget.  It is not
    permission to create extra rounds.  A formal experiment must bind the same
    total metric opportunities and physical-attempt ceiling in both arms.
    """

    max_actions: int
    max_open_actions: int = 1
    replication_trigger_delta: float = 0.0

    schema = "recclaw.helix.frontier-allocation-policy.v31"

    def __post_init__(self) -> None:
        if isinstance(self.max_actions, bool) or self.max_actions < 0:
            raise ValueError("max_actions must be a non-negative integer")
        if isinstance(self.max_open_actions, bool) or self.max_open_actions < 1:
            raise ValueError("max_open_actions must be a positive integer")
        if (
            isinstance(self.replication_trigger_delta, bool)
            or not isinstance(self.replication_trigger_delta, (int, float))
            or not math.isfinite(float(self.replication_trigger_delta))
            or float(self.replication_trigger_delta) < 0.0
        ):
            raise ValueError(
                "replication_trigger_delta must be finite and non-negative"
            )

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "max_actions": self.max_actions,
                "max_open_actions": self.max_open_actions,
                "replication_trigger_delta": float(
                    self.replication_trigger_delta
                ),
                "budget_semantics": "SUBSET_OF_FROZEN_METRIC_OPPORTUNITIES",
                "research_update_authority": "RESEARCH_LINE_ONLY",
            }
        )


@dataclass(frozen=True, slots=True)
class HelixAllocationDecisionV31:
    action: AllocationActionV31
    reason: str
    candidate_id: str
    candidate_semantic_digest: str
    mechanism_program_digest: str
    control_identity_digest: str
    target: str | None
    information_gain_score: float
    action_id: str | None
    policy_digest: str
    budget: Mapping[str, Any]

    schema = "recclaw.helix.frontier-allocation-decision.v31"

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.information_gain_score) <= 1.0:
            raise ValueError("information_gain_score must be in [0, 1]")
        if self.action is AllocationActionV31.NOOP:
            if self.action_id is not None or self.target is not None:
                raise ValueError("NOOP cannot reserve an action or target")
        elif not self.action_id or not self.target:
            raise ValueError("an allocated action requires action_id and target")

    @property
    def decision_relevant(self) -> bool:
        return self.action is not AllocationActionV31.NOOP

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "action": self.action.value,
                "reason": self.reason,
                "decision_relevant": self.decision_relevant,
                "candidate_id": self.candidate_id,
                "candidate_semantic_digest": self.candidate_semantic_digest,
                "mechanism_program_digest": self.mechanism_program_digest,
                "control_identity_digest": self.control_identity_digest,
                "target": self.target,
                "information_gain_score": self.information_gain_score,
                "action_id": self.action_id,
                "policy_digest": self.policy_digest,
                "budget": dict(self.budget),
            }
        )


class FrontierEvidenceAllocatorV31:
    """Reserve sparse exact-candidate evidence actions under a hard cap."""

    def __init__(
        self,
        *,
        policy: HelixAllocationPolicyV31,
        ledger: AllocationLedgerPort,
    ) -> None:
        self.policy = policy
        self.ledger = ledger

    @staticmethod
    def _identity(summary: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
        names = (
            "candidate_id",
            "candidate_semantic_digest",
            "mechanism_program_digest",
            "protocol_digest",
            "comparator_identity",
        )
        values = tuple(summary.get(name) for name in names)
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError("allocation summary lacks exact candidate identity")
        return values  # type: ignore[return-value]

    @staticmethod
    def _information_gain_score(summary: Mapping[str, Any], *, action: str) -> float:
        required = summary.get("required_seed_count")
        count = summary.get("evidence_count")
        if (
            isinstance(required, bool)
            or not isinstance(required, int)
            or required < 1
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            return 0.0
        uncertainty = max(0.0, min(1.0, 1.0 - (count / required)))
        if action == AllocationActionV31.REPLICATE.value:
            # A first positive observation is uncertain but potentially
            # frontier-relevant.  Replication resolves both sign and variance.
            return min(1.0, 0.5 + 0.5 * uncertainty)
        if action == AllocationActionV31.CONTROL.value:
            # Mixed replicated evidence has lower statistical uncertainty but
            # high mechanism-attribution value.
            return 0.75 if summary.get("sign_consistency") == "MIXED_SIGNS" else 0.6
        return 0.0

    def _noop(
        self,
        *,
        reason: str,
        candidate_id: str,
        semantic_digest: str,
        program_digest: str,
        control_identity_digest: str,
    ) -> HelixAllocationDecisionV31:
        return HelixAllocationDecisionV31(
            action=AllocationActionV31.NOOP,
            reason=reason,
            candidate_id=candidate_id,
            candidate_semantic_digest=semantic_digest,
            mechanism_program_digest=program_digest,
            control_identity_digest=control_identity_digest,
            target=None,
            information_gain_score=0.0,
            action_id=None,
            policy_digest=self.policy.digest,
            budget=self.ledger.allocation_budget_snapshot(
                max_actions=self.policy.max_actions
            ),
        )

    @staticmethod
    def _project_reserved_budget(
        budget: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Return the single-threaded post-reservation budget receipt.

        Standalone campaigns schedule one round at a time.  Projecting the
        reservation here lets the ledger and the returned decision persist
        exactly the same immutable receipt instead of two budget snapshots.
        The ledger remains the authority that atomically enforces the cap.
        """

        projected = dict(budget)
        projected["reserved_action_count"] = int(
            projected.get("reserved_action_count", 0)
        ) + 1
        projected["open_action_count"] = int(
            projected.get("open_action_count", 0)
        ) + 1
        projected["remaining_actions"] = max(
            0, int(projected.get("remaining_actions", 0)) - 1
        )
        return canonical_value(projected)

    def close_active_action(
        self,
        *,
        active_task: Mapping[str, Any] | None,
        summary: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        if not isinstance(active_task, Mapping):
            return None
        metadata = active_task.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        action_id = metadata.get("helix_allocation_action_id")
        if not isinstance(action_id, str) or not action_id:
            return None
        attempt_class = summary.get("current_attempt_class")
        task_status = str(active_task.get("status") or "")
        if attempt_class == "VALID_METRIC":
            status = AllocationClosureV31.SATISFIED
        elif task_status == "SATISFIED":
            status = AllocationClosureV31.SATISFIED
        elif task_status == "CLOSED":
            status = AllocationClosureV31.CANCELLED
        else:
            # Resource/runtime failures do not advance a metric round.  Keep
            # the exact action open so the existing attempt scheduler can
            # retry it without closure drift.
            return None
        target = active_task.get("required_seed_or_control")
        if not isinstance(target, str) or not target:
            raise ValueError("allocation task lacks its exact reserved target")
        allocation_target = target
        source_semantic_digest = metadata.get(
            "frontier_candidate_semantic_digest",
            summary.get("candidate_semantic_digest"),
        )
        source_program_digest = metadata.get(
            "frontier_mechanism_program_digest",
            summary.get("mechanism_program_digest"),
        )
        closure = canonical_value(
            {
                "action_id": action_id,
                "task_id": active_task.get("task_id"),
                "task_status": active_task.get("status"),
                "candidate_id": summary.get("candidate_id"),
                "candidate_semantic_digest": source_semantic_digest,
                "mechanism_program_digest": source_program_digest,
                "allocation_target": allocation_target,
                "required_seed_or_control": target,
                "current_attempt_class": summary.get("current_attempt_class"),
                "evidence_count": summary.get("evidence_count"),
                "scientific_conclusion_strength": summary.get(
                    "scientific_conclusion_strength"
                ),
            }
        )
        stored, _created = self.ledger.close_allocation_action(
            action_id=action_id,
            closure_status=status.value,
            closure=closure,
        )
        return stored

    def decide(
        self,
        *,
        summary: Mapping[str, Any],
        requested_control_kind: str,
        allow_new_action: bool = True,
        bound_control_target: str | None = None,
    ) -> HelixAllocationDecisionV31:
        (
            candidate_id,
            semantic_digest,
            program_digest,
            protocol_digest,
            comparator_identity,
        ) = self._identity(summary)
        control_identity_digest = sha256_digest(
            {
                "candidate_semantic_digest": semantic_digest,
                "mechanism_program_digest": program_digest,
                "protocol_digest": protocol_digest,
                "comparator_identity": comparator_identity,
            }
        )
        noop = lambda reason: self._noop(
            reason=reason,
            candidate_id=candidate_id,
            semantic_digest=semantic_digest,
            program_digest=program_digest,
            control_identity_digest=control_identity_digest,
        )
        if not allow_new_action:
            return noop("NO_NEW_EVIDENCE_ACTION_REQUESTED")
        if summary.get("current_attempt_class") != "VALID_METRIC":
            return noop("NON_SCIENTIFIC_ATTEMPT")
        remaining_opportunities = summary.get("remaining_metric_opportunities")
        if (
            isinstance(remaining_opportunities, bool)
            or not isinstance(remaining_opportunities, int)
            or remaining_opportunities <= 0
        ):
            return noop("NO_FUTURE_METRIC_OPPORTUNITY")
        open_action = self.ledger.open_allocation_for_identity(
            control_identity_digest=control_identity_digest
        )
        if open_action is not None:
            return noop("OPEN_ACTION_ALREADY_RESERVED")
        budget = self.ledger.allocation_budget_snapshot(
            max_actions=self.policy.max_actions
        )
        if int(budget.get("remaining_actions", 0)) <= 0:
            return noop("MATCHED_BUDGET_EXHAUSTED")
        if int(budget.get("open_action_count", 0)) >= self.policy.max_open_actions:
            return noop("OPEN_ACTION_CAP_REACHED")

        state = str(summary.get("scientific_conclusion_strength", ""))
        mean_delta = summary.get("mean_comparator_delta")
        next_seed = summary.get("next_eligible_seed")
        action = AllocationActionV31.NOOP
        reason = "NO_DISCRIMINATIVE_ACTION"
        target: str | None = None
        if (
            state == "PRELIMINARY_POSITIVE"
            and isinstance(mean_delta, (int, float))
            and not isinstance(mean_delta, bool)
            and math.isfinite(float(mean_delta))
            and float(mean_delta) > float(self.policy.replication_trigger_delta)
            and isinstance(next_seed, str)
            and next_seed
        ):
            action = AllocationActionV31.REPLICATE
            reason = "POSITIVE_SINGLE_OR_PARTIAL_SEED_SIGNAL"
            target = next_seed
        elif (
            state == "REPLICATED_INCONCLUSIVE"
            and requested_control_kind in {"MATCHED_CONTROL", "MECHANISM_OFF"}
        ):
            if not isinstance(bound_control_target, str) or not bound_control_target:
                return noop("CONTROL_BINDING_REQUIRED")
            action = AllocationActionV31.CONTROL
            reason = "REPLICATED_UNCERTAINTY_REQUIRES_ATTRIBUTION"
            target = bound_control_target
        if action is AllocationActionV31.NOOP or target is None:
            return noop(reason)

        action_id = sha256_digest(
            {
                "policy_digest": self.policy.digest,
                "control_identity_digest": control_identity_digest,
                "action": action.value,
                "target": target,
                "evidence_count": summary.get("evidence_count"),
            }
        )
        score = self._information_gain_score(summary, action=action.value)
        decision = HelixAllocationDecisionV31(
            action=action,
            reason=reason,
            candidate_id=candidate_id,
            candidate_semantic_digest=semantic_digest,
            mechanism_program_digest=program_digest,
            control_identity_digest=control_identity_digest,
            target=target,
            information_gain_score=score,
            action_id=action_id,
            policy_digest=self.policy.digest,
            budget=self._project_reserved_budget(budget),
        )
        self.ledger.reserve_allocation_action(
            action_id=action_id,
            control_identity_digest=control_identity_digest,
            action=action.value,
            target=target,
            max_actions=self.policy.max_actions,
            decision=decision.to_dict(),
        )
        final_budget = self.ledger.allocation_budget_snapshot(
            max_actions=self.policy.max_actions
        )
        if canonical_value(final_budget) != canonical_value(decision.budget):
            raise RuntimeError("allocation reservation budget receipt drift")
        return decision


__all__ = [
    "AllocationActionV31",
    "AllocationClosureV31",
    "FrontierEvidenceAllocatorV31",
    "HelixAllocationDecisionV31",
    "HelixAllocationPolicyV31",
]
