"""V13 scientific search state: lineage, controls, and Router evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from recclaw_core.mechanism_space.canonical import deep_freeze, deep_thaw, snapshot_json

from .canonical import canonical_value, sha256_digest
from .research_contracts import (
    CandidateProposalV4,
    MatchedControlPlanV1,
    RouterFeatureEvidenceV1,
    SearchUtilityFeaturesV1,
)


class ResearchScienceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class LineageRecordV1:
    proposal_candidate_id: str
    runtime_candidate_id: str
    mechanism_id: str
    mechanism_axis: str
    mechanism_program_digest: str
    mechanism_semantics_digest: str
    parent_candidate_id: str | None
    protocol_digest: str
    observation_seed: str
    run_status: str
    metric_name: str
    metric_value: float | None
    result_digest: str
    round_index: int
    mechanism_program: Mapping[str, Any]
    owner_arm_instance_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mechanism_program",
            deep_freeze(snapshot_json(deep_thaw(self.mechanism_program))),
        )
        if self.owner_arm_instance_id == "":
            raise ResearchScienceError("lineage owner cannot be empty")

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "mechanism_axis": self.mechanism_axis,
                "mechanism_id": self.mechanism_id,
                "mechanism_program": deep_thaw(self.mechanism_program),
                "mechanism_program_digest": self.mechanism_program_digest,
                "mechanism_semantics_digest": (
                    self.mechanism_semantics_digest
                ),
                "metric_name": self.metric_name,
                "metric_value": self.metric_value,
                "observation_seed": self.observation_seed,
                "owner_arm_instance_id": self.owner_arm_instance_id,
                "parent_candidate_id": self.parent_candidate_id,
                "proposal_candidate_id": self.proposal_candidate_id,
                "protocol_digest": self.protocol_digest,
                "result_digest": self.result_digest,
                "round_index": self.round_index,
                "run_status": self.run_status,
                "runtime_candidate_id": self.runtime_candidate_id,
            }
        )


@dataclass(frozen=True, slots=True)
class MatchedComparatorV1:
    mechanism_question_digest: str
    primary_candidate_id: str
    comparator_candidate_id: str
    comparator_program_digest: str
    protocol_digest: str
    changed_axis: str
    comparator_metric: float

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class LineageIndexV1:
    """Arm-private content-addressed lineage spanning prior SearchRounds."""

    def __init__(self, owner_arm_instance_id: str | None = None) -> None:
        if owner_arm_instance_id == "":
            raise ResearchScienceError("lineage index owner cannot be empty")
        self.owner_arm_instance_id = owner_arm_instance_id
        self._records: dict[str, LineageRecordV1] = {}

    def bind_owner(self, opaque_arm_instance_id: str) -> None:
        if not opaque_arm_instance_id:
            raise ResearchScienceError("lineage owner cannot be empty")
        if self._records:
            raise ResearchScienceError(
                "lineage owner must be bound before the first record"
            )
        if (
            self.owner_arm_instance_id is not None
            and self.owner_arm_instance_id != opaque_arm_instance_id
        ):
            raise ResearchScienceError("lineage owner cannot be rebound")
        self.owner_arm_instance_id = opaque_arm_instance_id

    @property
    def records(self) -> tuple[LineageRecordV1, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (
                    item.round_index,
                    item.proposal_candidate_id,
                    item.observation_seed,
                ),
            )
        )

    @property
    def digest(self) -> str:
        return sha256_digest([item.to_dict() for item in self.records])

    def record(self, item: LineageRecordV1) -> None:
        if (
            self.owner_arm_instance_id is not None
            and item.owner_arm_instance_id
            != self.owner_arm_instance_id
        ):
            raise ResearchScienceError(
                "lineage record crossed its Arm owner boundary"
            )
        key = sha256_digest(
            {
                "observation_seed": item.observation_seed,
                "proposal_candidate_id": item.proposal_candidate_id,
                "protocol_digest": item.protocol_digest,
            }
        )
        prior = self._records.get(key)
        if prior is not None and prior.digest != item.digest:
            raise ResearchScienceError("lineage observation identity substitution")
        self._records[key] = item

    def latest_for_candidate(
        self,
        candidate_id: str,
        *,
        protocol_digest: str | None = None,
    ) -> LineageRecordV1 | None:
        rows = [
            item
            for item in self.records
            if item.proposal_candidate_id == candidate_id
            and (
                protocol_digest is None
                or item.protocol_digest == protocol_digest
            )
        ]
        return rows[-1] if rows else None

    def latest_success(self) -> LineageRecordV1 | None:
        rows = [
            item
            for item in self.records
            if item.run_status in {"SUCCESS", "COMPLETED", "SMOKE_PASS"}
            and item.metric_value is not None
        ]
        return rows[-1] if rows else None

    def latest_exact_program(
        self,
        mechanism_program_digest: str,
        *,
        protocol_digest: str,
        observation_seed: str,
    ) -> LineageRecordV1 | None:
        """Return an exact successful Arm-private comparator observation."""

        rows = [
            item
            for item in self.records
            if item.mechanism_program_digest == mechanism_program_digest
            and item.protocol_digest == protocol_digest
            and item.observation_seed == observation_seed
            and item.run_status in {"SUCCESS", "COMPLETED", "SMOKE_PASS"}
            and item.metric_value is not None
        ]
        return rows[-1] if rows else None

    def mechanism_depth(self, parent_candidate_id: str | None) -> int:
        depth = 0
        current = parent_candidate_id
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ResearchScienceError("lineage parent cycle")
            seen.add(current)
            record = self.latest_for_candidate(current)
            if record is None:
                break
            depth += 1
            current = record.parent_candidate_id
        return depth

    def semantic_seen(self, semantics_digest: str) -> bool:
        return any(
            item.mechanism_semantics_digest == semantics_digest
            for item in self.records
        )

    def blocker_rate(self, mechanism_id: str) -> float:
        rows = [
            item for item in self.records if item.mechanism_id == mechanism_id
        ]
        if not rows:
            return 0.0
        failures = sum(
            item.run_status not in {"SUCCESS", "COMPLETED", "SMOKE_PASS"}
            for item in rows
        )
        return failures / len(rows)

    def exact_parent(
        self,
        proposal: CandidateProposalV4,
        *,
        protocol_digest: str,
    ) -> LineageRecordV1 | None:
        if proposal.parent_candidate_id is None:
            return None
        return self.latest_for_candidate(
            proposal.parent_candidate_id,
            protocol_digest=protocol_digest,
        )

    def matched_comparator(
        self,
        proposal: CandidateProposalV4,
        *,
        protocol_digest: str,
        observation_seed: str,
    ) -> MatchedComparatorV1 | None:
        parent = self.exact_parent(
            proposal,
            protocol_digest=protocol_digest,
        )
        comparator = parent
        if comparator is None:
            planned_digest = (
                proposal.matched_control_plan.comparator_program_digest
            )
            if planned_digest is not None:
                comparator = self.latest_exact_program(
                    planned_digest,
                    protocol_digest=protocol_digest,
                    observation_seed=observation_seed,
                )
        if comparator is None or comparator.metric_value is None:
            return None
        question_digest = sha256_digest(
            {
                "changed_axis": proposal.mechanism_axis,
                "mechanism_hypothesis": proposal.mechanism_hypothesis,
                "parent_candidate_id": proposal.parent_candidate_id,
                "protocol_digest": protocol_digest,
            }
        )
        if proposal.discriminative_plan is not None:
            declared = (
                proposal.discriminative_plan.matched_control_plan
                .mechanism_question_digest
            )
            if declared != question_digest:
                return None
        return MatchedComparatorV1(
            mechanism_question_digest=question_digest,
            primary_candidate_id=proposal.candidate_id,
            comparator_candidate_id=comparator.proposal_candidate_id,
            comparator_program_digest=comparator.mechanism_program_digest,
            protocol_digest=protocol_digest,
            changed_axis=proposal.mechanism_axis,
            comparator_metric=float(comparator.metric_value),
        )


@dataclass(frozen=True, slots=True)
class DeterministicRouterFeatureBuilderV1:
    """Derive Router core features from executable and historical facts."""

    def build(
        self,
        *,
        compile_valid: bool,
        handler_available: bool,
        materializer_available: bool,
        mechanism_id: str,
        mechanism_depth: int,
        estimated_cost: float,
        semantics_digest: str,
        parent_available: bool,
        lineage: LineageIndexV1,
        llm_diagnostic: SearchUtilityFeaturesV1,
    ) -> tuple[SearchUtilityFeaturesV1, RouterFeatureEvidenceV1]:
        blocker_rate = lineage.blocker_rate(mechanism_id)
        duplicate = lineage.semantic_seen(semantics_digest)
        runnable = (
            1.0 - blocker_rate
            if compile_valid and handler_available and materializer_available
            else 0.0
        )
        depth_signal = min(1.0, mechanism_depth / 2.0)
        novelty = 0.0 if duplicate else 1.0
        parent_signal = 1.0 if parent_available else 0.0
        useful = (
            0.45 * llm_diagnostic.useful_signal
            + 0.25 * parent_signal
            + 0.15 * depth_signal
            + 0.15 * novelty
        )
        frontier = (
            0.45 * llm_diagnostic.frontier_potential
            + 0.30 * novelty
            + 0.15 * depth_signal
            + 0.10 * parent_signal
        )
        information = (
            0.50 * llm_diagnostic.information_gain
            + 0.30 * parent_signal
            + 0.20 * novelty
        )
        evidence = RouterFeatureEvidenceV1(
            compile_valid=compile_valid,
            handler_available=handler_available,
            materializer_available=materializer_available,
            blocker_rate=blocker_rate,
            semantic_duplicate=duplicate,
            parent_available=parent_available,
            mechanism_depth=mechanism_depth,
            estimated_cost=estimated_cost,
            llm_diagnostic=llm_diagnostic,
        )
        features = SearchUtilityFeaturesV1(
            runnable_probability=max(0.0, min(1.0, runnable)),
            useful_signal=max(0.0, min(1.0, useful)),
            frontier_potential=max(0.0, min(1.0, frontier)),
            information_gain=max(0.0, min(1.0, information)),
            cost=max(0.0, min(1.0, estimated_cost)),
            blocker_risk=max(0.0, min(1.0, blocker_rate)),
        )
        return features, evidence


@dataclass(frozen=True, slots=True)
class ControlAblationBuilderV2:
    service_id: str = "control_ablation_builder.v2"

    def build(
        self,
        *,
        lineage: LineageIndexV1,
        primary_candidate_id: str,
        parent_candidate_id: str | None,
        changed_axis: str,
        mechanism_hypothesis: str,
        protocol_digest: str,
    ) -> MatchedControlPlanV1:
        return matched_control_plan(
            lineage=lineage,
            primary_candidate_id=primary_candidate_id,
            parent_candidate_id=parent_candidate_id,
            changed_axis=changed_axis,
            mechanism_hypothesis=mechanism_hypothesis,
            protocol_digest=protocol_digest,
        )


@dataclass(frozen=True, slots=True)
class EngineeringFeasibilityUpdateV1:
    candidate_id: str
    blocker_code: str
    repair_status: str
    discovery_credit: str = "NON_DISCOVERY_REPAIR"

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class RepairEngineerV2:
    service_id: str = "repair_engineer.v2"

    def record(
        self,
        *,
        candidate_id: str,
        blocker_code: str,
        repair_status: str,
    ) -> EngineeringFeasibilityUpdateV1:
        return EngineeringFeasibilityUpdateV1(
            candidate_id=candidate_id,
            blocker_code=blocker_code,
            repair_status=repair_status,
        )


def matched_control_plan(
    *,
    lineage: LineageIndexV1,
    primary_candidate_id: str,
    parent_candidate_id: str | None,
    changed_axis: str,
    mechanism_hypothesis: str,
    protocol_digest: str,
    queued_comparator_candidate_id: str | None = None,
    queued_comparator_program_digest: str | None = None,
    observation_seed: str | None = None,
) -> MatchedControlPlanV1:
    question_digest = sha256_digest(
        {
            "changed_axis": changed_axis,
            "mechanism_hypothesis": mechanism_hypothesis,
            "parent_candidate_id": parent_candidate_id,
            "protocol_digest": protocol_digest,
        }
    )
    parent = (
        lineage.latest_for_candidate(
            parent_candidate_id,
            protocol_digest=protocol_digest,
        )
        if parent_candidate_id is not None
        else None
    )
    exact_control = (
        lineage.latest_exact_program(
            queued_comparator_program_digest,
            protocol_digest=protocol_digest,
            observation_seed=observation_seed,
        )
        if (
            parent is None
            and queued_comparator_program_digest is not None
            and observation_seed is not None
        )
        else None
    )
    comparator = parent or exact_control
    return MatchedControlPlanV1(
        mechanism_question_digest=question_digest,
        primary_candidate_id=primary_candidate_id,
        comparator_candidate_id=(
            comparator.proposal_candidate_id
            if comparator is not None
            else queued_comparator_candidate_id
        ),
        comparator_program_digest=(
            comparator.mechanism_program_digest
            if comparator is not None
            else queued_comparator_program_digest
        ),
        protocol_digest=protocol_digest,
        changed_axis=changed_axis,
        plan_status=(
            "MATCHED_COMPARATOR_AVAILABLE"
            if comparator is not None
            else "QUEUE_MATCHED_CONTROL"
        ),
    )


__all__ = [
    "ControlAblationBuilderV2",
    "DeterministicRouterFeatureBuilderV1",
    "EngineeringFeasibilityUpdateV1",
    "LineageIndexV1",
    "LineageRecordV1",
    "MatchedComparatorV1",
    "ResearchScienceError",
    "RepairEngineerV2",
    "matched_control_plan",
]
