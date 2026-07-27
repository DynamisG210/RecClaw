"""Typed, search-side contracts for the Meta VNext ranker."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..canonical import canonical_value, sha256_digest, validate_sha256
from ..research_contracts import SearchUtilityFeaturesV1


class MetaVNextContractError(ValueError):
    """Raised when a Meta VNext scientific or identity contract is invalid."""


class ChangeClassV1(str, Enum):
    CONTROL = "CONTROL"
    PARAMETER_TUNING_ONLY = "PARAMETER_TUNING_ONLY"
    MECHANISM_CHANGE = "MECHANISM_CHANGE"
    ARCHITECTURE_REWRITE = "ARCHITECTURE_REWRITE"


class EpisodeSplitV1(str, Enum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    PROMOTION_HELDOUT = "PROMOTION_HELDOUT"


class HeldoutFreshnessV1(str, Enum):
    DEVELOPMENT = "DEVELOPMENT"
    FRESH_BLINDED = "FRESH_BLINDED"
    PREVIOUSLY_UNBLINDED = "PREVIOUSLY_UNBLINDED"


MECHANISM_AXIS_FEATURES_V1 = (
    "objective",
    "geometry",
    "propagation",
    "message_transform",
    "self_supervision",
    "sampling",
    "optimization",
    "architecture",
    "other",
)

FAST_ADAPTATION_FEATURES_V1 = tuple(
    f"axis_{axis}" for axis in MECHANISM_AXIS_FEATURES_V1
) + (
    "change_control",
    "change_tuning",
    "change_mechanism",
    "change_architecture",
)

PRIMITIVE_CLASS_FEATURES_V1 = (
    "objective",
    "embedding",
    "encoder_message",
    "propagation",
    "self_supervision",
    "sampling",
    "regularization",
    "training",
    "other",
)

RANK_FEATURE_NAMES_V1 = (
    "strong_static_score",
    "static_runnable",
    "static_useful",
    "static_frontier",
    "static_information",
    "static_cost",
    "static_blocker",
    "change_control",
    "change_tuning",
    "change_mechanism",
    "change_architecture",
    *(f"axis_{axis}" for axis in MECHANISM_AXIS_FEATURES_V1),
    "core_change_fraction",
    "support_change_fraction",
    "primitive_delta_fraction",
    "parameter_delta_fraction",
    "architecture_delta",
    "custom_component_delta",
    "intervention_magnitude",
    "matched_control",
    "ablation_or_falsification",
    "compute_ordinal",
    "memory_ordinal",
    *(f"primitive_{name}_signed_delta" for name in PRIMITIVE_CLASS_FEATURES_V1),
    "axis_coverage_gap",
    "intervention_x_stagnation",
    "architecture_x_stagnation",
    "falsification_x_stagnation",
    "cost_x_budget_pressure",
    "compute_x_gpu_pressure",
    "intervention_x_duplicate_pressure",
    "blocker_x_blocker_pressure",
    "lineage_depth",
    "lineage_x_round",
    *(
        f"axis_{axis}_x_task_scale"
        for axis in MECHANISM_AXIS_FEATURES_V1
    ),
    *(
        f"axis_{axis}_x_task_density"
        for axis in MECHANISM_AXIS_FEATURES_V1
    ),
)


class MetaVNextRecord:
    schema = "recclaw.meta-vnext.record.v1"

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    @property
    def digest(self) -> str:
        return sha256_digest({"schema": self.schema, **self.to_dict()})


@dataclass(frozen=True, slots=True)
class ResearchValueWeightsV1(MetaVNextRecord):
    frontier: float
    discriminative: float
    cost: float
    blocker: float

    schema = "recclaw.meta-vnext.research-value-weights.v1"

    def __post_init__(self) -> None:
        values = (
            float(self.frontier),
            float(self.discriminative),
            float(self.cost),
            float(self.blocker),
        )
        if values[0] <= 0.0:
            raise MetaVNextContractError("frontier weight must be positive")
        if any(value < 0.0 for value in values[1:]):
            raise MetaVNextContractError("research-value weights cannot be negative")


@dataclass(frozen=True, slots=True)
class ResearchContextV1(MetaVNextRecord):
    round_fraction: float
    remaining_execution_fraction: float
    remaining_token_fraction: float
    remaining_gpu_fraction: float
    starting_frontier: float
    recent_frontier_gain: float
    stagnation_fraction: float
    axis_coverage: tuple[tuple[str, float], ...]
    exact_duplicate_count: int
    near_duplicate_count: int
    blocker_count: int
    lineage_depth: int
    task_scale: float = 0.0
    task_density: float = 0.0

    schema = "recclaw.meta-vnext.research-context.v1"

    def __post_init__(self) -> None:
        bounded = (
            self.round_fraction,
            self.remaining_execution_fraction,
            self.remaining_token_fraction,
            self.remaining_gpu_fraction,
            self.starting_frontier,
            self.stagnation_fraction,
            self.task_scale,
            self.task_density,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in bounded):
            raise MetaVNextContractError("context fractions must be in [0,1]")
        if not -1.0 <= float(self.recent_frontier_gain) <= 1.0:
            raise MetaVNextContractError("recent_frontier_gain must be in [-1,1]")
        if any(
            count < 0
            for count in (
                self.exact_duplicate_count,
                self.near_duplicate_count,
                self.blocker_count,
                self.lineage_depth,
            )
        ):
            raise MetaVNextContractError("context counts must be non-negative")
        coverage = tuple(
            sorted((str(axis), float(value)) for axis, value in self.axis_coverage)
        )
        if len({axis for axis, _ in coverage}) != len(coverage):
            raise MetaVNextContractError("axis coverage keys must be unique")
        if any(not 0.0 <= value <= 1.0 for _, value in coverage):
            raise MetaVNextContractError("axis coverage must be in [0,1]")
        object.__setattr__(self, "axis_coverage", coverage)

    def coverage_for(self, mechanism_axis: str) -> float:
        return dict(self.axis_coverage).get(str(mechanism_axis), 0.0)


@dataclass(frozen=True, slots=True)
class CandidateMechanismDeltaV1(MetaVNextRecord):
    candidate_id: str
    candidate_semantics_digest: str
    mechanism_program_digest: str
    parent_program_digest: str
    parent_semantics_digest: str
    producer_id: str
    producer_role: str
    proposal_intent: str
    mechanism_family_id: str
    primary_mechanism_axis: str
    secondary_mechanism_axes: tuple[str, ...]
    change_class: ChangeClassV1
    construction_mode: str
    core_changed_slots: tuple[str, ...]
    support_changed_slots: tuple[str, ...]
    added_primitives: tuple[str, ...]
    removed_primitives: tuple[str, ...]
    replaced_slots: tuple[str, ...]
    parameter_change_count: int
    architecture_operator_delta: int
    custom_component_delta: int
    intervention_magnitude: float
    lineage_depth: int
    matched_control: bool
    ablation_or_falsification: bool
    estimated_compute_class: str
    estimated_memory_class: str
    static_utility_features: SearchUtilityFeaturesV1
    strong_static_score: float | None
    route_eligibility: str
    hard_gate_reason: str
    rank_features: tuple[tuple[str, float], ...]

    schema = "recclaw.meta-vnext.candidate-mechanism-delta.v1"

    def __post_init__(self) -> None:
        for name in (
            "candidate_semantics_digest",
            "mechanism_program_digest",
            "parent_program_digest",
            "parent_semantics_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.parameter_change_count < 0 or self.lineage_depth < 0:
            raise MetaVNextContractError("candidate counts must be non-negative")
        if not 0.0 <= float(self.intervention_magnitude) <= 1.0:
            raise MetaVNextContractError("intervention_magnitude must be in [0,1]")
        if self.route_eligibility not in {"ELIGIBLE", "INELIGIBLE"}:
            raise MetaVNextContractError("route_eligibility is outside the closed domain")
        features = tuple((str(name), float(value)) for name, value in self.rank_features)
        if tuple(name for name, _ in features) != RANK_FEATURE_NAMES_V1:
            raise MetaVNextContractError("rank feature schema is not frozen V1")
        if self.strong_static_score is None and self.route_eligibility == "ELIGIBLE":
            raise MetaVNextContractError("eligible candidates require a static score")
        object.__setattr__(self, "secondary_mechanism_axes", tuple(self.secondary_mechanism_axes))
        object.__setattr__(self, "core_changed_slots", tuple(sorted(self.core_changed_slots)))
        object.__setattr__(
            self, "support_changed_slots", tuple(sorted(self.support_changed_slots))
        )
        object.__setattr__(self, "added_primitives", tuple(sorted(self.added_primitives)))
        object.__setattr__(self, "removed_primitives", tuple(sorted(self.removed_primitives)))
        object.__setattr__(self, "replaced_slots", tuple(sorted(self.replaced_slots)))
        object.__setattr__(self, "rank_features", features)

    @property
    def rank_feature_dict(self) -> dict[str, float]:
        return dict(self.rank_features)


@dataclass(frozen=True, slots=True)
class CandidatePoolVNextV1(MetaVNextRecord):
    pool_id: str
    source_candidate_pool_digest: str
    hard_gate_assessment_digest: str
    pre_round_state_digest: str
    producer_invocation_digests: tuple[str, ...]
    candidate_order_policy_digest: str
    research_context: ResearchContextV1
    candidates: tuple[CandidateMechanismDeltaV1, ...]
    rejected_candidate_ids: tuple[str, ...]

    schema = "recclaw.meta-vnext.candidate-pool.v1"

    def __post_init__(self) -> None:
        validate_sha256(
            self.source_candidate_pool_digest,
            field_name="source_candidate_pool_digest",
        )
        validate_sha256(
            self.hard_gate_assessment_digest,
            field_name="hard_gate_assessment_digest",
        )
        validate_sha256(
            self.pre_round_state_digest, field_name="pre_round_state_digest"
        )
        validate_sha256(
            self.candidate_order_policy_digest,
            field_name="candidate_order_policy_digest",
        )
        for index, digest in enumerate(self.producer_invocation_digests):
            validate_sha256(
                digest, field_name=f"producer_invocation_digests[{index}]"
            )
        candidates = tuple(self.candidates)
        if not candidates:
            raise MetaVNextContractError("candidate pool cannot be empty")
        ids = tuple(item.candidate_id for item in candidates)
        semantics = tuple(item.candidate_semantics_digest for item in candidates)
        if len(set(ids)) != len(ids) or len(set(semantics)) != len(semantics):
            raise MetaVNextContractError("candidate pool identities must be unique")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self, "rejected_candidate_ids", tuple(self.rejected_candidate_ids)
        )

    @property
    def feature_matrix_digest(self) -> str:
        return sha256_digest(
            [
                {
                    "candidate_semantics_digest": item.candidate_semantics_digest,
                    "rank_features": item.rank_features,
                }
                for item in self.candidates
            ]
        )

    @property
    def eligible_candidates(self) -> tuple[CandidateMechanismDeltaV1, ...]:
        return tuple(
            item for item in self.candidates if item.route_eligibility == "ELIGIBLE"
        )


@dataclass(frozen=True, slots=True)
class SearchValueObservationV1(MetaVNextRecord):
    candidate_semantics_digest: str
    source_search_utility_event_digest: str
    frontier_value: float
    discriminative_value: float
    normalized_cost: float
    blocker_loss: float
    round_boundary: int

    schema = "recclaw.meta-vnext.search-value-observation.v1"

    def __post_init__(self) -> None:
        validate_sha256(
            self.candidate_semantics_digest,
            field_name="candidate_semantics_digest",
        )
        validate_sha256(
            self.source_search_utility_event_digest,
            field_name="source_search_utility_event_digest",
        )
        if not -1.0 <= float(self.frontier_value) <= 1.0:
            raise MetaVNextContractError("frontier_value must be in [-1,1]")
        for name in ("discriminative_value", "normalized_cost", "blocker_loss"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise MetaVNextContractError(f"{name} must be in [0,1]")
        if self.round_boundary < 0:
            raise MetaVNextContractError("round_boundary must be non-negative")

    def net_research_value(self, weights: ResearchValueWeightsV1) -> float:
        return (
            float(weights.frontier) * float(self.frontier_value)
            + float(weights.discriminative) * float(self.discriminative_value)
            - float(weights.cost) * float(self.normalized_cost)
            - float(weights.blocker) * float(self.blocker_loss)
        )


@dataclass(frozen=True, slots=True)
class CompletePoolEpisodeV1(MetaVNextRecord):
    episode_id: str
    episode_group_id: str
    lineage_group_id: str
    sequence_index: int
    split: EpisodeSplitV1
    heldout_freshness: HeldoutFreshnessV1
    pool: CandidatePoolVNextV1
    observations: tuple[SearchValueObservationV1, ...]

    schema = "recclaw.meta-vnext.complete-pool-episode.v1"

    def __post_init__(self) -> None:
        if self.sequence_index < 1:
            raise MetaVNextContractError("sequence_index must be positive")
        if (
            self.split is EpisodeSplitV1.PROMOTION_HELDOUT
            and self.heldout_freshness is HeldoutFreshnessV1.DEVELOPMENT
        ):
            raise MetaVNextContractError("held-out episodes require explicit freshness")
        if (
            self.split is not EpisodeSplitV1.PROMOTION_HELDOUT
            and self.heldout_freshness is not HeldoutFreshnessV1.DEVELOPMENT
        ):
            raise MetaVNextContractError(
                "TRAIN and VALIDATION episodes are development data"
            )
        observations = tuple(self.observations)
        expected = {
            item.candidate_semantics_digest for item in self.pool.eligible_candidates
        }
        actual = {item.candidate_semantics_digest for item in observations}
        if expected != actual or len(actual) != len(observations):
            raise MetaVNextContractError(
                "complete-pool episode requires one outcome per eligible candidate"
            )
        if len(expected) < 2:
            raise MetaVNextContractError(
                "pairwise learning requires at least two eligible candidates"
            )
        object.__setattr__(self, "observations", observations)

    @property
    def outcomes_by_semantics(self) -> dict[str, SearchValueObservationV1]:
        return {
            item.candidate_semantics_digest: item for item in self.observations
        }
