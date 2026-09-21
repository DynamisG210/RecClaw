"""Deterministic slow and Arm-private fast routing for Meta VNext."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..canonical import canonical_value, sha256_digest, validate_sha256
from .contracts import (
    MECHANISM_AXIS_FEATURES_V1,
    CandidateMechanismDeltaV1,
    CandidatePoolVNextV1,
    MetaVNextRecord,
    SearchValueObservationV1,
)
from .features import FEATURE_SCHEMA_DIGEST_V1
from .learning import PairwiseSlowPolicyV1


class MetaVNextRoutingError(ValueError):
    """Raised when routing would violate pool or Arm-private state identity."""


FAST_FEATURE_NAMES_V1 = tuple(
    f"axis_{axis}" for axis in MECHANISM_AXIS_FEATURES_V1
)
FAST_UPDATE_RULE_DIGEST_V1 = sha256_digest(
    {
        "schema": "recclaw.meta-vnext.response-prototype-private-state.v6",
        "features": FAST_FEATURE_NAMES_V1,
        "feedback": "SearchValueObservationV1_ONLY",
        "adaptation": "TASK_CONDITIONAL_K_NEAREST_RESPONSE_PROTOTYPES",
        "sharing": "OPAQUE_ARM_X_SEARCH_SEED_PRIVATE",
    }
)
FAST_TASK_SCALE_STD_FLOOR_V1 = 0.05
FAST_TASK_DENSITY_STD_FLOOR_V1 = 0.05
FAST_CONTROL_AXIS_MISMATCH_PENALTY_V1 = 1.0
FAST_RESPONSE_SCALE_V1 = 0.1
ROUTER_IMPLEMENTATION_DIGEST_V1 = sha256_digest(
    {
        "schema": "recclaw.meta-vnext.router-implementation.v2",
        "slow_policy": "PAIRWISE_RIDGE_WITH_STATIC_OUT_OF_SUPPORT_FALLBACK",
        "task_support": "CLOSED_PROTOTYPE_SCALE_X_DENSITY_ENVELOPE",
        "fast_update_rule_digest": FAST_UPDATE_RULE_DIGEST_V1,
        "fast_task_scale_std_floor": FAST_TASK_SCALE_STD_FLOOR_V1,
        "fast_task_density_std_floor": FAST_TASK_DENSITY_STD_FLOOR_V1,
        "fast_control_axis_mismatch_penalty": (
            FAST_CONTROL_AXIS_MISMATCH_PENALTY_V1
        ),
        "fast_response_scale": FAST_RESPONSE_SCALE_V1,
    }
)


@dataclass(frozen=True, slots=True)
class FastResidualStateV1(MetaVNextRecord):
    opaque_arm_instance_digest: str
    search_seed_digest: str
    slow_policy_digest: str
    round_boundary: int
    feature_names: tuple[str, ...]
    observed_responses: tuple[tuple[int, str, float], ...]
    visible_search_value_event_digests: tuple[str, ...]
    update_rule_digest: str

    schema = "recclaw.meta-vnext.fast-residual-state.v1"

    def __post_init__(self) -> None:
        for name in (
            "opaque_arm_instance_digest",
            "search_seed_digest",
            "slow_policy_digest",
            "update_rule_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.update_rule_digest != FAST_UPDATE_RULE_DIGEST_V1:
            raise MetaVNextRoutingError("fast update rule is not frozen V1")
        if self.round_boundary < 0:
            raise MetaVNextRoutingError("round_boundary must be non-negative")
        if tuple(self.feature_names) != FAST_FEATURE_NAMES_V1:
            raise MetaVNextRoutingError("fast feature schema mismatch")
        if len(self.observed_responses) != len(
            self.visible_search_value_event_digests
        ):
            raise MetaVNextRoutingError(
                "fast responses and visible event digests must align"
            )
        previous_sequence_index = 0
        for sequence_index, axis, value in self.observed_responses:
            if (
                sequence_index <= previous_sequence_index
                or sequence_index > self.round_boundary
                or axis not in MECHANISM_AXIS_FEATURES_V1
            ):
                raise MetaVNextRoutingError(
                    "fast response history must be ordered, bounded, and semantic"
                )
            if not math.isfinite(float(value)):
                raise MetaVNextRoutingError("fast response value must be finite")
            previous_sequence_index = sequence_index
        for index, digest in enumerate(self.visible_search_value_event_digests):
            validate_sha256(
                digest,
                field_name=f"visible_search_value_event_digests[{index}]",
            )

def initialize_fast_residual(
    *,
    opaque_arm_instance_digest: str,
    search_seed_digest: str,
    policy: PairwiseSlowPolicyV1,
) -> FastResidualStateV1:
    validate_sha256(
        opaque_arm_instance_digest, field_name="opaque_arm_instance_digest"
    )
    validate_sha256(search_seed_digest, field_name="search_seed_digest")
    validate_sha256(policy.digest, field_name="slow_policy_digest")
    return FastResidualStateV1(
        opaque_arm_instance_digest=opaque_arm_instance_digest,
        search_seed_digest=search_seed_digest,
        slow_policy_digest=policy.digest,
        round_boundary=0,
        feature_names=FAST_FEATURE_NAMES_V1,
        observed_responses=(),
        visible_search_value_event_digests=(),
        update_rule_digest=FAST_UPDATE_RULE_DIGEST_V1,
    )


def _check_private_identity(
    state: FastResidualStateV1,
    *,
    opaque_arm_instance_digest: str,
    search_seed_digest: str,
    slow_policy_digest: str,
) -> None:
    if opaque_arm_instance_digest != state.opaque_arm_instance_digest:
        raise MetaVNextRoutingError("fast state cannot cross Arm instances")
    if search_seed_digest != state.search_seed_digest:
        raise MetaVNextRoutingError("fast state cannot cross Search Seeds")
    if slow_policy_digest != state.slow_policy_digest:
        raise MetaVNextRoutingError("B/C fast state must share the bound slow policy")


def advance_fast_without_observation(
    state: FastResidualStateV1,
    *,
    opaque_arm_instance_digest: str,
    search_seed_digest: str,
    slow_policy_digest: str,
    round_boundary: int,
) -> FastResidualStateV1:
    """Advance one boundary without encoding why an event was unavailable."""

    _check_private_identity(
        state,
        opaque_arm_instance_digest=opaque_arm_instance_digest,
        search_seed_digest=search_seed_digest,
        slow_policy_digest=slow_policy_digest,
    )
    if round_boundary != state.round_boundary + 1:
        raise MetaVNextRoutingError("fast state advances exactly one round boundary")
    return FastResidualStateV1(
        opaque_arm_instance_digest=state.opaque_arm_instance_digest,
        search_seed_digest=state.search_seed_digest,
        slow_policy_digest=state.slow_policy_digest,
        round_boundary=round_boundary,
        feature_names=state.feature_names,
        observed_responses=state.observed_responses,
        visible_search_value_event_digests=state.visible_search_value_event_digests,
        update_rule_digest=state.update_rule_digest,
    )


def update_fast_residual(
    state: FastResidualStateV1,
    *,
    policy: PairwiseSlowPolicyV1,
    candidate: CandidateMechanismDeltaV1,
    observation: SearchValueObservationV1,
    opaque_arm_instance_digest: str,
    search_seed_digest: str,
    round_boundary: int,
) -> FastResidualStateV1:
    _check_private_identity(
        state,
        opaque_arm_instance_digest=opaque_arm_instance_digest,
        search_seed_digest=search_seed_digest,
        slow_policy_digest=policy.digest,
    )
    if round_boundary != state.round_boundary + 1:
        raise MetaVNextRoutingError("fast state updates exactly once per round boundary")
    if observation.round_boundary != round_boundary:
        raise MetaVNextRoutingError("search value belongs to another round boundary")
    if (
        observation.candidate_semantics_digest
        != candidate.candidate_semantics_digest
    ):
        raise MetaVNextRoutingError("search value does not match selected candidate")
    axis = (
        candidate.primary_mechanism_axis
        if candidate.primary_mechanism_axis in MECHANISM_AXIS_FEATURES_V1
        else "other"
    )
    observed_value = observation.net_research_value(policy.value_weights)
    return FastResidualStateV1(
        opaque_arm_instance_digest=state.opaque_arm_instance_digest,
        search_seed_digest=state.search_seed_digest,
        slow_policy_digest=state.slow_policy_digest,
        round_boundary=round_boundary,
        feature_names=state.feature_names,
        observed_responses=(
            *state.observed_responses,
            (round_boundary, axis, observed_value),
        ),
        visible_search_value_event_digests=(
            *state.visible_search_value_event_digests,
            observation.digest,
        ),
        update_rule_digest=state.update_rule_digest,
    )


def fast_prediction(
    state: FastResidualStateV1,
    candidate: CandidateMechanismDeltaV1,
    policy: PairwiseSlowPolicyV1,
    *,
    neighbor_count: int,
    task_control_axis: str,
) -> tuple[float, float]:
    task_scale = sum(
        value
        for name, value in candidate.rank_features
        if name.endswith("_x_task_scale")
    )
    task_density = sum(
        value
        for name, value in candidate.rank_features
        if name.endswith("_x_task_density")
    )
    prototype_scales = tuple(
        row[2] for row in policy.fast_response_prototypes
    )
    prototype_densities = tuple(
        row[3] for row in policy.fast_response_prototypes
    )
    scale_mean = sum(prototype_scales) / len(prototype_scales)
    density_mean = sum(prototype_densities) / len(prototype_densities)
    scale_std = max(
        FAST_TASK_SCALE_STD_FLOOR_V1,
        math.sqrt(
            sum((value - scale_mean) ** 2 for value in prototype_scales)
            / len(prototype_scales)
        ),
    )
    density_std = max(
        FAST_TASK_DENSITY_STD_FLOOR_V1,
        math.sqrt(
            sum((value - density_mean) ** 2 for value in prototype_densities)
            / len(prototype_densities)
        ),
    )
    distances = []
    next_sequence = state.round_boundary + 1
    for (
        group_id,
        _,
        prototype_scale,
        prototype_density,
        prototype_control_axis,
        sequence,
    ) in policy.fast_response_prototypes:
        by_sequence = {
            sequence_index: dict(axis_values)
            for sequence_index, axis_values in sequence
        }
        if next_sequence not in by_sequence:
            continue
        squared_distance = (
            ((task_scale - prototype_scale) / scale_std) ** 2
            + ((task_density - prototype_density) / density_std) ** 2
            + (
                FAST_CONTROL_AXIS_MISMATCH_PENALTY_V1
                if task_control_axis != prototype_control_axis
                else 0.0
            )
        )
        comparable = True
        for sequence_index, axis, observed_value in state.observed_responses:
            if (
                sequence_index not in by_sequence
                or axis not in by_sequence[sequence_index]
            ):
                comparable = False
                break
            squared_distance += (
                (
                    observed_value - by_sequence[sequence_index][axis]
                )
                / FAST_RESPONSE_SCALE_V1
            ) ** 2
        if comparable:
            distances.append((squared_distance, group_id, by_sequence))
    neighbors = sorted(distances)[:neighbor_count]
    axis = candidate.primary_mechanism_axis
    values = tuple(
        sequence[next_sequence][axis]
        for _, _, sequence in neighbors
        if axis in sequence[next_sequence]
    )
    if not values:
        return 0.0, 0.0
    prediction = sum(values) / len(values)
    uncertainty = math.sqrt(
        sum((value - prediction) ** 2 for value in values) / len(values)
    )
    return prediction - policy.score(candidate), uncertainty


def task_context_supported(
    candidate: CandidateMechanismDeltaV1,
    policy: PairwiseSlowPolicyV1,
) -> bool:
    """Return whether task scale and density lie inside development support."""

    task_scale = sum(
        value
        for name, value in candidate.rank_features
        if name.endswith("_x_task_scale")
    )
    task_density = sum(
        value
        for name, value in candidate.rank_features
        if name.endswith("_x_task_density")
    )
    prototype_scales = tuple(
        row[2] for row in policy.fast_response_prototypes
    )
    prototype_densities = tuple(
        row[3] for row in policy.fast_response_prototypes
    )
    return (
        min(prototype_scales) <= task_scale <= max(prototype_scales)
        and min(prototype_densities)
        <= task_density
        <= max(prototype_densities)
    )


@dataclass(frozen=True, slots=True)
class CandidateRouteScoreV1(MetaVNextRecord):
    candidate_id: str
    candidate_semantics_digest: str
    slow_score: float
    fast_correction: float
    uncertainty: float
    final_score: float

    schema = "recclaw.meta-vnext.candidate-route-score.v1"


@dataclass(frozen=True, slots=True)
class MetaVNextRouteDecisionV1(MetaVNextRecord):
    pool_digest: str
    policy_digest: str
    fast_state_digest: str | None
    shadow_mode: bool
    scored_candidates: tuple[CandidateRouteScoreV1, ...]
    selected_candidate_id: str
    selected_candidate_semantics_digest: str

    schema = "recclaw.meta-vnext.route-decision.v1"

    def __post_init__(self) -> None:
        validate_sha256(self.pool_digest, field_name="pool_digest")
        validate_sha256(self.policy_digest, field_name="policy_digest")
        if self.fast_state_digest is not None:
            validate_sha256(
                self.fast_state_digest, field_name="fast_state_digest"
            )


@dataclass(frozen=True, slots=True)
class MetaVNextRouterV1:
    fast_neighbor_count: int = 5
    fast_override_margin: float = 0.005

    def __post_init__(self) -> None:
        if self.fast_neighbor_count < 1:
            raise MetaVNextRoutingError("fast neighbor count must be positive")
        if self.fast_override_margin < 0.0:
            raise MetaVNextRoutingError("fast override margin cannot be negative")

    @property
    def configuration_digest(self) -> str:
        return sha256_digest(
            {
                "implementation_digest": ROUTER_IMPLEMENTATION_DIGEST_V1,
                "configuration": canonical_value(self),
            }
        )

    def route(
        self,
        pool: CandidatePoolVNextV1,
        policy: PairwiseSlowPolicyV1,
        *,
        fast_state: FastResidualStateV1 | None = None,
        shadow_mode: bool = True,
    ) -> MetaVNextRouteDecisionV1:
        if policy.feature_schema_digest != FEATURE_SCHEMA_DIGEST_V1:
            raise MetaVNextRoutingError("policy feature schema is not frozen V1")
        if fast_state is not None and fast_state.slow_policy_digest != policy.digest:
            raise MetaVNextRoutingError("fast state is bound to another slow policy")
        task_supported = task_context_supported(
            pool.eligible_candidates[0],
            policy,
        )
        slow_scores = tuple(
            (
                policy.score(candidate)
                if task_supported
                else float(candidate.strong_static_score)
            )
            for candidate in pool.eligible_candidates
        )
        slow_winner_index = max(
            range(len(pool.eligible_candidates)),
            key=lambda index: (slow_scores[index], -index),
        )
        prototype_values: list[float] = []
        prototype_uncertainties: list[float] = []
        if fast_state is not None and task_supported:
            control_axes = {
                candidate.primary_mechanism_axis
                for candidate in pool.eligible_candidates
                if candidate.matched_control
            }
            if len(control_axes) != 1:
                raise MetaVNextRoutingError(
                    "fast routing requires exactly one matched control axis"
                )
            task_control_axis = next(iter(control_axes))
            for candidate in pool.eligible_candidates:
                correction, uncertainty = fast_prediction(
                    fast_state,
                    candidate,
                    policy,
                    neighbor_count=self.fast_neighbor_count,
                    task_control_axis=task_control_axis,
                )
                prototype_values.append(policy.score(candidate) + correction)
                prototype_uncertainties.append(uncertainty)
        override = False
        if prototype_values:
            prototype_winner_index = max(
                range(len(prototype_values)),
                key=lambda index: (prototype_values[index], -index),
            )
            override = (
                prototype_values[prototype_winner_index]
                - prototype_values[slow_winner_index]
                > self.fast_override_margin
            )
        scored: list[CandidateRouteScoreV1] = []
        for index, candidate in enumerate(pool.eligible_candidates):
            slow_score = slow_scores[index]
            slow_uncertainty = (
                policy.uncertainty(candidate) if task_supported else 0.0
            )
            correction = 0.0
            fast_uncertainty = 0.0
            if override:
                correction = prototype_values[index] - slow_score
                fast_uncertainty = prototype_uncertainties[index]
            uncertainty = math.sqrt(
                slow_uncertainty * slow_uncertainty
                + fast_uncertainty * fast_uncertainty
            )
            final_score = slow_score + correction
            scored.append(
                CandidateRouteScoreV1(
                    candidate_id=candidate.candidate_id,
                    candidate_semantics_digest=candidate.candidate_semantics_digest,
                    slow_score=round(slow_score, 15),
                    fast_correction=round(correction, 15),
                    uncertainty=round(uncertainty, 15),
                    final_score=round(final_score, 15),
                )
            )
        if not scored:
            raise MetaVNextRoutingError("pool has no eligible candidate")
        selected_index, selected = max(
            enumerate(scored),
            key=lambda item: (item[1].final_score, -item[0]),
        )
        del selected_index
        return MetaVNextRouteDecisionV1(
            pool_digest=pool.digest,
            policy_digest=policy.digest,
            fast_state_digest=fast_state.digest if fast_state is not None else None,
            shadow_mode=bool(shadow_mode),
            scored_candidates=tuple(scored),
            selected_candidate_id=selected.candidate_id,
            selected_candidate_semantics_digest=(
                selected.candidate_semantics_digest
            ),
        )


def static_champion_candidate(
    pool: CandidatePoolVNextV1,
) -> CandidateMechanismDeltaV1:
    """Select exactly the runtime static score winner after identical hard gates."""

    candidates = pool.eligible_candidates
    if not candidates:
        raise MetaVNextRoutingError("pool has no static-Champion candidate")
    return max(
        enumerate(candidates),
        key=lambda item: (
            float(item[1].strong_static_score),
            -item[0],
        ),
    )[1]
