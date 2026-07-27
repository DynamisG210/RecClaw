"""Equal-episode pairwise ridge learning for Meta VNext."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

from ..canonical import sha256_digest, validate_sha256
from .contracts import (
    FAST_ADAPTATION_FEATURES_V1,
    MECHANISM_AXIS_FEATURES_V1,
    RANK_FEATURE_NAMES_V1,
    CandidateMechanismDeltaV1,
    ChangeClassV1,
    CompletePoolEpisodeV1,
    EpisodeSplitV1,
    HeldoutFreshnessV1,
    MetaVNextRecord,
    ResearchValueWeightsV1,
)
from .features import FEATURE_SCHEMA_DIGEST_V1


class MetaVNextLearningError(ValueError):
    """Raised when pairwise training lacks valid, separated support."""


def _zero_matrix(size: int) -> list[list[float]]:
    return [[0.0 for _ in range(size)] for _ in range(size)]


def _solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    size = len(rhs)
    augmented = [
        [float(value) for value in matrix[row]] + [float(rhs[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise MetaVNextLearningError("ridge system is singular")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][index] - factor * augmented[column][index]
                for index in range(size + 1)
            ]
    return [augmented[row][-1] for row in range(size)]


def _inverse_diagonal(matrix: Sequence[Sequence[float]]) -> list[float]:
    size = len(matrix)
    diagonal: list[float] = []
    for column in range(size):
        unit = [0.0] * size
        unit[column] = 1.0
        solution = _solve(matrix, unit)
        diagonal.append(max(0.0, solution[column]))
    return diagonal


def _candidate_vector(candidate: CandidateMechanismDeltaV1) -> tuple[float, ...]:
    return tuple(float(value) for _, value in candidate.rank_features)


def _episode_targets(
    episode: CompletePoolEpisodeV1,
    weights: ResearchValueWeightsV1,
) -> dict[str, float]:
    return {
        digest: observation.net_research_value(weights)
        for digest, observation in episode.outcomes_by_semantics.items()
    }


def _pairwise_system(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
    static_coefficient: float,
) -> tuple[list[list[float]], list[float], int]:
    size = len(RANK_FEATURE_NAMES_V1)
    static_index = RANK_FEATURE_NAMES_V1.index("strong_static_score")
    matrix = _zero_matrix(size)
    rhs = [0.0] * size
    pair_count = 0
    for episode in episodes:
        candidates = episode.pool.eligible_candidates
        targets = _episode_targets(episode, weights)
        pairs = tuple(combinations(candidates, 2))
        episode_weight = 1.0 / len(pairs)
        for left, right in pairs:
            left_vector = _candidate_vector(left)
            right_vector = _candidate_vector(right)
            difference = [
                left_vector[index] - right_vector[index] for index in range(size)
            ]
            static_difference = difference[static_index]
            difference[static_index] = 0.0
            target = (
                targets[left.candidate_semantics_digest]
                - targets[right.candidate_semantics_digest]
                - float(static_coefficient) * static_difference
            )
            for row in range(size):
                rhs[row] += episode_weight * difference[row] * target
                for column in range(size):
                    matrix[row][column] += (
                        episode_weight * difference[row] * difference[column]
                    )
            pair_count += 1
    return matrix, rhs, pair_count


def _fit_coefficients(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
    regularization: float,
    static_coefficient: float,
) -> tuple[tuple[float, ...], list[list[float]], int]:
    matrix, rhs, pair_count = _pairwise_system(
        episodes,
        weights,
        static_coefficient,
    )
    static_index = RANK_FEATURE_NAMES_V1.index("strong_static_score")
    for index in range(len(matrix)):
        matrix[index][index] += float(regularization)
    rhs[static_index] += float(regularization) * float(static_coefficient)
    coefficients = tuple(_solve(matrix, rhs))
    return coefficients, matrix, pair_count


def _fit_intercept(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
    coefficients: Sequence[float],
) -> float:
    residual_total = 0.0
    for episode in episodes:
        targets = _episode_targets(episode, weights)
        candidate_residuals = [
            targets[candidate.candidate_semantics_digest]
            - sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients, _candidate_vector(candidate), strict=True
                )
            )
            for candidate in episode.pool.eligible_candidates
        ]
        residual_total += sum(candidate_residuals) / len(candidate_residuals)
    return residual_total / len(episodes)


def _fit_fast_value_head(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
    regularization: float,
) -> tuple[float, tuple[float, ...]]:
    size = len(RANK_FEATURE_NAMES_V1) + 1
    matrix = _zero_matrix(size)
    rhs = [0.0] * size
    for episode in episodes:
        targets = _episode_targets(episode, weights)
        episode_weight = 1.0 / len(episode.pool.eligible_candidates)
        for candidate in episode.pool.eligible_candidates:
            vector = (1.0, *_candidate_vector(candidate))
            target = targets[candidate.candidate_semantics_digest]
            for row in range(size):
                rhs[row] += episode_weight * vector[row] * target
                for column in range(size):
                    matrix[row][column] += (
                        episode_weight * vector[row] * vector[column]
                    )
    matrix[0][0] += 1e-9
    for index in range(1, size):
        matrix[index][index] += float(regularization)
    solution = _solve(matrix, rhs)
    return float(solution[0]), tuple(float(value) for value in solution[1:])


def _linear_value(
    candidate: CandidateMechanismDeltaV1,
    intercept: float,
    coefficients: Sequence[float],
) -> float:
    return float(intercept) + sum(
        coefficient * value
        for coefficient, value in zip(
            coefficients,
            _candidate_vector(candidate),
            strict=True,
        )
    )


def _build_fast_response_prototypes(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
) -> tuple[
    tuple[
        str,
        str,
        float,
        float,
        str,
        tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
    ],
    ...,
]:
    groups: dict[str, list[CompletePoolEpisodeV1]] = {}
    for episode in episodes:
        groups.setdefault(episode.episode_group_id, []).append(episode)
    prototypes = []
    for group_id, group_episodes in sorted(groups.items()):
        lineages = {episode.lineage_group_id for episode in group_episodes}
        if len(lineages) != 1:
            raise MetaVNextLearningError(
                "fast prototype group must have one lineage"
            )
        ordered = sorted(group_episodes, key=lambda item: item.sequence_index)
        if [episode.sequence_index for episode in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise MetaVNextLearningError(
                "fast prototype sequence must be contiguous"
            )
        first_candidates = ordered[0].pool.eligible_candidates
        task_scales = {
            round(
                sum(
                    value
                    for name, value in candidate.rank_features
                    if name.endswith("_x_task_scale")
                ),
                15,
            )
            for candidate in first_candidates
        }
        task_densities = {
            round(
                sum(
                    value
                    for name, value in candidate.rank_features
                    if name.endswith("_x_task_density")
                ),
                15,
            )
            for candidate in first_candidates
        }
        control_axes = {
            candidate.primary_mechanism_axis
            for candidate in first_candidates
            if candidate.matched_control
        }
        if (
            len(task_scales) != 1
            or len(task_densities) != 1
            or len(control_axes) != 1
        ):
            raise MetaVNextLearningError(
                "fast prototype requires one task context and control axis"
            )
        sequence_rows = []
        for episode in ordered:
            axis_values = tuple(
                sorted(
                    (
                        candidate.primary_mechanism_axis,
                        episode.outcomes_by_semantics[
                            candidate.candidate_semantics_digest
                        ].net_research_value(weights),
                    )
                    for candidate in episode.pool.eligible_candidates
                )
            )
            if len({axis for axis, _ in axis_values}) != len(axis_values):
                raise MetaVNextLearningError(
                    "fast prototypes require one candidate per mechanism axis"
                )
            sequence_rows.append((episode.sequence_index, axis_values))
        prototypes.append(
            (
                group_id,
                next(iter(lineages)),
                next(iter(task_scales)),
                next(iter(task_densities)),
                next(iter(control_axes)),
                tuple(sequence_rows),
            )
        )
    return tuple(prototypes)


def _validation_metrics(
    episodes: Sequence[CompletePoolEpisodeV1],
    weights: ResearchValueWeightsV1,
    coefficients: Sequence[float],
    intercept: float,
) -> tuple[float, float, float]:
    regrets: list[float] = []
    correct = 0
    non_tied = 0
    squared_errors: list[float] = []
    for episode in episodes:
        candidates = episode.pool.eligible_candidates
        targets = _episode_targets(episode, weights)
        scores = {
            candidate.candidate_semantics_digest: intercept
            + sum(
                coefficient * value
                for coefficient, value in zip(
                    coefficients, _candidate_vector(candidate), strict=True
                )
            )
            for candidate in candidates
        }
        selected = max(
            enumerate(candidates),
            key=lambda item: (
                scores[item[1].candidate_semantics_digest],
                -item[0],
            ),
        )[1]
        regrets.append(
            max(targets.values()) - targets[selected.candidate_semantics_digest]
        )
        for left, right in combinations(candidates, 2):
            target_difference = (
                targets[left.candidate_semantics_digest]
                - targets[right.candidate_semantics_digest]
            )
            score_difference = (
                scores[left.candidate_semantics_digest]
                - scores[right.candidate_semantics_digest]
            )
            squared_errors.append((target_difference - score_difference) ** 2)
            if not math.isclose(target_difference, 0.0, abs_tol=1e-12):
                non_tied += 1
                if target_difference * score_difference > 0.0:
                    correct += 1
    return (
        sum(regrets) / len(regrets),
        correct / non_tied if non_tied else 1.0,
        math.sqrt(sum(squared_errors) / len(squared_errors)),
    )


@dataclass(frozen=True, slots=True)
class PairwiseSlowPolicyV1(MetaVNextRecord):
    policy_id: str
    parent_checkpoint_digest: str
    feature_schema_digest: str
    static_router_policy_digest: str
    value_weights: ResearchValueWeightsV1
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    covariance_diagonal: tuple[float, ...]
    fast_value_intercept: float
    fast_value_coefficients: tuple[float, ...]
    fast_feature_names: tuple[str, ...]
    fast_response_prototypes: tuple[
        tuple[
            str,
            str,
            float,
            float,
            str,
            tuple[tuple[int, tuple[tuple[str, float], ...]], ...],
        ],
        ...,
    ]
    selected_static_coefficient: float
    selected_regularization: float
    training_episode_set_digest: str
    validation_episode_set_digest: str
    fast_prototype_episode_set_digest: str
    development_group_ids: tuple[str, ...]
    development_lineage_group_ids: tuple[str, ...]
    development_semantics_digests: tuple[str, ...]
    development_parent_semantics_digests: tuple[str, ...]
    selection_metrics: tuple[
        tuple[float, float, float, float, float], ...
    ]

    schema = "recclaw.meta-vnext.pairwise-slow-policy.v1"

    def __post_init__(self) -> None:
        for name in (
            "parent_checkpoint_digest",
            "feature_schema_digest",
            "static_router_policy_digest",
            "training_episode_set_digest",
            "validation_episode_set_digest",
            "fast_prototype_episode_set_digest",
        ):
            validate_sha256(str(getattr(self, name)), field_name=name)
        if self.feature_schema_digest != FEATURE_SCHEMA_DIGEST_V1:
            raise MetaVNextLearningError("policy uses an unknown feature schema")
        if tuple(self.feature_names) != RANK_FEATURE_NAMES_V1:
            raise MetaVNextLearningError("policy feature names are not frozen V1")
        if len(self.coefficients) != len(self.feature_names):
            raise MetaVNextLearningError("coefficient dimension mismatch")
        if len(self.covariance_diagonal) != len(self.feature_names):
            raise MetaVNextLearningError("covariance dimension mismatch")
        if len(self.fast_value_coefficients) != len(self.feature_names):
            raise MetaVNextLearningError("fast value-head dimension mismatch")
        if tuple(self.fast_feature_names) != FAST_ADAPTATION_FEATURES_V1:
            raise MetaVNextLearningError("fast feature schema mismatch")
        prototype_groups = set()
        prototype_lineages = set()
        for (
            group_id,
            lineage_id,
            task_scale,
            task_density,
            control_axis,
            sequence,
        ) in self.fast_response_prototypes:
            if group_id in prototype_groups or lineage_id in prototype_lineages:
                raise MetaVNextLearningError(
                    "fast prototype groups and lineages must be unique"
                )
            prototype_groups.add(group_id)
            prototype_lineages.add(lineage_id)
            if (
                not math.isfinite(task_scale)
                or not math.isfinite(task_density)
                or control_axis not in MECHANISM_AXIS_FEATURES_V1
            ):
                raise MetaVNextLearningError(
                    "fast prototype task context is invalid"
                )
            if not sequence or [row[0] for row in sequence] != list(
                range(1, len(sequence) + 1)
            ):
                raise MetaVNextLearningError(
                    "fast prototype sequence must be contiguous"
                )
            for _, axis_values in sequence:
                axes = tuple(axis for axis, _ in axis_values)
                if len(set(axes)) != len(axes) or any(
                    axis not in MECHANISM_AXIS_FEATURES_V1 for axis in axes
                ):
                    raise MetaVNextLearningError(
                        "fast prototype axis schema mismatch"
                    )
        if self.selected_regularization <= 0.0:
            raise MetaVNextLearningError("regularization must be positive")
        if self.selected_static_coefficient not in {0.0, 1.0}:
            raise MetaVNextLearningError(
                "static coefficient must be a selected on/off anchor"
            )
        for name, values in (
            ("development_semantics_digests", self.development_semantics_digests),
            (
                "development_parent_semantics_digests",
                self.development_parent_semantics_digests,
            ),
        ):
            for index, digest in enumerate(values):
                validate_sha256(digest, field_name=f"{name}[{index}]")

    def score(self, candidate: CandidateMechanismDeltaV1) -> float:
        if tuple(name for name, _ in candidate.rank_features) != self.feature_names:
            raise MetaVNextLearningError("candidate feature schema mismatch")
        return float(self.intercept) + sum(
            coefficient * value
            for coefficient, (_, value) in zip(
                self.coefficients, candidate.rank_features, strict=True
            )
        )

    def uncertainty(self, candidate: CandidateMechanismDeltaV1) -> float:
        variance = sum(
            covariance * float(value) ** 2
            for covariance, (_, value) in zip(
                self.covariance_diagonal,
                candidate.rank_features,
                strict=True,
            )
        )
        return math.sqrt(max(0.0, variance))

    def fast_value(self, candidate: CandidateMechanismDeltaV1) -> float:
        if tuple(name for name, _ in candidate.rank_features) != self.feature_names:
            raise MetaVNextLearningError("candidate feature schema mismatch")
        return float(self.fast_value_intercept) + sum(
            coefficient * value
            for coefficient, (_, value) in zip(
                self.fast_value_coefficients,
                candidate.rank_features,
                strict=True,
            )
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PairwiseSlowPolicyV1":
        return cls(
            policy_id=str(payload["policy_id"]),
            parent_checkpoint_digest=str(payload["parent_checkpoint_digest"]),
            feature_schema_digest=str(payload["feature_schema_digest"]),
            static_router_policy_digest=str(payload["static_router_policy_digest"]),
            value_weights=ResearchValueWeightsV1(**payload["value_weights"]),
            feature_names=tuple(str(item) for item in payload["feature_names"]),
            coefficients=tuple(float(item) for item in payload["coefficients"]),
            intercept=float(payload["intercept"]),
            covariance_diagonal=tuple(
                float(item) for item in payload["covariance_diagonal"]
            ),
            fast_value_intercept=float(payload["fast_value_intercept"]),
            fast_value_coefficients=tuple(
                float(item) for item in payload["fast_value_coefficients"]
            ),
            fast_feature_names=tuple(
                str(item) for item in payload["fast_feature_names"]
            ),
            fast_response_prototypes=tuple(
                (
                    str(group_id),
                    str(lineage_id),
                    float(task_scale),
                    float(task_density),
                    str(control_axis),
                    tuple(
                        (
                            int(sequence_index),
                            tuple(
                                (str(axis), float(value))
                                for axis, value in axis_values
                            ),
                        )
                        for sequence_index, axis_values in sequence
                    ),
                )
                for (
                    group_id,
                    lineage_id,
                    task_scale,
                    task_density,
                    control_axis,
                    sequence,
                ) in payload[
                    "fast_response_prototypes"
                ]
            ),
            selected_static_coefficient=float(
                payload["selected_static_coefficient"]
            ),
            selected_regularization=float(payload["selected_regularization"]),
            training_episode_set_digest=str(payload["training_episode_set_digest"]),
            validation_episode_set_digest=str(
                payload["validation_episode_set_digest"]
            ),
            fast_prototype_episode_set_digest=str(
                payload["fast_prototype_episode_set_digest"]
            ),
            development_group_ids=tuple(
                str(item) for item in payload["development_group_ids"]
            ),
            development_lineage_group_ids=tuple(
                str(item)
                for item in payload["development_lineage_group_ids"]
            ),
            development_semantics_digests=tuple(
                str(item) for item in payload["development_semantics_digests"]
            ),
            development_parent_semantics_digests=tuple(
                str(item)
                for item in payload["development_parent_semantics_digests"]
            ),
            selection_metrics=tuple(
                tuple(float(value) for value in item)
                for item in payload["selection_metrics"]
            ),
        )


@dataclass(frozen=True, slots=True)
class PairwiseRidgeTrainerV1:
    regularization_grid: tuple[float, ...] = (
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
    )
    static_coefficient_grid: tuple[float, ...] = (0.0, 1.0)

    def __post_init__(self) -> None:
        grid = tuple(sorted({float(item) for item in self.regularization_grid}))
        if not grid or any(item <= 0.0 for item in grid):
            raise MetaVNextLearningError("regularization grid must be positive")
        object.__setattr__(self, "regularization_grid", grid)
        static_grid = tuple(
            sorted({float(item) for item in self.static_coefficient_grid})
        )
        if not static_grid or any(item not in {0.0, 1.0} for item in static_grid):
            raise MetaVNextLearningError(
                "static coefficient grid supports only frozen on/off anchors"
            )
        object.__setattr__(self, "static_coefficient_grid", static_grid)

    def fit(
        self,
        *,
        training_episodes: Sequence[CompletePoolEpisodeV1],
        validation_episodes: Sequence[CompletePoolEpisodeV1],
        value_weights: ResearchValueWeightsV1,
        parent_checkpoint_digest: str,
        static_router_policy_digest: str,
        policy_id: str = "RESEARCH_META_VNEXT_SLOW_V1",
        fast_prototype_episodes: Sequence[CompletePoolEpisodeV1] = (),
    ) -> PairwiseSlowPolicyV1:
        training = tuple(training_episodes)
        validation = tuple(validation_episodes)
        if not training or not validation:
            raise MetaVNextLearningError(
                "training and validation episode sets must be non-empty"
            )
        if any(item.split is not EpisodeSplitV1.TRAIN for item in training):
            raise MetaVNextLearningError("training fit accepts TRAIN episodes only")
        if any(item.split is not EpisodeSplitV1.VALIDATION for item in validation):
            raise MetaVNextLearningError(
                "model selection accepts VALIDATION episodes only"
            )
        extra_fast_prototypes = tuple(fast_prototype_episodes)
        if any(
            item.heldout_freshness is not HeldoutFreshnessV1.DEVELOPMENT
            or item.split
            not in {EpisodeSplitV1.TRAIN, EpisodeSplitV1.VALIDATION}
            for item in extra_fast_prototypes
        ):
            raise MetaVNextLearningError(
                "fast prototype support must be development episodes"
            )
        train_groups = {item.episode_group_id for item in training}
        validation_groups = {item.episode_group_id for item in validation}
        train_lineages = {item.lineage_group_id for item in training}
        validation_lineages = {item.lineage_group_id for item in validation}
        if train_groups & validation_groups or train_lineages & validation_lineages:
            raise MetaVNextLearningError(
                "training and validation groups/lineages must be disjoint"
            )
        extra_groups = {
            item.episode_group_id for item in extra_fast_prototypes
        }
        extra_lineages = {
            item.lineage_group_id for item in extra_fast_prototypes
        }
        if (
            extra_groups & (train_groups | validation_groups)
            or extra_lineages & (train_lineages | validation_lineages)
        ):
            raise MetaVNextLearningError(
                "extra fast prototype support must use disjoint task groups"
            )
        training_semantics = {
            candidate.candidate_semantics_digest
            for episode in training
            for candidate in episode.pool.eligible_candidates
        }
        validation_semantics = {
            candidate.candidate_semantics_digest
            for episode in validation
            for candidate in episode.pool.eligible_candidates
        }
        training_parents = {
            candidate.parent_semantics_digest
            for episode in training
            for candidate in episode.pool.eligible_candidates
        }
        validation_parents = {
            candidate.parent_semantics_digest
            for episode in validation
            for candidate in episode.pool.eligible_candidates
        }
        extra_semantics = {
            candidate.candidate_semantics_digest
            for episode in extra_fast_prototypes
            for candidate in episode.pool.eligible_candidates
        }
        extra_parents = {
            candidate.parent_semantics_digest
            for episode in extra_fast_prototypes
            for candidate in episode.pool.eligible_candidates
        }
        development_change_classes = {
            candidate.change_class
            for episode in (*training, *validation, *extra_fast_prototypes)
            for candidate in episode.pool.eligible_candidates
        }
        if not development_change_classes & {
            ChangeClassV1.MECHANISM_CHANGE,
            ChangeClassV1.ARCHITECTURE_REWRITE,
        }:
            raise MetaVNextLearningError(
                "tuning-only development data cannot train Meta VNext"
            )
        validate_sha256(
            parent_checkpoint_digest, field_name="parent_checkpoint_digest"
        )
        validate_sha256(
            static_router_policy_digest,
            field_name="static_router_policy_digest",
        )

        candidates: list[
            tuple[
                tuple[float, float, float, float, float],
                tuple[float, ...],
            ]
        ] = []
        for static_coefficient in self.static_coefficient_grid:
            for regularization in self.regularization_grid:
                coefficients, _, _ = _fit_coefficients(
                    training,
                    value_weights,
                    regularization,
                    static_coefficient,
                )
                intercept = _fit_intercept(
                    training,
                    value_weights,
                    coefficients,
                )
                regret, accuracy, rmse = _validation_metrics(
                    validation,
                    value_weights,
                    coefficients,
                    intercept,
                )
                candidates.append(
                    (
                        (
                            float(static_coefficient),
                            float(regularization),
                            float(regret),
                            float(accuracy),
                            float(rmse),
                        ),
                        coefficients,
                    )
                )
        selected_metrics, _ = min(
            candidates,
            key=lambda item: (
                item[0][2],
                -item[0][3],
                item[0][4],
                -item[0][0],
                item[0][1],
            ),
        )
        selected_static_coefficient = selected_metrics[0]
        selected_regularization = selected_metrics[1]
        refit_episodes = (*training, *validation)
        coefficients, ridge_matrix, pair_count = _fit_coefficients(
            refit_episodes,
            value_weights,
            selected_regularization,
            selected_static_coefficient,
        )
        intercept = _fit_intercept(refit_episodes, value_weights, coefficients)
        pairwise_mse = 0.0
        pairwise_weight = 0.0
        for episode in refit_episodes:
            targets = _episode_targets(episode, value_weights)
            pairs = tuple(combinations(episode.pool.eligible_candidates, 2))
            weight = 1.0 / len(pairs)
            for left, right in pairs:
                predicted = sum(
                    coefficient * (left_value - right_value)
                    for coefficient, left_value, right_value in zip(
                        coefficients,
                        _candidate_vector(left),
                        _candidate_vector(right),
                        strict=True,
                    )
                )
                observed = (
                    targets[left.candidate_semantics_digest]
                    - targets[right.candidate_semantics_digest]
                )
                pairwise_mse += weight * (observed - predicted) ** 2
                pairwise_weight += weight
        residual_variance = max(
            1e-9,
            pairwise_mse / max(1.0, pairwise_weight),
        )
        covariance_values = [
            residual_variance * value for value in _inverse_diagonal(ridge_matrix)
        ]
        covariance_values[
            RANK_FEATURE_NAMES_V1.index("strong_static_score")
        ] = 0.0
        covariance_diagonal = tuple(covariance_values)
        fast_value_intercept, fast_value_coefficients = _fit_fast_value_head(
            refit_episodes,
            value_weights,
            selected_regularization,
        )
        prototype_episodes = (*refit_episodes, *extra_fast_prototypes)
        fast_response_prototypes = _build_fast_response_prototypes(
            prototype_episodes,
            value_weights,
        )
        training_digest = sha256_digest(
            [item.digest for item in sorted(training, key=lambda item: item.episode_id)]
        )
        validation_digest = sha256_digest(
            [
                item.digest
                for item in sorted(validation, key=lambda item: item.episode_id)
            ]
        )
        fast_prototype_digest = sha256_digest(
            [
                item.digest
                for item in sorted(
                    prototype_episodes,
                    key=lambda item: item.episode_id,
                )
            ]
        )
        metrics = tuple(
            (
                round(item[0][0], 12),
                round(item[0][1], 12),
                round(item[0][2], 12),
                round(item[0][3], 12),
                round(item[0][4], 12),
            )
            for item in candidates
        )
        if pair_count == 0:
            raise MetaVNextLearningError("pairwise training has no candidate pairs")
        return PairwiseSlowPolicyV1(
            policy_id=policy_id,
            parent_checkpoint_digest=parent_checkpoint_digest,
            feature_schema_digest=FEATURE_SCHEMA_DIGEST_V1,
            static_router_policy_digest=static_router_policy_digest,
            value_weights=value_weights,
            feature_names=RANK_FEATURE_NAMES_V1,
            coefficients=tuple(round(item, 15) for item in coefficients),
            intercept=round(intercept, 15),
            covariance_diagonal=tuple(
                round(item, 15) for item in covariance_diagonal
            ),
            fast_value_intercept=round(fast_value_intercept, 15),
            fast_value_coefficients=tuple(
                round(item, 15) for item in fast_value_coefficients
            ),
            fast_feature_names=FAST_ADAPTATION_FEATURES_V1,
            fast_response_prototypes=fast_response_prototypes,
            selected_static_coefficient=selected_static_coefficient,
            selected_regularization=selected_regularization,
            training_episode_set_digest=training_digest,
            validation_episode_set_digest=validation_digest,
            fast_prototype_episode_set_digest=fast_prototype_digest,
            development_group_ids=tuple(
                sorted(train_groups | validation_groups | extra_groups)
            ),
            development_lineage_group_ids=tuple(
                sorted(train_lineages | validation_lineages | extra_lineages)
            ),
            development_semantics_digests=tuple(
                sorted(
                    training_semantics
                    | validation_semantics
                    | extra_semantics
                )
            ),
            development_parent_semantics_digests=tuple(
                sorted(training_parents | validation_parents | extra_parents)
            ),
            selection_metrics=metrics,
        )
