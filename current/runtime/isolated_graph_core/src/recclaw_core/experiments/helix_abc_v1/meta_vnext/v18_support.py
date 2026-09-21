"""Support-aware projection for the V18 Meta policy successor."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..canonical import canonical_value, sha256_digest
from .contracts import (
    RANK_FEATURE_NAMES_V1,
    CandidateMechanismDeltaV1,
    CandidatePoolVNextV1,
)
from .features import FEATURE_SCHEMA_DIGEST_V1
from .learning import PairwiseSlowPolicyV1
from .routing import (
    CandidateRouteScoreV1,
    FastResidualStateV1,
    MetaVNextRouteDecisionV1,
    MetaVNextRouterV1,
    MetaVNextRoutingError,
    fast_prediction,
)


SUPPORT_RESOURCE = (
    Path(__file__).resolve().parents[1]
    / "resources"
    / "meta_vnext_feature_support_v18.json"
)
SLOW_PROJECTION_ID_V18 = "CLIP_EACH_FEATURE_TO_FROZEN_DEVELOPMENT_RANGE"
FAST_SUPPORT_ID_V18 = (
    "ACTUAL_TASK_IN_PROTOTYPE_ENVELOPE_AND_ALL_POOL_AXES_CALIBRATED"
)
ROUTER_POLICY_ID_V18 = "META_VNEXT_SUPPORT_AWARE_ROUTER_V18"


class MetaV18SupportError(ValueError):
    """Raised when V18 would score outside its frozen support contract."""


@dataclass(frozen=True, slots=True)
class MetaV18SupportProjectionV1:
    actual_task_scale: float
    actual_task_density: float
    task_scale_support: tuple[float, float]
    task_density_support: tuple[float, float]
    calibrated_axes: tuple[str, ...]
    pool_axes: tuple[str, ...]
    generic_transfer_axes: tuple[str, ...]
    task_context_supported: bool
    fast_supported: bool
    fast_support_reason: str
    clipped_feature_count: int
    projected_feature_matrix_digest: str
    support_resource_sha256: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@lru_cache(maxsize=1)
def load_feature_support() -> dict[str, Any]:
    payload = json.loads(SUPPORT_RESOURCE.read_text(encoding="utf-8"))
    names = tuple(str(row[0]) for row in payload["feature_bounds"])
    if (
        payload.get("schema") != "recclaw.meta-vnext-feature-support.v18"
        or payload.get("feature_schema_digest") != FEATURE_SCHEMA_DIGEST_V1
        or names != RANK_FEATURE_NAMES_V1
        or payload.get("slow_context_projection") != SLOW_PROJECTION_ID_V18
        or payload.get("fast_context_projection") != "NONE_EXACT_SUPPORT_ONLY"
    ):
        raise MetaV18SupportError("V18 feature-support resource is not exact")
    return payload


def feature_support_sha256() -> str:
    return hashlib.sha256(SUPPORT_RESOURCE.read_bytes()).hexdigest()


def project_candidate_for_slow(
    candidate: CandidateMechanismDeltaV1,
) -> tuple[CandidateMechanismDeltaV1, int]:
    support = load_feature_support()
    bounds = {
        str(name): (float(lower), float(upper))
        for name, lower, upper in support["feature_bounds"]
    }
    clipped = []
    changed = 0
    for name, raw_value in candidate.rank_features:
        lower, upper = bounds[name]
        value = min(upper, max(lower, float(raw_value)))
        changed += int(value != float(raw_value))
        clipped.append((name, value))
    return replace(candidate, rank_features=tuple(clipped)), changed


def _task_support(
    *,
    policy: PairwiseSlowPolicyV1,
    task_scale: float,
    task_density: float,
) -> tuple[bool, tuple[float, float], tuple[float, float]]:
    scales = tuple(float(row[2]) for row in policy.fast_response_prototypes)
    densities = tuple(float(row[3]) for row in policy.fast_response_prototypes)
    scale_bounds = (min(scales), max(scales))
    density_bounds = (min(densities), max(densities))
    supported = (
        scale_bounds[0] <= float(task_scale) <= scale_bounds[1]
        and density_bounds[0] <= float(task_density) <= density_bounds[1]
    )
    return supported, scale_bounds, density_bounds


def route_support_aware(
    *,
    pool: CandidatePoolVNextV1,
    policy: PairwiseSlowPolicyV1,
    router: MetaVNextRouterV1,
    fast_state: FastResidualStateV1 | None,
    actual_task_scale: float,
    actual_task_density: float,
    shadow_mode: bool = False,
) -> tuple[MetaVNextRouteDecisionV1, MetaV18SupportProjectionV1]:
    """Score one frozen pool without extrapolating learned features."""

    if policy.feature_schema_digest != FEATURE_SCHEMA_DIGEST_V1:
        raise MetaV18SupportError("V18 requires the frozen rank feature schema")
    if fast_state is not None and fast_state.slow_policy_digest != policy.digest:
        raise MetaV18SupportError("fast state belongs to another slow policy")
    candidates = pool.eligible_candidates
    if not candidates:
        raise MetaV18SupportError("V18 pool has no eligible candidate")
    projected_rows = tuple(project_candidate_for_slow(item) for item in candidates)
    projected = tuple(item for item, _ in projected_rows)
    clipped_feature_count = sum(count for _, count in projected_rows)
    support = load_feature_support()
    calibrated_axes = tuple(str(item) for item in support["calibrated_mechanism_axes"])
    pool_axes = tuple(sorted({item.primary_mechanism_axis for item in candidates}))
    generic_axes = tuple(sorted(set(pool_axes) - set(calibrated_axes)))
    task_supported, scale_bounds, density_bounds = _task_support(
        policy=policy,
        task_scale=actual_task_scale,
        task_density=actual_task_density,
    )
    fast_supported = (
        fast_state is not None and task_supported and not generic_axes
    )
    if fast_state is None:
        fast_reason = "FAST_STATE_ABSENT"
    elif not task_supported:
        fast_reason = "ACTUAL_TASK_CONTEXT_OUT_OF_SUPPORT"
    elif generic_axes:
        fast_reason = "POOL_CONTAINS_UNCALIBRATED_AXIS"
    else:
        fast_reason = "FAST_SUPPORTED"

    slow_scores = tuple(policy.score(item) for item in projected)
    slow_winner = max(
        range(len(projected)),
        key=lambda index: (slow_scores[index], -index),
    )
    prototype_values: list[float] = []
    prototype_uncertainties: list[float] = []
    if fast_supported:
        control_axes = {
            item.primary_mechanism_axis
            for item in projected
            if item.matched_control
        }
        if len(control_axes) != 1:
            raise MetaVNextRoutingError(
                "fast routing requires exactly one matched control axis"
            )
        control_axis = next(iter(control_axes))
        for item in projected:
            correction, uncertainty = fast_prediction(
                fast_state,
                item,
                policy,
                neighbor_count=router.fast_neighbor_count,
                task_control_axis=control_axis,
            )
            prototype_values.append(policy.score(item) + correction)
            prototype_uncertainties.append(uncertainty)
    override = False
    if prototype_values:
        fast_winner = max(
            range(len(prototype_values)),
            key=lambda index: (prototype_values[index], -index),
        )
        override = (
            prototype_values[fast_winner] - prototype_values[slow_winner]
            > router.fast_override_margin
        )

    scores = []
    for index, (original, projected_candidate) in enumerate(
        zip(candidates, projected, strict=True)
    ):
        slow_score = slow_scores[index]
        slow_uncertainty = policy.uncertainty(projected_candidate)
        correction = (
            prototype_values[index] - slow_score if override else 0.0
        )
        fast_uncertainty = (
            prototype_uncertainties[index] if override else 0.0
        )
        scores.append(
            CandidateRouteScoreV1(
                candidate_id=original.candidate_id,
                candidate_semantics_digest=original.candidate_semantics_digest,
                slow_score=round(slow_score, 15),
                fast_correction=round(correction, 15),
                uncertainty=round(
                    math.sqrt(
                        slow_uncertainty * slow_uncertainty
                        + fast_uncertainty * fast_uncertainty
                    ),
                    15,
                ),
                final_score=round(slow_score + correction, 15),
            )
        )
    selected = max(
        enumerate(scores),
        key=lambda item: (item[1].final_score, -item[0]),
    )[1]
    decision = MetaVNextRouteDecisionV1(
        pool_digest=pool.digest,
        policy_digest=policy.digest,
        fast_state_digest=fast_state.digest if fast_supported else None,
        shadow_mode=bool(shadow_mode),
        scored_candidates=tuple(scores),
        selected_candidate_id=selected.candidate_id,
        selected_candidate_semantics_digest=(
            selected.candidate_semantics_digest
        ),
    )
    projection = MetaV18SupportProjectionV1(
        actual_task_scale=float(actual_task_scale),
        actual_task_density=float(actual_task_density),
        task_scale_support=scale_bounds,
        task_density_support=density_bounds,
        calibrated_axes=calibrated_axes,
        pool_axes=pool_axes,
        generic_transfer_axes=generic_axes,
        task_context_supported=task_supported,
        fast_supported=fast_supported,
        fast_support_reason=fast_reason,
        clipped_feature_count=clipped_feature_count,
        projected_feature_matrix_digest=sha256_digest(
            [
                {
                    "candidate_semantics_digest": item.candidate_semantics_digest,
                    "rank_features": item.rank_features,
                }
                for item in projected
            ]
        ),
        support_resource_sha256=feature_support_sha256(),
    )
    return decision, projection


__all__ = [
    "FAST_SUPPORT_ID_V18",
    "MetaV18SupportError",
    "MetaV18SupportProjectionV1",
    "ROUTER_POLICY_ID_V18",
    "SLOW_PROJECTION_ID_V18",
    "feature_support_sha256",
    "load_feature_support",
    "project_candidate_for_slow",
    "route_support_aware",
]
