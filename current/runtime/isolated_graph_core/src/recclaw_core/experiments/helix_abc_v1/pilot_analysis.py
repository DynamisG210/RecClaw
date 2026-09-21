"""Frozen, treatment-blind M6 Pilot support and four-axis analysis."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest, validate_sha256


PILOT_MAX_FAILURE_RATE = 0.25
PILOT_MIN_SUCCESS_PER_INSTANCE = 2


class FrontierProjectionV13(str, Enum):
    OBSERVED = "OBSERVED"
    SEARCH_ELIGIBLE = "SEARCH_ELIGIBLE"
    CONFIRMED = "CONFIRMED"


def analysis_observation_key(row: Mapping[str, Any]) -> str:
    """Return the stable identity used by explicit eligibility projections."""

    for field in ("raw_result_digest", "observation_id", "analysis_row_id"):
        value = row.get(field)
        if isinstance(value, str) and value:
            return value
    return sha256_digest(
        {
            "candidate_id": row.get("candidate_id"),
            "observation_seed": row.get("observation_seed"),
            "opaque_instance_id": row.get("opaque_instance_id"),
            "round_index": row.get("round_index"),
        }
    )


def _row_enters_frontier(
    row: Mapping[str, Any],
    *,
    frontier_projection: FrontierProjectionV13,
    eligibility_by_observation: Mapping[str, str] | None,
    post_selection_evaluator_digest: str | None,
) -> bool:
    if frontier_projection is FrontierProjectionV13.OBSERVED:
        return True
    if eligibility_by_observation is None:
        raise ValueError(
            f"{frontier_projection.value} requires explicit Guard eligibility"
        )
    key = analysis_observation_key(row)
    try:
        eligibility = str(eligibility_by_observation[key])
    except KeyError as exc:
        raise ValueError(f"missing Guard eligibility for observation {key}") from exc
    if eligibility not in {
        "SEARCH_ELIGIBLE",
        "SEARCH_ELIGIBLE_PRELIMINARY",
    }:
        return False
    if frontier_projection is FrontierProjectionV13.SEARCH_ELIGIBLE:
        return True
    if post_selection_evaluator_digest is None:
        raise ValueError(
            "CONFIRMED requires the frozen post-selection evaluator digest"
        )
    validate_sha256(
        post_selection_evaluator_digest,
        field_name="post_selection_evaluator_digest",
    )
    return (
        row.get("post_selection_status") == "CONFIRMED"
        and row.get("post_selection_evaluator_digest")
        == post_selection_evaluator_digest
    )


def four_axis_frontiers(
    rows: Sequence[Mapping[str, Any]],
    *,
    frontier_projection: FrontierProjectionV13 = FrontierProjectionV13.OBSERVED,
    eligibility_by_observation: Mapping[str, str] | None = None,
    post_selection_evaluator_digest: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Project costs over all rounds and metrics only from the named frontier."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opaque_instance_id"])].append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for instance, values in sorted(grouped.items()):
        best: float | None = None
        executions = 0
        tokens = 0
        gpu_cost = 0
        points = []
        for row in sorted(values, key=lambda item: int(item["round_index"])):
            executions += int(row["ordinary_execution_count"])
            tokens += int(row["billed_tokens"])
            gpu_cost += int(row["gpu_cost_microunits"])
            metric = row.get("ndcg")
            if (
                isinstance(metric, (int, float))
                and _row_enters_frontier(
                    row,
                    frontier_projection=frontier_projection,
                    eligibility_by_observation=eligibility_by_observation,
                    post_selection_evaluator_digest=(
                        post_selection_evaluator_digest
                    ),
                )
            ):
                best = float(metric) if best is None else max(best, float(metric))
            points.append(
                {
                    "execution_axis": executions,
                    "frontier_ndcg": best,
                    "gpu_cost_axis_microunits": gpu_cost,
                    "frontier_projection": frontier_projection.value,
                    "round_axis": int(row["round_index"]),
                    "support": int(best is not None),
                    "token_axis": tokens,
                }
            )
        result[instance] = points
    return canonical_value(result)


def descriptive_effect_summary(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Pilot-only mechanical summary; intentionally computes no treatment effect."""

    observed = four_axis_frontiers(
        rows,
        frontier_projection=FrontierProjectionV13.OBSERVED,
    )
    return canonical_value(
        {
            "analysis_class": "DEVELOPMENT_PILOT_MECHANICAL_SUPPORT_ONLY",
            "confirmed_frontier": "NOT_COMPUTED",
            "formal_inference": False,
            "observed_frontier": observed,
            "search_eligible_frontier": "REQUIRES_EXPLICIT_ELIGIBILITY",
            "treatment_effect": "NOT_AUTHORIZED",
        }
    )


def pilot_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_instances: int,
    expected_rounds_per_instance: int,
    guard_call_count: int,
    expected_guard_call_count: int,
    meta_versions: Mapping[str, int],
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["opaque_instance_id"])].append(row)
    expected_rows = expected_instances * expected_rounds_per_instance
    completeness = (
        len(grouped) == expected_instances
        and len(rows) == expected_rows
        and all(
            len(values) == expected_rounds_per_instance
            for values in grouped.values()
        )
    )
    successes = sum(row.get("run_status") == "SUCCESS" for row in rows)
    attempted = [
        row for row in rows if int(row["ordinary_execution_count"]) == 1
    ]
    failures = sum(
        row.get("run_status") != "SUCCESS" for row in attempted
    )
    failure_rate = failures / len(attempted) if attempted else 1.0
    support_by_instance = {
        instance: sum(row.get("run_status") == "SUCCESS" for row in values)
        for instance, values in sorted(grouped.items())
    }
    support_ok = (
        completeness
        and all(
            value >= PILOT_MIN_SUCCESS_PER_INSTANCE
            for value in support_by_instance.values()
        )
    )
    metrics_complete = all(
        row.get("run_status") != "SUCCESS"
        or isinstance(row.get("ndcg"), (int, float))
        for row in rows
    )
    meta_ok = (
        set(meta_versions) == {"B", "C"}
        and all(int(value) == expected_rounds_per_instance + 1 for value in meta_versions.values())
    )
    checks = {
        "failure_rate_within_limit": failure_rate <= PILOT_MAX_FAILURE_RATE,
        "guard_call_count_exact": guard_call_count == expected_guard_call_count,
        "identity_completeness": completeness,
        "meta_versioned": meta_ok,
        "successful_metric_rows_complete": metrics_complete,
        "scheduled_round_rows_complete": completeness,
        "support_non_degenerate": support_ok,
    }
    if not rows or not completeness:
        verdict = "INSUFFICIENT_INFORMATION"
    elif all(checks.values()):
        verdict = "GO"
    else:
        verdict = "NOT_READY"
    return {
        "checks": checks,
        "failure_count": failures,
        "failure_rate": failure_rate,
        "frontier_digest": sha256_digest(
            four_axis_frontiers(
                rows,
                frontier_projection=FrontierProjectionV13.OBSERVED,
            )
        ),
        "row_count": len(rows),
        "success_count": successes,
        "support_by_instance": support_by_instance,
        "verdict": verdict,
    }


__all__ = [
    "FrontierProjectionV13",
    "PILOT_MAX_FAILURE_RATE",
    "PILOT_MIN_SUCCESS_PER_INSTANCE",
    "analysis_observation_key",
    "descriptive_effect_summary",
    "four_axis_frontiers",
    "pilot_readiness",
]
